# %% [markdown]
# # HumAID 02 — Transformer baselines (BERT, Twitter-RoBERTa)
# Fine-tunes the two reference transformers on the official split. These are the
# baselines our model has to beat, not the proposed model.

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU keeps batch size and timing predictable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import json
import re
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

DEBUG = False  # True: 10% of the data and 1 epoch, to check the pipeline quickly
MODELS = {"bert": "bert-base-uncased", "twitter_roberta": "cardiffnlp/twitter-roberta-base"}
EPOCHS, BATCH, LR, MAX_LEN, SEED = (1 if DEBUG else 3), 32, 2e-5, 96, 42
LOG_EVERY = 100

OUT = "/kaggle/working" if os.path.exists("/kaggle/working") else "outputs"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(msg):
    m, s = divmod(int(time.time() - T0), 60)
    print(f"[{m:02d}:{s:02d}] {msg}", flush=True)


assert torch.cuda.is_available(), "No GPU attached — enable it in the notebook settings"
device = torch.device("cuda")
log(f"GPU: {torch.cuda.get_device_name(0)} | DEBUG={DEBUG} | epochs={EPOCHS}")

# %% [markdown]
# ## 1. Data

# %%
EVENTS = {
    "california_wildfires_2018": "wildfire", "canada_wildfires_2016": "wildfire",
    "greece_wildfires_2018": "wildfire",
    "cyclone_idai_2019": "hurricane", "hurricane_dorian_2019": "hurricane",
    "hurricane_florence_2018": "hurricane", "hurricane_harvey_2017": "hurricane",
    "hurricane_irma_2017": "hurricane", "hurricane_maria_2017": "hurricane",
    "hurricane_matthew_2016": "hurricane",
    "ecuador_earthquake_2016": "earthquake", "italy_earthquake_aug_2016": "earthquake",
    "kaikoura_earthquake_2016": "earthquake", "pakistan_earthquake_2019": "earthquake",
    "puebla_mexico_earthquake_2017": "earthquake",
    "kerala_floods_2018": "flood", "maryland_floods_2018": "flood",
    "midwestern_us_floods_2019": "flood", "srilanka_floods_2017": "flood",
}
BASE = "https://huggingface.co/datasets/QCRI/HumAID-events/resolve/main"


def load_humaid():
    """Use the CSV saved by notebook 01 if it is attached, otherwise download."""
    found = glob.glob("/kaggle/input/**/humaid.csv", recursive=True)
    if found:
        log(f"reading {found[0]}")
        return pd.read_csv(found[0])
    rows = []
    for i, (event, dtype) in enumerate(EVENTS.items(), 1):
        for split in ["train", "dev", "test"]:
            with urllib.request.urlopen(f"{BASE}/{event}/{split}.json") as r:
                for item in json.load(r):
                    rows.append({"tweet_id": item["tweet_id"], "text": item["tweet_text"],
                                 "label": item["class_label"], "event": event,
                                 "disaster_type": dtype, "split": split})
        log(f"downloaded event {i}/{len(EVENTS)}: {event}")
    return pd.DataFrame(rows)


def clean(text):
    """Light normalisation in the format Twitter-RoBERTa was pretrained on."""
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"^RT\s+", "", text.strip())
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


df = load_humaid()
df["clean"] = df["text"].map(clean)
labels = sorted(df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
df["y"] = df["label"].map(label2id)

train, dev, test = (df[df["split"] == s].reset_index(drop=True) for s in ["train", "dev", "test"])
if DEBUG:
    train = train.sample(frac=0.1, random_state=SEED).reset_index(drop=True)
    dev = dev.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    test = test.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
log(f"train={len(train)} dev={len(dev)} test={len(test)} classes={len(labels)}")

# Square-root-dampened inverse-frequency weights: help rare classes without swamping the big ones
counts = train["y"].value_counts().sort_index().values
class_weights = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float, device=device)

PROBE = [
    ("Bridge collapsed after the flood, vehicles stranded on both sides", "real"),
    ("Bridge closed for repair work till Monday, please use the alternate route", "routine"),
    ("Road blocked due to landslide, several houses damaged", "real"),
    ("Road blocked due to metro construction work near the station", "routine"),
    ("Power lines down across the town after the cyclone", "real"),
    ("Scheduled power cut tomorrow for maintenance of power lines", "routine"),
    ("Building collapsed in the earthquake, people trapped inside", "real"),
    ("Old building being demolished today to make way for a new mall", "routine"),
    ("My exam today was a complete disaster", "figurative"),
    ("That concert was fire, the crowd was flooding in", "figurative"),
]
probe_df = pd.DataFrame(PROBE, columns=["text", "reality"])

# %% [markdown]
# ## 2. Training loop with live progress

# %%
def make_loader(tok, texts, ys=None, shuffle=False):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
             for i in range(len(texts))]
    if ys is not None:
        for item, y in zip(items, ys):
            item["labels"] = int(y)
    return DataLoader(items, batch_size=BATCH if shuffle else BATCH * 4, shuffle=shuffle,
                      collate_fn=DataCollatorWithPadding(tok), num_workers=2)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16):
            preds.append(model(**batch).logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4)}


def run(name, checkpoint):
    torch.manual_seed(SEED)
    log(f"===== {name}: {checkpoint} =====")
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(labels)).to(device)
    train_loader = make_loader(tok, train["clean"], train["y"], shuffle=True)
    dev_loader, test_loader = make_loader(tok, dev["clean"]), make_loader(tok, test["clean"])

    total = EPOCHS * len(train_loader)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, history, step, start = -1, None, [], 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(**batch).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
            if step % LOG_EVERY == 0:
                rate = (time.time() - start) / step
                log(f"{name} epoch {epoch}/{EPOCHS} step {step}/{total} "
                    f"loss {running / LOG_EVERY:.4f} | {1 / rate:.1f} it/s | "
                    f"ETA this model {rate * (total - step) / 60:.1f} min")
                running = 0.0
        dev_scores = scores(dev["y"], predict(model, dev_loader))
        history.append({"epoch": epoch, **dev_scores})
        log(f"{name} epoch {epoch} DEV {dev_scores}")
        if dev_scores["macro_f1"] > best_f1:
            best_f1 = dev_scores["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_pred = predict(model, test_loader)
    result = {"checkpoint": checkpoint, "dev_history": history, "test": scores(test["y"], test_pred),
              "train_minutes": round((time.time() - start) / 60, 1)}
    log(f"{name} TEST {result['test']}")
    print(classification_report(test["y"], test_pred, target_names=labels, digits=3), flush=True)

    pd.DataFrame({"tweet_id": test["tweet_id"], "event": test["event"], "label": test["label"],
                  "pred": [labels[i] for i in test_pred]}).to_csv(f"{OUT}/{name}_test_predictions.csv", index=False)

    cm = confusion_matrix(test["y"], test_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} — test confusion matrix (row-normalised)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}_confusion_matrix.png", dpi=150)
    plt.show()

    probe_pred = predict(model, make_loader(tok, probe_df["text"].map(clean)))
    probe_df[name] = [labels[i] for i in probe_pred]

    del model, opt, best_state
    torch.cuda.empty_cache()
    return result


# %% [markdown]
# ## 3. Run both baselines

# %%
results = {}
for i, (name, checkpoint) in enumerate(MODELS.items(), 1):
    log(f"model {i}/{len(MODELS)}")
    results[name] = run(name, checkpoint)
    with open(f"{OUT}/transformer_baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

# %% [markdown]
# ## 4. Summary and confounder probe

# %%
summary = pd.DataFrame({n: r["test"] for n, r in results.items()}).T
print(summary, flush=True)
pd.set_option("display.max_colwidth", 80)
probe_df.to_csv(f"{OUT}/transformer_confounder_probe.csv", index=False)
log("done")
probe_df
