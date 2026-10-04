# %% [markdown]
# # HumAID 03 — Backbone search (ModernBERT, DeBERTa-v3) and ensemble
# Same recipe as the notebook 02 baselines, applied to newer backbones, to choose the
# base model for our method. Models train in parallel, one queue per GPU, and their
# predictions are averaged at the end.

# %%
import json
import os
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

DEBUG = False  # True: 10% of the data and 1 epoch, to check the pipeline quickly
# (name, checkpoint, learning rate) — one list per GPU, run top to bottom
QUEUES = [
    [("modernbert_base", "answerdotai/ModernBERT-base", 5e-5),
     ("deberta_v3_base", "microsoft/deberta-v3-base", 2e-5)],
    [("modernbert_large", "answerdotai/ModernBERT-large", 2e-5)],
]
OUT = "/kaggle/working"
T0 = time.time()


def log(msg):
    m, s = divmod(int(time.time() - T0), 60)
    print(f"[{m:02d}:{s:02d}] {msg}", flush=True)


n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "No GPU attached — enable it in the notebook settings"
if n_gpu == 1:
    QUEUES = [[job for q in QUEUES for job in q]]
log(f"GPUs: {n_gpu} x {torch.cuda.get_device_name(0)} | DEBUG={DEBUG}")

# %% [markdown]
# ## 1. Training script (one model per call)

# %%
WORKER = r'''
import glob, json, os, re, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

NAME, CHECKPOINT, LR = sys.argv[1], sys.argv[2], float(sys.argv[3])
DEBUG = os.environ.get("DEBUG") == "1"
EPOCHS, BATCH, MAX_LEN, SEED, LOG_EVERY = (1 if DEBUG else 3), 32, 96, 42, 100
OUT = "/kaggle/working"
T0 = float(os.environ["T0"])
device = torch.device("cuda")


def log(msg):
    m, s = divmod(int(time.time() - T0), 60)
    print(f"[{m:02d}:{s:02d}] {NAME} | {msg}", flush=True)


def clean(text):
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"^RT\s+", "", text.strip())
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


df = pd.read_csv(glob.glob("/kaggle/input/**/humaid.csv", recursive=True)[0])
df["clean"] = df["text"].map(clean)
labels = sorted(df["label"].unique())
df["y"] = df["label"].map({l: i for i, l in enumerate(labels)})
train, dev, test = (df[df["split"] == s].reset_index(drop=True) for s in ["train", "dev", "test"])
if DEBUG:
    train = train.sample(frac=0.1, random_state=SEED).reset_index(drop=True)

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
def predict_logits(model, loader):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**batch).logits.float().cpu())
    return torch.cat(out).numpy()


def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4)}


torch.manual_seed(SEED)
log(f"loading {CHECKPOINT} (lr={LR}, epochs={EPOCHS}, train={len(train)})")
tok = AutoTokenizer.from_pretrained(CHECKPOINT)
# .float(): some checkpoints (DeBERTa-v3) are stored in half precision, which breaks the gradient scaler
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(labels)).float().to(device)
train_loader = make_loader(tok, train["clean"], train["y"], shuffle=True)
dev_loader, test_loader = make_loader(tok, dev["clean"]), make_loader(tok, test["clean"])

total = EPOCHS * len(train_loader)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
scaler = torch.amp.GradScaler("cuda")
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

best_f1, best_state, best_dev_logits, history, step, start = -1, None, None, [], 0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(**batch).logits.float(), y)
        if not torch.isfinite(loss):
            log("loss is not finite — stopping this model")
            sys.exit(1)
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
            log(f"epoch {epoch}/{EPOCHS} step {step}/{total} loss {running / LOG_EVERY:.4f} | "
                f"{1 / rate:.1f} it/s | ETA {rate * (total - step) / 60:.1f} min")
            running = 0.0
    dev_logits = predict_logits(model, dev_loader)
    dev_scores = scores(dev["y"], dev_logits.argmax(-1))
    history.append({"epoch": epoch, **dev_scores})
    log(f"epoch {epoch} DEV {dev_scores}")
    if dev_scores["macro_f1"] > best_f1:
        best_f1, best_dev_logits = dev_scores["macro_f1"], dev_logits
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
test_logits = predict_logits(model, test_loader)
test_pred = test_logits.argmax(-1)
result = {"checkpoint": CHECKPOINT, "lr": LR, "dev_history": history,
          "test": scores(test["y"], test_pred), "train_minutes": round((time.time() - start) / 60, 1)}
log(f"TEST {result['test']}")
report = classification_report(test["y"], test_pred, target_names=labels, digits=3, output_dict=True)
result["per_class_f1"] = {l: round(report[l]["f1-score"], 3) for l in labels}

np.save(f"{OUT}/{NAME}_dev_logits.npy", best_dev_logits)
np.save(f"{OUT}/{NAME}_test_logits.npy", test_logits)
pd.DataFrame({"tweet_id": test["tweet_id"], "event": test["event"], "label": test["label"],
              "pred": [labels[i] for i in test_pred]}).to_csv(f"{OUT}/{NAME}_test_predictions.csv", index=False)

cm = confusion_matrix(test["y"], test_pred, normalize="true")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"{NAME} — test confusion matrix (row-normalised)")
plt.tight_layout()
plt.savefig(f"{OUT}/{NAME}_confusion_matrix.png", dpi=150)

probe_pred = predict_logits(model, make_loader(tok, probe_df["text"].map(clean))).argmax(-1)
probe_df[NAME] = [labels[i] for i in probe_pred]
probe_df.to_csv(f"{OUT}/{NAME}_probe.csv", index=False)
with open(f"{OUT}/{NAME}_result.json", "w") as f:
    json.dump(result, f, indent=2)
log("finished")
'''
with open("worker.py", "w") as f:
    f.write(WORKER)

# %% [markdown]
# ## 2. Train the queues in parallel

# %%
KEEP = ("[", "Traceback", "Error", "error")  # progress lines and failures only


def run_queue(gpu, jobs):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "DEBUG": "1" if DEBUG else "0", "T0": str(T0),
           "TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_VERBOSITY": "error", "PYTHONWARNINGS": "ignore"}
    for name, checkpoint, lr in jobs:
        log(f"GPU {gpu}: starting {name}")
        proc = subprocess.Popen(["python", "-u", "worker.py", name, checkpoint, str(lr)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in proc.stdout:
            tail = (tail + [line])[-40:]
            if line.startswith(KEEP):
                print(line, end="", flush=True)
        if proc.wait() != 0:
            log(f"GPU {gpu}: {name} FAILED — last output:\n" + "".join(tail))


threads = [threading.Thread(target=run_queue, args=(g, q)) for g, q in enumerate(QUEUES)]
for t in threads:
    t.start()
for t in threads:
    t.join()
log("all queues finished")

# %% [markdown]
# ## 3. Comparison and ensemble

# %%
def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4)}


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


names = [n for q in QUEUES for n, _, _ in q if os.path.exists(f"{OUT}/{n}_result.json")]
results = {n: json.load(open(f"{OUT}/{n}_result.json")) for n in names}

# Earlier baselines (notebooks 01 and 02), for reference
rows = {"tfidf_lr (nb01)": {"accuracy": 0.7462, "macro_f1": 0.7298, "weighted_f1": 0.7475},
        "bert (nb02)": {"accuracy": 0.7819, "macro_f1": 0.7655, "weighted_f1": 0.7791},
        "twitter_roberta (nb02)": {"accuracy": 0.7759, "macro_f1": 0.7607, "weighted_f1": 0.7712}}
rows.update({n: results[n]["test"] for n in names})

if len(names) > 1:
    first = pd.read_csv(f"{OUT}/{names[0]}_test_predictions.csv")
    labels = sorted(first["label"].unique())
    y_test = first["label"].map({l: i for i, l in enumerate(labels)}).values
    avg = np.mean([softmax(np.load(f"{OUT}/{n}_test_logits.npy")) for n in names], axis=0)
    rows["ensemble of " + " + ".join(names)] = results["ensemble"] = scores(y_test, avg.argmax(-1))

summary = pd.DataFrame(rows).T
print(summary.to_string(), flush=True)
with open(f"{OUT}/backbone_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# %%
probe = None
for n in names:
    p = pd.read_csv(f"{OUT}/{n}_probe.csv")
    probe = p if probe is None else probe.merge(p[["text", n]], on="text")
if probe is not None:
    probe.to_csv(f"{OUT}/backbone_confounder_probe.csv", index=False)
    pd.set_option("display.max_colwidth", 80)
    print(probe.to_string(), flush=True)
log("done")
