# %% [markdown]
# # HumAID 04 — Anchor entity, JEV model and classifier
# The three-stage pipeline, built up one component at a time so each one's effect is visible:
#
# | Run | Confounder training data | Anchor entity + JEV heads | Contrastive loss |
# |---|---|---|---|
# | `baseline` | | | |
# | `plus_confounders` | yes | | |
# | `jev` | yes | yes | |
# | `jev_contrastive` | yes | yes | yes |
#
# Every run is scored on the official HumAID test set and on the confounder test split,
# whose anchor entities never appear in training.

# %%
import json
import os
import subprocess
import threading
import time

import pandas as pd
import torch

DEBUG = False  # True: 10% of the data and 1 epoch, to check the pipeline quickly
# One list per GPU, run top to bottom
QUEUES = [
    [("baseline", {"confounders": False, "jev": False, "contrastive": False}),
     ("plus_confounders", {"confounders": True, "jev": False, "contrastive": False})],
    [("jev", {"confounders": True, "jev": True, "contrastive": False}),
     ("jev_contrastive", {"confounders": True, "jev": True, "contrastive": True})],
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
# ## 1. Project files used by the training script

# %%
# %embed CONFOUNDER_CSV confounder_set/confounder_set_v2_draft.csv
# %embed ANCHOR_ENTITIES_PY src/anchor_entities.py
for name, content in [("confounder.csv", CONFOUNDER_CSV), ("anchor_entities.py", ANCHOR_ENTITIES_PY)]:
    with open(name, "w", encoding="utf-8") as f:
        f.write(content)
conf = pd.read_csv("confounder.csv")
log(f"confounder set: {len(conf)} sentences, {conf['anchor_entity'].nunique()} anchor entities, "
    f"verified: {conf['verified'].notna().sum()}")

# %% [markdown]
# ## 2. Training script (one run per call)

# %%
WORKER = r'''
import glob, json, os, random, re, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from anchor_entities import find, tag

NAME, CFG = sys.argv[1], json.loads(sys.argv[2])
DEBUG = os.environ.get("DEBUG") == "1"
CHECKPOINT, LR = "answerdotai/ModernBERT-base", 5e-5
EPOCHS, HUM_BATCH, CONF_PER_BATCH, MAX_LEN, SEED, LOG_EVERY = (1 if DEBUG else 3), 28, 4, 96, 42, 100
W_REAL, W_CON, TEMP = 0.5, 0.3, 0.1   # loss weights for the reality head and contrastive term
OUT = "/kaggle/working"
T0 = float(os.environ["T0"])
device = torch.device("cuda")
rng = random.Random(SEED)
torch.manual_seed(SEED)


def log(msg):
    m, s = divmod(int(time.time() - T0), 60)
    print(f"[{m:02d}:{s:02d}] {NAME} | {msg}", flush=True)


def clean(text):
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"^RT\s+", "", text.strip())
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


# ---------- data
hum = pd.read_csv(glob.glob("/kaggle/input/**/humaid.csv", recursive=True)[0])
conf = pd.read_csv(f"{OUT}/confounder.csv")
labels = sorted(hum["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
NH = label2id["not_humanitarian"]
for d in (hum, conf):
    d["clean"] = d["text"].map(clean)
    d["y"] = d["label"].map(label2id)
    d["has_anchor"] = d["clean"].map(lambda t: find(t) is not None)
    # Stage 1: mark the anchor entity so the model can read it together with its event
    d["model_text"] = d["clean"].map(tag) if CFG["jev"] else d["clean"]
hum["real"] = (hum["y"] != NH).astype(int)
conf["real"] = (conf["reality"] == "real_disaster").astype(int)

train, dev, test = (hum[hum["split"] == s].reset_index(drop=True) for s in ["train", "dev", "test"])
conf_train = conf[conf["split"] == "train"].reset_index(drop=True)
conf_test = conf[conf["split"] == "test"].reset_index(drop=True)
assert not set(conf_train["anchor_entity"]) & set(conf_test["anchor_entity"])
if DEBUG:
    train = train.sample(frac=0.1, random_state=SEED).reset_index(drop=True)

tok = AutoTokenizer.from_pretrained(CHECKPOINT)
if CFG["jev"]:
    tok.add_special_tokens({"additional_special_tokens": ["[E]", "[/E]"]})
    E_ID = tok.convert_tokens_to_ids("[E]")


def encode(d):
    enc = tok(list(d["model_text"]), truncation=True, max_length=MAX_LEN)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "y": int(d["y"].iloc[i]), "real": int(d["real"].iloc[i])} for i in range(len(d))]


def collate(items):
    batch = tok.pad({"input_ids": [it["input_ids"] for it in items],
                     "attention_mask": [it["attention_mask"] for it in items]}, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    y = torch.tensor([it["y"] for it in items], device=device)
    real = torch.tensor([it["real"] for it in items], device=device)
    return batch, y, real


train_items, dev_items, test_items = encode(train), encode(dev), encode(test)
conf_test_items = encode(conf_test)
conf_groups = []   # per anchor entity: (real sentences, look-alike sentences)
if CFG["confounders"]:
    conf_train_items = encode(conf_train)
    for _, idx in conf_train.groupby("anchor_entity").indices.items():
        conf_groups.append(([conf_train_items[i] for i in idx if conf_train["real"].iloc[i] == 1],
                            [conf_train_items[i] for i in idx if conf_train["real"].iloc[i] == 0]))

counts = train["y"].value_counts().sort_index().values
cat_weights = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float, device=device)
n_real = int(train["real"].sum())
real_weights = torch.tensor([np.sqrt(n_real / (len(train) - n_real)), 1.0], dtype=torch.float, device=device)


# ---------- model
class TriageModel(nn.Module):
    """Plain: sentence vector -> category. JEV: [sentence ; anchor entity] -> reality and category."""

    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(CHECKPOINT).float()
        if CFG["jev"]:
            self.enc.resize_token_embeddings(len(tok))
        h = self.enc.config.hidden_size
        d = 2 * h if CFG["jev"] else h
        self.cat_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, len(labels)))
        if CFG["jev"]:
            self.real_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 2))
            self.proj = nn.Linear(d, 256)

    def forward(self, input_ids, attention_mask):
        hid = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hid.dtype)
        sent = (hid * mask).sum(1) / mask.sum(1)
        if not CFG["jev"]:
            return self.cat_head(sent), None, None
        # Position of the [E] marker; 0 means the tweet has no anchor entity, so fall back to the sentence
        ent_pos = (input_ids == E_ID).int().argmax(1)
        ent = hid[torch.arange(hid.size(0), device=hid.device), ent_pos]
        ent = torch.where((ent_pos > 0).unsqueeze(-1), ent, sent)
        joint = torch.cat([sent, ent], dim=-1)
        return self.cat_head(joint), self.real_head(joint), self.proj(joint)


def supcon(z, y):
    """Supervised contrastive loss: same reality pulled together, different reality pushed apart."""
    z = F.normalize(z, dim=1)
    sim = z @ z.T / TEMP
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    valid = pos.sum(1) > 0
    if not valid.any():
        return z.sum() * 0
    return (-(log_prob * pos).sum(1)[valid] / pos.sum(1)[valid]).mean()


@torch.no_grad()
def predict(items):
    """Returns final prediction (stage 3, gated by stage 2), ungated prediction, and P(real)."""
    model.eval()
    cat, real = [], []
    for i in range(0, len(items), 128):
        batch, _, _ = collate(items[i:i + 128])
        with torch.autocast("cuda", dtype=torch.float16):
            c, r, _ = model(**batch)
        cat.append(c.float().softmax(-1).cpu())
        if r is not None:
            real.append(r.float().softmax(-1).cpu())
    cat = torch.cat(cat).numpy()
    ungated = cat.argmax(-1)
    if not real:
        return ungated, ungated, 1 - cat[:, NH]
    p_real = torch.cat(real).numpy()[:, 1]
    return np.where(p_real < 0.5, NH, ungated), ungated, p_real


def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4)}


def confounder_scores(pred):
    is_real, said_nh = conf_test["real"].values == 1, pred == NH
    out = {"real_kept": float(np.mean(~said_nh[is_real])),
           "real_exact_category": float(np.mean(pred[is_real] == conf_test["y"].values[is_real])),
           "lookalike_rejected": float(np.mean(said_nh[~is_real]))}
    out["balanced"] = (out["real_kept"] + out["lookalike_rejected"]) / 2
    for t in ["routine", "figurative", "past_or_fiction", "drill"]:
        out[f"rejected_{t}"] = float(np.mean(said_nh[(conf_test["reality"] == t).values]))
    return {k: round(v, 3) for k, v in out.items()}


# ---------- training
log(f"config {CFG} | train={len(train)} | anchor found: HumAID train {train['has_anchor'].mean():.1%}, "
    f"confounder test {conf_test['has_anchor'].mean():.1%}")
model = TriageModel().to(device)
loader = DataLoader(train_items, batch_size=HUM_BATCH, shuffle=True, collate_fn=list)
total = EPOCHS * len(loader)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
scaler = torch.amp.GradScaler("cuda")

best_f1, best_state, history, step, start = -1, None, [], 0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    run = {"cat": 0.0, "real": 0.0, "con": 0.0}
    for items in loader:
        if conf_groups:
            # One anchor entity per batch: 2 real reports and 2 look-alikes, i.e. hard negatives for each other
            reals, fakes = rng.choice(conf_groups)
            items = items + rng.sample(reals, CONF_PER_BATCH // 2) + rng.sample(fakes, CONF_PER_BATCH // 2)
        batch, y, real = collate(items)
        with torch.autocast("cuda", dtype=torch.float16):
            cat_logits, real_logits, z = model(**batch)
        loss = l_cat = F.cross_entropy(cat_logits.float(), y, weight=cat_weights)
        run["cat"] += l_cat.item()
        if CFG["jev"]:
            l_real = F.cross_entropy(real_logits.float(), real, weight=real_weights)
            loss = loss + W_REAL * l_real
            run["real"] += l_real.item()
        if CFG["contrastive"]:
            l_con = supcon(z.float(), real)
            loss = loss + W_CON * l_con
            run["con"] += l_con.item()
        if not torch.isfinite(loss):
            log("loss is not finite — stopping this run")
            sys.exit(1)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % LOG_EVERY == 0:
            rate = (time.time() - start) / step
            parts = " ".join(f"{k} {v / LOG_EVERY:.3f}" for k, v in run.items() if v)
            log(f"epoch {epoch}/{EPOCHS} step {step}/{total} loss {parts} | {1 / rate:.1f} it/s | "
                f"ETA {rate * (total - step) / 60:.1f} min")
            run = dict.fromkeys(run, 0.0)
    dev_scores = scores(dev["y"], predict(dev_items)[0])
    conf_now = confounder_scores(predict(conf_test_items)[0])
    history.append({"epoch": epoch, **dev_scores, "confounder_balanced": conf_now["balanced"]})
    log(f"epoch {epoch} DEV {dev_scores} | CONFOUNDER real_kept {conf_now['real_kept']} "
        f"lookalike_rejected {conf_now['lookalike_rejected']}")
    if dev_scores["macro_f1"] > best_f1:   # selection uses HumAID dev only, never the confounder test
        best_f1 = dev_scores["macro_f1"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# ---------- final evaluation
model.load_state_dict(best_state)
test_pred, test_ungated, _ = predict(test_items)
conf_pred, _, conf_p_real = predict(conf_test_items)
report = classification_report(test["y"], test_pred, target_names=labels, digits=3, output_dict=True)
result = {"config": CFG, "checkpoint": CHECKPOINT, "lr": LR, "dev_history": history,
          "humaid_test": {**scores(test["y"], test_pred),
                          "macro_f1_ungated": round(f1_score(test["y"], test_ungated, average="macro"), 4)},
          "per_class_f1": {l: round(report[l]["f1-score"], 3) for l in labels},
          "confounder_test": {"n_real": int(conf_test["real"].sum()),
                              "n_lookalike": int((conf_test["real"] == 0).sum()), **confounder_scores(conf_pred)},
          "anchor_found": {"humaid_train": round(float(train["has_anchor"].mean()), 3),
                           "confounder_test": round(float(conf_test["has_anchor"].mean()), 3)},
          "train_minutes": round((time.time() - start) / 60, 1)}
log(f"HUMAID TEST {result['humaid_test']}")
log(f"CONFOUNDER TEST {result['confounder_test']}")

pd.DataFrame({"tweet_id": test["tweet_id"], "event": test["event"], "label": test["label"],
              "pred": [labels[i] for i in test_pred]}).to_csv(f"{OUT}/{NAME}_humaid_test_predictions.csv", index=False)
pd.DataFrame({"id": conf_test["id"], "anchor_entity": conf_test["anchor_entity"], "reality": conf_test["reality"],
              "label": conf_test["label"], "text": conf_test["text"], "pred": [labels[i] for i in conf_pred],
              "p_real": np.round(conf_p_real, 3)}).to_csv(f"{OUT}/{NAME}_confounder_test_predictions.csv", index=False)
with open(f"{OUT}/{NAME}_result.json", "w") as f:
    json.dump(result, f, indent=2)
log("finished")
'''
with open("worker.py", "w") as f:
    f.write(WORKER)

# %% [markdown]
# ## 3. Train the queues in parallel

# %%
KEEP = ("[", "Traceback", "Error", "error")  # progress lines and failures only


def run_queue(gpu, jobs):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "DEBUG": "1" if DEBUG else "0", "T0": str(T0),
           "TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_VERBOSITY": "error", "PYTHONWARNINGS": "ignore"}
    for name, cfg in jobs:
        log(f"GPU {gpu}: starting {name}")
        proc = subprocess.Popen(["python", "-u", "worker.py", name, json.dumps(cfg)], env=env,
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
# ## 4. Results

# %%
names = [n for q in QUEUES for n, _ in q if os.path.exists(f"{OUT}/{n}_result.json")]
results = {n: json.load(open(f"{OUT}/{n}_result.json")) for n in names}
rows = {}
for n in names:
    h, c = results[n]["humaid_test"], results[n]["confounder_test"]
    rows[n] = {"humaid_acc": h["accuracy"], "humaid_macro_f1": h["macro_f1"], "real_kept": c["real_kept"],
               "lookalike_rejected": c["lookalike_rejected"], "balanced": c["balanced"],
               "routine": c["rejected_routine"], "figurative": c["rejected_figurative"],
               "past_or_fiction": c["rejected_past_or_fiction"], "drill": c["rejected_drill"]}
summary = pd.DataFrame(rows).T
pd.set_option("display.width", 250)
print(summary.to_string(), flush=True)
summary.to_csv(f"{OUT}/pipeline_summary.csv")
with open(f"{OUT}/pipeline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# %%
# Side-by-side predictions on the confounder test split
wide = None
for n in names:
    p = pd.read_csv(f"{OUT}/{n}_confounder_test_predictions.csv")
    p = p.rename(columns={"pred": n}).drop(columns=["p_real"])
    wide = p if wide is None else wide.merge(p[["id", n]], on="id")
if wide is not None:
    wide.to_csv(f"{OUT}/confounder_test_all_runs.csv", index=False)
log("done")
