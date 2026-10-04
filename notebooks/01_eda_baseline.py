# %% [markdown]
# # HumAID 01 — Data, EDA and TF-IDF baseline
# Downloads the per-event HumAID files, explores them, and trains the
# TF-IDF + Logistic Regression reference model (in-distribution and cross-disaster).

# %%
import json
import os
import re
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

OUT = "/kaggle/working" if os.path.exists("/kaggle/working") else "outputs"
os.makedirs(OUT, exist_ok=True)
SEED = 42

# %% [markdown]
# ## 1. Download HumAID (19 events, official train/dev/test splits)

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

rows = []
for event, dtype in EVENTS.items():
    for split in ["train", "dev", "test"]:
        with urllib.request.urlopen(f"{BASE}/{event}/{split}.json") as r:
            for item in json.load(r):
                rows.append({"tweet_id": item["tweet_id"], "text": item["tweet_text"],
                             "label": item["class_label"], "event": event,
                             "disaster_type": dtype, "split": split})
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/humaid.csv", index=False)
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Exploratory data analysis

# %%
print("Tweets per split:\n", df["split"].value_counts(), "\n")
print("Number of classes:", df["label"].nunique())
class_counts = df["label"].value_counts()
print(class_counts, "\n")
print("Imbalance ratio (largest / smallest class): %.1f" % (class_counts.max() / class_counts.min()))
print("Duplicate tweet texts:", df["text"].duplicated().sum())

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
class_counts.sort_values().plot.barh(ax=axes[0], color="steelblue")
axes[0].set_title("Tweets per class")
df["event"].value_counts().sort_values().plot.barh(ax=axes[1], color="darkorange")
axes[1].set_title("Tweets per event")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_class_event_counts.png", dpi=150)
plt.show()

# %%
df["n_words"] = df["text"].str.split().str.len()
print(df["n_words"].describe())
df["n_words"].hist(bins=50, figsize=(8, 4))
plt.title("Tweet length (words)")
plt.savefig(f"{OUT}/eda_length.png", dpi=150)
plt.show()

# %%
# How differently are the classes distributed across disaster types? (source of domain shift)
ct = pd.crosstab(df["label"], df["disaster_type"], normalize="columns") * 100
plt.figure(figsize=(8, 6))
sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues")
plt.title("Class share (%) within each disaster type")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_class_by_disaster_type.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Preprocessing
# Light normalisation only: URLs and mentions become placeholders, hashtags keep their word.

# %%
def clean(text):
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " @USER ", text)
    text = re.sub(r"^RT\s+", "", text.strip())
    text = text.replace("#", " ")
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip().lower()


df["clean"] = df["text"].map(clean)
df[["text", "clean"]].head(5)

# %% [markdown]
# ## 4. Baseline: TF-IDF + Logistic Regression (official split)

# %%
def fit_predict(train_text, train_y, eval_texts):
    """Word + character TF-IDF with a class-weighted logistic regression."""
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    clf = LogisticRegression(C=5, max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, train_y)
    preds = [clf.predict(hstack([word.transform(t), char.transform(t)]).tocsr()) for t in eval_texts]
    return clf, word, char, preds


def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4)}


train, dev, test = (df[df["split"] == s] for s in ["train", "dev", "test"])
clf, word, char, (dev_pred, test_pred) = fit_predict(train["clean"], train["label"],
                                                     [dev["clean"], test["clean"]])
results = {"in_distribution": {"dev": scores(dev["label"], dev_pred),
                               "test": scores(test["label"], test_pred)}}
print(json.dumps(results, indent=2))
print(classification_report(test["label"], test_pred, digits=3))

# %%
labels = sorted(df["label"].unique())
cm = confusion_matrix(test["label"], test_pred, labels=labels, normalize="true")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("TF-IDF + LR — test confusion matrix (row-normalised)")
plt.tight_layout()
plt.savefig(f"{OUT}/baseline_confusion_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Cross-disaster evaluation (leave one disaster type out)
# Train on three disaster types, test on the unseen fourth.

# %%
results["leave_one_type_out"] = {}
for held_out in sorted(df["disaster_type"].unique()):
    tr = df[(df["disaster_type"] != held_out) & (df["split"] == "train")]
    te = df[(df["disaster_type"] == held_out) & (df["split"] == "test")]
    _, _, _, (pred,) = fit_predict(tr["clean"], tr["label"], [te["clean"]])
    in_dist = scores(te["label"], test_pred[(test["disaster_type"] == held_out).values])
    results["leave_one_type_out"][held_out] = {"unseen": scores(te["label"], pred), "seen": in_dist}
    print(f"{held_out:11s} macro-F1 seen={in_dist['macro_f1']:.3f}  unseen={scores(te['label'], pred)['macro_f1']:.3f}")

# %% [markdown]
# ## 6. Confounder probe
# Same vocabulary, different reality. A triage model should not treat routine
# construction or figurative language as a disaster report.

# %%
probe = [
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
probe_df = pd.DataFrame(probe, columns=["text", "reality"])
pc = probe_df["text"].map(clean)
probe_df["baseline_prediction"] = clf.predict(hstack([word.transform(pc), char.transform(pc)]).tocsr())
pd.set_option("display.max_colwidth", 100)
probe_df.to_csv(f"{OUT}/baseline_confounder_probe.csv", index=False)
probe_df

# %%
with open(f"{OUT}/baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
