# Disaster Response Tweet Triage

Confounder-robust classification of disaster tweets on the HumAID dataset.

**Team:** Krrish Chanchal (23CSB0B06) · Shashank Dixit (23CSB0B07) · Shubh Gupta (23CSB0F33)

## 1. What the project is

Given a tweet posted during a disaster, classify it into a humanitarian category
(infrastructure damage, urgent requests, injured or dead people, and so on) so that
responders can prioritise actionable messages.

### What the faculty asked for

1. **Novelty.** A plain fine-tuned BERT or Twitter-BERT is not enough as the main model.
   Those two stay in the project only as baselines to beat.
2. **Context discrimination.** The model must tell apart messages that share vocabulary
   but not reality, for example:
   - "Bridge collapsed after the flood" → a disaster report
   - "Bridge closed for repair work till Monday" → routine construction, not a disaster
   - "My exam was a complete disaster" → figurative, not a disaster

### Our angle: confounder-robust triage

HumAID cannot teach point 2 by itself, because every tweet in it was collected during a
real disaster. That gap is the contribution. Four parts:

| # | Component | What it is |
|---|---|---|
| 1 | **Confounder challenge set** | 300–500 hand-verified minimal pairs (routine construction or maintenance, figurative language, past events, drills) that share words with real disaster reports. A separate portion is used for training; the test pairs are never trained on. |
| 2 | **Modern backbone** | ModernBERT or DeBERTa-v3 with LoRA, instead of BERT. |
| 3 | **Two heads + contrastive loss** | Head A: is this an actual, ongoing disaster (real / routine / figurative)? Head B: humanitarian category. A supervised contrastive loss with hard negatives pushes each minimal pair apart in embedding space. |
| 4 | **Event masking** | Event and place names ("Harvey", "Kerala") are replaced by a placeholder so the model cannot memorise events; evaluated by holding out whole disaster types. |

The headline result we are aiming for: standard baselines score well on HumAID but fail
the confounder set, while our model holds up on both.

## 2. Dataset

HumAID (Alam et al., 2021), per-event release: `QCRI/HumAID-events` on Hugging Face.
Licence CC BY-NC-SA 4.0. The data is not committed; notebook 01 downloads it.

- 76,484 tweets, 19 events (2016–2019), four disaster types: hurricane, earthquake, flood, wildfire
- Official split: 53,531 train / 7,793 dev / 15,160 test
- **10 classes** in the released data (the proposal slides say 11; "Don't know / can't judge" is not in the release)
- Largest class is 59× the smallest (rescue/volunteering/donation 21,278 vs missing or found people)
- Tweets are short: median 19 words, max 99

## 3. Results so far

### Baseline: TF-IDF + Logistic Regression (notebook 01)

| Test bed | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Official dev | 0.743 | 0.722 | 0.744 |
| Official test | 0.746 | 0.730 | 0.748 |

Cross-disaster (train on three disaster types, test on the fourth), macro-F1:

| Held-out type | Seen in training | Unseen |
|---|---|---|
| Earthquake | 0.746 | 0.608 |
| Flood | 0.654 | 0.577 |
| Hurricane | 0.719 | 0.659 |
| Wildfire | 0.679 | 0.619 |

Weakest classes: other relevant information (F1 0.54), requests or urgent needs (0.55),
not humanitarian (0.58).

Confounder probe (10 hand-written sentences, a demonstration only): all 4 real disaster
sentences were labelled infrastructure damage correctly; of the 6 routine or figurative
ones, only 2 were labelled not humanitarian. "Road blocked due to metro construction
work" was labelled infrastructure damage. Full table in
`outputs/kaggle_01/baseline_confounder_probe.csv`.

## 4. Roadmap

- [x] Project setup, Kaggle link, data download
- [x] EDA
- [x] Baseline 1: TF-IDF + Logistic Regression, in-distribution and cross-disaster
- [ ] Baselines 2 and 3: BERT-base and Twitter-RoBERTa fine-tuning (GPU)
- [ ] Confounder set: write and hand-verify the minimal pairs (start early, it is manual work)
- [ ] Our model: ModernBERT/DeBERTa-v3 + LoRA
- [ ] Add the two heads and the supervised contrastive loss
- [ ] Add event masking
- [ ] Ablations: remove one component at a time
- [ ] Final evaluation on three test beds (official test, cross-disaster, confounder set), 3 seeds
- [ ] Error analysis: 20–30 misclassified examples grouped by cause
- [ ] Gradio demo and report

Primary metric is macro-F1; per-class F1 and confusion matrices are reported alongside.
Class imbalance is handled with a class-weighted (or focal) loss, not oversampling.

## 5. Repository layout

```
notebooks/
  01_eda_baseline.py        source for each run, written as "# %%" cells
  kaggle_01/                what gets pushed: kernel-metadata.json + generated .ipynb
src/
  make_notebook.py          converts a "# %%" .py file into a .ipynb
  kaggle_run.py             push → stream live logs → download outputs
confounder_set/             the hand-verified minimal pairs (to be written)
outputs/kaggle_01/          metrics, plots and log pulled back from Kaggle
data/                       local dataset copy (git-ignored)
```

## 6. How we run things

Training runs on Kaggle Notebooks (free T4 GPU, about 30 GPU-hours per week per account).
Code is written here and pushed with the Kaggle command-line tool, so nothing is
copy-pasted.

One-time setup per team member:

1. Phone-verify the Kaggle account (needed for GPU and internet in notebooks).
2. kaggle.com → Settings → API → create a token, and save it under `~/.kaggle/`
   (on Windows: `C:\Users\<you>\.kaggle\`). Never commit or share the token.
3. `pip install kaggle`
4. Change the `id` in `notebooks/kaggle_*/kernel-metadata.json` to your own Kaggle username.

Run a notebook and watch it live:

```
python src/kaggle_run.py notebooks/01_eda_baseline.py kaggle_01
```

This builds the notebook, pushes it, prints its log as it runs, and saves the outputs to
`outputs/kaggle_01/`. Every slow stage in a notebook should print its step count, elapsed
time and estimated time remaining.

To see a run on the website: kaggle.com → Your Work → Code. Notebooks are private by
default; add teammates under Share → Collaborators.

Rules for GPU runs: mixed precision, `max_length` 64–96 with dynamic padding, early
stopping on dev macro-F1, and debug on a 10% subset before a full run.
