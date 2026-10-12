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
real disaster. That gap is the contribution. The method is a three-stage pipeline trained
with a purpose-built confounder set:

| Part | What it is | Status |
|---|---|---|
| **Confounder set** | Sentences grouped by anchor entity (bridge, road, power lines, ...): for each entity, real disaster reports and look-alikes (routine construction or maintenance, figurative, past or fiction, drills). Split by entity, so test entities are never seen in training. | 400-sentence draft (50 entities), unverified |
| **Stage 1: anchor entity** | Find the thing the tweet is about and mark it: `[E] bridge [/E] closed for repair work`. Currently a word list (`src/anchor_entities.py`). | First version run |
| **Stage 2: JEV (Joint Entity–Event) model** | ModernBERT-base reads the sentence and the marked entity together and decides whether this is a real disaster. Optionally trained with a supervised contrastive loss, where sentences sharing an entity but not a reality are hard negatives. | First version run |
| **Stage 3: classifier** | Assigns the humanitarian category from the same joint representation. If stage 2 says "not real", the output is forced to "not humanitarian". | First version run |
| **Event masking** | Event and place names ("Harvey", "Kerala") replaced by a placeholder so the model cannot memorise events; evaluated by holding out whole disaster types. | Not started |

Full fine-tuning is used throughout. LoRA was in the original plan but has been dropped:
a base-size model trains in 13 to 26 minutes, and the baselines were fully fine-tuned.

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

### Transformer baselines (notebook 02)

Fine-tuned on the official split: 3 epochs, batch 32, lr 2e-5, max length 96, mixed
precision, square-root-dampened class weights, best epoch chosen on dev macro-F1.
One seed so far. About 13 minutes per model on a T4.

| Model | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.746 | 0.730 | 0.748 |
| BERT-base | 0.782 | 0.766 | 0.779 |
| Twitter-RoBERTa-base | 0.776 | 0.761 | 0.771 |

Both transformers are still weakest on other relevant information (F1 0.52–0.56),
requests or urgent needs (0.62) and not humanitarian (0.63).

Confounder probe: both get the 4 real sentences and the 2 figurative ones right, but
neither labels any of the 4 routine construction or maintenance sentences as not
humanitarian. "Road blocked due to metro construction work", "Scheduled power cut for
maintenance" and "Old building being demolished" are all labelled infrastructure damage
by both models. Full table in `outputs/kaggle_02/transformer_confounder_probe.csv`.

Not yet run for the transformers: cross-disaster (leave-one-type-out) evaluation.

### Backbone search (notebook 03)

Same recipe as notebook 02 (3 epochs, batch 32, max length 96, one seed); only the
learning rate differs per model. Two T4 GPUs in parallel, 52 minutes in total.

| Model | Learning rate | Test accuracy | Test macro-F1 | Training time |
|---|---|---|---|---|
| BERT-base (reference) | 2e-5 | 0.782 | 0.766 | 12.5 min |
| ModernBERT-base | 5e-5 | 0.784 | 0.762 | 21.6 min |
| DeBERTa-v3-base | 2e-5 | 0.778 | 0.763 | 20.7 min |
| ModernBERT-large | 2e-5 | 0.783 | 0.768 | 51.2 min |
| Ensemble of the three new models | | 0.790 | 0.774 | |

What this shows:

- **Newer and larger backbones do not help on their own.** All four single models are
  within 0.007 macro-F1 of each other, which is inside single-seed noise. The task
  appears to have a ceiling of about 0.77 with plain fine-tuning.
- **More training would not help either.** ModernBERT-large had its best dev score after
  epoch 1 and ModernBERT-base after epoch 2; both declined afterwards.
- **Averaging the three models gives 0.774**, the best number so far, by a small margin.
- **Confounder probe: none of the three new models labels any of the 4 routine
  construction or maintenance sentences as not humanitarian.** Full table in
  `outputs/kaggle_03/backbone_confounder_probe.csv`.

Backbone choice for our method: a base-size model. ModernBERT-large costs four times the
training time of BERT for no measurable gain, which would make ablations with three seeds
impractical on free GPUs.

### Pipeline, first run (notebook 04)

This run used the first 240-sentence draft of the confounder set (batch 1). The set has
since grown to 400 sentences, so these numbers will be replaced once it is verified.

ModernBERT-base, 3 epochs, lr 5e-5, one seed, one component added per run. Each batch is
28 HumAID tweets, plus (in the last three runs) 4 confounder training sentences about one
anchor entity: 2 real and 2 look-alikes. The confounder test split has 30 real reports
and 50 look-alikes about 10 anchor entities that never appear in training.

| Run | HumAID accuracy | HumAID macro-F1 | Real reports kept | Look-alikes rejected |
|---|---|---|---|---|
| `baseline` (HumAID only) | 0.782 | 0.760 | 30 of 30 | 15 of 50 (30%) |
| `plus_confounders` (+ confounder training data) | 0.782 | 0.767 | 29 of 30 | 40 of 50 (80%) |
| `jev` (+ anchor entity, JEV model, gated classifier) | 0.771 | 0.759 | 29 of 30 | 45 of 50 (90%) |
| `jev_contrastive` (+ contrastive loss) | 0.774 | 0.756 | 28 of 30 | 45 of 50 (90%) |

Look-alikes rejected, by type:

| Run | Routine (26) | Figurative (6) | Past or fiction (9) | Drill (9) |
|---|---|---|---|---|
| `baseline` | 12% | 67% | 56% | 33% |
| `plus_confounders` | 73% | 83% | 89% | 89% |
| `jev` | 88% | 67% | 100% | 100% |
| `jev_contrastive` | 85% | 100% | 100% | 89% |

What this shows, and what it does not:

- **The baseline fails where the faculty said it would.** It rejects 3 of 26 routine
  construction and maintenance sentences.
- **Most of the fix comes from the training data.** 160 confounder sentences take
  look-alike rejection from 30% to 80% on entities the model never saw, with no loss on HumAID.
- **The JEV pipeline scores higher again (90%), mainly on routine sentences, but this is
  not yet a reliable difference.** It is 5 sentences out of 50 from one seed, and the
  `plus_confounders` run itself moved between 80% and 90% from one epoch to the next.
- **The contrastive loss shows no benefit in this form.**
- **The JEV runs are about 1 point lower in HumAID accuracy** (0.771 and 0.774 against 0.782).
- The anchor-entity word list finds an entity in 100% of confounder test sentences but
  only 38% of HumAID training tweets; without one the model falls back to the sentence.

Caveats: the confounder set is an unverified draft written by an AI assistant, in one
style; the test split is 80 sentences; one seed per run.

## 4. Roadmap

- [x] Project setup, Kaggle link, data download
- [x] EDA
- [x] Baseline 1: TF-IDF + Logistic Regression, in-distribution and cross-disaster
- [x] Baselines 2 and 3: BERT-base and Twitter-RoBERTa fine-tuning (GPU), official split
- [ ] Cross-disaster evaluation for the transformer baselines
- [x] Confounder set: 400-sentence draft, 50 anchor entities, split 25/25 by entity
- [ ] Confounder set: blind verification by two team members, third resolves; then freeze the test split
- [x] Backbone search: ModernBERT-base, DeBERTa-v3-base, ModernBERT-large, and their ensemble
- [x] Pipeline, first version: anchor entity → JEV model → gated classifier, with a 4-run ablation ladder
- [ ] Rerun the ladder on the verified set with 3 seeds, to tell `plus_confounders` and `jev` apart
- [ ] Improve the pipeline: anchor coverage on HumAID, contrastive loss focused on hard negatives
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
  02_transformer_baselines.py
  03_backbone_search.py
  04_jev_pipeline.py
  kaggle_01/ … kaggle_04/   what gets pushed: kernel-metadata.json + generated .ipynb
src/
  make_notebook.py          converts a "# %%" .py file into a .ipynb
  anchor_entities.py        stage 1: word list that finds and marks the anchor entity
  kaggle_run.py             push → stream live logs → download outputs
confounder_set/             draft confounder set, blind verification sheet, labelling guide
outputs/kaggle_0*/          metrics, plots and logs pulled back from Kaggle
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
