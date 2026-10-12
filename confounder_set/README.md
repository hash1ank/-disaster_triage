# Confounder set

Sentences that share an **anchor entity** (bridge, road, power lines, shelter, ...) but
differ in reality: some report a real disaster, others only look like one.

**Status: v2 draft, unverified.** All 400 sentences were drafted by an AI assistant in
tweet style. None has been checked by the team yet. Do not report results on this file
as a benchmark until verification is done and the test split is frozen.

## Files

| File | What it is |
|---|---|
| `confounder_set_v2_draft.csv` | The set, with the draft `reality` and `label` for each sentence |
| `verification_sheet_blind.csv` | The same 400 sentences, shuffled, with no labels. Verifiers fill this in |
| `verification_key.csv` | Maps the blind sheet's `vid` back to the set's `id` |

## What is in the set

- 50 anchor entities, 8 sentences each: 3 real disaster reports and 5 look-alikes.
- 400 sentences: 150 real, 250 look-alikes (133 routine, 49 figurative, 39 past or fiction, 29 drill).
- Split **by anchor entity**: 25 entities (200 sentences) for training and 25 entities
  (200 sentences) for testing. Test entities never appear in training, so a model cannot
  pass by memorising "bridge + repair"; it has to learn the cause and context.

The set was drafted in two batches:

- **Batch 1** (`C01` to `C30`, 240 sentences): the version used by notebook 04's first run.
- **Batch 2** (`C31` to `C50`, 160 sentences): written to be harder. Cue words are
  deliberately crossed: some real reports contain routine words ("repair crews cant even
  reach the spot"), and some look-alikes contain disaster words ("any port in a storm",
  "rebuilt three times after earthquakes"). Fewer look-alikes say "drill" or
  "maintenance" outright. 15 of its 20 entities go to the test split.

## The five reality types

| `reality` | Definition | Example |
|---|---|---|
| `real_disaster` | Reports damage, need or danger caused by an ongoing or very recent disaster | Bridge near Aluva collapsed after the flood |
| `routine` | Planned or everyday: construction, repair, maintenance, demolition, inauguration, ordinary faults, ordinary life | Bridge closed for repair work till Monday |
| `figurative` | Disaster or entity word used as a metaphor or slang | Burned that bridge with my ex long ago |
| `past_or_fiction` | Historical event, anniversary, film, series, or memory; nothing is happening now | On this day in 1940 the Tacoma Narrows bridge collapsed |
| `drill` | Mock drill, test, rehearsal, training exercise or demo | Mock drill on the bridge today 11am |

`label` is the HumAID category. It is `not_humanitarian` for every look-alike. For real
reports it is one of: `infrastructure_and_utility_damage`, `requests_or_urgent_needs`,
`injured_or_dead_people`, `missing_or_found_people`, `displaced_people_and_evacuations`,
`caution_and_advice`, `rescue_volunteering_or_donation_effort`,
`sympathy_and_support`, `other_relevant_information`.

## How to verify

Two team members label every sentence independently; the third settles disagreements.

1. Each verifier makes their own copy of `verification_sheet_blind.csv`, named
   `verification_<yourname>.csv`. Do not open the draft file or the key while labelling.
2. For each row fill in:
   - `your_reality`: one of the five types above.
   - `your_label`: the HumAID category, only if you chose `real_disaster`.
   - `natural`: `yes`, or `no` if the sentence is unnatural, unclear even to a person, or could fairly be read both ways.
   - `notes`: anything worth recording, such as a better wording.
3. When both copies are done, they are merged with the key and compared with the draft.
   Agreement between the two verifiers is computed and goes in the report.
4. The third member decides every row where the two disagree with each other or with
   the draft. Rows marked unnatural by both are rewritten or dropped.
5. The test split is then frozen. No test sentence is edited after that.

Rules that keep the test fair:

- **Do not look at model predictions while verifying.** `outputs/kaggle_04/` contains
  the predictions of four models on 80 of these sentences. Editing test sentences with
  those in mind would tune the set to the models.
- **Judge the sentence, not the cue word.** A sentence with "repair" in it can still be
  a real report, and one with "flood" in it can still be a look-alike.
- **A sentence with no deciding cue is a bad sentence.** "Road blocked near the station"
  alone cannot be labelled; mark it `natural = no`.
- **The category of a real report is a second-order judgement.** Several could fairly be
  two categories (damage or urgent request). The main thing tested is real versus not.

## Open questions for the faculty

- A drill or rehearsal announcement is labelled `not_humanitarian` here. It could be
  argued to be `caution_and_advice`.
- A historical disaster is labelled `not_humanitarian`. On an anniversary of a major
  event this may be debatable.

## Known limitations

- Written by one author, so the style is narrower than real tweets.
- English only, mixing Indian and international contexts.
- Almost every sentence contains an anchor entity, while only about 39% of HumAID tweets do.

## How this feeds the model

Each anchor entity gives ready-made training triplets: a sentence, another with the same
reality, and a hard negative with the same entity and the other reality. During training
every batch carries four sentences about one entity, two real and two look-alikes.
