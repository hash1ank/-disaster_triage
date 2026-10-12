# Confounder set

Sentences that share an **anchor entity** (bridge, road, power lines, siren, ...) but
differ in reality: some report a real disaster, others only look like one.

**Status: v1 draft, unverified.** All 240 sentences in `confounder_set_v1_draft.csv` were
drafted by an AI assistant in tweet style. None has been checked by the team yet. Do not
report results on this file as a benchmark until the `verified` column is filled in.

## What is in the file

- 30 anchor entities, 8 sentences each: 3 real disaster reports and 5 look-alikes.
- 240 sentences: 90 real, 150 look-alikes (74 routine, 29 figurative, 26 past or fiction, 21 drill).
- Split **by anchor entity**: 20 entities (160 sentences) for training, 10 entities
  (80 sentences) for testing. The test entities never appear in training, so a model
  cannot pass by memorising "bridge + repair"; it has to learn the cause and context.

Test entities: flyover, airport runway, tunnel, school building, embankment, substation,
gas pipeline, mobile tower, shelter, ambulance.

## Columns

| Column | Meaning |
|---|---|
| `id` | `C<entity number>-<sentence number>` |
| `anchor_entity` | The thing the sentence is about |
| `entity_type` | transport, structure, utility, or hazard |
| `split` | train or test (fixed per entity) |
| `reality` | See the table below |
| `label` | HumAID category. Always `not_humanitarian` unless `reality` is `real_disaster` |
| `text` | The sentence |
| `verified` | **To fill in:** `yes`, `fix`, or `drop` |
| `corrected_reality`, `corrected_label` | **To fill in** when `verified` is `fix` |
| `verifier`, `notes` | Who checked it, and why if changed |

## The five reality types

| `reality` | Definition | Example |
|---|---|---|
| `real_disaster` | Reports damage, need or danger caused by an ongoing or very recent disaster | Bridge near Aluva collapsed after the flood |
| `routine` | Planned or everyday: construction, repair, maintenance, demolition, inauguration, ordinary faults | Bridge closed for repair work till Monday |
| `figurative` | Disaster or entity word used as a metaphor or slang | Burned that bridge with my ex long ago |
| `past_or_fiction` | Historical event, anniversary, film, series, or memory; nothing is happening now | On this day in 1940 the Tacoma Narrows bridge collapsed |
| `drill` | Mock drill, test siren, training exercise or demo | Mock drill on the bridge today 11am |

## How to verify (two people per sentence)

1. Read the sentence without looking at `reality` or `label`. Decide both yourself.
2. Compare with the draft. If you agree, write `yes`. If not, write `fix` and fill in the corrected columns.
3. Write `drop` if the sentence is unnatural, ambiguous even to a human, or gives the answer away too cheaply.
4. Where the two verifiers disagree, the third team member decides. Record the agreement rate; it goes in the report.

Things to watch for:

- **Sentences with no cause stated.** "Road blocked near the station" alone is ambiguous. Each sentence should contain a cue that settles it (after the flood / for repair work).
- **Giveaway words.** If every look-alike says "drill" or "maintenance" the test is too easy. Rewrite some so the cue is less obvious.
- **The `label` of real sentences is a second-order judgement.** Several could fairly be two categories (damage vs urgent request). The main thing being tested is real versus not real.
- **Open question for the faculty:** a drill announcement is labelled `not_humanitarian` here. It could be argued to be `caution_and_advice`.

## Known limitations of this draft

- Written by one author, so the style is narrower than real tweets.
- Indian and international contexts are mixed, but only English.
- 240 sentences is a start. The target is 300 to 500 verified sentences; add new anchor entities rather than more sentences for existing ones.

## How this feeds the model

Each anchor entity gives ready-made training triplets: an anchor sentence, a positive
with the same reality, and a hard negative with the same entity but the other reality.
