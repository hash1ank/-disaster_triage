# Confounder set

Sentences that share an **anchor entity** (bridge, road, power lines, shelter, ...) but
differ in reality: some report a real disaster, others only look like one.

**Status: v2, reviewed, test split frozen on 20 October 2026.** All 400 sentences were
drafted by an AI assistant in tweet style and then reviewed by the team, who accepted
them as they are. Frozen file: `confounder_set_v2.csv`, SHA-256
`60e0447b5ff2344df38019725b27f5e20ff9a1a5cdc678fb9e39cad6ad67d9e1`. No sentence in the
test split is to be edited from here on; new sentences go into a v3 file.

## What is in the set

- 50 anchor entities, 8 sentences each: 3 real disaster reports and 5 look-alikes.
- 400 sentences: 150 real, 250 look-alikes (133 routine, 49 figurative, 39 past or fiction, 29 drill).
- Split **by anchor entity**: 25 entities (200 sentences) for training and 25 entities
  (200 sentences) for testing. Test entities never appear in training, so a model cannot
  pass by memorising "bridge + repair"; it has to learn the cause and context.

The set was drafted in two batches:

- **Batch 1** (`C01` to `C30`, 240 sentences): the version used by notebook 04.
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

## Review record

- The team read the draft and agreed with almost all of it. No individual corrections
  were recorded, so the file is unchanged from the draft.
- The review was a single informal pass, not the blind double annotation planned
  earlier, so **no inter-annotator agreement figure exists**. The report should say the
  labels were drafted by an AI assistant and reviewed by the authors.
- Model predictions on batch 1 (from notebook 04) existed at the time of the review.

Rules from here on:

- **The test split is frozen.** Sentences are not edited or removed after seeing results.
- **Judge sentences, not cue words.** A sentence with "repair" in it can still be a real
  report, and one with "flood" in it can still be a look-alike.
- **The category of a real report is a second-order judgement.** The main thing tested
  is real versus not.

## Open questions for the faculty

- A drill or rehearsal announcement is labelled `not_humanitarian` here. It could be
  argued to be `caution_and_advice`.
- A historical disaster is labelled `not_humanitarian`. On an anniversary of a major
  event this may be debatable.

## Known limitations

- Written by one author, so the style is narrower than real tweets.
- Reviewed informally; no inter-annotator agreement was measured.
- English only, mixing Indian and international contexts.
- Almost every sentence contains an anchor entity, while only about 39% of HumAID tweets do.

## How this feeds the model

Each anchor entity gives ready-made training triplets: a sentence, another with the same
reality, and a hard negative with the same entity and the other reality. During training
every batch carries four sentences about one entity, two real and two look-alikes.
