"""Stage 1 of the pipeline: find the anchor entity of a tweet with a word list.

The anchor entity is the thing the tweet is about (a bridge, the power supply, a shelter).
`tag(text)` wraps the first mention in [E] ... [/E] markers so the model can read the
entity together with the event around it.

Run this file directly to print coverage on the local data:
    python src/anchor_entities.py
"""
import re

TERMS = [
    # utilities
    r"power lines?", r"power poles?", r"power grid", r"power supply", r"power", r"electricity", r"electric poles?",
    r"wires?", r"transformers?", r"substations?", r"water supply", r"drinking water", r"water mains?", r"water",
    r"gas pipelines?", r"gas lines?", r"pipelines?", r"gas", r"mobile towers?", r"cell towers?", r"towers?",
    r"network", r"internet", r"wifi", r"phone lines?", r"cell service",
    # transport
    r"bridges?", r"roads?", r"highways?", r"freeways?", r"interstate", r"nh-?\d+", r"streets?", r"flyovers?",
    r"overpass(?:es)?", r"railway tracks?", r"rail tracks?", r"tracks?", r"railways?", r"trains?", r"metro",
    r"subway", r"airports?", r"runways?", r"tunnels?", r"ports?", r"harbou?rs?", r"ferr(?:y|ies)",
    # structures
    r"school buildings?", r"buildings?", r"houses?", r"homes?", r"schools?", r"hospitals?", r"walls?", r"roofs?",
    r"dams?", r"levees?", r"embankments?", r"church(?:es)?", r"mosques?", r"temples?", r"hotels?",
    r"apartments?", r"malls?", r"factor(?:y|ies)", r"markets?", r"shops?",
    # hazards and response
    r"wildfires?", r"fires?", r"smoke", r"sirens?", r"evacuat\w+", r"shelters?", r"ambulances?", r"rescue\w*",
    r"ndrf",
]
# Longest alternative first, so "power lines" wins over "power" at the same position
ANCHOR_RE = re.compile(r"\b(?:" + "|".join(sorted(TERMS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def find(text):
    """Return (start, end) of the first anchor entity mention, or None."""
    m = ANCHOR_RE.search(text)
    return (m.start(), m.end()) if m else None


def tag(text):
    """Wrap the first anchor entity mention in [E] ... [/E]; unchanged if there is none."""
    span = find(text)
    if span is None:
        return text
    s, e = span
    return f"{text[:s]}[E] {text[s:e]} [/E]{text[e:]}"


if __name__ == "__main__":
    import os
    from collections import Counter

    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conf = pd.read_csv(os.path.join(root, "confounder_set", "confounder_set_v1_draft.csv"))
    found = conf["text"].map(lambda t: find(t) is not None)
    print(f"confounder set: anchor found in {found.mean():.1%} of {len(conf)} sentences")
    for _, r in conf[~found].iterrows():
        print("   no anchor:", r["id"], "|", r["text"])

    path = os.path.join(root, "data", "humaid.csv")
    if os.path.exists(path):
        hum = pd.read_csv(path)
        mentions = hum["text"].map(lambda t: (lambda s: t[s[0]:s[1]].lower() if s else None)(find(t)))
        print(f"\nHumAID: anchor found in {mentions.notna().mean():.1%} of {len(hum)} tweets")
        print(hum.assign(found=mentions.notna()).groupby("label")["found"].mean().round(3).to_string())
        print("\nmost common anchors:", Counter(mentions.dropna()).most_common(25))
