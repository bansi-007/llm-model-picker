"""
Example: pick an OpenAI model for support-ticket triage.

The model reads a customer message and returns JSON:
  {"category": billing|bug|feature_request|account, "urgent": true|false}

Cases run easy -> tricky (negation, buried signals, inferred urgency), which is
where a small model slips and the choice stops being obvious.

Run it:
  pip install -r requirements.txt
  export OPENAI_API_KEY=...          # a FRESH key, in your shell only
  python example.py                  # real run: a handful of cents
  python example.py --mock           # offline demo, no key, no spend
  python example.py --models mini 4o --repeats 3 --quality-bar 0.9

Never paste your key into a chat, a file, or the widget. See README.
"""

import argparse

from picker import Case, Task, evaluate, json_fields, report

SYSTEM = (
    "You triage customer support messages. Reply with ONLY a JSON object, no prose:\n"
    '{"category": one of ["billing","bug","feature_request","account"], '
    '"urgent": true or false}\n'
    "urgent = true only if the customer is blocked right now or money is affected."
)

TASK = Task(
    name="support-triage",
    system=SYSTEM,
    cases=[
        Case("I was charged twice for my subscription this month.",
             json_fields(category="billing"), name="double-charge"),
        Case("The app crashes every time I open the Settings page.",
             json_fields(category="bug"), name="crash"),
        Case("Please add a dark mode, it would be easier on the eyes at night.",
             json_fields(category="feature_request"), name="dark-mode"),
        Case("I can't log in and I have a customer demo in 15 minutes.",
             json_fields(category="account", urgent=True), name="locked-out"),
        Case("The crash I reported is fixed now, thanks. While I'm here, how do "
             "I export all my data to CSV?",
             json_fields(category="account", urgent=False), name="decoy-crash"),
        Case("You charged my card but my invoice shows $0 and all my premium "
             "features just disappeared before a client meeting.",
             json_fields(category="billing", urgent=True), name="buried-urgent"),
    ],
)


# --- mock backend (offline demo / test; no API, no spend) -------------------

def mock_backend(model_id, system, prompt, max_tokens):
    """Fakes a model call. Bigger models get more tricky cases right and run a
    bit slower. Illustrative only — not a real model comparison."""
    import random
    import time

    profile = {
        "gpt-4o-mini": dict(easy=0.98, tricky=0.55, lat=0.4, out=40),
        "gpt-4o":      dict(easy=0.99, tricky=0.86, lat=0.9, out=45),
        "o3":          dict(easy=1.00, tricky=0.97, lat=2.5, out=50),
    }[model_id]

    tricky = any(w in prompt for w in ("15 minutes", "fixed now", "$0"))
    p = profile["tricky"] if tricky else profile["easy"]
    time.sleep(profile["lat"] * random.uniform(0.7, 1.3) * 0.05)  # scaled down

    if "15 minutes" in prompt:
        good = '{"category":"account","urgent":true}'
    elif "fixed now" in prompt:
        good = '{"category":"account","urgent":false}'
    elif "$0" in prompt:
        good = '{"category":"billing","urgent":true}'
    elif "charged twice" in prompt:
        good = '{"category":"billing","urgent":false}'
    elif "crashes" in prompt:
        good = '{"category":"bug","urgent":false}'
    else:
        good = '{"category":"feature_request","urgent":false}'
    bad = '{"category":"bug","urgent":false}'

    out = good if random.random() < p else bad
    return out, 90, profile["out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mini", "4o", "big"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--quality-bar", type=float, default=0.9)
    ap.add_argument("--mock", action="store_true", help="offline demo, no API")
    args = ap.parse_args()

    backend = mock_backend if args.mock else None
    results = evaluate(TASK, args.models, repeats=args.repeats, backend=backend)
    report(results, quality_bar=args.quality_bar)


if __name__ == "__main__":
    main()
