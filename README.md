# llm-model-picker

A tiny framework for choosing an LLM for a task. You give it a task (test cases
plus a way to grade them) and a set of candidate models. It runs every model
over every case and reports the four things that actually decide the call:

- **quality** — how well it does *your* task (not a public benchmark)
- **latency** — measured yourself, as p50 and p95 (not the mean)
- **cost** — from real token usage on your prompts, not list price
- **context** — does your largest input fit the model's window?

Then it recommends the cheapest model that clears your quality bar.

## The rule it encodes

Quality is a **gate**, not a score to maximize. Reaching for the biggest model
by default is how you overpay. Cheaping out to the smallest is how you ship
wrong answers. Pick the cheapest, fastest model that **passes on your task.**

## Security: where the key goes

The OpenAI key belongs in **your shell only** — never in this repo, never in a
web page, never pasted into a chat. A key in client-side code is public the
moment it ships and can be drained. If you want a browser demo, it must call a
**server** you control that holds the key and rate-limits requests.

```bash
export OPENAI_API_KEY=...   # a key you created and keep private
```

`.env` is gitignored. Rotate any key that has ever been shared.

## Run it

```bash
pip install -r requirements.txt

python example.py                  # real run: a handful of cents
python example.py --mock           # offline demo, no key, no spend
python example.py --models mini 4o --repeats 3 --quality-bar 0.9
```

## Verify the prices

`MODELS` in `picker.py` has each model's price and context window. **Prices
change often — check every row against <https://openai.com/api/pricing/>
before trusting the cost column.** Add, remove, or rename candidates freely.

## Use it for your own task

```python
from picker import Case, Task, evaluate, report, json_fields, contains, llm_judge

task = Task(
    name="my-task",
    system="...",
    cases=[
        Case("prompt here", json_fields(label="billing")),      # deterministic
        Case("open-ended prompt", llm_judge("rubric: ...")),    # graded by a model
        Case("...", contains("expected phrase")),
    ],
)

report(evaluate(task, ["mini", "4o", "big"], repeats=3), quality_bar=0.9)
```

Graders return a score in `[0, 1]`: `json_fields` gives partial credit per
field, `contains` checks for phrases, `llm_judge` grades open-ended output.
The model call lives in one place — `OpenAIBackend` in `picker.py`; swap it for
any provider by matching the same signature.

## Files

- `picker.py` — the framework (models, backend, graders, runner, report)
- `example.py` — a worked example (support-ticket triage) + an offline mock
- `requirements.txt` — just the `openai` SDK
