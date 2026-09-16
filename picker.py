"""
picker — a tiny framework for choosing an LLM for a task.

You give it a task (test cases + a way to grade them) and a set of candidate
models. It runs every model over every case and reports the four things that
actually decide the call: quality, latency, cost, and context fit. Then it
recommends the cheapest model that clears your quality bar.

The rule it encodes: quality is a *gate*, not a score to maximize. Pick the
cheapest, fastest model that passes on YOUR task — not the biggest one.

Provider-agnostic: the model call goes through a `backend`. An OpenAI backend
ships below; write your own for any provider by matching the same signature.
The SDK is imported lazily, so graders and math can be tested offline.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol


# --- candidate models -------------------------------------------------------

@dataclass(frozen=True)
class Model:
    id: str
    input_per_mtok: float      # USD per 1M input tokens
    output_per_mtok: float     # USD per 1M output tokens
    context_window: int        # max input tokens


# EDIT THIS for your own set. Prices are USD per 1M tokens and CHANGE OFTEN —
# verify every row against https://openai.com/api/pricing/ before trusting the
# cost column. The values here are a starting point, not a source of truth.
MODELS: dict[str, Model] = {
    "mini": Model("gpt-4o-mini", 0.15,  0.60, 128_000),
    "4o":   Model("gpt-4o",      2.50, 10.00, 128_000),
    "big":  Model("o3",         10.00, 40.00, 200_000),
}


# --- task definition --------------------------------------------------------

Grader = Callable[[str], float]   # model output -> score in [0, 1]


@dataclass
class Case:
    prompt: str
    grade: Grader
    name: str = ""


@dataclass
class Task:
    name: str
    cases: list[Case]
    system: Optional[str] = None


# --- backend (how a model gets called) --------------------------------------

class Backend(Protocol):
    def __call__(self, model_id: str, system: Optional[str],
                 prompt: str, max_tokens: int) -> tuple[str, int, int]:
        """Return (output_text, input_tokens, output_tokens)."""
        ...


class OpenAIBackend:
    """Calls OpenAI Chat Completions. Reads OPENAI_API_KEY from the env.

    Note: some newer models want `max_completion_tokens` instead of `max_tokens`
    and reject `temperature`. If a model 400s, adjust the call here — this one
    method is the only provider-specific code in the framework.
    """

    def __init__(self, client=None):
        self._client = client

    def _c(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def __call__(self, model_id, system, prompt, max_tokens):
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": prompt}]
        r = self._c().chat.completions.create(
            model=model_id, max_tokens=max_tokens, messages=messages,
        )
        text = r.choices[0].message.content or ""
        return text, r.usage.prompt_tokens, r.usage.completion_tokens


# --- graders ----------------------------------------------------------------

def contains(*needles: str, case_insensitive: bool = True) -> Grader:
    def g(out: str) -> float:
        hay = out.lower() if case_insensitive else out
        ok = all((n.lower() if case_insensitive else n) in hay for n in needles)
        return 1.0 if ok else 0.0
    return g


def json_fields(**expected) -> Grader:
    """Partial credit: fraction of expected JSON fields the output got right."""
    def g(out: str) -> float:
        data = extract_json(out)
        if not data:
            return 0.0
        hits = sum(
            1 for k, v in expected.items()
            if str(data.get(k)).strip().lower() == str(v).strip().lower()
        )
        return hits / len(expected)
    return g


def llm_judge(rubric: str, judge_model: str = "gpt-4o",
              backend: Optional[Backend] = None) -> Grader:
    """Grade open-ended output with a model. One extra call per grade."""
    def g(out: str) -> float:
        b = backend or OpenAIBackend()
        system = ('You are a strict grader. Score the answer 0.0-1.0. '
                  'Reply with ONLY JSON: {"score": <float>, "reason": "<short>"}.')
        text, _, _ = b(judge_model, system,
                       f"Rubric:\n{rubric}\n\nAnswer:\n{out}", 400)
        data = extract_json(text) or {}
        try:
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
        except (TypeError, ValueError):
            return 0.0
    return g


# --- helpers ----------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# --- runner -----------------------------------------------------------------

@dataclass
class Result:
    key: str
    model_id: str
    quality: float
    lat_p50: float
    lat_p95: float
    cost_per_call: float
    cost_per_1k: float
    context_window: int
    fits: bool
    calls: int
    errors: int


def evaluate(task: Task, model_keys: list[str], *, repeats: int = 1,
             max_tokens: int = 512, backend: Optional[Backend] = None) -> list[Result]:
    """Run every model over every case `repeats` times; aggregate the metrics."""
    backend = backend or OpenAIBackend()
    results: list[Result] = []

    for key in model_keys:
        m = MODELS[key]
        scores, lats, costs, in_toks = [], [], [], []
        errors = 0

        for case in task.cases:
            for _ in range(repeats):
                t0 = time.perf_counter()
                try:
                    text, in_tok, out_tok = backend(
                        m.id, task.system, case.prompt, max_tokens)
                except Exception:            # noqa: BLE001 - resilient batch runner
                    errors += 1
                    continue
                dt = time.perf_counter() - t0
                cost = in_tok / 1e6 * m.input_per_mtok + out_tok / 1e6 * m.output_per_mtok
                scores.append(case.grade(text))
                lats.append(dt)
                costs.append(cost)
                in_toks.append(in_tok)

        results.append(Result(
            key=key, model_id=m.id,
            quality=statistics.fmean(scores) if scores else 0.0,
            lat_p50=_pct(lats, 50), lat_p95=_pct(lats, 95),
            cost_per_call=statistics.fmean(costs) if costs else 0.0,
            cost_per_1k=(statistics.fmean(costs) * 1000) if costs else 0.0,
            context_window=m.context_window,
            fits=(max(in_toks) < m.context_window) if in_toks else True,
            calls=len(scores), errors=errors,
        ))
    return results


# --- report -----------------------------------------------------------------

def report(results: list[Result], *, quality_bar: float = 0.9) -> Optional[Result]:
    """Print the comparison table and the recommended model."""
    print(f"\n{'model':<10}{'id':<16}{'quality':>9}{'p50 s':>8}{'p95 s':>8}"
          f"{'$/1k':>9}{'ctx':>8}{'fits':>6}")
    print("-" * 74)
    for r in sorted(results, key=lambda x: x.cost_per_1k):
        print(f"{r.key:<10}{r.model_id:<16}{r.quality:>9.2f}{r.lat_p50:>8.2f}"
              f"{r.lat_p95:>8.2f}{r.cost_per_1k:>9.2f}{r.context_window // 1000:>7}K"
              f"{('yes' if r.fits else 'NO'):>6}")

    passing = [r for r in results if r.quality >= quality_bar and r.fits]
    print("-" * 74)
    if not passing:
        best = max(results, key=lambda r: r.quality)
        print(f"No model clears the quality bar ({quality_bar:.2f}). "
              f"Best was {best.key} at {best.quality:.2f}. "
              f"Improve the prompt, or lower the bar if the task allows.")
        return None

    pick = min(passing, key=lambda r: r.cost_per_1k)
    fastest = min(passing, key=lambda r: r.lat_p50)
    print(f"Recommended: {pick.key} ({pick.model_id}) — cheapest model that "
          f"clears {quality_bar:.2f} quality and fits.")
    if fastest.key != pick.key:
        print(f"If latency matters more than cost: {fastest.key} "
              f"(p50 {fastest.lat_p50:.2f}s).")
    return pick
