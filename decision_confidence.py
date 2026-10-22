"""Confidence statistics for System One / Jev decision answers.

The hosted API returns `confidence` on choice and score answers already; these
are the exact published formulas, reproduced so you can (a) compute confidence
for a local/open-weights model that only returns `probabilities`, (b) compute a
noul confidence, which the API does not return, and (c) substitute a different
statistic when confidence is the wrong summary for your problem.

Verified against every worked example in https://docs.typesafe.ai/confidence
(2026-10-22). See docs/DECISION_MODELS.md §4.
"""
from __future__ import annotations

__all__ = [
    "noul_confidence",
    "choice_confidence",
    "score_confidence",
    "top_probability",
    "top_to_second_ratio",
]


def noul_confidence(p: float) -> float:
    """Distance from 0.5, on the same 0-1 scale as choice/score confidence.

    A noul answer carries no confidence field: with only two outcomes the
    probability describes the distribution completely. Use this only when you
    need to gate nouls and choices with one code path.

    NOTE: not the same as `p if p >= 0.5 else 1 - p` (which maps to 0.5-1.0).
    Both rank identically; they are not interchangeable in an ECE figure.
    """
    return abs(2.0 * p - 1.0)


def choice_confidence(probabilities: list[float]) -> float:
    """How far the top option sits above an even split of 1/n.

    0 = even split, 1 = all mass on one option. Only the top probability
    counts: (0.6,0.3,0.1) and (0.6,0.2,0.2) both give 0.4.
    """
    n = len(probabilities)
    if n < 2:
        raise ValueError("need at least 2 options")
    return (max(probabilities) - 1 / n) / (1 - 1 / n)


def score_confidence(probabilities: list[float]) -> float:
    """Distance-weighted spread; index position is the level number.

    Levels are ordered, so mass on a neighbouring level costs less than mass
    further away: (0,0.5,0.5) -> 0.25, (0.5,0,0.5) -> 0.0. Floored at 0 when
    the answer is at least as spread out as a uniform distribution.

    Pass probabilities ordered by level. From an API answer:
        levels = sorted(answer.probabilities)           # keys may be strings
        score_confidence([answer.probabilities[k] for k in levels])
    """
    n = len(probabilities)
    if n < 2:
        raise ValueError("need at least 2 levels")
    m = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even_spread)


def top_probability(probabilities: list[float]) -> float:
    """P(selected option). Reads directly as "how likely is the answer".

    Its meaning depends on option count -- 0.5 is weak among 2 options and
    strong among 10 -- so set its threshold per question, not globally.
    """
    return max(probabilities)


def top_to_second_ratio(probabilities: list[float]) -> float:
    """How clearly the winner beats the runner-up; ignores the rest of the tail.

    Often the statistic that matters, since many real decisions come down to
    the top two candidates. Returns inf when the runner-up has zero mass.
    """
    if len(probabilities) < 2:
        raise ValueError("need at least 2 options")
    top, second = sorted(probabilities, reverse=True)[:2]
    return float("inf") if second == 0 else top / second


def _selftest() -> None:
    """Published vendor values. Tolerance 0.006: the docs round to 2dp
    (0.415 is published as 0.42, 0.355 as 0.35, 0.9167 as 0.92)."""
    cases = [
        (choice_confidence([0.6, 0.3, 0.1]), 0.40),
        (choice_confidence([0.6, 0.2, 0.2]), 0.40),
        (choice_confidence([0.9, 0.06, 0.04]), 0.85),
        (choice_confidence([0.04, 0.35, 0.61]), 0.42),
        (score_confidence([0.0, 0.57, 0.43]), 0.35),
        (score_confidence([0.0, 0.5, 0.5]), 0.25),
        (score_confidence([0.5, 0.0, 0.5]), 0.00),
        (score_confidence([0.0, 0.91, 0.09]), 0.87),
        (score_confidence([0.0, 0.76, 0.24]), 0.64),
        (score_confidence([0, 0.05, 0.9, 0.05, 0]), 0.92),
        (noul_confidence(0.5), 0.00),
        (noul_confidence(0.93), 0.86),
    ]
    for got, want in cases:
        assert abs(got - want) < 0.006, f"{got:.4f} != {want}"
    print(f"ok - {len(cases)} published values reproduced")


if __name__ == "__main__":
    _selftest()
