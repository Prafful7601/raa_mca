"""Split-conformal risk control (C3): pick the smallest confidence threshold
lambda such that the selective error rate among ANSWERED calibration claims is
guaranteed, with high probability under exchangeability, to be <= alpha on
future data. Uses the (errors+1)/(n+1) finite-sample upper confidence bound
from the conformal risk control literature (Angelopoulos et al., 2022-style
"add-one" correction), applied over a monotone (in lambda) selective-risk loss.
Every claim with agreement_score < lambda abstains and is routed to review.
"""
from __future__ import annotations
import numpy as np


def _risk_ucb(n_wrong: int, n_answered: int) -> float:
    if n_answered == 0:
        return 0.0
    return (n_wrong + 1) / (n_answered + 1)


def calibrate_threshold(scores: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """scores: confidence/agreement in [0,1] per calibration claim.
    correct: bool array, True if the method's verdict matched the gold label.
    Returns lambda in [0, 1+eps]; answer iff score >= lambda, else abstain."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    grid = np.unique(np.concatenate([scores, [0.0, 1.0 + 1e-9]]))
    grid.sort()
    risk_ucb = np.empty_like(grid)
    for i, lam in enumerate(grid):
        mask = scores >= lam
        n = int(mask.sum())
        wrong = int((~correct[mask]).sum()) if n else 0
        risk_ucb[i] = _risk_ucb(wrong, n)
    # enforce monotonic safety: suffix-max so that any lambda' >= chosen lambda is also safe
    suffix_max = np.maximum.accumulate(risk_ucb[::-1])[::-1]
    safe_idx = np.where(suffix_max <= alpha)[0]
    if len(safe_idx) == 0:
        return float(grid[-1])  # no threshold achieves alpha -> abstain on everything
    return float(grid[safe_idx[0]])


def apply_threshold(scores: np.ndarray, lam: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) >= lam


def realised_risk_coverage(scores: np.ndarray, correct: np.ndarray, lam: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    answered = scores >= lam
    n_answered = int(answered.sum())
    n_total = len(scores)
    n_wrong = int((~correct[answered]).sum()) if n_answered else 0
    risk = n_wrong / n_answered if n_answered else 0.0
    coverage = n_answered / n_total if n_total else 0.0
    return {
        "lambda": lam, "n_total": n_total, "n_answered": n_answered,
        "n_abstained": n_total - n_answered, "n_wrong": n_wrong,
        "realised_risk": risk, "coverage": coverage,
    }
