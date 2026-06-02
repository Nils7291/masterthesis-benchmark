"""
Shared evaluation metrics for the encoder vs. LLM benchmark.

Both notebooks call into these functions to ensure that the encoder and the
LLM system are evaluated under identical statistical machinery. Any change
to a metric or to the bootstrap procedure changes both systems together,
which preserves comparability.

References
----------
Efron, B. and Tibshirani, R. (1993). An Introduction to the Bootstrap.
    Chapman & Hall, New York.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# Metrics computed on every run. Keep the names stable across notebooks.
METRIC_FUNCTIONS = {
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "precision_macro": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
    "precision_weighted": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall_macro": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
    "recall_weighted": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "f1_macro": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
    "f1_weighted": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
}


def compute_point_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    """Compute all point metrics for one run."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {name: float(fn(y_true, y_pred)) for name, fn in METRIC_FUNCTIONS.items()}


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap confidence intervals for all metrics.

    Resamples the (y_true, y_pred) pairs with replacement n_resamples times,
    recomputes each metric on each resample, and returns the percentile
    interval at the requested confidence level.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Gold and predicted labels, aligned by position.
    n_resamples : int
        Number of bootstrap resamples. Default 1000 as configured in the
        thesis methodology.
    confidence : float
        Two-sided confidence level. Default 0.95.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Per metric a dict with keys 'point', 'ci_low', 'ci_high'.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, n, size=(n_resamples, n))

    bootstrap_scores: Dict[str, List[float]] = {name: [] for name in METRIC_FUNCTIONS}
    for resample in indices:
        y_t = y_true[resample]
        y_p = y_pred[resample]
        for name, fn in METRIC_FUNCTIONS.items():
            bootstrap_scores[name].append(float(fn(y_t, y_p)))

    alpha = (1.0 - confidence) / 2.0
    low_q, high_q = 100 * alpha, 100 * (1.0 - alpha)

    point = compute_point_metrics(y_true, y_pred)
    return {
        name: {
            "point": point[name],
            "ci_low": float(np.percentile(bootstrap_scores[name], low_q)),
            "ci_high": float(np.percentile(bootstrap_scores[name], high_q)),
        }
        for name in METRIC_FUNCTIONS
    }


def stability_score(predictions_per_run: List[Sequence[int]]) -> Tuple[float, np.ndarray]:
    """
    Stability score across N runs on identical inputs.

    A prediction is considered stable for a given input if all N runs return
    the same label for that input. The stability score is the fraction of
    inputs for which this holds.

    Parameters
    ----------
    predictions_per_run : list of array-like
        N sequences of predictions, all of identical length, aligned by
        position.

    Returns
    -------
    score : float
        Fraction of inputs with identical predictions across all runs.
    is_stable_per_input : np.ndarray of bool
        Length-M array marking each input as stable or not.
    """
    if len(predictions_per_run) < 2:
        raise ValueError("Stability requires at least two runs")

    matrix = np.array([np.asarray(p) for p in predictions_per_run])
    if matrix.ndim != 2:
        raise ValueError("All runs must have the same length")

    is_stable_per_input = np.all(matrix == matrix[0], axis=0)
    score = float(np.mean(is_stable_per_input))
    return score, is_stable_per_input


def aggregate_runs_to_dataframe(
    runs_metrics: List[Dict[str, float]], run_names: List[str] | None = None
) -> pd.DataFrame:
    """
    Convert a list of per-run metric dicts into a DataFrame with one row per
    run. Useful for reporting variability across the N stability runs.
    """
    if run_names is None:
        run_names = [f"run_{i + 1}" for i in range(len(runs_metrics))]
    df = pd.DataFrame(runs_metrics, index=run_names)
    df.index.name = "run"
    return df


def summarise_metric_across_runs(
    runs_metrics: List[Dict[str, float]],
) -> pd.DataFrame:
    """
    Mean and standard deviation of each metric across the N runs.
    """
    df = aggregate_runs_to_dataframe(runs_metrics)
    summary = pd.DataFrame(
        {
            "mean": df.mean(axis=0),
            "std": df.std(axis=0, ddof=1) if len(df) > 1 else 0.0,
            "min": df.min(axis=0),
            "max": df.max(axis=0),
        }
    )
    summary.index.name = "metric"
    return summary
