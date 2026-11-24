"""Population Stability Index and a drift report. You write this file."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _psi_from_counts(reference, current, eps):
    r = np.maximum(reference / reference.sum(), eps)
    c = np.maximum(current / current.sum(), eps)
    return float(np.sum((c - r) * np.log(c / r)))


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index between two 1-D numeric samples.

    Drop NaNs from both; `ValueError` if either is then empty. Interior bin edges are
    `np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1])`; a value's bin is
    `np.searchsorted(edges, x, side="right")`. Compute the proportion of each side in each bin,
    floor each proportion at `eps`, then return `sum((c - r) * ln(c / r))` over the bins.
    """
    ref = np.asarray(pd.Series(reference).dropna(), dtype=float)
    cur = np.asarray(pd.Series(current).dropna(), dtype=float)
    if not ref.size or not cur.size:
        raise ValueError("Both samples must contain non-missing values")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1])
    bins = edges.size + 1
    ref_counts = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=bins)
    cur_counts = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=bins)
    return _psi_from_counts(ref_counts, cur_counts, eps)


def categorical_psi(reference, current, eps: float = 1e-4) -> float:
    """PSI for a categorical column.

    Drop missing values, compare as strings. Categories are the union of both sides (so a
    category that is new in `current` still counts). Same flooring and formula as `psi`.
    """
    ref = pd.Series(reference).dropna().astype(str).value_counts()
    cur = pd.Series(current).dropna().astype(str).value_counts()
    if ref.empty or cur.empty:
        raise ValueError("Both samples must contain non-missing values")
    ref, cur = ref.align(cur, join="outer", fill_value=0)
    return _psi_from_counts(ref.to_numpy(), cur.to_numpy(), eps)


def drift_report(ref_df, cur_df, numeric: list[str], categorical: list[str],
                 threshold: float = 0.2) -> dict:
    """Drift report over the given numeric and categorical columns.

    Returns `{"columns": {col: {"psi", "missing_rate_ref", "missing_rate_cur", "drifted"}},
    "drifted_columns": [...]}`. `drifted` is true when `psi >= threshold` or the column's
    missing rate changed by 0.1 or more (absolute). `drifted_columns` is sorted.
    """
    columns = {}
    for names, measure in ((numeric, psi), (categorical, categorical_psi)):
        for name in names:
            ref, cur = ref_df[name], cur_df[name]
            score = measure(ref, cur)
            missing_ref, missing_cur = float(ref.isna().mean()), float(cur.isna().mean())
            change = abs(missing_ref - missing_cur)
            columns[name] = {
                "psi": score, "missing_rate_ref": missing_ref, "missing_rate_cur": missing_cur,
                "drifted": score >= threshold or change >= 0.1 or math.isclose(change, 0.1, abs_tol=1e-12, rel_tol=0),
            }
    return {"columns": columns, "drifted_columns": sorted(k for k, v in columns.items() if v["drifted"])}
