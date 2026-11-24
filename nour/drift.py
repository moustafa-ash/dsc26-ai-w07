"""Population Stability Index and a drift report. You write this file."""
from __future__ import annotations


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index between two 1-D numeric samples.

    Drop NaNs from both; `ValueError` if either is then empty. Interior bin edges are
    `np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1])`; a value's bin is
    `np.searchsorted(edges, x, side="right")`. Compute the proportion of each side in each bin,
    floor each proportion at `eps`, then return `sum((c - r) * ln(c / r))` over the bins.
    """
    raise NotImplementedError


def categorical_psi(reference, current, eps: float = 1e-4) -> float:
    """PSI for a categorical column.

    Drop missing values, compare as strings. Categories are the union of both sides (so a
    category that is new in `current` still counts). Same flooring and formula as `psi`.
    """
    raise NotImplementedError


def drift_report(ref_df, cur_df, numeric: list[str], categorical: list[str],
                 threshold: float = 0.2) -> dict:
    """Drift report over the given numeric and categorical columns.

    Returns `{"columns": {col: {"psi", "missing_rate_ref", "missing_rate_cur", "drifted"}},
    "drifted_columns": [...]}`. `drifted` is true when `psi >= threshold` or the column's
    missing rate changed by 0.1 or more (absolute). `drifted_columns` is sorted.
    """
    raise NotImplementedError
