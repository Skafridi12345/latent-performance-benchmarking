from __future__ import annotations

import numpy as np
import pandas as pd


def compute_transition_matrix(
    rolling: pd.DataFrame,
    *,
    horizon_months: int = 12,
    date_col: str = "window_end",
    n_quantiles: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a quintile transition matrix and aggregate transition metrics."""

    df = rolling.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["period"] = df[date_col].dt.to_period("M")

    if "quintile" not in df.columns:
        if "AE" not in df.columns:
            raise ValueError("rolling data must contain either quintile or AE.")
        df["quintile"] = df.groupby(date_col)["AE"].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop") + 1
        )

    df = df.dropna(subset=["quintile"])
    df["quintile"] = df["quintile"].astype(int)

    transitions: list[tuple[int, int]] = []
    periods = sorted(df["period"].unique())
    available = set(periods)
    for p0 in periods:
        p1 = p0 + int(horizon_months)
        if p1 not in available:
            continue
        d0 = df[df["period"] == p0].set_index("portfolio")["quintile"]
        d1 = df[df["period"] == p1].set_index("portfolio")["quintile"]
        common = d0.index.intersection(d1.index)
        transitions.extend(zip(d0.loc[common].astype(int), d1.loc[common].astype(int)))

    labels = list(range(1, n_quantiles + 1))
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    for q0, q1 in transitions:
        counts.loc[int(q0), int(q1)] += 1.0

    matrix = counts.div(counts.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    matrix.index.name = "from_quintile"
    matrix.columns.name = "to_quintile"

    summary_rows = []
    for q in labels:
        row = matrix.loc[q]
        summary_rows.append(
            {
                "from_quintile": q,
                "stay_probability": float(row.loc[q]),
                "upgrade_probability": float(row[row.index > q].sum()),
                "downgrade_probability": float(row[row.index < q].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if "window_length" in df.columns and df["window_length"].notna().any():
        window_length = int(round(float(df["window_length"].median())))
        overlap_fraction = max(
            0.0, 1.0 - float(horizon_months) / float(window_length)
        )
    else:
        window_length = np.nan
        overlap_fraction = np.nan
    summary["top_quintile_persistence"] = float(matrix.loc[n_quantiles, n_quantiles])
    summary["bottom_quintile_persistence"] = float(matrix.loc[1, 1])
    summary["horizon_months"] = int(horizon_months)
    summary["window_length_months"] = window_length
    summary["window_overlap_fraction"] = overlap_fraction
    summary["structural_inference_eligible"] = bool(
        np.isfinite(overlap_fraction) and overlap_fraction == 0.0
    )
    summary["n_transitions"] = int(len(transitions))
    return matrix, summary


def compute_quintile_transitions(
    df: pd.DataFrame,
    value_col: str = "AE_H1_mean",
    horizon: int = 1,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Backward-compatible wrapper for older scripts."""

    work = df.rename(columns={value_col: "AE"}).copy()
    if "window_end" not in work:
        raise ValueError("Expected a window_end column.")
    work["quintile"] = work.groupby("window_end")["AE"].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop") + 1
    )
    matrix, _ = compute_transition_matrix(
        work,
        horizon_months=horizon,
        date_col="window_end",
        n_quantiles=n_quantiles,
    )
    return matrix
