from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_persistence_metrics(
    rolling: pd.DataFrame,
    *,
    horizons_months: list[int] | tuple[int, ...] = (1, 3, 6, 12),
    date_col: str = "window_end",
    score_col: str = "AE",
) -> pd.DataFrame:
    """
    Compute rank and score persistence across month-based horizons.

    The function matches windows by calendar month. If the rolling output uses
    an annual step, one-month or three-month horizons will be reported with
    zero matched pairs rather than fabricated by interpolation.
    """

    required = {date_col, "portfolio", score_col, "rank", "quintile"}
    missing = required - set(rolling.columns)
    if missing:
        raise ValueError(f"rolling data missing required columns: {sorted(missing)}")

    df = rolling.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["period"] = df[date_col].dt.to_period("M")
    df = df.dropna(subset=[score_col, "rank", "quintile"])
    df["rank"] = df["rank"].astype(float)
    df["quintile"] = df["quintile"].astype(int)

    rows: list[dict] = []
    if "window_length" in df.columns and df["window_length"].notna().any():
        window_length = int(round(float(df["window_length"].median())))
    else:
        window_length = np.nan
    for horizon in horizons_months:
        overlap_fraction = (
            max(0.0, 1.0 - float(horizon) / float(window_length))
            if np.isfinite(window_length) and window_length > 0
            else np.nan
        )
        inference_eligible = bool(
            np.isfinite(overlap_fraction) and overlap_fraction == 0.0
        )
        pair_rows = []
        periods = sorted(df["period"].unique())
        available = set(periods)
        for p0 in periods:
            p1 = p0 + int(horizon)
            if p1 not in available:
                continue

            d0 = df[df["period"] == p0].set_index("portfolio")
            d1 = df[df["period"] == p1].set_index("portfolio")
            common = d0.index.intersection(d1.index)
            if len(common) < 3:
                continue

            left = d0.loc[common]
            right = d1.loc[common]
            top_left = left["quintile"] == 5
            bottom_left = left["quintile"] == 1
            pair_rows.append(
                {
                    "date_t": p0.to_timestamp(how="end"),
                    "date_t_plus_h": p1.to_timestamp(how="end"),
                    "horizon_months": int(horizon),
                    "window_length_months": window_length,
                    "window_overlap_fraction": overlap_fraction,
                    "structural_inference_eligible": inference_eligible,
                    "n_portfolios": int(len(common)),
                    "spearman_rank_autocorrelation": spearmanr(
                        left["rank"], right["rank"]
                    ).statistic,
                    "pearson_score_autocorrelation": pearsonr(
                        left[score_col], right[score_col]
                    ).statistic,
                    "average_absolute_rank_change": float(
                        np.mean(np.abs(left["rank"] - right["rank"]))
                    ),
                    "top_quintile_stay_probability": float(
                        np.mean(right.loc[top_left, "quintile"] == 5)
                    )
                    if top_left.any()
                    else np.nan,
                    "bottom_quintile_stay_probability": float(
                        np.mean(right.loc[bottom_left, "quintile"] == 1)
                    )
                    if bottom_left.any()
                    else np.nan,
                }
            )

        if pair_rows:
            rows.extend(pair_rows)
        else:
            rows.append(
                {
                    "date_t": pd.NaT,
                    "date_t_plus_h": pd.NaT,
                    "horizon_months": int(horizon),
                    "window_length_months": window_length,
                    "window_overlap_fraction": overlap_fraction,
                    "structural_inference_eligible": inference_eligible,
                    "n_portfolios": 0,
                    "spearman_rank_autocorrelation": np.nan,
                    "pearson_score_autocorrelation": np.nan,
                    "average_absolute_rank_change": np.nan,
                    "top_quintile_stay_probability": np.nan,
                    "bottom_quintile_stay_probability": np.nan,
                }
            )

    return pd.DataFrame(rows)


def rank_persistence(
    df: pd.DataFrame,
    horizon: int = 1,
    date_col: str = "window_end",
    value_col: str = "AE_H1_mean",
) -> pd.DataFrame:
    """Backward-compatible wrapper for older scripts."""

    work = df.rename(columns={value_col: "AE"}).copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["rank"] = work.groupby(date_col)["AE"].rank(ascending=False, method="first")
    work["quintile"] = work.groupby(date_col)["AE"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    out = compute_persistence_metrics(
        work,
        horizons_months=[horizon],
        date_col=date_col,
        score_col="AE",
    )
    return out.rename(columns={"horizon_months": "horizon"})
