from __future__ import annotations

import numpy as np
import pandas as pd

MOBILITY_COLUMNS = [
    "portfolio",
    "mean_rank",
    "median_rank",
    "rank_volatility",
    "mean_score",
    "score_volatility",
    "maximum_rank_improvement",
    "maximum_rank_deterioration",
    "same_quintile_probability",
    "move_up_probability",
    "move_down_probability",
    "time_in_quintile_1",
    "time_in_quintile_2",
    "time_in_quintile_3",
    "time_in_quintile_4",
    "time_in_quintile_5",
    "mobility_horizon_months",
    "window_overlap_fraction",
    "structural_inference_eligible",
]


def portfolio_mobility_summary(
    rolling: pd.DataFrame,
    *,
    horizon_months: int | None = None,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Summarise mobility at an explicit calendar horizon."""

    score_col = score_col or (
        "posterior_alpha" if "posterior_alpha" in rolling.columns else "AE"
    )
    required = {"portfolio", "window_end", score_col, "rank", "quintile"}
    missing = required - set(rolling.columns)
    if missing:
        raise ValueError(f"rolling data missing required columns: {sorted(missing)}")

    df = rolling.dropna(subset=[score_col, "rank", "quintile"]).copy()
    if df.empty:
        return pd.DataFrame(columns=MOBILITY_COLUMNS)

    df["window_end"] = pd.to_datetime(df["window_end"])
    df = df.sort_values(["portfolio", "window_end"])
    df["rank"] = df["rank"].astype(float)
    df["quintile"] = df["quintile"].astype(int)
    df["period"] = df["window_end"].dt.to_period("M")
    if horizon_months is None:
        ordered_periods = sorted(df["period"].unique())
        gaps = [
            int(right.ordinal - left.ordinal)
            for left, right in zip(ordered_periods[:-1], ordered_periods[1:])
            if right.ordinal > left.ordinal
        ]
        horizon_months = int(round(float(np.median(gaps)))) if gaps else 1
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive.")

    left = df.copy()
    left["target_period"] = left["period"] + int(horizon_months)
    right = df[["portfolio", "period", "quintile", "rank"]].rename(
        columns={"quintile": "next_quintile", "rank": "next_rank"}
    )
    pairs = left.merge(
        right,
        left_on=["portfolio", "target_period"],
        right_on=["portfolio", "period"],
        how="left",
        suffixes=("", "_future"),
    )
    pairs["rank_change"] = pairs["rank"] - pairs["next_rank"]

    if "window_length" in df.columns and df["window_length"].notna().any():
        window_length = float(df["window_length"].median())
        overlap_fraction = max(
            0.0, 1.0 - float(horizon_months) / window_length
        )
    else:
        overlap_fraction = np.nan

    rows: list[dict] = []
    for portfolio, group in df.groupby("portfolio"):
        portfolio_pairs = pairs[pairs["portfolio"] == portfolio]
        transitions = portfolio_pairs.dropna(subset=["next_quintile"])
        row = {
            "portfolio": portfolio,
            "mean_rank": float(group["rank"].mean()),
            "median_rank": float(group["rank"].median()),
            "rank_volatility": float(group["rank"].std(ddof=1)),
            "mean_score": float(group[score_col].mean()),
            "score_volatility": float(group[score_col].std(ddof=1)),
            "maximum_rank_improvement": float(
                np.nanmax(portfolio_pairs["rank_change"])
            )
            if portfolio_pairs["rank_change"].notna().any()
            else np.nan,
            "maximum_rank_deterioration": float(
                -np.nanmin(portfolio_pairs["rank_change"])
            )
            if portfolio_pairs["rank_change"].notna().any()
            else np.nan,
            "mobility_horizon_months": int(horizon_months),
            "window_overlap_fraction": overlap_fraction,
            "structural_inference_eligible": bool(
                np.isfinite(overlap_fraction) and overlap_fraction == 0.0
            ),
        }

        if transitions.empty:
            row.update(
                {
                    "same_quintile_probability": np.nan,
                    "move_up_probability": np.nan,
                    "move_down_probability": np.nan,
                }
            )
        else:
            row.update(
                {
                    "same_quintile_probability": float(
                        np.mean(transitions["next_quintile"] == transitions["quintile"])
                    ),
                    "move_up_probability": float(
                        np.mean(transitions["next_quintile"] > transitions["quintile"])
                    ),
                    "move_down_probability": float(
                        np.mean(transitions["next_quintile"] < transitions["quintile"])
                    ),
                }
            )

        shares = group["quintile"].value_counts(normalize=True)
        for q in range(1, 6):
            row[f"time_in_quintile_{q}"] = float(shares.get(q, 0.0))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=MOBILITY_COLUMNS)
    return (
        pd.DataFrame(rows)[MOBILITY_COLUMNS]
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def compute_mobility_metrics(transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible transition-matrix mobility summary."""

    mat = transition_matrix.copy()
    mat.index = mat.index.astype(float).astype(int)
    mat.columns = mat.columns.astype(float).astype(int)

    rows = []
    for q in mat.index:
        row = mat.loc[q]
        rows.append(
            {
                "Quintile": q,
                "Stay": float(row.loc[q]),
                "Improve": float(row[row.index > q].sum()),
                "Deteriorate": float(row[row.index < q].sum()),
            }
        )
    return pd.DataFrame(rows)
