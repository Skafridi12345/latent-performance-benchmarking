from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.latent_performance import rolling_cross_sectional_performance
from analysis.rolling_windows import rolling_sfa


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def _fisher_mean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan
    clipped = np.clip(finite, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def rolling_window_sensitivity(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    model_type: str = "half_normal",
    windows: tuple[int, ...] = (60, 120, 180),
    step: int = 12,
    min_obs: int | None = None,
    maxiter: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare rolling AE/rank stability across rolling-window lengths."""

    rolling_by_window: dict[int, pd.DataFrame] = {}
    first_end = max(windows)
    for window in windows:
        min_required = min_obs or window
        out = rolling_sfa(
            df,
            factor_cols,
            window=window,
            min_obs=min_required,
            step=step,
            model_type=model_type,
            factor_model=factor_model,
            maxiter=maxiter,
            first_end=first_end,
        )
        out["sensitivity_window"] = int(window)
        rolling_by_window[int(window)] = out

    rows: list[dict] = []
    for left_window, right_window in itertools.combinations(windows, 2):
        left = rolling_by_window[int(left_window)].dropna(subset=["AE", "rank"])
        right = rolling_by_window[int(right_window)].dropna(subset=["AE", "rank"])
        merged = left.merge(
            right,
            on=["portfolio", "window_end"],
            suffixes=(f"_{left_window}", f"_{right_window}"),
        )
        if merged.empty:
            rows.append(
                {
                    "window_left": left_window,
                    "window_right": right_window,
                    "n_common_observations": 0,
                    "rank_correlation": np.nan,
                    "AE_correlation": np.nan,
                    "top_quintile_jaccard": np.nan,
                    "bottom_quintile_jaccard": np.nan,
                }
            )
            continue

        rank_correlations = []
        ae_correlations = []
        top_scores = []
        bottom_scores = []
        for _, group in merged.groupby("window_end"):
            if len(group) >= 3:
                rank_correlations.append(
                    spearmanr(
                        group[f"rank_{left_window}"],
                        group[f"rank_{right_window}"],
                    ).statistic
                )
                ae_correlations.append(
                    pearsonr(
                        group[f"AE_{left_window}"],
                        group[f"AE_{right_window}"],
                    ).statistic
                )
            top_left = set(
                group.loc[
                    group[f"quintile_{left_window}"].astype(int) == 5, "portfolio"
                ]
            )
            top_right = set(
                group.loc[
                    group[f"quintile_{right_window}"].astype(int) == 5, "portfolio"
                ]
            )
            bottom_left = set(
                group.loc[
                    group[f"quintile_{left_window}"].astype(int) == 1, "portfolio"
                ]
            )
            bottom_right = set(
                group.loc[
                    group[f"quintile_{right_window}"].astype(int) == 1, "portfolio"
                ]
            )
            top_scores.append(_jaccard(top_left, top_right))
            bottom_scores.append(_jaccard(bottom_left, bottom_right))

        rows.append(
            {
                "window_left": int(left_window),
                "window_right": int(right_window),
                "n_common_observations": int(len(merged)),
                "n_common_window_ends": int(merged["window_end"].nunique()),
                "rank_correlation": _fisher_mean(rank_correlations),
                "AE_correlation": _fisher_mean(ae_correlations),
                "rank_correlation_window_std": float(
                    np.nanstd(rank_correlations, ddof=1)
                )
                if len(rank_correlations) > 1
                else np.nan,
                "AE_correlation_window_std": float(
                    np.nanstd(ae_correlations, ddof=1)
                )
                if len(ae_correlations) > 1
                else np.nan,
                "top_quintile_jaccard": float(np.nanmean(top_scores)),
                "bottom_quintile_jaccard": float(np.nanmean(bottom_scores)),
            }
        )

    combined = pd.concat(rolling_by_window.values(), ignore_index=True)
    return pd.DataFrame(rows), combined


def performance_window_sensitivity(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    windows: tuple[int, ...] = (60, 120, 180),
    step: int = 12,
    hac_lags: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare primary performance scores across window lengths by date."""

    rolling_by_window: dict[int, pd.DataFrame] = {}
    first_end = max(windows)
    for window in windows:
        scores = rolling_cross_sectional_performance(
            df,
            factor_cols,
            factor_model=factor_model,
            window=int(window),
            step=int(step),
            min_obs=int(window),
            hac_lags=hac_lags,
            first_end=first_end,
        )
        scores["sensitivity_window"] = int(window)
        rolling_by_window[int(window)] = scores

    rows = []
    for left_window, right_window in itertools.combinations(windows, 2):
        left = rolling_by_window[int(left_window)]
        right = rolling_by_window[int(right_window)]
        merged = left.merge(
            right,
            on=["portfolio", "window_end"],
            suffixes=(f"_{left_window}", f"_{right_window}"),
        )
        rank_correlations: list[float] = []
        score_correlations: list[float] = []
        top_scores: list[float] = []
        bottom_scores: list[float] = []
        for _, group in merged.groupby("window_end"):
            if len(group) >= 3:
                rank_correlations.append(
                    spearmanr(
                        group[f"performance_rank_{left_window}"],
                        group[f"performance_rank_{right_window}"],
                    ).statistic
                )
                score_correlations.append(
                    pearsonr(
                        group[f"posterior_alpha_{left_window}"],
                        group[f"posterior_alpha_{right_window}"],
                    ).statistic
                )
            top_scores.append(
                _jaccard(
                    set(
                        group.loc[
                            group[f"quintile_{left_window}"] == 5, "portfolio"
                        ]
                    ),
                    set(
                        group.loc[
                            group[f"quintile_{right_window}"] == 5, "portfolio"
                        ]
                    ),
                )
            )
            bottom_scores.append(
                _jaccard(
                    set(
                        group.loc[
                            group[f"quintile_{left_window}"] == 1, "portfolio"
                        ]
                    ),
                    set(
                        group.loc[
                            group[f"quintile_{right_window}"] == 1, "portfolio"
                        ]
                    ),
                )
            )
        rows.append(
            {
                "window_left": int(left_window),
                "window_right": int(right_window),
                "n_common_observations": int(len(merged)),
                "n_common_window_ends": int(merged["window_end"].nunique())
                if not merged.empty
                else 0,
                "rank_correlation": _fisher_mean(rank_correlations),
                "score_correlation": _fisher_mean(score_correlations),
                "rank_correlation_window_std": float(
                    np.nanstd(rank_correlations, ddof=1)
                )
                if len(rank_correlations) > 1
                else np.nan,
                "score_correlation_window_std": float(
                    np.nanstd(score_correlations, ddof=1)
                )
                if len(score_correlations) > 1
                else np.nan,
                "top_quintile_jaccard": float(np.nanmean(top_scores))
                if top_scores
                else np.nan,
                "bottom_quintile_jaccard": float(np.nanmean(bottom_scores))
                if bottom_scores
                else np.nan,
            }
        )
    combined = pd.concat(rolling_by_window.values(), ignore_index=True)
    return pd.DataFrame(rows), combined


def model_comparison(
    half_normal_scores: pd.DataFrame,
    truncated_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compare static half-normal and truncated-normal SFA outputs."""

    left = half_normal_scores[
        [
            "portfolio",
            "AE",
            "AE_rank",
            "log_likelihood",
            "AIC",
            "BIC",
            "converged",
            "boundary_mixture_p_value",
            "one_sided_component_supported",
        ]
    ].rename(
        columns={
            "AE": "AE_half_normal",
            "AE_rank": "AE_rank_half_normal",
            "log_likelihood": "log_likelihood_half_normal",
            "AIC": "AIC_half_normal",
            "BIC": "BIC_half_normal",
            "converged": "converged_half_normal",
            "boundary_mixture_p_value": "boundary_mixture_p_value_half_normal",
            "one_sided_component_supported": (
                "one_sided_component_supported_half_normal"
            ),
        }
    )
    right = truncated_scores[
        ["portfolio", "AE", "AE_rank", "log_likelihood", "AIC", "BIC", "converged"]
    ].rename(
        columns={
            "AE": "AE_truncated_normal",
            "AE_rank": "AE_rank_truncated_normal",
            "log_likelihood": "log_likelihood_truncated_normal",
            "AIC": "AIC_truncated_normal",
            "BIC": "BIC_truncated_normal",
            "converged": "converged_truncated_normal",
        }
    )
    merged = left.merge(right, on="portfolio", how="inner")
    merged["AE_difference"] = merged["AE_half_normal"] - merged["AE_truncated_normal"]
    merged["rank_difference"] = (
        merged["AE_rank_half_normal"] - merged["AE_rank_truncated_normal"]
    )
    valid = merged["one_sided_component_supported_half_normal"].fillna(False)
    merged["valid_for_cross_distribution_rank_comparison"] = valid
    if int(valid.sum()) >= 3:
        merged["rank_spearman"] = spearmanr(
            merged.loc[valid, "AE_rank_half_normal"],
            merged.loc[valid, "AE_rank_truncated_normal"],
        ).statistic
    else:
        merged["rank_spearman"] = np.nan
    return merged
