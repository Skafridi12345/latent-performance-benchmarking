from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _ljung_box(residuals: np.ndarray, lags: int) -> tuple[float, float]:
    n = len(residuals)
    use_lags = int(max(0, min(lags, n - 2)))
    if use_lags == 0:
        return np.nan, np.nan
    centered = residuals - np.mean(residuals)
    denominator = float(centered @ centered)
    if denominator <= 0:
        return np.nan, np.nan
    correlations = [
        float(centered[lag:] @ centered[:-lag] / denominator)
        for lag in range(1, use_lags + 1)
    ]
    q_stat = n * (n + 2.0) * sum(
        rho**2 / (n - lag)
        for lag, rho in enumerate(correlations, start=1)
    )
    return float(q_stat), float(stats.chi2.sf(q_stat, use_lags))


def _arch_lm(residuals: np.ndarray, lags: int) -> tuple[float, float]:
    squared = np.asarray(residuals, dtype=float) ** 2
    use_lags = int(max(0, min(lags, len(squared) // 4)))
    if use_lags == 0 or len(squared) <= use_lags + 2:
        return np.nan, np.nan
    y = squared[use_lags:]
    X = np.column_stack(
        [
            np.ones(len(y)),
            *[
                squared[use_lags - lag : len(squared) - lag]
                for lag in range(1, use_lags + 1)
            ],
        ]
    )
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    tss = float((y - y.mean()) @ (y - y.mean()))
    r_squared = 1.0 - float(residual @ residual) / tss if tss > 0 else 0.0
    statistic = len(y) * max(r_squared, 0.0)
    return float(statistic), float(stats.chi2.sf(statistic, use_lags))


def residual_diagnostics(residuals: pd.DataFrame) -> pd.DataFrame:
    """Compute residual moments and normality diagnostics by fitted model."""

    if residuals.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    group_columns = ["portfolio"]
    if "model_type" in residuals.columns:
        group_columns.append("model_type")
    if "factor_model" in residuals.columns:
        group_columns.append("factor_model")
    for keys, group in residuals.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys))
        if "date" in group.columns:
            group = group.sort_values("date")
        r = group["residual"].dropna().to_numpy(float)
        if len(r) < 8:
            jb_stat, jb_p = np.nan, np.nan
        else:
            jb = stats.jarque_bera(r)
            jb_stat, jb_p = float(jb.statistic), float(jb.pvalue)
        ljung_box_stat, ljung_box_p = _ljung_box(r, lags=12)
        arch_lm_stat, arch_lm_p = _arch_lm(r, lags=12)
        durbin_watson = (
            float(np.sum(np.diff(r) ** 2) / np.sum(r**2))
            if len(r) > 1 and np.sum(r**2) > 0
            else np.nan
        )
        rows.append(
            {
                **key_values,
                "residual_mean": float(np.mean(r)) if len(r) else np.nan,
                "residual_std": float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
                "residual_skewness": float(stats.skew(r, bias=False))
                if len(r) > 2
                else np.nan,
                "residual_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False))
                if len(r) > 3
                else np.nan,
                "jarque_bera_stat": jb_stat,
                "jarque_bera_p_value": jb_p,
                "durbin_watson": durbin_watson,
                "ljung_box_lag_12_stat": ljung_box_stat,
                "ljung_box_lag_12_p_value": ljung_box_p,
                "arch_lm_lag_12_stat": arch_lm_stat,
                "arch_lm_lag_12_p_value": arch_lm_p,
            }
        )
    return pd.DataFrame(rows)


def model_diagnostics(
    static_scores: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    runtime_seconds: float | None = None,
) -> pd.DataFrame:
    """Combine model fit metadata with residual diagnostics."""

    diag = static_scores.copy()
    if not residuals.empty:
        resid_diag = residual_diagnostics(residuals)
        diag = diag.merge(
            resid_diag,
            on=["portfolio", "model_type", "factor_model"],
            how="left",
            suffixes=("", "_diagnostic"),
        )

    diag["number_of_portfolios"] = int(static_scores["portfolio"].nunique())
    diag["number_of_observations"] = int(static_scores["n_obs"].sum())
    if runtime_seconds is not None:
        diag["pipeline_runtime_seconds"] = float(runtime_seconds)

    keep = [
        "portfolio",
        "model_type",
        "factor_model",
        "sample_start",
        "sample_end",
        "number_of_portfolios",
        "number_of_observations",
        "n_obs",
        "log_likelihood",
        "AIC",
        "BIC",
        "sigma_v",
        "sigma_u",
        "lambda",
        "normal_log_likelihood",
        "boundary_lr_stat",
        "boundary_mixture_p_value",
        "one_sided_component_supported",
        "converged",
        "runtime_seconds",
        "pipeline_runtime_seconds",
        "residual_mean",
        "residual_std",
        "residual_skewness",
        "residual_kurtosis",
        "jarque_bera_p_value",
        "durbin_watson",
        "ljung_box_lag_12_p_value",
        "arch_lm_lag_12_p_value",
    ]
    existing = [col for col in keep if col in diag.columns]
    return diag[existing].sort_values("portfolio").reset_index(drop=True)
