from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.latent_performance import (
    benjamini_hochberg,
    estimate_hac_factor_model,
)
from sfa.loaders import design_matrix


def _ols_summary(y: np.ndarray, X: np.ndarray) -> dict:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    n, k = X.shape
    dof = max(n - k, 1)
    sse = float(resid @ resid)
    sigma2 = sse / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    tss = float(((y - y.mean()) @ (y - y.mean())))
    r_squared = 1.0 - sse / tss if tss > 0 else np.nan
    vol = float(np.std(y, ddof=1))
    sharpe_like = float(y.mean() / vol * np.sqrt(12.0)) if vol > 0 else np.nan
    return {
        "beta": beta,
        "t_stats": t_stats,
        "residuals": resid,
        "fitted": fitted,
        "residual_volatility": float(np.std(resid, ddof=1)),
        "r_squared": float(r_squared),
        "mean_excess_return": float(np.mean(y)),
        "volatility": vol,
        "sharpe_like_annualized": sharpe_like,
    }


def estimate_alpha_baseline(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    min_obs: int = 60,
    hac_lags: int = 12,
) -> pd.DataFrame:
    """Estimate factor-alpha diagnostics with Newey-West HAC inference."""

    rows: list[dict] = []
    feature_names = ["alpha", *factor_cols]

    for portfolio, group in df.groupby("portfolio"):
        g = group.sort_values("date")
        if len(g) < min_obs:
            continue
        y = g["excess_return"].to_numpy(float)
        X = design_matrix(g, factor_cols)
        out = _ols_summary(y, X)
        hac = estimate_hac_factor_model(y, X, hac_lags=hac_lags)

        row = {
            "portfolio": portfolio,
            "factor_model": factor_model,
            "sample_start": g["date"].min(),
            "sample_end": g["date"].max(),
            "n_obs": int(len(g)),
            "alpha": float(hac["alpha"]),
            "alpha_hac_se": float(hac["alpha_hac_se"]),
            "alpha_t_stat": float(hac["alpha_hac_t_stat"]),
            "alpha_p_value": float(hac["alpha_p_value"]),
            "hac_lags": int(hac_lags),
            "residual_volatility": out["residual_volatility"],
            "r_squared": out["r_squared"],
            "mean_excess_return": out["mean_excess_return"],
            "volatility": out["volatility"],
            "sharpe_like_annualized": out["sharpe_like_annualized"],
        }
        for name, coef, standard_error, t_stat, p_value in zip(
            feature_names[1:],
            hac["beta"][1:],
            hac["hac_standard_errors"][1:],
            hac["t_stats"][1:],
            hac["p_values"][1:],
        ):
            row[f"beta_{name}"] = float(coef)
            row[f"hac_se_{name}"] = float(standard_error)
            row[f"t_{name}"] = float(t_stat)
            row[f"p_value_{name}"] = float(p_value)
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["alpha_fdr_q_value"] = benjamini_hochberg(
        result["alpha_p_value"].to_numpy(float)
    )
    result["alpha_rank"] = (
        result["alpha"].rank(ascending=False, method="first").astype(int)
    )
    return result.sort_values("alpha_rank").reset_index(drop=True)


def compare_alpha_to_ae(
    alpha_df: pd.DataFrame,
    ae_df: pd.DataFrame,
    *,
    ae_col: str = "AE",
) -> pd.DataFrame:
    """Compare conventional alpha ranks with static SFA AE ranks."""

    cols = [
        "portfolio",
        "alpha",
        "alpha_t_stat",
        "alpha_rank",
        "mean_excess_return",
        "volatility",
        "sharpe_like_annualized",
        "r_squared",
    ]
    merged = alpha_df[cols].merge(
        ae_df[["portfolio", ae_col, "AE_rank", "u_hat", "converged"]],
        on="portfolio",
        how="inner",
    )
    merged = merged.rename(columns={ae_col: "AE"})
    merged["rank_difference"] = merged["alpha_rank"] - merged["AE_rank"]
    merged["abs_rank_difference"] = merged["rank_difference"].abs()
    merged["disagreement_case"] = merged["abs_rank_difference"] >= 5
    if len(merged) >= 3:
        rho, p_value = spearmanr(merged["alpha_rank"], merged["AE_rank"])
    else:
        rho, p_value = np.nan, np.nan
    merged["alpha_ae_spearman"] = float(rho)
    merged["alpha_ae_spearman_p_value"] = float(p_value)
    return merged.sort_values("abs_rank_difference", ascending=False).reset_index(
        drop=True
    )
