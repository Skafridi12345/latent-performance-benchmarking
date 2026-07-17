from __future__ import annotations

import time

import numpy as np
import pandas as pd

from analysis.latent_performance import benjamini_hochberg
from sfa.loaders import design_matrix
from sfa.models import make_sfa_model, normalise_model_type


def apply_sfa_multiple_testing(scores: pd.DataFrame) -> pd.DataFrame:
    """Apply the cross-sectional BH decision rule to SFA boundary tests."""

    out = scores.copy()
    if "sfa_boundary_p_value" not in out:
        out["sfa_boundary_p_value"] = np.nan
    p_values = pd.to_numeric(out["sfa_boundary_p_value"], errors="coerce")
    valid = p_values.notna() & p_values.between(0.0, 1.0)
    out["sfa_boundary_q_value"] = benjamini_hochberg(p_values.to_numpy(float))
    nominal = pd.Series(pd.NA, index=out.index, dtype="boolean")
    fdr = pd.Series(pd.NA, index=out.index, dtype="boolean")
    nominal.loc[valid] = p_values.loc[valid] < 0.05
    fdr.loc[valid] = out.loc[valid, "sfa_boundary_q_value"] < 0.05
    out["sfa_supported_nominal_5pct"] = nominal
    out["sfa_supported_fdr_5pct"] = fdr
    rankable = (
        out.get("AE", pd.Series(np.nan, index=out.index)).notna()
        & out.get("converged", pd.Series(False, index=out.index)).fillna(False)
        & out["sfa_supported_fdr_5pct"].fillna(False)
    )
    out["sfa_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[rankable, "sfa_rank"] = (
        out.loc[rankable, "AE"].rank(ascending=False, method="first").astype("Int64")
    )
    return out


def estimate_static_sfa(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    model_type: str = "half_normal",
    min_obs: int = 60,
    maxiter: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate static SFA models portfolio by portfolio."""

    model_key = normalise_model_type(model_type)
    feature_names = ["alpha", *factor_cols]
    rows: list[dict] = []
    ts_rows: list[pd.DataFrame] = []

    for portfolio, group in df.groupby("portfolio"):
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) < min_obs:
            continue

        y = g["excess_return"].to_numpy(float)
        X = design_matrix(g, factor_cols)
        started = time.perf_counter()
        try:
            fit = make_sfa_model(
                model_key,
                y,
                X,
                feature_names=feature_names,
            ).fit(maxiter=maxiter)
            runtime = time.perf_counter() - started
            converged = bool(fit.converged)
            message = fit.message
        except Exception as exc:
            runtime = time.perf_counter() - started
            fit = None
            converged = False
            message = repr(exc)

        if fit is None:
            row = {
                "portfolio": portfolio,
                "model_type": model_key,
                "factor_model": factor_model,
                "sample_start": g["date"].min(),
                "sample_end": g["date"].max(),
                "n_obs": int(len(g)),
                "converged": False,
                "message": message,
                "runtime_seconds": runtime,
            }
        else:
            row = {
                "portfolio": portfolio,
                "model_type": model_key,
                "factor_model": factor_model,
                "sample_start": g["date"].min(),
                "sample_end": g["date"].max(),
                "n_obs": int(len(g)),
                "alpha": float(fit.alpha),
                "sigma_v": float(fit.sigma_v),
                "sigma_u": float(fit.sigma_u),
                "lambda": float(fit.lambda_),
                "log_likelihood": float(fit.log_likelihood),
                "AIC": float(fit.aic),
                "BIC": float(fit.bic),
                "u_hat": float(np.mean(fit.u_hat)),
                "u_hat_standardized": float(np.mean(fit.u_hat_standardized)),
                "AE": float(np.mean(fit.AE)),
                "AE_median": float(np.median(fit.AE)),
                "AE_raw_plugin_legacy": float(np.mean(fit.AE_raw_plugin)),
                "normal_log_likelihood": float(
                    getattr(fit, "normal_log_likelihood", np.nan)
                ),
                "boundary_lr_stat": float(getattr(fit, "boundary_lr_stat", np.nan)),
                "sfa_boundary_p_value": float(
                    getattr(fit, "boundary_mixture_p_value", np.nan)
                ),
                "sfa_supported_nominal_5pct": getattr(
                    fit, "one_sided_component_supported", pd.NA
                ),
                "residual_mean": float(np.mean(fit.residuals)),
                "residual_std": float(np.std(fit.residuals, ddof=1)),
                "converged": converged,
                "message": message,
                "runtime_seconds": runtime,
            }
            if hasattr(fit, "mu"):
                row["mu"] = float(fit.mu)
            for name, coef in zip(feature_names, fit.beta):
                row[f"coef_{name}"] = float(coef)

            ts = g[["date", "portfolio", "excess_return"]].copy()
            ts["model_type"] = model_key
            ts["factor_model"] = factor_model
            ts["frontier"] = fit.frontier
            ts["fitted_value"] = fit.fitted_values
            ts["residual"] = fit.residuals
            ts["composed_error"] = fit.composed_error
            ts["u_hat"] = fit.u_hat
            ts["u_hat_standardized"] = fit.u_hat_standardized
            ts["AE"] = fit.AE
            ts["AE_raw_plugin_legacy"] = fit.AE_raw_plugin
            ts_rows.append(ts)

        rows.append(row)

    scores = pd.DataFrame(rows)
    if not scores.empty and "AE" in scores:
        if model_key == "half_normal":
            scores = apply_sfa_multiple_testing(scores)
        else:
            valid = scores["AE"].notna() & scores["converged"].fillna(False)
            scores["sfa_rank"] = pd.Series(pd.NA, index=scores.index, dtype="Int64")
            scores.loc[valid, "sfa_rank"] = (
                scores.loc[valid, "AE"]
                .rank(ascending=False, method="first")
                .astype("Int64")
            )
        # Compatibility alias for downstream comparison code. It has exactly
        # the same FDR eligibility as the explicit SFA rank.
        scores["AE_rank"] = scores["sfa_rank"]
        scores = scores.sort_values("sfa_rank", na_position="last").reset_index(
            drop=True
        )

    timeseries = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame()
    return scores, timeseries
