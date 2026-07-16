from __future__ import annotations

import numpy as np
import pandas as pd

from sfa.loaders import design_matrix
from sfa.models import make_sfa_model, normalise_model_type


def _assign_window_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = out["AE"].notna() & out["convergence_status"].astype(bool)
    if "one_sided_component_supported" in out:
        support = out["one_sided_component_supported"].fillna(False).astype(bool)
        valid &= support
    out["rank"] = pd.NA
    out.loc[valid, "rank"] = (
        out.loc[valid]
        .groupby("window_end")["AE"]
        .rank(ascending=False, method="first")
        .astype("Int64")
    )

    def quintiles(x: pd.Series) -> pd.Series:
        if x.notna().sum() < 5:
            return pd.Series(pd.NA, index=x.index, dtype="Int64")
        return pd.qcut(x, 5, labels=False, duplicates="drop").add(1).astype("Int64")

    out["quintile"] = pd.NA
    out.loc[valid, "quintile"] = (
        out.loc[valid].groupby("window_end")["AE"].transform(quintiles).astype("Int64")
    )
    return out


def rolling_sfa(
    df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    *,
    window: int = 120,
    min_obs: int = 120,
    step: int = 12,
    model_type: str = "half_normal",
    factor_model: str = "ff3",
    maxiter: int = 250,
    warm_start: bool = True,
    first_end: int | None = None,
) -> pd.DataFrame:
    """
    Rolling-window stochastic frontier efficiency estimation.

    Returns one row per portfolio-window with mean AE and mean u_hat over the
    window, plus convergence and information-criterion diagnostics.
    """

    factor_cols = factor_cols or ["mkt_rf", "smb", "hml"]
    model_key = normalise_model_type(model_type)
    feature_names = ["alpha", *factor_cols]
    results: list[dict] = []

    for portfolio, group in df.groupby("portfolio"):
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) < min_obs:
            continue

        theta0 = None
        start_end = max(window, first_end or window)
        for end in range(start_end, len(g) + 1, step):
            w = g.iloc[end - window : end].copy()
            if len(w) < min_obs:
                continue

            y = w["excess_return"].to_numpy(float)
            X = design_matrix(w, factor_cols)
            fit = None
            message = ""
            try:
                fit = make_sfa_model(
                    model_key,
                    y,
                    X,
                    feature_names=feature_names,
                ).fit(theta0=theta0 if warm_start else None, maxiter=maxiter)
                if warm_start and fit.converged:
                    theta0 = fit.theta
                message = fit.message
            except Exception as exc:
                message = repr(exc)

            base = {
                "portfolio": portfolio,
                "window_start": w["date"].iloc[0],
                "window_end": w["date"].iloc[-1],
                "window_length": int(len(w)),
                "model_type": model_key,
                "factor_model": factor_model,
            }

            if fit is None:
                base.update(
                    {
                        "AE": np.nan,
                        "u_hat": np.nan,
                        "convergence_status": False,
                        "message": message,
                        "log_likelihood": np.nan,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "sigma_v": np.nan,
                        "sigma_u": np.nan,
                        "lambda": np.nan,
                        "u_hat_standardized": np.nan,
                        "boundary_lr_stat": np.nan,
                        "boundary_mixture_p_value": np.nan,
                        "one_sided_component_supported": False,
                    }
                )
            else:
                base.update(
                    {
                        "AE": float(np.mean(fit.AE)),
                        "u_hat": float(np.mean(fit.u_hat)),
                        "u_hat_standardized": float(
                            np.mean(fit.u_hat_standardized)
                        ),
                        "convergence_status": bool(fit.converged),
                        "message": message,
                        "log_likelihood": float(fit.log_likelihood),
                        "AIC": float(fit.aic),
                        "BIC": float(fit.bic),
                        "sigma_v": float(fit.sigma_v),
                        "sigma_u": float(fit.sigma_u),
                        "lambda": float(fit.lambda_),
                        "boundary_lr_stat": float(
                            getattr(fit, "boundary_lr_stat", np.nan)
                        ),
                        "boundary_mixture_p_value": float(
                            getattr(fit, "boundary_mixture_p_value", np.nan)
                        ),
                        "one_sided_component_supported": getattr(
                            fit, "one_sided_component_supported", pd.NA
                        ),
                    }
                )
            results.append(base)

    out = pd.DataFrame(results)
    if out.empty:
        return out
    return _assign_window_ranks(out)
