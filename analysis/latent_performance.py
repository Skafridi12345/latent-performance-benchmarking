from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm, spearmanr

from sfa.loaders import design_matrix


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg false-discovery q-values."""

    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out

    pv = np.clip(p[valid], 0.0, 1.0)
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def newey_west_covariance(
    X: np.ndarray,
    residuals: np.ndarray,
    *,
    max_lags: int,
) -> np.ndarray:
    """Compute a Bartlett-kernel Newey-West covariance matrix.

    The implementation uses the unnormalised sandwich form
    ``(X'X)^-1 S (X'X)^-1`` and applies an ``n / (n-k)`` finite-sample
    correction. It is suitable for the monthly factor regressions used here.
    """

    X = np.asarray(X, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if X.ndim != 2 or residuals.ndim != 1 or len(X) != len(residuals):
        raise ValueError("X and residuals have incompatible shapes.")
    if not np.isfinite(X).all() or not np.isfinite(residuals).all():
        raise ValueError("X and residuals must contain finite values.")

    n, k = X.shape
    lags = int(max(0, min(max_lags, n - 1)))
    score = X * residuals[:, None]
    meat = score.T @ score
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = score[lag:].T @ score[:-lag]
        meat += weight * (gamma + gamma.T)

    bread = np.linalg.pinv(X.T @ X)
    correction = n / max(n - k, 1)
    covariance = correction * bread @ meat @ bread
    return (covariance + covariance.T) / 2.0


def estimate_hac_factor_model(
    y: np.ndarray,
    X: np.ndarray,
    *,
    hac_lags: int = 12,
) -> dict:
    """Estimate a linear factor model with HAC inference for alpha."""

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1 or X.ndim != 2 or len(y) != X.shape[0]:
        raise ValueError("y and X have incompatible shapes.")
    if len(y) <= X.shape[1]:
        raise ValueError("Factor model requires more observations than parameters.")
    if not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ValueError("y and X must contain finite values.")

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    residuals = y - fitted
    covariance = newey_west_covariance(X, residuals, max_lags=hac_lags)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_stats = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * norm.sf(np.abs(t_stats))

    sse = float(residuals @ residuals)
    tss = float((y - y.mean()) @ (y - y.mean()))
    r_squared = 1.0 - sse / tss if tss > 0 else np.nan
    residual_volatility = float(np.std(residuals, ddof=1))
    return {
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "hac_covariance": covariance,
        "hac_standard_errors": standard_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "alpha": float(beta[0]),
        "alpha_hac_se": float(standard_errors[0]),
        "alpha_hac_t_stat": float(t_stats[0]),
        "alpha_p_value": float(p_values[0]),
        "hac_lags": int(hac_lags),
        "r_squared": float(r_squared),
        "residual_volatility": residual_volatility,
    }


def _profile_prior(alpha: np.ndarray, se: np.ndarray) -> dict:
    """Estimate a normal cross-sectional prior by marginal maximum likelihood."""

    alpha = np.asarray(alpha, dtype=float)
    se = np.asarray(se, dtype=float)
    if len(alpha) < 2 or len(alpha) != len(se):
        raise ValueError("At least two alpha estimates with matching SEs are required.")
    if not np.isfinite(alpha).all() or not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError("Alpha estimates and standard errors must be finite; SEs > 0.")

    se2 = se**2

    def profiled(tau2: float) -> tuple[float, float]:
        variance = se2 + max(float(tau2), 0.0)
        weights = 1.0 / variance
        mu = float(np.sum(weights * alpha) / np.sum(weights))
        objective = 0.5 * float(
            np.sum(np.log(variance) + (alpha - mu) ** 2 / variance)
        )
        return objective, mu

    empirical_variance = float(np.var(alpha, ddof=1))
    upper = max(empirical_variance * 25.0, float(np.max(se2)) * 10.0, 1e-10)
    opt = minimize_scalar(
        lambda tau2: profiled(tau2)[0],
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": max(upper * 1e-10, 1e-16)},
    )
    tau2 = max(float(opt.x), 0.0)
    objective, mu = profiled(tau2)
    prior_variance = se2 + tau2
    mu_se = float(1.0 / np.sqrt(np.sum(1.0 / prior_variance)))
    return {
        "mu": mu,
        "mu_se": mu_se,
        "tau2": tau2,
        "tau": float(np.sqrt(tau2)),
        "marginal_objective": objective,
        "converged": bool(opt.success),
    }


def hierarchical_shrinkage(
    estimates: pd.DataFrame,
    *,
    alpha_col: str = "alpha",
    se_col: str = "alpha_hac_se",
    p_value_col: str = "alpha_p_value",
) -> tuple[pd.DataFrame, dict]:
    """Shrink noisy portfolio alphas toward an estimated common prior.

    The posterior alpha is the primary comparable latent-performance estimand.
    Negative posterior alpha is reported separately as annualized shortfall in
    basis points; no unidentified non-negative term is added beside an alpha.
    """

    required = {"portfolio", alpha_col, se_col, p_value_col}
    missing = sorted(required - set(estimates.columns))
    if missing:
        raise ValueError(f"Performance estimates missing columns: {missing}")

    out = estimates.copy()
    alpha = out[alpha_col].to_numpy(float)
    se = out[se_col].to_numpy(float)
    prior = _profile_prior(alpha, se)
    tau2 = prior["tau2"]
    se2 = se**2
    denominator = tau2 + se2
    shrinkage_weight = np.divide(
        tau2,
        denominator,
        out=np.zeros_like(se2),
        where=denominator > 0,
    )
    posterior_alpha = (
        shrinkage_weight * alpha + (1.0 - shrinkage_weight) * prior["mu"]
    )
    conditional_variance = np.divide(
        tau2 * se2,
        denominator,
        out=np.zeros_like(se2),
        where=denominator > 0,
    )
    # Include first-order uncertainty in the estimated common mean. This avoids
    # spuriously zero intervals when the cross-sectional heterogeneity is small.
    posterior_variance = conditional_variance + (
        (1.0 - shrinkage_weight) * prior["mu_se"]
    ) ** 2
    posterior_sd = np.sqrt(np.maximum(posterior_variance, 0.0))
    standardized_alpha = np.zeros_like(posterior_alpha)
    np.divide(
        posterior_alpha,
        posterior_sd,
        out=standardized_alpha,
        where=posterior_sd > 0,
    )
    zero_sd = posterior_sd <= 0
    standardized_alpha[zero_sd & (posterior_alpha > 0)] = np.inf
    standardized_alpha[zero_sd & (posterior_alpha < 0)] = -np.inf
    probability_positive = norm.cdf(standardized_alpha)

    out["prior_mean_alpha"] = prior["mu"]
    out["prior_tau"] = prior["tau"]
    out["shrinkage_weight"] = shrinkage_weight
    out["posterior_alpha"] = posterior_alpha
    out["posterior_alpha_sd"] = posterior_sd
    out["posterior_alpha_ci_low"] = posterior_alpha - 1.96 * posterior_sd
    out["posterior_alpha_ci_high"] = posterior_alpha + 1.96 * posterior_sd
    out["posterior_probability_alpha_positive"] = probability_positive
    out["posterior_alpha_annualized_bps"] = posterior_alpha * 12.0 * 10_000.0
    out["posterior_shortfall_annualized_bps"] = np.maximum(
        -out["posterior_alpha_annualized_bps"], 0.0
    )
    out["alpha_fdr_q_value"] = benjamini_hochberg(
        out[p_value_col].to_numpy(float)
    )
    out["performance_rank"] = (
        out["posterior_alpha"].rank(ascending=False, method="first").astype(int)
    )
    out = out.sort_values("performance_rank").reset_index(drop=True)
    return out, prior


def estimate_cross_sectional_performance(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    min_obs: int = 60,
    hac_lags: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Estimate HAC factor alphas and shrink them on a common cross-section."""

    rows: list[dict] = []
    residual_rows: list[pd.DataFrame] = []
    feature_names = ["alpha", *factor_cols]
    for portfolio, group in df.groupby("portfolio"):
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) < min_obs:
            continue
        y = g["excess_return"].to_numpy(float)
        X = design_matrix(g, factor_cols)
        fit = estimate_hac_factor_model(y, X, hac_lags=hac_lags)
        row = {
            "portfolio": portfolio,
            "factor_model": factor_model,
            "sample_start": g["date"].min(),
            "sample_end": g["date"].max(),
            "n_obs": int(len(g)),
            "alpha": fit["alpha"],
            "alpha_hac_se": fit["alpha_hac_se"],
            "alpha_hac_t_stat": fit["alpha_hac_t_stat"],
            "alpha_p_value": fit["alpha_p_value"],
            "hac_lags": int(hac_lags),
            "r_squared": fit["r_squared"],
            "residual_volatility": fit["residual_volatility"],
        }
        for name, coefficient, standard_error, t_stat, p_value in zip(
            feature_names,
            fit["beta"],
            fit["hac_standard_errors"],
            fit["t_stats"],
            fit["p_values"],
        ):
            row[f"coef_{name}"] = float(coefficient)
            row[f"hac_se_{name}"] = float(standard_error)
            row[f"hac_t_{name}"] = float(t_stat)
            row[f"p_value_{name}"] = float(p_value)
        rows.append(row)

        residual_frame = g[["date", "portfolio", "excess_return"]].copy()
        residual_frame["factor_model"] = factor_model
        residual_frame["fitted_value"] = fit["fitted"]
        residual_frame["residual"] = fit["residuals"]
        residual_rows.append(residual_frame)

    estimates = pd.DataFrame(rows)
    if estimates.empty:
        return estimates, pd.DataFrame(), {}
    scores, prior = hierarchical_shrinkage(estimates)
    residuals = pd.concat(residual_rows, ignore_index=True)
    return scores, residuals, prior


def rolling_cross_sectional_performance(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    window: int = 120,
    step: int = 12,
    min_obs: int | None = None,
    hac_lags: int = 12,
    first_end: int | None = None,
) -> pd.DataFrame:
    """Estimate shrinkage-adjusted factor performance in rolling windows."""

    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive integers.")
    dates = pd.Index(sorted(pd.to_datetime(df["date"]).unique()))
    if len(dates) < window:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    required = int(min_obs or window)
    start_end = max(window, int(first_end or window))
    for end in range(start_end, len(dates) + 1, step):
        window_dates = dates[end - window : end]
        sample = df[df["date"].isin(window_dates)].copy()
        scores, _, prior = estimate_cross_sectional_performance(
            sample,
            factor_cols,
            factor_model=factor_model,
            min_obs=required,
            hac_lags=min(hac_lags, max(window // 4, 0)),
        )
        if scores.empty:
            continue
        scores["window_start"] = pd.Timestamp(window_dates[0])
        scores["window_end"] = pd.Timestamp(window_dates[-1])
        scores["window_length"] = int(window)
        scores["rolling_step_months"] = int(step)
        scores["prior_mean_alpha"] = prior["mu"]
        scores["prior_tau"] = prior["tau"]
        scores["rank"] = scores["performance_rank"].astype(int)
        scores["quintile"] = (
            pd.qcut(
                scores["posterior_alpha"],
                5,
                labels=False,
                duplicates="drop",
            )
            .add(1)
            .astype("Int64")
        )
        rows.append(scores)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def forward_performance_validation(
    rolling_scores: pd.DataFrame,
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    forward_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate each ranking against strictly future factor-adjusted returns.

    Factor loadings are frozen at the end of each training window. No future
    return is used to estimate the score, prior, rank, or beta coefficients.
    """

    if forward_months <= 0:
        raise ValueError("forward_months must be positive.")
    if rolling_scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = pd.Index(sorted(data["date"].unique()))
    date_position = {pd.Timestamp(date): idx for idx, date in enumerate(dates)}
    observation_rows: list[dict] = []
    for row in rolling_scores.itertuples(index=False):
        end = pd.Timestamp(row.window_end)
        if end not in date_position:
            continue
        start_idx = date_position[end] + 1
        future_dates = dates[start_idx : start_idx + forward_months]
        if len(future_dates) < forward_months:
            continue
        future = data[
            (data["portfolio"] == row.portfolio)
            & (data["date"].isin(future_dates))
        ].sort_values("date")
        if len(future) != forward_months:
            continue
        beta = np.array([getattr(row, f"coef_{name}") for name in factor_cols])
        factor_component = future[factor_cols].to_numpy(float) @ beta
        realized_alpha = future["excess_return"].to_numpy(float) - factor_component
        observation_rows.append(
            {
                "portfolio": row.portfolio,
                "window_end": end,
                "forward_start": pd.Timestamp(future_dates[0]),
                "forward_end": pd.Timestamp(future_dates[-1]),
                "forward_months": int(forward_months),
                "posterior_alpha": float(row.posterior_alpha),
                "performance_rank": int(row.performance_rank),
                "quintile": int(row.quintile),
                "forward_alpha_monthly": float(np.mean(realized_alpha)),
                "forward_alpha_annualized_bps": float(
                    np.mean(realized_alpha) * 12.0 * 10_000.0
                ),
            }
        )

    observations = pd.DataFrame(observation_rows)
    if observations.empty:
        return observations, pd.DataFrame()

    summary_rows: list[dict] = []
    for window_end, group in observations.groupby("window_end"):
        if len(group) < 3:
            continue
        rho, p_value = spearmanr(
            group["posterior_alpha"], group["forward_alpha_monthly"]
        )
        top = group.loc[group["quintile"] == 5, "forward_alpha_monthly"]
        bottom = group.loc[group["quintile"] == 1, "forward_alpha_monthly"]
        summary_rows.append(
            {
                "window_end": pd.Timestamp(window_end),
                "forward_start": group["forward_start"].min(),
                "forward_end": group["forward_end"].max(),
                "forward_months": int(forward_months),
                "n_portfolios": int(len(group)),
                "rank_vs_forward_alpha_spearman": float(rho),
                "rank_vs_forward_alpha_p_value": float(p_value),
                "top_minus_bottom_forward_alpha_annualized_bps": float(
                    (top.mean() - bottom.mean()) * 12.0 * 10_000.0
                )
                if len(top) and len(bottom)
                else np.nan,
                "look_ahead_free": True,
            }
        )
    return observations, pd.DataFrame(summary_rows)


def summarize_forward_validation(
    forward_summary: pd.DataFrame,
    *,
    hac_lags: int = 4,
) -> pd.DataFrame:
    """Summarise forward validation with time-series HAC uncertainty.

    Rank correlations are averaged on the Fisher-z scale. The top-minus-bottom
    spread is averaged in annualised basis points. Both standard errors allow
    for serial dependence across validation windows.
    """

    required = {
        "window_end",
        "forward_start",
        "forward_end",
        "rank_vs_forward_alpha_spearman",
        "top_minus_bottom_forward_alpha_annualized_bps",
        "look_ahead_free",
    }
    missing = sorted(required - set(forward_summary.columns))
    if missing:
        raise ValueError(f"Forward validation summary missing columns: {missing}")
    if forward_summary.empty:
        return pd.DataFrame()

    data = forward_summary.sort_values("window_end").copy()
    correlations = data["rank_vs_forward_alpha_spearman"].to_numpy(float)
    correlations = np.clip(
        correlations[np.isfinite(correlations)], -0.999999, 0.999999
    )
    spreads = data[
        "top_minus_bottom_forward_alpha_annualized_bps"
    ].to_numpy(float)
    spreads = spreads[np.isfinite(spreads)]
    if len(correlations) < 2 or len(spreads) < 2:
        raise ValueError("At least two valid validation windows are required.")

    z_values = np.arctanh(correlations)
    z_fit = estimate_hac_factor_model(
        z_values,
        np.ones((len(z_values), 1)),
        hac_lags=hac_lags,
    )
    spread_fit = estimate_hac_factor_model(
        spreads,
        np.ones((len(spreads), 1)),
        hac_lags=hac_lags,
    )
    z_mean = z_fit["alpha"]
    z_se = z_fit["alpha_hac_se"]
    spread_mean = spread_fit["alpha"]
    spread_se = spread_fit["alpha_hac_se"]
    return pd.DataFrame(
        [
            {
                "n_validation_windows": int(len(data)),
                "first_training_window_end": pd.to_datetime(
                    data["window_end"]
                ).min(),
                "last_training_window_end": pd.to_datetime(
                    data["window_end"]
                ).max(),
                "all_look_ahead_free": bool(data["look_ahead_free"].all()),
                "hac_lags": int(hac_lags),
                "mean_rank_spearman_fisher": float(np.tanh(z_mean)),
                "rank_spearman_ci_low": float(np.tanh(z_mean - 1.96 * z_se)),
                "rank_spearman_ci_high": float(np.tanh(z_mean + 1.96 * z_se)),
                "rank_spearman_hac_p_value": float(z_fit["alpha_p_value"]),
                "mean_top_minus_bottom_forward_alpha_annualized_bps": float(
                    spread_mean
                ),
                "top_minus_bottom_hac_se_bps": float(spread_se),
                "top_minus_bottom_ci_low_bps": float(
                    spread_mean - 1.96 * spread_se
                ),
                "top_minus_bottom_ci_high_bps": float(
                    spread_mean + 1.96 * spread_se
                ),
                "top_minus_bottom_hac_p_value": float(
                    spread_fit["alpha_p_value"]
                ),
                "positive_top_minus_bottom_fraction": float(np.mean(spreads > 0)),
            }
        ]
    )


def block_bootstrap_rank_uncertainty(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    n_bootstrap: int = 200,
    block_length: int = 12,
    hac_lags: int = 12,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Estimate score and rank uncertainty with a common-date circular bootstrap."""

    if n_bootstrap <= 0 or block_length <= 0:
        raise ValueError("n_bootstrap and block_length must be positive.")
    dates = pd.Index(sorted(pd.to_datetime(df["date"]).unique()))
    portfolios = sorted(df["portfolio"].unique())
    n_dates = len(dates)
    if n_dates < max(block_length, 3):
        raise ValueError("Insufficient dates for the requested block bootstrap.")

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for portfolio in portfolios:
        group = (
            df[df["portfolio"] == portfolio]
            .set_index("date")
            .reindex(dates)
        )
        if group[["excess_return", *factor_cols]].isna().any().any():
            raise ValueError("Block bootstrap requires a balanced portfolio panel.")
        arrays[portfolio] = (
            group["excess_return"].to_numpy(float),
            group[factor_cols].to_numpy(float),
        )

    rng = np.random.default_rng(random_seed)
    draws: list[pd.DataFrame] = []
    n_blocks = int(np.ceil(n_dates / block_length))
    for replicate in range(n_bootstrap):
        starts = rng.integers(0, n_dates, size=n_blocks)
        indices = np.concatenate(
            [
                (start + np.arange(block_length, dtype=int)) % n_dates
                for start in starts
            ]
        )[:n_dates]
        estimates = []
        for portfolio in portfolios:
            y_full, factors_full = arrays[portfolio]
            y = y_full[indices]
            X = np.column_stack([np.ones(n_dates), factors_full[indices]])
            fit = estimate_hac_factor_model(
                y,
                X,
                hac_lags=min(hac_lags, max(n_dates // 4, 0)),
            )
            estimates.append(
                {
                    "portfolio": portfolio,
                    "alpha": fit["alpha"],
                    "alpha_hac_se": fit["alpha_hac_se"],
                    "alpha_p_value": fit["alpha_p_value"],
                }
            )
        scores, _ = hierarchical_shrinkage(pd.DataFrame(estimates))
        scores["bootstrap_replicate"] = int(replicate)
        draws.append(
            scores[
                [
                    "portfolio",
                    "posterior_alpha",
                    "performance_rank",
                    "bootstrap_replicate",
                ]
            ]
        )

    bootstrap = pd.concat(draws, ignore_index=True)
    top_cutoff = max(1, int(np.ceil(len(portfolios) / 5)))
    bottom_cutoff = len(portfolios) - top_cutoff + 1
    rows = []
    for portfolio, group in bootstrap.groupby("portfolio"):
        rows.append(
            {
                "portfolio": portfolio,
                "bootstrap_replicates": int(group["bootstrap_replicate"].nunique()),
                "bootstrap_posterior_alpha_median": float(
                    group["posterior_alpha"].median()
                ),
                "bootstrap_posterior_alpha_ci_low": float(
                    group["posterior_alpha"].quantile(0.025)
                ),
                "bootstrap_posterior_alpha_ci_high": float(
                    group["posterior_alpha"].quantile(0.975)
                ),
                "bootstrap_rank_median": float(group["performance_rank"].median()),
                "bootstrap_rank_ci_low": float(
                    group["performance_rank"].quantile(0.025)
                ),
                "bootstrap_rank_ci_high": float(
                    group["performance_rank"].quantile(0.975)
                ),
                "bootstrap_top_quintile_probability": float(
                    np.mean(group["performance_rank"] <= top_cutoff)
                ),
                "bootstrap_bottom_quintile_probability": float(
                    np.mean(group["performance_rank"] >= bottom_cutoff)
                ),
                "bootstrap_block_length_months": int(block_length),
                "bootstrap_random_seed": int(random_seed),
            }
        )
    return pd.DataFrame(rows).sort_values("portfolio").reset_index(drop=True)
