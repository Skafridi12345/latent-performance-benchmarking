"""Incremental 'rolling vs. static ranking' forward test.

The canonical forward validation in :mod:`analysis.latent_performance` asks
whether the time-varying (rolling) latent-performance ranking predicts future
FF3 alpha. That test is near-tautological: the 25 size/book-to-market cells
have a *persistent* characteristic mispricing under FF3, so a ranking built
from past alpha will correlate with future alpha even if it carries no dynamic
information beyond "small-value tends to look good, small-growth tends to look
bad".

This module isolates the *incremental* content of the rolling ranking. For
every training window it compares the rolling ranking against a **fixed,
a-priori characteristic ranking** that never updates, and tests whether the
rolling ranking predicts forward alpha any better than the static one.

The null hypothesis of interest is therefore not "zero predictive power" but
"no predictive power beyond the static characteristic ordering". If the mean
increment (rolling minus static) is statistically indistinguishable from zero,
the apparent forecasting ability of the rolling ranking is fully explained by
persistent characteristic mispricing.

Design principles
-----------------
* Reuse the existing window scheme, data loading, empirical-Bayes machinery,
  and forward-alpha construction. The static benchmark is layered *on top of*
  the observation-level output of
  :func:`analysis.latent_performance.forward_performance_validation`, so the
  rolling and static rankings are scored on identical forward observations,
  frozen betas, and identical windows. Nothing about the rolling result is
  re-estimated or perturbed.
* Inference reuses :func:`analysis.latent_performance.estimate_hac_factor_model`
  for HAC (Bartlett) standard errors, mirroring
  :func:`analysis.latent_performance.summarize_forward_validation`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figure generation must not require a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

from analysis.latent_performance import (  # noqa: E402
    estimate_cross_sectional_performance,
    estimate_hac_factor_model,
    forward_performance_validation,
    rolling_cross_sectional_performance,
)
from sfa.loaders import build_factor_dataset  # noqa: E402

STATIC_SCHEMES = ("lexicographic", "diagonal", "unconditional_alpha")
# All benchmarks, including the leakage-free time-varying expanding window.
ALL_SCHEMES = (*STATIC_SCHEMES, "expanding_window")
BENCHMARK_LABELS = {
    "lexicographic": "static characteristic ordering (size asc, BM desc)",
    "diagonal": "static characteristic ordering (diagonal value-minus-size)",
    "unconditional_alpha": "full-sample oracle alpha ordering (look-ahead)",
    "expanding_window": "expanding-window (non-peeking) ranking",
}

ANNUALIZE_BPS = 12.0 * 10_000.0


# ---------------------------------------------------------------------------
# Static characteristic ranking
# ---------------------------------------------------------------------------
def parse_grid_position(name: str) -> tuple[int, int]:
    """Map a Fama-French 25-portfolio label to ``(size_quintile, bm_quintile)``.

    Quintiles are 1..5. Size 1 is the smallest, size 5 the largest. Book-to-
    market 1 is growth (low B/M), 5 is value (high B/M). This mirrors the
    label convention used by :func:`analysis.figures._portfolio_grid_position`
    and the Ken French corner naming (``SMALL``/``BIG`` x ``LoBM``/``HiBM``).
    """

    size: int | None = None
    if name.startswith("SMALL") or name.startswith("ME1"):
        size = 1
    elif name.startswith("ME2"):
        size = 2
    elif name.startswith("ME3"):
        size = 3
    elif name.startswith("ME4"):
        size = 4
    elif name.startswith("BIG") or name.startswith("ME5"):
        size = 5

    bm: int | None = None
    if "LoBM" in name or "BM1" in name:
        bm = 1
    elif "BM2" in name:
        bm = 2
    elif "BM3" in name:
        bm = 3
    elif "BM4" in name:
        bm = 4
    elif "HiBM" in name or "BM5" in name:
        bm = 5

    if size is None or bm is None:
        raise ValueError(f"Cannot parse size/BM grid position from label '{name}'.")
    return size, bm


def static_characteristic_ranking(
    portfolios: list[str],
    *,
    scheme: str = "lexicographic",
) -> pd.DataFrame:
    """Return a fixed, time-invariant characteristic ranking of the 25 cells.

    The ranking is a *prior* ordering derived only from the grid coordinates,
    never from realized alpha, so it encodes structurally known characteristic
    mispricing without peeking at the outcome being predicted.

    Schemes
    -------
    ``"lexicographic"`` (default)
        Sort by size quintile ascending, then book-to-market quintile
        descending. This is the literal algorithm requested in the task spec:
        best (rank 1) is the smallest / highest-B/M cell (``SMALL HiBM``) and
        worst (rank 25) is the largest / lowest-B/M cell (``BIG LoBM``). Note
        that under this rule ``SMALL LoBM`` is rank 5, *not* rank 25 -- the
        small-growth anomaly is a diagonal feature that a lexicographic rule
        does not reproduce.
    ``"diagonal"``
        Order by a single value-minus-size score ``(bm - size)`` (ties broken
        by size ascending, then bm descending). This leans on the diagonal
        value/size gradient rather than the additive lexicographic one and is
        reported as a robustness alternative so the incremental verdict does
        not hinge on one arbitrary a-priori rule.

    Returns a frame with ``portfolio``, ``size_quintile``, ``bm_quintile``,
    ``static_rank`` (1 = best) and ``static_score`` (higher = better, equal to
    ``n - static_rank + 1``) so downstream correlations share the "higher is
    better" orientation of ``posterior_alpha``.
    """

    rows = []
    for name in portfolios:
        size, bm = parse_grid_position(name)
        rows.append({"portfolio": name, "size_quintile": size, "bm_quintile": bm})
    grid = pd.DataFrame(rows)

    if scheme == "lexicographic":
        # size ascending, then BM descending -> SMALL HiBM best, BIG LoBM worst.
        ordered = grid.sort_values(
            ["size_quintile", "bm_quintile"], ascending=[True, False]
        )
    elif scheme == "diagonal":
        grid = grid.assign(_score=grid["bm_quintile"] - grid["size_quintile"])
        ordered = grid.sort_values(
            ["_score", "size_quintile", "bm_quintile"],
            ascending=[False, True, False],
        ).drop(columns="_score")
    else:
        raise ValueError("scheme must be 'lexicographic' or 'diagonal'.")

    ordered = ordered.reset_index(drop=True)
    n = len(ordered)
    ordered["static_rank"] = np.arange(1, n + 1, dtype=int)
    ordered["static_scheme"] = scheme
    # Higher score = better, so a positive Spearman with forward alpha means the
    # ordering predicts, matching the rolling test's use of posterior_alpha.
    ordered["static_score"] = n - ordered["static_rank"] + 1
    return ordered


def unconditional_alpha_ranking(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    hac_lags: int = 12,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Return a fixed ranking from the *full-sample* posterior alpha ordering.

    This is an **oracle / look-ahead** static benchmark: it is estimated once
    on the entire sample and therefore uses information that post-dates every
    forward window. It is not a tradable strategy. Its purpose is to bound what
    a single time-invariant ranking can achieve. If the rolling ranking beats
    even this, the extra power is genuinely dynamic; if the rolling ranking
    cannot beat it, a single (correct but irregular) fixed ordering suffices
    and the time variation adds nothing.
    """

    scores, _, _, _ = estimate_cross_sectional_performance(
        df,
        factor_cols,
        factor_model=factor_model,
        min_obs=min_obs,
        hac_lags=hac_lags,
    )
    if scores.empty:
        raise ValueError(
            "Full-sample estimation produced no portfolios; the sample is too "
            f"short for min_obs={min_obs}."
        )
    out = scores[["portfolio", "posterior_alpha", "performance_rank"]].copy()
    out = out.rename(columns={"performance_rank": "static_rank"})
    out["static_scheme"] = "unconditional_alpha"
    # Higher posterior alpha = better, matching the rolling test orientation.
    out["static_score"] = out["posterior_alpha"].to_numpy(float)
    return out[["portfolio", "static_rank", "static_scheme", "static_score"]]


def expanding_window_ranking(
    df: pd.DataFrame,
    factor_cols: list[str],
    window_ends: list[pd.Timestamp],
    *,
    factor_model: str,
    hac_lags: int = 12,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Return a **non-peeking, time-varying** ranking for each window end.

    For every training-window end, the ranking is re-estimated on *all* data
    from the sample start up to that date (an expanding window), rather than the
    trailing 120-month rolling window. It uses only past information, so unlike
    the ``unconditional_alpha`` oracle it is leakage-free.

    Comparing the rolling ranking against this benchmark is the fair test of
    whether the *trailing-window time variation* helps: the expanding ranking
    accumulates history and converges toward the stable long-run ordering, while
    the rolling ranking discards data older than the window and chases recent
    alpha. Returned columns are ``window_end``, ``portfolio``, ``static_rank``
    (1 = best) and ``static_score`` (higher = better) so it plugs straight into
    :func:`build_incremental_window_metrics` as a time-varying benchmark.
    """

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    ends = pd.Index(sorted(pd.to_datetime(pd.Series(window_ends)).unique()))
    rows: list[pd.DataFrame] = []
    for window_end in ends:
        sample = data[data["date"] <= window_end]
        scores, _, _, _ = estimate_cross_sectional_performance(
            sample,
            factor_cols,
            factor_model=factor_model,
            min_obs=min_obs,
            hac_lags=hac_lags,
        )
        if scores.empty:
            continue
        frame = scores[["portfolio", "posterior_alpha", "performance_rank"]].copy()
        frame = frame.rename(columns={"performance_rank": "static_rank"})
        frame["window_end"] = pd.Timestamp(window_end)
        frame["static_score"] = frame["posterior_alpha"].to_numpy(float)
        rows.append(frame[["window_end", "portfolio", "static_rank", "static_score"]])
    if not rows:
        return pd.DataFrame(
            columns=["window_end", "portfolio", "static_rank", "static_score"]
        )
    out = pd.concat(rows, ignore_index=True)
    out["static_scheme"] = "expanding_window"
    return out


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def newey_west_bandwidth(n_obs: int) -> int:
    """Automatic Bartlett bandwidth via the Newey-West (1994) plug-in rule."""

    if n_obs < 2:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def _hac_one_sided(series: np.ndarray, *, hac_lags: int) -> dict:
    """HAC mean, SE, and one-sided (mean > 0) p-value for a scalar series."""

    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("At least two finite observations are required.")
    fit = estimate_hac_factor_model(
        values, np.ones((len(values), 1)), hac_lags=hac_lags
    )
    mean = fit["alpha"]
    se = fit["alpha_hac_se"]
    t_stat = fit["alpha_hac_t_stat"]
    return {
        "mean": float(mean),
        "median": float(np.median(values)),
        "hac_se": float(se),
        "hac_t_stat": float(t_stat),
        "hac_lags": int(fit["hac_lags"]),
        # estimate_hac_factor_model returns a two-sided p-value; convert to the
        # one-sided upper-tail test H1: mean > 0.
        "hac_p_value_one_sided": float(norm.sf(t_stat)),
        "hac_p_value_two_sided": float(fit["alpha_p_value"]),
        "n_obs": int(len(values)),
    }


def _block_bootstrap_mean(
    series: np.ndarray,
    *,
    block_length: int,
    n_bootstrap: int,
    random_seed: int,
    ci: float = 0.95,
) -> dict:
    """Circular moving-block bootstrap of the mean of a serially dependent series.

    Blocks preserve the short-range autocorrelation induced by overlapping
    training windows. Returns the percentile confidence interval and the
    bootstrap one-sided p-value (share of resampled means <= 0).
    """

    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        raise ValueError("At least two finite observations are required.")
    block_length = max(1, min(int(block_length), n))
    rng = np.random.default_rng(random_seed)
    n_blocks = int(np.ceil(n / block_length))
    means = np.empty(n_bootstrap, dtype=float)
    offsets = np.arange(block_length)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(start + offsets) % n for start in starts])[:n]
        means[b] = values[idx].mean()
    lower_q = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(means, [lower_q, 1.0 - lower_q])
    return {
        "bootstrap_mean": float(means.mean()),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_p_value_one_sided": float(np.mean(means <= 0.0)),
        "bootstrap_block_length": int(block_length),
        "bootstrap_replicates": int(n_bootstrap),
        "bootstrap_random_seed": int(random_seed),
        "bootstrap_ci_level": float(ci),
    }


# ---------------------------------------------------------------------------
# Per-window comparison
# ---------------------------------------------------------------------------
def build_incremental_window_metrics(
    forward_observations: pd.DataFrame,
    benchmark_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rolling vs. benchmark predictive metrics for each forward window.

    ``forward_observations`` is the observation-level frame returned by
    :func:`analysis.latent_performance.forward_performance_validation`, holding
    one row per (window_end, portfolio) with the frozen rolling
    ``posterior_alpha``/``performance_rank`` and the realized
    ``forward_alpha_monthly``.

    ``benchmark_ranking`` supplies the competing ordering with columns
    ``static_rank`` (1 = best) and ``static_score`` (higher = better). It may be:

    * **time-invariant** -- one row per ``portfolio`` (the characteristic or
      full-sample oracle rankings); merged on ``portfolio`` and reused for every
      window; or
    * **time-varying** -- one row per (``window_end``, ``portfolio``) (the
      expanding-window ranking); merged on both keys so each window is scored
      against its own contemporaneous, non-peeking ordering.

    Both the rolling and benchmark orderings are scored against the *same*
    ``forward_alpha_monthly``, so the comparison isolates the ordering, not the
    forward-outcome definition.
    """

    required = {
        "window_end",
        "portfolio",
        "posterior_alpha",
        "performance_rank",
        "forward_alpha_monthly",
    }
    missing = sorted(required - set(forward_observations.columns))
    if missing:
        raise ValueError(f"Forward observations missing columns: {missing}")
    bench_missing = sorted(
        {"static_rank", "static_score"} - set(benchmark_ranking.columns)
    )
    if bench_missing:
        raise ValueError(f"Benchmark ranking missing columns: {bench_missing}")
    if forward_observations.empty:
        return pd.DataFrame()

    time_varying = "window_end" in benchmark_ranking.columns
    merge_keys = ["window_end", "portfolio"] if time_varying else ["portfolio"]
    bench_cols = [*merge_keys, "static_rank", "static_score"]
    merged = forward_observations.merge(
        benchmark_ranking[bench_cols],
        on=merge_keys,
        how="left",
        validate="one_to_one" if time_varying else "many_to_one",
    )
    if merged["static_rank"].isna().any():
        unmatched = sorted(
            merged.loc[merged["static_rank"].isna(), "portfolio"].unique()
        )
        raise ValueError(f"Benchmark ranking is missing portfolios: {unmatched}")

    rows: list[dict] = []
    for window_end, group in merged.groupby("window_end"):
        # Need at least two distinct portfolios for a correlation and enough
        # names to populate a top and bottom quintile.
        if group["portfolio"].nunique() < 5:
            continue
        forward = group["forward_alpha_monthly"].to_numpy(float)
        rolling_score = group["posterior_alpha"].to_numpy(float)
        static_score = group["static_score"].to_numpy(float)

        rolling_rho = spearmanr(rolling_score, forward).correlation
        static_rho = spearmanr(static_score, forward).correlation

        n = len(group)
        k = max(1, round(n / 5))  # top / bottom quintile size (5 for 25 cells)
        # Same members, two orderings -> a like-for-like long-short comparison.
        rolling_top = group.nsmallest(k, "performance_rank")["forward_alpha_monthly"]
        rolling_bottom = group.nlargest(k, "performance_rank")["forward_alpha_monthly"]
        static_top = group.nsmallest(k, "static_rank")["forward_alpha_monthly"]
        static_bottom = group.nlargest(k, "static_rank")["forward_alpha_monthly"]

        rolling_spread = (rolling_top.mean() - rolling_bottom.mean()) * ANNUALIZE_BPS
        static_spread = (static_top.mean() - static_bottom.mean()) * ANNUALIZE_BPS

        rows.append(
            {
                "window_end": pd.Timestamp(window_end),
                "n_portfolios": int(n),
                "quintile_size": int(k),
                "rolling_rank_spearman": float(rolling_rho),
                "static_rank_spearman": float(static_rho),
                "rank_spearman_increment": float(rolling_rho - static_rho),
                "rolling_spread_bps": float(rolling_spread),
                "static_spread_bps": float(static_spread),
                "spread_increment_bps": float(rolling_spread - static_spread),
            }
        )

    return pd.DataFrame(rows).sort_values("window_end").reset_index(drop=True)


def _summarize_increment(
    metrics: pd.DataFrame,
    column: str,
    *,
    hac_lags: int,
    block_length: int,
    n_bootstrap: int,
    random_seed: int,
) -> dict:
    """Bundle HAC and block-bootstrap inference for one increment series."""

    series = metrics[column].to_numpy(float)
    hac = _hac_one_sided(series, hac_lags=hac_lags)
    boot = _block_bootstrap_mean(
        series,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    return {"metric": column, **hac, **boot}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_cumulative_increment(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    benchmark_label: str = "benchmark ordering",
) -> Path:
    """Plot the cumulative rolling-minus-benchmark increment over time."""

    data = metrics.sort_values("window_end")
    dates = pd.to_datetime(data["window_end"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
    axes[0].plot(
        dates,
        data["rank_spearman_increment"].cumsum(),
        color="#2f6f9f",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Cumulative rank-corr\nincrement (rolling - benchmark)")
    axes[0].set_title(
        "Incremental predictive power of the rolling ranking\n"
        f"vs. {benchmark_label}\n"
        "(cumulative sum; upward slope = rolling beats the benchmark)"
    )

    axes[1].axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
    axes[1].plot(
        dates,
        data["spread_increment_bps"].cumsum(),
        color="#9f552f",
        linewidth=2.0,
    )
    axes[1].set_ylabel("Cumulative spread\nincrement (bps)")
    axes[1].set_xlabel("Training window end")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_incremental_test(
    portfolio_file: Path,
    factor_file: Path,
    *,
    factor_model: str = "ff3",
    scheme: str = "lexicographic",
    rolling_window: int = 120,
    rolling_step: int = 12,
    min_obs: int = 60,
    hac_lags: int = 12,
    forward_months: int = 12,
    forward_hac_lags: int | None = None,
    bootstrap_block_length: int = 2,
    bootstrap_replicates: int = 5_000,
    bootstrap_seed: int = 2026,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the incremental rolling-vs-static forward test end to end.

    Steps: (1) load the same factor panel; (2) reuse
    :func:`rolling_cross_sectional_performance` and
    :func:`forward_performance_validation` to obtain the rolling ranking and
    its forward observations; (3) score a fixed a-priori characteristic
    ranking on the identical forward observations; (4) run HAC and block-
    bootstrap inference on the rolling-minus-static increments; (5) return all
    statistics, a per-window table, and a plain-language verdict.

    ``forward_hac_lags`` defaults to the Newey-West automatic bandwidth. Set
    ``scheme='diagonal'`` for the robustness alternative.
    """

    dataset = build_factor_dataset(
        portfolio_file, factor_file, factor_model=factor_model
    )
    df = dataset.data
    factor_cols = dataset.factor_cols

    rolling = rolling_cross_sectional_performance(
        df,
        factor_cols,
        window=rolling_window,
        step=rolling_step,
        factor_model=dataset.factor_model,
        min_obs=rolling_window,
        hac_lags=hac_lags,
    )
    if rolling.empty:
        raise ValueError("Rolling estimation produced no windows; check the sample.")

    forward_observations, _ = forward_performance_validation(
        rolling, df, factor_cols, forward_months=forward_months
    )
    if forward_observations.empty:
        raise ValueError("Forward validation produced no observations.")

    if scheme == "unconditional_alpha":
        static_ranking = unconditional_alpha_ranking(
            df,
            factor_cols,
            factor_model=dataset.factor_model,
            hac_lags=hac_lags,
            min_obs=min_obs,
        )
    elif scheme == "expanding_window":
        static_ranking = expanding_window_ranking(
            df,
            factor_cols,
            forward_observations["window_end"].tolist(),
            factor_model=dataset.factor_model,
            hac_lags=hac_lags,
            min_obs=min_obs,
        )
    else:
        portfolios = sorted(df["portfolio"].unique())
        static_ranking = static_characteristic_ranking(portfolios, scheme=scheme)
    benchmark_label = BENCHMARK_LABELS.get(scheme, "benchmark ordering")
    metrics = build_incremental_window_metrics(forward_observations, static_ranking)
    if len(metrics) < 2:
        raise ValueError("Too few comparable windows for inference.")

    if forward_hac_lags is None:
        forward_hac_lags = newey_west_bandwidth(len(metrics))

    rank_increment = _summarize_increment(
        metrics,
        "rank_spearman_increment",
        hac_lags=forward_hac_lags,
        block_length=bootstrap_block_length,
        n_bootstrap=bootstrap_replicates,
        random_seed=bootstrap_seed,
    )
    spread_increment = _summarize_increment(
        metrics,
        "spread_increment_bps",
        hac_lags=forward_hac_lags,
        block_length=bootstrap_block_length,
        n_bootstrap=bootstrap_replicates,
        random_seed=bootstrap_seed + 1,
    )

    # Level (non-incremental) references for context.
    rolling_rho_hac = _hac_one_sided(
        metrics["rolling_rank_spearman"].to_numpy(float), hac_lags=forward_hac_lags
    )
    static_rho_hac = _hac_one_sided(
        metrics["static_rank_spearman"].to_numpy(float), hac_lags=forward_hac_lags
    )

    alpha_level = 0.05
    rank_significant = (
        rank_increment["hac_p_value_one_sided"] < alpha_level
        and rank_increment["bootstrap_ci_low"] > 0.0
    )
    spread_significant = (
        spread_increment["hac_p_value_one_sided"] < alpha_level
        and spread_increment["bootstrap_ci_low"] > 0.0
    )
    adds_value = rank_significant or spread_significant
    if adds_value:
        verdict = (
            "The rolling ranking adds significant incremental predictive power "
            f"over the {benchmark_label}."
        )
    else:
        verdict = (
            "The rolling ranking does not provide statistically significant "
            f"incremental information over the {benchmark_label} -- the apparent "
            "forecasting ability is consistent with persistent characteristic "
            "mispricing."
        )

    figure_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(
            tables_dir / f"incremental_validation_windows_{scheme}.csv", index=False
        )
        summary_frame = pd.DataFrame([rank_increment, spread_increment])
        summary_frame.insert(0, "static_scheme", scheme)
        summary_frame.to_csv(
            tables_dir / f"incremental_validation_summary_{scheme}.csv", index=False
        )
        static_ranking.to_csv(
            tables_dir / f"benchmark_ranking_{scheme}.csv", index=False
        )
        figure_path = plot_cumulative_increment(
            metrics,
            figures_dir / f"incremental_rolling_vs_static_{scheme}.png",
            benchmark_label=benchmark_label,
        )

    results = {
        "static_scheme": scheme,
        "benchmark_label": benchmark_label,
        "n_windows": int(len(metrics)),
        "forward_hac_lags": int(forward_hac_lags),
        "static_ranking": static_ranking,
        "window_metrics": metrics,
        "rank_spearman_increment": rank_increment,
        "spread_increment": spread_increment,
        "rolling_rank_spearman_level": rolling_rho_hac,
        "static_rank_spearman_level": static_rho_hac,
        "rank_increment_significant": bool(rank_significant),
        "spread_increment_significant": bool(spread_significant),
        "rolling_adds_incremental_value": bool(adds_value),
        "verdict": verdict,
        "figure_path": str(figure_path) if figure_path else None,
    }

    if verbose:
        print(_format_report(results))
    return results


def _format_report(results: dict) -> str:
    rank = results["rank_spearman_increment"]
    spread = results["spread_increment"]
    roll = results["rolling_rank_spearman_level"]
    stat = results["static_rank_spearman_level"]
    lines = [
        "=" * 78,
        "INCREMENTAL FORWARD TEST: rolling ranking vs. a benchmark ordering",
        "=" * 78,
        f"Benchmark scheme       : {results['static_scheme']}",
        f"Benchmark ordering     : {results['benchmark_label']}",
        f"Validation windows     : {results['n_windows']}",
        f"HAC (Bartlett) lags    : {results['forward_hac_lags']} "
        "(Newey-West automatic bandwidth unless overridden)",
        "",
        "Level rank correlation with forward alpha (higher = more predictive):",
        f"  rolling ranking      : mean {roll['mean']:+.4f}  "
        f"(HAC 1-sided p = {roll['hac_p_value_one_sided']:.3g})",
        f"  benchmark ranking    : mean {stat['mean']:+.4f}  "
        f"(HAC 1-sided p = {stat['hac_p_value_one_sided']:.3g})",
        "",
        "(1) INCREMENTAL RANK CORRELATION  (rolling - benchmark Spearman)",
        f"    mean               : {rank['mean']:+.4f}   median {rank['median']:+.4f}",
        f"    HAC SE / t         : {rank['hac_se']:.4f} / {rank['hac_t_stat']:+.3f}",
        f"    HAC 1-sided p      : {rank['hac_p_value_one_sided']:.4g}",
        f"    block-bootstrap 95%: [{rank['bootstrap_ci_low']:+.4f}, "
        f"{rank['bootstrap_ci_high']:+.4f}]  "
        f"(1-sided p = {rank['bootstrap_p_value_one_sided']:.4g})",
        f"    distinguishable > 0: {results['rank_increment_significant']}",
        "",
        "(2) INCREMENTAL LONG-SHORT SPREAD  (rolling - benchmark, annualized bps)",
        f"    mean               : {spread['mean']:+.1f} bps   "
        f"median {spread['median']:+.1f} bps",
        f"    HAC SE / t         : {spread['hac_se']:.1f} / "
        f"{spread['hac_t_stat']:+.3f}",
        f"    HAC 1-sided p      : {spread['hac_p_value_one_sided']:.4g}",
        f"    block-bootstrap 95%: [{spread['bootstrap_ci_low']:+.1f}, "
        f"{spread['bootstrap_ci_high']:+.1f}] bps  "
        f"(1-sided p = {spread['bootstrap_p_value_one_sided']:.4g})",
        f"    distinguishable > 0: {results['spread_increment_significant']}",
        "",
        "VERDICT:",
        f"  {results['verdict']}",
        "=" * 78,
    ]
    return "\n".join(lines)


def _default_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "data" / "25_size_bm_portfolios.csv",
        root / "data" / "ff3_factors.csv",
        root / "results",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    port_default, factor_default, results_default = _default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Incremental rolling-vs-static forward test for the latent "
            "performance ranking."
        )
    )
    parser.add_argument("--portfolio-file", type=Path, default=port_default)
    parser.add_argument("--factor-file", type=Path, default=factor_default)
    parser.add_argument("--results-dir", type=Path, default=results_default)
    parser.add_argument("--factor-model", type=str, default="ff3")
    parser.add_argument(
        "--scheme",
        type=str,
        default="lexicographic",
        choices=[*ALL_SCHEMES, "all"],
        help=(
            "Benchmark ranking definition. 'all' runs every scheme. "
            "'unconditional_alpha' is a full-sample oracle (look-ahead) benchmark; "
            "'expanding_window' is the leakage-free time-varying benchmark."
        ),
    )
    parser.add_argument("--rolling-window", type=int, default=120)
    parser.add_argument("--rolling-step", type=int, default=12)
    parser.add_argument("--hac-lags", type=int, default=12)
    parser.add_argument("--forward-months", type=int, default=12)
    parser.add_argument(
        "--forward-hac-lags",
        type=int,
        default=None,
        help="Override the automatic Newey-West bandwidth for the increment test.",
    )
    parser.add_argument("--bootstrap-block-length", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--no-write", action="store_true", help="Skip writing outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    schemes = list(ALL_SCHEMES) if args.scheme == "all" else [args.scheme]
    output_dir = None if args.no_write else args.results_dir
    for scheme in schemes:
        run_incremental_test(
            args.portfolio_file,
            args.factor_file,
            factor_model=args.factor_model,
            scheme=scheme,
            rolling_window=args.rolling_window,
            rolling_step=args.rolling_step,
            hac_lags=args.hac_lags,
            forward_months=args.forward_months,
            forward_hac_lags=args.forward_hac_lags,
            bootstrap_block_length=args.bootstrap_block_length,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            output_dir=output_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()
