from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _portfolio_grid_position(name: str) -> tuple[int | None, int | None]:
    size = None
    bm = None
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
    return size, bm


def static_ae_ranking(static_scores: pd.DataFrame, out: Path) -> None:
    """Plot static adjusted-efficiency scores by portfolio."""

    data = static_scores.sort_values("AE", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(data["portfolio"], data["AE"], color="#2f6f9f")
    ax.set_xlabel("Adjusted efficiency (AE)")
    ax.set_title("Static SFA Adjusted Efficiency Ranking")
    ax.set_xlim(
        max(0.0, data["AE"].min() - 0.002), min(1.001, data["AE"].max() + 0.001)
    )
    fig.tight_layout()
    fig.savefig(out / "static_ae_ranking.png")
    plt.close(fig)


def ae_heatmap(static_scores: pd.DataFrame, out: Path) -> None:
    """Plot static AE values on the 5x5 size/book-to-market grid."""

    records = []
    for _, row in static_scores.iterrows():
        size, bm = _portfolio_grid_position(str(row["portfolio"]))
        if size is not None and bm is not None:
            records.append({"size": size, "book_to_market": bm, "AE": row["AE"]})
    heat = pd.DataFrame(records).pivot(
        index="size", columns="book_to_market", values="AE"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heat, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": "AE"}, ax=ax
    )
    ax.set_xlabel("Book-to-market quintile")
    ax.set_ylabel("Size quintile")
    ax.set_title("Static AE Across Size x Book-to-Market Portfolios")
    fig.tight_layout()
    fig.savefig(out / "ae_heatmap_size_bm.png")
    plt.close(fig)


def rolling_ae_timeseries(rolling: pd.DataFrame, out: Path) -> None:
    """Plot cross-sectional rolling AE mean and dispersion."""

    df = rolling.dropna(subset=["AE"]).copy()
    df["window_end"] = pd.to_datetime(df["window_end"])
    summary = (
        df.groupby("window_end")["AE"]
        .agg(
            mean="mean", p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90)
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = summary["window_end"].to_numpy()
    ax.plot(x, summary["mean"], color="#1f4e79", label="Cross-sectional mean")
    ax.fill_between(
        x,
        summary["p10"].to_numpy(float),
        summary["p90"].to_numpy(float),
        color="#6aaed6",
        alpha=0.25,
        label="10th-90th percentile",
    )
    ax.set_xlabel("Window end")
    ax.set_ylabel("Adjusted efficiency (AE)")
    ax.set_title("Rolling SFA Adjusted Efficiency")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "rolling_ae_timeseries.png")
    plt.close(fig)


def rank_persistence_plot(persistence: pd.DataFrame, out: Path) -> None:
    """Plot rolling rank and performance-score persistence by horizon."""

    summary = (
        persistence.groupby("horizon_months")
        .agg(
            spearman=("spearman_rank_autocorrelation", "mean"),
            pearson=("pearson_score_autocorrelation", "mean"),
            rank_change=("average_absolute_rank_change", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        summary["horizon_months"],
        summary["spearman"],
        marker="o",
        label="Spearman rank",
    )
    ax.plot(
        summary["horizon_months"],
        summary["pearson"],
        marker="s",
        label="Pearson performance score",
    )
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Rolling Persistence Diagnostics")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "rank_persistence.png")
    plt.close(fig)


def transition_heatmap(matrix: pd.DataFrame, out: Path) -> None:
    """Plot the quintile transition matrix as a heatmap."""

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
        ax=ax,
    )
    ax.set_xlabel("To quintile")
    ax.set_ylabel("From quintile")
    ax.set_title("Quintile Transition Matrix")
    fig.tight_layout()
    fig.savefig(out / "transition_matrix_heatmap.png")
    plt.close(fig)


def mobility_plot(mobility: pd.DataFrame, out: Path) -> None:
    """Plot portfolios with the highest rolling rank volatility."""

    data = mobility.sort_values("rank_volatility", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(data["portfolio"], data["rank_volatility"], color="#7c4d79")
    ax.set_ylabel("Rank volatility")
    ax.set_title("Most Mobile Portfolios by Rolling Rank Volatility")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "mobility_summary.png")
    plt.close(fig)


def alpha_vs_ae_scatter(comparison: pd.DataFrame, out: Path) -> None:
    """Plot traditional alpha ranks against SFA AE ranks."""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(comparison["alpha_rank"], comparison["AE_rank"], color="#2a9d8f", s=55)
    lim = [
        min(comparison["alpha_rank"].min(), comparison["AE_rank"].min()) - 1,
        max(comparison["alpha_rank"].max(), comparison["AE_rank"].max()) + 1,
    ]
    ax.plot(lim, lim, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Alpha rank")
    ax.set_ylabel("AE rank")
    ax.set_title("Alpha Rank vs SFA AE Rank")
    fig.tight_layout()
    fig.savefig(out / "alpha_vs_ae_rank_scatter.png")
    plt.close(fig)


def window_sensitivity_plot(robustness: pd.DataFrame, out: Path) -> None:
    """Plot rolling-window sensitivity correlations."""

    data = robustness.copy()
    data["comparison"] = (
        data["window_left"].astype(str) + " vs " + data["window_right"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(data))
    width = 0.35
    ax.bar(x - width / 2, data["rank_correlation"], width, label="Rank")
    score_col = (
        "score_correlation" if "score_correlation" in data else "AE_correlation"
    )
    ax.bar(x + width / 2, data[score_col], width, label="Performance score")
    ax.set_xticks(x)
    ax.set_xticklabels(data["comparison"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Correlation")
    ax.set_title("Rolling Window Sensitivity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "window_sensitivity.png")
    plt.close(fig)


def residual_diagnostics_plot(residuals: pd.DataFrame, out: Path) -> None:
    """Plot a residual histogram for static SFA fits."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(
        residuals["residual"].dropna(), bins=50, kde=True, color="#4c78a8", ax=ax
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_title("Factor-Model Residual Distribution")
    fig.tight_layout()
    fig.savefig(out / "residual_diagnostics.png")
    plt.close(fig)


def performance_ranking_plot(scores: pd.DataFrame, out: Path) -> None:
    """Plot shrinkage-adjusted annualized alpha with 95% intervals."""

    data = scores.sort_values("posterior_alpha_annualized_bps", ascending=True)
    center = data["posterior_alpha_annualized_bps"].to_numpy(float)
    low = data["posterior_alpha_ci_low"].to_numpy(float) * 12.0 * 10_000.0
    high = data["posterior_alpha_ci_high"].to_numpy(float) * 12.0 * 10_000.0
    errors = np.vstack([center - low, high - center])
    colors = np.where(center >= 0, "#2a9d8f", "#c44e52")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(data["portfolio"], center, color=colors, alpha=0.85)
    ax.errorbar(center, data["portfolio"], xerr=errors, fmt="none", color="black")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Posterior annualized factor alpha (basis points)")
    ax.set_title("Shrinkage-Adjusted Portfolio Performance with 95% Intervals")
    fig.tight_layout()
    fig.savefig(out / "performance_ranking.png")
    plt.close(fig)


def performance_heatmap(scores: pd.DataFrame, out: Path) -> None:
    records = []
    for _, row in scores.iterrows():
        size, bm = _portfolio_grid_position(str(row["portfolio"]))
        if size is not None and bm is not None:
            records.append(
                {
                    "size": size,
                    "book_to_market": bm,
                    "posterior_alpha_bps": row["posterior_alpha_annualized_bps"],
                }
            )
    heat = pd.DataFrame(records).pivot(
        index="size", columns="book_to_market", values="posterior_alpha_bps"
    )
    bound = float(np.nanmax(np.abs(heat.to_numpy(float))))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".0f",
        cmap="vlag",
        center=0,
        vmin=-bound,
        vmax=bound,
        cbar_kws={"label": "Annualized posterior alpha (bps)"},
        ax=ax,
    )
    ax.set_xlabel("Book-to-market quintile")
    ax.set_ylabel("Size quintile")
    ax.set_title("Posterior Factor Alpha Across Size x Book-to-Market Portfolios")
    fig.tight_layout()
    fig.savefig(out / "performance_heatmap_size_bm.png")
    plt.close(fig)


def rolling_performance_plot(rolling: pd.DataFrame, out: Path) -> None:
    data = rolling.copy()
    data["window_end"] = pd.to_datetime(data["window_end"])
    data["score_bps"] = data["posterior_alpha"] * 12.0 * 10_000.0
    summary = (
        data.groupby("window_end")["score_bps"]
        .agg(
            median="median",
            p10=lambda x: x.quantile(0.10),
            p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary["window_end"], summary["median"], color="#1f4e79")
    ax.fill_between(
        summary["window_end"].to_numpy(),
        summary["p10"].to_numpy(float),
        summary["p90"].to_numpy(float),
        color="#6aaed6",
        alpha=0.25,
        label="Cross-sectional 10th-90th percentile",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Annualized posterior alpha (bps)")
    ax.set_xlabel("Training-window end")
    ax.set_title("Rolling Shrinkage-Adjusted Performance")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "rolling_performance_timeseries.png")
    plt.close(fig)


def forward_validation_plot(forward: pd.DataFrame, out: Path) -> None:
    if forward.empty:
        return
    data = forward.sort_values("window_end")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        data["window_end"],
        data["rank_vs_forward_alpha_spearman"],
        color="#2f6f9f",
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_title("Look-Ahead-Free Validation Against Future Factor Alpha")
    axes[1].plot(
        data["window_end"],
        data["top_minus_bottom_forward_alpha_annualized_bps"],
        color="#7c4d79",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Top minus bottom (bps/year)")
    axes[1].set_xlabel("Training-window end")
    fig.tight_layout()
    fig.savefig(out / "forward_performance_validation.png")
    plt.close(fig)


def rank_uncertainty_plot(scores: pd.DataFrame, out: Path) -> None:
    data = scores.sort_values("bootstrap_rank_median", ascending=False)
    center = data["bootstrap_rank_median"].to_numpy(float)
    low = data["bootstrap_rank_ci_low"].to_numpy(float)
    high = data["bootstrap_rank_ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.errorbar(
        center,
        data["portfolio"],
        xerr=np.vstack([center - low, high - center]),
        fmt="o",
        color="#2a9d8f",
        ecolor="#555555",
    )
    ax.invert_xaxis()
    ax.set_xlabel("Bootstrap performance rank (95% interval; 1 is best)")
    ax.set_title("Rank Uncertainty from Common-Date Block Bootstrap")
    fig.tight_layout()
    fig.savefig(out / "rank_uncertainty.png")
    plt.close(fig)


def sfa_boundary_plot(sfa_scores: pd.DataFrame, out: Path) -> None:
    if "boundary_mixture_p_value" not in sfa_scores:
        return
    data = sfa_scores.sort_values("boundary_mixture_p_value", ascending=False)
    p_values = data["boundary_mixture_p_value"].clip(lower=1e-12)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(data["portfolio"], -np.log10(p_values), color="#777777")
    ax.axvline(
        -np.log10(0.05),
        color="#c44e52",
        linestyle="--",
        label="5% boundary test",
    )
    ax.set_xlabel("-log10 boundary-test p-value")
    ax.set_title("Evidence for a One-Sided Residual Component")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "sfa_boundary_diagnostics.png")
    plt.close(fig)


def generate_all_figures(
    *,
    performance_scores: pd.DataFrame,
    rolling_performance: pd.DataFrame,
    persistence: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    mobility: pd.DataFrame,
    robustness: pd.DataFrame,
    forward_validation: pd.DataFrame,
    sfa_diagnostics: pd.DataFrame,
    residuals: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate the standard set of GitHub-readable result figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    performance_ranking_plot(performance_scores, output_dir)
    performance_heatmap(performance_scores, output_dir)
    rolling_performance_plot(rolling_performance, output_dir)
    rank_persistence_plot(persistence, output_dir)
    transition_heatmap(transition_matrix, output_dir)
    mobility_plot(mobility, output_dir)
    window_sensitivity_plot(robustness, output_dir)
    forward_validation_plot(forward_validation, output_dir)
    rank_uncertainty_plot(performance_scores, output_dir)
    sfa_boundary_plot(sfa_diagnostics, output_dir)
    residual_diagnostics_plot(residuals, output_dir)
