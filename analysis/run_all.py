from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from analysis.diagnostics import model_diagnostics, residual_diagnostics
from analysis.figures import generate_all_figures
from analysis.latent_performance import (
    block_bootstrap_rank_uncertainty,
    estimate_cross_sectional_performance,
    forward_performance_validation,
    rolling_cross_sectional_performance,
    summarize_forward_validation,
)
from analysis.mobility import portfolio_mobility_summary
from analysis.persistence import compute_persistence_metrics
from analysis.robustness import model_comparison, performance_window_sensitivity
from analysis.static_sfa import estimate_static_sfa
from analysis.transitions import compute_transition_matrix
from sfa.loaders import build_factor_dataset, dataset_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PORT_FILE = DATA_DIR / "25_size_bm_portfolios.csv"
FF_FILE = DATA_DIR / "ff3_factors.csv"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a reproducible pipeline run."""

    port_file: Path = PORT_FILE
    factor_file: Path = FF_FILE
    results_dir: Path = RESULTS_DIR
    factor_model: str = "ff3"
    rolling_window: int = 120
    rolling_step: int = 12
    min_obs: int = 120
    static_maxiter: int = 500
    hac_lags: int = 12
    bootstrap_replicates: int = 200
    bootstrap_block_length: int = 12
    bootstrap_seed: int = 2026
    forward_months: int = 12
    forward_hac_lags: int = 4
    persistence_horizons: tuple[int, ...] = (12, 60, 120)
    transition_horizon: int = 120
    robustness_windows: tuple[int, ...] = (60, 120, 180)
    robustness_step: int = 120

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"


def _write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a CSV file, creating the parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the research pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the latent performance benchmarking research pipeline."
    )
    parser.add_argument("--portfolio-file", type=Path, default=PORT_FILE)
    parser.add_argument("--factor-file", type=Path, default=FF_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--factor-model",
        default="ff3",
        choices=["capm", "ff3", "ff5", "ff5_mom"],
    )
    parser.add_argument("--rolling-window", type=int, default=120)
    parser.add_argument("--rolling-step", type=int, default=12)
    parser.add_argument("--min-obs", type=int, default=120)
    parser.add_argument("--static-maxiter", type=int, default=500)
    parser.add_argument("--hac-lags", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--bootstrap-block-length", type=int, default=12)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--forward-months", type=int, default=12)
    parser.add_argument("--forward-hac-lags", type=int, default=4)
    parser.add_argument(
        "--persistence-horizons",
        type=int,
        nargs="+",
        default=[12, 60, 120],
    )
    parser.add_argument(
        "--transition-horizon",
        type=int,
        default=120,
        help="Transition horizon in months.",
    )
    parser.add_argument(
        "--robustness-windows",
        type=int,
        nargs="+",
        default=[60, 120, 180],
    )
    parser.add_argument("--robustness-step", type=int, default=120)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a pipeline configuration from parsed CLI arguments."""

    return PipelineConfig(
        port_file=args.portfolio_file,
        factor_file=args.factor_file,
        results_dir=args.results_dir,
        factor_model=args.factor_model,
        rolling_window=args.rolling_window,
        rolling_step=args.rolling_step,
        min_obs=args.min_obs,
        static_maxiter=args.static_maxiter,
        hac_lags=args.hac_lags,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_seed=args.bootstrap_seed,
        forward_months=args.forward_months,
        forward_hac_lags=args.forward_hac_lags,
        persistence_horizons=tuple(args.persistence_horizons),
        transition_horizon=args.transition_horizon,
        robustness_windows=tuple(args.robustness_windows),
        robustness_step=args.robustness_step,
    )


def run_pipeline(config: PipelineConfig) -> dict:
    """Run the full analytical pipeline and return a compact summary."""

    started = time.perf_counter()
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading and aligning factor/portfolio data")

    dataset = build_factor_dataset(
        config.port_file,
        config.factor_file,
        factor_model=config.factor_model,
    )
    df = dataset.data
    _write_csv(df, config.tables_dir / "factor_model_dataset.csv")
    manifest = dataset_manifest(
        dataset,
        portfolio_file=config.port_file,
        factor_file=config.factor_file,
    )
    _write_csv(pd.DataFrame([manifest]), config.tables_dir / "dataset_manifest.csv")

    LOGGER.info("Estimating HAC factor performance and hierarchical shrinkage")
    performance, performance_residuals, prior = estimate_cross_sectional_performance(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        min_obs=config.min_obs,
        hac_lags=config.hac_lags,
    )
    LOGGER.info("Estimating common-date block-bootstrap rank uncertainty")
    bootstrap = block_bootstrap_rank_uncertainty(
        df,
        dataset.factor_cols,
        n_bootstrap=config.bootstrap_replicates,
        block_length=config.bootstrap_block_length,
        hac_lags=config.hac_lags,
        random_seed=config.bootstrap_seed,
    )
    performance = performance.merge(bootstrap, on="portfolio", how="left")
    _write_csv(performance, config.tables_dir / "performance_scores.csv")
    _write_csv(performance_residuals, config.tables_dir / "performance_residuals.csv")
    _write_csv(pd.DataFrame([prior]), config.tables_dir / "performance_prior.csv")
    _write_csv(bootstrap, config.tables_dir / "performance_rank_uncertainty.csv")

    ranking_comparison = performance[
        ["portfolio", "alpha", "posterior_alpha", "performance_rank"]
    ].copy()
    ranking_comparison["raw_alpha_rank"] = ranking_comparison["alpha"].rank(
        ascending=False, method="first"
    ).astype(int)
    ranking_comparison["rank_change_after_shrinkage"] = (
        ranking_comparison["raw_alpha_rank"] - ranking_comparison["performance_rank"]
    )
    _write_csv(
        ranking_comparison.sort_values("performance_rank"),
        config.tables_dir / "raw_vs_shrunk_alpha_ranks.csv",
    )

    LOGGER.info(
        "Estimating rolling primary performance: window=%s, step=%s",
        config.rolling_window,
        config.rolling_step,
    )
    rolling = rolling_cross_sectional_performance(
        df,
        dataset.factor_cols,
        window=config.rolling_window,
        step=config.rolling_step,
        factor_model=dataset.factor_model,
        min_obs=config.rolling_window,
        hac_lags=config.hac_lags,
    )
    _write_csv(rolling, config.tables_dir / "rolling_performance_scores.csv")

    LOGGER.info("Computing overlap-labelled persistence, transitions, and mobility")
    persistence = compute_persistence_metrics(
        rolling,
        horizons_months=config.persistence_horizons,
        score_col="posterior_alpha",
    )
    _write_csv(persistence, config.tables_dir / "rank_persistence.csv")

    transition_matrix, transition_summary = compute_transition_matrix(
        rolling,
        horizon_months=config.transition_horizon,
    )
    _write_csv(
        transition_matrix, config.tables_dir / "transition_matrix.csv", index=True
    )
    _write_csv(transition_summary, config.tables_dir / "transition_summary.csv")

    mobility = portfolio_mobility_summary(
        rolling, horizon_months=config.transition_horizon
    )
    _write_csv(mobility, config.tables_dir / "mobility_summary.csv")

    forward_observations, forward_summary = forward_performance_validation(
        rolling,
        df,
        dataset.factor_cols,
        forward_months=config.forward_months,
    )
    _write_csv(
        forward_observations,
        config.tables_dir / "forward_performance_observations.csv",
    )
    _write_csv(
        forward_summary,
        config.tables_dir / "forward_performance_validation.csv",
    )
    forward_aggregate = summarize_forward_validation(
        forward_summary,
        hac_lags=config.forward_hac_lags,
    )
    _write_csv(
        forward_aggregate,
        config.tables_dir / "forward_performance_aggregate.csv",
    )

    LOGGER.info("Running date-stratified rolling-window sensitivity checks")
    robustness_summary, robustness_scores = performance_window_sensitivity(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        windows=config.robustness_windows,
        step=config.robustness_step,
        hac_lags=config.hac_lags,
    )
    _write_csv(robustness_summary, config.tables_dir / "robustness_summary.csv")
    _write_csv(
        robustness_scores,
        config.tables_dir / "rolling_window_sensitivity_scores.csv",
    )

    LOGGER.info("Estimating half-normal residual-asymmetry diagnostic")
    static_scores, static_timeseries = estimate_static_sfa(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        model_type="half_normal",
        min_obs=config.min_obs,
        maxiter=config.static_maxiter,
    )
    _write_csv(static_scores, config.tables_dir / "sfa_asymmetry_diagnostics.csv")
    _write_csv(
        static_timeseries, config.tables_dir / "sfa_observation_diagnostics.csv"
    )

    LOGGER.info("Estimating truncated-normal sensitivity diagnostic")
    truncated_scores, _ = estimate_static_sfa(
        df,
        dataset.factor_cols,
        factor_model=dataset.factor_model,
        model_type="truncated_normal",
        min_obs=config.min_obs,
        maxiter=config.static_maxiter,
    )
    comparison = model_comparison(static_scores, truncated_scores)
    _write_csv(comparison, config.tables_dir / "sfa_distribution_sensitivity.csv")

    sfa_diagnostics = model_diagnostics(
        static_scores,
        static_timeseries,
    )
    performance_diagnostics = residual_diagnostics(performance_residuals)
    _write_csv(
        sfa_diagnostics, config.tables_dir / "sfa_model_diagnostics.csv"
    )
    _write_csv(
        performance_diagnostics,
        config.tables_dir / "performance_residual_diagnostics.csv",
    )

    LOGGER.info("Generating figures")
    generate_all_figures(
        performance_scores=performance,
        rolling_performance=rolling,
        persistence=persistence,
        transition_matrix=transition_matrix,
        mobility=mobility,
        robustness=robustness_summary,
        forward_validation=forward_summary,
        sfa_diagnostics=static_scores,
        residuals=performance_residuals,
        output_dir=config.figures_dir,
    )

    elapsed = time.perf_counter() - started
    n_estimates = int(rolling[["portfolio", "window_end"]].drop_duplicates().shape[0])
    n_windows = int(rolling["window_end"].nunique())
    summary = {
        "n_portfolios": dataset.n_portfolios,
        "sample_start": dataset.sample_start.date().isoformat(),
        "sample_end": dataset.sample_end.date().isoformat(),
        "factor_model": dataset.factor_model,
        "factor_cols": ", ".join(dataset.factor_cols),
        "sfa_diagnostic": "half-normal with truncated-normal sensitivity",
        "primary_model": "HAC factor alpha with empirical-Bayes shrinkage",
        "hac_lags": config.hac_lags,
        "bootstrap_replicates": config.bootstrap_replicates,
        "forward_months": config.forward_months,
        "forward_hac_lags": config.forward_hac_lags,
        "rolling_window": config.rolling_window,
        "rolling_step": config.rolling_step,
        "n_rolling_windows": n_windows,
        "n_rolling_estimates": n_estimates,
        "tables_dir": str(config.tables_dir),
        "figures_dir": str(config.figures_dir),
        "runtime_seconds": elapsed,
    }
    _write_json(
        {
            "dataset": manifest,
            "configuration": asdict(config),
            "summary": summary,
        },
        config.results_dir / "run_manifest.json",
    )
    LOGGER.info("Pipeline complete in %.2f seconds", elapsed)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for the full research pipeline."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    summary = run_pipeline(config_from_args(args))

    print("\nLatent Performance Benchmarking pipeline complete")
    print("------------------------------------------------")
    print(f"Portfolios: {summary['n_portfolios']}")
    print(f"Sample period: {summary['sample_start']} to {summary['sample_end']}")
    print(f"Factor model: {summary['factor_model']} ({summary['factor_cols']})")
    print(f"SFA diagnostic: {summary['sfa_diagnostic']}")
    print(f"Rolling window: {summary['rolling_window']} months")
    print(f"Rolling step: {summary['rolling_step']} month(s)")
    print(f"Rolling portfolio-window estimates: {summary['n_rolling_estimates']}")
    print(f"Tables: {summary['tables_dir']}")
    print(f"Figures: {summary['figures_dir']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
