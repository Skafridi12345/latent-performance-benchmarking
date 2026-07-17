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
from analysis.external_validation import (
    asset_pricing_patterns,
    external_validation_checks,
    ff5_model_sensitivity,
    historical_anchor_analysis,
    official_reference_manifest,
    validation_status_summary,
)
from analysis.figures import generate_all_figures
from analysis.incremental_validation import ALL_SCHEMES, run_incremental_test
from analysis.latent_performance import (
    block_bootstrap_rank_uncertainty,
    estimate_cross_sectional_performance,
    forward_performance_validation,
    forward_validation_robustness,
    joint_alpha_tests,
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
REFERENCE_DIR = DATA_DIR / "reference" / "official"
LOGGER = logging.getLogger(__name__)

# The incremental forward test needs enough non-overlapping forward windows for
# HAC inference to be meaningful; below this it is skipped (e.g. smoke tests).
MIN_INCREMENTAL_WINDOWS = 10


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
    bootstrap_replicates: int = 5_000
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
        description="Run the uncertainty-aware portfolio benchmarking pipeline."
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
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
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
    performance, performance_residuals, prior, joint_hac_covariance = (
        estimate_cross_sectional_performance(
            df,
            dataset.factor_cols,
            factor_model=dataset.factor_model,
            min_obs=config.min_obs,
            hac_lags=config.hac_lags,
        )
    )
    LOGGER.info("Estimating common-date block-bootstrap rank uncertainty")
    bootstrap, bootstrap_stability = block_bootstrap_rank_uncertainty(
        df,
        dataset.factor_cols,
        n_bootstrap=config.bootstrap_replicates,
        block_length=config.bootstrap_block_length,
        hac_lags=config.hac_lags,
        random_seed=config.bootstrap_seed,
        stability_checkpoint=min(1_000, config.bootstrap_replicates),
        return_stability=True,
    )
    performance = performance.merge(bootstrap, on="portfolio", how="left")
    _write_csv(performance, config.tables_dir / "performance_scores.csv")
    _write_csv(performance_residuals, config.tables_dir / "performance_residuals.csv")
    posterior_covariance = prior.pop("posterior_covariance")
    shrinkage_matrix = prior.pop("shrinkage_matrix")
    matrix_order = prior.pop("matrix_portfolio_order")
    _write_csv(pd.DataFrame([prior]), config.tables_dir / "performance_prior.csv")
    _write_csv(bootstrap, config.tables_dir / "performance_rank_uncertainty.csv")
    _write_csv(
        bootstrap_stability,
        config.tables_dir / "bootstrap_stability_1000_vs_final.csv",
    )
    _write_csv(
        joint_hac_covariance,
        config.tables_dir / "joint_alpha_hac_covariance.csv",
        index=True,
    )
    _write_csv(
        pd.DataFrame(posterior_covariance, index=matrix_order, columns=matrix_order),
        config.tables_dir / "posterior_alpha_covariance.csv",
        index=True,
    )
    _write_csv(
        pd.DataFrame(shrinkage_matrix, index=matrix_order, columns=matrix_order),
        config.tables_dir / "empirical_bayes_shrinkage_matrix.csv",
        index=True,
    )
    joint_order = joint_hac_covariance.columns.tolist()
    joint_alpha = (
        performance.set_index("portfolio").loc[joint_order, "alpha"].to_numpy(float)
    )
    residual_matrix = (
        performance_residuals.pivot(
            index="date", columns="portfolio", values="residual"
        )
        .reindex(columns=joint_order)
        .to_numpy(float)
    )
    first_portfolio = df[df["portfolio"] == joint_order[0]].sort_values("date")
    joint_tests = joint_alpha_tests(
        joint_alpha,
        residual_matrix,
        first_portfolio[dataset.factor_cols].to_numpy(float),
        joint_hac_covariance.to_numpy(float),
    )
    _write_csv(joint_tests, config.tables_dir / "joint_alpha_tests.csv")

    canonical_inputs = (
        config.port_file.resolve() == PORT_FILE.resolve()
        and config.factor_file.resolve() == FF_FILE.resolve()
    )
    external_validation_manifest: dict = {
        "run": False,
        "reason": "non-canonical input paths",
    }
    if canonical_inputs:
        LOGGER.info("Running official-source and independent replication checks")
        reference_manifest = official_reference_manifest(REFERENCE_DIR)
        external_checks = external_validation_checks(
            dataset,
            performance,
            local_portfolio_file=config.port_file,
            local_factor_file=config.factor_file,
            reference_dir=REFERENCE_DIR,
            hac_lags=config.hac_lags,
        )
        validation_summary = validation_status_summary(external_checks)
        patterns = asset_pricing_patterns(dataset, performance)
        historical_scores, historical_tests = historical_anchor_analysis(
            dataset, hac_lags=config.hac_lags
        )
        ff_scores, ff_comparison, ff_joint_tests = ff5_model_sensitivity(
            dataset,
            local_portfolio_file=config.port_file,
            official_ff5_file=(REFERENCE_DIR / "F-F_Research_Data_5_Factors_2x3.csv"),
            hac_lags=config.hac_lags,
        )
        _write_csv(
            reference_manifest,
            config.tables_dir / "official_reference_manifest.csv",
        )
        _write_csv(
            external_checks,
            config.tables_dir / "external_validation_checks.csv",
        )
        _write_csv(patterns, config.tables_dir / "asset_pricing_pattern_checks.csv")
        _write_csv(
            historical_scores,
            config.tables_dir / "historical_anchor_1963_07_to_1991_12.csv",
        )
        _write_csv(
            historical_tests,
            config.tables_dir / "historical_anchor_joint_alpha_tests.csv",
        )
        _write_csv(
            ff_scores,
            config.tables_dir / "ff3_ff5_common_sample_scores.csv",
        )
        _write_csv(
            ff_comparison,
            config.tables_dir / "ff3_ff5_sensitivity_comparison.csv",
        )
        _write_csv(
            ff_joint_tests,
            config.tables_dir / "ff3_ff5_joint_alpha_tests.csv",
        )
        external_validation_manifest = {
            "run": True,
            "official_sources": reference_manifest.to_dict(orient="records"),
            **validation_summary,
            "unresolved_or_noncomparable_checks": validation_summary["open_checks"],
        }

    ranking_comparison = performance[
        ["portfolio", "alpha", "posterior_alpha", "performance_rank"]
    ].copy()
    ranking_comparison["raw_alpha_rank"] = (
        ranking_comparison["alpha"].rank(ascending=False, method="first").astype(int)
    )
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
    forward_robustness, forward_extremes = forward_validation_robustness(
        forward_summary
    )
    _write_csv(
        forward_robustness,
        config.tables_dir / "forward_performance_robustness.csv",
    )
    _write_csv(
        forward_extremes,
        config.tables_dir / "forward_performance_extreme_windows.csv",
    )

    n_forward_windows = int(forward_observations["window_end"].nunique())
    if n_forward_windows >= MIN_INCREMENTAL_WINDOWS:
        LOGGER.info(
            "Running incremental rolling-vs-benchmark forward test (%d benchmarks)",
            len(ALL_SCHEMES),
        )
        for scheme in ALL_SCHEMES:
            run_incremental_test(
                config.port_file,
                config.factor_file,
                factor_model=dataset.factor_model,
                scheme=scheme,
                rolling_window=config.rolling_window,
                rolling_step=config.rolling_step,
                min_obs=config.min_obs,
                hac_lags=config.hac_lags,
                forward_months=config.forward_months,
                bootstrap_replicates=config.bootstrap_replicates,
                bootstrap_seed=config.bootstrap_seed,
                output_dir=config.results_dir,
                verbose=False,
            )
    else:
        LOGGER.warning(
            "Skipping incremental forward test: only %d forward windows "
            "(need >= %d for meaningful HAC inference).",
            n_forward_windows,
            MIN_INCREMENTAL_WINDOWS,
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
    _write_csv(static_timeseries, config.tables_dir / "sfa_observation_diagnostics.csv")

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
    _write_csv(sfa_diagnostics, config.tables_dir / "sfa_model_diagnostics.csv")
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
        transition_summary=transition_summary,
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
        "primary_model": (
            "full joint-HAC factor alpha with multivariate empirical-Bayes shrinkage"
        ),
        "hac_lags": config.hac_lags,
        "bootstrap_replicates": config.bootstrap_replicates,
        "forward_months": config.forward_months,
        "forward_hac_lags": config.forward_hac_lags,
        "rolling_window": config.rolling_window,
        "rolling_step": config.rolling_step,
        "n_rolling_windows": n_windows,
        "n_rolling_estimates": n_estimates,
        "primary_joint_alpha_test": "HAC_Wald",
        "primary_joint_alpha_p_value": float(
            joint_tests.set_index("test").loc["HAC_Wald", "p_value"]
        ),
        "tables_dir": str(config.tables_dir),
        "figures_dir": str(config.figures_dir),
        "runtime_seconds": elapsed,
    }
    _write_json(
        {
            "dataset": manifest,
            "configuration": asdict(config),
            "bootstrap_stability_1000_vs_final": {
                column: float(bootstrap_stability[column].max())
                for column in bootstrap_stability.columns
                if column.startswith("absolute_change_")
            },
            "external_validation": external_validation_manifest,
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

    print("\nUncertainty-Aware Portfolio Benchmarking pipeline complete")
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
