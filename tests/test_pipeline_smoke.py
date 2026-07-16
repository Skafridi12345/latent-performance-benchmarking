from __future__ import annotations

import pandas as pd

from analysis.run_all import PipelineConfig, run_pipeline


def test_pipeline_smoke_creates_expected_outputs(synthetic_csv_files, tmp_path):
    port_file, factor_file = synthetic_csv_files
    results_dir = tmp_path / "results"

    summary = run_pipeline(
        PipelineConfig(
            port_file=port_file,
            factor_file=factor_file,
            results_dir=results_dir,
            rolling_window=18,
            rolling_step=18,
            min_obs=12,
            static_maxiter=60,
            hac_lags=3,
            bootstrap_replicates=20,
            bootstrap_block_length=6,
            bootstrap_seed=7,
            forward_months=6,
            persistence_horizons=(18,),
            transition_horizon=18,
            robustness_windows=(12, 18),
            robustness_step=18,
        )
    )

    expected_tables = [
        "dataset_manifest.csv",
        "performance_scores.csv",
        "performance_rank_uncertainty.csv",
        "rolling_performance_scores.csv",
        "forward_performance_validation.csv",
        "forward_performance_aggregate.csv",
        "sfa_asymmetry_diagnostics.csv",
        "rank_persistence.csv",
        "transition_matrix.csv",
        "mobility_summary.csv",
        "robustness_summary.csv",
        "performance_residual_diagnostics.csv",
    ]
    for name in expected_tables:
        assert (results_dir / "tables" / name).exists()

    assert (results_dir / "figures" / "performance_ranking.png").exists()
    assert summary["n_portfolios"] == 5

    transition = pd.read_csv(
        results_dir / "tables" / "transition_matrix.csv", index_col=0
    )
    nonzero_rows = transition.sum(axis=1) > 0
    assert (transition.loc[nonzero_rows].sum(axis=1).round(10) == 1).all()
