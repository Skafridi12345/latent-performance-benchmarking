from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.latent_performance import (
    block_bootstrap_rank_uncertainty,
    estimate_hac_factor_model,
    forward_performance_validation,
    hierarchical_shrinkage,
    rolling_cross_sectional_performance,
    summarize_forward_validation,
)
from analysis.persistence import compute_persistence_metrics
from sfa.loaders import design_matrix, validate_factor_panel
from sfa.sfa_halfnormal import HalfNormalSFA


def test_factor_panel_validation_rejects_join_explosion(synthetic_dataset):
    panel = synthetic_dataset.data
    report = validate_factor_panel(panel, synthetic_dataset.factor_cols)

    assert report["is_valid"]
    assert report["duplicate_portfolio_months"] == 0
    assert report["n_rows"] == report["n_unique_portfolio_months"]

    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate portfolio-month"):
        validate_factor_panel(duplicated, synthetic_dataset.factor_cols)


def test_hac_factor_model_recovers_alpha_and_reports_uncertainty():
    rng = np.random.default_rng(2026)
    n = 720
    factors = rng.normal(0.0, [0.04, 0.02, 0.02], size=(n, 3))
    innovation = rng.normal(0.0, 0.012, n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.45 * noise[t - 1] + innovation[t]

    alpha = 0.002
    y = alpha + factors @ np.array([1.0, 0.3, -0.2]) + noise
    X = np.column_stack([np.ones(n), factors])
    result = estimate_hac_factor_model(y, X, hac_lags=12)

    assert result["alpha"] == pytest.approx(alpha, abs=0.002)
    assert result["alpha_hac_se"] > 0
    assert 0 <= result["alpha_p_value"] <= 1
    assert result["hac_lags"] == 12


def test_hierarchical_shrinkage_reduces_extreme_noisy_estimates():
    estimates = pd.DataFrame(
        {
            "portfolio": ["A", "B", "C", "D", "E"],
            "alpha": [-0.010, -0.001, 0.000, 0.002, 0.020],
            "alpha_hac_se": [0.020, 0.003, 0.003, 0.004, 0.025],
            "alpha_p_value": [0.62, 0.74, 1.0, 0.62, 0.42],
        }
    )

    out, prior = hierarchical_shrinkage(estimates)

    assert prior["tau"] >= 0
    assert set(out["performance_rank"]) == set(range(1, 6))
    assert out.loc[out["portfolio"] == "E", "posterior_alpha"].iat[0] < 0.020
    assert out.loc[out["portfolio"] == "A", "posterior_alpha"].iat[0] > -0.010
    assert out["posterior_alpha_ci_low"].le(out["posterior_alpha"]).all()
    assert out["posterior_alpha_ci_high"].ge(out["posterior_alpha"]).all()
    assert out["alpha_fdr_q_value"].between(0, 1).all()


def test_sfa_adjusted_efficiency_is_unit_invariant(synthetic_dataset):
    group = synthetic_dataset.data.query("portfolio == 'SMALL LoBM'")
    y = group["excess_return"].to_numpy(float)
    X = design_matrix(group, synthetic_dataset.factor_cols)

    decimal_fit = HalfNormalSFA(y, X).fit(maxiter=200)
    percent_X = X.copy()
    percent_X[:, 1:] *= 100.0
    percent_fit = HalfNormalSFA(y * 100.0, percent_X).fit(maxiter=200)

    assert np.allclose(decimal_fit.AE, percent_fit.AE, atol=2e-5, rtol=2e-5)
    assert np.allclose(
        decimal_fit.u_hat_standardized,
        percent_fit.u_hat_standardized,
        atol=2e-5,
        rtol=2e-5,
    )


def test_sfa_constant_shift_is_absorbed_by_alpha_not_called_performance(
    synthetic_dataset,
):
    group = synthetic_dataset.data.query("portfolio == 'SMALL LoBM'")
    y = group["excess_return"].to_numpy(float)
    X = design_matrix(group, synthetic_dataset.factor_cols)

    base = HalfNormalSFA(y, X).fit(maxiter=200)
    shifted = HalfNormalSFA(y - 0.01, X).fit(maxiter=200)

    assert shifted.alpha - base.alpha == pytest.approx(-0.01, abs=2e-5)
    assert np.allclose(base.AE, shifted.AE, atol=2e-5, rtol=2e-5)


def test_persistence_reports_window_overlap_and_inference_eligibility():
    rows = []
    for date_idx, date in enumerate(
        [
            pd.Timestamp("2000-12-31"),
            pd.Timestamp("2001-12-31"),
            pd.Timestamp("2010-12-31"),
        ]
    ):
        for q in range(1, 6):
            rows.append(
                {
                    "portfolio": f"P{q}",
                    "window_end": date,
                    "window_length": 120,
                    "AE": q / 10 + date_idx * 0.001,
                    "rank": 6 - q,
                    "quintile": q,
                }
            )

    result = compute_persistence_metrics(
        pd.DataFrame(rows), horizons_months=[12, 120]
    )

    twelve = result.query("horizon_months == 12").iloc[0]
    one_twenty = result.query("horizon_months == 120").iloc[0]
    assert twelve["window_overlap_fraction"] == pytest.approx(0.90)
    assert not bool(twelve["structural_inference_eligible"])
    assert one_twenty["window_overlap_fraction"] == pytest.approx(0.0)
    assert bool(one_twenty["structural_inference_eligible"])


def test_forward_validation_uses_only_dates_after_training_window(synthetic_dataset):
    rolling = rolling_cross_sectional_performance(
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        factor_model=synthetic_dataset.factor_model,
        window=18,
        step=6,
        min_obs=18,
        hac_lags=3,
    )
    observations, summary = forward_performance_validation(
        rolling,
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        forward_months=6,
    )

    assert not observations.empty
    assert not summary.empty
    assert (observations["forward_start"] > observations["window_end"]).all()
    assert (observations["forward_end"] > observations["forward_start"]).all()
    assert summary["look_ahead_free"].all()
    assert summary["n_portfolios"].eq(5).all()
    aggregate = summarize_forward_validation(summary, hac_lags=1)
    assert bool(aggregate.loc[0, "all_look_ahead_free"])
    assert aggregate.loc[0, "n_validation_windows"] == len(summary)


def test_block_bootstrap_reports_rank_uncertainty_reproducibly(synthetic_dataset):
    first = block_bootstrap_rank_uncertainty(
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        n_bootstrap=20,
        block_length=6,
        hac_lags=3,
        random_seed=99,
    )
    second = block_bootstrap_rank_uncertainty(
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        n_bootstrap=20,
        block_length=6,
        hac_lags=3,
        random_seed=99,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5
    assert first["bootstrap_replicates"].eq(20).all()
    assert first["bootstrap_rank_ci_low"].le(first["bootstrap_rank_median"]).all()
    assert first["bootstrap_rank_ci_high"].ge(first["bootstrap_rank_median"]).all()
    assert first["bootstrap_top_quintile_probability"].between(0, 1).all()
