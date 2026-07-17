from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.latent_performance import (
    _profile_prior,
    benjamini_hochberg,
    hierarchical_shrinkage,
    joint_alpha_tests,
    joint_newey_west_alpha_covariance,
)


def test_bh_matches_hand_calculation_and_preserves_input_order():
    p_values = np.array([0.04, 0.001, 0.03, 0.20, 0.01])
    q_values = benjamini_hochberg(p_values)

    assert q_values == pytest.approx([0.05, 0.005, 0.05, 0.20, 0.025])


def test_bh_handles_none_several_ties_and_invalid_values():
    assert np.all(benjamini_hochberg(np.array([0.4, 0.7, 0.9])) > 0.05)
    tied = benjamini_hochberg(np.array([0.01, 0.01, 0.9, np.nan, -0.1, 1.1]))
    assert tied[:2] == pytest.approx([0.015, 0.015])
    assert tied[2] == pytest.approx(0.9)
    assert np.isnan(tied[3:]).all()
    several = benjamini_hochberg(np.array([0.001, 0.002, 0.003, 0.9]))
    assert np.sum(several < 0.05) == 3


def test_joint_hac_diagonal_agrees_with_scalar_sandwich():
    rng = np.random.default_rng(20)
    n = 360
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    residuals = rng.normal(size=(n, 4)) @ np.diag([0.01, 0.02, 0.03, 0.04])
    covariance, metadata = joint_newey_west_alpha_covariance(X, residuals, max_lags=6)

    from analysis.latent_performance import newey_west_covariance

    scalar = np.array(
        [newey_west_covariance(X, residuals[:, j], max_lags=6)[0, 0] for j in range(4)]
    )
    assert np.diag(covariance) == pytest.approx(scalar, rel=1e-9, abs=1e-15)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0
    assert metadata["condition_number"] <= 1e12 * (1 + 1e-12)


def test_joint_hac_retains_cross_portfolio_dependence_and_order():
    rng = np.random.default_rng(21)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    common = rng.normal(0, 0.02, n)
    residuals = np.column_stack(
        [common + rng.normal(0, 0.003, n), -common + rng.normal(0, 0.003, n)]
    )
    covariance, _ = joint_newey_west_alpha_covariance(X, residuals, max_lags=4)

    assert covariance[0, 1] < 0
    assert abs(covariance[0, 1]) > 0.5 * np.sqrt(covariance[0, 0] * covariance[1, 1])


def test_joint_empirical_bayes_matches_diagonal_formula_and_shrinks():
    estimates = pd.DataFrame(
        {
            "portfolio": ["A", "B", "C", "D"],
            "alpha": [-0.01, -0.001, 0.003, 0.02],
            "alpha_hac_se": [0.02, 0.004, 0.005, 0.025],
            "alpha_p_value": [0.6, 0.8, 0.5, 0.4],
        }
    )
    default, default_prior = hierarchical_shrinkage(estimates)
    explicit, explicit_prior = hierarchical_shrinkage(
        estimates,
        sampling_covariance=np.diag(estimates["alpha_hac_se"].to_numpy() ** 2),
    )

    pd.testing.assert_series_equal(
        default["posterior_alpha"], explicit["posterior_alpha"], check_names=False
    )
    assert default_prior["posterior_covariance"] == pytest.approx(
        explicit_prior["posterior_covariance"]
    )
    scalar_prior = _profile_prior(
        estimates["alpha"].to_numpy(float),
        estimates["alpha_hac_se"].to_numpy(float),
    )
    assert explicit_prior["mu"] == pytest.approx(scalar_prior["mu"], rel=1e-7)
    assert explicit_prior["tau2"] == pytest.approx(scalar_prior["tau2"], rel=1e-7)
    weights = scalar_prior["tau2"] / (
        scalar_prior["tau2"] + estimates["alpha_hac_se"].to_numpy(float) ** 2
    )
    expected = weights * estimates["alpha"].to_numpy(float) + (
        1.0 - weights
    ) * scalar_prior["mu"]
    actual = explicit.set_index("portfolio").loc[
        estimates["portfolio"], "posterior_alpha"
    ]
    assert actual.to_numpy(float) == pytest.approx(expected, rel=1e-7)
    assert np.min(np.linalg.eigvalsh(default_prior["posterior_covariance"])) >= -1e-14
    merged = default.set_index("portfolio")
    assert abs(merged.loc["D", "posterior_alpha"] - default_prior["mu"]) < abs(
        merged.loc["D", "alpha"] - default_prior["mu"]
    )


def test_joint_alpha_tests_are_reproducible_and_detect_nonzero_alpha():
    rng = np.random.default_rng(22)
    t_obs, n_assets, n_factors = 600, 5, 3
    factors = rng.normal(0, [0.04, 0.02, 0.02], size=(t_obs, n_factors))
    residuals = rng.normal(0, 0.015, size=(t_obs, n_assets))
    X = np.column_stack([np.ones(t_obs), factors])
    covariance, _ = joint_newey_west_alpha_covariance(X, residuals, max_lags=6)
    alpha = np.full(n_assets, 0.003)
    first = joint_alpha_tests(alpha, residuals, factors, covariance)
    second = joint_alpha_tests(alpha, residuals, factors, covariance)

    pd.testing.assert_frame_equal(first, second)
    assert first["p_value"].between(0, 1).all()
    assert first["p_value"].max() < 0.05
    assert first.set_index("test").loc["HAC_Wald", "role"] == "primary_joint_test"
    residual_covariance = residuals.T @ residuals / t_obs
    centered_factors = factors - factors.mean(axis=0)
    factor_covariance = centered_factors.T @ centered_factors / t_obs
    grs_expected = (
        (t_obs - n_assets - n_factors)
        / n_assets
        * (alpha @ np.linalg.inv(residual_covariance) @ alpha)
        / (
            1
            + factors.mean(axis=0)
            @ np.linalg.inv(factor_covariance)
            @ factors.mean(axis=0)
        )
    )
    indexed = first.set_index("test")
    assert indexed.loc["GRS", "statistic"] == pytest.approx(grs_expected)
    assert indexed.loc["HAC_Wald", "statistic"] == pytest.approx(
        alpha @ np.linalg.inv(covariance) @ alpha
    )


def test_joint_alpha_tests_have_exact_null_behavior_at_zero_alpha():
    rng = np.random.default_rng(23)
    t_obs, n_assets, n_factors = 300, 4, 2
    factors = rng.normal(size=(t_obs, n_factors))
    residuals = rng.normal(size=(t_obs, n_assets))
    X = np.column_stack([np.ones(t_obs), factors])
    covariance, _ = joint_newey_west_alpha_covariance(X, residuals, max_lags=3)
    result = joint_alpha_tests(np.zeros(n_assets), residuals, factors, covariance)

    assert np.allclose(result["statistic"], 0.0)
    assert np.allclose(result["p_value"], 1.0)
