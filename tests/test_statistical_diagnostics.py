from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.diagnostics import residual_diagnostics
from sfa.sfa_halfnormal import (
    HalfNormalSFA,
    _truncated_normal_exp_moment,
)


def test_half_normal_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(42)
    n = 300
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = X @ np.array([0.01, 0.02, -0.03]) + rng.normal(0, 0.05, n)
    model = HalfNormalSFA(y, X)
    theta = model._ols_start()
    theta[-2:] += [0.2, -0.1]

    analytic = model._neg_loglik_grad(theta)
    numeric = np.zeros_like(theta)
    step = 1e-6
    for idx in range(len(theta)):
        upper = theta.copy()
        lower = theta.copy()
        upper[idx] += step
        lower[idx] -= step
        numeric[idx] = (
            model._neg_loglik(upper) - model._neg_loglik(lower)
        ) / (2 * step)

    assert np.allclose(analytic, numeric, atol=2e-5, rtol=2e-5)


def test_truncated_normal_exponential_moment_matches_simulation():
    rng = np.random.default_rng(11)
    mu = 0.4
    sigma = 0.7
    rate = 0.8
    draws = rng.normal(mu, sigma, 1_000_000)
    draws = draws[draws >= 0]
    simulated = float(np.mean(np.exp(-rate * draws)))
    analytic = float(
        _truncated_normal_exp_moment(
            np.array([mu]), sigma, rate=rate
        )[0]
    )

    assert analytic == pytest.approx(simulated, abs=8e-4)


def test_boundary_test_suppresses_unsupported_one_sided_component():
    rng = np.random.default_rng(7)
    n = 800
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    beta = np.array([0.001, 0.4, -0.2])
    symmetric = X @ beta + rng.normal(0, 0.02, n)
    fit = HalfNormalSFA(symmetric, X).fit(maxiter=500)

    assert not fit.one_sided_component_supported
    assert fit.sigma_u == 0
    assert np.all(fit.AE == 1)


def test_residual_diagnostics_include_serial_dependence_and_arch_tests():
    rng = np.random.default_rng(8)
    n = 240
    residual = np.zeros(n)
    for idx in range(1, n):
        residual[idx] = 0.5 * residual[idx - 1] + rng.normal(0, 0.02)
    frame = pd.DataFrame(
        {
            "portfolio": "P1",
            "factor_model": "ff3",
            "date": pd.date_range("2000-01-31", periods=n, freq="ME"),
            "residual": residual,
        }
    )
    result = residual_diagnostics(frame).iloc[0]

    assert 0 <= result["ljung_box_lag_12_p_value"] <= 1
    assert 0 <= result["arch_lm_lag_12_p_value"] <= 1
    assert result["durbin_watson"] < 2
