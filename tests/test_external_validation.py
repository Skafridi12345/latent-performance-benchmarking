from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from analysis.external_validation import validation_status_summary
from analysis.latent_performance import (
    estimate_cross_sectional_performance,
    hierarchical_shrinkage,
)
from sfa.loaders import design_matrix, load_ff_factors, load_portfolios


def test_original_ken_french_preamble_files_are_parsed():
    portfolios = load_portfolios(
        "data/reference/official/25_Portfolios_5x5.csv"
    )
    factors = load_ff_factors(
        "data/reference/official/F-F_Research_Data_Factors.csv"
    )

    assert portfolios["portfolio"].nunique() == 25
    assert portfolios["date"].min().to_period("M").strftime("%Y%m") == "192607"
    assert factors[["mkt_rf", "smb", "hml", "rf"]].notna().all().all()
    assert factors["date"].max().to_period("M").strftime("%Y%m") == "202605"


def test_statsmodels_reproduces_every_coefficient_and_hac_se(synthetic_dataset):
    scores, _, _, _ = estimate_cross_sectional_performance(
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        factor_model="ff3",
        min_obs=12,
        hac_lags=3,
    )
    scores = scores.set_index("portfolio")
    names = ["alpha", *synthetic_dataset.factor_cols]
    for portfolio, group in synthetic_dataset.data.groupby("portfolio"):
        group = group.sort_values("date")
        X = design_matrix(group, synthetic_dataset.factor_cols)
        fit = sm.OLS(group["excess_return"].to_numpy(float), X).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": 3,
                "kernel": "bartlett",
                "use_correction": True,
            },
            use_t=False,
        )
        for idx, name in enumerate(names):
            assert scores.loc[portfolio, f"coef_{name}"] == pytest.approx(
                fit.params[idx], abs=1e-12
            )
            assert scores.loc[portfolio, f"hac_se_{name}"] == pytest.approx(
                fit.bse[idx], abs=1e-12
            )


def test_joint_shrinkage_rejects_materially_indefinite_covariance():
    estimates = pd.DataFrame(
        {
            "portfolio": ["A", "B"],
            "alpha": [0.001, -0.001],
            "alpha_hac_se": [0.01, 0.01],
            "alpha_p_value": [0.9, 0.9],
        }
    )
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]]) * 1e-4

    with pytest.raises(ValueError, match="materially indefinite"):
        hierarchical_shrinkage(estimates, sampling_covariance=indefinite)


def _status_table(statuses: list[str]) -> pd.DataFrame:
    differences = [0.0 if status == "PASS" else 0.01 for status in statuses]
    return pd.DataFrame(
        {
            "check_id": [f"check_{idx}" for idx in range(len(statuses))],
            "status": statuses,
            "absolute_difference": differences,
            "tolerance": [1e-12] * len(statuses),
            "details": ["Visible validation result."] * len(statuses),
        }
    )


def test_true_fail_blocks_integrity_approval():
    summary = validation_status_summary(_status_table(["PASS", "FAIL"]))

    assert not summary["pipeline_integrity_approved"]
    assert summary["blocking_failures"] == ["check_1"]


def test_open_comparability_statuses_remain_visible_but_nonblocking():
    summary = validation_status_summary(
        _status_table(["PASS", "UNRESOLVED", "NOT_COMPARABLE"])
    )

    assert summary["pipeline_integrity_approved"]
    assert summary["blocking_failures"] == []
    assert summary["open_checks"] == ["check_1", "check_2"]


def test_mismatched_comparison_cannot_be_described_as_pass():
    checks = _status_table(["PASS"])
    checks.loc[0, "absolute_difference"] = 0.01

    with pytest.raises(ValueError, match="PASS checks exceed"):
        validation_status_summary(checks)


def test_unknown_validation_status_fails_closed():
    with pytest.raises(ValueError, match="Unknown validation status"):
        validation_status_summary(_status_table(["UNKNOWN"]))
