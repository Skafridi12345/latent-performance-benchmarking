from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.static_sfa import apply_sfa_multiple_testing


def _scores(p_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio": [f"P{idx}" for idx in range(len(p_values))],
            "AE": np.linspace(0.8, 0.99, len(p_values)),
            "converged": True,
            "sfa_boundary_p_value": p_values,
        }
    )


def test_sfa_current_approximate_pattern_can_lose_nominal_discoveries():
    result = apply_sfa_multiple_testing(_scores([0.001, 0.03, *([0.6] * 23)]))

    assert result["sfa_supported_nominal_5pct"].sum() == 2
    assert result["sfa_supported_fdr_5pct"].sum() == 1
    assert result["sfa_rank"].notna().sum() == 1


def test_sfa_fdr_handles_none_several_ties_and_input_order():
    none = apply_sfa_multiple_testing(_scores([0.2] * 25))
    assert none["sfa_supported_fdr_5pct"].sum() == 0
    assert none["sfa_rank"].isna().all()

    several = apply_sfa_multiple_testing(_scores([0.001, 0.002, 0.003, *([0.8] * 22)]))
    assert several["sfa_supported_fdr_5pct"].sum() == 3
    assert several["sfa_rank"].notna().sum() == 3

    tied = _scores([0.01, 0.01, 0.9])
    tied["portfolio"] = ["C", "A", "B"]
    result = apply_sfa_multiple_testing(tied)
    assert result["portfolio"].tolist() == ["C", "A", "B"]
    assert (
        result.loc[0, "sfa_boundary_q_value"] == result.loc[1, "sfa_boundary_q_value"]
    )


def test_sfa_invalid_p_values_remain_missing_and_unranked():
    result = apply_sfa_multiple_testing(_scores([0.01, np.nan, -0.1, 1.1]))

    assert result.loc[1:, "sfa_boundary_q_value"].isna().all()
    assert result.loc[1:, "sfa_supported_nominal_5pct"].isna().all()
    assert result.loc[1:, "sfa_supported_fdr_5pct"].isna().all()
    assert result.loc[1:, "sfa_rank"].isna().all()
