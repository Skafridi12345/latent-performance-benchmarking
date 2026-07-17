from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reports.generate_report import (
    format_p_value,
    maximum_rank_mover_fact,
    posterior_positive_fact,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def test_p_value_presentation_never_displays_zero():
    assert format_p_value(0.0) == "<0.0001"
    assert format_p_value(np.nextafter(0.0, 1.0), scientific=True) != "0.0000"
    assert format_p_value(0.00001234) == "<0.0001"
    assert format_p_value(0.00001234, scientific=True) == "1.23e-05"
    assert format_p_value(0.123456) == "0.1235"
    assert format_p_value(None) == "NA"
    assert format_p_value(np.nan) == "NA"


@pytest.mark.parametrize("value", [-0.1, 1.1, "invalid"])
def test_invalid_p_values_fail_closed(value):
    with pytest.raises((TypeError, ValueError)):
        format_p_value(value)


def test_posterior_positive_fact_agrees_across_report_surfaces():
    performance = pd.read_csv(TABLES / "performance_scores.csv")
    count, names, sentence = posterior_positive_fact(performance)

    assert count == 2
    assert names == ["SMALL HiBM", "BIG LoBM"]
    source = (ROOT / "reports" / "generate_report.py").read_text()
    summary = (ROOT / "reports" / "TECHNICAL_REPORT.md").read_text()
    readme = (ROOT / "README.md").read_text()
    assert source.count("positive_interval_sentence") >= 3
    assert f"{count} posterior intervals are wholly above zero" in summary
    assert all(name in summary for name in names)
    assert all(name in readme for name in names)
    assert sentence.startswith(f"{count} posterior intervals")
    assert "only one approximate posterior interval" not in source + summary + readme


def test_maximum_ff3_ff5_mover_agrees_across_report_surfaces():
    comparison = pd.read_csv(TABLES / "ff3_ff5_sensitivity_comparison.csv")
    row, sentence = maximum_rank_mover_fact(comparison)

    assert row["portfolio"] == "ME5 BM2"
    assert int(row["performance_rank_ff3"]) == 9
    assert int(row["performance_rank_ff5"]) == 20
    source = (ROOT / "reports" / "generate_report.py").read_text()
    summary = (ROOT / "reports" / "TECHNICAL_REPORT.md").read_text()
    readme = (ROOT / "README.md").read_text()
    assert "max_mover_sentence" in source
    for fragment in ("ME5 BM2", "rank 9", "rank 20", "11-position"):
        assert fragment in summary
        assert fragment in readme
    assert sentence == (
        "ME5 BM2 moves from rank 9 under FF3 to rank 20 under FF5, "
        "an 11-position change."
    )


def test_book_chapter_has_no_prior_version_framing():
    source = (ROOT / "reports" / "generate_report.py").read_text().lower()
    banned = (
        "previous version",
        "earlier version",
        "legacy version",
        "pre-rebuild",
        "methodological rebuild",
    )
    assert all(phrase not in source for phrase in banned)
