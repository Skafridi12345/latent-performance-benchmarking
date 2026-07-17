from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.figures import generate_all_figures

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def main() -> None:
    generate_all_figures(
        performance_scores=pd.read_csv(TABLES / "performance_scores.csv"),
        rolling_performance=pd.read_csv(TABLES / "rolling_performance_scores.csv"),
        persistence=pd.read_csv(TABLES / "rank_persistence.csv"),
        transition_matrix=pd.read_csv(TABLES / "transition_matrix.csv", index_col=0),
        transition_summary=pd.read_csv(TABLES / "transition_summary.csv"),
        mobility=pd.read_csv(TABLES / "mobility_summary.csv"),
        robustness=pd.read_csv(TABLES / "robustness_summary.csv"),
        forward_validation=pd.read_csv(TABLES / "forward_performance_validation.csv"),
        sfa_diagnostics=pd.read_csv(TABLES / "sfa_asymmetry_diagnostics.csv"),
        residuals=pd.read_csv(TABLES / "performance_residuals.csv"),
        output_dir=FIGURES,
    )
    print(f"Figures written to: {FIGURES}")


if __name__ == "__main__":
    main()
