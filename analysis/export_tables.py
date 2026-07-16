from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "latex_tables"


def df_to_latex(
    df: pd.DataFrame,
    filename: str,
    caption: str,
    label: str,
    float_format: str = "%.3f",
) -> Path:
    """Write a DataFrame to a standalone LaTeX table."""

    OUT.mkdir(exist_ok=True)
    tex = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
    )

    path = OUT / filename
    path.write_text(tex, encoding="utf-8")
    return path


def main() -> None:
    """Export selected pipeline CSV outputs as LaTeX tables."""

    performance = pd.read_csv(TABLES / "performance_scores.csv")[
        [
            "portfolio",
            "posterior_alpha_annualized_bps",
            "bootstrap_rank_median",
            "bootstrap_rank_ci_low",
            "bootstrap_rank_ci_high",
        ]
    ].rename(
        columns={
            "portfolio": "Portfolio",
            "posterior_alpha_annualized_bps": "Posterior Alpha (bps/year)",
            "bootstrap_rank_median": "Median Rank",
            "bootstrap_rank_ci_low": "Rank CI Low",
            "bootstrap_rank_ci_high": "Rank CI High",
        }
    )
    persistence = pd.read_csv(TABLES / "rank_persistence.csv").rename(
        columns={
            "horizon_months": "Horizon",
            "spearman_rank_autocorrelation": "Spearman $\\rho$",
            "pearson_score_autocorrelation": "Pearson Score",
            "average_absolute_rank_change": "Avg. |Rank Change|",
        }
    )
    transition = pd.read_csv(
        TABLES / "transition_matrix.csv", index_col=0
    ).reset_index()
    mobility = pd.read_csv(TABLES / "mobility_summary.csv")[
        [
            "portfolio",
            "mean_rank",
            "rank_volatility",
            "same_quintile_probability",
            "move_up_probability",
            "move_down_probability",
        ]
    ]

    outputs = [
        df_to_latex(
            performance,
            "table_performance_rankings.tex",
            "HAC and Shrinkage-Adjusted Performance Rankings",
            "tab:performance_rankings",
        ),
        df_to_latex(
            persistence,
            "table_rank_persistence.tex",
            "Rank Persistence Across Rolling Windows",
            "tab:rank_persistence",
        ),
        df_to_latex(
            transition,
            "table_quintile_transitions.tex",
            "Quintile Transition Probabilities",
            "tab:quintile_transitions",
        ),
        df_to_latex(
            mobility,
            "table_quintile_mobility.tex",
            "Portfolio-Level Quintile Mobility Metrics",
            "tab:quintile_mobility",
        ),
    ]

    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
