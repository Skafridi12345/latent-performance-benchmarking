"""Generate the technical report as an academic research paper (PDF).

The layout follows the conventions of an empirical finance working paper:
a title block, an abstract with keywords and JEL codes, numbered sections and
subsections, "booktabs"-style tables, figures with numbered captions, in-text
author-year citations, and a reference list. Every numerical claim is read from
the canonical result tables in ``results/tables`` (and the run manifest), so the
document regenerates deterministically from a completed pipeline run.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
OUTPUT = ROOT / "output" / "pdf" / "latent-performance-benchmarking-technical-report.pdf"

FORWARD_MEAN = (
    "mean_average_top_quintile_minus_average_bottom_quintile_"
    "forward_alpha_annualized_bps"
)
FORWARD_MEDIAN = (
    "median_average_top_quintile_minus_average_bottom_quintile_"
    "forward_alpha_annualized_bps"
)
FORWARD_IQR = (
    "iqr_average_top_quintile_minus_average_bottom_quintile_"
    "forward_alpha_annualized_bps"
)

# Incremental rolling-vs-benchmark forward test (analysis/incremental_validation).
INCREMENTAL_SCHEMES = [
    ("lexicographic", "Characteristic gradient (size asc, B/M desc)"),
    ("diagonal", "Characteristic gradient (diagonal)"),
    ("expanding_window", "Expanding window (leakage-free)"),
    ("unconditional_alpha", "Full-sample alpha (oracle, look-ahead)"),
]
RANK_INCREMENT = "rank_spearman_increment"
SPREAD_INCREMENT = "spread_increment_bps"

# Restrained, journal-like palette: near-black text, one navy accent for the
# title and rules. Tables and body are monochrome.
INK = colors.HexColor("#1A1A1A")
NAVY = colors.HexColor("#14243B")
RULE = colors.HexColor("#000000")
GREY = colors.HexColor("#5B6570")
FAINT = colors.HexColor("#F4F6F8")

SERIF = "Times-Roman"
SERIF_BOLD = "Times-Bold"
SERIF_ITALIC = "Times-Italic"
SERIF_BI = "Times-BoldItalic"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PaperTitle",
            parent=styles["Title"],
            fontName=SERIF_BOLD,
            fontSize=17,
            leading=21,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Author",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=11,
            leading=15,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Affil",
            parent=styles["Normal"],
            fontName=SERIF_ITALIC,
            fontSize=9,
            leading=12,
            textColor=GREY,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AbstractHead",
            parent=styles["Normal"],
            fontName=SERIF_BOLD,
            fontSize=9.5,
            leading=12,
            textColor=INK,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Abstract",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=9,
            leading=12.6,
            textColor=INK,
            alignment=TA_JUSTIFY,
            leftIndent=14,
            rightIndent=14,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=8.5,
            leading=11.5,
            textColor=INK,
            alignment=TA_JUSTIFY,
            leftIndent=14,
            rightIndent=14,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName=SERIF_BOLD,
            fontSize=11.5,
            leading=14,
            textColor=INK,
            spaceBefore=11,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName=SERIF_BI,
            fontSize=10,
            leading=13,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName=SERIF,
            fontSize=9.5,
            leading=13.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            firstLineIndent=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyIndent",
            parent=styles["Body"],
            firstLineIndent=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Equation",
            parent=styles["Normal"],
            fontName=SERIF_ITALIC,
            fontSize=9.3,
            leading=13.5,
            textColor=INK,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperBullet",
            parent=styles["Body"],
            leftIndent=16,
            firstLineIndent=0,
            bulletIndent=4,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCaption",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=8.4,
            leading=11,
            textColor=INK,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FigCaption",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=8.4,
            leading=11,
            textColor=INK,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Note",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=7.8,
            leading=10,
            textColor=GREY,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=7.6,
            leading=9.4,
            textColor=INK,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CellLeft",
            parent=styles["Cell"],
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Reference",
            parent=styles["Normal"],
            fontName=SERIF,
            fontSize=8.6,
            leading=11.4,
            textColor=INK,
            alignment=TA_JUSTIFY,
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=4,
        )
    )
    return styles


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFont(SERIF_ITALIC, 7.6)
    canvas.setFillColor(GREY)
    if doc.page > 1:
        canvas.drawString(
            doc.leftMargin,
            height - 12 * mm,
            "Risk-Adjusted Benchmarking of the Fama-French Portfolios",
        )
    canvas.drawRightString(
        width - doc.rightMargin, 11 * mm, f"{doc.page}"
    )
    canvas.drawString(
        doc.leftMargin,
        11 * mm,
        "Historical research diagnostics; not investment advice.",
    )
    canvas.restoreState()


def _image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _p(text: str, styles, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def _bullet(text: str, styles, style: str = "PaperBullet") -> Paragraph:
    return Paragraph(text, styles[style], bulletText="•")


def _table(data, widths, styles, *, alignments=None, font_size=7.6) -> Table:
    """Render a booktabs-style academic table (horizontal rules only)."""

    formatted = []
    for row_idx, row in enumerate(data):
        cells = []
        for col_idx, value in enumerate(row):
            base = "CellLeft" if alignments and alignments[col_idx] == "left" else "Cell"
            style = ParagraphStyle(
                f"c-{row_idx}-{col_idx}-{id(data)}",
                parent=styles[base],
                fontSize=font_size,
                leading=font_size + 1.8,
                fontName=SERIF_BOLD if row_idx == 0 else SERIF,
            )
            cells.append(Paragraph(escape(str(value)), style))
        formatted.append(cells)
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), 1.1, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 1.1, RULE),
    ]
    table.setStyle(TableStyle(commands))
    return table


def _section(story, number: str, title: str, styles):
    story.append(Paragraph(f"{number}&nbsp;&nbsp;{title}", styles["Section"]))


def _subsection(story, number: str, title: str, styles):
    story.append(Paragraph(f"{number}&nbsp;&nbsp;{title}", styles["Subsection"]))


def _figure(story, path: str, caption: str, styles, *, width=5.0 * inch, height=4.2 * inch):
    img = _image(FIGURES / path, width, height)
    img.hAlign = "CENTER"
    story.append(Spacer(1, 3))
    story.append(img)
    story.append(_p(caption, styles, "FigCaption"))


def format_p_value(value: object, *, decimals: int = 4, scientific: bool = False) -> str:
    """Format a valid p-value without ever displaying a numerical zero."""

    if value is None or pd.isna(value):
        return "NA"
    p_value = float(value)
    if not 0.0 <= p_value <= 1.0:
        raise ValueError(f"Invalid p-value: {p_value}")
    threshold = 10.0 ** (-decimals)
    if p_value == 0.0:
        return f"<{threshold:.{decimals}f}"
    if p_value < threshold:
        return f"{p_value:.2e}" if scientific else f"<{threshold:.{decimals}f}"
    return f"{p_value:.{decimals}f}"


def posterior_positive_fact(performance: pd.DataFrame) -> tuple[int, list[str], str]:
    ordered = performance.sort_values("performance_rank")
    names = ordered.loc[ordered["posterior_alpha_ci_low"] > 0, "portfolio"].tolist()
    count = len(names)
    noun = "interval is" if count == 1 else "intervals are"
    return count, names, f"{count} posterior {noun} wholly above zero ({', '.join(names)})"


def maximum_rank_mover_fact(comparison: pd.DataFrame) -> tuple[pd.Series, str]:
    changes = comparison["rank_change_ff5_minus_ff3"].abs()
    row = comparison.loc[changes.idxmax()]
    movement = int(changes.loc[changes.idxmax()])
    sentence = (
        f"{row['portfolio']} moves from rank {int(row['performance_rank_ff3'])} "
        f"under FF3 to rank {int(row['performance_rank_ff5'])} under FF5, "
        f"an {movement}-position change."
    )
    return row, sentence


def _load_incremental() -> dict:
    """Load the incremental rolling-vs-benchmark test tables, if present."""

    windows: dict[str, pd.DataFrame] = {}
    summary: dict[str, pd.DataFrame] = {}
    for scheme, _ in INCREMENTAL_SCHEMES:
        w = TABLES / f"incremental_validation_windows_{scheme}.csv"
        s = TABLES / f"incremental_validation_summary_{scheme}.csv"
        if not w.exists() or not s.exists():
            raise FileNotFoundError(
                "Incremental validation tables are missing. Run "
                "`python -m analysis.incremental_validation --scheme all` "
                "or the full `python -m analysis.run_all` pipeline first."
            )
        windows[scheme] = pd.read_csv(w)
        summary[scheme] = pd.read_csv(s).set_index("metric")
    return {"windows": windows, "summary": summary}


def build_report(output: Path = OUTPUT) -> Path:
    performance = pd.read_csv(TABLES / "performance_scores.csv")
    prior = pd.read_csv(TABLES / "performance_prior.csv").iloc[0]
    manifest = pd.read_csv(TABLES / "dataset_manifest.csv").iloc[0]
    persistence = pd.read_csv(TABLES / "rank_persistence.csv")
    forward = pd.read_csv(TABLES / "forward_performance_aggregate.csv").iloc[0]
    residual_diag = pd.read_csv(TABLES / "performance_residual_diagnostics.csv")
    sfa = pd.read_csv(TABLES / "sfa_asymmetry_diagnostics.csv")
    robustness = pd.read_csv(TABLES / "robustness_summary.csv")
    joint_tests = pd.read_csv(TABLES / "joint_alpha_tests.csv").set_index("test")
    stability = pd.read_csv(TABLES / "bootstrap_stability_1000_vs_final.csv")
    external = pd.read_csv(TABLES / "external_validation_checks.csv")
    ff5_joint = pd.read_csv(TABLES / "ff3_ff5_joint_alpha_tests.csv")
    ff5_comparison = pd.read_csv(TABLES / "ff3_ff5_sensitivity_comparison.csv")
    historical_tests = pd.read_csv(
        TABLES / "historical_anchor_joint_alpha_tests.csv"
    ).set_index("test")
    transition = pd.read_csv(TABLES / "transition_summary.csv")
    raw_shrunk = pd.read_csv(TABLES / "raw_vs_shrunk_alpha_ranks.csv")
    incremental = _load_incremental()
    with (ROOT / "results" / "run_manifest.json").open(encoding="utf-8") as handle:
        run_manifest = json.load(handle)

    # ---- Derived facts (identical provenance to the tables) -----------------
    top = performance.iloc[0]
    bottom = performance.iloc[-1]
    _, _, positive_interval_sentence = posterior_positive_fact(performance)
    fdr_supported_alpha = performance.loc[performance["alpha_fdr_q_value"] < 0.05].copy()
    supported_sfa = sfa[sfa["sfa_supported_fdr_5pct"].fillna(False).astype(bool)]
    normal_reject = int((residual_diag["jarque_bera_p_value"] < 0.05).sum())
    serial_reject = int((residual_diag["ljung_box_lag_12_p_value"] < 0.05).sum())
    arch_reject = int((residual_diag["arch_lm_lag_12_p_value"] < 0.05).sum())
    rank_endpoint_change = max(
        stability["absolute_change_bootstrap_rank_ci_low"].max(),
        stability["absolute_change_bootstrap_rank_ci_high"].max(),
    )
    alpha_endpoint_change_bps = (
        max(
            stability["absolute_change_bootstrap_posterior_alpha_ci_low"].max(),
            stability["absolute_change_bootstrap_posterior_alpha_ci_high"].max(),
        )
        * 12
        * 10_000
    )
    external_by_id = external.set_index("check_id")
    portfolio_comparison = external_by_id.loc["portfolio_returns_exact_overlap"]
    factor_comparison = external_by_id.loc["ff3_factors_exact_overlap"]
    portfolio_difference = portfolio_comparison["absolute_difference"]
    factor_difference = factor_comparison["absolute_difference"]
    replication_passes = int(
        (
            (external["category"] == "independent_replication")
            & (external["status"] == "PASS")
        ).sum()
    )
    prior_mean_bps = prior["mu"] * 12 * 10_000
    prior_tau_bps = prior["tau"] * 12 * 10_000
    ff3_hac = ff5_joint.query(
        "sensitivity_model == 'ff3_common_sample' and test == 'HAC_Wald'"
    ).iloc[0]
    ff5_hac = ff5_joint.query(
        "sensitivity_model == 'ff5_common_sample' and test == 'HAC_Wald'"
    ).iloc[0]
    ff_rank_rho = ff5_comparison["performance_rank_ff3"].corr(
        ff5_comparison["performance_rank_ff5"], method="spearman"
    )
    _, max_mover_sentence = maximum_rank_mover_fact(ff5_comparison)
    ff3_fdr_count = int((ff5_comparison["alpha_fdr_q_value_ff3"] < 0.05).sum())
    ff5_fdr_count = int((ff5_comparison["alpha_fdr_q_value_ff5"] < 0.05).sum())
    mean_transition_diagonal = float(transition["stay_probability"].mean())
    max_shrink_shift = int(raw_shrunk["rank_change_after_shrinkage"].abs().max())

    # Incremental test facts.
    inc_windows = incremental["windows"]
    inc_summary = incremental["summary"]
    rolling_level = float(inc_windows["lexicographic"]["rolling_rank_spearman"].mean())

    def inc_stat(scheme: str, metric: str, field: str) -> float:
        return float(inc_summary[scheme].loc[metric, field])

    def bench_level(scheme: str) -> float:
        return float(inc_windows[scheme]["static_rank_spearman"].mean())

    def rank_sig(scheme: str) -> bool:
        return (
            inc_stat(scheme, RANK_INCREMENT, "hac_p_value_one_sided") < 0.05
            and inc_stat(scheme, RANK_INCREMENT, "bootstrap_ci_low") > 0.0
        )

    exp_incr = inc_stat("expanding_window", RANK_INCREMENT, "mean")
    exp_p = inc_stat("expanding_window", RANK_INCREMENT, "hac_p_value_one_sided")
    oracle_incr = inc_stat("unconditional_alpha", RANK_INCREMENT, "mean")
    oracle_bench = bench_level("unconditional_alpha")
    exp_bench = bench_level("expanding_window")
    lex_incr = inc_stat("lexicographic", RANK_INCREMENT, "mean")

    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="Risk-Adjusted Benchmarking of the Fama-French Portfolios",
        author="Muhammad Shoaib",
        subject="Latent performance, rank uncertainty, and the limits of forecastability",
    )
    story: list = []

    # ===================== Front matter =====================
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "Uncertainty-Aware Risk-Adjusted Benchmarking of the "
            "Fama&ndash;French Portfolios: Latent Performance, Rank "
            "Uncertainty, and the Limits of Forecastability",
            styles["PaperTitle"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Muhammad Shoaib", styles["Author"]))
    story.append(
        Paragraph("Working paper &middot; This version: 16 July 2026", styles["Affil"])
    )
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Abstract", styles["AbstractHead"]))
    story.append(
        _p(
            "We benchmark the 25 Fama&ndash;French size and book-to-market "
            "portfolios on a validated balanced panel of "
            f"{int(manifest['n_observations']):,} portfolio-months "
            f"({int(manifest['n_months']):,} months, July 1926&ndash;November 2025). "
            "The estimand is the Fama&ndash;French three-factor (FF3) alpha; the "
            "primary uncertainty object is the full 25&times;25 heteroskedasticity- "
            "and autocorrelation-consistent (HAC) covariance of the alpha vector; "
            "and the reported score is a multivariate empirical-Bayes posterior "
            "alpha. Rankings are accompanied by common-date block-bootstrap rank "
            "intervals and evaluated out of sample. The joint HAC/Wald test rejects "
            "the hypothesis that all alphas are zero "
            f"(p = {format_p_value(joint_tests.loc['HAC_Wald', 'p_value'], scientific=True)}), "
            "but partial pooling, wide bootstrap rank intervals, and strong "
            "long-horizon mobility show that the point ranking is not a precise "
            "hierarchy. In strictly forward windows the top-minus-bottom quintile "
            f"spread averages {forward[FORWARD_MEAN]:.0f} bps/year, yet an incremental "
            "test decomposes this signal: the time-varying rolling ranking adds no "
            "predictive power over a leakage-free expanding-window ranking "
            f"(mean increment {exp_incr:+.3f}, HAC one-sided p = {exp_p:.2f}) and is "
            "dominated by a single fixed full-sample ordering. The forward result is "
            "therefore consistent with persistent characteristic mispricing rather "
            "than dynamic forecasting skill. Alpha is benchmark-dependent (an "
            "11-rank FF3-to-FF5 move), and only one of 25 stochastic-frontier "
            "boundary tests survives false-discovery control.",
            styles,
            "Abstract",
        )
    )
    story.append(
        _p(
            "<b>Keywords:</b> cross-sectional asset pricing; empirical Bayes; "
            "HAC inference; rank uncertainty; out-of-sample validation; "
            "false-discovery control.",
            styles,
            "Meta",
        )
    )
    story.append(
        _p(
            "<b>JEL classification:</b> G11, G12, C11, C58.",
            styles,
            "Meta",
        )
    )
    story.append(
        _p(
            "<b>Disclaimer:</b> This paper reports historical research diagnostics "
            "on passively constructed research portfolios. It is not investment "
            "advice, a trading strategy, or evidence of manager-specific skill.",
            styles,
            "Meta",
        )
    )
    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=8, spaceAfter=6))

    # ===================== 1. Introduction =====================
    _section(story, "1.", "Introduction", styles)
    story.append(
        _p(
            "A portfolio can earn a high average return because it bears systematic "
            "risk, because its sample happened to be favourable, because the "
            "benchmark is incomplete, or because it contains genuinely unusual "
            "performance. Factor benchmarking removes the part linearly associated "
            "with priced factors and studies the intercept, or alpha. For the 25 "
            "Fama&ndash;French portfolios&mdash;the canonical test assets from which "
            "the size and value factors are built (Fama and French, 1993)&mdash;this "
            "intercept is a model-relative quantity: it is unexplained by the chosen "
            "factors, not by every conceivable risk or friction.",
            styles,
        )
    )
    story.append(
        _p(
            "This paper makes the uncertainty around such a benchmark explicit and "
            "then asks a sharper question than the usual out-of-sample test. Our "
            "contributions are fourfold. First, we estimate the full cross-portfolio "
            "HAC covariance of the 25 alphas (Newey and West, 1987) rather than 25 "
            "unrelated standard errors, and use it for joint testing and for "
            "multivariate empirical-Bayes shrinkage (Efron and Morris, 1973). "
            "Second, we quantify ranking uncertainty with a common-date circular "
            "block bootstrap (Politis and Romano, 1992) that refits every stage of "
            "the estimator. Third, we separate description from validation with a "
            "strictly forward evaluation in which scores, ranks, and factor loadings "
            "are frozen before the evaluation window. Fourth&mdash;and centrally&mdash;we "
            "introduce an <i>incremental</i> forward test that isolates whether the "
            "time-varying ranking adds anything beyond a fixed characteristic "
            "ordering.",
            styles,
        )
    )
    story.append(
        _p(
            "The headline results are as follows. The joint zero-alpha hypothesis is "
            "rejected under the dependence-robust HAC/Wald test, and "
            f"{len(fdr_supported_alpha)} of 25 individual alphas survive 5% "
            "Benjamini&ndash;Hochberg false-discovery control (Benjamini and "
            "Hochberg, 1995). Yet bootstrap rank intervals are wide and long-horizon "
            "quintile mobility is high, so the point ranking is not a reliable league "
            "table. The strictly forward top-minus-bottom quintile spread is positive "
            f"on average ({forward[FORWARD_MEAN]:.0f} bps/year). The incremental test "
            "then shows that this predictability is <i>not</i> dynamic: the 120-month "
            "rolling ranking beats a naive size/value characteristic gradient "
            f"(increment {lex_incr:+.3f}) but does not beat a leakage-free "
            f"expanding-window ranking ({exp_incr:+.3f}, p = {exp_p:.2f}), and is "
            "itself dominated by a single fixed full-sample ordering "
            f"({oracle_incr:+.3f}). The predictive content lives in a stable, "
            "irregular cross-sectional alpha pattern that FF3 misprices persistently, "
            "not in the re-estimation of the ranking through time.",
            styles,
        )
    )
    story.append(
        _p(
            "The remainder of the paper is organised as follows. Section 2 describes "
            "the data and integrity checks. Section 3 sets out the estimator: the "
            "factor regression, the joint HAC covariance, empirical-Bayes shrinkage, "
            "joint and multiple testing, the bootstrap, and the incremental forward "
            "design. Section 4 reports the empirical results, Section 5 the "
            "robustness and validation analysis, Section 6 discusses interpretation "
            "and limitations, and Section 7 concludes.",
            styles,
        )
    )

    # ===================== 2. Data =====================
    _section(story, "2.", "Data", styles)
    _subsection(story, "2.1", "Portfolios and sample", styles)
    story.append(
        _p(
            "The 25 test portfolios form a 5&times;5 grid. At each formation date "
            "eligible stocks are sorted independently into five size and five "
            "book-to-market groups; returns within each of the 25 intersections are "
            "value weighted. Labels such as SMALL HiBM and BIG LoBM describe the two "
            "sorting dimensions, not individual firms or managed funds. The unit of "
            "observation is one portfolio-month; the dependent variable is the "
            "portfolio return minus the one-month risk-free rate, in decimal monthly "
            "units. The canonical benchmark is FF3 (Mkt&minus;RF, SMB, HML). "
            "Annualised basis points multiply monthly decimal alpha by 12 and by "
            "10,000, a linear reporting convention rather than geometric compounding.",
            styles,
        )
    )
    story.append(Paragraph("Table 1. Sample and canonical specification.", styles["TableCaption"]))
    spec_rows = [
        ["Element", "Specification", "Element", "Specification"],
        ["Cross-section", "25 size x B/M portfolios", "Primary HAC lag", "12 months"],
        ["Time span", "1926:07-2025:11", "Rolling window", "120 months, step 12"],
        ["Months", f"{int(manifest['n_months']):,}", "Forward horizon", "12 months"],
        ["Observations", f"{int(manifest['n_observations']):,}", "Bootstrap draws", "5,000 (seed 2026)"],
        ["Benchmark", "FF3: Mkt-RF, SMB, HML", "Block length", "12 months, circular"],
    ]
    story.append(
        _table(
            spec_rows,
            [1.25 * inch, 1.7 * inch, 1.2 * inch, 1.55 * inch],
            styles,
            alignments=["left", "left", "left", "left"],
            font_size=7.6,
        )
    )
    story.append(Spacer(1, 3 * mm))

    _subsection(story, "2.2", "Integrity and provenance", styles)
    story.append(
        _p(
            "Data integrity is treated as part of the specification. The analytical "
            f"grain is exactly one portfolio-month, giving {int(manifest['n_unique_portfolio_months']):,} "
            "unique keys with zero duplicates, a balanced panel, and no missing "
            "calendar months; the loader fails closed on any violation. Each run "
            "records the source byte count and SHA-256 digest of both inputs.",
            styles,
        )
    )
    story.append(
        _p(
            "A current (May 2026) official snapshot from the Kenneth R. French Data "
            "Library is retained separately and never overwrites the fixed local "
            "inputs. The local sample ends in November 2025, but its exact "
            "acquisition vintage is unverified, and the two sources differ over their "
            f"overlap&mdash;by up to {portfolio_difference * 100:.4f} percentage points "
            f"for portfolio returns and {factor_difference * 100:.3f} percentage "
            "points for FF3/RF fields. Official historical revisions are a plausible "
            "but unproven cause. Both comparisons are therefore classified "
            "NOT_COMPARABLE (visible, non-blocking) rather than PASS. This is a "
            "genuine limitation: the headline numbers rest on inputs that cannot "
            "currently be reconstructed byte-for-byte from the public source. All "
            "within-pipeline integrity and independent-replication checks pass "
            "(Section 5.4).",
            styles,
        )
    )

    # ===================== 3. Methodology =====================
    _section(story, "3.", "Methodology", styles)
    _subsection(story, "3.1", "Factor model and the meaning of alpha", styles)
    story.append(
        _p(
            "For portfolio i in month t, with excess return y<sub>it</sub> = "
            "r<sub>it</sub> &minus; r<sub>ft</sub>, the benchmark regression is",
            styles,
        )
    )
    story.append(
        _p(
            "y<sub>it</sub> = &alpha;<sub>i</sub> + &beta;<sub>i</sub>&prime; f<sub>t</sub> + &epsilon;<sub>it</sub>,",
            styles,
            "Equation",
        )
    )
    story.append(
        _p(
            "where f<sub>t</sub> stacks the FF3 factor returns and &alpha;<sub>i</sub> "
            "is the mean excess return left after the fitted linear factor component "
            "is removed. Ordinary least squares gives "
            "&theta;^<sub>i</sub> = (X&prime;X)<sup>&minus;1</sup>X&prime;y<sub>i</sub>. "
            "Selecting the intercept, the estimation error is a weighted sum of "
            "residual shocks, &alpha;^<sub>i</sub> &minus; &alpha;<sub>i</sub> = "
            "&sum;<sub>t</sub> a<sub>t</sub>&epsilon;<sub>it</sub>, where the scalar "
            "weight a<sub>t</sub> depends only on the common factor design. Because "
            "every portfolio shares the same dates and factors, these weights combine "
            "with the same-month residual vector to yield a <i>joint</i> covariance. "
            "A positive alpha denotes in-sample benchmark-relative outperformance, not "
            "managerial skill or future profit; a negative alpha can reflect omitted "
            "risk rather than inefficiency.",
            styles,
        )
    )

    _subsection(story, "3.2", "Full joint HAC covariance", styles)
    story.append(
        _p(
            "Let e<sub>t</sub> be the 25-vector of same-month residuals and "
            "g<sub>t</sub> = a<sub>t</sub>e<sub>t</sub> the alpha score vector. With "
            "L = 12 lags, Bartlett weights w<sub>l</sub> = 1 &minus; l/(L+1), and "
            "lag-l cross-product &Gamma;<sub>l</sub> = &sum;<sub>t&gt;l</sub> "
            "g<sub>t</sub>g<sub>t&minus;l</sub>&prime;, the covariance is",
            styles,
        )
    )
    story.append(
        _p(
            "V<sub>HAC</sub> = [T/(T&minus;P)]&nbsp;{&Gamma;<sub>0</sub> + "
            "&sum;<sub>l=1</sub><sup>L</sup> w<sub>l</sub>(&Gamma;<sub>l</sub> + "
            "&Gamma;<sub>l</sub>&prime;)},",
            styles,
            "Equation",
        )
    )
    story.append(
        _p(
            "with P = K + 1 regression coefficients. This 25&times;25 matrix keeps "
            "individual alpha variances on its diagonal and cross-portfolio "
            "co-movement off-diagonal. The implementation symmetrises the result, "
            "checks eigenvalues, and permits only scale-relative flooring of "
            "negligible numerical negatives. The canonical matrix is positive "
            f"definite without flooring, with condition number "
            f"{prior['joint_hac_condition_number']:.1f}. HAC inference is robust to "
            "the modelled dependence; it does not make residuals Gaussian or repair "
            "an incomplete benchmark.",
            styles,
        )
    )

    _subsection(story, "3.3", "Multivariate empirical Bayes", styles)
    story.append(
        _p(
            "Ranking 25 noisy estimates invites a winner's curse. Partial pooling "
            "treats the latent alphas as one cross-section,",
            styles,
        )
    )
    story.append(
        _p(
            "&alpha;^ | &alpha; ~ N(&alpha;, V<sub>HAC</sub>), &nbsp;&nbsp; "
            "&alpha; ~ N(&mu;1, &tau;<sup>2</sup>I),",
            styles,
            "Equation",
        )
    )
    story.append(
        _p(
            "with posterior mean &mu;1 + &tau;<sup>2</sup>(V<sub>HAC</sub> + "
            "&tau;<sup>2</sup>I)<sup>&minus;1</sup>(&alpha;^ &minus; &mu;1) "
            "and conditional covariance &tau;<sup>2</sup>I &minus; "
            "&tau;<sup>4</sup>(V<sub>HAC</sub> + &tau;<sup>2</sup>I)<sup>&minus;1</sup>. "
            "The hyperparameters (&mu;, &tau;) are estimated by profile marginal "
            "maximum likelihood, so the prior is learned from the same cross-section. "
            f"The fitted prior mean is {prior_mean_bps:.0f} bps/year with dispersion "
            f"{prior_tau_bps:.0f} bps/year, implying meaningful shrinkage toward a "
            "slightly negative centre; shrinkage moves at least one portfolio by "
            f"{max_shrink_shift} ranks. Because V<sub>HAC</sub> is non-diagonal, "
            "shrinkage is multivariate. The reported intervals include first-order "
            "uncertainty in the profiled mean but do not fully integrate uncertainty "
            "in &tau;, so they are model-based summaries rather than exact "
            "finite-sample guarantees.",
            styles,
        )
    )

    _subsection(story, "3.4", "Joint and multiple testing", styles)
    story.append(
        _p(
            "The primary joint test is the dependence-robust Wald statistic "
            "W = &alpha;^&prime; V<sub>HAC</sub><sup>&minus;1</sup> &alpha;^ "
            "~ &chi;<sup>2</sup>(25) under the null that all alphas are zero. The "
            "Gibbons&ndash;Ross&ndash;Shanken F-test (Gibbons, Ross, and Shanken, "
            "1989) is reported as a conventional secondary benchmark; its exact F "
            "reference assumes IID Gaussian errors, which the diagnostics reject "
            "(Section 5.3). Individual raw-alpha p-values and the 25 SFA boundary "
            "p-values form two separate Benjamini&ndash;Hochberg families, since they "
            "test different nulls.",
            styles,
        )
    )

    _subsection(story, "3.5", "Bootstrap rank uncertainty", styles)
    story.append(
        _p(
            "A rank is discontinuous, so individual standard errors do not describe "
            "how often a portfolio stays in the top five. We draw 5,000 circular "
            "12-month blocks of common date indices, apply the same dates to all 25 "
            "portfolios (preserving cross-sectional shocks), and refit the "
            "regressions, the full HAC covariance, and the empirical-Bayes "
            "hyperparameters within every replicate before re-ranking. Rank and score "
            "intervals are empirical 2.5&ndash;97.5% quantiles of this resampling "
            "distribution. Increasing the draw count from 1,000 to 5,000 shifts rank "
            f"interval endpoints by at most {rank_endpoint_change:.0f} rank but "
            f"alpha-tail endpoints by up to {alpha_endpoint_change_bps:.0f} annualised "
            "bps, motivating the larger final count.",
            styles,
        )
    )

    _subsection(story, "3.6", "Forward validation and the incremental test", styles)
    story.append(
        _p(
            "For each 120-month training window we freeze all scores, ranks, prior "
            "parameters, and betas at the cutoff, then compute each portfolio's next "
            "12-month benchmark-adjusted return with the frozen loadings, "
            "A<sub>i</sub><sup>+</sup> = mean<sub>t</sub>(y<sub>it</sub> &minus; "
            "&beta;^<sub>i</sub>&prime; f<sub>t</sub>). Within each evaluation "
            "year we compute the Spearman correlation between training scores and "
            "future alpha and the average top-minus-bottom quintile spread. Annual "
            "statistics are aggregated with a Fisher transformation and four-lag HAC "
            "inference.",
            styles,
        )
    )
    story.append(
        _p(
            "The forward test alone cannot separate genuine dynamic skill from "
            "persistent characteristic mispricing: because FF3 misprices fixed "
            "characteristic cells in a stable way, a ranking anchored to past alpha "
            "will correlate with future alpha mechanically. We therefore add an "
            "<i>incremental</i> test that scores the rolling ranking against four "
            "fixed or leakage-free benchmark orderings on the <i>same</i> forward "
            "observations, and tests whether the rolling ordering adds power. The "
            "benchmarks are (i) a size-ascending / B/M-descending characteristic "
            "gradient; (ii) a diagonal value-minus-size gradient; (iii) a leakage-free "
            "<i>expanding-window</i> ranking re-estimated on all data up to each "
            "cutoff; and (iv) a full-sample <i>oracle</i> ordering (which uses "
            "look-ahead and is thus an upper bound on what a static ranking can "
            "achieve). Each per-window increment (rolling minus benchmark) is "
            "aggregated with an automatic-bandwidth Newey&ndash;West HAC t-test and a "
            "block bootstrap. Benchmark (iii) is the fair, tradable comparator on "
            "which the interpretation rests.",
            styles,
        )
    )

    # ===================== 4. Results =====================
    _section(story, "4.", "Results", styles)
    _subsection(story, "4.1", "Cross-sectional posterior performance", styles)
    story.append(
        _p(
            f"Table 2 reports the full ranking. {top['portfolio']} has the highest "
            f"posterior alpha ({top['posterior_alpha_annualized_bps']:.0f} bps/year) "
            f"and {bottom['portfolio']} the lowest "
            f"({bottom['posterior_alpha_annualized_bps']:.0f} bps/year), reproducing "
            "the well-known small-value premium and small-growth shortfall relative "
            f"to FF3. {positive_interval_sentence[0].upper()}{positive_interval_sentence[1:]}; "
            "most intervals cross zero. Five raw-alpha tests survive BH control, but "
            "four of these correspond to significantly negative alpha, so raw-test "
            "significance and posterior rank are distinct summaries.",
            styles,
        )
    )
    story.append(Paragraph("Table 2. Posterior FF3 performance ranking (annualised basis points).", styles["TableCaption"]))
    ranking_rows = [["Rank", "Portfolio", "Raw alpha", "Post. alpha", "Post. 95%", "BH q", "Boot. rank 95%"]]
    for row in performance.itertuples(index=False):
        ranking_rows.append(
            [
                int(row.performance_rank),
                row.portfolio,
                f"{row.alpha * 12 * 10_000:.0f}",
                f"{row.posterior_alpha_annualized_bps:.0f}",
                f"[{row.posterior_alpha_ci_low * 12 * 10_000:.0f}, {row.posterior_alpha_ci_high * 12 * 10_000:.0f}]",
                "<0.001" if row.alpha_fdr_q_value < 0.001 else f"{row.alpha_fdr_q_value:.3f}",
                f"[{row.bootstrap_rank_ci_low:.1f}, {row.bootstrap_rank_ci_high:.1f}]",
            ]
        )
    story.append(
        _table(
            ranking_rows,
            [0.4 * inch, 0.95 * inch, 0.7 * inch, 0.72 * inch, 1.15 * inch, 0.6 * inch, 1.05 * inch],
            styles,
            alignments=["center", "left", "right", "right", "center", "center", "center"],
            font_size=6.7,
        )
    )
    story.append(
        _p(
            "Note: Raw and posterior alpha in annualised basis points; posterior 95% "
            "is the empirical-Bayes interval; BH q is the false-discovery-adjusted "
            "raw-alpha p-value; the bootstrap rank interval is the central 95% range "
            "over 5,000 common-date circular-block draws.",
            styles,
            "Note",
        )
    )
    _figure(
        story,
        "performance_ranking.png",
        "Figure 1. Posterior FF3 alpha (annualised bps) with approximate 95% posterior "
        "intervals. An interval crossing zero indicates the latent alpha is not cleanly "
        "separated from zero; interval overlap warns that adjacent ranks are not "
        "distinguished.",
        styles,
        width=4.7 * inch,
        height=4.9 * inch,
    )
    _figure(
        story,
        "performance_heatmap_size_bm.png",
        "Figure 2. Posterior annualised alpha on the 5&times;5 size/book-to-market grid. "
        "The grid restores the economic sorting structure that a one-dimensional list "
        "hides and should be read jointly with intervals and benchmark sensitivity.",
        styles,
        width=4.6 * inch,
        height=3.7 * inch,
    )

    _subsection(story, "4.2", "Joint tests", styles)
    story.append(Paragraph("Table 3. Joint tests of the hypothesis that all 25 alphas are zero.", styles["TableCaption"]))
    joint_rows = [
        ["Test", "Statistic", "p-value", "Role"],
        ["HAC/Wald", f"{joint_tests.loc['HAC_Wald', 'statistic']:.2f}", format_p_value(joint_tests.loc['HAC_Wald', 'p_value'], scientific=True), "Primary; dependence-robust"],
        ["GRS", f"{joint_tests.loc['GRS', 'statistic']:.2f}", format_p_value(joint_tests.loc['GRS', 'p_value'], scientific=True), "Secondary; IID-Gaussian reference"],
    ]
    story.append(
        _table(
            joint_rows,
            [1.0 * inch, 1.0 * inch, 1.0 * inch, 2.3 * inch],
            styles,
            alignments=["left", "right", "right", "left"],
        )
    )
    story.append(
        _p(
            "Both tests reject. Rejection means the 25-dimensional alpha vector is "
            "inconsistent with all-zero under the stated model; it does not establish "
            "that alphas are positive, that the ordering is stable, or that any "
            "portfolio reflects persistent skill.",
            styles,
        )
    )

    _subsection(story, "4.3", "Rank uncertainty", styles)
    _figure(
        story,
        "rank_uncertainty.png",
        "Figure 3. Median bootstrap rank and central 95% rank interval from 5,000 "
        "common-date circular-block draws. Broad, overlapping intervals are direct "
        "evidence against reading the ordered list as a precise hierarchy.",
        styles,
        width=4.7 * inch,
        height=4.9 * inch,
    )
    story.append(
        _p(
            f"{top['portfolio']} is rank 1 at the point estimate and appears in the "
            f"top quintile in {top['bootstrap_top_quintile_probability']:.0%} of "
            f"bootstrap histories, yet its central rank interval still reaches rank "
            f"{top['bootstrap_rank_ci_high']:.0f}. Rank intervals overlap broadly "
            "across portfolios.",
            styles,
        )
    )

    _subsection(story, "4.4", "Persistence and mobility", styles)
    grouped = persistence.groupby(
        ["horizon_months", "window_overlap_fraction", "structural_inference_eligible"],
        as_index=False,
    ).agg(
        mean_spearman=("spearman_rank_autocorrelation", "mean"),
        mean_score=("pearson_score_autocorrelation", "mean"),
        mean_rank_change=("average_absolute_rank_change", "mean"),
    )
    story.append(Paragraph("Table 4. Rank and score persistence by calendar horizon.", styles["TableCaption"]))
    persistence_rows = [["Horizon", "Window overlap", "Mean rank rho", "Mean score r", "Mean |rank chg|", "Structural"]]
    for row in grouped.itertuples(index=False):
        persistence_rows.append(
            [
                f"{int(row.horizon_months)}m",
                f"{row.window_overlap_fraction:.0%}",
                f"{row.mean_spearman:.3f}",
                f"{row.mean_score:.3f}",
                f"{row.mean_rank_change:.2f}",
                "Yes" if row.structural_inference_eligible else "No",
            ]
        )
    story.append(
        _table(
            persistence_rows,
            [0.7 * inch, 1.0 * inch, 0.95 * inch, 0.9 * inch, 1.0 * inch, 0.85 * inch],
            styles,
            alignments=["center"] * 6,
        )
    )
    story.append(
        _p(
            "Rank persistence falls from about 0.86 at 12 months (90% window overlap) "
            "to roughly 0.16 at 120 months (zero overlap). The non-overlapping "
            f"horizon is the credible structural measure and is weak: the mean "
            f"probability of staying in the same quintile after 120 months is "
            f"{mean_transition_diagonal:.0%} (Table 5), only modestly above the 20% "
            "implied by uniform mobility.",
            styles,
        )
    )
    story.append(Paragraph("Table 5. Quintile transitions over a non-overlapping 120-month horizon.", styles["TableCaption"]))
    transition_rows = [["Starting quintile", "Stay", "Move up", "Move down"]]
    for row in transition.itertuples(index=False):
        transition_rows.append(
            [
                int(row.from_quintile),
                f"{row.stay_probability:.0%}",
                f"{row.upgrade_probability:.0%}",
                f"{row.downgrade_probability:.0%}",
            ]
        )
    story.append(
        _table(
            transition_rows,
            [1.5 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch],
            styles,
            alignments=["center"] * 4,
        )
    )
    _figure(
        story,
        "rank_persistence.png",
        "Figure 4. Mean rank and score persistence by calendar horizon, with window "
        "overlap shown explicitly. The 120-month point compares disjoint decades.",
        styles,
        width=4.9 * inch,
        height=3.3 * inch,
    )

    _subsection(story, "4.5", "Strictly forward validation", styles)
    story.append(
        _p(
            f"Across {int(forward['n_validation_windows'])} non-overlapping 12-month "
            f"evaluation periods, the Fisher-averaged rank correlation between "
            f"training scores and future alpha is {forward['mean_rank_spearman_fisher']:.3f} "
            f"(HAC 95% CI [{forward['rank_spearman_ci_low']:.3f}, "
            f"{forward['rank_spearman_ci_high']:.3f}]), and the average top-minus-bottom "
            f"quintile spread is {forward[FORWARD_MEAN]:.0f} bps/year (median "
            f"{forward[FORWARD_MEDIAN]:.0f}, IQR {forward[FORWARD_IQR]:.0f}), positive "
            f"in {forward['positive_average_quintile_spread_fraction']:.0%} of windows "
            "(Table 6). Both means are statistically distinguishable from zero, but "
            "dispersion across years is large&mdash;the IQR exceeds the mean&mdash;so the "
            "average is not a uniformly reliable timing rule.",
            styles,
        )
    )
    story.append(Paragraph("Table 6. Strictly forward validation (four-lag HAC inference).", styles["TableCaption"]))
    forward_rows = [
        ["Metric", "Estimate", "HAC 95% interval", "HAC p"],
        ["Rank vs future alpha", f"{forward['mean_rank_spearman_fisher']:.3f}", f"[{forward['rank_spearman_ci_low']:.3f}, {forward['rank_spearman_ci_high']:.3f}]", format_p_value(forward['rank_spearman_hac_p_value'], scientific=True)],
        ["Top-bottom quintile (bps/y)", f"{forward[FORWARD_MEAN]:.0f}", f"[{forward['average_quintile_spread_ci_low_bps']:.0f}, {forward['average_quintile_spread_ci_high_bps']:.0f}]", format_p_value(forward['average_quintile_spread_hac_p_value'], scientific=True)],
    ]
    story.append(
        _table(
            forward_rows,
            [1.9 * inch, 1.0 * inch, 1.35 * inch, 0.9 * inch],
            styles,
            alignments=["left", "right", "center", "right"],
        )
    )
    _figure(
        story,
        "forward_performance_validation.png",
        "Figure 5. Window-by-window future rank correlation and average top-minus-bottom "
        "quintile future alpha. Frequent negative windows and large spread variation "
        "show the average is not uniformly reliable.",
        styles,
        width=4.9 * inch,
        height=4.0 * inch,
    )

    _subsection(story, "4.6", "The incremental test: is the ranking dynamically informative?", styles)
    story.append(
        _p(
            "Table 7 decomposes the forward signal. The rolling ranking's own level "
            f"correlation with future alpha is {rolling_level:+.3f}. It exceeds the "
            "two static characteristic gradients&mdash;which have essentially zero "
            "forward power, because FF3 alpha is the residual after size and value "
            "exposure are removed&mdash;so the rolling signal is not merely a restated "
            "size/value tilt. But it does not beat the leakage-free expanding-window "
            f"ranking (level {exp_bench:+.3f}; increment {exp_incr:+.3f}, HAC one-sided "
            f"p = {exp_p:.2f}), and it is dominated by the full-sample oracle ordering "
            f"(level {oracle_bench:+.3f}; increment {oracle_incr:+.3f}, significantly "
            "negative). A single fixed but irregular cross-sectional ordering predicts "
            "future alpha as well as or better than the ranking that is re-estimated "
            "every year.",
            styles,
        )
    )
    story.append(Paragraph("Table 7. Incremental predictive power of the rolling ranking versus fixed and leakage-free benchmarks.", styles["TableCaption"]))
    inc_rows = [["Benchmark ordering", "Bench. corr", "Rank-corr incr.", "HAC p", "Bootstrap 95% CI", "Spread incr."]]
    for scheme, label in INCREMENTAL_SCHEMES:
        inc_rows.append(
            [
                label,
                f"{bench_level(scheme):+.3f}",
                f"{inc_stat(scheme, RANK_INCREMENT, 'mean'):+.3f}",
                format_p_value(inc_stat(scheme, RANK_INCREMENT, "hac_p_value_one_sided")),
                f"[{inc_stat(scheme, RANK_INCREMENT, 'bootstrap_ci_low'):+.3f}, {inc_stat(scheme, RANK_INCREMENT, 'bootstrap_ci_high'):+.3f}]",
                f"{inc_stat(scheme, SPREAD_INCREMENT, 'mean'):+.0f}",
            ]
        )
    story.append(
        _table(
            inc_rows,
            [1.95 * inch, 0.7 * inch, 0.75 * inch, 0.6 * inch, 1.0 * inch, 0.65 * inch],
            styles,
            alignments=["left", "right", "right", "right", "center", "right"],
            font_size=6.8,
        )
    )
    story.append(
        _p(
            "Note: The rolling ranking's own level correlation with future alpha is "
            f"{rolling_level:+.3f}. &Delta; rank corr is the mean per-window Spearman "
            "increment (rolling minus benchmark); HAC p is the one-sided test that the "
            "increment exceeds zero; &Delta; spread is the mean top-minus-bottom "
            "spread increment in annualised bps. Benchmarks are ordered from naive "
            "(characteristic gradients) to strong (expanding window; full-sample "
            "oracle). The oracle uses look-ahead information and is an upper bound, "
            "not a tradable strategy.",
            styles,
            "Note",
        )
    )
    _figure(
        story,
        "incremental_rolling_vs_static_expanding_window.png",
        "Figure 6. Cumulative rolling-minus-expanding-window increment in rank "
        "correlation (top) and quintile spread (bottom). A flat or downward path shows "
        "the time-varying ranking does not out-predict a ranking that simply "
        "accumulates all past data.",
        styles,
        width=5.0 * inch,
        height=3.7 * inch,
    )
    story.append(
        _p(
            "The interpretation is unambiguous. The out-of-sample predictability "
            "documented in Section 4.5 is real but reflects a stable, irregular "
            "pattern of characteristic mispricing that a fixed ranking captures at "
            "least as well; discarding data older than the rolling window and chasing "
            "recent alpha adds estimation noise, not forecasting skill. Claims of "
            "<i>dynamic</i> forecasting ability are not supported by the data.",
            styles,
        )
    )

    # ===================== 5. Robustness and validation =====================
    _section(story, "5.", "Robustness and validation", styles)
    _subsection(story, "5.1", "Benchmark sensitivity: FF3 versus FF5", styles)
    story.append(
        _p(
            "On the prespecified common sample (1963:07&ndash;2025:11), FF3 and FF5 "
            f"posterior ranks have Spearman correlation {ff_rank_rho:.3f}. "
            f"{max_mover_sentence} FDR-supported "
            f"raw alphas move from {ff3_fdr_count} under FF3 to {ff5_fdr_count} under "
            "FF5, and the HAC/Wald test rejects joint zero alpha under both FF3 "
            f"(p = {format_p_value(ff3_hac['p_value'], scientific=True)}) and FF5 "
            f"(p = {format_p_value(ff5_hac['p_value'], scientific=True)}). The "
            "11-rank maximum movement demonstrates that alpha is benchmark-dependent: "
            "part of what looks like FF3 performance is exposure to dimensions absent "
            "from FF3.",
            styles,
        )
    )
    _subsection(story, "5.2", "Window-length sensitivity", styles)
    story.append(Paragraph("Table 8. Rolling-window-length sensitivity of the ranking.", styles["TableCaption"]))
    robust_rows = [["Windows compared", "Rank corr", "Score corr", "Top Jaccard", "Bottom Jaccard"]]
    for row in robustness.itertuples(index=False):
        robust_rows.append(
            [
                f"{int(row.window_left)}m vs {int(row.window_right)}m",
                f"{row.rank_correlation:.3f}",
                f"{row.score_correlation:.3f}",
                f"{row.top_quintile_jaccard:.3f}",
                f"{row.bottom_quintile_jaccard:.3f}",
            ]
        )
    story.append(
        _table(
            robust_rows,
            [1.4 * inch, 0.95 * inch, 0.95 * inch, 1.0 * inch, 1.0 * inch],
            styles,
            alignments=["left", "right", "right", "right", "right"],
        )
    )
    story.append(
        _p(
            "Rank correlations across 60-, 120-, and 180-month windows are positive "
            "but far from one, with top-quintile membership overlap (Jaccard) that can "
            "fall below 0.40. Window length is not a neutral tuning knob; disagreement "
            "reflects genuine temporal instability.",
            styles,
        )
    )
    _subsection(story, "5.3", "Residual diagnostics and model risk", styles)
    story.append(
        _p(
            f"Residual normality is rejected for {normal_reject} of 25 portfolios "
            f"(Jarque&ndash;Bera), serial independence for {serial_reject} of 25 "
            f"(Ljung&ndash;Box, lag 12), and conditional homoskedasticity for "
            f"{arch_reject} of 25 (ARCH-LM, lag 12). An IID-Gaussian error model is "
            "not credible for these returns, which is why the paper relies on HAC "
            "inference and block resampling and treats GRS as secondary. These "
            "estimators protect parts of the inference against dependence; they do "
            "not repair omitted nonlinearities, time-varying betas, or benchmark "
            "incompleteness.",
            styles,
        )
    )
    story.append(
        _p(
            "The historical 1963&ndash;1991 anchor sharpens the point: HAC/Wald "
            f"rejects (p = {format_p_value(historical_tests.loc['HAC_Wald', 'p_value'], scientific=True)}) "
            f"while GRS does not (p = {format_p_value(historical_tests.loc['GRS', 'p_value'])}), "
            "consistent with the diagnostic evidence against IID-normal residuals.",
            styles,
        )
    )
    _subsection(story, "5.4", "Independent replication and reference validation", styles)
    story.append(
        _p(
            f"All coefficients and HAC standard errors reproduce independently with "
            f"statsmodels to a tolerance of 1e-12 for {replication_passes} of 25 "
            "portfolios, and the qualitative 5&times;5 size/value patterns, "
            "excess-return identity, block selection, and weighting checks all pass. "
            "Independent numerical replication is distinct from source reconciliation: "
            "the two current-snapshot comparisons remain NOT_COMPARABLE (Section 2.2) "
            "because the local acquisition vintage is unverified, and no external "
            "comparison whose values differ is reported as a pass.",
            styles,
        )
    )
    _subsection(story, "5.5", "Stochastic frontier analysis as a narrow diagnostic", styles)
    story.append(
        _p(
            "A half-normal stochastic-frontier model (&epsilon; = v &minus; u, "
            "u &ge; 0) is applied only as a residual-asymmetry diagnostic. Because a "
            "free intercept and a one-sided shortfall are not separately identified, "
            "SFA is not used as a substitute for alpha. The boundary null "
            "&sigma;<sub>u</sub> = 0 is tested with the appropriate 50:50 "
            "&chi;<sup>2</sup> mixture and BH-corrected across the 25 portfolios "
            f"(Jondrow et al., 1982). Only {len(supported_sfa)} of 25 boundary tests "
            "survives 5% FDR (ME5 BM4), so no general efficiency ranking is warranted.",
            styles,
        )
    )

    # ===================== 6. Discussion =====================
    _section(story, "6.", "Discussion", styles)
    story.append(
        _p(
            "Table 9 summarises what the evidence does and does not support. The "
            "central message is a distinction the raw forward test obscures: a "
            "<i>stable</i> cross-sectional ordering of the 25 cells carries "
            "out-of-sample information, but the <i>time variation</i> of the rolling "
            "ranking does not. The most defensible use of the framework is "
            "comparative and uncertainty-aware&mdash;estimate benchmark-relative "
            "performance, shrink noisy extremes, and report rank uncertainty. The "
            "least defensible use is to select a single portfolio from the point rank "
            "and treat its historical posterior alpha, or the rolling ranking's "
            "apparent timing ability, as a guaranteed future excess return.",
            styles,
        )
    )
    story.append(Paragraph("Table 9. What the evidence supports.", styles["TableCaption"]))
    synthesis = [
        ["Statement", "Assessment", "Basis"],
        ["Some FF3 alphas differ from zero", "Supported", "FDR evidence and joint HAC/Wald rejection"],
        ["The point rank is a precise hierarchy", "Not supported", "Wide bootstrap rank intervals; strong mobility"],
        ["A stable ordering has out-of-sample information", "Supported", "Positive forward spread; expanding-window and oracle levels"],
        ["The time-varying (rolling) ranking forecasts", "Not supported", "No increment over expanding window; dominated by oracle"],
        ["The forward relation is stable through time", "Not supported", "Large IQR; negative windows; decade heterogeneity"],
        ["Alpha is independent of benchmark choice", "Not supported", "11-rank FF3-to-FF5 movement"],
        ["SFA yields a full efficiency ranking", "Not supported", "Only 1 of 25 boundary tests survives FDR"],
        ["The implementation reproduces independently", "Supported", "OLS/HAC match statsmodels to 1e-12"],
        ["Local data equal the latest official snapshot", "Not supported", "Two NOT_COMPARABLE comparisons; vintage unverified"],
    ]
    story.append(
        _table(
            synthesis,
            [2.55 * inch, 1.15 * inch, 2.05 * inch],
            styles,
            alignments=["left", "left", "left"],
            font_size=7.0,
        )
    )
    _subsection(story, "6.1", "Limitations", styles)
    for text in [
        "Returns exclude transaction costs, taxes, shorting frictions, capacity, and market impact; the top-minus-bottom construction shorts small-growth, which is costly and capacity-constrained in practice.",
        "The source portfolios are value-weighted research constructions reconstituted by a published methodology, not live investable funds; full-sample alphas embed later data revisions.",
        "The factor model is linear with constant betas within each window; time-varying exposures, nonlinear payoffs, and omitted priced factors can remain in residuals.",
        "Posterior intervals do not fully integrate hyperparameter uncertainty, and the full HAC covariance is treated as known in the shrinkage step, so reported intervals are mildly optimistic.",
        "The oracle benchmark in Table 7 uses look-ahead information and is an upper bound; the expanding-window benchmark is the leakage-free comparator on which the negative incremental verdict rests.",
        "The local input vintage cannot currently be reconciled with the public source (Section 2.2).",
    ]:
        story.append(_bullet(text, styles))

    # ===================== 7. Conclusion =====================
    _section(story, "7.", "Conclusion", styles)
    story.append(
        _p(
            "This paper provides an uncertainty-aware benchmark for the 25 "
            "Fama&ndash;French portfolios built on a validated panel, a full joint "
            "HAC covariance, multivariate empirical-Bayes shrinkage, multiplicity "
            "control, overlap-aware persistence analysis, and strictly forward "
            "validation. Cross-sectional FF3 alpha variation is present and jointly "
            "significant, but its apparent precision is substantially reduced by "
            "partial pooling, bootstrap rank uncertainty, temporal mobility, and "
            "benchmark sensitivity.",
            styles,
        )
    )
    story.append(
        _p(
            "The paper's main contribution is the incremental forward test. It shows "
            "that the positive out-of-sample quintile spread is driven by persistent "
            "characteristic mispricing captured by a <i>stable</i> ordering, not by "
            "dynamic skill in the re-estimated ranking: the rolling ranking adds no "
            "predictive power over a leakage-free expanding-window ranking and is "
            "dominated by a fixed full-sample ordering. Future work should test "
            "whether any residual signal survives realistic turnover and trading "
            "costs, whether additional prespecified factors absorb the FF3 residual "
            "alphas without specification search, and whether a decision-theoretic "
            "allocation over rank probabilities improves on hard top-quintile "
            "selection. Any real-time claim should also await reconciliation of the "
            "data vintage with the official source.",
            styles,
        )
    )

    # ===================== References =====================
    _section(story, "", "References", styles)
    references = [
        "Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. <i>Journal of the Royal Statistical Society B</i>, 57(1), 289&ndash;300.",
        "Efron, B., and Morris, C. (1973). Stein's estimation rule and its competitors&mdash;an empirical Bayes approach. <i>Journal of the American Statistical Association</i>, 68(341), 117&ndash;130.",
        "Fama, E. F., and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. <i>Journal of Financial Economics</i>, 33(1), 3&ndash;56.",
        "Fama, E. F., and French, K. R. (2015). A five-factor asset pricing model. <i>Journal of Financial Economics</i>, 116(1), 1&ndash;22.",
        "Gibbons, M. R., Ross, S. A., and Shanken, J. (1989). A test of the efficiency of a given portfolio. <i>Econometrica</i>, 57(5), 1121&ndash;1152.",
        "Jondrow, J., Lovell, C. A. K., Materov, I. S., and Schmidt, P. (1982). On the estimation of technical inefficiency in the stochastic frontier production function model. <i>Journal of Econometrics</i>, 19(2&ndash;3), 233&ndash;238.",
        "Newey, W. K., and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. <i>Econometrica</i>, 55(3), 703&ndash;708.",
        "Politis, D. N., and Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In <i>Exploring the Limits of Bootstrap</i>, Wiley, 263&ndash;270.",
        "Kenneth R. French Data Library (2026). 25 portfolios formed on size and book-to-market; Fama&ndash;French research factors. Official URLs, access date, hashes, units, and coverage are recorded in results/tables/official_reference_manifest.csv.",
    ]
    for reference in references:
        story.append(_p(reference, styles, "Reference"))

    # ===================== Appendix A =====================
    _section(story, "Appendix A.", "Reproducibility", styles)
    story.append(
        _p(
            "The reviewed environment is CPython 3.13; the deterministic bootstrap "
            f"uses seed 2026. The recorded end-to-end runtime is "
            f"{run_manifest['summary']['runtime_seconds']:.1f} seconds. Source hashes, "
            "settings, and runtime are persisted in results/run_manifest.json so "
            "another analyst can identify exactly what was run. The full pipeline is:",
            styles,
        )
    )
    story.append(
        _p(
            "python -m pip install -r requirements-dev-lock.txt<br/>"
            "pytest -q &nbsp;&middot;&nbsp; ruff check .<br/>"
            "MPLBACKEND=Agg python -m analysis.run_all<br/>"
            "python -m analysis.export_tables<br/>"
            "python reports/generate_report.py",
            styles,
            "Equation",
        )
    )
    story.append(
        _p(
            "The incremental forward test (Section 4.6) is produced within "
            "analysis.run_all and can be regenerated independently with "
            "<i>python -m analysis.incremental_validation --scheme all</i>, which "
            "writes incremental_validation_summary_*.csv, "
            "incremental_validation_windows_*.csv, benchmark_ranking_*.csv, and the "
            "corresponding figures.",
            styles,
        )
    )

    # ===================== Appendix B =====================
    _section(story, "Appendix B.", "Notation", styles)
    notation = [
        ["Symbol", "Definition", "Symbol", "Definition"],
        ["y_it", "Excess return r_it - r_ft", "V_HAC", "Full 25x25 alpha covariance"],
        ["f_t", "Factor return vector", "mu, tau", "EB prior mean and dispersion"],
        ["alpha_i", "FF3 regression intercept", "L", "HAC lag length (L = 12)"],
        ["beta_i", "Factor loadings", "W", "HAC/Wald joint statistic"],
        ["eps_it", "Regression residual", "A_i^+", "Frozen-beta forward alpha"],
    ]
    story.append(
        _table(
            notation,
            [0.65 * inch, 2.0 * inch, 0.7 * inch, 2.0 * inch],
            styles,
            alignments=["left", "left", "left", "left"],
            font_size=7.2,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        _p(
            "All numerical claims in this paper are generated from the canonical "
            "result tables listed in Appendix A.",
            styles,
            "Note",
        )
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output


if __name__ == "__main__":
    generated = build_report()
    label = (
        str(generated.relative_to(ROOT))
        if generated.is_relative_to(ROOT)
        else str(generated)
    )
    print(f"Wrote {label}")
