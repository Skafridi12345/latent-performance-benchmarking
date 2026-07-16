from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
OUTPUT = (
    ROOT / "output" / "pdf" / "latent-performance-benchmarking-technical-report.pdf"
)
ROOT_ALIAS = (
    ROOT
    / "Risk-Adjusted Portfolio Benchmarking via Latent Performance Decomposition.pdf"
)

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F6F9F")
TEAL = colors.HexColor("#2A9D8F")
RED = colors.HexColor("#B54A4A")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GREY = colors.HexColor("#F3F5F7")
MID_GREY = colors.HexColor("#66717E")
TEXT = colors.HexColor("#1E2933")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=19,
            textColor=BLUE,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCompact",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=TEXT,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MID_GREY,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=15,
            textColor=NAVY,
            borderColor=BLUE,
            borderWidth=0.8,
            borderPadding=10,
            backColor=LIGHT_BLUE,
            spaceBefore=8,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Equation",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=12,
            textColor=TEXT,
            borderColor=colors.HexColor("#D9E1E8"),
            borderWidth=0.5,
            borderPadding=7,
            backColor=LIGHT_GREY,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=5,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=TEXT,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableTextLeft",
            parent=styles["TableText"],
            alignment=TA_LEFT,
        )
    )
    return styles


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D7DEE5"))
        canvas.setLineWidth(0.5)
        canvas.line(
            doc.leftMargin, height - 16 * mm, width - doc.rightMargin, height - 16 * mm
        )
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MID_GREY)
        canvas.drawString(
            doc.leftMargin, height - 12.5 * mm, "LATENT PERFORMANCE BENCHMARKING"
        )
        right = "QUANTITATIVE REVIEW - JULY 2026"
        canvas.drawRightString(width - doc.rightMargin, height - 12.5 * mm, right)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    footer = "Research diagnostics - not investment advice"
    canvas.drawString(doc.leftMargin, 10 * mm, footer)
    page = str(doc.page)
    canvas.drawRightString(width - doc.rightMargin, 10 * mm, page)
    canvas.restoreState()


def _image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _p(text: str, styles, style: str = "BodyCompact") -> Paragraph:
    return Paragraph(text, styles[style])


def _bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"- {text}", styles["BodyCompact"])


def _table(data, widths, styles, *, header=True, alignments=None) -> Table:
    formatted = []
    for row_idx, row in enumerate(data):
        cells = []
        for col_idx, value in enumerate(row):
            align = (
                "TableTextLeft"
                if alignments and alignments[col_idx] == "left"
                else "TableText"
            )
            cell_style = styles[align]
            if row_idx == 0 and header:
                cell_style = ParagraphStyle(
                    f"header-{col_idx}-{id(data)}",
                    parent=cell_style,
                    fontName="Helvetica-Bold",
                    textColor=colors.white,
                )
            cells.append(Paragraph(str(value), cell_style))
        formatted.append(cells)
    table = Table(
        formatted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT"
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCD5DD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row in range(1, len(formatted)):
        commands.append(
            ("BACKGROUND", (0, row), (-1, row), colors.white if row % 2 else LIGHT_GREY)
        )
    table.setStyle(TableStyle(commands))
    return table


def _section_title(number: str, title: str, styles):
    return [
        Paragraph(f"{number}  {title}", styles["Section"]),
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10),
    ]


def build_report(output: Path = OUTPUT) -> Path:
    performance = pd.read_csv(TABLES / "performance_scores.csv")
    prior = pd.read_csv(TABLES / "performance_prior.csv").iloc[0]
    manifest = pd.read_csv(TABLES / "dataset_manifest.csv").iloc[0]
    persistence = pd.read_csv(TABLES / "rank_persistence.csv")
    forward = pd.read_csv(TABLES / "forward_performance_aggregate.csv").iloc[0]
    residual_diag = pd.read_csv(TABLES / "performance_residual_diagnostics.csv")
    sfa = pd.read_csv(TABLES / "sfa_asymmetry_diagnostics.csv")
    robustness = pd.read_csv(TABLES / "robustness_summary.csv")

    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="Risk-Adjusted Portfolio Benchmarking",
        author="Dr. Muhammad Shoaib",
        subject="Quantitative review and rebuilt methodology",
    )
    story = []

    # Cover
    story.extend(
        [
            Spacer(1, 20 * mm),
            Paragraph("RISK-ADJUSTED PORTFOLIO BENCHMARKING", styles["CoverTitle"]),
            Paragraph(
                "Quantitative review and rebuilt methodology", styles["CoverSub"]
            ),
            Spacer(1, 4 * mm),
            HRFlowable(width="34%", thickness=3, color=TEAL, hAlign="LEFT"),
            Spacer(1, 13 * mm),
            Paragraph(
                "A critical rebuild of the latent-performance benchmark for the 25 Fama-French size and book-to-market portfolios.",
                styles["Callout"],
            ),
            Spacer(1, 11 * mm),
            _p("Prepared by Dr. Muhammad Shoaib", styles, "CoverSub"),
            _p("Review date: 16 July 2026", styles),
            _p("Sample: July 1926 to November 2025", styles),
            _p("Status: validated research implementation", styles),
            Spacer(1, 22 * mm),
            _p(
                "This report supersedes the pre-rebuild document. The invalid old report and outputs have been removed and must not be cited.",
                styles,
                "Small",
            ),
        ]
    )
    story.append(PageBreak())

    # Executive summary
    story.extend(_section_title("1", "Executive conclusion", styles))
    story.append(
        _p(
            "The rebuild changes both the method and the strength of the claims. The primary score is now Fama-French three-factor alpha with 12-lag Newey-West inference and cross-sectional empirical-Bayes shrinkage. Stochastic frontier analysis (SFA) is retained only as a secondary test of one-sided residual asymmetry.",
            styles,
        )
    )
    story.append(
        _p(
            f"The validated panel has <b>{int(manifest['n_observations']):,}</b> unique observations: {int(manifest['n_portfolios'])} portfolios by {int(manifest['n_months']):,} months. No duplicates, missing calendar months, factor disagreement, or panel imbalance remain.",
            styles,
            "Callout",
        )
    )
    top = performance.iloc[0]
    only_positive = performance.loc[
        performance["posterior_alpha_ci_low"] > 0, "portfolio"
    ].tolist()
    story.append(
        _bullet(
            f"{top['portfolio']} ranks first at {top['posterior_alpha_annualized_bps']:.0f} posterior bps/year, but its interval includes zero. Only {', '.join(only_positive)} has a posterior interval wholly above zero.",
            styles,
        )
    )
    story.append(
        _bullet(
            "Point ranks are much more precise-looking than the evidence. Common-date block-bootstrap rank intervals are wide for most portfolios.",
            styles,
        )
    )
    story.append(
        _bullet(
            f"Forward validation produces a modest mean rank correlation of {forward['mean_rank_spearman_fisher']:.3f} and a top-minus-bottom future alpha spread of {forward['mean_top_minus_bottom_forward_alpha_annualized_bps']:.0f} bps/year.",
            styles,
        )
    )
    supported = sfa[sfa["one_sided_component_supported"].astype(bool)]
    story.append(
        _bullet(
            f"Only {len(supported)} of {len(sfa)} portfolios support a one-sided SFA component at 5%; unsupported SFA ranks are suppressed.",
            styles,
        )
    )
    normal_reject = int((residual_diag["jarque_bera_p_value"] < 0.05).sum())
    serial_reject = int((residual_diag["ljung_box_lag_12_p_value"] < 0.05).sum())
    arch_reject = int((residual_diag["arch_lm_lag_12_p_value"] < 0.05).sum())
    story.append(
        _bullet(
            f"Residual diagnostics reject normality for {normal_reject}/25 portfolios, lag-12 independence for {serial_reject}/25, and no ARCH effects for {arch_reject}/25.",
            styles,
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        _p(
            "Decision implication: use the posterior scores as uncertain historical benchmark diagnostics. Do not interpret the ranking as a stable skill hierarchy, and do not use SFA scores as a general efficiency league table.",
            styles,
            "Callout",
        )
    )
    story.append(PageBreak())

    # Data integrity
    story.extend(_section_title("2", "Data integrity and review corrections", styles))
    story.append(
        _p(
            "The pre-rebuild merged file contained 243,950 rows but only 29,825 unique portfolio-month keys. The implied inflation factor was 8.18, and downstream tables reported 9,758 observations per portfolio where only 1,193 months existed. That was a data error, not additional information.",
            styles,
        )
    )
    integrity_data = [
        ["Check", "Validated result"],
        ["Unique portfolio-months", f"{int(manifest['n_unique_portfolio_months']):,}"],
        ["Duplicate keys", str(int(manifest["duplicate_portfolio_months"]))],
        [
            "Portfolios x months",
            f"{int(manifest['n_portfolios'])} x {int(manifest['n_months']):,}",
        ],
        ["Balanced panel", str(bool(manifest["balanced_panel"]))],
        ["Missing calendar months", str(int(manifest["missing_calendar_months"]))],
        ["Sample", "July 1926 - November 2025"],
    ]
    story.append(
        _table(
            integrity_data,
            [2.5 * inch, 3.3 * inch],
            styles,
            alignments=["left", "left"],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        _p(
            "The loader now fails on duplicate keys, null required values, factor disagreement within a month, an unbalanced panel, or a gap in the calendar sequence. Every run records SHA-256 hashes and byte sizes for both source files.",
            styles,
        )
    )
    story.append(
        _p(
            f"Portfolio file SHA-256: {manifest['portfolio_source_sha256']}",
            styles,
            "Small",
        )
    )
    story.append(
        _p(f"Factor file SHA-256: {manifest['factor_source_sha256']}", styles, "Small")
    )
    story.append(
        _p(
            "Source: Kenneth R. French Data Library, with the official 25 size/book-to-market portfolio construction documentation. The local review files are treated as immutable inputs and identified by hash.",
            styles,
        )
    )
    story.append(PageBreak())

    # Method
    story.extend(_section_title("3", "Primary estimand and uncertainty", styles))
    story.append(
        _p("For portfolio i and month t, the benchmark regression is:", styles)
    )
    story.append(
        _p("r_it - r_ft = alpha_i + beta_i' f_t + epsilon_it", styles, "Equation")
    )
    story.append(
        _p(
            "The factors are market excess return, SMB, and HML. A 12-lag Bartlett-kernel Newey-West covariance matrix protects coefficient inference against heteroskedasticity and autocorrelation. Raw alpha p-values are adjusted across the 25 portfolios with the Benjamini-Hochberg false-discovery procedure.",
            styles,
        )
    )
    story.append(_p("Cross-sectional partial pooling uses:", styles))
    story.append(
        _p(
            "alpha_hat_i | alpha_i ~ Normal(alpha_i, se_i^2)<br/>alpha_i ~ Normal(mu, tau^2)<br/>posterior mean = w_i alpha_hat_i + (1 - w_i) mu<br/>w_i = tau^2 / (tau^2 + se_i^2)",
            styles,
            "Equation",
        )
    )
    story.append(Spacer(1, 3 * mm))
    prior_mean_bps = prior["mu"] * 12 * 10000
    prior_tau_bps = prior["tau"] * 12 * 10000
    story.append(
        _p(
            f"Profile marginal maximum likelihood estimates a prior mean of {prior_mean_bps:.0f} bps/year and a cross-sectional standard deviation of {prior_tau_bps:.0f} bps/year. Shrinkage is stronger where the portfolio-specific HAC standard error is larger.",
            styles,
            "Callout",
        )
    )
    story.append(
        _p(
            "This is the key identification repair. The model no longer asks a free portfolio intercept and a persistent non-negative term to explain the same constant shift. Persistent benchmark-relative performance is represented by one estimand: alpha.",
            styles,
        )
    )
    story.append(Paragraph("Rank and temporal validation", styles["Subsection"]))
    story.append(
        _p(
            "Rank intervals use 200 common-date circular block-bootstrap samples with 12-month blocks. Common sampled dates preserve dependence across portfolios. Rolling scores use 120-month training windows ending every 12 months. Persistence outputs state the exact overlap fraction. Forward validation freezes all training estimates, then evaluates the next 12 months only.",
            styles,
        )
    )
    story.append(Paragraph("Secondary SFA diagnostic", styles["Subsection"]))
    story.append(
        _p(
            "The half-normal model decomposes residuals as epsilon = v - u. The reported conditional score is E[exp(-u/sigma) | epsilon], with sigma equal to the total fitted residual scale. This is invariant to expressing returns in decimals or percent. A 50:50 boundary likelihood-ratio mixture test is required before a half-normal rank is assigned. Truncated-normal results are reported only as sensitivity evidence.",
            styles,
        )
    )
    story.append(PageBreak())

    # Performance results
    story.extend(_section_title("4", "Full-sample performance results", styles))
    rank_rows = [
        [
            "Rank",
            "Portfolio",
            "Posterior bps/year",
            "95% interval",
            "Bootstrap rank 95%",
        ]
    ]
    selection = pd.concat([performance.head(5), performance.tail(5)])
    for row in selection.itertuples(index=False):
        low = row.posterior_alpha_ci_low * 12 * 10000
        high = row.posterior_alpha_ci_high * 12 * 10000
        rank_rows.append(
            [
                int(row.performance_rank),
                row.portfolio,
                f"{row.posterior_alpha_annualized_bps:.0f}",
                f"[{low:.0f}, {high:.0f}]",
                f"[{row.bootstrap_rank_ci_low:.1f}, {row.bootstrap_rank_ci_high:.1f}]",
            ]
        )
    story.append(
        _table(
            rank_rows,
            [0.43 * inch, 1.05 * inch, 1.15 * inch, 1.05 * inch, 1.15 * inch],
            styles,
            alignments=["left", "left", "left", "left", "left"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    fdr_count = int((performance["alpha_fdr_q_value"] < 0.05).sum())
    story.append(
        _p(
            f"Five raw alpha tests survive 5% false-discovery control ({fdr_count}/25), but only one approximate posterior interval is wholly positive. The two statements answer different questions: FDR controls a family of raw-alpha tests, while posterior intervals reflect cross-sectional partial pooling.",
            styles,
        )
    )
    story.append(_image(FIGURES / "performance_ranking.png", 6.4 * inch, 5.9 * inch))
    story.append(
        _p(
            "Figure 1. Posterior annualised factor alpha with approximate 95% intervals.",
            styles,
            "Small",
        )
    )
    story.append(PageBreak())

    story.extend(_section_title("5", "Rank uncertainty", styles))
    story.append(
        _p(
            "Bootstrap uncertainty is the strongest warning against over-reading the ordered list. Even the leading portfolios can occupy a broad range of ranks when common historical blocks are resampled. The output therefore reports the probability of being in the top or bottom quintile as well as a point rank.",
            styles,
        )
    )
    story.append(_image(FIGURES / "rank_uncertainty.png", 6.4 * inch, 7.0 * inch))
    story.append(
        _p(
            "Figure 2. Common-date block-bootstrap median ranks and 95% intervals.",
            styles,
            "Small",
        )
    )
    story.append(PageBreak())

    # Persistence
    story.extend(
        _section_title("6", "Rolling persistence without the overlap illusion", styles)
    )
    grouped = persistence.groupby(
        ["horizon_months", "window_overlap_fraction", "structural_inference_eligible"],
        as_index=False,
    ).agg(
        mean_spearman=("spearman_rank_autocorrelation", "mean"),
        mean_score=("pearson_score_autocorrelation", "mean"),
        mean_rank_change=("average_absolute_rank_change", "mean"),
    )
    persistence_rows = [
        ["Horizon", "Window overlap", "Mean rank rho", "Mean score r", "Eligible"]
    ]
    for row in grouped.itertuples(index=False):
        persistence_rows.append(
            [
                f"{int(row.horizon_months)} months",
                f"{row.window_overlap_fraction:.0%}",
                f"{row.mean_spearman:.3f}",
                f"{row.mean_score:.3f}",
                "Yes" if row.structural_inference_eligible else "No",
            ]
        )
    story.append(
        _table(
            persistence_rows,
            [1.2 * inch, 1.15 * inch, 1.15 * inch, 1.05 * inch, 0.7 * inch],
            styles,
            alignments=["left", "left", "left", "left", "left"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        _p(
            "The old persistence interpretation confused estimator overlap with economic stability. At a 12-month horizon, adjacent 120-month windows share 90% of their observations. At 120 months, overlap is zero and average rank persistence is only about 0.13. Only this non-overlapping comparison is labelled structurally interpretable.",
            styles,
            "Callout",
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(_image(FIGURES / "rank_persistence.png", 6.3 * inch, 3.7 * inch))
    story.append(
        _p("Figure 3. Rank and score persistence by calendar horizon.", styles, "Small")
    )
    story.append(PageBreak())

    # Forward
    story.extend(_section_title("7", "Strictly forward validation", styles))
    forward_rows = [
        ["Metric", "Estimate", "HAC 95% interval", "HAC p-value"],
        [
            "Rank vs future alpha",
            f"{forward['mean_rank_spearman_fisher']:.3f}",
            f"[{forward['rank_spearman_ci_low']:.3f}, {forward['rank_spearman_ci_high']:.3f}]",
            f"{forward['rank_spearman_hac_p_value']:.4f}",
        ],
        [
            "Top - bottom future alpha",
            f"{forward['mean_top_minus_bottom_forward_alpha_annualized_bps']:.0f} bps/year",
            f"[{forward['top_minus_bottom_ci_low_bps']:.0f}, {forward['top_minus_bottom_ci_high_bps']:.0f}]",
            f"{forward['top_minus_bottom_hac_p_value']:.4f}",
        ],
    ]
    story.append(
        _table(
            forward_rows,
            [1.7 * inch, 1.25 * inch, 1.45 * inch, 0.9 * inch],
            styles,
            alignments=["left", "left", "left", "left"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        _p(
            f"There are {int(forward['n_validation_windows'])} validation windows. Every evaluation month is strictly after its training window. The future spread is positive in {forward['positive_top_minus_bottom_fraction']:.1%} of windows. The result supports modest predictive ordering on average, not a uniformly successful timing rule.",
            styles,
        )
    )
    story.append(
        _image(FIGURES / "forward_performance_validation.png", 6.4 * inch, 5.0 * inch)
    )
    story.append(
        _p(
            "Figure 4. Window-by-window future rank correlations and top-minus-bottom spreads.",
            styles,
            "Small",
        )
    )
    story.append(PageBreak())

    # Diagnostics
    story.extend(_section_title("8", "Model diagnostics and SFA evidence", styles))
    diagnostic_rows = [
        ["Diagnostic", "5% rejections", "Interpretation"],
        [
            "Jarque-Bera normality",
            f"{normal_reject}/25",
            "Heavy tails and/or asymmetry are widespread",
        ],
        [
            "Ljung-Box lag 12",
            f"{serial_reject}/25",
            "Serial dependence remains for many portfolios",
        ],
        ["ARCH LM lag 12", f"{arch_reject}/25", "Conditional variance is time-varying"],
    ]
    story.append(
        _table(
            diagnostic_rows,
            [1.35 * inch, 0.9 * inch, 3.25 * inch],
            styles,
            alignments=["left", "left", "left"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        _p(
            "Newey-West standard errors and block bootstraps address broad dependence in inference, but they do not make the Gaussian likelihood correct. The residual tests are therefore a substantive limitation, not a footnote.",
            styles,
        )
    )
    sfa_rows = [["Portfolio", "Boundary LR", "Mixture p", "Standardised AE"]]
    for row in supported.sort_values("boundary_mixture_p_value").itertuples(
        index=False
    ):
        sfa_rows.append(
            [
                row.portfolio,
                f"{row.boundary_lr_stat:.2f}",
                f"{row.boundary_mixture_p_value:.4f}",
                f"{row.AE:.3f}",
            ]
        )
    story.append(
        Paragraph(
            "Portfolios with supported one-sided residual components",
            styles["Subsection"],
        )
    )
    story.append(
        _table(
            sfa_rows,
            [1.3 * inch, 1.0 * inch, 1.0 * inch, 1.15 * inch],
            styles,
            alignments=["left", "left", "left", "left"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        _p(
            "The boundary evidence is sparse. Unsupported portfolios are reported under the symmetric Gaussian null with u = 0 and AE = 1, and no SFA rank is assigned. The two supported cases indicate negative residual asymmetry under this specification; they do not prove economic inefficiency.",
            styles,
            "Callout",
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        _image(FIGURES / "sfa_boundary_diagnostics.png", 6.2 * inch, 4.4 * inch)
    )
    story.append(
        _p(
            "Figure 5. Boundary-test evidence for a one-sided residual component.",
            styles,
            "Small",
        )
    )
    story.append(PageBreak())

    # Robustness and limits
    story.extend(_section_title("9", "Robustness and remaining limitations", styles))
    robustness_rows = [
        ["Windows compared", "Rank correlation", "Score correlation", "Top overlap"]
    ]
    for row in robustness.itertuples(index=False):
        robustness_rows.append(
            [
                f"{int(row.window_left)} vs {int(row.window_right)} months",
                f"{row.rank_correlation:.3f}",
                f"{row.score_correlation:.3f}",
                f"{row.top_quintile_jaccard:.3f}",
            ]
        )
    story.append(
        _table(
            robustness_rows,
            [1.55 * inch, 1.25 * inch, 1.25 * inch, 1.05 * inch],
            styles,
            alignments=["left", "left", "left", "left"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    limitations = [
        "The three-factor model is a benchmark, not a complete model. Omitted systematic risks can appear as alpha.",
        "Posterior intervals are empirical-Bayes approximations. Hyperparameter uncertainty is only approximated.",
        "The 25 constructed portfolios are cross-sectionally related. The common-date bootstrap preserves this dependence, but 200 replicates provide moderate tail precision.",
        "The forward evaluation periods do not overlap, but adjacent 120-month training histories do. HAC inference is used for the aggregate validation series.",
        "Full-sample ranks use all historical data and are descriptive. Only the forward-validation tables are out of sample.",
        "The SFA boundary test has limited power and depends on a restrictive distribution. The two supported cases are residual diagnostics, not managerial labels.",
        "Reported portfolio returns exclude implementation costs, taxes, capacity constraints, and real-time data revisions.",
    ]
    for item in limitations:
        story.append(_bullet(item, styles))
    story.append(Spacer(1, 4 * mm))
    story.append(
        _p(
            "Overall assessment: the rebuilt project is suitable as a transparent research benchmark with explicit uncertainty. It is not a production investment signal and should not be presented as one.",
            styles,
            "Callout",
        )
    )
    story.append(PageBreak())

    # Reproducibility
    story.extend(_section_title("10", "Reproducibility and references", styles))
    story.append(Paragraph("Reviewed execution", styles["Subsection"]))
    story.append(
        _p(
            "The codebase passes 25 tests and a full ruff lint check in a clean CPython 3.13 environment. Tests cover duplicate-key rejection, HAC recovery, shrinkage behaviour, SFA analytic gradients, a one-million-draw conditional-moment simulation, return-unit invariance, boundary suppression, overlap labelling, no-look-ahead dates, bootstrap reproducibility, and the end-to-end pipeline.",
            styles,
        )
    )
    story.append(
        _p(
            "python -m pip install -r requirements-dev-lock.txt<br/>pytest -q<br/>ruff check .<br/>MPLBACKEND=Agg python -m analysis.run_all<br/>python -m analysis.export_tables<br/>python reports/generate_report.py",
            styles,
            "Equation",
        )
    )
    story.append(Paragraph("Primary sources", styles["Subsection"]))
    references = [
        "Fama, E. F., and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3-56.",
        "Newey, W. K., and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708.",
        "Jondrow, J., Lovell, C. A. K., Materov, I. S., and Schmidt, P. (1982). On the estimation of technical inefficiency in the stochastic frontier production function model. Journal of Econometrics, 19(2-3), 233-238.",
        "Kenneth R. French Data Library. 25 portfolios formed on size and book-to-market and Fama-French research factors. Local review files identified by SHA-256 in data/SOURCES.md.",
    ]
    for reference in references:
        story.append(_bullet(reference, styles))
    story.append(Spacer(1, 5 * mm))
    story.append(_p("Canonical artefacts", styles, "Subsection"))
    artefacts = [
        ["Purpose", "Path"],
        ["Primary scores", "results/tables/performance_scores.csv"],
        ["Rank uncertainty", "results/tables/performance_rank_uncertainty.csv"],
        ["Forward aggregate", "results/tables/forward_performance_aggregate.csv"],
        [
            "Data and run manifests",
            "results/tables/dataset_manifest.csv; results/run_manifest.json",
        ],
    ]
    story.append(
        _table(artefacts, [1.7 * inch, 3.9 * inch], styles, alignments=["left", "left"])
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    shutil.copy2(output, ROOT_ALIAS)
    return output


if __name__ == "__main__":
    generated = build_report()
    label = (
        str(generated.relative_to(ROOT))
        if generated.is_relative_to(ROOT)
        else str(generated)
    )
    print(f"Wrote {label}")
