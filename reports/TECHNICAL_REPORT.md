# Uncertainty-Aware Risk-Adjusted Portfolio Benchmarking

## Technical summary

This project benchmarks the 25 Fama-French size/book-to-market portfolios with
a full cross-portfolio HAC alpha covariance matrix, multivariate
empirical-Bayes shrinkage, 5,000-draw common-date bootstrap rank uncertainty,
and strictly forward validation. It is a historical research benchmark, not an
investment signal.

The validated local panel contains 29,825 unique portfolio-month observations:
25 portfolios by 1,193 months from July 1926 through November 2025. The primary
HAC/Wald joint-alpha test rejects at p = 6.42e-10; conventional GRS also rejects
(p = 1.16e-07) but is secondary because its IID-normal assumptions conflict
with the residual diagnostics.

`SMALL HiBM` has the highest posterior alpha at 132 annualized basis points.
2 posterior intervals are wholly above zero: `SMALL HiBM`, `BIG LoBM`.
Bootstrap rank intervals are wide. Rank interval endpoints change by no more than one rank between the
1,000-draw checkpoint and the final 5,000-draw run. Alpha tail endpoints are
less stable, with a worst shift of about 27 annualized basis points.

Strictly forward validation freezes scores and factor loadings at each
training-window cutoff. Across 89 evaluation windows, the Fisher-averaged rank
correlation is 0.102 (HAC 95% interval 0.055 to 0.149). The average top-quintile
portfolio alpha minus the average bottom-quintile portfolio alpha is 243
bps/year (HAC 95% interval 153 to 333), with median 209 bps/year and a positive
spread in 73.0% of windows. Decade results are heterogeneous.
The displayed HAC p-values are `2.32e-05` for the rank correlation and
`1.30e-07` for the quintile spread; no p-value is rounded to zero.

## External and reference validation

The fixed local inputs have a sample ending November 2025; exact acquisition
vintage is unverified. The current May 2026 official snapshot differs over the
overlap. The maximum differences are 1.4167 percentage points for portfolio
returns and 0.340 percentage points for FF3/RF fields. Official historical
revisions are a plausible cause, but the discrepancy cannot be attributed
uniquely without the exact archived release and verified acquisition
provenance. Both exact-overlap checks are `NOT_COMPARABLE`, not `PASS`: values
differ, but the unknown local acquisition vintage prevents a valid exact-
equality acceptance test. All pipeline-integrity checks pass. The local inputs
remain immutable and are identified by hash.

The largest portfolio-return difference is for `SMALL LoBM` in September 2024;
478 of 29,825 cells (1.60%) differ beyond tolerance. The largest factor
difference is for `HML` in October 2025; 50 of 4,772 factor-month cells (1.05%)
differ. Both discrepancy counts are concentrated in the 2020s, with the full
three-decade summaries retained in `external_validation_checks.csv`.

All coefficients and Bartlett-HAC standard errors for all 25 portfolios match
an independent statsmodels implementation to the 1e-12 tolerance. The official
portfolio block is explicitly value weighted; labels/order, decimal conversion,
and excess-return construction pass. Ten qualitative size/value pattern checks
also pass.

On the prespecified July 1963 common sample, FF3 and FF5 posterior ranks have
Spearman correlation 0.875. `ME5 BM2` moves from rank 9 under FF3 to rank 20
under FF5, an 11-position change. HAC/Wald rejects
joint zero alpha under both FF3 (p = 1.63e-08) and FF5 (p = 6.67e-07). On the
July 1963-December 1991 historical anchor, HAC/Wald rejects (p = 0.0003) while
GRS does not at 5% (p = 0.0821).

## SFA diagnostic

The 25 half-normal boundary-mixture p-values are adjusted together with
Benjamini-Hochberg. Two are nominally below 5%, but only `ME5 BM4` survives 5%
FDR (raw p = 0.000254; q = 0.006361). Only that portfolio receives an SFA rank.
The result is evidence of residual asymmetry under the fitted model, not proof
of managerial inefficiency.

## Reproducibility

Canonical settings are FF3, 12 HAC lags, 120-month rolling windows stepped by
12 months, 12-month forward horizons, and 5,000 bootstrap draws with 12-month
blocks and seed 2026. Source hashes and configuration are in the dataset and run
manifests. Official snapshot URLs, hashes, units, weighting, and coverage are in
`results/tables/official_reference_manifest.csv`.

The source-generated PDF is
`output/pdf/latent-performance-benchmarking-technical-report.pdf`.
