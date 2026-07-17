# Data provenance

The analysis uses monthly US portfolio returns and Fama–French research factors.
Both local files are treated as immutable inputs: the pipeline records their
SHA-256 hashes, dimensions, date coverage, and validation results in
`results/tables/dataset_manifest.csv` on every run. Current official snapshots
are pinned separately under `data/reference/official/`; they never replace the
local analysis inputs.

## 25 size–book-to-market portfolios

- Local file: `data/25_size_bm_portfolios.csv`
- Official source: [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- Portfolio construction: [25 Portfolios Formed on Size and Book-to-Market](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/tw_5_ports.html)
- Local SHA-256 at review: `05b45903b4ece361e80733136fc6e13a6e475a9c18b096d86345be458f210061`
- Local file size at review: 1,521,315 bytes

## Fama–French three factors

- Local file: `data/ff3_factors.csv`
- Official source: [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- Factor archive and documentation: [Fama/French 3 Factors archive](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors_archive.html)
- Local SHA-256 at review: `3a412f4bb89d740af47ba331ca9f9b7d9fa0657cf22fa3c5f142d0db8ce68c70`
- Local file size at review: 37,218 bytes

## Validation rules

The loader fails rather than silently proceeding if it finds duplicate
portfolio-month keys, null required values, factor values that disagree across
portfolios in the same month, an unbalanced panel, or missing calendar months.
The review copy covers July 1926 through November 2025: 1,193 months and 25
portfolios, giving exactly 29,825 portfolio-month observations.

## Official comparison snapshots

The official May 2026 downloads were accessed on 16 July 2026. Byte hashes,
URLs, units, weighting, and coverage are written to
`results/tables/official_reference_manifest.csv`. The analysis uses fixed local
inputs with a sample ending November 2025; exact acquisition vintage is
unverified. The current official snapshot differs from those inputs over the
overlap. Official historical revisions are a plausible cause, but the
discrepancy cannot be attributed uniquely without the exact archived release
and verified acquisition provenance. Exact common-sample differences remain
visible as `NOT_COMPARABLE` checks in
`results/tables/external_validation_checks.csv`; the local inputs remain
immutable and are identified by hash.
