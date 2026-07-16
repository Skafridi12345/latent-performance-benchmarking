# Data provenance

The analysis uses monthly US portfolio returns and Fama–French research factors.
Both local files are treated as immutable inputs: the pipeline records their
SHA-256 hashes, dimensions, date coverage, and validation results in
`results/tables/dataset_manifest.json` on every run.

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
- Local SHA-256 at review: `3a412f4bb89d740af47ba33136fc6e13a6e475a9c18b096d86345be458f210061`
- Local file size at review: 37,218 bytes

## Validation rules

The loader fails rather than silently proceeding if it finds duplicate
portfolio-month keys, null required values, factor values that disagree across
portfolios in the same month, an unbalanced panel, or missing calendar months.
The review copy covers July 1926 through November 2025: 1,193 months and 25
portfolios, giving exactly 29,825 portfolio-month observations.
