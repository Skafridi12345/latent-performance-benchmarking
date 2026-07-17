# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] - 2026-07-16

### Fixed

- Applied BH false-discovery control jointly across all 25 SFA boundary tests;
  only FDR-supported fits can receive an SFA rank.
- Replaced diagonal cross-sectional shrinkage with a full joint-HAC covariance
  and multivariate empirical-Bayes posterior.
- Made forward-spread labels explicit and added leakage, decade,
  leave-one-decade-out, and extreme-window diagnostics.
- Reframed window sensitivity as cross-window estimator agreement and centered
  transitions on the 0.20 uniform-mixing baseline.

### Added

- Dependence-robust HAC/Wald and conventional GRS joint-alpha tests.
- A 5,000-draw bootstrap with saved 1,000-versus-final stability diagnostics.
- Pinned official May 2026 FF3, FF5, and 25-portfolio snapshots with hashes,
  source metadata, and exact current-snapshot comparisons.
- Independent statsmodels OLS/HAC replication, qualitative 5x5 checks,
  July 1963-December 1991 historical validation, and FF5 sensitivity.
- Audit-level methodology and data-dictionary documentation.

### Removed

- Legacy root entry points, compatibility run wrappers, and the superseded
  root-level PDF alias. The canonical pipeline is `python -m analysis.run_all`.

## [0.4.0] - 2026-07-16

### Fixed

- Removed a many-to-many factor merge that inflated 29,825 unique
  portfolio-months to 243,950 rows and invalidated the legacy outputs.
- Replaced the unidentified intercept-plus-persistent-inefficiency ranking with
  one primary estimand: HAC factor alpha with empirical-Bayes shrinkage.
- Made SFA conditional scores invariant to return units and corrected the
  conditional exponential-moment calculation.
- Added a one-sided boundary likelihood-ratio test and suppress unsupported SFA
  rankings.
- Labelled rolling-window overlap explicitly and restricted structural
  persistence interpretation to non-overlapping windows.
- Added strictly forward validation with frozen training betas and HAC aggregate
  uncertainty.
- Removed the invalid pre-rebuild outputs and superseded report.

### Added

- Strict panel validation and SHA-256 source manifests.
- Common-date circular block-bootstrap score and rank intervals.
- Jarque-Bera, Durbin-Watson, Ljung-Box, and ARCH-LM residual diagnostics.
- Numerical gradient, simulated-moment, unit-invariance, no-look-ahead, and
  bootstrap-reproducibility tests.
- Exact runtime and development dependency locks.
- A source-generated technical report.

## [0.3.0] - 2026-05-25

### Added

- Added pytest suite with synthetic fixtures for loaders, factor construction, SFA contracts, rolling windows, rankings, transitions, mobility, alpha comparison, and pipeline smoke testing.
- Added `analysis/benchmark.py` for compact runtime benchmarking.
- Added a testable `PipelineConfig` and `run_pipeline` API around the canonical pipeline.
- Added packaging metadata for editable installs with development extras.

### Changed

- Replaced pipeline progress prints with logging while keeping a concise final console summary.
- Standardised the canonical run command as `python -m analysis.run_all`.
- Converted legacy analysis run scripts into compatibility wrappers.
- Updated LaTeX export code to use the new `results/tables/` outputs and avoid import-time side effects.
- Applied ruff formatting and lint cleanup.

## [0.2.0] - 2026-05-25

### Added

- Added a reproducible analysis pipeline at `analysis/run_all.py`.
- Added explicit Fama-French monthly panel extraction and factor-model construction.
- Added static half-normal and truncated-normal SFA implementations with log-likelihood, AIC, BIC, lambda, residuals, fitted values, convergence metadata, and adjusted efficiency outputs.
- Added traditional factor-alpha baseline estimation and alpha-vs-AE rank comparison.
- Added rolling SFA outputs with window metadata, ranks, quintiles, convergence status, likelihood diagnostics, AIC, and BIC.
- Added persistence, transition, mobility, robustness, model-comparison, and residual-diagnostic outputs.
- Added GitHub-readable figures under `results/figures/`.

### Changed

- Updated `main_minimal.py` to delegate to the reproducible pipeline.
- Updated the README to reference the new analytical outputs.
- Replaced duplicated top-level SFA implementations with compatibility wrappers around the package implementations.

### Notes

- The technical PDF and legacy outputs were retained.
- The default pipeline uses an annual rolling step for practical runtime; exact monthly persistence horizons can be generated with `--rolling-step 1`.

## [0.1.0] - 2026-05-25

### Added

- Added professional repository documentation for the initial presentation pass.
- Added project citation metadata.
- Added MIT license.
- Added minimal Python project metadata and tool configuration.

### Changed

- Reworked the README for clearer quant research positioning, methodology context, reproducible workflow instructions, output references, limitations, and future extensions.
- Expanded `.gitignore` for Python development, virtual environments, caches, build artifacts, operating-system files, notebooks, local environment files, and temporary outputs.

### Notes

- No major modelling or analytical-engine changes were made in this pass.
- Existing figures, CSV outputs, LaTeX tables, and the technical PDF were retained.
