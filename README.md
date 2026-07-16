# Latent Performance Benchmarking

This project benchmarks the 25 Fama-French size and book-to-market portfolios
using factor-adjusted returns, dependence-robust inference, cross-sectional
shrinkage, rank uncertainty, and strictly forward validation.

The primary estimand is posterior factor alpha. Stochastic frontier analysis
(SFA) is retained only as a secondary residual-asymmetry diagnostic. It is not
used as the main performance ranking because a portfolio intercept and a
persistent non-negative shortfall are not separately identified without extra
structure.

The July 2026 rebuild also fixes a historical many-to-many merge that had
expanded 29,825 valid portfolio-months to 243,950 rows. All affected outputs and
the superseded report have been removed and must not be cited.

## Headline findings

The validated sample contains 25 portfolios and 1,193 monthly observations per
portfolio from July 1926 through November 2025.

- The empirical-Bayes prior mean is -47 annualised basis points (bps), with an
  estimated cross-sectional standard deviation of 129 bps.
- `SMALL HiBM` ranks first at +117 posterior bps/year, but its 95% posterior
  interval includes zero. `BIG LoBM`, ranked second at +101 bps/year, is the
  only portfolio whose posterior interval is wholly above zero.
- Five raw factor-alpha tests remain significant at a 5% Benjamini-Hochberg
  false-discovery rate, but bootstrap rank intervals remain wide. A point rank
  should therefore not be read as a precise league table.
- Rank persistence is mechanically high when estimation windows overlap. Mean
  Spearman persistence falls from 0.88 at a 12-month horizon with 90% overlap
  to 0.13 at a non-overlapping 120-month horizon.
- In 89 strictly forward 12-month validation windows, the Fisher-averaged rank
  correlation is 0.096 (HAC 95% CI 0.041 to 0.151). The top-minus-bottom future
  factor-alpha spread averages 172 bps/year (HAC 95% CI 63 to 280).
- Only 2 of 25 portfolios reject the no-one-sided-component boundary null at
  5%. SFA rankings are therefore suppressed for the other 23 portfolios.
- Factor residuals reject Gaussian normality for all 25 portfolios; 16 show
  significant lag-12 serial dependence and all 25 show ARCH effects. HAC and
  block-bootstrap uncertainty are essential, but do not solve model
  misspecification.

These are historical research diagnostics, not trading returns or investment
advice.

## Method

For portfolio `i` and month `t`, the primary model is

```text
r_it - r_ft = alpha_i + beta_i' f_t + epsilon_it,
```

where `f_t` contains the market excess return, SMB, and HML factors. Coefficient
uncertainty uses a 12-lag Bartlett-kernel Newey-West covariance estimator.
Raw alpha p-values are adjusted across the 25 portfolios with the
Benjamini-Hochberg procedure.

Noisy cross-sectional alpha estimates are partially pooled through

```text
alpha_hat_i | alpha_i ~ Normal(alpha_i, se_i^2)
alpha_i               ~ Normal(mu, tau^2).
```

`mu` and `tau` are estimated by profile marginal maximum likelihood. The
posterior mean is

```text
E[alpha_i | data] = w_i alpha_hat_i + (1 - w_i) mu,
w_i = tau^2 / (tau^2 + se_i^2).
```

This resolves the earlier identification problem: persistent performance is
represented by one estimand, alpha, rather than being split arbitrarily between
an intercept and a non-negative latent term.

Rank uncertainty is estimated with a common-date circular block bootstrap. The
same sampled months are used for every portfolio, preserving cross-sectional
dependence. Rolling estimates use 120-month training windows ending every 12
months. Persistence and transition outputs explicitly report the fraction of
overlap, and only the 120-month horizon is labelled eligible for structural
interpretation.

Forward validation freezes each training window's factor loadings and ranking,
then evaluates factor-adjusted returns in the next 12 months. Future returns do
not enter the score, prior, beta estimates, or rank.

### Secondary SFA diagnostic

The half-normal diagnostic models factor residuals as `epsilon = v - u`, with
Gaussian noise `v` and non-negative `u`. Its conditional score uses
`E[exp(-u / sigma) | epsilon]`, where `sigma` is the fitted total residual scale;
this makes the diagnostic invariant to expressing returns in decimals or
percent. A one-sided likelihood-ratio boundary test is applied against the
Gaussian model. Unsupported fits report the null model and receive no SFA rank.
A truncated-normal specification is retained only as a distributional
sensitivity check.

## Data and validation

The local inputs originate from the
[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
Portfolio construction is documented on the official
[25 size/book-to-market portfolio page](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/tw_5_ports.html).
Full provenance and reviewed hashes are in [`data/SOURCES.md`](data/SOURCES.md).

The loader fails on:

- duplicate portfolio-month keys;
- null required values;
- factor values that disagree across portfolios within a month;
- an unbalanced panel; or
- missing calendar months.

Each run records file hashes, byte sizes, dimensions, sample dates, validation
flags, configuration, and runtime in `results/tables/dataset_manifest.csv` and
`results/run_manifest.json`.

## Reproduce the analysis

Python 3.13 is the reviewed environment.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev-lock.txt
pytest -q
ruff check .
MPLBACKEND=Agg python -m analysis.run_all
python -m analysis.export_tables
python reports/generate_report.py
```

Use `requirements.txt` for supported dependency ranges,
`requirements-lock.txt` for the reviewed runtime lock, and
`requirements-dev-lock.txt` for the exact QA and report-generation environment.

## Canonical outputs

| Purpose | File |
|---|---|
| Validated factor panel | `results/tables/factor_model_dataset.csv` |
| Data quality and hashes | `results/tables/dataset_manifest.csv` |
| Primary estimates and intervals | `results/tables/performance_scores.csv` |
| Bootstrap rank uncertainty | `results/tables/performance_rank_uncertainty.csv` |
| Rolling estimates | `results/tables/rolling_performance_scores.csv` |
| Overlap-labelled persistence | `results/tables/rank_persistence.csv` |
| Non-overlapping transitions | `results/tables/transition_matrix.csv` |
| Look-ahead-free validation | `results/tables/forward_performance_validation.csv` |
| HAC forward summary | `results/tables/forward_performance_aggregate.csv` |
| Residual tests | `results/tables/performance_residual_diagnostics.csv` |
| SFA boundary evidence | `results/tables/sfa_asymmetry_diagnostics.csv` |
| Full technical report | [`output/pdf/latent-performance-benchmarking-technical-report.pdf`](output/pdf/latent-performance-benchmarking-technical-report.pdf) |

![Posterior performance ranking](results/figures/performance_ranking.png)

![Bootstrap rank uncertainty](results/figures/rank_uncertainty.png)

![Look-ahead-free validation](results/figures/forward_performance_validation.png)

## Interpretation limits

- The Fama-French three-factor model is a benchmark, not a complete return
  model. Omitted factors can appear as alpha.
- The posterior intervals are empirical-Bayes approximations; uncertainty in
  the estimated hyperparameters is only approximated.
- The 25 portfolios are related constructed portfolios, not independent
  securities. The common-date bootstrap helps preserve dependence but 200
  replicates give only moderate tail precision.
- Heavy tails, serial dependence, and conditional heteroskedasticity remain in
  residuals. HAC protects standard errors against broad dependence, not against
  all forms of misspecification.
- Forward validation reuses overlapping 120-month training histories even
  though the 12-month evaluation periods do not overlap. HAC inference is used
  for the aggregate time series.
- Full-sample posterior ranks are descriptive and use all historical data. Only
  the dedicated forward tables are out of sample.
- The boundary test has limited power and the SFA distribution is restrictive.
  The two supported cases are diagnostics of residual asymmetry, not proof of
  managerial inefficiency.
- Portfolio returns exclude implementation costs, taxes, capacity constraints,
  and real-time data revisions.

## References

- Fama, E. F., and French, K. R. (1993). Common risk factors in the returns on
  stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Newey, W. K., and West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55(3), 703-708.
- Jondrow, J., Lovell, C. A. K., Materov, I. S., and Schmidt, P. (1982). On the
  estimation of technical inefficiency in the stochastic frontier production
  function model. *Journal of Econometrics*, 19(2-3), 233-238.

## Licence and citation

The code is released under the MIT Licence. Citation metadata is available in
[`CITATION.cff`](CITATION.cff).
