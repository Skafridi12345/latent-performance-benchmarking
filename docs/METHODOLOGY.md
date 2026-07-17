# Methodology

## Estimand and panel

The unit is one value-weighted portfolio-month. Returns and factors are decimals.
`excess_return = ret - rf`. The canonical panel is balanced across 25
portfolios and 1,193 months. Loaders fail on duplicate entity-month keys, null
required values, within-month factor disagreement, imbalance, or calendar gaps.

## Full joint HAC covariance

Each portfolio is estimated by OLS with an intercept and the selected factors.
Let `a_t` be the OLS intercept weight at month `t` and `e_t` the 25-vector of
same-month residuals. The alpha score vector is `g_t = a_t e_t`. The full HAC
covariance is the Bartlett-weighted sum of `g_t g_s'` through lag 12, multiplied
by the finite-sample correction `T/(T-P)`, where `P` is the total number of
regression coefficients including the intercept.

The implementation enforces symmetry, checks finiteness and eigenvalues, and
raises on material indefiniteness. Only a scale-relative numerical floor may be
used for near-zero eigenvalues, and its size/count are persisted. The canonical
matrix is positive definite without flooring and has condition number 139.4.

## Multivariate empirical Bayes

```text
alpha_hat | alpha ~ N(alpha, V_HAC)
alpha             ~ N(mu 1, tau^2 I)
```

For each `tau^2`, `mu` is profiled with generalized least squares. `tau^2` is
selected by bounded marginal maximum likelihood. The posterior mean is

```text
mu 1 + tau^2 (V_HAC + tau^2 I)^-1 (alpha_hat - mu 1).
```

The conditional posterior covariance is

```text
tau^2 I - tau^4 (V_HAC + tau^2 I)^-1,
```

plus a first-order outer-product term for uncertainty in the profiled common
mean. This does not fully integrate uncertainty in `tau`.

## Multiple testing and ranking

Raw alpha p-values form one 25-test BH family. SFA boundary-mixture p-values
form a separate 25-test BH family. Invalid p-values remain missing. Stable
sorting and the reverse cumulative minimum give monotone q-values under ties.
SFA output fields are `sfa_boundary_p_value`, `sfa_boundary_q_value`,
`sfa_supported_nominal_5pct`, `sfa_supported_fdr_5pct`, and `sfa_rank`. Only
FDR-supported converged fits receive `sfa_rank`.

## Joint alpha tests

The primary joint test is `alpha_hat' V_HAC^-1 alpha_hat`, compared with
`chi-square(25)`. It is robust to the modeled serial and cross-portfolio
dependence.

The conventional GRS statistic uses ML-scaled residual and factor covariance
matrices and the `F(N, T-N-K)` reference distribution. It assumes IID normal
errors, constant residual covariance, and nonsingular factor/residual
covariances. It is retained as a secondary benchmark because diagnostics reject
those assumptions.

## Bootstrap and forward validation

The circular block bootstrap samples common month indices for every portfolio,
uses 12-month blocks and seed 2026, and refits regressions, joint HAC covariance,
and empirical-Bayes hyperparameters in every draw. The canonical run uses 5,000
draws. A saved table compares the first 1,000 draws with the final run.

Rolling scores use 120-month training windows ending every 12 months. Forward
validation freezes scores, ranks, and betas at the training cutoff and evaluates
only the following 12 months. “Top minus bottom” is defined precisely as the
average forward alpha of the five top-quintile portfolios minus the average of
the five bottom-quintile portfolios. The best-minus-worst single-portfolio
spread is reported separately. Robustness tables include median, IQR, positive
fraction, decade, leave-one-decade-out, and extreme windows.

## External and model validation

Official current snapshots are pinned separately and never replace local data.
The fixed local inputs have a sample ending November 2025; exact acquisition
vintage is unverified. A current exact-overlap comparison that differs is
therefore `NOT_COMPARABLE`, not `PASS`: it remains visible and non-blocking
because the available provenance does not support a valid exact-equality
acceptance test. `FAIL` is reserved for a valid required check that fails and
blocks integrity approval; `UNRESOLVED` records a visible non-blocking
discrepancy whose cause cannot be determined. Official historical revisions are
a plausible, not proven, cause of the observed overlap differences.

Checks cover source hashes/URLs/access dates, units, weighting, coverage, labels,
exact overlap, independent statsmodels OLS/HAC, qualitative 5x5 patterns, the
July 1963-December 1991 historical anchor, and prespecified FF5 sensitivity on
the July 1963 common sample.
