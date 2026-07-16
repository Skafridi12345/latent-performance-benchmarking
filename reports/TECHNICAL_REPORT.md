# Risk-Adjusted Portfolio Benchmarking

## Quantitative review and rebuilt methodology

Review date: 16 July 2026

This report replaces the pre-rebuild document, which was based on a duplicated
many-to-many merge. The canonical source is the validated pipeline output under
`results/tables/`; `reports/generate_report.py` converts those tables and figures
into the final PDF. The invalid pre-rebuild artefacts have been removed.

## Answer first

The data rebuild materially changes the strength of the conclusions. The
primary performance score is now Fama-French three-factor alpha with 12-lag
Newey-West inference and cross-sectional empirical-Bayes shrinkage. Ranks are
accompanied by common-date circular block-bootstrap uncertainty. SFA is no
longer treated as the main performance model.

The sample contains 29,825 unique observations: 25 portfolios by 1,193 months,
from July 1926 to November 2025. Only one posterior alpha interval is wholly
above zero. Full-sample ranks have wide bootstrap intervals. Persistence is
high when rolling windows overlap, but mean rank persistence falls to about
0.13 at the non-overlapping 120-month horizon.

The dedicated forward test is more encouraging but still modest. Across 89
non-overlapping 12-month evaluation periods, the Fisher-averaged rank
correlation is 0.096 (HAC 95% interval 0.041 to 0.151), and the top-minus-bottom
future factor-alpha spread averages 172 bps/year (HAC 95% interval 63 to 280).

Only 2 of 25 portfolios provide 5% boundary-test support for a one-sided SFA
residual component. SFA scores are therefore residual-asymmetry diagnostics,
not general portfolio rankings.

## Methodological decisions

1. Fail fast on duplicate keys, missing values, factor disagreement, panel
   imbalance, or missing calendar months.
2. Estimate one persistent performance term, alpha, rather than an unidentified
   intercept and portfolio-level inefficiency term.
3. Use Newey-West covariance, false-discovery control, hierarchical shrinkage,
   and common-date block-bootstrap rank intervals.
4. Label rolling-window overlap and reserve structural persistence claims for
   non-overlapping windows.
5. Freeze training betas and ranks before evaluating the next 12 months.
6. Use a scale-invariant conditional exponential moment for SFA and require
   boundary evidence before assigning an SFA rank.

## Remaining limits

The three-factor benchmark can omit relevant systematic risks. Empirical-Bayes
intervals approximate hyperparameter uncertainty. Residuals remain non-normal,
serially dependent for many portfolios, and conditionally heteroskedastic. The
200-replicate bootstrap has moderate rather than fine tail resolution. The
historical portfolio returns are not implementable net performance. The two SFA
signals do not establish managerial inefficiency.

## References

- Fama, E. F., and French, K. R. (1993), Journal of Financial Economics 33(1),
  3-56.
- Newey, W. K., and West, K. D. (1987), Econometrica 55(3), 703-708.
- Jondrow, J., Lovell, C. A. K., Materov, I. S., and Schmidt, P. (1982), Journal
  of Econometrics 19(2-3), 233-238.
