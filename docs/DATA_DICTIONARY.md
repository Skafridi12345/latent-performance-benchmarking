# Data dictionary

## Primary performance fields

| Field | Definition |
|---|---|
| `alpha` | Monthly OLS factor intercept in decimal return units. |
| `alpha_hac_se` | Portfolio-specific diagonal of the full 12-lag HAC covariance. |
| `alpha_p_value` | Two-sided normal-reference raw alpha p-value. |
| `alpha_fdr_q_value` | BH q-value across the 25 raw alpha tests. |
| `posterior_alpha` | Multivariate empirical-Bayes posterior mean, monthly decimal. |
| `posterior_alpha_sd` | Square root of posterior covariance diagonal. |
| `posterior_alpha_ci_low/high` | Approximate 95% posterior interval. |
| `performance_rank` | Rank of posterior alpha; 1 is highest. |
| `bootstrap_rank_ci_low/high` | 2.5% and 97.5% rank quantiles from 5,000 draws. |

Matrix row/column order is explicitly stored by portfolio labels in
`joint_alpha_hac_covariance.csv`, `posterior_alpha_covariance.csv`, and
`empirical_bayes_shrinkage_matrix.csv`.

## SFA fields

| Field | Definition |
|---|---|
| `sfa_boundary_p_value` | Raw 50:50 boundary-mixture likelihood-ratio p-value. |
| `sfa_boundary_q_value` | BH q-value across all 25 boundary tests. |
| `sfa_supported_nominal_5pct` | Raw p-value below 0.05. |
| `sfa_supported_fdr_5pct` | BH q-value below 0.05. |
| `sfa_rank` | AE rank only among converged FDR-supported fits; otherwise missing. |

`AE_rank` is a compatibility alias with exactly the same FDR eligibility; it is
not a separate inferential result.

## Forward fields

`average_top_quintile_minus_average_bottom_quintile_forward_alpha_annualized_bps`
is the average alpha of the five portfolios in predicted quintile 5 minus the
average of the five in quintile 1, annualized and multiplied by 10,000.
`best_ranked_minus_worst_ranked_forward_alpha_annualized_bps` is the separate
single-extreme diagnostic. Leakage flags document frozen loadings, exclusion of
future outcomes from score estimation, and strict date ordering.

## Validation status

- `PASS`: the stated check was completed successfully and its tolerance was
  satisfied.
- `FAIL`: a valid required check failed; final integrity approval is blocked.
- `UNRESOLVED`: a discrepancy is visible, but its cause cannot be determined
  from available evidence; it remains open but non-blocking.
- `NOT_COMPARABLE`: the available source provenance does not support a valid
  exact-equality acceptance test; unequal values remain visible and the check
  is non-blocking.

The two current-official versus fixed-local overlap rows are `NOT_COMPARABLE`.
The local sample ends in November 2025, but its exact acquisition vintage is
unverified. This status does not imply equality: both rows retain the maximum
difference, location, count, percentage, period summary, and provenance caveat.
