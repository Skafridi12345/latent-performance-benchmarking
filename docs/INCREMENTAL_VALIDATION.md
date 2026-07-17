# Incremental forward test: does the rolling ranking add anything?

## Motivation

The canonical forward validation shows the rolling latent-performance ranking
predicts future FF3 alpha (Fisher-averaged Spearman ≈ 0.10, p ≈ 2e-5). That
result on its own is weak evidence of a *dynamic* signal: the 25 size/BM cells
have persistent FF3 mispricing, so any ranking anchored to past alpha will
correlate with future alpha even if it carries no time-varying information.

This test isolates the **incremental** content of the rolling ranking by
scoring it against fixed benchmark rankings on the *identical* forward
observations (same windows, same frozen betas, same forward alphas). The unit
of comparison is, per window:

- `rank_spearman_increment` = Spearman(rolling, forward) − Spearman(static, forward)
- `spread_increment_bps` = rolling top-5-minus-bottom-5 forward spread − static spread

Inference on the increment series uses HAC (Bartlett, Newey–West automatic
bandwidth = 3 over 89 windows) and a circular block bootstrap (block length 2
windows ≈ 24 months, 5,000 replicates, seed 2026). Run with:

```bash
MPLBACKEND=Agg python -m analysis.incremental_validation --scheme all
```

Outputs: `results/tables/incremental_validation_{windows,summary}_<scheme>.csv`,
`results/tables/static_characteristic_ranking_<scheme>.csv`, and
`results/figures/incremental_rolling_vs_static_<scheme>.png`.

## Benchmarks

| Scheme | What the benchmark ranking encodes | Peeks? | Time-varying? |
|---|---|:--:|:--:|
| `lexicographic` | Size asc, BM desc — additive size+value characteristic gradient (task-spec algorithm); `SMALL HiBM` → `BIG LoBM` | no | no |
| `diagonal` | Value-minus-size score `(bm − size)` — diagonal gradient; `SMALL HiBM` → `BIG LoBM` | no | no |
| `expanding_window` | Ranking re-estimated on **all data up to** each window end (leakage-free) | no | yes |
| `unconditional_alpha` | **Full-sample** posterior-alpha ordering (oracle upper bound); `SMALL HiBM` → `SMALL LoBM` | **yes** | no |

> Note on the spec: the requested algorithm "(size asc, BM desc)" places
> `SMALL LoBM` at rank 5, **not** 25. The parenthetical `SMALL LoBM = 25`
> describes the diagonal small-growth anomaly, which no smooth characteristic
> gradient reproduces and which would require peeking at the realized outcome.
> The `unconditional_alpha` oracle is the benchmark that actually recovers
> `SMALL LoBM` as worst.

## Results

Level rank correlation with forward alpha (higher = more predictive):
**rolling = +0.099 (p ≈ 7e-6)**; characteristic gradient ≈ 0
(lexicographic −0.018, p = 0.79; diagonal +0.006, p = 0.38);
**expanding-window = +0.140 (p ≈ 1e-9)**; **oracle = +0.199 (p ≈ 4e-16)**.
The rolling ranking is the *weakest* of the informed orderings.

| Benchmark | Rank-corr increment (rolling − benchmark) | HAC 1-sided p | Bootstrap 95% CI | Rolling beats it? |
|---|---:|---:|---|:--:|
| `lexicographic` | **+0.117** | 6.5e-06 | [+0.062, +0.169] | ✅ yes |
| `diagonal` | **+0.092** | 3.5e-04 | [+0.038, +0.145] | ✅ yes |
| `expanding_window` | **−0.042** | 0.97 | [−0.085, +0.001] | ❌ no |
| `unconditional_alpha` | **−0.100** | 1.00 | [−0.148, −0.053] | ❌ no |

Spread increment (annualized bps) agrees: +343 and +221 bps vs the
characteristic gradients (both p < 1e-4), but **−44 bps** vs the expanding
window (p = 0.85) and **−165 bps** vs the oracle (p ≈ 1.0) — neither negative
result is a tradable edge for the static side, but both refute an edge for the
rolling side.

## Interpretation

Three findings, and the last two are decisive.

1. **The rolling ranking beats the a-priori characteristic gradient.** A smooth
   size/value ordering has *zero* forward power on FF3 alpha — unsurprising,
   since alpha is the residual *after* size and value exposure are removed. So
   the rolling ranking's signal is not merely a restatement of the size/value
   tilt.

2. **A leakage-free expanding-window ranking does at least as well — with no
   time variation of the harmful kind.** Accumulating all past history (which
   converges toward the stable long-run ordering) matches or slightly beats the
   120-month rolling ranking (+0.140 vs +0.099). The incremental correlation is
   *negative* and insignificant, so discarding data older than 120 months and
   chasing recent alpha adds noise, not signal. This test uses only past data,
   so it is a fair, tradable comparison.

3. **A single fixed (oracle) ordering does best.** The full-sample alpha ranking
   predicts forward alpha about **twice as well** as the rolling ranking (0.199
   vs 0.099). The rolling ranking's power is fully explained by the *stable but
   irregular* cell-level alpha pattern (small-growth structurally worst,
   small-value best, with an irregular interior — e.g. `BIG LoBM` near the top —
   that no smooth gradient captures), **not** by time variation.

**Bottom line for the project's claims.** The headline forward result is real
but is **not** dynamic forecasting skill. It is driven by persistent,
irregular characteristic mispricing that a *stable* ranking captures at least as
well. Re-estimating every window adds no information over an expanding-window
ranking (p = 0.97) and is dominated by the full-sample ordering. Any language in
the technical report implying that the *time-varying* ranking forecasts is not
supported; the defensible claim is that a **stable cross-sectional alpha
ordering** of the 25 cells persists out-of-sample.

## Caveat

`unconditional_alpha` uses full-sample (look-ahead) information, so treat it as
an *upper bound* on what a static ranking can achieve, not a tradable strategy.
The `expanding_window` benchmark is the leakage-free, tradable comparator and is
the one the bottom line rests on. All four rankings are scored on identical
forward observations and frozen rolling betas, so the comparison isolates the
ordering, not the forward-outcome definition.
