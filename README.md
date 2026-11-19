# GTA VI Trophy Rarity Forecast

`main.py` (Python 3.13, standard library only, no network) forecasts, for every
PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at a case's
target. It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): five quantiles
(5/25/50/75/95%) of each trophy's earn rate, scored by mean pinball loss on the
logit scale.

The model, per trophy:

- **Trend** — a recency-weighted least-squares line of logit(rate) against
  ln(1 + hours since release), read at the target, with the slope shrunk toward
  zero by its own signal-to-noise ratio (flat or noisy series fall back to
  persistence). The resulting median shift is damped by how far it extrapolates
  past the observed window and capped at ±2.25 logit.
- **Denominator** — earners (rate × players tracked) are fitted the same way and
  divided by players tracked projected to the target, then blended 50/50 with the
  direct trend; earners grow more smoothly than rates when the tracked count jumps.
- **Stale-page guard** — if the tracked count never changed across the history and
  the game is under 48 hours old, the history is an un-refreshed snapshot: the
  median takes a dilution allowance (−1.15 logit) and the spread widens, instead
  of trusting a flat line hours into a launch.
- **Spread** — the house random-walk floor 0.5·√(horizon days), the trend fit's
  residual sigma inflated by extrapolation leverage, and the stale allowance,
  combined in quadrature (capped at 2.5 logit).

The program is deterministic: two runs on the same case produce byte-identical
output, and a run takes well under a second even on the final case's hundreds of
reads (limit: 8 s).
