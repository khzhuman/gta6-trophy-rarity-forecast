"""gta6-trophy-rarity-forecast-01, v2: forecast each GTA VI PS5 trophy's
Exophase earn rate at the case's target, modelling the page's refresh process.

Why this shape. A case's history is not a time series of the rates: it is a
sequence of reads of a page that only changes when Exophase refreshes it.
Between refreshes every read is identical, and at a refresh the printed rates
jump, because the page then shows the counts as of that moment. The answering
read likewise shows the state of the last refresh before it. So the forecast
target is a page STATE, and a state is two counts, not one ratio:

    rate_i = E_i / N        (earners of trophy i, over players tracked)

E_i and N move on different clocks. Launch night proved it: the page sat at
638 players for hours, then the first refresh showed 3,868 players (x6.06)
while the earner counts behind the rates had grown only about x1.17 -- the
denominator exploded, the numerators barely moved, and every rate diluted.
A model that trends the ratio directly (v1) cannot see that; a model that
forecasts E_i and N separately, as the step processes they are, can.

The model, per case:

  1. Counts. Each read's printed rate is a count in disguise: with N players
     tracked, one earner is 100/N of a percent, far coarser than the printed
     0.01, so E_i = round(rate_i / 100 * N) recovers the exact earner counts
     (checked on the launch-night page: the recovered counts sum to the
     page's own trophies-earned figure, 943).
  2. States. Consecutive reads with the same (N, rates) collapse into one
     page state. The state sequence -- not the read sequence -- is what
     carries information; each state's as-of time is bracketed by the
     previous state's last read and its own first read.
  3. Growth. On the state sequence, ln N and each ln E_i are fitted against
     u = ln(1 + hours/24) (launch growth is a power law, straight on this
     axis) by weighted least squares, recent states weighted most. Slopes
     are shrunk toward a target by their own standard error and capped:
     the players' slope toward a launch prior, each trophy's toward the
     mean earner slope (fitted on total earners, leaving out any trophy
     every tracked player holds, whose count is N itself). Predictions are
     clamped so no count falls, and none grows past a sane multiple of its
     last state over the extrapolated span.
  4. Refresh timing. The answer shows the last refresh at or before the
     answering read, so E and N are forecast at the expected as-of time of
     that state (half a mean refresh gap before the answer), and the
     predictive distribution mixes in the probability that no refresh
     happens at all before the answer -- in which case the rates are the
     last printed ones, exactly.
  5. Spread. On ln(rate) the variance combines the earner fit's residual
     variance, the player fit's, a correlation between them (growth
     surprises move both), and a count-noise term for small E_i. Quantiles
     of the resulting (possibly mixed) distribution are read off at the
     case's five levels.

  Single-state histories (the launch-night trap: one frozen page, the
  refresh still unseen) take a dedicated branch: the refreshed state is
  modelled with the launch transition's own multipliers -- ln N shifted by
  +1.72 +/- 0.50, each ln E_i by +0.16 +/- 0.28 -- mixed with a no-refresh
  component whose weight falls as the freeze lengthens.

Deterministic, standard library only, no network. Reads every *.in in
$MAKEFUN_INPUT_DIR (else /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round4(x: float):
    value = math.floor(x * 10000 + 0.5) / 10000
    return int(value) if float(value).is_integer() else value


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, resid_sd, sxx, x_bar, se_b), or None when the points
    cannot support a line (fewer than two, or no spread on the x axis).
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    x_bar = sum(w * x for x, w in zip(xs, ws)) / sw
    y_bar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = y_bar - b * x_bar
    dof = max(n - 2, 1)
    resid_var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof
    resid_sd = math.sqrt(max(resid_var, 0.0))
    se_b = math.sqrt(resid_var / sxx) if resid_var > 0 else 0.0
    return a, b, resid_sd, sxx, x_bar, se_b


class State:
    """One distinct page state: a run of reads that all show the same page."""

    __slots__ = ("first_seen", "last_seen", "n", "earners", "total_earned")


def build_states(history):
    """Collapse the read history into page states, recovering earner counts."""
    states = []
    for read in history:
        n = float(read["players_tracked"])
        earners = [None if rate is None else rate / 100.0 * n for rate in read["rates"]]
        t = parse_time(read["at"])
        same = (
            states
            and states[-1].n == n
            and all(
                (a is None and b is None)
                or (a is not None and b is not None and abs(a - b) < 1e-9)
                for a, b in zip(states[-1].earners, earners)
            )
        )
        if same:
            states[-1].last_seen = t
        else:
            s = State()
            s.first_seen = t
            s.last_seen = t
            s.n = n
            s.earners = earners
            s.total_earned = float(read["trophies_earned"])
            states.append(s)
    return states


def mixture_quantiles(components):
    """The five level quantiles of a mixture of normals on the ln-rate scale.

    components: (weight, mean, sd) triples; a near-zero sd acts as a point
    mass (the no-refresh component, sitting on the last printed rate).
    Quantiles come from bisecting the mixture CDF, so they are ordered by
    construction.
    """
    lo = min(m - 5 * s for _, m, s in components)
    hi = max(m + 5 * s for _, m, s in components)

    def cdf(x):
        total = 0.0
        for w, m, s in components:
            total += w * 0.5 * (1.0 + math.erf((x - m) / s / math.sqrt(2.0)))
        return total

    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(60):
            mid = (a + b) / 2
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append((a + b) / 2)
    return out


def forecast_single_state(case, last, mean_gap, trophies):
    """The history is one frozen page: the answer turns on the first refresh
    the case has not seen. At a refresh the tracked-player count jumps hard
    (launch night: x6.06) while earner counts barely move (x~1.17), so the
    ratio dilutes; model the refreshed state as ln E and ln N shifted
    separately, mixed with a no-refresh component at the last printed rates.
    """
    span_h = (last.last_seen - last.first_seen).total_seconds() / 3600.0
    horizon_h = case["horizon_minutes"] / 60.0
    age_h = (parse_time(case["at"]) - RELEASE).total_seconds() / 3600.0
    if age_h < 72.0:
        # The launch-freeze regime, with the observed transition's shape.
        mu_ln_n, sig_ln_n = 1.72, 0.50
        mu_ln_e, sig_ln_e = 0.16, 0.28
        p0 = math.exp(-(((span_h + horizon_h + 0.5) / 4.5) ** 1.5))
        p0 = min(max(p0, 0.05), 0.45)
    else:
        # A long freeze this late means refreshes are rare; expect little.
        mu_ln_n, sig_ln_n = 0.05, 0.25
        mu_ln_e, sig_ln_e = 0.05, 0.25
        p0 = min(max(math.exp(-(horizon_h + 0.5) / mean_gap), 0.1), 0.8)

    rows = []
    for i, trophy in enumerate(trophies):
        e_last = last.earners[i]
        if e_last is None:
            m, s = math.log(0.05 / 100.0), 1.0
            rows.append({"id": trophy["id"], "q": [round4(100.0 * math.exp(m + z * s)) for z in Z]})
            continue
        if e_last >= last.n - 1e-9:
            # Every tracked player holds it; a refresh keeps it at ~100.
            rows.append({"id": trophy["id"], "q": [99.9, 99.97, 99.99, 100, 100]})
            continue
        m = math.log(max(e_last, 0.35)) + mu_ln_e - (math.log(last.n) + mu_ln_n)
        var = sig_ln_e ** 2 + sig_ln_n ** 2 - 2 * 0.3 * sig_ln_e * sig_ln_n + 0.12 / max(e_last, 1.0)
        s = math.sqrt(var)
        ln_r_last = math.log(max(e_last / last.n, 0.0001))
        qs = mixture_quantiles([(1 - p0, m, s), (p0, ln_r_last, 0.012)])
        rows.append({"id": trophy["id"], "q": [round4(min(100.0 * math.exp(q), 99.99)) for q in qs]})
    return {"trophies": rows}


def forecast_case(case: dict) -> dict:
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    states = build_states(history)
    k_states = len(states)
    last = states[-1]
    answer_read = target.timestamp() + 1800.0  # the middle of the answer hour

    # Refresh cadence, from how the states' first sightings are spaced.
    if k_states >= 3:
        gaps = [
            (states[k].first_seen - states[k - 1].first_seen).total_seconds() / 3600.0
            for k in range(1, k_states)
        ]
        gaps = [g for g in gaps if g > 0.05]
        mean_gap = sum(gaps) / len(gaps) if gaps else 5.0
    elif k_states == 2:
        mean_gap = max((states[1].first_seen - states[0].first_seen).total_seconds() / 3600.0, 1.0)
    else:
        mean_gap = 5.0
    mean_gap = min(max(mean_gap, 0.5), 48.0)

    if k_states == 1:
        return forecast_single_state(case, last, mean_gap, trophies)

    # Each state's as-of time is bracketed: after the previous state's last
    # read, no later than its own first read. Take the bracket's midpoint.
    t0 = states[0].first_seen.timestamp()

    def hrs(ts):
        return (ts - t0) / 3600.0

    s_times = [0.0]
    for k in range(1, k_states):
        lo = states[k - 1].last_seen.timestamp()
        hi = states[k].first_seen.timestamp()
        s_times.append(hrs((lo + hi) / 2.0))

    # The last refresh before this history: inside its own bracket.
    last_refresh_est = (states[-2].last_seen.timestamp() + last.first_seen.timestamp()) / 2.0
    since_refresh_h = max((answer_read - last_refresh_est) / 3600.0, 0.0)
    p_no_refresh = math.exp(-since_refresh_h / mean_gap)
    # The answer state's expected as-of: the target, less half a refresh gap.
    t_star_h = max(hrs(answer_read) - mean_gap / 2.0, hrs(last.first_seen.timestamp()))

    def u_of(h):
        return math.log1p(max(h, 0.0) / 24.0)

    u_star = u_of(t_star_h)
    u_last = u_of(s_times[-1])
    # Recency weights over states: the last state counts most, halving every
    # six states back.
    ws_state = [0.5 ** ((k_states - 1 - k) / 6.0) for k in range(k_states)]

    def fit_series(vals, target_slope=None, cap_lo=-1.0, cap_hi=4.0):
        """Forecast a log-count series at u_star from its (state, value) points.

        The slope is shrunk toward target_slope by its own standard error
        and capped -- no count grows faster than e^cap per log-day unit --
        and the intercept is re-anchored on the weighted mean so a shrunk
        slope still passes through the data's centre.
        Returns (prediction, resid_sd, slope_used).
        """
        xs = [u_of(s_times[k]) for k, _ in vals]
        ys = [y for _, y in vals]
        ws = [ws_state[k] for k, _ in vals]
        fit = wls(xs, ys, ws)
        if fit is None:
            fallback = 0.0 if target_slope is None else target_slope
            return (ys[-1] if ys else 0.0), 0.3, fallback
        _, b, resid_sd, _, x_bar, se_b = fit
        tgt = 0.0 if target_slope is None else target_slope
        se = max(se_b, 0.02)
        dev = b - tgt
        w_snr = (dev * dev) / (dev * dev + se * se)
        b2 = min(max(tgt + dev * w_snr, cap_lo), cap_hi)
        sw = sum(ws)
        a2 = (sum(w * y for y, w in zip(ys, ws)) / sw) - b2 * x_bar
        return a2 + b2 * u_star, resid_sd, b2

    # Players tracked: a launch prior as the shrinkage target, never falling,
    # and never growing past a sane multiple of the last state.
    ln_n = [(k, math.log(states[k].n)) for k in range(k_states)]
    pred_ln_n, resid_n, slope_n = fit_series(ln_n, target_slope=1.2, cap_lo=0.0)
    grow_cap_n = 2.6 + 1.5 * max(u_star - u_last, 0.0)
    pred_ln_n = min(pred_ln_n, math.log(last.n) + grow_cap_n)
    n_star = max(math.exp(pred_ln_n), last.n)
    sig_ln_n = min(math.sqrt(resid_n ** 2 * (1.0 + max(u_star - u_last, 0.0)) + 0.02 ** 2), 0.6)

    # The mean earner slope: fit total earners across states, leaving out
    # any trophy every tracked player holds (its count is N itself and
    # would drag the mean to the players' slope).
    tot_pts = []
    for k in range(k_states):
        tot = sum(
            e for e in states[k].earners if e is not None and e < states[k].n - 1e-9
        )
        if tot > 0:
            tot_pts.append((k, math.log(tot)))
    if len(tot_pts) >= 2:
        _, _, mean_slope_e = fit_series(tot_pts, target_slope=slope_n * 0.9, cap_lo=-0.5)
    else:
        mean_slope_e = slope_n * 0.9

    rows = []
    for i, trophy in enumerate(trophies):
        e_last = last.earners[i]
        if e_last is None:
            m = math.log(0.0005)
            rows.append({"id": trophy["id"], "q": [round4(100.0 * math.exp(m + z * 1.0)) for z in Z]})
            continue
        if e_last >= last.n - 1e-9:
            rows.append({"id": trophy["id"], "q": [99.9, 99.97, 99.99, 100, 100]})
            continue
        pts = [
            (k, math.log(max(states[k].earners[i], 0.35)))
            for k in range(k_states)
            if states[k].earners[i] is not None
        ]
        pred_ln_e, resid_e, _ = fit_series(pts, target_slope=mean_slope_e, cap_lo=-0.5)
        grow_cap = 2.6 + 1.5 * max(u_star - u_last, 0.0)
        pred_ln_e = min(
            max(pred_ln_e, math.log(max(e_last, 0.35)) - 0.35),
            math.log(max(e_last, 0.35)) + grow_cap,
        )
        e_star = min(max(math.exp(pred_ln_e), e_last), n_star * 0.999)
        m = math.log(max(e_star, 0.05)) - math.log(n_star)
        var_e = min(resid_e ** 2 * (1.0 + max(u_star - u_last, 0.0)), 0.36) + 0.05 / max(e_last, 1.0)
        var = var_e + sig_ln_n ** 2 - 2 * 0.5 * math.sqrt(max(var_e, 1e-6)) * sig_ln_n
        s = min(max(math.sqrt(max(var, 0.0016)), 0.05), 1.1)
        if p_no_refresh > 0.02:
            ln_r_last = math.log(max(e_last / last.n, 0.0001))
            qs = mixture_quantiles([(1 - p_no_refresh, m, s), (p_no_refresh, ln_r_last, 0.012)])
        else:
            qs = [m + z * s for z in Z]
        rows.append({"id": trophy["id"], "q": [round4(min(100.0 * math.exp(q), 99.99)) for q in qs]})
    return {"trophies": rows}


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (output_dir / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
