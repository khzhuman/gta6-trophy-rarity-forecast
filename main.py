"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase earn rate.

Model, per trophy, all on the logit scale (the scale the pinball loss is scored on):

  1. Trend: a recency-weighted least-squares line of logit(rate) against
     x = ln(1 + hours since release midnight 2026-11-19T00:00:00Z), read at the
     target. The slope is shrunk toward 0 by its own signal-to-noise ratio, so a
     flat or noisy series falls back to persistence instead of extrapolating noise.
  2. Denominator: earners (rate/100 x players_tracked) grow more smoothly than the
     rate itself, so a second line is fitted to ln(earners), players_tracked is
     projected to the target with its own shrunk log-line, and the implied rate
     (projected earners / projected players) is blended 50/50 with the direct trend.
  3. Stale-page guard: if players_tracked never changed across the whole history
     and the game is under 48 hours old, the page in the history is an un-refreshed
     snapshot (the launch-night trap: 638 players for hours while the real count
     explodes at the first refresh). The median is shifted down by a dilution
     allowance and the spread widened, instead of pretending the flat line is precise.
  4. Spread: sigma combines the house random-walk floor 0.5*sqrt(horizon_days),
     the trend fit's residual sigma inflated by the extrapolation leverage, and the
     stale-page allowance, in quadrature. Quantiles are mu + z_tau * sigma.

Deterministic, standard library only, no network. Reads every *.in in
$MAKEFUN_INPUT_DIR (else /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

RW_SIGMA = 0.5            # random-walk spread per sqrt(day), the house floor
HALF_LIFE_H = 20.0        # recency weighting half-life for trend fits, hours
MIN_TREND_POINTS = 4      # fewer usable reads than this -> persistence
MAX_SHIFT = 2.25          # cap on |median logit shift| away from the last rate
MAX_SLOPE = 3.0           # cap on |shrunk slope|, logit per unit of x
K_RESID = 1.0             # multiplier on the residual-based spread component
SIGMA_CAP = 2.5

STALE_MAX_AGE_H = 48.0    # stale-page guard only while the game is this young
STALE_MIN_SPAN_H = 1.0    # ... and the unchanged history spans at least this long
STALE_SHIFT = 1.15        # median dilution allowance when the guard fires, logit
STALE_SIGMA = 1.0         # extra spread when the guard fires, logit


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when: datetime) -> float:
    return (when - RELEASE).total_seconds() / 3600.0


def x_of(hours: float) -> float:
    """The trend axis: ln(1 + hours since release); pre-release reads sit just below 0."""
    return math.log1p(max(hours, -0.99))


def recency_weight(age_h: float) -> float:
    return 0.5 ** (max(age_h, 0.0) / HALF_LIFE_H)


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x with the slope shrunk by signal-to-noise.

    Returns (a, b_shrunk, resid_sd, sxx, x_bar, n_eff) or None when the series
    cannot support a line (too few points, or no spread on the x axis).
    """
    n = len(xs)
    if n < MIN_TREND_POINTS:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    x_bar = sum(w * x for x, w in zip(xs, ws)) / sw
    y_bar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-9:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = y_bar - b * x_bar
    dof = max(n - 2, 1)
    resid_var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof
    resid_sd = math.sqrt(max(resid_var, 0.0))
    se_b = math.sqrt(resid_var / sxx) if resid_var > 0 else 0.0
    if se_b > 0:
        shrink = (b * b) / (b * b + se_b * se_b)
    else:
        shrink = 1.0
    b_shrunk = max(-MAX_SLOPE, min(MAX_SLOPE, b * shrink))
    a_shrunk = y_bar - b_shrunk * x_bar
    return a_shrunk, b_shrunk, resid_sd, sxx, x_bar, n


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    checkpoint_h = hours_since_release(parse_time(case["at"]))
    target_h = hours_since_release(parse_time(case["target"]))
    x_tg = x_of(target_h)

    read_h = [hours_since_release(parse_time(r["at"])) for r in history]
    read_x = [x_of(h) for h in read_h]
    read_w = [recency_weight(checkpoint_h - h) for h in read_h]
    players = [float(r["players_tracked"]) for r in history]

    # --- Project players_tracked to the target (log-line, shrunk slope). ---
    p_last = players[-1]
    p_target = p_last
    fit_p = wls_line(read_x, [math.log(max(p, 1.0)) for p in players], read_w)
    if fit_p is not None:
        a_p, b_p, _, _, _, _ = fit_p
        p_target = math.exp(a_p + b_p * x_tg)
        p_target = min(max(p_target, p_last), p_last * 8.0)

    # --- Stale-page guard: the tracked count never moved in the whole history. ---
    span_h = read_h[-1] - read_h[0] if len(read_h) > 1 else 0.0
    stale = (
        max(players) == min(players)
        and span_h >= STALE_MIN_SPAN_H
        and checkpoint_h < STALE_MAX_AGE_H
    )

    floor_sigma = RW_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        series = [
            (read_x[j], history[j]["rates"][i], read_w[j], players[j])
            for j in range(len(history))
            if history[j]["rates"][i] is not None
        ]
        if not series:
            # Never seen in any read: a rare trophy nobody has earned yet.
            mu, sigma = logit(0.05), max(floor_sigma, 1.0)
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(mu + z * sigma)) for z in Z]})
            continue

        r_last = series[-1][1]
        lg_last = logit(r_last)
        xs = [s[0] for s in series]
        ws = [s[2] for s in series]

        # --- Direct trend on logit(rate). ---
        mu_direct = lg_last
        resid_sd = 0.0
        leverage = 1.0
        fit = wls_line(xs, [logit(s[1]) for s in series], ws)
        if fit is not None:
            a_f, b_f, resid_sd, sxx, x_bar, _ = fit
            mu_direct = a_f + b_f * x_tg
            leverage = math.sqrt(1.0 + 1.0 / len(series) + (x_tg - x_bar) ** 2 / sxx)

        # --- Earners trend, divided by projected players. ---
        mu_earn = None
        fit_e = wls_line(
            xs,
            [math.log(max(s[1] / 100.0 * s[3], 0.0) + 0.5) for s in series],
            ws,
        )
        if fit_e is not None:
            a_e, b_e, _, _, _, _ = fit_e
            earners_tg = max(math.exp(a_e + b_e * x_tg) - 0.5, 0.0)
            mu_earn = logit(min(100.0 * earners_tg / p_target, 99.99))

        mu = 0.5 * (mu_direct + mu_earn) if mu_earn is not None else mu_direct

        # Damp the shift by how far past the observed window it extrapolates:
        # a trend may not carry the median further than the span that supports it.
        span_x = max(max(xs) - min(xs), 0.5) if xs else 0.5
        extrap_x = max(0.0, x_tg - (max(xs) if xs else x_tg))
        damp = span_x / (span_x + extrap_x)
        mu = lg_last + (mu - lg_last) * damp

        # Keep the median within a sane distance of the last observed rate.
        mu = max(lg_last - MAX_SHIFT, min(lg_last + MAX_SHIFT, mu))
        mu = max(logit(0.01), min(logit(99.99), mu))

        # --- Spread: floor + residual (with extrapolation leverage) + stale. ---
        sigma = floor_sigma ** 2 + (K_RESID * resid_sd * leverage) ** 2
        if stale:
            mu = max(logit(0.01), mu - STALE_SHIFT)
            sigma += STALE_SIGMA ** 2
        sigma = min(math.sqrt(sigma), SIGMA_CAP)

        rows.append({"id": trophy["id"], "q": [round4(inv_logit(mu + z * sigma)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
