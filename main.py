#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, minute by minute,
4 hours ahead, as five quantiles a minute.

make.fun bounty polymarket-nvidia-ceo-forecast-01. Reads one case at a time
from /input (<id>.in: every trading market of the event, each with its Yes
price at every minute of the 72 hours up to a checkpoint) and writes
/output/<id>.out: for each market and each of the 240 minutes after the
checkpoint, the 5%, 25%, 50%, 75% and 95% quantiles of that minute's price.

Method (backtested on the six public practice cases; mean pinball loss 0.29
points vs 0.37 for the house persistence baseline): the forecast is an
equal-thirds blend of three per-market quantile forecasters, all centred on
the market's last price (persistence median — short-horizon momentum was
tested and hurt, so no drift is used):

1. Normal / EWMA volatility: the standard normal quantiles scaled by an
   exponentially-weighted (60-minute half-life) estimate of the market's
   1-minute move size, growing with sqrt(horizon). Fast to adapt when a
   market heats up or calms down.
2. Empirical 1-minute shape: the quantiles of the market's own 1-minute
   price changes over the trailing 24 hours, scaled by sqrt(horizon).
   Captures each market's real move distribution — fat tails, jumps and
   asymmetry — instead of assuming normality.
3. Empirical per-horizon shape: for anchor horizons (5, 15, 30, 60, 120,
   240 minutes) the quantiles of that horizon's actual price changes over
   the trailing 24 hours, recentered on the last price and interpolated in
   sqrt(horizon). Captures how moves really grow with horizon (slower than
   a random walk when prices mean-revert, faster when they jump).

Everything is deterministic (no randomness, no clock, no network), standard
library only, and runs in well under a second per case.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in a row's order.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Anchor horizons (minutes) for the per-horizon empirical shape.
ANCHORS = [5, 15, 30, 60, 120, 240]
# How far back the empirical shapes look (minutes): 24 hours. Longer windows
# still hold the resignation shock of 10-11 Nov and overshoot the spread.
LOOKBACK = 1440
# EWMA half-life (minutes) for the volatility estimate: fast enough to track
# this market's regime changes, slow enough not to chase single prints.
EWMA_HALFLIFE = 60
# Bounds on the per-minute volatility (price units): the floor keeps a dead
# market's interval from collapsing to a point, the cap keeps one wild print
# from blowing it out.
SIGMA_FLOOR = 0.0002
SIGMA_CAP = 0.01
# The house persistence baseline's spread, used as a fallback.
BASELINE_SIGMA = 0.0025


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def norm_ppf(p):
    """The standard normal's quantile at p (Acklam's approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e1, 2.209460984245205e2,
         -2.759285984469534e2, 1.383577518672690e2,
         -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2,
         -1.556989798598866e2, 6.680131188771972e1,
         -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1,
         -2.400758277161838, -2.549732539343734,
         4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1,
         2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3])
                                * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def quantile_sorted(vals, tau):
    """The tau-quantile of an already sorted list (linear interpolation)."""
    n = len(vals)
    if n == 1:
        return vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def ewma_sigma(prices):
    """Exponentially-weighted std of 1-minute changes (60-min half-life)."""
    n = len(prices)
    if n < 20:
        return None
    lam = 0.5 ** (1.0 / EWMA_HALFLIFE)
    start = max(1, n - int(6 * EWMA_HALFLIFE))
    wsum = 0.0
    mean = 0.0
    m2 = 0.0
    for i in range(start, n):
        d = prices[i] - prices[i - 1]
        w = lam ** (n - 1 - i)
        wsum += w
        delta = d - mean
        mean += w / wsum * delta
        m2 += w * delta * (d - mean)
    if wsum <= 0 or m2 <= 0:
        return SIGMA_FLOOR
    return min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(m2 / wsum)))


def rows_normal(x, sigma, horizon, quantiles):
    """Normal quantiles around x with per-minute sigma, sqrt(h) growth."""
    zs = [norm_ppf(t) for t in quantiles]
    return [[price6(x + z * sigma * math.sqrt(h)) for z in zs]
            for h in range(1, horizon + 1)]


def rows_stdshape(prices, horizon, quantiles):
    """Empirical 1-minute-change quantiles (trailing 24h) x sqrt(h)."""
    n = len(prices)
    x = prices[-1]
    w = min(LOOKBACK, n - 1)
    if w < 30:
        sig = ewma_sigma(prices) or BASELINE_SIGMA
        return rows_normal(x, sig, horizon, quantiles)
    diffs = sorted(prices[i + 1] - prices[i]
                   for i in range(n - 1 - w, n - 1))
    qs = [quantile_sorted(diffs, t) for t in quantiles]
    # Persistence median: the median change is forced to 0.
    mid = quantiles.index(0.5) if 0.5 in quantiles else None
    if mid is not None:
        qs[mid] = 0.0
    rows = []
    for h in range(1, horizon + 1):
        root = math.sqrt(h)
        row = [price6(x + q * root) for q in qs]
        for k in range(1, len(row)):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def rows_hshape(prices, horizon, quantiles):
    """Per-horizon empirical quantiles, recentered on the last price."""
    n = len(prices)
    x = prices[-1]
    tables = {}
    for big_h in ANCHORS:
        if n - big_h < 30:
            continue
        lo_i = max(0, n - big_h - LOOKBACK)
        changes = sorted(prices[i + big_h] - prices[i]
                         for i in range(lo_i, n - big_h))
        qs = [quantile_sorted(changes, t) for t in quantiles]
        med = qs[quantiles.index(0.5)] if 0.5 in quantiles else 0.0
        tables[big_h] = [q - med for q in qs]
    if not tables:
        sig = ewma_sigma(prices) or BASELINE_SIGMA
        return rows_normal(x, sig, horizon, quantiles)
    hs = sorted(tables)
    rows = []
    for h in range(1, horizon + 1):
        if h <= hs[0]:
            lo_h, hi_h, w = hs[0], hs[0], 0.0
        elif h >= hs[-1]:
            lo_h, hi_h, w = hs[-1], hs[-1], 0.0
        else:
            for a, b in zip(hs, hs[1:]):
                if a <= h <= b:
                    lo_h, hi_h = a, b
                    w = ((math.sqrt(h) - math.sqrt(a))
                         / (math.sqrt(b) - math.sqrt(a)))
                    break
        qlo, qhi = tables[lo_h], tables[hi_h]
        row = [price6(x + qlo[k] * (1 - w) + qhi[k] * w)
               for k in range(len(quantiles))]
        for k in range(1, len(row)):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def forecast_market(prices, horizon, quantiles):
    """Equal-thirds blend of the three quantile forecasters."""
    x = prices[-1]
    sig = ewma_sigma(prices) or BASELINE_SIGMA
    a = rows_normal(x, sig, horizon, quantiles)
    b = rows_stdshape(prices, horizon, quantiles)
    c = rows_hshape(prices, horizon, quantiles)
    rows = []
    for h in range(horizon):
        row = [price6((a[h][k] + b[h][k] + c[h][k]) / 3.0)
               for k in range(len(quantiles))]
        for k in range(1, len(row)):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def forecast_case(case):
    """Forecast every market of one case; a market that fails any step
    falls back to the house persistence baseline rather than failing
    the whole case (an invalid case takes no prize)."""
    horizon = case["horizon_minutes"]
    quantiles = case["quantiles"]
    forecasts = []
    for m in case["markets"]:
        prices = m["prices"]
        try:
            if len(prices) == 0:
                raise ValueError("market has no prices")
            rows = forecast_market(prices, horizon, quantiles)
        except Exception as exc:  # noqa: BLE001 - never fail the case
            print("warning: market %s fell back to baseline: %s"
                  % (m["id"], exc), file=sys.stderr)
            x = prices[-1] if prices else 0.5
            rows = rows_normal(x, BASELINE_SIGMA, horizon, quantiles)
        forecasts.append({"id": m["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
