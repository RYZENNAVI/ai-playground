"""Generate the five series this module forecasts, each from a mechanism written down in advance.

Demonstrates how to build a series whose every later claim can be checked:
    1. Draw a daily cash-flow table from explicit weekday and day-of-month factors.
    2. Give inflow a flat level and outflow a rising one, so stationarity tests disagree on them.
    3. Draw a long daily index from piecewise drift plus a seasonal cycle of known length.
    4. Draw a short listing history that is too young to contain a yearly cycle at all.
    5. Draw a monthly retail total, and an ARMA series whose order is fixed in advance.
    6. Write every factor, changepoint and coefficient to a truth file the other scripts score against.

Module 08: Time Series Forecasting - Dataset Construction.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 20260827

# Cash-flow panel: a full fourteen months, of which the last six are the modelling window.
FLOW_START = "2013-07-01"
FLOW_END = "2014-08-31"

# Inflow peaks early in the working week and collapses at the weekend.
PURCHASE_WEEKDAY = np.array([1.16, 1.21, 1.11, 1.04, 0.94, 0.71, 0.75])
# Outflow peaks on Monday instead, and its weekend dip is shallower.
REDEEM_WEEKDAY = np.array([1.22, 1.09, 1.04, 1.01, 0.97, 0.81, 0.84])

PURCHASE_BASE = 3.2e8
REDEEM_BASE = 2.9e8
# Outflow grows half a percent a week; inflow does not grow at all.
REDEEM_DAILY_GROWTH = 0.0007
# Outflow also carries a level that never returns to where it started. This, and
# not the growth term above, is what a unit-root test is built to detect.
REDEEM_RANDOM_WALK_SIGMA = 0.035

# Four promotion days that lift inflow well above what the two cycles predict.
CAMPAIGN_DAYS = ["2013-11-11", "2013-12-12", "2014-06-18", "2014-08-08"]
CAMPAIGN_LIFT = 1.55

# Long index: thirty years of trading days, in five drift regimes.
INDEX_START = "1990-12-19"
INDEX_DAYS = 7145
INDEX_CHANGEPOINTS = [900, 2400, 3900, 5600]
INDEX_SLOPES = [0.00092, -0.00021, 0.00058, -0.00034, 0.00041]
INDEX_SEASON_PERIOD = 250
INDEX_SEASON_AMPLITUDE = 0.058
INDEX_AR1 = 0.42

# Short listing: under one trading year, so a yearly cycle cannot be estimated from it.
LISTING_START = "2024-06-12"
LISTING_DAYS = 198
LISTING_OPEN_PRICE = 21.4
LISTING_FIRST_DAY_RETURN = 1.769

# Monthly retail total, three and a half years of it.
RETAIL_START = "2004-01"
RETAIL_MONTHS = 42
RETAIL_BASE = 740.0
RETAIL_MONTHLY_SLOPE = 2.35
RETAIL_MONTH_OF_YEAR = np.array(
    [0.982, 0.968, 0.994, 1.001, 1.006, 1.011, 1.018, 1.021, 1.007, 1.013, 1.028, 1.051]
)

# ARMA series with the order fixed here, so a search can be asked to recover it.
ARMA_AR = [0.62, -0.31]
ARMA_MA = [0.45]
ARMA_POINTS = 320
ARMA_LEVEL = 4800.0
ARMA_SIGMA = 620.0


def day_of_month_factor(day: int) -> float:
    """Return the month-position factor for one calendar day.

    Money moves at the edges of a month and sits still in the middle: the first
    three days and the last two carry the payroll and settlement peaks, and the
    middle of the month is the trough. The shape is a fixed lookup rather than a
    formula so that a fitted factor can be compared against it entry by entry.
    """
    table = {
        1: 1.34, 2: 1.28, 3: 1.19, 4: 1.08, 5: 1.02, 6: 0.98, 7: 0.95,
        8: 0.93, 9: 0.92, 10: 0.91, 11: 0.90, 12: 0.89, 13: 0.89, 14: 0.90,
        15: 0.94, 16: 0.92, 17: 0.90, 18: 0.91, 19: 0.92, 20: 0.94, 21: 0.95,
        22: 0.96, 23: 0.97, 24: 0.99, 25: 1.04, 26: 1.07, 27: 1.09, 28: 1.12,
        29: 1.16, 30: 1.21, 31: 1.26,
    }
    return table[day]


def build_cash_flow(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Build the daily inflow and outflow table from the two published cycles.

    Both columns are a product of the same three ingredients: a base level, a
    weekday factor and a day-of-month factor. The outflow gets two extra terms
    on top: a steady growth rate, and a wandering level built by accumulating
    small shocks. The second of those is the one that makes a unit-root test
    fail, and separating the two is the point of building them separately. The
    multiplicative noise is drawn in log space so that a quiet weekend day and a
    busy month-end day carry the same relative spread.
    """
    dates = pd.date_range(FLOW_START, FLOW_END, freq="D")
    weekday = dates.dayofweek.to_numpy()
    dom = dates.day.to_numpy()

    weekday_pull_p = PURCHASE_WEEKDAY[weekday]
    weekday_pull_r = REDEEM_WEEKDAY[weekday]
    dom_pull = np.array([day_of_month_factor(int(d)) for d in dom])

    horizon = np.arange(len(dates))
    growth = 1.0 + REDEEM_DAILY_GROWTH * horizon
    wander = np.exp(np.cumsum(rng.normal(0.0, REDEEM_RANDOM_WALK_SIGMA, size=len(dates))))

    campaign = np.isin(dates.strftime("%Y-%m-%d"), CAMPAIGN_DAYS)
    campaign_pull = np.where(campaign, CAMPAIGN_LIFT, 1.0)

    noise_p = rng.lognormal(mean=0.0, sigma=0.058, size=len(dates))
    noise_r = rng.lognormal(mean=0.0, sigma=0.064, size=len(dates))

    purchase = PURCHASE_BASE * weekday_pull_p * dom_pull * campaign_pull * noise_p
    redeem = REDEEM_BASE * weekday_pull_r * dom_pull * growth * wander * noise_r

    frame = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y%m%d"),
            "total_purchase_amt": np.round(purchase).astype("int64"),
            "total_redeem_amt": np.round(redeem).astype("int64"),
        }
    )

    truth = {
        "purchase_base": PURCHASE_BASE,
        "redeem_base": REDEEM_BASE,
        "purchase_weekday_factor": PURCHASE_WEEKDAY.tolist(),
        "redeem_weekday_factor": REDEEM_WEEKDAY.tolist(),
        "day_of_month_factor": [day_of_month_factor(d) for d in range(1, 32)],
        "redeem_daily_growth": REDEEM_DAILY_GROWTH,
        "redeem_random_walk_sigma": REDEEM_RANDOM_WALK_SIGMA,
        "purchase_has_trend": False,
        "redeem_has_trend": True,
        "purchase_has_unit_root": False,
        "redeem_has_unit_root": True,
        "campaign_days": CAMPAIGN_DAYS,
        "campaign_lift": CAMPAIGN_LIFT,
        "noise_sigma_log": {"purchase": 0.058, "redeem": 0.064},
    }
    return frame, truth


def build_index(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Build a long daily index from piecewise drift, one seasonal cycle and AR(1) noise.

    The series is assembled in log space and exponentiated at the end, so the
    drift regimes read as constant percentage rates rather than constant point
    moves. The cycle is exactly INDEX_SEASON_PERIOD observations long, which
    gives the decomposition script a period it is supposed to find. The AR(1)
    residual is what stops a differenced series from looking like clean noise.
    """
    dates = pd.bdate_range(INDEX_START, periods=INDEX_DAYS, freq="B")
    steps = np.arange(INDEX_DAYS)

    drift = np.empty(INDEX_DAYS)
    edges = [0, *INDEX_CHANGEPOINTS, INDEX_DAYS]
    for slope, lo, hi in zip(INDEX_SLOPES, edges[:-1], edges[1:]):
        drift[lo:hi] = slope
    log_trend = np.cumsum(drift)

    season = INDEX_SEASON_AMPLITUDE * np.sin(2 * np.pi * steps / INDEX_SEASON_PERIOD)

    shocks = rng.normal(0.0, 0.011, size=INDEX_DAYS)
    resid = np.empty(INDEX_DAYS)
    resid[0] = shocks[0]
    for i in range(1, INDEX_DAYS):
        resid[i] = INDEX_AR1 * resid[i - 1] + shocks[i]

    price = 100.0 * np.exp(log_trend + season + resid)

    frame = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "close": np.round(price, 2),
        }
    )

    truth = {
        "changepoint_index": INDEX_CHANGEPOINTS,
        "changepoint_date": [dates[i].strftime("%Y-%m-%d") for i in INDEX_CHANGEPOINTS],
        "segment_log_slope": INDEX_SLOPES,
        "season_period": INDEX_SEASON_PERIOD,
        "season_amplitude_log": INDEX_SEASON_AMPLITUDE,
        "resid_ar1": INDEX_AR1,
        "rows": INDEX_DAYS,
    }
    return frame, truth


def build_listing(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Build a short listing history with a first-day jump and no yearly cycle.

    Nothing in this generator repeats on a yearly period, and the series is
    under one trading year long, so a model that reports a yearly component on
    it is reporting something that was never put in. The first day carries a
    large opening return, which also makes it a leverage point for any fit that
    is not told to treat it separately.
    """
    dates = pd.bdate_range(LISTING_START, periods=LISTING_DAYS, freq="B")

    price = np.empty(LISTING_DAYS)
    price[0] = LISTING_OPEN_PRICE * (1.0 + LISTING_FIRST_DAY_RETURN)
    anchor = price[0] * 0.62
    for i in range(1, LISTING_DAYS):
        pull = 0.014 * (anchor - price[i - 1])
        price[i] = price[i - 1] + pull + rng.normal(0.0, 0.9)

    frame = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "close": np.round(np.maximum(price, 1.0), 2),
        }
    )

    truth = {
        "rows": LISTING_DAYS,
        "trading_days_per_year": 250,
        "covers_full_year": LISTING_DAYS >= 250,
        "first_day_return": LISTING_FIRST_DAY_RETURN,
        "mean_reversion_anchor": round(float(anchor), 2),
        "yearly_component_present": False,
    }
    return frame, truth


def build_retail(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Build a monthly retail total with a straight trend and a month-of-year cycle.

    Forty-two monthly points is a realistic amount of history for this kind of
    table and a very small amount of data for a seasonal model: twelve monthly
    effects have to be estimated from three and a half observations each. The
    trend is linear on purpose, so a differenced series should come out flat.
    """
    periods = pd.period_range(RETAIL_START, periods=RETAIL_MONTHS, freq="M")
    steps = np.arange(RETAIL_MONTHS)
    month_pull = RETAIL_MONTH_OF_YEAR[periods.month.to_numpy() - 1]
    amount = (RETAIL_BASE + RETAIL_MONTHLY_SLOPE * steps) * month_pull
    amount = amount + rng.normal(0.0, 4.5, size=RETAIL_MONTHS)

    frame = pd.DataFrame(
        {
            "month": periods.strftime("%Y-%m"),
            "amount": np.round(amount, 1),
        }
    )

    truth = {
        "base": RETAIL_BASE,
        "monthly_slope": RETAIL_MONTHLY_SLOPE,
        "month_of_year_factor": RETAIL_MONTH_OF_YEAR.tolist(),
        "months": RETAIL_MONTHS,
        "observations_per_month_effect": round(RETAIL_MONTHS / 12, 2),
    }
    return frame, truth


def build_arma(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Build a series from an ARMA recursion whose order and coefficients are fixed above.

    The recursion is written out term by term rather than called from a library,
    so the order that a later search is asked to recover is visible in this file.
    A burn-in of two hundred points is dropped, because the first values still
    carry the arbitrary zero state the recursion was started from.
    """
    burn = 200
    total = ARMA_POINTS + burn
    eps = rng.normal(0.0, ARMA_SIGMA, size=total)
    y = np.zeros(total)
    for t in range(2, total):
        ar_part = ARMA_AR[0] * y[t - 1] + ARMA_AR[1] * y[t - 2]
        ma_part = ARMA_MA[0] * eps[t - 1]
        y[t] = ar_part + ma_part + eps[t]
    series = ARMA_LEVEL + y[burn:]

    frame = pd.DataFrame(
        {
            "step": np.arange(ARMA_POINTS),
            "value": np.round(series, 2),
        }
    )

    truth = {
        "ar_coefficients": ARMA_AR,
        "ma_coefficients": ARMA_MA,
        "order": [len(ARMA_AR), 0, len(ARMA_MA)],
        "level": ARMA_LEVEL,
        "innovation_sigma": ARMA_SIGMA,
        "points": ARMA_POINTS,
        "burn_in_dropped": burn,
    }
    return frame, truth


def write_frame(frame: pd.DataFrame, name: str) -> Path:
    """Write one table to the data directory and report its shape and size."""
    path = DATA_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8")
    size_kb = path.stat().st_size / 1024
    print(f"  {name:<28} {len(frame):>6} rows  {frame.shape[1]} cols  {size_kb:8.1f} KB")
    return path


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    truth: dict = {"seed": SEED}

    print("--- 1. Daily cash flow, from two multiplicative cycles ---")
    flow, truth["cash_flow"] = build_cash_flow(rng)
    write_frame(flow, "fund_flow_daily.csv")
    print(f"  window        {FLOW_START} .. {FLOW_END}")
    print(f"  weekday pull  inflow  Mon..Sun {PURCHASE_WEEKDAY}")
    print(f"                outflow Mon..Sun {REDEEM_WEEKDAY}")
    print(f"  month-edge    day 1 {day_of_month_factor(1):.2f}  "
          f"day 13 {day_of_month_factor(13):.2f}  day 31 {day_of_month_factor(31):.2f}")

    print("\n--- 2. One column wanders, the other does not ---")
    print(f"  inflow  growth per day  0.0000   random-walk sigma  0.000  -> level series")
    print(f"  outflow growth per day  {REDEEM_DAILY_GROWTH:.4f}   "
          f"random-walk sigma  {REDEEM_RANDOM_WALK_SIGMA:.3f}  "
          f"-> level never returns")
    print("  the growth term alone would still leave a series that tests stationary "
          "around its own trend")
    print("  the random-walk term is the one a unit-root test is built to catch, "
          "when it has enough data to catch it")

    print("\n--- 3. Long daily index, five drift regimes and one cycle ---")
    index, truth["index"] = build_index(rng)
    write_frame(index, "index_daily.csv")
    for date, slope in zip(truth["index"]["changepoint_date"], INDEX_SLOPES[1:]):
        print(f"  changepoint {date}  new log slope {slope:+.5f}")
    print(f"  seasonal cycle length {INDEX_SEASON_PERIOD} observations")

    print("\n--- 4. Short listing history, no yearly cycle to find ---")
    listing, truth["listing"] = build_listing(rng)
    write_frame(listing, "new_listing_daily.csv")
    print(f"  {LISTING_DAYS} rows against {truth['listing']['trading_days_per_year']} "
          f"trading days in a year -> covers a full year: "
          f"{truth['listing']['covers_full_year']}")
    print(f"  first-day return {LISTING_FIRST_DAY_RETURN:+.1%}")

    print("\n--- 5. Monthly retail total, and an ARMA series of known order ---")
    retail, truth["retail"] = build_retail(rng)
    write_frame(retail, "retail_sales_monthly.csv")
    print(f"  linear slope {RETAIL_MONTHLY_SLOPE} per month, "
          f"{truth['retail']['observations_per_month_effect']} observations "
          f"behind each month-of-year effect")
    arma, truth["arma"] = build_arma(rng)
    write_frame(arma, "arma_series.csv")
    print(f"  ARMA order {tuple(truth['arma']['order'])}  "
          f"AR {ARMA_AR}  MA {ARMA_MA}")

    print("\n--- 6. The truth file the other scripts are scored against ---")
    truth_path = DATA_DIR / "ground_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"  {truth_path.name} holds {len(truth) - 1} generator records:")
    for key in truth:
        if key == "seed":
            continue
        print(f"    {key:<12} {len(truth[key])} recorded quantities")
    print(f"\n  seed {SEED} is fixed, so rerunning this script rewrites identical files")


if __name__ == "__main__":
    main()
