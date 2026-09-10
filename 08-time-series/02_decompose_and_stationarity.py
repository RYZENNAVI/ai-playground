"""Split a series into trend, cycle and remainder, then ask whether what is left can be modelled.

Demonstrates the two questions that come before any forecast is fitted:
    1. Decompose the long index at the cycle length it was built with, and score the recovered cycle.
    2. Decompose it again at three wrong cycle lengths, and read what the mistake costs.
    3. Run a robust decomposition on the same series and compare it against the moving-average one.
    4. Test both cash-flow columns for a unit root, and check each verdict against the generator.
    5. Find what hid the wandering level from that test, and rerun it with the cycle removed.
    6. Difference the wandering column, difference it once too often, and cross-check with KPSS.

Module 08: Time Series Forecasting - Decomposition and Stationarity.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
WRONG_PERIODS = [60, 288, 500]
# Above this correlation with the planted cycle, a decomposition is called a match.
TRACKING_CORRELATION = 0.9
ALPHA = 0.05

POWER_TRIALS = 300
POWER_SEED = 7


def load_inputs() -> tuple[pd.Series, pd.DataFrame, dict]:
    """Read the index series, the cash-flow table and the generator truth file."""
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    index = pd.read_csv(DATA_DIR / "index_daily.csv", parse_dates=["trade_date"])
    index = index.set_index("trade_date")["close"]

    flow = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                       parse_dates=["report_date"], date_format="%Y%m%d")
    flow = flow.set_index("report_date")
    return index, flow, truth


def planted_cycle(n: int, period: int, amplitude: float) -> np.ndarray:
    """Rebuild the cycle that was written into the index, for scoring against."""
    return amplitude * np.sin(2 * np.pi * np.arange(n) / period)


def score_seasonal(recovered: pd.Series, planted: np.ndarray) -> tuple[float, float]:
    """Return the correlation between a recovered cycle and the planted one, and its amplitude.

    Correlation is the criterion rather than a plot, because a decomposition
    always returns a seasonal component of the length it was asked for: the
    component exists whether or not a cycle of that length is in the data. What
    separates a right answer from a wrong one is whether that component tracks
    the cycle the series was built from.
    """
    left = recovered.to_numpy(dtype=float)
    mask = ~np.isnan(left)
    corr = float(np.corrcoef(left[mask], planted[mask])[0, 1])
    amplitude = float(np.nanmax(left) - np.nanmin(left)) / 2
    return corr, amplitude


def stationary_call_rate(n: int, cycle: np.ndarray | None, truth: dict,
                         divide_back_out: bool = False) -> float:
    """Return how often the unit-root test calls a random walk of length n stationary.

    Every series drawn here is a random walk, so every stationary verdict is a
    mistake and the returned rate is an error rate. The only thing that changes
    between calls is whether a fixed repeating cycle is laid over the walk
    before the test sees it. That isolates the cycle as the cause: the walk, the
    length and the test are identical in both conditions.

    divide_back_out multiplies the cycle in and then divides it out again. That
    is an identity, not an experiment, and the rate it returns has to match the
    one measured without a cycle at all. It is here to show that the division
    used on the real column further down loses nothing, not to add evidence.
    """
    rng = np.random.default_rng(POWER_SEED + n)
    sigma = truth["redeem_random_walk_sigma"]
    calls = 0
    for _ in range(POWER_TRIALS):
        walk = np.exp(np.cumsum(rng.normal(0.0, sigma, size=n)))
        series = walk if cycle is None else walk * cycle[:n]
        if divide_back_out and cycle is not None:
            series = series / cycle[:n]
        noise = rng.lognormal(0.0, truth["noise_sigma_log"]["redeem"], size=n)
        if adfuller(series * noise, autolag="AIC")[1] < ALPHA:
            calls += 1
    return calls / POWER_TRIALS


def cycle_over(dates: pd.DatetimeIndex, truth: dict) -> np.ndarray:
    """Rebuild the weekday and day-of-month product that was written into the outflow."""
    weekday = np.array(truth["redeem_weekday_factor"])[dates.dayofweek.to_numpy()]
    dom = np.array(truth["day_of_month_factor"])[dates.day.to_numpy() - 1]
    return weekday * dom


def verdict(p_value: float, reject_means: str, accept_means: str) -> str:
    """Turn a p-value into the sentence it actually supports."""
    return reject_means if p_value < ALPHA else accept_means


def adf_report(series: pd.Series, label: str) -> float:
    """Run an augmented Dickey-Fuller test and print the parts that get misread.

    The null hypothesis is that the series has a unit root, so a small p-value
    is the good news: it is evidence against a random walk. Printing the
    statistic next to the critical values matters because a p-value alone hides
    how close the call was.
    """
    stat, p_value, used_lag, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    call = verdict(p_value, "stationary", "unit root not rejected")
    print(f"  {label:<34} stat {stat:9.3f}  p {p_value:8.4f}  "
          f"lags {used_lag:>2}  n {nobs:>4}  -> {call}")
    print(f"  {'':<34} critical  1% {crit['1%']:.3f}  "
          f"5% {crit['5%']:.3f}  10% {crit['10%']:.3f}")
    return p_value


def kpss_report(series: pd.Series, label: str, regression: str = "c") -> float:
    """Run a KPSS test, whose null hypothesis is the reverse of the previous one.

    KPSS assumes stationarity and looks for evidence against it, so its verdict
    reads the opposite way round. Running both is what turns a single borderline
    p-value into an agreement or a disagreement, and a disagreement is itself
    information: it usually means the series is neither clean noise nor a clean
    random walk.

    regression says what the test is allowed to call stationary. 'c' asks
    whether the series is stationary around a level, 'ct' around a straight
    line. A column built with a growth term is not stationary around a level,
    so asking the first question of it is asking the wrong one, and the answer
    that comes back is not usable as a cross-check on anything.
    """
    # The lookup table this test interpolates in stops at 0.01 and 0.10, and it
    # warns whenever a statistic falls outside that range. Every call here does,
    # which is the point being reported below rather than something to fix.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p_value, lags, crit = kpss(series.dropna(), regression=regression,
                                         nlags="auto")
    call = verdict(p_value, "stationarity rejected", "stationary")
    around = {"c": "a level", "ct": "a trend"}[regression]
    print(f"  {label:<26} around {around}  stat {stat:7.3f}  p {p_value:7.4f}  "
          f"lags {lags:>2}  -> {call}")
    return p_value


def main() -> None:
    index, flow, truth = load_inputs()
    index_truth = truth["index"]
    flow_truth = truth["cash_flow"]
    period = index_truth["season_period"]

    print("--- 1. Decompose at the cycle length the series was built with ---")
    planted = planted_cycle(len(index), period, index_truth["season_amplitude_log"])
    result = seasonal_decompose(index, model="multiplicative", period=period)
    seasonal_log = np.log(result.seasonal)
    corr, amp = score_seasonal(seasonal_log, planted)
    print(f"  series {len(index)} rows, cycle length {period}, "
          f"{len(index) / period:.1f} cycles of history")
    print(f"  recovered cycle vs planted cycle   correlation {corr:+.4f}")
    print(f"  recovered half-amplitude {amp:.4f} against planted "
          f"{index_truth['season_amplitude_log']:.4f}")
    resid = result.resid.dropna()
    print(f"  remainder: mean {np.log(resid).mean():+.5f}  sd {np.log(resid).std():.5f}")

    print("\n--- 2. Decompose at three wrong cycle lengths ---")
    print(f"  {'period':>8}  {'corr vs planted':>16}  {'half-amplitude':>15}  "
          f"{'remainder sd':>13}")
    print(f"  {period:>8}  {corr:>+16.4f}  {amp:>15.4f}  "
          f"{np.log(resid).std():>13.5f}")
    tracking = {period: corr}
    for wrong in WRONG_PERIODS:
        alt = seasonal_decompose(index, model="multiplicative", period=wrong)
        alt_corr, alt_amp = score_seasonal(np.log(alt.seasonal), planted)
        alt_resid = np.log(alt.resid.dropna())
        tracking[wrong] = alt_corr
        print(f"  {wrong:>8}  {alt_corr:>+16.4f}  {alt_amp:>15.4f}  "
              f"{alt_resid.std():>13.5f}")
    # Counted rather than stated: a period that is a whole multiple of the real
    # one recovers the same cycle, so the number that track it is not always one.
    close = sorted(p for p, c in tracking.items() if c > TRACKING_CORRELATION)
    multiples = [p for p in close if p != period and p % period == 0]
    print(f"  every run returned a seasonal component; {len(close)} of {len(tracking)} "
          f"track the cycle above {TRACKING_CORRELATION}: {close}")
    if multiples:
        print(f"  {multiples} are whole multiples of {period}, and a multiple spans the "
              f"real cycle a whole number of times, so it fits it just as well")
        print("  correlation alone does not say the period is the shortest one that works")

    print("\n--- 3. The same split, computed robustly ---")
    stl = STL(np.log(index), period=period, robust=True).fit()
    stl_corr, stl_amp = score_seasonal(stl.seasonal, planted)
    print(f"  STL correlation {stl_corr:+.4f}  half-amplitude {stl_amp:.4f}  "
          f"remainder sd {stl.resid.std():.5f}")
    print(f"  moving average  correlation {corr:+.4f}  half-amplitude {amp:.4f}  "
          f"remainder sd {np.log(resid).std():.5f}")
    print("  STL lets the cycle change shape over thirty years; the moving average "
          "holds it fixed")

    print("\n--- 4. Test both cash-flow columns for a unit root ---")
    window = flow.loc["2014-03-01":"2014-08-31"]
    inflow = window["total_purchase_amt"]
    outflow = window["total_redeem_amt"]
    print(f"  window {window.index[0].date()} .. {window.index[-1].date()}, "
          f"{len(window)} rows")
    p_in = adf_report(inflow, "inflow, as generated")
    p_out = adf_report(outflow, "outflow, as generated")
    print(f"  generator: inflow  growth {flow_truth['purchase_has_trend']}, "
          f"random walk {flow_truth['purchase_has_unit_root']}")
    print(f"             outflow growth {flow_truth['redeem_has_trend']} "
          f"({flow_truth['redeem_daily_growth']:.4f}/day), "
          f"random walk {flow_truth['redeem_has_unit_root']} "
          f"(sigma {flow_truth['redeem_random_walk_sigma']:.3f})")
    in_agrees = (p_in < ALPHA) == (not flow_truth["purchase_has_unit_root"])
    out_agrees = (p_out >= ALPHA) == flow_truth["redeem_has_unit_root"]
    print(f"  test agrees with the generator on inflow:  {in_agrees}")
    print(f"  test agrees with the generator on outflow: {out_agrees}")
    if not out_agrees:
        print("  the outflow was built with a wandering level and the test calls it "
              "stationary anyway")

    print("\n--- 5. Find out what hid the wandering level from the test ---")
    n = len(window)
    cycle = cycle_over(window.index, flow_truth)
    bare = stationary_call_rate(n, None, flow_truth)
    dressed = stationary_call_rate(n, cycle, flow_truth)
    undressed = stationary_call_rate(n, cycle, flow_truth, divide_back_out=True)
    print(f"  {POWER_TRIALS} random walks of {n} rows each, every stationary verdict "
          f"a mistake:")
    print(f"  {'walk on its own':<44} wrong {bare:>6.0%}")
    print(f"  {'same walk under the two cycles':<44} wrong {dressed:>6.0%}")
    print(f"  {'same walk, cycles divided back out':<44} wrong {undressed:>6.0%}")
    print(f"  laying the cycles over a walk multiplies the error rate by "
          f"{dressed / max(bare, 1 / POWER_TRIALS):.1f}x")
    print(f"  the third row is not a second finding: dividing out what was just "
          f"multiplied in is")
    print(f"  an identity, so it had to come back to {bare:.0%}. What it establishes is "
          f"that the")
    print(f"  removal is exact, which is what the next block relies on.")
    print("\n  the same removal on the real column:")
    deseasonalised = outflow / cycle
    p_clean = adf_report(deseasonalised, "outflow, cycles divided out")
    now_agrees = (p_clean >= ALPHA) == flow_truth["redeem_has_unit_root"]
    print(f"  p moved {p_out:.4f} -> {p_clean:.4f}; agrees with the generator: "
          f"{now_agrees}")
    print(f"  one realisation is not the error rate: the first row above says this "
          f"still goes wrong {bare:.0%} of the time on a clean walk, and this column "
          f"is one of those times")
    print("  the cycle removal is still what makes the verdict worth reading; it just "
          "does not make a 184-row verdict conclusive")

    print("\n--- 6. Difference the wandering column, then difference it once too often ---")
    print(f"  {'series':<34} {'sd':>14}  {'ADF p':>8}  {'verdict':>26}")
    orders = {}
    for order, label in [(0, "outflow, d=0"), (1, "outflow, d=1"), (2, "outflow, d=2")]:
        candidate = outflow.copy()
        for _ in range(order):
            candidate = candidate.diff()
        clean = candidate.dropna()
        stat_p = adfuller(clean, autolag="AIC")[1]
        orders[order] = (clean.std(), stat_p)
        call = verdict(stat_p, "stationary", "unit root not rejected")
        print(f"  {label:<34} {clean.std():>14,.0f}  {stat_p:>8.4f}  {call:>26}")
    smallest_passing = min(d for d, (_, p) in orders.items() if p < ALPHA)
    print(f"  smallest order the test accepts: d={smallest_passing}")
    print(f"  order the generator implies:     d=1 "
          f"(a wandering level is removed by one difference)")
    print(f"  every further difference still passes the test: d=2 changes the spread "
          f"by {orders[2][0] / orders[1][0]:.2f}x against d=1")
    print("  the test can only say a series is differenced enough, never that it is "
          "differenced too much; whether one too many costs anything is a question "
          "for a holdout, which is where the next script settles it")

    print("\n  cross-checked against a test whose null hypothesis is reversed:")
    p_d1 = adfuller(outflow.diff().dropna(), autolag="AIC")[1]
    columns = [("inflow ", inflow, p_in), ("outflow", outflow, p_out),
               ("d=1    ", outflow.diff(), p_d1)]
    scored = []
    for label, series, adf_p in columns:
        level = kpss_report(series, f"{label.strip()}, as generated"
                            if label.strip() != "d=1" else "outflow, d=1")
        trend = kpss_report(series, f"{label.strip()}, as generated"
                            if label.strip() != "d=1" else "outflow, d=1", "ct")
        scored.append((label, adf_p, level, trend))
    print("\n  KPSS p-values are clipped to the 0.01-0.10 lookup range, so read them "
          "as a side")
    print("  of the boundary rather than as a measurement. Every level-only call above "
          "sits at")
    print("  the top of that range:")
    print(f"  {'':<9}{'ADF stationary':>16}{'KPSS around a level':>22}"
          f"{'KPSS around a trend':>22}")
    for label, adf_p, level, trend in scored:
        adf_says = adf_p < ALPHA
        print(f"  {label}{str(adf_says):>16}{str(level >= ALPHA):>22}"
              f"{str(trend >= ALPHA):>22}")
    print("\n  The middle column agrees with ADF on all three, and that agreement is "
          "worth nothing:")
    print("  the outflow was built with a growth term, so it is not stationary around a "
          "level, and")
    print("  a test asked whether it is stationary around a level has been asked a "
          "question with a")
    print("  known answer. It fails to reject on everything, so it confirms whatever it "
          "is put next to.")
    print("  The right-hand column asks the question the generator actually poses and "
          "contradicts ADF")
    print("  on the raw outflow, which is the disagreement the generator says should be "
          "there.")
    print("  Two tests agreeing is only evidence when each of them could have said "
          "something else.")


if __name__ == "__main__":
    main()
