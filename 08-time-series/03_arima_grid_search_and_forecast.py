"""This script searches a grid of ARIMA and seasonal ARIMA (SARIMAX) orders by AIC, on a series of
known order and on the monthly retail total, and forecasts from the winner. It then checks what the
search searched, how the forecast is dated and scored, and what one difference too many costs. The
run prints 7 parts:
    1. Search for the order of a series whose generating recursion is known, and check what came back.
    2. Resample one daily series to three coarser scales and read what each one can still show.
    3. Search a seasonal grid on the monthly table and forecast past the end of it.
    4. Truncate the candidate list the way a slice does, and compare the winner against the full search.
    5. Label those forecast values: build the future dates two ways, and audit where they land.
    6. Ask the fitted model for in-sample and out-of-sample values, and keep the two apart.
    7. Price one difference too many on a holdout, then price the cycle none of those models knew.
"""

import json
import sys
import warnings
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Two warning classes are silenced, both deliberately and both narrowly.
# ConvergenceWarning, because every fit reports its own convergence flag in
# the table below, which is a stronger record than a message on stderr. And the
# starting-parameter notices, because enforce_stationarity and
# enforce_invertibility are switched off on purpose so that the whole grid gets
# fitted rather than only the well-behaved corner of it. The filters match
# these messages only, so any other warning still shows.
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive")
warnings.filterwarnings("ignore", message="Too few observations to estimate starting")

DATA_DIR = Path(__file__).resolve().parent / "data"
TRUNCATED_TO = 20
FORECAST_MONTHS = 4
SEASONAL_PERIOD = 12

# The seasonal grid searched in step 3. Named here because step 2 quotes the
# largest number of ARMA coefficients in it, the largest p plus the largest q,
# and that number should come from the grid rather than from a sentence.
GRID_P = range(5)
GRID_D = range(2)
GRID_Q = range(4)
LONGEST_MEMORY = max(GRID_P) + max(GRID_Q)

# The optimiser's iteration budget. statsmodels defaults to 50, at which 34 of
# the 40 seasonal candidates stop unconverged. At 1000, 17 still stop, all on a
# failed line search rather than the cap.
MAX_ITER = 1000


def load_inputs() -> tuple[pd.Series, pd.Series, pd.DataFrame, dict]:
    """Read the ARMA series, the monthly retail table, the cash flow and the truth file."""
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    arma = pd.read_csv(DATA_DIR / "arma_series.csv").set_index("step")["value"]

    retail = pd.read_csv(DATA_DIR / "retail_sales_monthly.csv")
    retail.index = pd.PeriodIndex(retail["month"], freq="M").to_timestamp(how="end").normalize()
    retail = retail["amount"].asfreq("ME")

    flow = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                       parse_dates=["report_date"], date_format="%Y%m%d")
    # The index is contiguous daily, but pandas does not record that on its own.
    # Leaving it unset makes statsmodels infer a frequency and say so on stderr
    # for every fit; stating it is both quieter and one fewer thing inferred.
    flow = flow.set_index("report_date").asfreq("D")
    return arma, retail, flow, truth


def search_orders(series: pd.Series, candidates: list[tuple[int, int, int]],
                  seasonal: tuple[int, int, int, int] | None = None) -> pd.DataFrame:
    """Fit every candidate order and return the whole table, converged fits sorted first.
    An unconverged fit still returns an AIC without raising, so the optimiser's flag is read."""
    rows = []
    for order in candidates:
        try:
            if seasonal is None:
                fit = ARIMA(series, order=order).fit(method_kwargs={"maxiter": MAX_ITER})
            else:
                fit = sm.tsa.statespace.SARIMAX(
                    series, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=MAX_ITER)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append({"order": order, "aic": np.nan, "converged": False,
                         "note": type(exc).__name__})
            continue
        retvals = getattr(fit, "mle_retvals", {})
        converged = bool(retvals.get("converged", True))
        note = ""
        if not converged:
            note = ("hit the iteration cap" if retvals.get("warnflag") == 1
                    else "line search failed")
        rows.append({"order": order, "aic": fit.aic, "converged": converged, "note": note})
    table = pd.DataFrame(rows)
    table = (table.sort_values(["converged", "aic"], ascending=[False, True],
                               kind="stable")
             .reset_index(drop=True))
    return table


def month_ends_by_day_count(last: pd.Timestamp, count: int) -> list[pd.Timestamp]:
    """Step forward by the length of the month the cursor is in: it looks right and drifts.
    From January 31 it lands on March 3 and never labels February."""
    out = []
    cursor = last
    for _ in range(count):
        days_in_month = cursor.days_in_month
        cursor = cursor + timedelta(days=days_in_month)
        out.append(cursor)
    return out


def acf_gap(series: pd.Series, order: tuple[int, int, int], planted: ArmaProcess) -> float:
    """Fit one order and return its largest ACF gap to the planted process over lags 1 to 12."""
    fit = ARIMA(series, order=order).fit(method_kwargs={"maxiter": MAX_ITER})
    fitted = ArmaProcess(np.r_[1, -fit.arparams], np.r_[1, fit.maparams])
    return float(np.abs(fitted.acf(13)[1:] - planted.acf(13)[1:]).max())


def main() -> None:
    arma, retail, flow, truth = load_inputs()
    arma_truth = truth["arma"]
    retail_truth = truth["retail"]

    print("--- 1. Search for an order that was fixed before the data existed ---")
    planted_order = tuple(arma_truth["order"])
    grid = [(p, 0, q) for p, q in product(range(4), range(4))]
    table = search_orders(arma, grid)
    winner = table.loc[0, "order"]
    print(f"  {len(grid)} candidate orders, {table['aic'].notna().sum()} of them fitted, "
          f"{int(table['converged'].sum())} of them converged")
    print(f"  {'order':>12}  {'AIC':>10}  {'gap to best':>12}  {'converged':>10}")
    for _, row in table.head(5).iterrows():
        print(f"  {str(row['order']):>12}  {row['aic']:>10.2f}  "
              f"{row['aic'] - table.loc[0, 'aic']:>12.2f}  {str(row['converged']):>10}")
    print(f"  best by AIC {winner}, planted {planted_order} -> "
          f"labels match: {tuple(winner) == planted_order}")
    within_two = int((table["aic"] <= table.loc[0, "aic"] + 2).sum())
    planted_aic = float(table.loc[table["order"] == planted_order, "aic"].iloc[0])
    print(f"  {within_two - 1} other candidates sit within 2 AIC of the winner, "
          f"and the planted order is {planted_aic - table.loc[0, 'aic']:.2f} behind it")

    planted_process = ArmaProcess(
        np.r_[1, -np.array(arma_truth["ar_coefficients"])],
        np.r_[1, np.array(arma_truth["ma_coefficients"])])
    print(f"  {'order':>12}  {'AIC':>10}  {'max ACF gap to planted, lags 1-12':>36}")
    top_gaps = []
    for _, row in table.head(3).iterrows():
        gap = acf_gap(arma, tuple(row["order"]), planted_process)
        top_gaps.append(gap)
        print(f"  {str(row['order']):>12}  {row['aic']:>10.2f}  {gap:>36.4f}")
    own_gap = acf_gap(arma, planted_order, planted_process)
    print(f"  the top three differ from the planted autocorrelation by at most "
          f"{max(top_gaps):.3f}; the planted order's own fit differs by {own_gap:.3f}, "
          f"and the planted lag-1 value is {planted_process.acf(2)[1]:.3f}")
    print("  an information criterion ranks fits, it does not identify a mechanism: "
          "the order label is not the thing that was recovered, the dynamics are")

    print("\n--- 2. Read the same daily series at four scales ---")
    index = pd.read_csv(DATA_DIR / "index_daily.csv", parse_dates=["trade_date"])
    index = index.set_index("trade_date")["close"]
    index_truth = truth["index"]
    planted_cycle = pd.Series(
        index_truth["season_amplitude_log"]
        * np.sin(2 * np.pi * np.arange(len(index)) / index_truth["season_period"]),
        index=index.index)
    rules = {"day": None, "month": "ME", "quarter": "QE", "year": "YE"}
    scales, kept = {}, {}
    print(f"  {'scale':<9} {'points':>7}  {'sd/mean':>9}  {'largest step':>13}  "
          f"{'planted cycle kept':>19}")
    for name, rule in rules.items():
        series = index if rule is None else index.resample(rule).mean()
        cycle = planted_cycle if rule is None else planted_cycle.resample(rule).mean()
        scales[name] = series
        kept[name] = cycle.std() / planted_cycle.std()
        rel = series.std() / series.mean()
        step = series.diff().abs().max() / series.mean()
        print(f"  {name:<9} {len(series):>7}  {rel:>9.3f}  {step:>13.3f}  {kept[name]:>19.2f}")
    print(f"  the yearly series has only {len(scales['year'])} points: far too little "
          f"history to fit a model with up to {LONGEST_MEMORY} ARMA coefficients and trust "
          f"the result")
    print(f"  aggregating is not free smoothing: the planted cycle repeats every "
          f"{index_truth['season_period']} trading days, close to one year, so the yearly "
          f"mean keeps {kept['year']:.0%} of it")

    print("\n--- 3. Search a seasonal grid on the monthly table and forecast forward ---")
    full_grid = [(p, d, q) for p, d, q in product(GRID_P, GRID_D, GRID_Q)]
    retail_table = search_orders(retail, full_grid,
                                 seasonal=(1, 0, 1, SEASONAL_PERIOD))
    best_order = tuple(retail_table.loc[0, "order"])
    converged_count = int(retail_table["converged"].sum())
    unconverged_best = retail_table[~retail_table["converged"]].sort_values("aic")
    reasons = ", ".join(f"{count} {note}" for note, count
                        in unconverged_best["note"].value_counts().items())
    print(f"  {len(full_grid)} candidates, {converged_count} converged, "
          f"best {best_order}, AIC {retail_table.loc[0, 'aic']:.2f}, "
          f"runner-up gap {retail_table.loc[1, 'aic'] - retail_table.loc[0, 'aic']:.2f}")
    if len(unconverged_best):
        top_bad = unconverged_best.iloc[0]
        print(f"  {len(unconverged_best)} candidates stopped without converging ({reasons}); "
              f"the best AIC among them is {tuple(top_bad['order'])} at {top_bad['aic']:.2f}")
        print(f"  that number is not a score the optimiser arrived at, so it is sorted "
              f"below every converged fit rather than allowed to win")
    model = sm.tsa.statespace.SARIMAX(
        retail, order=best_order, seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        enforce_stationarity=False, enforce_invertibility=False).fit(
            disp=False, maxiter=MAX_ITER)
    forecast = model.get_forecast(steps=FORECAST_MONTHS)
    mean = forecast.predicted_mean
    # conf_int defaults to alpha=0.05, which is a 95% interval. The band printed
    # here is the 80% one, so the level has to be passed rather than assumed.
    band = forecast.conf_int(alpha=0.20)
    print(f"  {'month':<10} {'forecast':>10}  {'80% band':>22}")
    for stamp, value in mean.items():
        lo, hi = band.loc[stamp]
        print(f"  {stamp.strftime('%Y-%m'):<10} {value:>10.1f}  "
              f"{f'{lo:.1f} .. {hi:.1f}':>22}")
    # The generator is (base + slope x step) x month-of-year factor, plus noise.
    # Anchoring a straight line on the last observation would inherit that
    # observation's noise and drop the month factor entirely, so the reference
    # is rebuilt from the generator's own parameters instead.
    future_steps = np.arange(len(retail), len(retail) + FORECAST_MONTHS)
    month_factors = np.asarray(retail_truth["month_of_year_factor"])[
        mean.index.month.to_numpy() - 1]
    planted = (retail_truth["base"]
               + retail_truth["monthly_slope"] * future_steps) * month_factors
    print(f"  the generator is (base {retail_truth['base']} + slope "
          f"{retail_truth['monthly_slope']}/month) x a month-of-year factor, so its "
          f"noiseless")
    print(f"  expectation for these four months is "
          f"{np.round(planted, 1).tolist()}")
    print(f"  mean absolute gap to that expectation: "
          f"{np.abs(mean.to_numpy() - planted).mean():.1f}")
    inside = int(((band.iloc[:, 0].to_numpy() <= planted)
                  & (planted <= band.iloc[:, 1].to_numpy())).sum())
    print(f"  the 80% band covers the generator's expectation in {inside} of the "
          f"{FORECAST_MONTHS} months")

    print("\n--- 4. Truncate the candidate list, and see which order wins then ---")
    truncated = full_grid[:TRUNCATED_TO]
    trunc_table = search_orders(retail, truncated, seasonal=(1, 0, 1, SEASONAL_PERIOD))
    trunc_best = tuple(trunc_table.loc[0, "order"])
    p_values_full = sorted({o[0] for o in full_grid})
    p_values_trunc = sorted({o[0] for o in truncated})
    print(f"  full list      {len(full_grid):>3} candidates, p ranges over {p_values_full}")
    print(f"  truncated list {len(truncated):>3} candidates, p ranges over {p_values_trunc}")
    print(f"  full search best      {best_order}  AIC {retail_table.loc[0, 'aic']:.2f}  "
          f"({int(retail_table['converged'].sum())} of {len(full_grid)} converged)")
    print(f"  truncated search best {trunc_best}  AIC {trunc_table.loc[0, 'aic']:.2f}  "
          f"({int(trunc_table['converged'].sum())} of {len(truncated)} converged)")
    print(f"  same winner: {trunc_best == best_order}")
    dropped = [o for o in full_grid if o not in truncated]
    dropped_scores = retail_table[retail_table["order"].isin(dropped)]
    print(f"  {len(dropped)} candidates never fitted, the best of them "
          f"{tuple(dropped_scores.iloc[0]['order'])} at AIC "
          f"{dropped_scores.iloc[0]['aic']:.2f}, ranked "
          f"{int(dropped_scores.index[0]) + 1} of {len(full_grid)} overall")
    print(f"  itertools.product varies the last factor fastest, so slicing the front "
          f"of the list holds the first factor near its smallest value")
    print("  the truncated run reports a best AIC either way, and nothing in that "
          "number says which orders were never tried: the count of candidates has to "
          "be printed next to the winner or the reader cannot tell these two runs apart")

    # get_forecast dates its values only when the index carries a frequency. Fitted
    # on a plain array, or on a date index with a gap, it numbers them by position,
    # and the caller builds the calendar. A wrong label raises nothing.
    print("\n--- 5. Attach dates to those four values, two ways, and audit both ---")
    print("  statsmodels dated the four values above from the month-end frequency on the index.")
    print("  Fitted on a plain array, or on a date index with a gap, it numbers them by position")
    print("  instead, and the caller builds the dates. These are the two routines people reach for:")
    last = retail.index[-1]
    by_hand = month_ends_by_day_count(last, FORECAST_MONTHS)
    by_offset = pd.date_range(last, periods=FORECAST_MONTHS + 1, freq="ME")[1:]
    print(f"  last observed month end {last.date()}")
    print(f"  stepping by month length {[d.date().isoformat() for d in by_hand]}")
    print(f"  month-end offset         {[d.date().isoformat() for d in by_offset]}")
    off_by = [(d - (d + pd.offsets.MonthEnd(0))).days for d in by_hand]
    print(f"  days off the true month end, by hand: {off_by}")
    print(f"  month-end offset equals the dates statsmodels attached: "
          f"{bool((by_offset == mean.index).all())}")

    # One example is one example. The same routine is run from every start date in
    # the year the forecast lands in, so the failure rate is counted rather than
    # inferred from the four labels above.
    scanned = pd.date_range(f"{last.year}-01-01", f"{last.year}-12-31", freq="D")
    skipped = drifted = 0
    for start in scanned:
        produced = month_ends_by_day_count(start, 6)
        months = [d.year * 12 + d.month for d in [start, *produced]]
        skipped += any(b - a > 1 for a, b in zip(months, months[1:]))
        drifted += any(d != d + pd.offsets.MonthEnd(0) for d in produced)
    print(f"  scanned every start date in {last.year} ({len(scanned)} of them), "
          f"six steps each:")
    print(f"    {'sequences that skip a whole month':<48} {skipped:>4}")
    print(f"    {'sequences that miss the month end at least once':<48} {drifted:>4}")
    if skipped:
        print("  once a month is skipped, every later value is filed under the month after "
              "its own")

    assert len(set(by_offset)) == len(by_offset), "forecast dates must be distinct"
    assert all(d == d + pd.offsets.MonthEnd(0) for d in by_offset), \
        "forecast dates must land on month ends"
    assert (by_offset == mean.index).all(), "forecast dates must match the fitted index"
    print("  the script asserts that the offset labels are distinct, on month ends and equal "
          "to the dates statsmodels attached: a wrong label raises nothing and survives "
          "every join")

    print("\n--- 6. In-sample values and out-of-sample values from the same fit ---")
    in_sample = model.get_prediction(start=retail.index[-12])
    resid = retail.loc[retail.index[-12]:] - in_sample.predicted_mean
    in_rmse = float(np.sqrt((resid ** 2).mean()))
    holdout_fit = sm.tsa.statespace.SARIMAX(
        retail.iloc[:-12], order=best_order, seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        enforce_stationarity=False, enforce_invertibility=False).fit(
            disp=False, maxiter=MAX_ITER)
    out_pred = holdout_fit.get_forecast(steps=12).predicted_mean
    out_rmse = float(np.sqrt(((retail.iloc[-12:].to_numpy() - out_pred.to_numpy()) ** 2).mean()))
    # The holdout parameters run one step ahead over the full series, so this row
    # differs from the first only in whether the parameters saw those months.
    one_step = holdout_fit.apply(retail).get_prediction(start=retail.index[-12])
    one_rmse = float(np.sqrt(((retail.iloc[-12:] - one_step.predicted_mean) ** 2).mean()))
    print(f"  {'last 12 months, one step ahead, parameters that saw them':<64} "
          f"RMSE {in_rmse:8.2f}")
    print(f"  {'last 12 months, one step ahead, parameters that did not':<64} "
          f"RMSE {one_rmse:8.2f}")
    print(f"  {'last 12 months, forecast 1 to 12 steps by a model that did not':<64} "
          f"RMSE {out_rmse:8.2f}")
    print(f"  the first two rows differ only in whether the parameters saw those months "
          f"({one_rmse / in_rmse:.1f}x)")
    if one_rmse >= out_rmse:
        print("  so the gap comes from seeing the data, not from forecasting further ahead")
    else:
        print("  part of the gap comes from forecasting further ahead")

    print("\n--- 7. Price the extra difference the test could not rule out ---")
    window = flow.loc["2014-03-01":"2014-08-31"]["total_redeem_amt"]
    train, test = window.iloc[:-30], window.iloc[-30:]
    print(f"  train {len(train)} rows, holdout {len(test)} rows, "
          f"forecast horizon {len(test)} days")
    print(f"  {'order':>22}  {'AIC':>10}  {'holdout RMSE':>14}  {'vs best':>9}")
    scores = {}
    for d in [0, 1, 2]:
        fit = ARIMA(train, order=(2, d, 2)).fit(method_kwargs={"maxiter": MAX_ITER})
        pred = fit.forecast(steps=len(test))
        scores[f"(2, {d}, 2)"] = (
            fit.aic, float(np.sqrt(((test.to_numpy() - pred.to_numpy()) ** 2).mean())))
    weekly = sm.tsa.statespace.SARIMAX(
        train, order=(2, 0, 2), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False).fit(
            disp=False, maxiter=MAX_ITER)
    weekly_pred = weekly.get_forecast(steps=len(test)).predicted_mean
    scores["(2, 0, 2) x (1,0,1,7)"] = (
        weekly.aic,
        float(np.sqrt(((test.to_numpy() - weekly_pred.to_numpy()) ** 2).mean())))
    best_rmse = min(rmse for _, rmse in scores.values())
    for label, (aic, rmse) in scores.items():
        print(f"  {label:>22}  {aic:>10.1f}  {rmse:>14,.0f}  {rmse / best_rmse:>8.2f}x")
    d_rmse = [rmse for _, rmse in list(scores.values())[:3]]
    d_span = max(d_rmse) - min(d_rmse)
    weekly_cut = scores["(2, 0, 2)"][1] - scores["(2, 0, 2) x (1,0,1,7)"][1]
    print(f"  AIC is printed as fit information only. It is computed on the training "
          f"rows and is")
    print(f"  not comparable across different d in any case, because differencing "
          f"changes the series")
    print(f"  being scored. The criterion used here is the holdout RMSE: 30 days no "
          f"model saw.")
    print(f"  the three choices of d span {d_span / 1e6:.1f} million on that criterion; "
          f"adding the weekly cycle to (2, 0, 2) cut it by {weekly_cut / 1e6:.1f} million")
    if weekly_cut > d_span:
        print("  the order of differencing was the wrong knob to argue over: the largest "
              "structure in this column repeats every seven days, and none of the first "
              "three models were told that")


if __name__ == "__main__":
    main()
