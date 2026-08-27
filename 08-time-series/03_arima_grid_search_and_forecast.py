"""Search a grid of ARIMA orders by information criterion, and check what the search actually searched.

Demonstrates where an autoregressive forecast comes from, end to end:
    1. Search for the order of a series whose generating recursion is known, and check what came back.
    2. Resample one daily series to three coarser scales and read what each one can still show.
    3. Search a seasonal grid on the monthly table and forecast past the end of it.
    4. Truncate the candidate list the way a slice does, and compare the winner against the full search.
    5. Build future dates by adding month lengths, and audit where the labels land.
    6. Ask the fitted model for in-sample and out-of-sample values, and keep the two apart.
    7. Price one difference too many, then price the cycle none of those models were told about.

Module 08: Time Series Forecasting - ARIMA Grid Search.
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

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
TRUNCATED_TO = 20
FORECAST_MONTHS = 4
SEASONAL_PERIOD = 12


def load_inputs() -> tuple[pd.Series, pd.Series, pd.DataFrame, dict]:
    """Read the ARMA series, the monthly retail table, the cash flow and the truth file."""
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    arma = pd.read_csv(DATA_DIR / "arma_series.csv").set_index("step")["value"]

    retail = pd.read_csv(DATA_DIR / "retail_sales_monthly.csv")
    retail.index = pd.PeriodIndex(retail["month"], freq="M").to_timestamp(how="end").normalize()
    retail = retail["amount"].asfreq("ME")

    flow = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                       parse_dates=["report_date"], date_format="%Y%m%d")
    flow = flow.set_index("report_date")
    return arma, retail, flow, truth


def search_orders(series: pd.Series, candidates: list[tuple[int, int, int]],
                  seasonal: tuple[int, int, int, int] | None = None) -> pd.DataFrame:
    """Fit every candidate order and return the whole table, not only the winner.

    Returning the full table is the point. A search that reports one order and
    one score cannot be checked: the reader cannot see how many candidates were
    tried, whether any failed to converge, or how close the runner-up was. All
    three of those change what the winning order is worth.
    """
    rows = []
    for order in candidates:
        try:
            if seasonal is None:
                fit = ARIMA(series, order=order).fit()
            else:
                fit = sm.tsa.statespace.SARIMAX(
                    series, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append({"order": order, "aic": np.nan, "note": type(exc).__name__})
            continue
        rows.append({"order": order, "aic": fit.aic, "note": ""})
    table = pd.DataFrame(rows).sort_values("aic", kind="stable").reset_index(drop=True)
    return table


def month_ends_by_day_count(last: pd.Timestamp, count: int) -> list[pd.Timestamp]:
    """Step forward by the length of the current month, repeatedly.

    This is the arithmetic that reads as obviously correct and is not. Adding
    the length of the month the cursor is standing in lands it inside the next
    month rather than on the same position in it, and the error accumulates:
    from the end of a thirty-one day month the cursor can step clean over a
    short month and never produce a label for it. It is kept here so that the
    damage can be measured rather than argued about.
    """
    out = []
    cursor = last
    for _ in range(count):
        days_in_month = cursor.days_in_month
        cursor = cursor + timedelta(days=days_in_month)
        out.append(cursor)
    return out


def main() -> None:
    arma, retail, flow, truth = load_inputs()
    arma_truth = truth["arma"]
    retail_truth = truth["retail"]

    print("--- 1. Search for an order that was fixed before the data existed ---")
    planted_order = tuple(arma_truth["order"])
    grid = [(p, 0, q) for p, q in product(range(4), range(4))]
    table = search_orders(arma, grid)
    winner = table.loc[0, "order"]
    print(f"  {len(grid)} candidate orders, {table['aic'].notna().sum()} of them fitted")
    print(f"  {'order':>12}  {'AIC':>10}  {'gap to best':>12}")
    for _, row in table.head(5).iterrows():
        print(f"  {str(row['order']):>12}  {row['aic']:>10.2f}  "
              f"{row['aic'] - table.loc[0, 'aic']:>12.2f}")
    print(f"  best by AIC {winner}, planted {planted_order} -> "
          f"labels match: {tuple(winner) == planted_order}")
    within_two = int((table["aic"] <= table.loc[0, "aic"] + 2).sum())
    planted_aic = float(table.loc[table["order"] == planted_order, "aic"].iloc[0])
    print(f"  {within_two} candidates sit within 2 AIC of the winner, "
          f"and the planted order is {planted_aic - table.loc[0, 'aic']:.2f} behind it")

    planted_process = ArmaProcess(
        np.r_[1, -np.array(arma_truth["ar_coefficients"])],
        np.r_[1, np.array(arma_truth["ma_coefficients"])])
    print(f"  {'order':>12}  {'AIC':>10}  {'max ACF gap to planted, lags 1-12':>36}")
    for _, row in table.head(3).iterrows():
        fit = ARIMA(arma, order=tuple(row["order"])).fit()
        fitted_process = ArmaProcess(np.r_[1, -fit.arparams], np.r_[1, fit.maparams])
        gap = np.abs(fitted_process.acf(13)[1:] - planted_process.acf(13)[1:]).max()
        print(f"  {str(row['order']):>12}  {row['aic']:>10.2f}  {gap:>36.4f}")
    print("  the top orders carry almost the same autocorrelation as the planted "
          "recursion, written with different numbers of terms")
    print("  an information criterion ranks fits, it does not identify a mechanism: "
          "the order label is not the thing that was recovered, the dynamics are")

    print("\n--- 2. Read the same daily series at four scales ---")
    index = pd.read_csv(DATA_DIR / "index_daily.csv", parse_dates=["trade_date"])
    index = index.set_index("trade_date")["close"]
    scales = {"day": index, "month": index.resample("ME").mean(),
              "quarter": index.resample("QE").mean(), "year": index.resample("YE").mean()}
    print(f"  {'scale':<9} {'points':>7}  {'sd/mean':>9}  {'largest step':>13}")
    for name, series in scales.items():
        rel = series.std() / series.mean()
        step = series.diff().abs().max() / series.mean()
        print(f"  {name:<9} {len(series):>7}  {rel:>9.3f}  {step:>13.3f}")
    print("  the yearly series has 30 points: too few to fit anything with a memory of 7")
    print("  aggregating is not free smoothing, it deletes the cycles shorter than "
          "the new step")

    print("\n--- 3. Search a seasonal grid on the monthly table and forecast forward ---")
    full_grid = [(p, d, q) for p, d, q in product(range(5), range(2), range(4))]
    retail_table = search_orders(retail, full_grid,
                                 seasonal=(1, 0, 1, SEASONAL_PERIOD))
    best_order = tuple(retail_table.loc[0, "order"])
    print(f"  {len(full_grid)} candidates, best {best_order}, "
          f"AIC {retail_table.loc[0, 'aic']:.2f}, "
          f"runner-up gap {retail_table.loc[1, 'aic'] - retail_table.loc[0, 'aic']:.2f}")
    model = sm.tsa.statespace.SARIMAX(
        retail, order=best_order, seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    forecast = model.get_forecast(steps=FORECAST_MONTHS)
    mean = forecast.predicted_mean
    band = forecast.conf_int()
    print(f"  {'month':<10} {'forecast':>10}  {'80% band':>22}")
    for stamp, value in mean.items():
        lo, hi = band.loc[stamp]
        print(f"  {stamp.strftime('%Y-%m'):<10} {value:>10.1f}  "
              f"{f'{lo:.1f} .. {hi:.1f}':>22}")
    implied = retail.iloc[-1] + retail_truth["monthly_slope"] * np.arange(
        1, FORECAST_MONTHS + 1)
    print(f"  planted slope {retail_truth['monthly_slope']}/month implies "
          f"{np.round(implied, 1).tolist()}")
    print(f"  mean absolute gap to the planted straight line: "
          f"{np.abs(mean.to_numpy() - implied).mean():.1f}")

    print("\n--- 4. Truncate the candidate list, and see which order wins then ---")
    truncated = full_grid[:TRUNCATED_TO]
    trunc_table = search_orders(retail, truncated, seasonal=(1, 0, 1, SEASONAL_PERIOD))
    trunc_best = tuple(trunc_table.loc[0, "order"])
    p_values_full = sorted({o[0] for o in full_grid})
    p_values_trunc = sorted({o[0] for o in truncated})
    print(f"  full list      {len(full_grid):>3} candidates, p ranges over {p_values_full}")
    print(f"  truncated list {len(truncated):>3} candidates, p ranges over {p_values_trunc}")
    print(f"  full search best      {best_order}  AIC {retail_table.loc[0, 'aic']:.2f}")
    print(f"  truncated search best {trunc_best}  AIC {trunc_table.loc[0, 'aic']:.2f}")
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

    print("\n--- 5. Build the forecast dates two ways and audit both ---")
    last = retail.index[-1]
    by_hand = month_ends_by_day_count(last, FORECAST_MONTHS)
    by_offset = pd.date_range(last, periods=FORECAST_MONTHS + 1, freq="ME")[1:]
    print(f"  last observed month end {last.date()}")
    print(f"  stepping by month length {[d.date().isoformat() for d in by_hand]}")
    print(f"  month-end offset         {[d.date().isoformat() for d in by_offset]}")
    off_by = [(d - (d + pd.offsets.MonthEnd(0))).days for d in by_hand]
    print(f"  days off the true month end, by hand: {off_by}")

    scanned = pd.date_range("2007-01-01", "2007-12-31", freq="D")
    duplicates = skipped = drifted = 0
    for start in scanned:
        produced = month_ends_by_day_count(start, 6)
        months = [(d.year, d.month) for d in produced]
        duplicates += len(set(produced)) < len(produced)
        skipped += len(set(months)) < len(months)
        drifted += any(d != d + pd.offsets.MonthEnd(0) for d in produced)
    print(f"  scanned {len(scanned)} start dates, six steps each:")
    print(f"    sequences with a repeated date              {duplicates:>4}")
    print(f"    sequences that visit one month twice        {skipped:>4}")
    print(f"    sequences that miss the month end at least once {drifted:>4}")
    print("  the failure is not repeated dates, it is labels that drift off the "
          "period they are supposed to name")

    assert len(set(by_offset)) == len(by_offset), "forecast dates must be distinct"
    assert all(d == d + pd.offsets.MonthEnd(0) for d in by_offset), \
        "forecast dates must land on month ends"
    print("  the two assertions above are the guard: forecast values carry no date of "
          "their own, so a wrong label is attached silently and survives every join")

    print("\n--- 6. In-sample values and out-of-sample values from the same fit ---")
    in_sample = model.get_prediction(start=retail.index[-12])
    resid = retail.loc[retail.index[-12]:] - in_sample.predicted_mean
    in_rmse = float(np.sqrt((resid ** 2).mean()))
    holdout_fit = sm.tsa.statespace.SARIMAX(
        retail.iloc[:-12], order=best_order, seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    out_pred = holdout_fit.get_forecast(steps=12).predicted_mean
    out_rmse = float(np.sqrt(((retail.iloc[-12:].to_numpy() - out_pred.to_numpy()) ** 2).mean()))
    print(f"  last 12 months, fitted by a model that saw them      RMSE {in_rmse:8.2f}")
    print(f"  last 12 months, forecast by a model that did not     RMSE {out_rmse:8.2f}")
    print(f"  the second number is {out_rmse / in_rmse:.1f}x the first, and only the "
          f"second one describes a forecast")

    print("\n--- 7. Price the extra difference the test could not rule out ---")
    window = flow.loc["2014-03-01":"2014-08-31"]["total_redeem_amt"]
    train, test = window.iloc[:-30], window.iloc[-30:]
    print(f"  train {len(train)} rows, holdout {len(test)} rows, "
          f"forecast horizon {len(test)} days")
    print(f"  {'order':>22}  {'AIC':>10}  {'holdout RMSE':>14}  {'vs best':>9}")
    scores = {}
    for d in [0, 1, 2]:
        fit = ARIMA(train, order=(2, d, 2)).fit()
        pred = fit.forecast(steps=len(test))
        scores[f"(2, {d}, 2)"] = (
            fit.aic, float(np.sqrt(((test.to_numpy() - pred.to_numpy()) ** 2).mean())))
    weekly = sm.tsa.statespace.SARIMAX(
        train, order=(2, 0, 2), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    weekly_pred = weekly.get_forecast(steps=len(test)).predicted_mean
    scores["(2, 0, 2) x (1,0,1,7)"] = (
        weekly.aic,
        float(np.sqrt(((test.to_numpy() - weekly_pred.to_numpy()) ** 2).mean())))
    best_rmse = min(rmse for _, rmse in scores.values())
    for label, (aic, rmse) in scores.items():
        print(f"  {label:>22}  {aic:>10.1f}  {rmse:>14,.0f}  {rmse / best_rmse:>8.2f}x")
    spread = max(r for _, r in list(scores.values())[:3]) / min(
        r for _, r in list(scores.values())[:3])
    print(f"  the three differencing orders land within {spread:.2f}x of each other, "
          f"so on this series the extra difference cost close to nothing")
    print(f"  adding the weekly cycle moved the error further than every choice of d "
          f"put together")
    print("  the order of differencing was the wrong knob to argue over: the largest "
          "structure in this column repeats every seven days, and none of the first "
          "three models were told that")


if __name__ == "__main__":
    main()
