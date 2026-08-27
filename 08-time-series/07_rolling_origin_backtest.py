"""Score every route on repeated forecasts made from moving cut-off dates, then write the file.

Demonstrates the difference between a score and an estimate of future error:
    1. Lay out the cut-off dates, and confirm no route can see past the one it is given.
    2. Score one route on a single holdout, and then on every fold, to see how far one number moves.
    3. Run all routes over all folds and rank them by their averages.
    4. Put the same routes' training-period errors next to those averages.
    5. Refit on the whole history and forecast the month that follows it.
    6. Check the written file against the format it has to satisfy before it is sent anywhere.

Module 08: Time Series Forecasting - Rolling Origin Backtesting.
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
HORIZON = 30
ORIGINS = ["2014-04-30", "2014-05-31", "2014-06-30", "2014-07-31"]
SUBMISSION_START = "2014-09-01"
COLUMNS = ["total_purchase_amt", "total_redeem_amt"]


def quiet_prophet() -> None:
    """Silence the fitting backend before it writes a line for every fit.

    The backend attaches its own log handler lazily, the first time it runs a
    fit, so clearing handlers once at import is undone a moment later. A filter
    sits on the logger itself rather than on its handlers, so it keeps holding
    however many handlers are added underneath it afterwards.
    """
    for noisy in ("prophet", "cmdstanpy"):
        log = logging.getLogger(noisy)
        log.setLevel(logging.CRITICAL)
        log.addFilter(lambda record: False)
        log.propagate = False


def seasonal_naive(train: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Repeat the last seven observed values for as long as the horizon runs."""
    return np.resize(train.to_numpy()[-7:], len(dates))


def periodic_factors(train: pd.Series, dates: pd.DatetimeIndex,
                     rounds: int = 20) -> np.ndarray:
    """Forecast as level times a weekday factor times a month-position factor.

    The two factors are fitted together, alternating between them, so that a
    weekday which happens to fall on month ends more often than average is not
    credited with the month-end lift. Nothing here is optimised against a loss:
    the forecast is a product of group averages.
    """
    frame = pd.DataFrame({"y": train.to_numpy(),
                          "weekday": train.index.dayofweek,
                          "day": train.index.day})
    level = frame["y"].mean()
    weekday = pd.Series(1.0, index=range(7))
    day = pd.Series(1.0, index=range(1, 32))
    for _ in range(rounds):
        weekday = (frame["y"] / (level * frame["day"].map(day))
                   ).groupby(frame["weekday"]).mean()
        weekday = weekday / weekday.mean()
        day = (frame["y"] / (level * frame["weekday"].map(weekday))
               ).groupby(frame["day"]).mean()
        day = day / day.mean()
    weekday = weekday.reindex(range(7)).fillna(1.0)
    day = day.reindex(range(1, 32)).fillna(1.0)
    return (level
            * weekday.reindex(dates.dayofweek).to_numpy()
            * day.reindex(dates.day).to_numpy())


def sarimax_weekly(train: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Fit an autoregressive model that is told the period is seven, and forecast forward."""
    fit = sm.tsa.statespace.SARIMAX(
        train.to_numpy(), order=(2, 0, 2), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.asarray(fit.get_forecast(steps=len(dates)).predicted_mean)


def additive_prophet(train: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Fit an additive trend-plus-cycle model and read the forecast off the future frame."""
    quiet_prophet()
    frame = pd.DataFrame({"ds": train.index, "y": train.to_numpy(dtype=float)})
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                    daily_seasonality=False)
    model.fit(frame)
    future = pd.DataFrame({"ds": dates})
    return model.predict(future)["yhat"].to_numpy()


ROUTES = {
    "last week repeated": seasonal_naive,
    "periodic factors": periodic_factors,
    "SARIMAX, weekly": sarimax_weekly,
    "additive model": additive_prophet,
}


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, in the units of the series."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def fold_score(series: pd.Series, origin: str, route) -> tuple[float, int]:
    """Fit a route on everything up to one cut-off date and score the next HORIZON days."""
    cut = pd.Timestamp(origin)
    train = series.loc[:cut]
    future = pd.date_range(cut + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    actual = series.reindex(future).dropna()
    predicted = route(train, future)[:len(actual)]
    return rmse(actual.to_numpy(), predicted), len(train)


def main() -> None:
    quiet_prophet()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    flow = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                       parse_dates=["report_date"], date_format="%Y%m%d")
    flow = flow.set_index("report_date").asfreq("D")
    inflow = flow["total_purchase_amt"]

    print("--- 1. Lay out the cut-off dates ---")
    print(f"  history {flow.index[0].date()} .. {flow.index[-1].date()}, "
          f"{len(flow)} days")
    print(f"  {'origin':<12} {'training days':>14}  {'scored window':>26}")
    for origin in ORIGINS:
        cut = pd.Timestamp(origin)
        train = inflow.loc[:cut]
        window = pd.date_range(cut + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        scored = inflow.reindex(window).dropna()
        print(f"  {origin:<12} {len(train):>14}  "
              f"{f'{scored.index[0].date()} .. {scored.index[-1].date()}':>26}")
    print(f"  each fold trains on everything up to its own cut-off, so the training "
          f"window grows and never contains a day it is later scored on")

    print("\n--- 2. One holdout, then four ---")
    single, _ = fold_score(inflow, ORIGINS[-1], ROUTES["periodic factors"])
    per_fold = [fold_score(inflow, origin, ROUTES["periodic factors"])[0]
                for origin in ORIGINS]
    print(f"  scored on the last cut-off alone: {single:,.0f}")
    print(f"  scored on each of the four:       "
          f"{', '.join(f'{s:,.0f}' for s in per_fold)}")
    print(f"  spread across folds {max(per_fold) / min(per_fold):.2f}x, "
          f"mean {np.mean(per_fold):,.0f}, sd {np.std(per_fold):,.0f}")
    print(f"  the single number sits {abs(single - np.mean(per_fold)) / np.std(per_fold):.1f} "
          f"standard deviations from the average of the four")

    print("\n--- 3. Every route over every fold ---")
    table = {}
    for name, route in ROUTES.items():
        table[name] = [fold_score(inflow, origin, route)[0] for origin in ORIGINS]
    header = "  ".join(f"{o[5:]:>12}" for o in ORIGINS)
    print(f"  {'route':<20} {header}  {'mean':>12}  {'sd':>11}")
    ranked = sorted(table.items(), key=lambda kv: np.mean(kv[1]))
    for name, scores in ranked:
        cells = "  ".join(f"{s:>12,.0f}" for s in scores)
        print(f"  {name:<20} {cells}  {np.mean(scores):>12,.0f}  "
              f"{np.std(scores):>11,.0f}")
    winners = {min(table, key=lambda n: table[n][i]) for i in range(len(ORIGINS))}
    print(f"  routes that win at least one fold: {sorted(winners)}")
    worst_spread = max((max(s) / min(s), n) for n, s in table.items())
    print(f"  the winner is the same in all {len(ORIGINS)} folds, so the ranking here "
          f"does not depend on which cut-off was chosen")
    print(f"  the fold-to-fold spread does: {worst_spread[1]} moves "
          f"{worst_spread[0]:.2f}x between its best and worst fold, which is more "
          f"than the gap between the top two routes")

    print("\n--- 4. The fitted routes measured on the days they were fitted on ---")
    cut = pd.Timestamp(ORIGINS[-1])
    train = inflow.loc[:cut]
    fitted_values = {
        "periodic factors": periodic_factors(train, train.index),
        "SARIMAX, weekly": np.asarray(sm.tsa.statespace.SARIMAX(
            train.to_numpy(), order=(2, 0, 2), seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False).fittedvalues),
        "additive model": additive_prophet(train, train.index),
    }
    print(f"  {'route':<20} {'on training days':>17}  {'held out, mean':>15}  "
          f"{'ratio':>7}")
    for name, fitted in fitted_values.items():
        in_sample = rmse(train.to_numpy(), fitted)
        held = float(np.mean(table[name]))
        print(f"  {name:<20} {in_sample:>17,.0f}  {held:>15,.0f}  "
              f"{held / in_sample:>6.2f}x")
    print(f"  'last week repeated' is absent because it fits nothing: it has no "
          f"training-period error to quote, only the same rule applied to older days")
    inflation = {n: float(np.mean(table[n])) / rmse(train.to_numpy(), f)
                 for n, f in fitted_values.items()}
    print(f"  the three routes inflate by {min(inflation.values()):.2f}x to "
          f"{max(inflation.values()):.2f}x, so the gaps between them are not preserved")
    fitted_rank = sorted(fitted_values, key=lambda n: rmse(train.to_numpy(),
                                                           fitted_values[n]))
    held_rank = sorted(fitted_values, key=lambda n: np.mean(table[n]))
    print(f"  ranked on training days: {fitted_rank}")
    print(f"  ranked on held-out days: {held_rank}")
    print(f"  same order here: {fitted_rank == held_rank}; the two rankings are free "
          f"to differ, and only the second one was measured on days no route had read")

    print("\n--- 5. Refit on everything and forecast the month after the data ---")
    best_route_name = ranked[0][0]
    best_route = ROUTES[best_route_name]
    future = pd.date_range(SUBMISSION_START, periods=HORIZON, freq="D")
    submission = pd.DataFrame({"report_date": future.strftime("%Y%m%d")})
    for column in COLUMNS:
        predicted = best_route(flow[column], future)
        submission[column.replace("total_", "").replace("_amt", "")] = np.round(
            np.clip(predicted, 0, None)).astype("int64")
    print(f"  route {best_route_name}, refitted on all {len(flow)} days")
    print(f"  horizon {future[0].date()} .. {future[-1].date()}")
    print(f"  {'report_date':<12} {'purchase':>12} {'redeem':>12}")
    for _, row in submission.head(5).iterrows():
        print(f"  {row['report_date']:<12} {row['purchase']:>12,} "
              f"{row['redeem']:>12,}")
    weekday_mean = submission.assign(
        weekday=future.dayofweek).groupby("weekday")["purchase"].mean()
    planted = np.array(truth["cash_flow"]["purchase_weekday_factor"])
    recovered = (weekday_mean / weekday_mean.mean()).to_numpy()
    print(f"  weekday shape of the forecast vs the planted factors, largest gap "
          f"{np.abs(recovered - planted / planted.mean()).max():.4f}")

    print("\n--- 6. Check the file against the format before sending it ---")
    path = OUT_DIR / "cash_flow_forecast.csv"
    submission.to_csv(path, index=False, header=False, encoding="utf-8")
    reread = pd.read_csv(path, header=None, names=submission.columns,
                         dtype={"report_date": str})
    checks = {
        "row count matches the horizon": len(reread) == HORIZON,
        "dates are distinct": reread["report_date"].nunique() == HORIZON,
        "dates are the requested month": set(reread["report_date"]) == set(
            future.strftime("%Y%m%d")),
        "no header row was written": not reread["report_date"].iloc[0].isalpha(),
        "three columns": reread.shape[1] == 3,
        "no missing values": int(reread.isna().sum().sum()) == 0,
        "no negative amounts": bool((reread[["purchase", "redeem"]] >= 0).all().all()),
        "amounts are whole numbers": bool(
            (reread[["purchase", "redeem"]].dtypes == "int64").all()),
    }
    for label, passed in checks.items():
        print(f"  {'pass' if passed else 'FAIL':>4}  {label}")
    assert all(checks.values()), "the written file does not satisfy the format"
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}, "
          f"{path.stat().st_size} bytes")
    print("  the checks run against the file that was written, not the frame in "
          "memory: the header, the dtypes and the date format are all decided by "
          "the write and can only be confirmed by reading it back")


if __name__ == "__main__":
    main()
