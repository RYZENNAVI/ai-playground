"""This script runs a rolling-origin backtest of four forecasting routes (last week repeated,
periodic factors, weekly SARIMAX and Prophet) on the cash-flow table, scoring each on repeated
forecasts from moving cut-off dates, then refits the best route for each column and writes the
forecast file. A score is not an estimate of future error, and the run shows the difference:
    1. Lay out the cut-off dates, and check that no route can see past the one it is given.
    2. Score one route on a single holdout, and then on every fold, to see how far one number
       moves. This part is for comparison only, and part 3 scores every fold again.
    3. Run all routes over all folds and rank them by their averages.
    4. Put the same routes' training-period errors next to those averages. This part is for
       comparison only and nothing later uses it.
    5. Backtest the redeem column too, pick the best route for each column, refit it on the
       whole history and forecast the month that follows.
    6. Check the written file against the format it has to satisfy before it is sent anywhere.
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

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
HORIZON = 30
ORIGINS = ["2014-04-30", "2014-05-31", "2014-06-30", "2014-07-31"]
SUBMISSION_START = "2014-09-01"
COLUMNS = ["total_purchase_amt", "total_redeem_amt"]

# Same alternating fit and stopping rule as 05.
# EPS floors a denominator that is itself a fitted value.
PERIODIC_MAX_ROUNDS = 20
PERIODIC_TOL = 1e-6
EPS = 1e-8


def quiet_prophet() -> None:
    """Silence Prophet and its fitting backend, which otherwise log a line for every fit."""
    for noisy in ("prophet", "cmdstanpy"):
        log = logging.getLogger(noisy)
        log.setLevel(logging.CRITICAL)
        log.addFilter(lambda record: False)
        log.propagate = False


def seasonal_naive(train: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Repeat the last seven observed values for as long as the horizon runs."""
    return np.resize(train.to_numpy()[-7:], len(dates))


def periodic_factors(train: pd.Series, dates: pd.DatetimeIndex,
                     max_rounds: int = PERIODIC_MAX_ROUNDS,
                     tol: float = PERIODIC_TOL) -> np.ndarray:
    """Forecast as level times a weekday factor times a month-position factor, fitted as in 05.

    Fitting the two together keeps a weekday that often falls on month ends from taking the lift.
    """
    frame = pd.DataFrame({"y": train.to_numpy(),
                          "weekday": train.index.dayofweek,
                          "day": train.index.day})
    level = frame["y"].mean()
    weekday = pd.Series(1.0, index=range(7))
    day = pd.Series(1.0, index=range(1, 32))
    for _ in range(max_rounds):
        before = np.r_[weekday.to_numpy(), day.to_numpy()]
        explained_by_day = (level * frame["day"].map(day)).clip(lower=EPS)
        weekday = (frame["y"] / explained_by_day).groupby(frame["weekday"]).mean()
        weekday = (weekday / weekday.mean()).reindex(range(7)).fillna(1.0)
        explained_by_weekday = (level * frame["weekday"].map(weekday)).clip(lower=EPS)
        day = (frame["y"] / explained_by_weekday).groupby(frame["day"]).mean()
        day = (day / day.mean()).reindex(range(1, 32)).fillna(1.0)
        if np.abs(np.r_[weekday.to_numpy(), day.to_numpy()] - before).max() < tol:
            break
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


def fold_score(series: pd.Series, origin: str, route,
               left_out: pd.DatetimeIndex) -> tuple[float, float]:
    """Fit a route up to one cut-off and score the next HORIZON days, with and without left_out.

    Predictions are lined up with the actuals by date, so a missing day cannot shift the rest.
    """
    cut = pd.Timestamp(origin)
    train = series.loc[:cut]
    future = pd.date_range(cut + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    actual = series.reindex(future).dropna()
    predicted = pd.Series(route(train, future), index=future).reindex(actual.index)
    kept = ~actual.index.isin(left_out)
    return (rmse(actual.to_numpy(), predicted.to_numpy()),
            rmse(actual.to_numpy()[kept], predicted.to_numpy()[kept]))


def main() -> None:
    quiet_prophet()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    flow = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                       parse_dates=["report_date"], date_format="%Y%m%d")
    flow = flow.set_index("report_date").asfreq("D")
    inflow = flow["total_purchase_amt"]
    promo_days = pd.to_datetime(truth["cash_flow"]["campaign_days"])

    print("--- 1. Lay out the cut-off dates ---")
    print(f"  history {flow.index[0].date()} .. {flow.index[-1].date()}, {len(flow)} days")
    print(f"  {'origin':<12} {'training days':>14}  {'scored window':>26}")
    scored_days = pd.DatetimeIndex([])
    for origin in ORIGINS:
        cut = pd.Timestamp(origin)
        train = inflow.loc[:cut]
        window = pd.date_range(cut + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        scored = inflow.reindex(window).dropna()
        assert train.index[-1] < scored.index[0], f"fold {origin} trains on a scored day"
        scored_days = scored_days.union(scored.index)
        print(f"  {origin:<12} {len(train):>14}  "
              f"{f'{scored.index[0].date()} .. {scored.index[-1].date()}':>26}")
    print("  each fold trains on everything up to its own cut-off, so the training window "
          "grows and never contains a day it is later scored on")

    print("\n--- 2. One holdout, then four ---")
    single, _ = fold_score(inflow, ORIGINS[-1], ROUTES["periodic factors"], promo_days)
    per_fold, per_fold_ordinary = zip(*(
        fold_score(inflow, origin, ROUTES["periodic factors"], promo_days)
        for origin in ORIGINS))
    print(f"  scored on the last cut-off alone: {single:,.0f}")
    print(f"  scored on each of the four:       {', '.join(f'{s:,.0f}' for s in per_fold)}")
    spread = max(per_fold) / min(per_fold)
    print(f"  spread across folds {spread:.2f}x, "
          f"mean {np.mean(per_fold):,.0f}, sd {np.std(per_fold):,.0f}")
    print(f"  the single number sits {abs(single - np.mean(per_fold)) / np.std(per_fold):.1f} "
          f"standard deviations from the average of the four")
    promo_scored = [d for d in promo_days if d in scored_days]
    ordinary_spread = max(per_fold_ordinary) / min(per_fold_ordinary)
    print(f"  without the promotion days {', '.join(str(d.date()) for d in promo_scored)} "
          f"the four folds are {', '.join(f'{s:,.0f}' for s in per_fold_ordinary)} "
          f"({ordinary_spread:.2f}x)")
    if ordinary_spread - 1 < (spread - 1) / 2:
        print(f"  one day in each of {len(promo_scored)} folds makes most of the spread")

    print("\n--- 3. Every route over every fold ---")
    table, ordinary_table = {}, {}
    for name, route in ROUTES.items():
        scores = [fold_score(inflow, origin, route, promo_days) for origin in ORIGINS]
        table[name] = [s[0] for s in scores]
        ordinary_table[name] = [s[1] for s in scores]
    header = "  ".join(f"{o[5:]:>12}" for o in ORIGINS)
    print(f"  {'route':<20} {header}  {'mean':>12}  {'sd':>11}  {'ordinary mean':>13}")
    ranked = sorted(table.items(), key=lambda kv: np.mean(kv[1]))
    for name, scores in ranked:
        cells = "  ".join(f"{s:>12,.0f}" for s in scores)
        print(f"  {name:<20} {cells}  {np.mean(scores):>12,.0f}  "
              f"{np.std(scores):>11,.0f}  {np.mean(ordinary_table[name]):>13,.0f}")
    print("  'ordinary mean' leaves out the promotion days, which no route is told about")
    winners = {min(table, key=lambda n: table[n][i]) for i in range(len(ORIGINS))}
    fold_rankings = {tuple(sorted(table, key=lambda n: table[n][i]))
                     for i in range(len(ORIGINS))}
    print(f"  routes that win at least one fold: {sorted(winners)}")
    if len(fold_rankings) == 1:
        print(f"  the whole ranking is the same in all {len(ORIGINS)} folds, so it does not "
              f"depend on which cut-off was chosen")
    elif len(winners) == 1:
        print(f"  {next(iter(winners))} wins all {len(ORIGINS)} folds, but the order below "
              f"it changes from fold to fold")
    else:
        print("  different routes win different folds, so which one looks best depends "
              "on which cut-off was chosen")
    first, second = ranked[0][0], ranked[1][0]
    first_spread = max(table[first]) / min(table[first])
    top_two_gap = np.mean(table[second]) / np.mean(table[first])
    paired = [a / b for a, b in zip(table[first], table[second])]
    comparison = "more than" if first_spread > top_two_gap else "no more than"
    in_every = "but in every fold" if max(paired) < 1 else "and fold by fold"
    print(f"  {first} moves {first_spread:.2f}x between its best and worst fold, "
          f"{comparison} the {top_two_gap:.2f}x gap between the top two means, {in_every} "
          f"it scores {min(paired):.2f}x to {max(paired):.2f}x of the {second}")

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
    # SARIMAX's first fitted values come from its start-up state, so every route skips two weeks.
    warm_up = 14
    in_sample = {n: rmse(train.to_numpy()[warm_up:], f[warm_up:])
                 for n, f in fitted_values.items()}
    print(f"  {'route':<20} {'on training days':>17}  {'held out, mean':>15}  "
          f"{'ratio':>7}")
    for name in fitted_values:
        held = float(np.mean(table[name]))
        print(f"  {name:<20} {in_sample[name]:>17,.0f}  {held:>15,.0f}  "
              f"{held / in_sample[name]:>6.2f}x")
    print(f"  training days are scored from day {warm_up + 1}, because the first fitted "
          f"value of SARIMAX is {fitted_values['SARIMAX, weekly'][0]:,.0f}")
    print(f"  SARIMAX's training error is one step ahead, so it has read every real day "
          f"before the one it predicts; held out, it forecasts {HORIZON} days ahead")
    print("  'last week repeated' is absent because it fits nothing: it has no "
          "training-period error to quote, only the same rule applied to older days")
    inflation = {n: float(np.mean(table[n])) / in_sample[n] for n in fitted_values}
    print(f"  the three routes inflate by {min(inflation.values()):.2f}x to "
          f"{max(inflation.values()):.2f}x, so the gaps between them are not preserved")
    fitted_rank = sorted(fitted_values, key=in_sample.get)
    held_rank = sorted(fitted_values, key=lambda n: np.mean(table[n]))
    print(f"  ranked on training days: {fitted_rank}")
    print(f"  ranked on held-out days: {held_rank}")
    print(f"  same order here: {fitted_rank == held_rank}; the two rankings are free "
          f"to differ, and only the second one was measured on days no route had read")

    print("\n--- 5. Choose a route per target, refit, and forecast ---")
    # Every fold above scored the purchase column. The generator gives redeem a growth
    # term and a wandering level that purchase lacks, so each target gets its own backtest.
    future = pd.date_range(SUBMISSION_START, periods=HORIZON, freq="D")
    submission = pd.DataFrame({"report_date": future.strftime("%Y%m%d")})
    best_by_column = {}
    for column in COLUMNS:
        column_table = table if column == "total_purchase_amt" else {
            name: [fold_score(flow[column], origin, route, promo_days)[0]
                   for origin in ORIGINS]
            for name, route in ROUTES.items()}
        column_ranked = sorted(column_table.items(), key=lambda kv: np.mean(kv[1]))
        best_name = column_ranked[0][0]
        best_by_column[column] = best_name
        print(f"  {column}:")
        for name, scores in column_ranked:
            mark = "  <- selected" if name == best_name else ""
            print(f"    {name:<20} mean RMSE {np.mean(scores):>14,.0f}{mark}")
        predicted = ROUTES[best_name](flow[column], future)
        submission[column.replace("total_", "").replace("_amt", "")] = np.round(
            np.clip(predicted, 0, None)).astype("int64")
    if len(set(best_by_column.values())) == 1:
        print("  both targets select the same route")
    else:
        print(f"  the two targets select different routes: "
              f"{', '.join(f'{c} -> {r}' for c, r in best_by_column.items())}")
        carried = best_by_column["total_purchase_amt"]
        redeem_scores = {n: np.mean(s) for n, s in column_table.items()}
        print(f"  carrying the purchase winner across would have used {carried!r} on "
              f"redeem, where it scores "
              f"{redeem_scores[carried] / min(redeem_scores.values()):.2f}x the best route "
              f"for that column")
    print(f"  each selected route refitted on all {len(flow)} days")
    print(f"  horizon {future[0].date()} .. {future[-1].date()}")
    print(f"  {'report_date':<12} {'purchase':>12} {'redeem':>12}")
    for _, row in submission.head(5).iterrows():
        print(f"  {row['report_date']:<12} {row['purchase']:>12,} "
              f"{row['redeem']:>12,}")
    # Each weekday falls on different days of the month, so the month factors are planted too.
    weekday_mean = submission.assign(
        weekday=future.dayofweek).groupby("weekday")["purchase"].mean()
    planted_weekday = np.array(truth["cash_flow"]["purchase_weekday_factor"])
    planted_day = np.array(truth["cash_flow"]["day_of_month_factor"])
    planted_mean = pd.Series(planted_weekday[future.dayofweek]
                             * planted_day[future.day - 1]).groupby(future.dayofweek).mean()
    recovered = (weekday_mean / weekday_mean.mean()).to_numpy()
    planted = (planted_mean / planted_mean.mean()).to_numpy()
    weekday_only = np.abs(planted - planted_weekday / planted_weekday.mean()).max()
    print(f"  weekday shape of the forecast vs the planted pattern on the same "
          f"{HORIZON} days, largest gap {np.abs(recovered - planted).max():.4f} "
          f"(the weekday factors alone differ from that pattern by {weekday_only:.4f})")

    print("\n--- 6. Check the file against the format before sending it ---")
    path = OUT_DIR / "cash_flow_forecast.csv"
    submission.to_csv(path, index=False, header=False, encoding="utf-8")
    reread = pd.read_csv(path, header=None, names=submission.columns,
                         dtype={"report_date": str})
    expected_dates = future.strftime("%Y%m%d").tolist()
    lines = path.read_text(encoding="utf-8").splitlines()
    checks = {
        "row count matches the horizon": len(reread) == HORIZON,
        "dates are distinct": reread["report_date"].nunique() == HORIZON,
        # Comparing sets would accept any permutation of the right dates, and a
        # submission is read positionally by whatever consumes it.
        "dates are in the requested order": reread["report_date"].tolist()
                                            == expected_dates,
        "every date is eight digits": bool(
            reread["report_date"].str.fullmatch(r"\d{8}").all()),
        # This check and the next read the raw lines: the parser cannot tell a header
        # from data, and it turns a fourth field into an index without complaint.
        "no header row was written": lines[0].split(",")[0] == expected_dates[0],
        "every line has three fields": all(line.count(",") == 2 for line in lines),
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
