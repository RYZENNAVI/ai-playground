"""Estimate a weekday effect and a month-position effect three ways, and score all three against the truth.

Demonstrates a forecast built from group averages, and what it costs against a fitted model:
    1. Average the column by weekday and by day of month, and read the two profiles.
    2. Fit the two effects as additive dummies in one regression.
    3. Divide each group mean by the overall mean, taking the two effects one at a time.
    4. Fit the two multiplicative effects jointly, by alternating between them until they settle.
    5. Score every recovered factor against the planted one, and forecast a month nobody has seen.
    6. Measure the imbalance that makes the one-at-a-time estimate wrong, and price it.

Module 08: Time Series Forecasting - Periodic Factor Baselines.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_START = "2014-03-01"
TRAIN_END = "2014-07-31"
TEST_START = "2014-08-01"
TEST_END = "2014-08-31"
ALTERNATING_ROUNDS = 20
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_daily() -> tuple[pd.DataFrame, dict]:
    """Read the cash-flow table with the weekday and month position already attached."""
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                        parse_dates=["report_date"], date_format="%Y%m%d")
    frame["weekday"] = frame["report_date"].dt.dayofweek
    frame["day"] = frame["report_date"].dt.day
    return frame.set_index("report_date"), truth


def normalise(factors: pd.Series) -> pd.Series:
    """Rescale a set of factors to average one, so two sets can be compared entry by entry.

    A multiplicative decomposition is only defined up to a constant: doubling
    every weekday factor and halving the level leaves every fitted value alone.
    Comparing raw factors against the planted ones would therefore measure that
    arbitrary constant rather than the shape, which is the part that was learned.
    """
    return factors / factors.mean()


def ratio_factors(frame: pd.DataFrame, column: str) -> tuple[float, pd.Series, pd.Series]:
    """Take each effect on its own: group mean divided by overall mean.

    This is the estimate that reads as obviously right. It is exact only when
    every weekday meets every day of the month equally often, and over a window
    of a few months it does not: the imbalance leaks part of one effect into the
    other, which the last step of this script measures.
    """
    level = frame[column].mean()
    weekday = frame.groupby("weekday")[column].mean() / level
    day = frame.groupby("day")[column].mean() / level
    return level, normalise(weekday), normalise(day)


def alternating_factors(frame: pd.DataFrame, column: str,
                        rounds: int = ALTERNATING_ROUNDS
                        ) -> tuple[float, pd.Series, pd.Series]:
    """Fit both multiplicative effects together, by holding one fixed while updating the other.

    Each round divides the observed value by what the other effect already
    explains before averaging, so a weekday that happens to fall on month ends
    more often is no longer credited with the month-end lift. Renormalising after
    every update keeps the level from drifting into the factors, which is what
    makes the two rounds comparable.
    """
    level = frame[column].mean()
    weekday = pd.Series(1.0, index=sorted(frame["weekday"].unique()))
    day = pd.Series(1.0, index=sorted(frame["day"].unique()))

    for _ in range(rounds):
        explained_by_day = level * frame["day"].map(day)
        weekday = normalise(
            (frame[column] / explained_by_day).groupby(frame["weekday"]).mean())
        explained_by_weekday = level * frame["weekday"].map(weekday)
        day = normalise(
            (frame[column] / explained_by_weekday).groupby(frame["day"]).mean())
    return level, weekday, day


def predict_from_factors(dates: pd.DatetimeIndex, level: float,
                         weekday: pd.Series, day: pd.Series) -> np.ndarray:
    """Multiply the level by both factors for every date being forecast."""
    wk = pd.Series(dates.dayofweek, index=dates).map(weekday).to_numpy(dtype=float)
    dm = pd.Series(dates.day, index=dates).map(day).to_numpy(dtype=float)
    return level * wk * dm


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, in the units of the series."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def main() -> None:
    frame, truth = load_daily()
    flow_truth = truth["cash_flow"]
    column = "total_purchase_amt"

    train = frame.loc[TRAIN_START:TRAIN_END]
    test = frame.loc[TEST_START:TEST_END]

    planted_weekday = normalise(pd.Series(flow_truth["purchase_weekday_factor"],
                                          index=range(7)))
    planted_day = normalise(pd.Series(flow_truth["day_of_month_factor"],
                                      index=range(1, 32)))

    print("--- 1. Average the column by weekday and by month position ---")
    print(f"  train {train.index[0].date()} .. {train.index[-1].date()} "
          f"({len(train)} rows), holdout {test.index[0].date()} .. "
          f"{test.index[-1].date()} ({len(test)} rows)")
    observed_weekday = normalise(train.groupby("weekday")[column].mean())
    print(f"  {'weekday':<10} {'observed factor':>16} {'planted factor':>16} "
          f"{'gap':>8}  {'rows':>5}")
    for wd, name in enumerate(WEEKDAY_NAMES):
        gap = observed_weekday[wd] - planted_weekday[wd]
        rows = int((train["weekday"] == wd).sum())
        print(f"  {name:<10} {observed_weekday[wd]:>16.4f} "
              f"{planted_weekday[wd]:>16.4f} {gap:>+8.4f}  {rows:>5}")
    print(f"  weekend rows sit around {observed_weekday[5:].mean():.2f} of an average "
          f"day and Tuesday around {observed_weekday[1]:.2f}: the swing across the "
          f"week is a factor of {observed_weekday.max() / observed_weekday.min():.2f}")

    print("\n--- 2. Fit both effects as additive dummies ---")
    ols = smf.ols(f"{column} ~ C(weekday) + C(day)", data=train.reset_index()).fit()
    print(f"  {len(ols.params)} coefficients from {len(train)} rows, "
          f"R-squared {ols.rsquared:.4f}, adjusted {ols.rsquared_adj:.4f}")
    print(f"  every effect is estimated against the reference cell, so a weekday "
          f"coefficient reads as a difference in amount, not a ratio")
    ols_test = ols.predict(test.reset_index())
    print(f"  holdout RMSE {rmse(test[column].to_numpy(), ols_test.to_numpy()):,.0f}")

    print("\n--- 3. Take each effect on its own, as a ratio ---")
    level_r, weekday_r, day_r = ratio_factors(train, column)
    print(f"  level {level_r:,.0f}")
    print(f"  weekday factors {np.round(weekday_r.to_numpy(), 3).tolist()}")
    print(f"  largest weekday gap to planted "
          f"{np.abs(weekday_r - planted_weekday).max():.4f}")
    common_days = day_r.index.intersection(planted_day.index)
    print(f"  largest month-position gap to planted "
          f"{np.abs(day_r[common_days] - planted_day[common_days]).max():.4f}")

    print("\n--- 4. Fit both effects together, alternating between them ---")
    level_a, weekday_a, day_a = alternating_factors(train, column)
    print(f"  {ALTERNATING_ROUNDS} rounds, level {level_a:,.0f}")
    print(f"  weekday factors {np.round(weekday_a.to_numpy(), 3).tolist()}")
    print(f"  largest weekday gap to planted "
          f"{np.abs(weekday_a - planted_weekday).max():.4f}")
    print(f"  largest month-position gap to planted "
          f"{np.abs(day_a[common_days] - planted_day[common_days]).max():.4f}")
    settle = alternating_factors(train, column, rounds=1)[1]
    print(f"  after one round the weekday factors are already within "
          f"{np.abs(settle - weekday_a).max():.5f} of where they end up")

    print("\n--- 5. Score every route on the month nobody fitted ---")
    truth_pred = predict_from_factors(
        test.index, train[column].mean(), planted_weekday, planted_day)
    routes = {
        "additive dummies": ols_test.to_numpy(),
        "ratio, one at a time": predict_from_factors(
            test.index, level_r, weekday_r, day_r),
        "alternating, joint": predict_from_factors(
            test.index, level_a, weekday_a, day_a),
        "planted factors": truth_pred,
    }
    arima_fit = ARIMA(train[column], order=(2, 0, 2),
                      seasonal_order=(1, 0, 1, 7)).fit()
    routes["ARIMA with a weekly term"] = arima_fit.forecast(steps=len(test)).to_numpy()
    routes["last week repeated"] = np.resize(
        train[column].to_numpy()[-7:], len(test))

    actual = test[column].to_numpy()
    campaign_in_test = [d for d in flow_truth["campaign_days"]
                        if TEST_START <= d <= TEST_END]
    ordinary = ~test.index.isin(pd.to_datetime(campaign_in_test))
    baseline = rmse(actual[ordinary], truth_pred[ordinary])
    print(f"  {'route':<26} {'all 31 days':>14}  {'ordinary days':>14}  {'vs planted':>11}")
    for label, predicted in routes.items():
        print(f"  {label:<26} {rmse(actual, predicted):>14,.0f}  "
              f"{rmse(actual[ordinary], predicted[ordinary]):>14,.0f}  "
              f"{rmse(actual[ordinary], predicted[ordinary]) / baseline:>10.2f}x")
    if campaign_in_test:
        stamp = pd.Timestamp(campaign_in_test[0])
        row = test.index.get_loc(stamp)
        err = actual[row] - truth_pred[row]
        print(f"  {stamp.date()} is a promotion day none of these routes knows about: "
              f"it alone carries "
              f"{err ** 2 / np.sum((actual - truth_pred) ** 2):.0%} of the squared "
              f"error of the best route, which is why the two columns differ")
    print(f"  on ordinary days the planted factors set the floor, and the three "
          f"fitted routes land within "
          f"{max(rmse(actual[ordinary], routes[k][ordinary]) / baseline for k in ['additive dummies', 'ratio, one at a time', 'alternating, joint']):.2f}x "
          f"of it")
    print(f"  the autoregressive model was given the same weekly period and reaches "
          f"{rmse(actual[ordinary], routes['ARIMA with a weekly term'][ordinary]) / baseline:.2f}x: "
          f"on this series the two approaches are close, and the factor model gets "
          f"there with {len(weekday_a) + len(day_a)} numbers and no optimiser")

    print("\n--- 6. Measure the imbalance that biases the one-at-a-time estimate ---")
    counts = pd.crosstab(train["weekday"], train["day"])
    expected = len(train) / (7 * counts.shape[1])
    print(f"  {len(train)} rows spread over {counts.shape[0]} weekdays x "
          f"{counts.shape[1]} month positions, {expected:.2f} rows per cell if balanced")
    print(f"  actual cell counts range {counts.to_numpy().min()} .. "
          f"{counts.to_numpy().max()}")
    mean_dom_by_weekday = train.groupby("weekday")["day"].apply(
        lambda days: planted_day[days].mean())
    print(f"  {'weekday':<10} {'mean month factor met':>22}  {'ratio est.':>11}  "
          f"{'joint est.':>11}  {'planted':>9}")
    for wd, name in enumerate(WEEKDAY_NAMES):
        print(f"  {name:<10} {mean_dom_by_weekday[wd]:>22.4f}  "
              f"{weekday_r[wd]:>11.4f}  {weekday_a[wd]:>11.4f}  "
              f"{planted_weekday[wd]:>9.4f}")
    ratio_err = float(np.abs(weekday_r - planted_weekday).mean())
    joint_err = float(np.abs(weekday_a - planted_weekday).mean())
    imbalance = float(mean_dom_by_weekday.max() - mean_dom_by_weekday.min())
    print(f"  mean weekday error: one at a time {ratio_err:.4f}, "
          f"joint {joint_err:.4f} "
          f"({1 - joint_err / ratio_err:.0%} smaller)")
    print(f"  a weekday that lands on high month positions more often than average "
          f"absorbs part of the month effect, and the one-at-a-time estimate has no "
          f"way to give it back")
    print(f"  here the imbalance is small: across the seven weekdays the average "
          f"month factor met spans only {imbalance:.4f}, because five months of "
          f"dates almost balance on their own")
    print(f"  the correction is worth having and it is not worth much on this window; "
          f"on a shorter one, or on a calendar with holidays removed, the same "
          f"argument gets larger")


if __name__ == "__main__":
    main()
