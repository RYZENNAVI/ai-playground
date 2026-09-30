"""This script estimates a weekday effect and a month-position effect in the daily inflow three
ways (additive dummies in an OLS regression, ratios of group means, and multiplicative factors
fitted by alternating updates), scores all three against the planted truth, and forecasts from
them as a periodic factor baseline, checked against ARIMA. The run prints 6 parts:
    1. Average the column by weekday and by day of month, and read the two profiles.
    2. Fit the two effects as additive dummies in one regression.
    3. Divide each group mean by the overall mean, taking the two effects one at a time.
    4. Fit the two multiplicative effects jointly, by alternating between them until they settle.
    5. Forecast the held-out month with every route, next to ARIMA and last week repeated.
    6. Measure the imbalance that makes the one-at-a-time estimate wrong, and price it.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_START = "2014-03-01"
TRAIN_END = "2014-07-31"
TEST_START = "2014-08-01"
TEST_END = "2014-08-31"
ALTERNATING_MAX_ROUNDS = 20
# The alternating fit stops when no factor moves by more than this between two rounds.
ALTERNATING_TOL = 1e-6
# Floor for the denominator each round divides by. It never binds on this data,
# where every factor stays near 1, but the routine divides by a fitted quantity
# and a fitted quantity is allowed to come back at zero.
EPS = 1e-8
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

    A multiplicative split is fixed only up to a constant, and raw factors would measure it.
    """
    return factors / factors.mean()


def ratio_factors(frame: pd.DataFrame, column: str) -> tuple[float, pd.Series, pd.Series]:
    """Take each effect on its own: group mean divided by overall mean.

    It is exact only when every weekday meets every day of the month equally often (part 6).
    """
    level = frame[column].mean()
    weekday = frame.groupby("weekday")[column].mean() / level
    day = frame.groupby("day")[column].mean() / level
    return level, normalise(weekday), normalise(day)


def alternating_factors(frame: pd.DataFrame, column: str,
                        max_rounds: int = ALTERNATING_MAX_ROUNDS,
                        tol: float = ALTERNATING_TOL
                        ) -> tuple[float, pd.Series, pd.Series, int]:
    """Fit both multiplicative effects together, holding one fixed while updating the other.

    Each update divides out what the other already explains; it stops once no factor moves more
    than tol.
    """
    level = frame[column].mean()
    weekday = pd.Series(1.0, index=sorted(frame["weekday"].unique()))
    day = pd.Series(1.0, index=sorted(frame["day"].unique()))

    used = 0
    for used in range(1, max_rounds + 1):
        before = np.r_[weekday.to_numpy(), day.to_numpy()]
        explained_by_day = (level * frame["day"].map(day)).clip(lower=EPS)
        weekday = normalise(
            (frame[column] / explained_by_day).groupby(frame["weekday"]).mean())
        explained_by_weekday = (level * frame["weekday"].map(weekday)).clip(lower=EPS)
        day = normalise(
            (frame[column] / explained_by_weekday).groupby(frame["day"]).mean())
        moved = np.abs(np.r_[weekday.to_numpy(), day.to_numpy()] - before).max()
        if moved < tol:
            break
    return level, weekday, day, used


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

    campaign_days = pd.to_datetime(flow_truth["campaign_days"])
    train = frame.loc[TRAIN_START:TRAIN_END]
    promoted = train.index[train.index.isin(campaign_days)]
    train = train.drop(promoted)
    test = frame.loc[TEST_START:TEST_END]

    planted_weekday = normalise(pd.Series(flow_truth["purchase_weekday_factor"],
                                          index=range(7)))
    planted_day = normalise(pd.Series(flow_truth["day_of_month_factor"],
                                      index=range(1, 32)))

    print("--- 1. Average the column by weekday and by month position ---")
    print(f"  train {train.index[0].date()} .. {train.index[-1].date()} "
          f"({len(train)} rows), holdout {test.index[0].date()} .. "
          f"{test.index[-1].date()} ({len(test)} rows)")
    print(f"  promotion days left out of training: "
          f"{', '.join(str(d.date()) for d in promoted) or 'none'}; "
          f"the one in the holdout is scored separately in part 5")
    observed_weekday = normalise(train.groupby("weekday")[column].mean())
    print(f"  {'weekday':<10} {'observed factor':>16} {'planted factor':>16} "
          f"{'gap':>8}  {'rows':>5}")
    for wd, name in enumerate(WEEKDAY_NAMES):
        gap = observed_weekday[wd] - planted_weekday[wd]
        rows = int((train["weekday"] == wd).sum())
        print(f"  {name:<10} {observed_weekday[wd]:>16.4f} "
              f"{planted_weekday[wd]:>16.4f} {gap:>+8.4f}  {rows:>5}")
    top = int(observed_weekday.idxmax())
    print(f"  weekend rows sit around {observed_weekday[5:].mean():.2f} of an average "
          f"day and {WEEKDAY_NAMES[top]} around {observed_weekday[top]:.2f}: the swing "
          f"across the week is a factor of "
          f"{observed_weekday.max() / observed_weekday.min():.2f}")

    # The second cycle the heading promises. Printing all 31 positions would bury
    # the shape, so the rows shown are the three the generator pushes highest and
    # the three it pushes lowest, chosen from the planted values rather than from
    # the observed ones so that the selection cannot flatter the fit.
    observed_day = normalise(train.groupby("day")[column].mean())
    common = observed_day.index.intersection(planted_day.index)
    extremes = sorted(set(planted_day[common].nlargest(3).index)
                      | set(planted_day[common].nsmallest(3).index))
    print(f"  {'month position':<14} {'observed factor':>16} {'planted factor':>16} "
          f"{'gap':>8}  {'rows':>5}")
    for d in extremes:
        rows = int((train["day"] == d).sum())
        print(f"  {f'day {d}':<14} {observed_day[d]:>16.4f} {planted_day[d]:>16.4f} "
              f"{observed_day[d] - planted_day[d]:>+8.4f}  {rows:>5}")
    day_rows = train.groupby("day").size()
    print(f"  these are the three positions the generator lifts most and the three it cuts "
          f"most; the swing across the month is a factor of "
          f"{observed_day[common].max() / observed_day[common].min():.2f}, against "
          f"{observed_weekday.max() / observed_weekday.min():.2f} across the week, and each "
          f"position is averaged over only {day_rows.min()} to {day_rows.max()} rows")

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
    print("  these weekday factors are the part 1 profiles, now read as an estimate")
    print(f"  largest weekday gap to planted "
          f"{np.abs(weekday_r - planted_weekday).max():.4f}")
    common_days = day_r.index.intersection(planted_day.index)
    day_gap_r = np.abs(day_r[common_days] - planted_day[common_days])
    print(f"  month-position gap to planted: largest {day_gap_r.max():.4f}, "
          f"mean {day_gap_r.mean():.4f}")

    print("\n--- 4. Fit both effects together, alternating between them ---")
    level_a, weekday_a, day_a, rounds_used = alternating_factors(train, column)
    status = (f"converged after {rounds_used} rounds" if rounds_used < ALTERNATING_MAX_ROUNDS
              else f"stopped at the cap of {ALTERNATING_MAX_ROUNDS} rounds")
    print(f"  {status} (tolerance {ALTERNATING_TOL:g}), level {level_a:,.0f}")
    print(f"  weekday factors {np.round(weekday_a.to_numpy(), 3).tolist()}")
    print(f"  largest weekday gap to planted "
          f"{np.abs(weekday_a - planted_weekday).max():.4f}")
    day_gap_a = np.abs(day_a[common_days] - planted_day[common_days])
    print(f"  month-position gap to planted: largest {day_gap_a.max():.4f}, "
          f"mean {day_gap_a.mean():.4f} ({1 - day_gap_a.mean() / day_gap_r.mean():.0%} "
          f"smaller than one at a time)")
    settle = alternating_factors(train, column, max_rounds=1)[1]
    if np.allclose(settle, weekday_r):
        print(f"  after one round the weekday factors equal the part 3 ratios, because the "
              f"month factors still start at 1; the later rounds move them by up to "
              f"{np.abs(settle - weekday_a).max():.5f}")

    print("\n--- 5. Score every route on the month nobody fitted ---")
    truth_pred = predict_from_factors(
        test.index, train[column].mean(), planted_weekday, planted_day)
    routes = {
        "additive dummies": ols_test.to_numpy(),
        "ratio, one at a time": predict_from_factors(
            test.index, level_r, weekday_r, day_r),
        "alternating, joint": predict_from_factors(
            test.index, level_a, weekday_a, day_a),
        "planted-factor reference": truth_pred,
    }
    # ARIMA needs consecutive days, so it keeps the promotion day the factor routes drop.
    arima_fit = ARIMA(frame.loc[TRAIN_START:TRAIN_END, column], order=(2, 0, 2),
                      seasonal_order=(1, 0, 1, 7)).fit()
    routes["ARIMA with a weekly term"] = arima_fit.forecast(steps=len(test)).to_numpy()
    routes["last week repeated"] = np.resize(
        train[column].to_numpy()[-7:], len(test))

    actual = test[column].to_numpy()
    campaign_in_test = [d for d in flow_truth["campaign_days"]
                        if TEST_START <= d <= TEST_END]
    ordinary = ~test.index.isin(pd.to_datetime(campaign_in_test))
    baseline = rmse(actual[ordinary], truth_pred[ordinary])
    print(f"  {'route':<26} {'all 31 days':>14}  {'ordinary days':>14}  "
          f"{'vs reference':>13}")
    # Not a lower bound on anything. It is handed the generator's own two factor
    # sets, but its level is still the training mean, the holdout still carries
    # noise, and it is as blind to the promotion day as every other route. It is
    # the score a route gets for knowing the cycles and nothing else.
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
              f"error of the planted-factor reference, which is why the two columns differ")
    ordinary_scores = {label: rmse(actual[ordinary], predicted[ordinary])
                       for label, predicted in routes.items()}
    best = min(ordinary_scores, key=ordinary_scores.get)
    fitted_routes = ["additive dummies", "ratio, one at a time", "alternating, joint"]
    print(f"  on ordinary days the {best} is the best of the {len(routes)} routes, and the "
          f"three fitted routes land within "
          f"{max(ordinary_scores[k] for k in fitted_routes) / baseline:.2f}x of the reference")
    print(f"  ARIMA has the weekly period but no month-position term, and reaches "
          f"{ordinary_scores['ARIMA with a weekly term'] / baseline:.2f}x on ordinary days; "
          f"the factor routes use {len(weekday_a)} weekday factors, {len(day_a)} "
          f"month-position factors and one level, {len(weekday_a) + len(day_a) + 1} numbers")

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
          f"month factor met spans only {imbalance:.4f}")
    clean = train.assign(**{column: train[column].mean()
                            * train["weekday"].map(planted_weekday)
                            * train["day"].map(planted_day)})
    clean_ratio = float(np.abs(ratio_factors(clean, column)[1] - planted_weekday).mean())
    clean_joint = float(np.abs(alternating_factors(clean, column)[1] - planted_weekday).mean())
    print(f"  on the same dates with the planted factors and no noise, the mean weekday "
          f"error is {clean_ratio:.4f} one at a time and {clean_joint:.4f} joint: that "
          f"is the price of the imbalance alone")


if __name__ == "__main__":
    main()
