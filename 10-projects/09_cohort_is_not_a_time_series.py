"""This script groups customers by the month they joined into a cohort series and sets it
beside one instrument's daily closes. It fits a fixed-order ARIMA model to both and runs a
shuffle test, a form of permutation test: refit on the same values in random orders and see
which fit gets worse. It then fits Prophet to the daily series and checks its weekly and
yearly terms against the rows behind them. A date on the x axis does not make a series
temporal.

    1. Group customers by join month and read the result as a series.
    2. Count how many customers two neighbouring months have in common.
    3. Load one instrument's daily closes, where every point is the same instrument.
    4. Fit ARIMA to both series as given and after 30 shuffles, and compare the errors.
    5. Fit Prophet to the daily series and count the training rows behind each weekday.
    6. Fit the yearly term on one year and on two. 01 plants no yearly pattern.
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
DB_PATH = DATA / "market.sqlite"

TICKER = "MRD"
ARIMA_ORDER = (1, 1, 1)
SHUFFLE_TRIALS = 30
SEED = 20260828


def cohort_series() -> pd.DataFrame:
    """Average assets by the month each customer opened their account."""
    path = DATA / "customers.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path.name}. Run 01_build_project_datasets.py first.")
    customers = pd.read_csv(path, parse_dates=["account_open_date"])
    customers["cohort"] = customers["account_open_date"].dt.to_period("M")
    grouped = customers.groupby("cohort")
    frame = pd.DataFrame({
        "period": grouped.size().index.astype(str),
        "customers": grouped.size().to_numpy(),
        "value": grouped["total_aum"].mean().to_numpy(),
    })
    frame["members"] = [set(group["customer_id"]) for _, group in grouped]
    return frame


def price_series() -> pd.DataFrame:
    """Read one instrument's daily closes, which is a series of one thing over time."""
    if not DB_PATH.exists():
        raise SystemExit(f"Missing {DB_PATH.name}. Run 01_build_project_datasets.py first.")
    with sqlite3.connect(DB_PATH) as connection:
        frame = pd.read_sql_query(
            "SELECT trade_date, close FROM daily_price WHERE ticker = ? ORDER BY trade_date",
            connection, params=(TICKER,),
        )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def overlap_between_neighbours(members: list) -> list:
    """Return the share of each point's customers who also appear in the next point."""
    shares = []
    for current, following in zip(members, members[1:]):
        if not current:
            shares.append(0.0)
            continue
        shares.append(len(current & following) / len(current))
    return shares


def fit_arima(values: np.ndarray) -> float:
    """Fit ARIMA at a fixed order and return its in-sample mean absolute error.

    The order is fixed so a shuffled fit differs from the real one only in the data.
    """
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(values, order=ARIMA_ORDER).fit()
    residuals = np.asarray(model.resid, dtype=float)
    return float(np.mean(np.abs(residuals[ARIMA_ORDER[1]:])))


def shuffle_test(values: np.ndarray, label: str) -> dict:
    """Refit the same model to the values in random orders and compare the errors.

    If the order carries information, shuffling it away makes the fit worse.
    """
    rng = np.random.default_rng(SEED)
    real = fit_arima(values)
    shuffled = []
    for _ in range(SHUFFLE_TRIALS):
        permuted = values.copy()
        rng.shuffle(permuted)
        try:
            shuffled.append(fit_arima(permuted))
        except Exception:
            continue
    shuffled = np.array(shuffled)
    beaten = int((shuffled <= real).sum())
    return {
        "label": label,
        "real": real,
        "shuffled_mean": float(shuffled.mean()),
        "ratio": float(shuffled.mean() / real) if real else float("nan"),
        "beaten_by": beaten,
        "trials": len(shuffled),
    }


def fit_prophet(frame: pd.DataFrame, yearly: bool, weekly: bool):
    """Fit an additive decomposition to a dated series and return the model and its fit.

    The fit runs 14 days past the data, so it has weekend dates the training rows lack.
    """
    from prophet import Prophet

    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly,
                    daily_seasonality=False)
    model.fit(frame.rename(columns={"trade_date": "ds", "close": "y"})[["ds", "y"]])
    future = model.make_future_dataframe(periods=14)
    return model, model.predict(future)


def main() -> None:
    # 1. Assets by the month customers joined
    print("--- 1. Assets by the month customers joined ---")
    cohorts = cohort_series()
    print(f"    points {len(cohorts)}, running {cohorts['period'].iloc[0]} to "
          f"{cohorts['period'].iloc[-1]}")
    print(f"    {'period':<10}{'customers':>11}{'mean assets':>15}")
    for row in cohorts.head(4).itertuples():
        print(f"    {row.period:<10}{row.customers:>11,}{row.value:>15,.0f}")
    print(f"    ... {len(cohorts) - 4} more months")

    # 2. What two neighbouring months have in common
    print("\n--- 2. What two neighbouring months have in common ---")
    shares = overlap_between_neighbours(cohorts["members"].tolist())
    sharing = sum(share > 0 for share in shares)
    print(f"    neighbouring months sharing any customer: {sharing} of {len(shares)}")
    if sharing == 0:
        print("    Each customer has one open date, so this is 0 by construction.")
    print("    Each point is a different set of people. The line between two points")
    print("    does not trace anything moving; it connects two separate populations.")

    # 3. A daily series, for contrast
    print("\n--- 3. A daily series, for contrast ---")
    prices = price_series()
    print(f"    points {len(prices)}, running {prices['trade_date'].min().date()} to "
          f"{prices['trade_date'].max().date()}")
    print(f"    Every point is the same instrument, {TICKER}, because the query filters on it,")
    print("    so neighbouring points always share their subject. That is what makes the")
    print("    change between two points a real quantity.")

    # 4. The shuffle test
    print(f"\n--- 4. The shuffle test, ARIMA{ARIMA_ORDER}, {SHUFFLE_TRIALS} shuffles each ---")
    cohort_result = shuffle_test(cohorts["value"].to_numpy(dtype=float), "cohort by join month")
    price_result = shuffle_test(prices["close"].to_numpy(dtype=float), f"{TICKER} daily close")
    print(f"    {'series':<24}{'error as given':>16}{'shuffled mean':>16}"
          f"{'ratio':>9}{'shuffles as good':>18}")
    for result in (cohort_result, price_result):
        tally = f"{result['beaten_by']} of {result['trials']}"
        print(f"    {result['label']:<24}{result['real']:>16,.2f}"
              f"{result['shuffled_mean']:>16,.2f}{result['ratio']:>9.2f}{tally:>18}")
    print(f"\n    {price_result['beaten_by']} of {price_result['trials']} shuffles fit "
          f"{price_result['label']} as well as its real order.")
    if price_result["beaten_by"] == 0:
        print("    01 builds each close on the one before, so the order is part of the data.")
    print(f"    {cohort_result['beaten_by']} of {cohort_result['trials']} shuffles fit the "
          f"cohort series at least as well.")
    if cohort_result["beaten_by"] > 0:
        print("    01 draws each customer's assets without regard to the join month, so there")
        print("    is no order to find.")

    # 5. The weekly term of the daily series
    print("\n--- 5. The weekly term of the daily series ---")
    model, forecast = fit_prophet(prices, yearly=True, weekly=True)
    training_days = prices["trade_date"].dt.dayofweek.value_counts().sort_index()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = forecast.assign(dow=forecast["ds"].dt.dayofweek).groupby("dow")["weekly"].mean()
    print(f"    {'day':<6}{'training rows':>15}{'weekly term':>14}")
    for day in range(7):
        rows = int(training_days.get(day, 0))
        print(f"    {names[day]:<6}{rows:>15,}{weekly.get(day, float('nan')):>14.4f}")
    weekend_rows = int(training_days.get(5, 0) + training_days.get(6, 0))
    forecast_weekend = int((forecast["ds"].dt.dayofweek >= 5).sum())
    weekday_spread = float(weekly.loc[0:4].max() - weekly.loc[0:4].min())
    print(f"\n    weekend rows: {weekend_rows} in training, {forecast_weekend} in the forecast "
          f"(the 14 days after the data)")
    print(f"    sum of the seven values {round(weekly.sum(), 4) + 0.0:.4f}, "
          f"spread across weekdays {weekday_spread:.4f}")
    print("    The weekly term is a periodic function fitted at five of seven positions and")
    print("    evaluated at all seven. Its seven values sum to zero, so Saturday and Sunday")
    print("    take what the five weekdays leave, with no weekend row behind them. 01 plants")
    print("    no weekly pattern in MRD, so the spread across weekdays is noise as well.")

    # 6. The yearly term on one year and on two
    print("\n--- 6. The yearly term on one year and on two ---")
    one_year = prices[prices["trade_date"] < "2024-01-01"]
    spans = {
        "one year of data": one_year,
        "two years of data": prices,
    }
    yearly_terms = {}
    span_days = {}
    ranges = {}
    for label, frame in spans.items():
        days = (frame["trade_date"].max() - frame["trade_date"].min()).days
        span_days[label] = days
        _, fitted = fit_prophet(frame, yearly=True, weekly=False)
        by_month = (
            fitted.assign(month=fitted["ds"].dt.month).groupby("month")["yearly"].mean()
        )
        yearly_terms[label] = by_month
        ranges[label] = by_month.max() - by_month.min()
        print(f"    {label:<20} rows {len(frame):>5}   span {days:>3} days "
              f"({days / 365.25:.2f} years)   range {ranges[label]:>8.2f}")

    left, right = yearly_terms["one year of data"], yearly_terms["two years of data"]
    correlation = float(np.corrcoef(left.to_numpy(), right.to_numpy())[0, 1])
    print(f"\n    {'month':<8}{'from one year':>16}{'from two years':>17}")
    for month in range(1, 13):
        print(f"    {month:<8}{left.get(month, float('nan')):>16.2f}"
              f"{right.get(month, float('nan')):>17.2f}")
    print(f"\n    correlation between the two yearly terms: {correlation:+.4f}")
    print("    01 draws MRD as a random walk with no yearly pattern, so both curves describe")
    print("    noise and trend.")
    one, two = ranges["one year of data"], ranges["two years of data"]
    if two > one:
        print(f"    The two-year curve ranges wider, {two:.2f} against {one:.2f}, and the two "
              f"agree at {correlation:+.2f}.")
    print("    Prophet warns below 730 days of history. These spans are "
          f"{span_days['one year of data']} and {span_days['two years of data']} days,")
    print("    so it warns on both fits, the two-year one included.")


if __name__ == "__main__":
    main()
