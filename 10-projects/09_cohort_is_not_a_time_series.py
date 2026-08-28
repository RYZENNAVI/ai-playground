"""Line two series up by month, fit both, and use a shuffle to find out which one has time in it.

Demonstrates that an x axis labelled with dates does not make a series temporal:
    1. Group customers by the month they joined and plot the result as a series.
    2. Count how many customers two neighbouring points have in common.
    3. Put a genuine daily series beside it and count the same overlap.
    4. Fit an autoregressive model to both, then refit with the order shuffled.
    5. Compare each fit against its own shuffled twin, which is the test that separates them.
    6. Decompose the daily series and count the training samples behind its weekly term.
    7. Fit the yearly term on one year and on two, and compare what it claims.

Module 10: Applied Projects - Cohorts, Series, and Identifiable Components.
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
DB_PATH = DATA / "market.sqlite"

TICKER = "MRD"
ARIMA_ORDER = (1, 1, 1)
SHUFFLE_TRIALS = 30
SEED = 20260828


def cohort_series() -> pd.DataFrame:
    """Average assets by the month each customer opened their account.

    Every step here is ordinary. The result has a date on the x axis, one value per
    month, and no gaps. It is the shape a forecasting library accepts, which is
    exactly why the question of whether it should be forecast never gets asked.
    """
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
    """Return the share of each point's population that also appears in the next point.

    This is the question the chart hides. A line drawn between two points asserts
    that something moved from one value to the other. That assertion needs the two
    points to be about the same thing, and this number says whether they are.
    """
    shares = []
    for current, following in zip(members, members[1:]):
        if not current:
            shares.append(0.0)
            continue
        shares.append(len(current & following) / len(current))
    return shares


def fit_arima(values: np.ndarray) -> float:
    """Fit an autoregressive model and return its in-sample mean absolute error.

    The order is fixed rather than searched. Order selection is a separate subject,
    and holding it constant is what makes the shuffled comparison below a comparison
    of the data rather than of two different models.
    """
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(values, order=ARIMA_ORDER).fit()
    residuals = np.asarray(model.resid, dtype=float)
    return float(np.mean(np.abs(residuals[ARIMA_ORDER[1]:])))


def shuffle_test(values: np.ndarray, label: str) -> dict:
    """Refit the same model to the same values in random orders, and compare the errors.

    If order carries information, destroying it should make the fit worse. If the fit
    is just as good on shuffled values, then the model was never using time; it was
    describing the spread of the numbers, which any ordering of them shares.
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
        "shuffled_min": float(shuffled.min()),
        "ratio": float(shuffled.mean() / real) if real else float("nan"),
        "beaten_by": beaten,
        "trials": len(shuffled),
    }


def fit_prophet(frame: pd.DataFrame, yearly: bool, weekly: bool):
    """Fit an additive decomposition to a dated series and return the model and its fit."""
    from prophet import Prophet

    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly,
                    daily_seasonality=False)
    model.fit(frame.rename(columns={"trade_date": "ds", "close": "y"})[["ds", "y"]])
    future = model.make_future_dataframe(periods=14)
    return model, model.predict(future)


def main() -> None:
    print("--- 1. Assets by the month customers joined ---")
    cohorts = cohort_series()
    print(f"    points {len(cohorts)}, running {cohorts['period'].iloc[0]} to "
          f"{cohorts['period'].iloc[-1]}")
    print(f"    {'period':<10}{'customers':>11}{'mean assets':>15}")
    for row in cohorts.head(4).itertuples():
        print(f"    {row.period:<10}{row.customers:>11,}{row.value:>15,.0f}")
    print(f"    ... {len(cohorts) - 4} more months")

    print("\n--- 2. What two neighbouring points have in common ---")
    shares = overlap_between_neighbours(cohorts["members"].tolist())
    print(f"    mean share of a month's customers who also appear in the next month: "
          f"{np.mean(shares):.4f}")
    print(f"    highest such share across all {len(shares)} pairs: {max(shares):.4f}")
    print("    Each point is a different set of people. The line between two points")
    print("    does not trace anything moving; it connects two separate populations.")

    print("\n--- 3. A daily series, for contrast ---")
    prices = price_series()
    print(f"    points {len(prices)}, running {prices['trade_date'].min().date()} to "
          f"{prices['trade_date'].max().date()}")
    print("    share of one point's subject that appears in the next point: 1.0000")
    print("    It is the same instrument every day, which is what makes the change")
    print("    between two points a real quantity.")

    print(f"\n--- 4-5. The shuffle test, ARIMA{ARIMA_ORDER}, {SHUFFLE_TRIALS} shuffles each ---")
    results = [
        shuffle_test(cohorts["value"].to_numpy(dtype=float), "cohort by join month"),
        shuffle_test(prices["close"].to_numpy(dtype=float), f"{TICKER} daily close"),
    ]
    print(f"    {'series':<24}{'error as given':>16}{'shuffled mean':>16}"
          f"{'ratio':>9}{'shuffles as good':>18}")
    for result in results:
        tally = f"{result['beaten_by']} of {result['trials']}"
        print(f"    {result['label']:<24}{result['real']:>16,.2f}"
              f"{result['shuffled_mean']:>16,.2f}{result['ratio']:>9.2f}{tally:>18}")
    print("\n    On the daily series, destroying the order makes the fit far worse, so the")
    print("    order was carrying something. On the cohort series the shuffled fits land")
    print("    in the same place, which means the model was never reading time out of it.")
    print("    A forecast from that model extrapolates the spread of twelve group means.")

    print("\n--- 6. The weekly term of the daily series ---")
    model, forecast = fit_prophet(prices, yearly=True, weekly=True)
    training_days = prices["trade_date"].dt.dayofweek.value_counts().sort_index()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = forecast.assign(dow=forecast["ds"].dt.dayofweek).groupby("dow")["weekly"].mean()
    print(f"    {'day':<6}{'training rows':>15}{'weekly term':>14}")
    for day in range(7):
        rows = int(training_days.get(day, 0))
        print(f"    {names[day]:<6}{rows:>15,}{weekly.get(day, float('nan')):>14.4f}")
    weekend_rows = int(training_days.get(5, 0) + training_days.get(6, 0))
    weekend_term = float(abs(weekly.get(5, 0)) + abs(weekly.get(6, 0)))
    print(f"\n    weekend training rows {weekend_rows}, weekend weekly term "
          f"{weekend_term:.4f}")
    print("    The model still assigns Saturday and Sunday a value. It has to: the")
    print("    weekly term is a periodic function fitted to five of seven positions and")
    print("    then evaluated at all seven. Nothing in the output marks the two")
    print("    positions that no observation ever constrained.")

    print("\n--- 7. The yearly term on one year and on two ---")
    one_year = prices[prices["trade_date"] < "2024-01-01"]
    spans = {
        "one year of data": one_year,
        "two years of data": prices,
    }
    yearly_terms = {}
    for label, frame in spans.items():
        years = (frame["trade_date"].max() - frame["trade_date"].min()).days / 365.25
        _, fitted = fit_prophet(frame, yearly=True, weekly=False)
        by_month = (
            fitted.assign(month=fitted["ds"].dt.month).groupby("month")["yearly"].mean()
        )
        yearly_terms[label] = by_month
        print(f"    {label:<20} rows {len(frame):>5}   complete cycles covered "
              f"{years:>4.2f}   term ranges {by_month.max() - by_month.min():>8.2f}")

    left, right = yearly_terms["one year of data"], yearly_terms["two years of data"]
    correlation = float(np.corrcoef(left.to_numpy(), right.to_numpy())[0, 1])
    print(f"\n    {'month':<8}{'from one year':>16}{'from two years':>17}")
    for month in range(1, 13):
        print(f"    {month:<8}{left.get(month, float('nan')):>16.2f}"
              f"{right.get(month, float('nan')):>17.2f}")
    print(f"\n    correlation between the two yearly terms: {correlation:+.4f}")
    print("    One year of data contains one pass through the calendar, so a yearly")
    print("    term fitted on it cannot be separated from the trend it sits on. Both")
    print("    fits succeed and both print a clean seasonal curve; the number that")
    print("    tells them apart is how many complete cycles the data covered.")


if __name__ == "__main__":
    main()
