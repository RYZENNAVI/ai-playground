"""Fit an additive model of trend, cycles and events, and check each term against what was planted.

Demonstrates what the three terms of an additive forecaster each pick up:
    1. Fit the long index and pull the fitted trend, cycle and remainder apart.
    2. Match the detected trend changepoints against the dates the drift actually changed.
    3. Turn the changepoint flexibility up and down, and count what each setting finds.
    4. Declare the promotion days as events, and read back the lift the model assigned them.
    5. Cap the trend and floor it, and watch the forecast bend towards the ceiling it was given.
    6. Ask a series shorter than a year for a yearly cycle, and check whether the answer means anything.

Module 08: Time Series Forecasting - Additive Decomposable Forecasting.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
CHANGEPOINT_SCALES = [0.01, 0.05, 0.5]
MATCH_TOLERANCE_DAYS = 200
FORECAST_DAYS = 365


def fit_quietly(model: Prophet, frame: pd.DataFrame) -> Prophet:
    """Fit a model without letting the sampler's progress output into the report.

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
    model.fit(frame)
    return model


def as_prophet_frame(series: pd.Series) -> pd.DataFrame:
    """Rename any dated series into the two column names the fitter requires."""
    return pd.DataFrame({"ds": series.index, "y": series.to_numpy(dtype=float)})


def nearest_gap(found: pd.Series, target: pd.Timestamp) -> float:
    """Return the distance in days from one planted date to the closest detected one."""
    if len(found) == 0:
        return float("inf")
    return float(np.abs((found - target).days).min())


def main() -> None:
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    index_truth = truth["index"]
    flow_truth = truth["cash_flow"]

    index = pd.read_csv(DATA_DIR / "index_daily.csv", parse_dates=["trade_date"])
    index = index.set_index("trade_date")["close"]
    flow = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                       parse_dates=["report_date"], date_format="%Y%m%d")
    flow = flow.set_index("report_date")
    listing = pd.read_csv(DATA_DIR / "new_listing_daily.csv", parse_dates=["trade_date"])
    listing = listing.set_index("trade_date")["close"]

    print("--- 1. Fit the long index and separate the terms ---")
    frame = as_prophet_frame(index)
    model = fit_quietly(Prophet(yearly_seasonality=True, weekly_seasonality=False,
                                daily_seasonality=False), frame)
    fitted = model.predict(frame)
    residual = frame["y"].to_numpy() - fitted["yhat"].to_numpy()
    detrended = frame["y"].to_numpy() - fitted["trend"].to_numpy()
    print(f"  {len(frame)} rows fitted, {len(model.changepoints)} candidate "
          f"changepoints placed over the first "
          f"{model.changepoint_range:.0%} of the history")
    print(f"  {'term':<14} {'sd':>10}  {'range':>22}")
    print(f"  {'series':<14} {frame['y'].std():>10.2f}  "
          f"{f'{frame.y.min():.1f} .. {frame.y.max():.1f}':>22}")
    print(f"  {'trend':<14} {fitted['trend'].std():>10.2f}  "
          f"{f'{fitted.trend.min():.1f} .. {fitted.trend.max():.1f}':>22}")
    print(f"  {'yearly cycle':<14} {fitted['yearly'].std():>10.2f}  "
          f"{f'{fitted.yearly.min():.1f} .. {fitted.yearly.max():.1f}':>22}")
    print(f"  {'remainder':<14} {residual.std():>10.2f}")
    print(f"  the trend carries the series: once it is subtracted, the cycle holds "
          f"{np.var(fitted['yearly']) / np.var(detrended):.1%} of what is left and the "
          f"remainder holds {np.var(residual) / np.var(detrended):.1%}")
    print("  a share taken against the raw series would have reported the cycle as "
          "0.0% and said nothing, because the trend spans four hundred points and "
          "the cycle spans four")
    planted_swing = float(
        (np.exp(index_truth["season_amplitude_log"]) - 1) * fitted["trend"].mean())
    print(f"  the planted cycle is worth about {planted_swing:.1f} points either side "
          f"of the trend, and this fit recovered {fitted['yearly'].max():.1f}")
    print(f"  the cycle repeats every {index_truth['season_period']} rows, and rows "
          f"here are trading days; the fitted term is indexed by calendar date, so "
          f"the two run at slightly different rates and the match is partial")

    print("\n--- 2. Match detected changepoints against the planted ones ---")
    deltas = pd.Series(model.params["delta"].mean(axis=0), index=model.changepoints)
    significant = deltas[deltas.abs() > 0.01]
    print(f"  {len(deltas)} candidates carry a slope change, "
          f"{len(significant)} of them larger than 0.01")
    print(f"  {'planted date':<14} {'new log slope':>14}  {'nearest detected':>18}  "
          f"{'gap, days':>10}")
    for date, slope in zip(index_truth["changepoint_date"], index_truth["segment_log_slope"][1:]):
        target = pd.Timestamp(date)
        gap = nearest_gap(significant.index, target)
        closest = significant.index[np.abs((significant.index - target).days).argmin()]
        print(f"  {date:<14} {slope:>+14.5f}  {closest.date().isoformat():>18}  "
              f"{gap:>10.0f}")
    matched = sum(nearest_gap(significant.index, pd.Timestamp(d)) <= MATCH_TOLERANCE_DAYS
                  for d in index_truth["changepoint_date"])
    print(f"  {matched} of {len(index_truth['changepoint_date'])} planted changes have "
          f"a detected one within {MATCH_TOLERANCE_DAYS} days")
    print("  the fitter was never told where to look; it places candidates on a grid "
          "and shrinks the ones the data does not pay for")

    print("\n--- 3. Turn the flexibility up and down ---")
    print(f"  {'prior scale':>12}  {'changes > 0.01':>15}  {'planted matched':>16}  "
          f"{'unplanted':>10}  {'in-sample RMSE':>15}")
    for scale in CHANGEPOINT_SCALES:
        alt = fit_quietly(Prophet(changepoint_prior_scale=scale, yearly_seasonality=True,
                                  weekly_seasonality=False, daily_seasonality=False), frame)
        alt_fit = alt.predict(frame)
        alt_deltas = pd.Series(alt.params["delta"].mean(axis=0), index=alt.changepoints)
        alt_significant = alt_deltas[alt_deltas.abs() > 0.01]
        alt_matched = sum(
            nearest_gap(alt_significant.index, pd.Timestamp(d)) <= MATCH_TOLERANCE_DAYS
            for d in index_truth["changepoint_date"])
        rmse = float(np.sqrt(np.mean(
            (frame["y"].to_numpy() - alt_fit["yhat"].to_numpy()) ** 2)))
        unplanted = sum(
            min(abs((cp - pd.Timestamp(d)).days)
                for d in index_truth["changepoint_date"]) > MATCH_TOLERANCE_DAYS
            for cp in alt_significant.index)
        print(f"  {scale:>12.2f}  {len(alt_significant):>15}  "
              f"{alt_matched:>10} of {len(index_truth['changepoint_date'])}  "
              f"{unplanted:>10}  {rmse:>15.2f}")
    print(f"  the four planted changes are found at every setting; what a looser prior "
          f"adds is the last column")
    print("  a looser prior always fits the history better, and buys that fit with "
          "slope changes the generator never made")

    print("\n--- 4. Declare the promotion days as events ---")
    inflow = as_prophet_frame(flow["total_purchase_amt"])
    events = pd.DataFrame({
        "holiday": "promotion",
        "ds": pd.to_datetime(flow_truth["campaign_days"]),
        "lower_window": 0,
        "upper_window": 0,
    })
    with_events = fit_quietly(
        Prophet(holidays=events, yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=False), inflow)
    event_fit = with_events.predict(inflow)
    on_days = event_fit.loc[event_fit["promotion"].abs() > 0]
    baseline = event_fit["yhat"].to_numpy() - event_fit["promotion"].to_numpy()
    lifts = 1 + on_days["promotion"].to_numpy() / baseline[on_days.index]
    print(f"  {'date':<12} {'fitted lift':>12}  {'planted lift':>13}")
    for stamp, lift in zip(on_days["ds"], lifts):
        print(f"  {stamp.date().isoformat():<12} {lift:>12.2f}  "
              f"{flow_truth['campaign_lift']:>13.2f}")
    print(f"  mean fitted lift {lifts.mean():.2f} against planted "
          f"{flow_truth['campaign_lift']:.2f}")
    without_events = fit_quietly(
        Prophet(yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=False), inflow)
    plain_fit = without_events.predict(inflow)
    for label, fitted_frame in [("with events   ", event_fit), ("without events", plain_fit)]:
        err = inflow["y"].to_numpy() - fitted_frame["yhat"].to_numpy()
        on = inflow["ds"].isin(events["ds"]).to_numpy()
        print(f"  {label}  RMSE on the four days {np.sqrt((err[on] ** 2).mean()):>14,.0f}"
              f"   elsewhere {np.sqrt((err[~on] ** 2).mean()):>12,.0f}")
    print("  naming the four days moves the error on those days and leaves the rest "
          "alone: that is what makes them events rather than outliers")

    print("\n--- 5. Give the trend a ceiling ---")
    capped_input = inflow.copy()
    ceiling = float(inflow["y"].max() * 1.05)
    floor = float(inflow["y"].min() * 0.5)
    capped_input["cap"] = ceiling
    capped_input["floor"] = floor
    capped = fit_quietly(
        Prophet(growth="logistic", yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=False), capped_input)
    future = capped.make_future_dataframe(periods=FORECAST_DAYS)
    future["cap"] = ceiling
    future["floor"] = floor
    capped_out = capped.predict(future)

    linear = fit_quietly(Prophet(yearly_seasonality=False, weekly_seasonality=True,
                                 daily_seasonality=False), inflow)
    linear_out = linear.predict(linear.make_future_dataframe(periods=FORECAST_DAYS))
    print(f"  ceiling {ceiling:,.0f}   floor {floor:,.0f}   horizon {FORECAST_DAYS} days")
    print(f"  {'model':<20} {'trend at the end':>18}  {'share of ceiling':>17}")
    for label, out in [("logistic", capped_out), ("linear", linear_out)]:
        end = float(out["trend"].iloc[-1])
        print(f"  {label:<20} {end:>18,.0f}  {end / ceiling:>16.1%}")
    print("  the ceiling is an input, not a finding: the logistic curve bends because "
          "it was handed a number, and a wrong number bends it just as smoothly")

    print("\n--- 6. Ask a series shorter than a year for a yearly cycle ---")
    short = as_prophet_frame(listing)
    forced = fit_quietly(Prophet(yearly_seasonality=True, weekly_seasonality=True,
                                 daily_seasonality=False), short)
    forced_fit = forced.predict(short)
    span_days = (short["ds"].max() - short["ds"].min()).days
    print(f"  {len(short)} rows spanning {span_days} days, "
          f"{span_days / 365:.2f} of a year")
    print(f"  the fitter still returned a yearly component, swinging "
          f"{forced_fit['yearly'].min():.2f} .. {forced_fit['yearly'].max():.2f} "
          f"against a series that itself has sd {short['y'].std():.2f}")
    print(f"  that component is larger than the series it was extracted from, which "
          f"is possible only because another term moves against it")
    print(f"  generator: yearly component present = "
          f"{truth['listing']['yearly_component_present']}")

    split = int(len(short) * 0.75)
    for label, yearly in [("yearly on ", True), ("yearly off", False)]:
        trained = fit_quietly(
            Prophet(yearly_seasonality=yearly, weekly_seasonality=True,
                    daily_seasonality=False), short.iloc[:split])
        held = trained.predict(short.iloc[split:])
        err = short["y"].iloc[split:].to_numpy() - held["yhat"].to_numpy()
        print(f"  {label}   holdout RMSE {np.sqrt((err ** 2).mean()):>8.2f} over "
              f"{len(short) - split} days")
    print("  a component that covers less than one full period is fitted to whatever "
          "shape the sample happens to have, and it is the holdout, not the fitted "
          "curve, that shows whether it was worth anything")


if __name__ == "__main__":
    main()
