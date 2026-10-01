"""This script fits Prophet, an additive model of trend, cycles and events, to the long index, the
cash-flow table and the short listing history, and checks each term (trend changepoints,
seasonality, event effects) against what was planted. The run prints 6 parts:
    1. Fit the long index, pull the trend, cycle and remainder apart, then refit with the
       planted period.
    2. Match the detected trend changepoints against the dates the drift actually changed.
    3. Turn the changepoint flexibility up and down, and count what each setting finds.
    4. Declare the promotion days as events, and read back the lift the model assigned them.
    5. Give the trend a ceiling, and see how far the forecast moves with the ceiling alone.
    6. Ask a series shorter than a year for a yearly cycle, and check whether the answer
       means anything.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
CHANGEPOINT_SCALES = [0.01, 0.05, 0.5]
CEILING_MULTIPLES = [1.05, 2.0, 5.0]
MATCH_TOLERANCE_DAYS = 200
FORECAST_DAYS = 365


def fit_quietly(model: Prophet, frame: pd.DataFrame) -> Prophet:
    """Fit a model with the prophet and cmdstanpy loggers silenced.
    The filter sits on the logger, so handlers the backend adds later are covered too.
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


def nearest_detected(found: pd.Index, target: pd.Timestamp) -> tuple:
    """Return the gap in days to the closest detected changepoint, and that changepoint.

    An empty set gives an infinite gap and None, since a tight prior can detect nothing.
    """
    if len(found) == 0:
        return float("inf"), None
    offsets = np.abs((found - target).days)
    position = int(offsets.argmin())
    return float(offsets[position]), found[position]


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
    print(f"  a share taken against the raw series would have reported the cycle as "
          f"{np.var(fitted['yearly']) / np.var(frame['y']):.1%} and said nothing, because "
          f"the trend spans {fitted['trend'].max() - fitted['trend'].min():.0f} points and "
          f"the cycle spans {fitted['yearly'].max() - fitted['yearly'].min():.0f}")
    planted_swing = float(
        (np.exp(index_truth["season_amplitude_log"]) - 1) * fitted["trend"].mean())
    print(f"  the planted cycle is worth about {planted_swing:.1f} points either side "
          f"of the trend, and this fit recovered {fitted['yearly'].max():.1f}")
    period = index_truth["season_period"]
    period_days = float(np.mean((index.index[period:] - index.index[:-period]).days))
    years = (index.index[-1] - index.index[0]).days / 365.25
    print(f"  the cycle repeats every {period} rows, and rows here are trading days, so one "
          f"cycle is {period_days:.0f} calendar days; the yearly term assumes 365.25, and "
          f"over {years:.1f} years the planted cycle drifts "
          f"{years * (365.25 / period_days - 1):.1f} turns against it, so a fixed yearly "
          f"shape averages most of it away")
    refit = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                    daily_seasonality=False)
    refit.add_seasonality(name="planted", period=period_days, fourier_order=3)
    refit_fit = fit_quietly(refit, frame).predict(frame)
    refit_residual = frame["y"].to_numpy() - refit_fit["yhat"].to_numpy()
    print(f"  refit with a {period_days:.0f}-day seasonality instead: the cycle reaches "
          f"{refit_fit['planted'].max():.1f} against the planted {planted_swing:.1f}, and "
          f"the remainder sd falls from {residual.std():.2f} to {refit_residual.std():.2f}, "
          f"so {1 - np.var(refit_residual) / np.var(residual):.0%} of what the yearly fit "
          f"called remainder was the cycle")

    print("\n--- 2. Match detected changepoints against the planted ones ---")
    deltas = pd.Series(model.params["delta"].mean(axis=0), index=model.changepoints)
    significant = deltas[deltas.abs() > 0.01]
    print(f"  {len(deltas)} candidates carry a slope change, "
          f"{len(significant)} of them larger than 0.01")
    print(f"  {'planted date':<14} {'new log slope':>14}  {'nearest detected':>18}  "
          f"{'gap, days':>10}")
    for date, slope in zip(index_truth["changepoint_date"], index_truth["segment_log_slope"][1:]):
        target = pd.Timestamp(date)
        gap, closest = nearest_detected(significant.index, target)
        label = closest.date().isoformat() if closest is not None else "none detected"
        print(f"  {date:<14} {slope:>+14.5f}  {label:>18}  "
              f"{gap:>10.0f}")
    matched = sum(
        nearest_detected(significant.index, pd.Timestamp(d))[0] <= MATCH_TOLERANCE_DAYS
        for d in index_truth["changepoint_date"])
    print(f"  {matched} of {len(index_truth['changepoint_date'])} planted changes have "
          f"a detected one within {MATCH_TOLERANCE_DAYS} days")
    print("  the fitter was never told where to look; it places candidates on a grid "
          "and shrinks the ones the data does not pay for")
    spacing = model.changepoints.diff().dt.days.mean()
    first, last = model.changepoints.iloc[0], model.changepoints.iloc[-1]
    rng = np.random.default_rng(0)
    random_dates = first + pd.to_timedelta(rng.integers(0, (last - first).days, 2000), unit="D")
    random_rate = np.mean([nearest_detected(significant.index, d)[0] <= MATCH_TOLERANCE_DAYS
                           for d in random_dates])
    print(f"  but the candidates sit {spacing:.0f} days apart, so each gap above is set by "
          f"the grid, and {random_rate:.0%} of random dates in the same span also have a "
          f"detected change within {MATCH_TOLERANCE_DAYS} days; "
          f"{len(index_truth['changepoint_date'])} random dates would all match "
          f"{random_rate ** len(index_truth['changepoint_date']):.0%} of the time")

    print("\n--- 3. Turn the flexibility up and down ---")
    print(f"  {'prior scale':>12}  {'changes > 0.01':>15}  {'planted matched':>16}  "
          f"{'unplanted':>10}  {'in-sample RMSE':>15}")
    matched_by_scale, unplanted_by_scale, rmse_by_scale = [], [], []
    for scale in CHANGEPOINT_SCALES:
        alt = fit_quietly(Prophet(changepoint_prior_scale=scale, yearly_seasonality=True,
                                  weekly_seasonality=False, daily_seasonality=False), frame)
        alt_fit = alt.predict(frame)
        alt_deltas = pd.Series(alt.params["delta"].mean(axis=0), index=alt.changepoints)
        alt_significant = alt_deltas[alt_deltas.abs() > 0.01]
        alt_matched = sum(
            nearest_detected(alt_significant.index,
                             pd.Timestamp(d))[0] <= MATCH_TOLERANCE_DAYS
            for d in index_truth["changepoint_date"])
        rmse = float(np.sqrt(np.mean(
            (frame["y"].to_numpy() - alt_fit["yhat"].to_numpy()) ** 2)))
        unplanted = sum(
            min(abs((cp - pd.Timestamp(d)).days)
                for d in index_truth["changepoint_date"]) > MATCH_TOLERANCE_DAYS
            for cp in alt_significant.index)
        matched_by_scale.append(alt_matched)
        unplanted_by_scale.append(unplanted)
        rmse_by_scale.append(rmse)
        print(f"  {scale:>12.2f}  {len(alt_significant):>15}  "
              f"{alt_matched:>10} of {len(index_truth['changepoint_date'])}  "
              f"{unplanted:>10}  {rmse:>15.2f}")
    planted_count = len(index_truth["changepoint_date"])
    if all(found == planted_count for found in matched_by_scale):
        print(f"  every setting matches all {planted_count} planted changes, so the 'planted "
              f"matched' column separates none of them")
    print(f"  a looser prior adds slope changes the generator never made: 'unplanted' goes "
          f"{' -> '.join(str(n) for n in unplanted_by_scale)}")
    if rmse_by_scale[-1] < rmse_by_scale[0] and unplanted_by_scale[-1] > unplanted_by_scale[0]:
        print(f"  the in-sample RMSE falls {rmse_by_scale[0]:.2f} -> {rmse_by_scale[-1]:.2f}: "
              f"a better fit to the history, bought with "
              f"{unplanted_by_scale[-1] - unplanted_by_scale[0]} more changes that were "
              f"never in it")

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
    day_factor = np.array(flow_truth["day_of_month_factor"])
    on_factor = day_factor[pd.to_datetime(flow_truth["campaign_days"]).day - 1].mean()
    print(f"  the generator also multiplies each day by a day-of-month factor the model "
          f"does not have; on these four days it averages {on_factor:.2f}, and "
          f"{flow_truth['campaign_lift']:.2f} x {on_factor:.2f} = "
          f"{flow_truth['campaign_lift'] * on_factor:.2f}")
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
    floor = float(inflow["y"].min() * 0.5)
    print(f"  floor {floor:,.0f}   horizon {FORECAST_DAYS} days   generator trend present = "
          f"{flow_truth['purchase_has_trend']}")
    print(f"  {'model':<20} {'ceiling':>14} {'trend at the end':>18}  {'share of ceiling':>17}")
    ends = []
    for multiple in CEILING_MULTIPLES:
        ceiling = float(inflow["y"].max() * multiple)
        capped = fit_quietly(
            Prophet(growth="logistic", yearly_seasonality=False, weekly_seasonality=True,
                    daily_seasonality=False), inflow.assign(cap=ceiling, floor=floor))
        future = capped.make_future_dataframe(periods=FORECAST_DAYS)
        end = float(capped.predict(future.assign(cap=ceiling, floor=floor))["trend"].iloc[-1])
        ends.append(end)
        print(f"  {f'logistic, max x {multiple:g}':<20} {ceiling:>14,.0f} {end:>18,.0f}  "
              f"{end / ceiling:>16.1%}")
    linear_out = without_events.predict(
        without_events.make_future_dataframe(periods=FORECAST_DAYS))
    print(f"  {'linear':<20} {'':>14} {float(linear_out['trend'].iloc[-1]):>18,.0f}")
    print(f"  the ceiling is an input, not a finding: with the data unchanged, the ceiling "
          f"alone moved the end of the trend by {max(ends) - min(ends):,.0f}")

    print("\n--- 6. Ask a series shorter than a year for a yearly cycle ---")
    short = as_prophet_frame(listing)
    forced = fit_quietly(Prophet(yearly_seasonality=True, weekly_seasonality=True,
                                 daily_seasonality=False), short)
    forced_fit = forced.predict(short)
    span_days = (short["ds"].max() - short["ds"].min()).days
    print(f"  {len(short)} rows spanning {span_days} days, "
          f"{span_days / 365:.2f} of a year")
    print(f"  the fitter still returned a yearly component with sd "
          f"{forced_fit['yearly'].std():.2f}, against a series that itself has sd "
          f"{short['y'].std():.2f}")
    if forced_fit["yearly"].std() > short["y"].std():
        print(f"  a component larger than its series is possible only because another term "
              f"moves against it: the yearly term and the trend correlate at "
              f"{np.corrcoef(forced_fit['yearly'], forced_fit['trend'])[0, 1]:.2f}")
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
