# Time series forecasting: baselines and backtests

Script 01 generates every series in this module from a mechanism written down in the file and
records each parameter in `data/ground_truth.json`, so every later score can be checked against
a known answer. Scripts 02 to 04 take the classic route: decomposition, stationarity tests,
ARIMA and SARIMAX in statsmodels, and Prophet. Scripts 05 and 06 take two others, a product of
periodic factors and a small LSTM on windowed rows. Script 07 backtests the candidates over four
cut-off dates and writes the forecast file. This document explains what each script does and the
ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_build_time_series_datasets.py` | Five synthetic series with known ground truth: a daily cash-flow table, a 27-year index, a short listing history, a monthly retail total and an ARMA series |
| 02 | `02_decompose_and_stationarity.py` | Seasonal decomposition at the right period and three wrong ones, STL, and the ADF and KPSS tests, where a strong cycle hides a random walk |
| 03 | `03_arima_grid_search_and_forecast.py` | An AIC grid search over ARIMA and SARIMAX orders, silent non-convergence, a truncated grid, forecast dates, in-sample against out-of-sample, and the differencing order priced on a holdout |
| 04 | `04_prophet_trend_seasonality_changepoints.py` | Prophet's trend, seasonality and event terms checked against planted ones: changepoints, a 350-day cycle, promotion days, a carrying capacity and a series shorter than a year |
| 05 | `05_periodic_factor_baseline.py` | Periodic factors three ways (additive dummies, ratios, alternating fit) scored against planted factors and against ARIMA |
| 06 | `06_lstm_windowed_forecast.py` | Sliding windows, a random split that leaks every test target, a time split scaled from the training side, and an LSTM against two baselines and an oracle |
| 07 | `07_rolling_origin_backtest.py` | A rolling-origin backtest of four routes, a route chosen per column, and a forecast file checked after it is written |

## Shared setup

*   Script 01 writes everything the others read into `data/`, so it runs first. The other six
    are independent of each other. Script 07 writes `outputs/cash_flow_forecast.csv`. `data/`
    and `outputs/` are rebuilt by a rerun and are not tracked.
*   Everything runs offline on CPU. No script needs an API key or a download.
*   The dependencies are numpy, pandas, matplotlib, statsmodels, prophet and torch. Current
    statsmodels has no standalone ARMA class, so an ARMA is written `ARIMA(data, order=(p, 0,
    q))`, and pandas 3 resamples with `"ME"`, `"QE"` and `"YE"`.

## Script 01: Series with a known mechanism

A downloaded series can say that a method scored well. It cannot say whether the method found
what is really there. A generated series can, because the answer was written down first. The
seed is fixed (`SEED = 20260827`), so a rerun rewrites identical files.

| File | Rows | What it is for |
| :--- | ---: | :--- |
| `fund_flow_daily.csv` | 427 | Daily inflow and outflow, a weekday cycle times a day-of-month cycle |
| `index_daily.csv` | 7145 | About 27 years of business days: five drift regimes and one cycle |
| `new_listing_daily.csv` | 198 | A history too short to contain a yearly cycle |
| `retail_sales_monthly.csv` | 42 | A monthly total with a straight trend and a month-of-year cycle |
| `arma_series.csv` | 320 | A series drawn from an ARMA recursion of fixed order |

*   Part 1 builds each cash-flow column as base level × weekday factor × day-of-month factor ×
    noise. Inflow runs from 1.21 on Tuesday down to 0.71 on Saturday. Outflow peaks on Monday
    (1.22) and dips less at the weekend. The day-of-month factor is a lookup table (day 1 is
    1.34, day 13 is 0.89, day 31 is 1.26), so a fitted factor can be compared entry by entry.
    Four promotion days (2013-11-11, 2013-12-12, 2014-06-18, 2014-08-08) are lifted by 1.55.
*   Part 2 gives outflow two extra terms: a growth rate of 0.0007 a day and a random walk with
    σ = 0.035. Growth alone would still test stationary around a trend. The random walk is
    what a unit-root test is built to catch, and script 02 checks whether it does.
*   Part 3 builds the index in log space: changepoints at rows 900, 2400, 3900 and 5600, five
    slopes, an AR(1) term of 0.42, and a cycle of exactly 250 rows. The rows are business
    days, so the cycle is about 350 calendar days, not 365.
*   Part 4 writes 198 business days against 250 in a trading year, and records
    `covers_full_year` as False. Nothing in this generator repeats yearly.
*   Part 5 builds 42 months with a slope of 2.35 a month and twelve month-of-year effects,
    3.5 observations each. The ARMA series is written out term by term with AR (0.62, −0.31)
    and MA (0.45), order (2, 0, 1), and its first 200 points are dropped because they still
    carry the zero start state.
*   Part 6 writes every factor, changepoint and coefficient to `ground_truth.json`.

## Script 02: Decomposition and stationarity

`seasonal_decompose` splits a series into trend, seasonal and residual. Its one choice is
`period`, and the right answer is how many rows one cycle occupies. A decomposition always
returns a seasonal component of the length it was asked for, so the criterion is correlation
with the planted cycle, not a plot.

*   Part 1 decomposes the index at the planted 250 and Part 2 at three wrong periods:

    | `period` | Correlation with the planted cycle | Half-amplitude | Remainder sd |
    | ---: | ---: | ---: | ---: |
    | 250 (planted) | +0.9981 | 0.0617 | 0.01293 |
    | 60 | +0.0002 | 0.0026 | 0.01251 |
    | 288 | +0.0540 | 0.0115 | 0.04760 |
    | 500 | +0.9941 | 0.0651 | 0.01796 |

*   Every run returned a component, and two of four track the cycle. 500 is a whole multiple
    of 250 and stays in phase, so a high correlation is necessary but not sufficient.
*   Part 3 runs STL, which lets the cycle change shape over time. It scores +0.9849 with
    remainder sd 0.01671, below the moving average's +0.9981. The planted cycle never changes
    shape, so the method that holds it fixed fits better here.
*   Part 4 runs the augmented Dickey-Fuller test. Its null is a unit root, so a small p-value
    is evidence against a random walk, and a more negative statistic means more stationary.
    Inflow scores −4.776 against a 1% critical value of −3.470. Outflow, which carries a
    random walk, is called stationary too, against the generator.
*   Part 5 finds why with a Monte Carlo on 300 random walks of 184 rows, where every
    "stationary" verdict is wrong:

    | What the test is shown | Wrong |
    | :--- | ---: |
    | The walk alone | 9% |
    | The walk under the weekday and month-position cycles | 28% |
    | The same walk with the cycles divided back out | 9% |

*   The cycles triple the error rate: a strong cycle reads as strong mean reversion. The third
    row only confirms that dividing out is exact. So decompose first, then test. After the
    cycles are removed, this outflow column is still called stationary: at 184 rows the test
    is wrong 9% of the time, and this column is one of those times.
*   Part 6 differences the outflow column up to twice:

    | Series | sd | ADF p |
    | :--- | ---: | ---: |
    | d=0 | 93,197,206 | 0.0299 |
    | d=1 | 85,194,988 | 0.0000 |
    | d=2 | 130,537,676 | 0.0000 |

*   All three pass. A test can say a series is differenced enough, never too much, and d=2
    inflates the spread 1.53x. Script 03 settles the order on a holdout instead.
*   KPSS reverses the null (stationarity), so it is run as a cross-check. Around a level it
    agrees with ADF on all three rows, which is worth nothing on raw outflow, because both
    allow only a level and the column was built with growth. Around a trend
    (`regression="ct"`), KPSS contradicts ADF on raw outflow, as the generator says it should.
    A cross-check has to be able to fail before its passing means anything.

## Script 03: ARIMA grid search and forecast

AR regresses a series on its own past values and MA on past shocks. ARMA combines the two, and
ARIMA(p, d, q) runs an ARMA on the series differenced d times. SARIMAX adds a seasonal order,
but only when `seasonal_order` is passed. AIC = 2k − 2 ln L penalises parameters and rewards
fit. A search fits every order on a grid and keeps the smallest AIC, which is only comparable
on the same data.

*   Part 1 searches sixteen orders on the series built as ARMA(2, 0, 1):

    | Order | AIC | Gap to best | Max ACF gap to planted, lags 1 to 12 |
    | :--- | ---: | ---: | ---: |
    | (1, 0, 3) | 5009.77 | 0.00 | 0.0827 |
    | (3, 0, 3) | 5009.87 | 0.11 | 0.0742 |
    | (2, 0, 1), planted | 5010.64 | 0.88 | 0.0494 |
    | (3, 0, 0) | 5011.46 | 1.69 | |

*   The planted order was not returned, and the top four sit within 2 AIC. The top three
    orders give nearly the same autocorrelations (planted lag 1 is 0.621), so AIC recovered
    the dynamics, not the order label.
*   Part 2 resamples the index to month, quarter and year. The yearly series has 29 points and
    keeps only 14% of the planted cycle's spread, because the cycle repeats close to once a
    year. Aggregating is not free smoothing.
*   Part 3 searches a 40-order seasonal grid on the monthly total. A fit that runs out of
    optimiser iterations raises nothing and still returns an AIC, so the script records
    `mle_retvals["converged"]` for every row and sorts failures last:

    | Iteration budget | Converged | Best AIC overall | Best converged fit |
    | ---: | ---: | :--- | :--- |
    | 50, the default | 6 / 40 | (4, 0, 0) at 187.45, not converged | (0, 0, 0) at 197.83 |
    | 1000 | 23 / 40 | (3, 0, 3) at 185.25 | (3, 0, 3) at 185.25 |

*   At 1000 iterations the other 17 stop on a failed line search, and Part 3 prints the reason
    counts. The winner forecasts four months with an 80% band, requested as
    `conf_int(alpha=0.20)` (mean width 13.96, against 21.35 for the default 95%). The
    reference is the generator's noiseless expectation rebuilt from `ground_truth.json`: the
    mean gap is 4.5, and the band covers it in 3 of 4 months.
*   Part 4 slices the grid with `[:20]`. That keeps the first twenty in lexicographic order,
    which pins p to 0 to 2. The truncated run picks (1, 0, 3) at 185.47 and never fits the
    real winner. Both runs print a best AIC, so the candidate count has to be printed too.
*   Part 5 labels the forecast. A hand-rolled loop that adds the current month's length drifts
    off the month end in all 365 start dates tested and skips a whole month in 14. The fix is
    `pd.date_range(..., freq="ME")` plus three assertions: distinct, on month ends, equal to
    the fitted index. A wrong label raises nothing, so it has to be asserted.
*   Part 6 separates in-sample from out-of-sample on the last twelve months:

    | Last twelve months | RMSE |
    | :--- | ---: |
    | One step ahead, parameters that saw them | 4.14 |
    | One step ahead, parameters that did not | 13.64 |
    | 1 to 12 steps ahead, parameters that did not | 10.44 |

*   The first two rows differ only in whether the parameters saw the months, so the 3.3x gap
    comes from seeing the data. Only rows from a model that did not see them describe a
    forecast.
*   Part 7 prices the differencing order on 30 held-out days of outflow:

    | Order | Holdout RMSE | vs best |
    | :--- | ---: | ---: |
    | (2, 0, 2) | 107,195,972 | 1.32x |
    | (2, 1, 2) | 111,569,384 | 1.37x |
    | (2, 2, 2) | 100,336,063 | 1.24x |
    | (2, 0, 2) × (1, 0, 1, 7) | 81,217,127 | 1.00x |

*   AIC is not comparable across d, so the holdout is the criterion. The three choices of d
    span 11.2 million. Adding a seven-day term cut 26.0 million, more than twice that span.
    The largest structure repeats weekly, and the differencing order was the wrong knob.

## Script 04: Prophet against planted structure

Prophet fits y(t) = g(t) + s(t) + h(t) + ε: a piecewise linear or logistic trend, Fourier
seasonality, and user-supplied events. It places 25 candidate changepoints over the first 80% of
the history and shrinks most of them with `changepoint_prior_scale` (default 0.05).
`make_future_dataframe` includes the historical dates by default.

*   Part 1 fits the index. The yearly term recovers a cycle of 2.2 points, against 15.3
    planted, because the cycle is 250 business days (about 350 calendar days) and the yearly
    term assumes 365.25. Over 27.4 years the two drift 1.2 turns apart. A refit with
    `add_seasonality(name="planted", period=350, fourier_order=3)` recovers 15.0 and cuts the
    remainder sd from 11.50 to 4.87.
*   Part 2 matches detected changepoints against the four planted ones:

    | Planted date | Nearest detected | Gap, days |
    | :--- | :--- | ---: |
    | 1994-06-01 | 1994-06-21 | 20 |
    | 2000-03-01 | 1999-09-23 | 160 |
    | 2005-11-30 | 2005-11-10 | 20 |
    | 2012-06-06 | 2011-12-29 | 160 |

*   All four match within 200 days, but that is weaker than it looks. The candidates sit 320
    days apart, 71% of random dates also have a detected change within 200 days, and four
    random dates would all match 26% of the time.
*   Part 3 varies the prior scale:

    | Prior scale | Changes above 0.01 | Planted matched | Unplanted | In-sample RMSE |
    | ---: | ---: | ---: | ---: | ---: |
    | 0.01 | 13 | 4 / 4 | 7 | 11.72 |
    | 0.05 | 17 | 4 / 4 | 11 | 11.50 |
    | 0.50 | 22 | 4 / 4 | 16 | 11.48 |

*   The planted column separates none of the settings. A looser prior only adds changes the
    generator never made, while the in-sample fit improves.
*   Part 4 declares the four promotion days as events (`lower_window` and `upper_window` 0).
    The fitted lifts are 1.37 to 1.45 against a planted 1.55, because the model has no
    day-of-month term: on those four days that factor averages 0.91, and 1.55 × 0.91 = 1.41.
    The lifts can move with the Prophet or Stan version.

    | | RMSE on the four days | RMSE elsewhere |
    | :--- | ---: | ---: |
    | Events declared | 23,172,549 | 42,710,487 |
    | Not declared | 136,764,812 | 42,749,153 |

*   Naming the days moved the error on those days and left the rest alone.
*   Part 5 fits a logistic trend with a ceiling (`cap` on every row, history and future) on a
    series with no planted trend. Setting the ceiling at 1.05, 2 and 5 times the maximum puts
    the end of the trend at 306.8, 318.6 and 325.6 million. The ceiling is an input, and it
    alone moves the forecast 18.7 million.
*   Part 6 asks the 198-row listing for a yearly term. The returned component has sd 4.97,
    larger than the series' 4.24, and correlates −0.72 with the trend. On a 50-day holdout the
    RMSE is 8.03 with the yearly term and 7.18 without it. A component covering less than one
    period fits whatever shape the sample has.

## Script 05: Periodic factors

A periodic factor is a group mean divided by the overall mean: a weekday factor of 1.2 means
that weekday runs 20% above average. The forecast is a base level times a weekday factor times a
day-of-month factor. An autoregressive model learns "the value seven steps ago". A periodic
factor learns "today is a Monday", which needs only the calendar.

*   Part 1 averages inflow by weekday and by day of month. Training is 152 days and 31 are
    held out. The one promotion day in training, 2014-06-18, is left out, because the
    holdout's promotion day is scored separately. Left in, it pushed the joint Wednesday
    factor to 1.1483 against a planted 1.1228.
*   Part 2 fits additive dummies, `C(weekday) + C(day)` in one OLS regression. The effects add,
    while the data multiplies: Saturday takes about 30% off the whole level, not a fixed
    amount.
*   Part 3 takes each effect alone as ratios of group means. That is exact only when every
    weekday meets every day of the month equally often, so each set is contaminated by the
    other.
*   Part 4 fits both jointly by alternation. Each update divides out what the other set
    explains:

    | Design | Why |
    | :--- | :--- |
    | Alternate until no factor moves more than 1e-6, capped at 20 rounds | Each set depends on the other. It settles after 13 rounds here |
    | Renormalise to mean 1 after every update | Otherwise one set can grow tenfold and the other shrink tenfold with the product unchanged |
    | `EPS = 1e-8` in the denominator | Each round divides by a fitted value, which may be zero |

*   After one round the joint weekday factors equal the ratio estimate, because the month
    factors start at 1. The later rounds do all the correction.

    | Method | Mean weekday gap | Mean month-position gap |
    | :--- | ---: | ---: |
    | One effect at a time | 0.0156 | 0.0311 |
    | Joint alternation | 0.0093 | 0.0171 |

*   Part 5 forecasts the held-out month. The reference is handed the planted factors, but its
    level is still the training mean and it is as blind to the promotion day as every route:

    | Route | All 31 days | Ordinary days | vs reference |
    | :--- | ---: | ---: | ---: |
    | Planted-factor reference | 35,923,966 | 19,935,489 | 1.00x |
    | Additive dummies | 37,022,997 | 20,811,944 | 1.04x |
    | Ratio, one at a time | 38,470,996 | 23,016,503 | 1.15x |
    | Joint alternation | 37,679,383 | 21,737,999 | 1.09x |
    | ARIMA with a weekly term | 39,159,792 | 27,239,251 | 1.37x |
    | Last week repeated | 68,488,662 | 62,836,277 | 3.15x |

*   The promotion day carries 70% of the reference's squared error, so the ratios use ordinary
    days. All three factor routes land within 1.15x of the reference with 39 numbers (7
    weekday factors, 31 day factors, one level). ARIMA has the weekly period but no
    month-position term. The wrong-shaped dummies still score best of the three here.
*   Part 6 prices the contamination. The mean month factor each weekday meets spans only
    0.0224 across the week. Refitted on the same dates with planted factors and no noise, the
    ratio estimate is off by 0.0051 and the joint fit by 0.0000. The mechanism is right, and
    the gain is small here. How much bias there is to remove depends on the data.

## Script 06: Windowed rows and an LSTM

A series has no feature columns. A sliding window makes them: each row takes the previous
observations as inputs and the next one as its target. Neighbouring rows share all but one
input, which is why the split has to follow time.

*   Part 1 windows 427 observations at `WINDOW = 14`, `HORIZON = 1` into 413 rows. Rows 0 and 1
    share 13 of 14 inputs, and row 0's target is one of row 1's inputs. An assertion pins
    HORIZON to 1, because the baselines are single-step comparisons.
*   Part 2 splits at random, for comparison only. The test rows touch 359 observations, and all
    359 appear in training rows. All 60 test targets were already read as training inputs, so
    the score measures interpolation inside history the model has seen.
*   Part 3 splits by time into 293 training rows, 60 validation and 60 final test. Train and
    final test share no observation, train and validation share 14 at the boundary. Scaling
    comes from the training rows only (centre 321,108,884). All rows would give 322,119,055, a
    0.31% shift that leaks later periods into training. Nothing fits on, scores or selects
    from the final test before Part 5.
*   Part 4 trains one LSTM layer and a linear head, 9,841 parameters, for 120 full-batch
    epochs. Validation bottoms at epoch 64 (0.2551) and ends at 0.2981, while training loss
    keeps falling. The script restores the epoch 64 weights, which is checkpoint selection,
    not early stopping: all 120 epochs still run. The low point falls between the printed
    rows, so the loss is measured every epoch.
*   Part 5 opens the final test once:

    | Kind | Route | RMSE over 60 days | Ordinary days | ÷ oracle | ÷ model |
    | :--- | :--- | ---: | ---: | ---: | ---: |
    | Oracle | Planted factors | 28,297,049 | 18,631,235 | 1.00x | 0.90x |
    | Learned | LSTM | 31,297,420 | 26,237,096 | 1.11x | 1.00x |
    | Baseline | Same weekday last week | 56,013,383 | 56,239,111 | 1.98x | 1.79x |
    | Baseline | Yesterday repeated | 73,887,789 | 73,805,996 | 2.61x | 2.36x |

*   The oracle reads the planted factors from `ground_truth.json` and estimates only a level.
    It is not something anyone could build on the day, and not a lower bound. The promotion
    day 2014-08-08 carries 57% of its squared error, and without it the LSTM is 1.41x the
    oracle.
*   The LSTM learned the weekly rhythm, since a 14-day window holds two whole weeks. It never
    sees a calendar, so the month-position effect reaches it only through the last fortnight.
*   Part 6 scores the same weights three ways:

    | Rows | RMSE |
    | :--- | ---: |
    | Training | 26,269,768 |
    | Validation | 38,070,473 (1.45x) |
    | Final test | 31,297,420 (1.19x) |

*   Run to 120 epochs instead, training RMSE would be 21,465,694 and final test 31,970,965.
    Restoring epoch 64 makes training 22% worse and the test 2.1% better.
*   Validation scores 1.22x the adjacent final test. Each window holds one promotion day, and
    without the 15 rows per window that touch it the two score 23,303,677 and 22,878,120
    (1.02x). One day moves a held-out score this much, which is what script 07 is for.

## Script 07: Rolling-origin backtest and the forecast file

A rolling-origin backtest refits each route at several cut-off dates and scores the next 30
days each time. Each fold trains on everything up to its cut-off, so the training window grows
and never contains a scored day. The cut-offs are 2014-04-30, 05-31, 06-30 and 07-31.

*   Part 1 lays out the folds (304 to 396 training days) and asserts for each that training
    ends before scoring starts.
*   Part 2 scores periodic factors on each fold, for comparison: 19,793,706, 41,675,398,
    17,100,659 and 37,436,714, a 2.44x spread. The June and August folds each hold a promotion
    day. Without them the folds are within 1.18x, so one day can decide a single score.
*   Part 3 runs all four routes over all folds:

    | Route | 04-30 | 05-31 | 06-30 | 07-31 | Mean | sd | Ordinary mean |
    | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | Periodic factors | 19,793,706 | 41,675,398 | 17,100,659 | 37,436,714 | 29,001,619 | 10,702,733 | 18,808,414 |
    | Additive model | 42,008,686 | 49,728,395 | 43,326,919 | 44,900,461 | 44,991,115 | 2,920,380 | 40,784,414 |
    | Last week repeated | 54,897,742 | 73,890,602 | 63,526,833 | 69,119,873 | 65,358,762 | 7,066,243 | 63,014,639 |
    | SARIMAX, weekly | 60,468,955 | 85,873,323 | 82,726,444 | 91,945,929 | 80,253,663 | 11,893,685 | 78,742,641 |

*   The whole ranking is the same in all four folds. The winner's 2.44x spread is wider than
    the 1.55x gap between the top two means, but the folds are paired: in every fold periodic
    factors score 0.39x to 0.84x of the additive model. Report the sd next to the mean and
    compare routes fold by fold.
*   Part 4 puts training-period error next to the backtest, for comparison. Training days
    count from day 15, because SARIMAX's first fitted value comes from its start-up state and
    is 0:

    | Route | On training days | Held out, mean | Inflation |
    | :--- | ---: | ---: | ---: |
    | Periodic factors | 23,164,815 | 29,001,619 | 1.25x |
    | SARIMAX, weekly | 62,227,177 | 80,253,663 | 1.29x |
    | Additive model | 44,568,733 | 44,991,115 | 1.01x |

*   The routes inflate by different amounts, so the gaps between them are not preserved.
    SARIMAX's training error is also one step ahead, while held out it forecasts 30 steps.
*   Part 5 backtests the redeem column too:

    | Route | Purchase, mean RMSE | Redeem, mean RMSE |
    | :--- | ---: | ---: |
    | Periodic factors | 29,001,619 (1st) | 105,946,151 (4th) |
    | SARIMAX, weekly | 80,253,663 (4th) | 75,975,758 (1st) |
    | Additive model | 44,991,115 | 91,018,547 |
    | Last week repeated | 65,358,762 | 102,612,221 |

*   The two columns rank the routes almost in reverse. Redeem was built with growth and a
    random walk, which a product of factors cannot hold. Carrying the purchase winner across
    would have forecast redeem at 1.39x the best route's error. A backtest is evidence about
    the series it ran on, so the script picks a route per column and refits it on all 427
    days.
*   The purchase forecast's weekday shape is within 0.0094 of the planted pattern on the same
    30 days. The weekday factors alone would be 0.0646 off, because each weekday falls on
    different days of the month.
*   Part 6 writes the file (no header, `YYYYMMDD` dates, whole amounts), reads it back
    and checks nine rules. The header, types and date format are decided by the write, so only
    the written file can confirm them. Three checks needed care:
    *   Dates are compared as a list, not a set, because a set accepts any permutation and the
        file is read by position.
    *   "No header row" compares the first raw cell with the first expected date.
        `not isalpha()` passes `'report_date'` too, because of the underscore.
    *   "Every line has three fields" counts commas on each raw line. `shape[1] == 3` passes
        a four-column file, because the parser turns the first field into the index.
