# Applied projects: the errors that do not raise

These eleven scripts build small, complete pieces of the systems the earlier modules covered
one layer at a time. Script 01 generates every data source with its answer written down first.
Scripts 02 to 09 work on tables: a join, a dashboard, a tool handing rows to a model, a chart
helper, control rules, a classifier label, association rules and a forecast. Scripts 10 and 11
retrieve and answer from documents. Each script runs cleanly and prints a plausible number
that is wrong in a way no exception reports, then prints the intermediate quantity that shows
it. This document explains what each script does and the ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_build_project_datasets.py` | Five synthetic sources, each with one planted property, and the ground truth later scripts are scored against |
| 02 | `02_join_grain_and_aggregation_audit.py` | A join at the wrong grain that shifts a mean, and a year total 197x the truth that also reorders the districts |
| 03 | `03_dashboard_metrics_and_cache.py` | A clamped ratio column, parts that do not sum, a band list that drops 1,276 customers, and two cache rules that disagree |
| 04 | `04_tool_return_shapes.py` | One question and one query, five return shapes, scored against a computed answer |
| 05 | `05_chart_criterion_and_index_alignment.py` | A chart rule that counts rows where the axis needs distinct values, and a column attached by index that is entirely misdated |
| 06 | `06_bollinger_and_spc_rules.py` | A rolling band reported with the numbers behind each flag, and eight Nelson rules that catch different days rather than more |
| 07 | `07_label_leakage_and_importance_views.py` | A label one column and one threshold reproduce, and four importance rankings checked against an independent column |
| 08 | `08_association_rules_sample_unit.py` | The same holdings mined under three sample units, one of which makes every lift exactly 1 |
| 09 | `09_cohort_is_not_a_time_series.py` | A cohort series whose neighbours share no one, a shuffle test, and Prophet terms with no data under them |
| 10 | `10_search_backends_and_ui.py` | Hybrid retrieval with BM25 and vectors, RRF and weighted fusion, a cutoff in tokens, and a failure traced one layer at a time |
| 11 | `11_answer_routing_and_citation.py` | Query routing to a report and an answer type, structured JSON answers, and every cited page checked |

## Shared setup

*   Script 01 writes everything the others read, so it runs first. It is seeded
    (`SEED = 20260828`) and rewrites every file identically. `data/` holds its output
    (`market.sqlite`, `customers.csv`, `staff.csv`, `staff_reviews.csv`, `district_daily.csv`,
    `facility_beds.csv`) plus the cache script 03 builds, and git ignores it. `outputs/` holds
    the three figures scripts 05 and 06 draw, and those are tracked.
*   Scripts 04 and 11 call a chat model. They read `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` or
    `OPENAI_API_KEY` from `.env`, in that order (the OpenAI key also reads `OPENAI_BASE_URL`
    and `OPENAI_MODEL`). Script 10 needs embeddings as well as answers and stays on Gemini, so
    it reads `GEMINI_API_KEY` only. All three back off and retry on a rate limit. The other
    eight scripts make no network call.
*   The dependencies are pandas, numpy, matplotlib, statsmodels, prophet, lightgbm,
    scikit-learn, rank_bm25, openai and python-dotenv, plus gradio for `--ui` in script 10.

## Script 01: Data with the answer written down first

Every later claim needs a number to be checked against, so the data is generated from an
explicit specification and the script prints the truth next to it.

| Part | Source | What is planted | What the run prints |
| :---: | :--- | :--- | :--- |
| 1 | `daily_price` in `market.sqlite` | Four instruments on a geometric random walk over two years of weekdays; CLD halted for 11 sessions from 2024-05-13; one three-day shock per instrument | CLD has 511 rows against 522 for ARB; the shocks (MRD +6.7% from 2024-02-13); the 2024 move of each instrument |
| 2 | `customers.csv` | Holdings drawn conditionally: fund at 0.22, times `WEALTH_TO_FUND_MULTIPLIER = 2.6` for wealth holders; insurance lowered among fund holders | P(fund given wealth) 0.5733, not wealth 0.2240, lift 1.6734; 10,000 rows and 16 distinct holding combinations |
| 3 | `staff.csv`, `staff_reviews.csv` | One row per employee against one per quarter worked; salary rises with years of service | 480 employees, of whom 421 hold the full 8 reviews |
| 4 | `district_daily.csv` | A daily count `new_cases` beside a running total `cumulative_cases` | The year total 459,458, and 90,522,763 from summing the running total |
| 5 | `facility_beds.csv` | A reported ratio rounded and then clamped at `REPORTED_RATIO_CAP = 99`; out-of-service beds in the total | 184 rows reading exactly 99; 2,595 rows where occupied plus free falls short of the total |

*   Part 1 draws each price as a percentage step from the one before, so a price never goes
    negative. Weekends are never generated. Part 6 prints the truth table for 2024:

| Ticker | First close | Last close | Change |
| :--- | ---: | ---: | ---: |
| ARB | 137.45 | 150.60 | 9.57% |
| CLD | 39.60 | 51.52 | 30.10% |
| MRD | 101.90 | 131.46 | 29.01% |
| SVN | 31.67 | 18.80 | -40.64% |

*   SVN at -40.64% is the largest absolute move of 2024, and script 04 asks a model for it.
*   Part 3 makes the review count uneven, but not as a gradient over tenure. The review table
    spans 8 quarters, so only the 59 most recent hires hold fewer than 8. Script 02 relies on
    that.
*   Part 5 rounds first and clamps second, so the 184 rows reading 99 are two populations:
    95 were clamped, and 89 have a true ratio between 98.5 and 99.5 and only round to 99.
    Python rounds an exact half to the even neighbour, so 98.5 reads 98 and 99.5 reads 100.
*   No later script uses the CLD halt; it only means the four series have different lengths.

## Script 02: Join grain and a total that agrees with nothing

*   Part 1 counts rows per key. The staff table has 1.0 row per `staff_id`, the review table
    7.5. Joining onto a master table keeps its grain only when the other side holds at most
    one row per key; the master side being unique is not enough.
*   Part 2 runs `staff.merge(reviews, on="staff_id", how="left")`. 480 rows become 3,608 with
    no warning. A one-to-many join is right when review-level rows are what the question
    wants; the mistake is reading employee-level answers from it.
*   Parts 3 and 4 fix it two ways, and both return 480 rows. Narrowing the reviews to 2024Q4
    answers a question about one period; aggregating to one row per employee answers one
    about the whole period.
*   Part 5 measures the damage on a quantity with a known answer. Mean base salary belongs to
    the master table, so any change means the grain changed:

| Table | Mean base salary |
| :--- | ---: |
| staff master only | 69,732.50 |
| after the naive join | 70,318.90 |
| after narrow then join | 69,732.50 |
| after aggregate then join | 69,732.50 |

*   The naive mean is weighted by review count. The 421 employees with all 8 reviews average
    70,930.40 and the other 59 average 61,184.75. Because the count is capped at 8, the join
    does not up-weight long service. It down-weights the 59 recent hires, who earn less.
    Headcount by department shows the same with no statistics: the total reads 3,608.
*   Parts 6 and 7 add up a year of district cases three ways. Only `new_cases` is a daily
    count; `cumulative_cases`, `recovered_total` and `deaths_total` are running totals.

| Method | Year total |
| :--- | ---: |
| sum of `new_cases` | 459,458 |
| sum of each district's maximum `cumulative_cases` | 459,458 |
| sum of `cumulative_cases` over all rows | 90,522,763 (197.0x) |

*   The first two agree by construction, since 01 builds the running total from the daily
    column. The third counts a January case again on every later day. Which aggregation is
    right depends on what one row means, which the dtype does not say.
*   Part 8 shows the wrong total is not just a wrong scale: 13 of 18 districts change rank.
    Summing a running total weights a district by how early its cases came. Thornbury (mean
    case day 160) overtakes Dunmore (182), and Kirkburn, the latest at day 197, drops out of
    the top 8.

## Script 03: Dashboard metrics and a cache

*   Part 1 recomputes the reported utilisation ratio from the bed counts. 99 is a legal
    percentage, so a clamp only shows when the ratio is computed another way:

```
    rows reporting exactly 99%                  184
    rows where the two ratios differ > 0.5       95
```

*   Rounding moves a ratio by at most 0.5, so the 95 rows that differ by more are the clamped
    ones, each moved down by at most one point. The other 89 only rounded to 99. Counting rows
    that read 99 would have overstated the damage by 94%. On this data the clamp is small:
    the mean is 77.13% reported against 77.17% recomputed.
*   Part 2 checks that the parts reach the total. Occupied plus free is 504,643 against
    521,400 total beds, and the difference is exactly the 16,757 out-of-service beds. A free
    beds tile computed as total minus occupied (120,484) and one read from the free column
    (103,727) answer different questions. Neither is wrong; publishing one without saying
    which question it answers is.
*   Part 3 bands customers by assets with `pd.cut` on edges 0, 100,000, 500,000 and
    1,000,000. A value outside the outermost edges becomes NaN, and `value_counts()` drops NaN
    without comment. The three bands sum to 8,724 of 10,000 customers, and all 1,276 missing
    ones sit above the top edge, the segment a wealth funnel exists to find.
*   Part 4 repairs it with an open top band and prints the observed range under each label:

| Label | Count | Min assets | Max assets |
| :--- | ---: | ---: | ---: |
| Mass | 1,010 | 10,058 | 99,927 |
| Affluent | 5,496 | 100,010 | 499,496 |
| High net worth | 2,218 | 500,048 | 999,933 |
| Ultra high net worth | 1,276 | 1,000,880 | 8,400,884 |

*   A label is a claim about a range, and the printed range makes it checkable. The code also
    passes `include_lowest`, which would keep a balance of exactly 0 in Mass; this file has
    none.
*   Part 5 computes the nine tiles and writes them to a cache together with a fingerprint of
    the source files (size and modification time). Without the fingerprint a cache can only
    answer whether it exists, not whether it is still valid. Size and time are a cheap check,
    not a content check: an edit that kept both would pass where a hash would not.
*   Part 6 reads the tiles back. The cold build took 2.9 ms from tables already in memory and
    the read 0.1 ms, but six runs gave ratios between 23x and 80x, so the ratio alone is weak.
    What matters is which side grows with the data:

| Bed rows | Cold build | Warm read |
| ---: | ---: | ---: |
| 3,000 | 2.3 ms | 0.1 ms |
| 30,000 | 7.3 ms | 0.1 ms |
| 120,000 | 25.0 ms | 0.1 ms |

*   Forty times the rows took about eleven times the build. The warm column is the one read
    timed earlier, printed on every line, because the cache holds the same nine tiles whatever
    the source size. Nothing got faster; the work moved off the request.
*   Part 7 edits one row of `facility_beds.csv`. "A cache file exists" still says fresh,
    while "the fingerprint still matches" says stale, and the mean utilisation moves from
    77.1712% in the stale cache to 77.1388% rebuilt. The edit flips the first row between
    empty and full, so it is a real change whatever the file holds, and it runs inside
    `try` and `finally`, so the source is restored and a second run prints the same numbers.

## Script 04: What a tool returns decides what a model can answer

*   Part 1 computes the answer from the database: the instrument with the largest absolute
    percentage move over 2024 is SVN at -40.64%. Part 2 runs one correct query (every 2024
    close, 1,037 rows, sorted by ticker then date) and packages the result five ways. Part 3
    asks the model the same question against each shape, at temperature 0.
*   The question removes any doubt about "moved the furthest": the largest absolute
    percentage change, where a fall of 40 percent is a larger move than a rise of 30.
*   Part 4 scores each reply against the truth table. Three runs on 2026-09-30 gave this
    table every time:

| Shape | Named | Right instrument | Claimed | Truth | Error | Characters |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `head(10)` | ARB | no | -3.80 | +9.57 | 13.37 | 455 |
| `head(5) + tail(5)` | SVN | yes | -0.63 | -40.64 | 40.01 | 455 |
| `head(5) + tail(5) + describe()` | ARB | no | -86.30 | +9.57 | 95.87 | 710 |
| first and last row per ticker | SVN | yes | -40.64 | -40.64 | 0.00 | 379 |
| endpoints with change computed | SVN | yes | -40.64 | -40.64 | 0.00 | 527 |

*   Only shapes 4 and 5 are right about the instrument and its move, and shape 4 is the
    smallest. Its rows are chosen per group rather than off the ends of a flat table.
*   The scoring asks two questions. "Right instrument" is whether the reply picked the one
    that moved most. The error is measured against the truth for whichever instrument the
    reply named. Shape 2 names SVN and is 40.01 points off, so that column alone proves
    nothing. Shape 5's 0.00 is copied, not computed: its table carries the same endpoint
    arithmetic the truth table uses.
*   Part 5 parses the two ten-row shapes and checks them against the truth table's first and
    last dates. `head(10)` reaches only ARB; `head(5) + tail(5)` reaches ARB and SVN. Neither
    holds any instrument's first and last close together, because the head is ARB's first
    week and the tail SVN's last.
*   The replies show what the model did instead. With `head(10)` it says outright that it
    has only ARB. Shape 2 takes a late-December SVN close as SVN's first close of the year,
    and shape 3 pairs ARB's first close with SVN's last and reports the gap as ARB's move.
    Neither mentions that the data was partial. The fix that reaches both ends removed the
    warning without making the number more trustworthy.
*   Which instrument shapes 1 to 3 name is not stable across runs, even at temperature 0. An
    earlier recorded run named ARB for all three, with errors of 13.37, 13.92 and 12.47, and
    shape 4 claimed -40.60. What held in every recorded run is narrower: only shapes 4 and 5
    were right about the instrument and within 0.04 points on its move.

## Script 05: A chart criterion on the wrong count, and index alignment

*   Parts 1 and 2 give a chart helper three results and two rules with the same threshold of
    20. One picks a line when there are more than 20 rows, the other when there are more than
    20 distinct x values. The row count stands in for the axis positions only while every row
    carries its own x value:

| Result | Rows | Distinct dates | Instruments | By row count | By distinct x |
| :--- | ---: | ---: | ---: | :--- | :--- |
| one instrument, one month | 21 | 21 | 1 | line | line |
| one instrument, one year | 262 | 262 | 1 | line | line |
| four instruments, ten days | 40 | 10 | 4 | line | bar |

*   Forty rows over ten dates: the row rule sees 40 and picks a line, and the same threshold
    on the ten positions picks a bar. That is the verdict of this script's own rule, not a
    claim that every ten-date result belongs in bars. Consecutive rows are different
    instruments on the same date, so the line zigzags.
*   Part 3 draws both pictures to `outputs/` with the same flat helper, and says that both
    plot all 40 rows by position, so the bar picture shows the other rule's choice, not 10
    date positions.
*   Part 4 thins each result to 10 points with `np.linspace` over row positions. On the
    interleaved result the points fall one per date, but they come from four series in the
    order ARB, ARB, ARB, CLD, CLD, CLD, MRD, MRD, MRD, SVN. The points switch instrument 3
    times, and the close drops from about 140 to about 49 where ARB hands over to CLD.
*   Part 5 attaches a 20-day moving average to a report frame. The date filter keeps 239 rows
    whose index runs 23 to 261, and the average has 220 values. The report is built from a
    list, so its index runs 0 to 238. Assignment matches the index, not the row order: the two
    sides share 216 index values and 197 of 239 cells arrive. All 197 hold another row's
    value. Report row 42, dated 2024-04-01, holds 141.8560, the average for 2024-02-28, the
    row that carried index 42 in the source frame. A total miss would leave an empty column;
    a partial overlap leaves one that looks fine.
*   Part 6 attaches the same average three ways that work (`.to_numpy()`,
    `.reset_index(drop=True)`, `.set_axis(report.index)`). Each gives 220 of 239 arrived and
    0 misplaced, checked cell by cell against the average looked up by date. The same check
    reports all 197 of Part 5's cells misplaced, so it can fail. Positional attachment is
    right here only because the report was built from the window in the same row order.
*   Part 6 also stacks two three-row frames. With mismatched indexes `concat` gives 6 rows
    and 9 empty cells; after `reset_index` on both it gives 3 rows and 0. Two three-row
    frames that stack into more than three rows never shared an index.

## Script 06: A rolling band and eight control rules

*   Part 1 builds a Bollinger band on MRD: a 20-day rolling mean as centre, plus and minus
    two rolling standard deviations. The centre moves with the series, so the band asks
    whether today is unusual against the recent past. That is what makes it usable on a
    series that trends.
*   Part 2 counts the flags. Of 522 rows, 503 have a full window behind them, and 54 of those
    fall outside the band (38 above, 16 below), 10.7% rather than 5%. The returns are not fat
    tailed: they measure as normal (excess kurtosis 0.004, lag-1 autocorrelation of absolute
    returns -0.056). The 95% figure assumes independent draws around a fixed mean, and a
    close is the next step of a random walk compared with a trailing mean and spread.
*   Part 3 prints each flag with its centre, bounds and distance in sigmas, then the two rows
    that look contradictory as date and price alone:

| Date | Side | Close | Centre | Upper | Lower | Sigmas |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 2024-11-07 | below | 126.51 | 130.42 | 134.02 | 126.82 | -2.17 |
| 2023-06-01 | above | 96.35 | 94.06 | 95.81 | 92.30 | 2.62 |

*   The dearer close is flagged low and the cheaper one high. With the centre beside them
    there is no contradiction: each was judged against its own window.
*   Part 4 standardises the series against its band and applies the eight Nelson rules. Rule
    1 is single-point at three sigma; the other seven look at runs or windows.

| Rule | Description | Days | Share |
| :--- | :--- | ---: | ---: |
| 1 | one point beyond 3 sigma | 0 | 0.0% |
| 2 | nine in a row on one side of centre | 230 | 45.7% |
| 3 | six in a row rising or falling | 48 | 9.5% |
| 4 | fourteen in a row alternating | 0 | 0.0% |
| 5 | two of three beyond 2 sigma, same side | 46 | 9.1% |
| 6 | four of five beyond 1 sigma, same side | 173 | 34.4% |
| 7 | fifteen in a row inside 1 sigma | 4 | 0.8% |
| 8 | eight in a row beyond 1 sigma, both sides | 0 | 0.0% |
| any | at least one rule | 309 | 61.4% |

*   Rules 2 and 6 were written for a process held at a fixed target. Here the centre is a
    mean that follows the series and 61% of days sit above it, so a trend alone keeps those
    counters running. The per-rule share shows that; a single flagged column would hide it.
*   Rule 8 needs its eight points on both sides of centre. Without that condition it flagged
    63 days, every one already flagged by another rule.
*   Rule 7 flags a series that is too quiet. Fifteen points inside one sigma would have a
    probability of about 0.68 to the 15th, under half a percent, if the points were
    independent, which here they are not.
*   Part 5 merges consecutive flagged days into events. The band's 54 days form 22 events
    (mean run 2.5 days, longest 6). The eight rules' 309 days form 20 events (mean 15.4,
    longest 58). A count of days says how long the series stayed unusual; a count of events
    says how often it became so.
*   Part 6 ranks the largest one-day and three-day moves, only on the 503 days the band can
    judge. The largest one-day move of the series, 2023-01-06, falls on row 5, before the
    first full window.

| List | Date | Move | Band | Rules that day | Next 3 days |
| :--- | :--- | ---: | :--- | :--- | :--- |
| one day | 2024-08-01 | 2.91% | inside | none | none |
| one day | 2024-05-02 | 2.83% | outside | none | +1: 5 |
| one day | 2024-02-13 | 2.81% | inside | none | none |
| three days | 2024-02-15 | 5.90% | outside | none | +2: 5, 6 |
| three days | 2024-05-02 | 5.22% | outside | none | +1: 5 |
| three days | 2024-12-13 | 5.08% | outside | none | +1: 5 |

*   The band catches 4 of the 6 moves, on 5 distinct days. The eight rules fire on none of
    the 5 days, and after 3 of them they fire within 3 days. Rule 1 is stricter than the band
    and fired on no day of the series; the other seven report an excursion once it has
    lasted. A rule set catches different days, and on the largest moves it answers later.
*   Rows 291 to 293 (2024-02-13 to 2024-02-15) are the shock 01 planted in MRD. Its first day
    is the one-day move the band left inside; its last is the largest three-day move, caught
    by the band that day and by rules 5 and 6 two days later.

## Script 07: A label recovered from one column, and four importance rankings

*   Part 1 builds the label the way a project does when the outcome has not happened yet:
    `future_aum` is `total_aum` times a random growth factor between 0.95 and 1.20, and the
    label is `future_aum >= 1,000,000`. The positive rate is 0.1464. The only column feeding
    the label is already a feature, which is target leakage.
*   Part 2 recovers the label with one threshold on `total_aum`:

| Check | Value |
| :--- | ---: |
| best threshold on `total_aum` | >= 942,865 |
| accuracy of that one comparison | 0.9864 |
| accuracy of always answering no | 0.8536 |
| AUC of the raw column, no model | 0.9990 |
| rows the growth factor can still decide | 0.0572 |

*   Below 1,000,000 / 1.20 no row can reach the threshold, and above 1,000,000 / 0.95 every
    row does. Only the band between them, 5.72% of the table, is left to chance.
*   Part 3 trains LightGBM on 12 features for 200 rounds: held-out AUC 0.9987 and accuracy
    0.9840. The raw column scores 0.9989 on the same 2,500 held-out rows, so the model is
    slightly below one column and no model.
*   Part 4 drops `total_aum`. AUC falls to 0.9869 (-0.0119) and accuracy to 0.9520, an error
    rate going from 1.60% to 4.80%. The small AUC drop does not mean the leak is gone: 01
    draws the balance and transaction columns from `total_aum` (correlations 0.893 for
    deposits, 0.657 for monthly transaction amount), and they stay in the table. Only a label
    from an observed outcome repairs it.
*   Parts 5 and 6 rank the features of each model four ways:

| Measure | What it counts | Measured on |
| :--- | :--- | :--- |
| split | how often the column was cut on | training |
| gain | how much those cuts improved the objective | training |
| permutation | the AUC drop when the column is shuffled, 5 repeats | held-out |
| contribution | mean absolute SHAP value from LightGBM's `pred_contrib` | held-out |

*   `age` is the control. 01 draws it on its own, so any rank it earns is noise. In Part 5
    it ranks split 3, gain 4, permutation 4 and contribution 4; in Part 6 split 3, gain 4,
    permutation 11 and contribution 7. Contribution is measured on held-out rows, but it
    follows what the model uses, not whether that helps.
*   Part 5 prints the permutation standard deviation. With `total_aum` in the model, only 3
    of 12 permutation means clear twice their sd, so ranks 2 to 12 are an order among noise.
    Without it, 9 of 11 do.

| Rank correlation | Part 5 | Part 6 |
| :--- | ---: | ---: |
| split and gain | 0.99 | 0.84 |
| split and permutation | 0.55 | 0.36 |
| gain and permutation | 0.58 | 0.64 |
| contribution and permutation | 0.69 | 0.80 |

*   In Part 6 split even picks a different first place (`monthly_txn_amount`, against
    `deposit_balance` for the other three). A continuous column offers many cut points and
    collects splits whether or not they help: `age`, with 54 values, ranks third by split in
    both models. Split and gain describe the fit, not the data.
*   The permutation measure needs a scikit-learn estimator, so the booster is wrapped in a
    class inheriting `ClassifierMixin, BaseEstimator`, the mixin first so its classifier tag
    wins. Current scikit-learn checks estimator tags before it scores anything.

## Script 08: The sample unit decides the answer

*   Support, confidence and lift are proportions, and the question is what they are
    proportions of. The script counts every combination of up to three of four products,
    with `MIN_SUPPORT = 0.05` and `MIN_CONFIDENCE = 0.30`.
*   Part 1 builds one basket per customer: 10,000 baskets, and all 16 possible combinations
    occur. Part 2 mines them; wealth to fund has support 0.1947, confidence 0.5733 and lift
    1.6734.
*   Part 3 drops duplicate baskets. 10,000 rows collapse to 16, which still hold every
    combination but no longer how many customers each stands for. Every product then has
    support exactly 0.5, every pair 0.25 and every rule lift 1.0000. With each combination
    present once, every product is in half of them and every pair in a quarter, so
    independence holds by construction whatever the customers did.
*   Part 4 groups the baskets and keeps the count as a weight: 16 rows carrying 10,000
    customers, the largest group 3,875 and the smallest 11. The rules are identical to Part
    2's, checked rule by rule on support, confidence and lift. The same check between Part 2
    and Part 3 returns False (12 rules against 24), so it can fail. Deduplicating was not the
    problem; discarding the count was.
*   Part 5 puts the three units side by side. Lift ranges over 0.6716 per customer and 0.0000
    per combination.
*   Part 6 checks the planted pair against 01's probabilities:

| Quantity | Measured | Drawn at |
| :--- | ---: | ---: |
| P(fund given wealth) | 0.5733 | 0.5720 |
| P(fund given no wealth) | 0.2240 | 0.2200 |
| lift of wealth to fund, direct count | 1.6734 | |
| lift of the mined rule | 1.6734 | |
| lift in the deduplicated table | 1.0000 | |

*   01 sets probabilities, not a lift, so the lift is measured on the draw. The highest mined
    lift, deposit and fund to wealth at 1.6748, is only 0.0013 above the planted pair, because
    01 draws deposit on its own. 01 also lowers insurance among fund holders (lift 0.7887),
    but that pair's support is 0.0438, below the minimum, so it never reaches the rule table.

## Script 09: A cohort is not a time series

*   Part 1 groups customers by the month they opened their account: 72 points from 2019-01
    to 2024-12, one mean per month, no gaps. That is the shape a forecasting library accepts.
*   Part 2 counts the customers two neighbouring months share: 0 of 71 pairs share any.
    Each customer has one open date, so this is 0 by construction. Part 3 loads MRD's daily
    closes for contrast, 522 points in which every neighbour is the same instrument because
    the query filters on it.
*   A line between two points asserts that something moved from one value to the other, and
    that needs both points to be about the same thing. The zero shows these months are
    different customers. It does not show that every aggregate with changing members lacks
    temporal structure.
*   Part 4 runs a shuffle test, a form of permutation test that needs no identity column: fit
    ARIMA(1,1,1) to the values as given and to 30 random orders, and compare the errors. The
    order is fixed, so a shuffled fit differs only in the data.

| Series | Error as given | Shuffled mean | Ratio | Shuffles as good |
| :--- | ---: | ---: | ---: | ---: |
| cohort by join month | 55,116.27 | 54,864.61 | 1.00 | 16 of 30 |
| MRD daily close | 0.83 | 9.35 | 11.29 | 0 of 30 |

*   01 builds each close on the one before, so order is part of the data. 01 draws each
    customer's assets without regard to the join month, so there is no order to find. This is
    a statement about ARIMA(1,1,1) on this series, not a proof that no model could find one.
*   Part 5 fits Prophet to the daily series. There are no weekend rows in training; the only
    four come from the 14 forecast days after the data. The weekly term is still defined at
    all seven days:

| Day | Training rows | Weekly term |
| :--- | ---: | ---: |
| Mon | 105 | 1.3361 |
| Tue | 105 | 1.3462 |
| Wed | 104 | 1.2234 |
| Thu | 104 | 1.3154 |
| Fri | 104 | 1.3871 |
| Sat | 0 | -3.3041 |
| Sun | 0 | -3.3041 |

*   The seven values sum to 0.0000, so Saturday and Sunday take what the weekdays leave: the
    weekdays add up to 6.6082 and the weekend to -6.6082. No weekend row stands behind
    -3.3041. The weekdays are no better. 01 plants no weekly pattern, so their spread of
    0.1637 is noise.
*   Part 6 fits the yearly term on the first year and on both. One year (260 rows, 361 days)
    gives a curve with a range of 7.92; two years (522 rows, 729 days) give 11.27, and the two
    curves correlate at +0.4937. 01 draws MRD as a random walk with no yearly pattern, so both
    curves describe noise and trend, and more data made the invented season wider.
*   Prophet warns three times that yearly seasonality is enabled with less than 730 days of
    history: once for Part 5 and once for each fit here. The two-year span is 729 days, so by
    the library's own rule even the longer fit is short. The span is not in the plot; it has
    to be printed.

## Script 10: Hybrid retrieval, fusion and a limit in the wrong unit

*   Part 1 chunks eight short policy documents written into the script into windows of 60
    words overlapping by 15: 15 chunks of 101 to 437 characters, about 1,011 estimated tokens
    in all. Every question names its source document, so retrieval is scored, not judged. The
    overlap keeps a sentence at a boundary whole only if at most 15 of its words fall before
    the boundary; 3 of the 35 sentences are whole in no window, none of them one a question
    asks about.
*   Part 2 builds a BM25 keyword index and a vector index (`gemini-embedding-001` at 768
    dimensions) over the same chunks, so the only variable is the scoring. Truncated
    embeddings are not unit length, so cosine is taken explicitly. It fuses the two rankings
    two ways. Reciprocal rank fusion (RRF) adds 1 / (60 + rank) from each backend. The
    weighted fusion rescales each backend's scores to 0 to 1 per question and adds them with
    a keyword weight of 0.5. Raw scores cannot be added: BM25 has no upper bound, and the
    cosine scores here sit between about 0.46 and 0.75.
*   Part 3 sends the five questions through all four backends, top 3:

| Question | Kind | Keyword | Vector | RRF | Weighted |
| :--- | :--- | :--- | :--- | :--- | :--- |
| What does Clause 7.3 cover? | exact term | 1 | 1 | 1 | 1 |
| What is the aggregate limit under Clause 4.1? | exact term | 1 | 2 | 2 | 2 |
| Someone stole my suitcase at the airport. Am I covered? | paraphrase | missed | 1 | missed | missed |
| I got sick on holiday and had to be flown home. Who pays? | paraphrase | 3 | 1 | 2 | 2 |
| How long does the insurer have to decide on my claim? | paraphrase | 2 | 1 | 1 | 1 |

*   The vector backend finds everything the keyword backend finds, so fusion has no hit to
    add; it can only pull one down. On the suitcase question the keyword backend ranks the
    right chunk 12th of 15, which drags it out of both fused top 3s. Fusion adds a hit only
    when each backend finds what the other misses.
*   The weighted result depends on the weight. A trial with keyword weights of 0.3, 0.5 and
    0.7 kept the suitcase question only at 0.3, which lost the first place on Clause 4.1.
    Choosing the weight on these five questions would be tuning on the test set. Three trial
    corpora with near-identical schedules that differ only in a reference code also failed to
    give the keyword side a win, so the script keeps the original corpus.
*   Part 4 cuts the vector backend's top 8 for each question to three chunks and to a budget
    of 220 estimated tokens (characters divided by 4, not a tokenizer count):

| Question | Fixed count: chunks | Tokens | Budget: chunks | Tokens |
| :--- | ---: | ---: | ---: | ---: |
| What does Clause 7.3 cover? | 3 | 215 | 3 | 215 |
| What is the aggregate limit under Clause 4.1? | 3 | 220 | 3 | 220 |
| Someone stole my suitcase at the airport. Am I covered? | 3 | 186 | 3 | 186 |
| I got sick on holiday and had to be flown home. Who pays? | 3 | 243 | 2 | 209 |
| How long does the insurer have to decide on my claim? | 3 | 182 | 4 | 213 |

*   The budget never passes 220 and varies between two and four chunks; the fixed count
    reaches 243. Three chunks cost whatever those three hold, so a count does not track the
    unit a context window limits. The budget loop stops before the first chunk that would
    exceed it but always keeps one, so a first chunk larger than the budget is kept whole.
    No chunk here is that large.
*   Part 5 answers each question from the keyword and the vector backend, using each one's
    top 3 from Part 3 trimmed by the same budget, so the context never holds more than 3
    chunks. It prints one extra line when a backend retrieved the expected source and still
    declined: the vector backend on the suitcase question, the same in three separate calls.
    The clause covers checked baggage in a carrier's custody, and the question does not say
    where the suitcase was, so the refusal may be defensible. It is a decision at the answer
    layer that no change to retrieval reaches.
*   Part 6 takes the first question a backend missed and goes down one layer at a time:

```
    Layer 1, the answer      : not in the retrieved context.
    Layer 2, the retrieval   : expected baggage-loss, got ['property-allrisks', 'medical-abroad', 'medical-abroad']
    Layer 3, the raw scoring : the expected document's best chunk sits at rank 12 of 15
    Layer 4, the tokens      : the question has 'covered', the chunk has 'covered.'
                               without sentence-final dots the chunk ranks 3 of 15, inside the top 3
```

| Layer | If the failure is here | The repair |
| :--- | :--- | :--- |
| The answer | The right chunks were retrieved and the reply is still wrong | prompt, schema, model |
| The retrieval | The right document exists but did not make the cutoff | scoring, k, budget, a second backend |
| The raw scoring | The document's best chunk ranks far down, or is not in the index at all | the scoring; if it is missing, ingestion, chunking, parsing |
| The tokens | Query and chunk share a word, but not as the same token | the tokenizer |

*   Three layers alone would stop at rank 12 and blame the scoring. The tokenizer
    `[a-z0-9.]+` keeps the dot so clause numbers such as `7.3` stay whole, and so every
    sentence-final word keeps its dot too: 39 of the 650 tokens end in one. The clause says
    `not covered.` and the question `Am I covered?`, and the two never match. With the dots
    stripped, the same BM25 ranks the chunk third, and the keyword backend and both fusions
    find all five questions. The script keeps the tokenizer because this is the failure Part
    6 exists to find.
*   `--ui` serves the same backends behind a small Gradio page with a backend selector, a
    cutoff selector and panes for the chunks, the context size and the answer. It calls the
    same `search`, `by_token_budget` and `answer` functions, so it is a second front end, not
    a second implementation.

## Script 11: Query routing, structured answers and checked citations

*   Part 1 holds three company reports of four pages each, numbered sparsely and never
    shared (Alderway 14, 29, 47, 63; Brightlane 11, 38, 52, 71; Coldharbour 9, 26, 44, 58).
    In a corpus numbered 1, 2, 3 a guessed page would very likely exist and pass as read;
    here most guesses land on a page that was never supplied.
*   Part 2 routes each question twice, to a report and to an answer type (number, boolean,
    name, names, string), and scores the two separately: 6 of 6 each. The type decides which
    formatting rule the answering call receives. The answer is produced from whatever report
    the router chose, so a routing error reaches the answer instead of being silently
    corrected, and a reply naming no known report opens no pages.
*   Part 3 answers under a four-field JSON schema: reasoning, answer, references and
    confidence, in that order, so the model writes the answer after its working. The
    DeepSeek replies keep that order. If the pages lack the answer, the schema asks for
    `N/A`, no references and confidence 0.
*   Part 3 then checks every cited page against the pages the model was given. A page it was
    never given cannot have been read, whatever the answer says. The check needs no answer
    key, and it checks provenance, not support: a supplied page can still fail to say what
    the answer claims.
*   One question asks for a dividend the reports never mention. Its type is `number`, so it
    separates routing to the right type from finding an answer. The model answered `N/A` with
    no references and confidence 0. The scoring treats `N/A` as conforming, since it is what
    the prompt asks for, but an `N/A` with references still fails. The checker is strict only
    for `number` (a bare number) and `boolean` (exactly yes or no); for the other types it
    checks that the answer is non-empty.
*   Part 4 gives each check a failure, since Parts 2 and 3 produced none:

```
    Q: What was Alderway Foods' revenue in 2024?
       answered from the pages of Brightlane Logistics
       answer 'N/A', answer WRONG, invented []

    a reply citing [14, 11, 3] for Alderway Foods: valid [14], invented [11, 3]
```

*   From the wrong pages the model declines and cites nothing, so only the answer key catches
    the wrong route. The hand-made reply cites one supplied page, one of another report and
    one of no report; the citation check drops the last two with no answer key.
*   Part 5 splits each comparison into one sub-question per report, since a comparison has no
    single report to route to. The revenue question gets 812.4, 1204.7 and 640.1 with one
    valid page each, and the combined reply names Brightlane Logistics, citing page 11. The
    combined reply's references are checked against the pages the sub-answers kept, the only
    pages it saw. The headcount question names two companies, but all three are asked, so
    Brightlane's 9,940 (the largest) reaches the combining call too; it answered Alderway
    Foods in all four runs.
*   Part 6 prints six scores, because each points to a different fix: a routing error is
    repaired in the router, a schema error in the prompt, and an invented citation is caught
    without knowing whether the answer was right.

| Score | Result |
| :--- | :--- |
| report routing | 6 of 6 |
| answer type routing | 6 of 6 |
| schema conformance | 6 of 6 |
| answers correct | 6 of 6 |
| comparisons correct | 2 of 2 |
| page citations | 5 made, 0 invented |

*   Temperature 0 does not guarantee identical replies. In four runs on 2026-09-30 every count
    stayed the same; only the wording of the string answer changed.
