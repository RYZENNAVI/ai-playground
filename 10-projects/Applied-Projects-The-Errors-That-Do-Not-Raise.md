# Applied Projects: The Errors That Do Not Raise

Every other module in this repository is about making something work. This one is about
what happens after it works.

Eleven scripts here build small, complete pieces of the kind of system the earlier modules
each covered one layer of: a table joined and aggregated for a dashboard, a tool handing
query results to a model, a classifier with a label somebody had to invent, an association
rule, a forecast, two retrieval backends, an answering pipeline that cites its sources.
None of them introduce a technique. Each one runs correctly, prints a plausible number,
and is wrong in a way that raises no exception.

The module's single proposition is that this class of error has one countermeasure, and it
is not code review:

> **Print an intermediate quantity and put it next to a value computed some other way.**

Everything below is an instance of that, and every number quoted is one the scripts print.

---

## 1. What this module is about

### 1.1 The shape of the failures

Nine of the eleven scripts end up demonstrating the same shape from a different angle:

| Where | What it looks like | What it is |
| :--- | :--- | :--- |
| A join | A valid table with more rows | An average silently reweighted |
| An aggregation | A total of the right magnitude | The wrong column added up, **197x** the truth |
| A dashboard band | A clean funnel | **1,276 customers** in no band at all |
| A ratio column | Percentages between 0 and 99 | A cap, hiding everything above it |
| A tool's return value | A ten-row preview | One instrument's January and another's December |
| A computed column | Mostly populated | **197 of 197** filled cells carrying another row's value |
| A label | AUC **0.9987** | A threshold on one column, reproduced |
| An association rule | Support 0.5, lift 1.0 | Arithmetic on 16 rows, not 10,000 customers |
| A forecast | A smooth curve | A series whose neighbouring points share **0.0000** of their population |

In none of these cases does the program stop. In none of them is the output malformed.
The only thing that separates the right answer from the wrong one is whether somebody
printed the count, the overlap, the distribution, or the second estimate.

### 1.2 Why the data is synthetic

`01_build_project_datasets.py` generates all five sources from an explicit specification and
prints the specification alongside them. That is not a convenience — it is what makes the
rest of the module measurable:

- The yearly move of every instrument is computed and printed, so a model's answer to the
  same question has a number to be scored against.
- The relationship the customer table was drawn with (`WEALTH_TO_FUND_MULTIPLIER = 2.6`)
  produces a known lift of **1.6734**, so an association rule either recovers it or does not.
- The district table carries both a daily count and a running total, and the correct year
  total is printed next to the one that adding the wrong column gives.
- The bed table's reported ratio is clamped at `REPORTED_RATIO_CAP = 99` on purpose, and the
  number of rows that hit the clamp is printed.

The generator is seeded (`SEED = 20260828`) and idempotent: rerunning it reproduces every
file exactly. Nothing in `data/` is source material — it is all output of script 01.

### 1.3 What is deliberately not here

Two things a reader might expect are absent, and both absences are decisions rather than
omissions.

**No external search service.** Script 10 compares two retrieval backends. A production
search engine would be the natural home for both, and it is what a system at real scale
would use — one engine that serves inverted-index scoring and vector similarity from the
same store, with a documented ceiling around 2.1 billion documents per index. It is also
several hundred megabytes and a resident JVM. The comparison the script is making is
between *what the two scoring methods retrieve*, and that comparison does not need the
service: both indexes are built in process, over the same fifteen chunks, so the only
variable is the scoring. What the script gives up is throughput and persistence; what it
keeps is the measurement.

**No agent framework.** Scripts 04 and 11 call a chat completion endpoint directly. Agent
loops, tool registration and protocol-level tool servers are the subject of module 04 and
are not restated here. What survives into this module is the one thing those modules do not
measure: what the tool *returns*, and what the model can therefore answer.

---

## 2. Data with the answer written down first

`01_build_project_datasets.py` writes five files and prints the ground truth behind each.

### 2.1 Prices with two planted features

Four invented instruments walk forward one weekday at a time under a multiplicative random
walk, so a price never goes negative and a percentage move means the same thing at any level:

```python
INSTRUMENTS = {
    "ARB": {"name": "Arbor Technologies", "start": 142.0, "drift": 0.00042, "vol": 0.0165},
    "CLD": {"name": "Calder Energy",      "start": 58.5,  "drift": -0.00011, "vol": 0.0231},
    "MRD": {"name": "Meridian Foods",     "start": 91.2,  "drift": 0.00018, "vol": 0.0104},
    "SVN": {"name": "Severn Logistics",   "start": 27.4,  "drift": 0.00035, "vol": 0.0192},
}
```

Two properties are planted rather than left to chance:

**A trading halt.** `HALT_TICKER = "CLD"` loses `HALT_DAYS = 11` sessions from
`HALT_START = date(2024, 5, 13)`. The run prints the consequence:

```
CLD is missing 11 trading days starting 2024-05-13: 511 rows against 522 for ARB.
```

A gap is what makes "row 1 to row N" and "first date to last date" stop being the same
question — which matters the moment a downstream tool takes the head and tail of a result.

**A shock per instrument**, applied as extra return across `SHOCK_LENGTH = 3` consecutive
days rather than as a single spike. A one-day spike leaves a rolling mean almost untouched
and would flag nothing; a three-day run does not, which is the contrast script 06 measures.

Weekends are never generated. That fact — zero Saturday and Sunday rows — is printed here
and becomes the whole point of section 13.2.

### 2.2 The truth table every later script is scored against

```
ticker  year   first date   last date        first     last   change %
ARB     2024   2024-01-01   2024-12-31      137.45   150.60      9.57%
CLD     2024   2024-01-01   2024-12-31       39.60    51.52     30.10%
MRD     2024   2024-01-01   2024-12-31      101.90   131.46     29.01%
SVN     2024   2024-01-01   2024-12-31       31.67    18.80    -40.64%
```

`SVN` at **−40.64%** is the largest absolute move of 2024. Script 04 asks a model exactly
this question and scores its five answers against this table.

### 2.3 Customers, with one relationship planted

Holdings are drawn conditionally, and that conditioning is the entire signal an association
rule can later recover:

```python
HOLDING_BASE = {"deposit": 0.93, "wealth": 0.34, "fund": 0.22, "insurance": 0.17}
WEALTH_TO_FUND_MULTIPLIER = 2.6
FUND_TO_INSURANCE_MULTIPLIER = 0.72
```

The run prints what that produces:

```
    P(fund | wealth)     0.5733
    P(fund | not wealth) 0.2240
    lift(wealth -> fund) 1.6734   <- the number an association rule should recover

    rows in the table                 10000
    distinct holding combinations     16
    Those two numbers are what an analysis of this table has to choose between.
```

The last two lines are placed there on purpose. `10000` and `16` are both true descriptions
of the same table, and section 11 is about which one a mining routine ends up counting.

### 2.4 Two tables built to be joined wrongly

`staff.csv` holds one row per employee; `staff_reviews.csv` holds one row per employee per
quarter *worked*. The count is deliberately uneven — someone hired partway through has fewer
reviews — and salary rises with years of service:

```python
"base_salary": np.round(
    rng.normal(58_000 + 1_650 * years_of_service, 8_500, size=STAFF_ROWS), -2
),
```

Those two facts together are what make a join at the wrong grain *shift an average* rather
than merely duplicate rows. Without the correlation, a uniform duplication would leave every
mean untouched and the demonstration would prove nothing.

### 2.5 A daily count and a running total in the same table

The district table carries `new_cases` and `cumulative_cases` side by side, because that is
what an upstream reporting system usually hands over. Only one of them can be added across
rows. Nothing in the column names says which:

```
    sum of new_cases                             459,458
    sum of per-district max(cumulative)          459,458
    sum of cumulative_cases over all rows     90,522,763   (197.0x the truth)
```

Two independently written columns agreeing is what makes either of them usable. The third
figure agrees with nothing.

### 2.6 A clamped ratio and parts that do not sum

```python
"reported_utilization_pct": min(REPORTED_RATIO_CAP, round(true_ratio)),
```

and beds that are in the total but available to nobody:

```
    rows                                     3,000
    rows reading exactly 99%                   184
    rows where occupied + free < total       2,595
```

---

## 3. Join grain, and a total that agrees with nothing

`02_join_grain_and_aggregation_audit.py`

### 3.1 Grain is a count, not a column name

A join is safe when at least one side holds exactly one row per key. Reading the column
names does not tell you that; counting does:

```
    staff master              480 rows     480 distinct staff_id    1.0 rows per key
    quarterly reviews       3,608 rows     480 distinct staff_id    7.5 rows per key
```

### 3.2 The join that looks correct

```python
def naive_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    return staff.merge(reviews, on="staff_id", how="left")
```

Nothing about this call is malformed. Pandas does exactly what it was asked: it pairs every
left row with every matching right row.

```
    480 master rows joined to reviews -> 3,608 rows (7.5x)
    No warning was raised. The result is a valid table, of the wrong thing.
```

### 3.3 What breaks, measured on a quantity with a known answer

Average base salary is a property of the master table. It cannot depend on how many reviews
someone happened to receive:

```
    mean base_salary, computed on each table:
        staff master only            69,732.50
        after the naive join         70,318.90
        after narrow-then-join       69,732.50
        after aggregate-then-join    69,732.50
```

**+586.40**, and the sign is not an accident: longer-serving people have more reviews and
earn more, so the join weights the higher salaries up.

Headcount per department shows the same thing without any statistics at all — every
department is inflated, and the total reads 3,608 where the answer is 480.

### 3.4 Two fixes, and why the order matters

```python
def narrow_then_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
def aggregate_then_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
```

Both return 480 rows. The choice between them is the question being asked — one period, or
the whole period — not a matter of taste. What is not optional is that the narrowing happens
*before* the join: filtering afterwards means the wrong-grain table already existed, and
anything written out from it is wrong.

### 3.5 The same total, three ways

```
    sum of new_cases                                   459,458
    sum of per-district max(cumulative_cases)          459,458
    sum of cumulative_cases across all rows         90,522,763    197.0x the truth
```

`sum` and `max` are both valid; both run; both return a number of a plausible magnitude.
Which one is correct depends on **what one row means**, and that information lives in the
column's semantics, not in its dtype.

The reconciliation is the check that makes the first two trustworthy:

```
    districts where the daily column and the running total disagree: 0
```

### 3.6 A wrong total that also reorders the answer

A wrong total that only changed the scale would still rank the districts the same way, and a
chart built on it would still point at the right places. This one does not:

```
        #   by sum(new_cases)        cases    by sum(cumulative)   true rank
        1   Dunmore                 45,017    Thornbury                    2
        2   Thornbury               41,939    Clifton Vale                 3
        3   Clifton Vale            40,660    Dunmore                      1
        ...
    13 of 18 districts sit at a different rank under the two totals.
```

Adding a running total across rows weights a district by *how early its cases arrived*. An
early outbreak outranks a larger late one, and the ranking is what a reader acts on.

---

## 4. A ratio the source already reports, and a band that drops people

`03_dashboard_metrics_and_cache.py`

### 4.1 Recomputing a column that was handed to you

A dashboard that reads `reported_utilization_pct` straight through has no way to notice that
the upstream system clamps it. The clamp is invisible in isolation — 99 is a legal
percentage. It shows up only when the ratio is recomputed from the two columns it was
supposedly derived from:

```
    rows                                      3,000
    rows reporting exactly 99%                  184
    rows where the two ratios differ > 0.5       95

    Among the rows reading 99%, the recomputed ratio runs from 98.5% to 100.0%.
```

The aggregate barely moves — 77.13% reported against 77.17% recomputed — which is exactly
why the aggregate is the wrong place to look. **Every facility past the cap lands on the
same value, and the busiest ones stop being distinguishable from each other.**

### 4.2 Parts that do not reach the total

```
    total beds                              521,400
    occupied + free                         504,643
    difference                               16,757   = out_of_service_beds (16,757)
```

This is not a defect in the data; it is a fact about it that needs stating. But a tile
reading "free beds" and a tile computed as "total minus occupied" answer different questions
and differ by exactly that count:

```
    'free' tile computed as total - occupied   120,484
    'free' tile read from the free column      103,727
    A dashboard showing the first number overstates availability by 16,757 beds.
```

Neither figure is wrong. What is wrong is publishing one without saying which question it
answers.

### 4.3 A band list that reads as if it covers everyone

```python
AUM_BAND_EDGES = [0, 100_000, 500_000, 1_000_000]
AUM_BAND_LABELS = ["Mass", "Affluent", "High net worth"]
```

`pd.cut` returns NaN for a value outside the outermost edges, and `value_counts()` drops NaN
without comment:

```
    Mass                (0 , 100,000]                   1,010
    Affluent            (100,000 , 500,000]             5,496
    High net worth      (500,000 , 1,000,000]           2,218
    sum of the bands                                    8,724
    customers in the file                              10,000

    customers in no band at all                  1,276
        above the top edge                       1,276
```

The funnel adds up to less than the customer base, and **the shortfall is precisely the
segment a wealth funnel exists to find.**

The repair is an open top edge plus `include_lowest`, and then a check that the labels
describe the ranges that actually landed under them:

```
    label                       count      min assets      max assets
    Mass                        1,010          10,058          99,927
    Affluent                    5,496         100,010         499,496
    High net worth              2,218         500,048         999,933
    Ultra high net worth        1,276       1,000,880       8,400,884
    total                      10,000
```

A label is a claim about a range. Printing the observed range next to the name is what turns
that claim into something checkable.

**The general rule:** before any chart that groups by a category, print
`df[col].value_counts()` and assert that the group sizes sum to the row count.

---

## 5. Moving the work off the request path

`03_dashboard_metrics_and_cache.py`, second half

### 5.1 A cache that records what it was built from

```python
def source_fingerprint(paths: list) -> dict:
    return {
        str(path.name): {"size": path.stat().st_size, "mtime": path.stat().st_mtime}
        for path in paths
    }
```

Both the tiles and the fingerprint are written together. **A cache that stores results
without recording what produced them can only answer "is there a cache", never "is it still
valid".**

### 5.2 The ratio on its own is a weak argument

```
    compute all tiles from the CSV files              3.7 ms
    load tiles from the cache                         0.1 ms
    the cache answers the request 31x faster
    Nothing got faster. The work moved off the request and onto a build step.
```

At three thousand rows the cold build is already cheap, so 31x proves very little. What
matters is which side grows with the data, and the script measures that directly:

```
           3,000 bed rows -> cold build      3.4 ms   warm read    0.1 ms
          30,000 bed rows -> cold build     11.0 ms   warm read    0.1 ms
         120,000 bed rows -> cold build     34.3 ms   warm read    0.1 ms
```

The cold column tracks the row count; the warm column does not move, because reading a dict
of finished numbers does not depend on the source.

### 5.3 Two rules that disagree the moment the source changes

```
    'a cache file exists' says fresh:        True
    'the fingerprint still matches' says:    True

    One row edited and written back to facility_beds.csv.
    'a cache file exists' still says fresh:  True
    'the fingerprint still matches' says:    False

    mean utilization, stale cache  77.1712%
    mean utilization, rebuilt      77.1388%
```

A service on the first rule would have served the stale figure with no error and no warning,
because the file was there the whole time. The edit the script makes flips the first row
between empty and full precisely so that it is a real change whatever the file currently
holds — an edit that happened to be a no-op would leave the two figures identical and prove
nothing. The source is restored and the cache rebuilt before the script exits, so a second
run reports the same numbers.

---

## 6. What a tool returns decides what a model can answer

`04_tool_return_shapes.py`

This is the only script in the module where a language model is the subject rather than a
participant. One question, one query, five ways of packaging the result.

### 6.1 The setup

The query is correct SQL. It selects exactly the rows the question is about:

```sql
SELECT ticker, trade_date, close
FROM daily_price
WHERE trade_date LIKE '2024%'
ORDER BY ticker, trade_date
```

1,037 rows across four instruments, sorted by ticker then date. The question is phrased to
remove any ambiguity about what "moved the furthest" means:

> Of the instruments in this data, which one moved the furthest over 2024 in percentage
> terms, measured as the largest absolute percentage change? A fall counts as a move, so a
> drop of 40 percent is a larger move than a rise of 30.

### 6.2 Five shapes

| Shape | What it is |
| :--- | :--- |
| `head(10)` | The preview a result viewer usually shows |
| `head(5) + tail(5)` | The usual fix for the shape above |
| `head(5) + tail(5) + describe()` | Summary statistics added on top |
| first and last row per ticker | Rows chosen **per group** |
| endpoints with change computed | The arithmetic moved into the tool |

### 6.3 The result

```
    shape                               ticker  named right   claimed    truth    error
    head(10)                               ARB           no     -3.80    +9.57    13.37
    head(5) + tail(5)                      ARB           no     -4.35    +9.57    13.92
    head(5) + tail(5) + describe()         ARB           no     -2.90    +9.57    12.47
    first and last row per ticker          SVN          yes    -40.60   -40.64     0.04
    endpoints with change computed         SVN          yes    -40.64   -40.64     0.00
```

The ladder is sharp and it is not about size:

```
    shape                               characters  named right
    head(10)                                   455           no
    head(5) + tail(5)                          455           no
    head(5) + tail(5) + describe()             710           no
    first and last row per ticker              379          yes
    endpoints with change computed             527          yes
```

**The shape that answers the question is the smallest one.** What changed is that its rows
were chosen per group rather than off the ends of a flat table.

### 6.4 The middle shape is the interesting one

`head(5) + tail(5)` looks like the strict improvement it is normally taken for: the digest
now reaches both ends of the result. But the result is sorted by ticker first:

```
    head(10) covers tickers       : ['ARB']
    head(5)+tail(5) covers tickers: ['ARB', 'SVN']
```

Both are ten rows. The second reaches two instruments and neither of them completely — the
head is one instrument's January and the tail is a different instrument's December, and
nothing in the output says so.

This is why the fix is worse than what it replaced. With `head(10)`, the reply talks about
January dates and is **visibly** working from partial data. With `head(5) + tail(5)`, the
reply spans the full year and reads as reasonable while being built from two different
instruments.

> **A repair that moves an error from visible to invisible is not an improvement.**
> Judging a fix by whether the output looks right is what lets this through; judging it by
> whether the *input* was right does not.

### 6.5 Two notes on reading these numbers

**The reply is non-deterministic at temperature 0.** Across runs, the first three shapes
have produced errors between 12 and 96 percentage points, and the specific wrong number
changes. What has been stable across every run is the ladder itself: shapes 1–3 name the
wrong instrument, shapes 4–5 name the right one. Quote the ladder; do not quote a specific
error from shape 3 as if it were a constant.

**The scoring separates two questions.** `named right` asks whether the reply picked the
instrument that actually moved most; `error` is measured against the truth for whichever
instrument the reply *named*. A reply can therefore be numerically precise about the wrong
subject, which is what shape 4's `0.04` and shapes 1–3's double-digit errors are
distinguishing.

---

## 7. Choosing a chart from the wrong count

`05_chart_criterion_and_index_alignment.py`

### 7.1 Two rules, one threshold

```python
def by_row_count(frame: pd.DataFrame) -> str:
    return "line" if len(frame) > ROW_THRESHOLD else "bar"

def by_distinct_x(frame: pd.DataFrame, x_column: str) -> str:
    return "line" if frame[x_column].nunique() > ROW_THRESHOLD else "bar"
```

The row rule is right about the thing it was tested on: a long single series should not be
drawn as bars. It reads the row count as a stand-in for how many positions the axis needs,
which is the same number **only while every row carries a distinct x value.**

```
    result                           rows   distinct dates  instruments
    one instrument, one month          21               21            1
    one instrument, one year          262              262            1
    four instruments, ten days         40               10            4
```

```
    result                           by row count   by distinct x   agree
    one instrument, one month                line            line     yes
    one instrument, one year                 line            line     yes
    four instruments, ten days               line             bar      NO
```

Forty rows, ten dates. The row rule sees `40 > 20` and picks a line; the axis needs ten
positions. Both pictures are drawn to `outputs/` so the difference is visible rather than
asserted, and the first rows of the frame show why the line zigzags:

```
trade_date ticker  close
2024-03-01    ARB 139.15
2024-03-01    CLD  47.75
2024-03-01    MRD 106.21
2024-03-01    SVN  29.67
2024-03-04    ARB 139.26
```

Consecutive positions are different instruments on the same date, not one instrument over
time.

### 7.2 Thinning has the same defect

`np.linspace` over row positions is even over time only for one series:

```
    one instrument, one year         262 rows -> 10 points, covering 1 instrument(s) and 10 date(s)
    four instruments, ten days        40 rows -> 10 points, covering 4 instrument(s) and 10 date(s)
```

On the interleaved result it walks across instruments while appearing to sample evenly.

> **When a parameter's criterion is a row count, ask what one row represents.** Once that
> meaning shifts with the query, the row count stops being a stable criterion.

### 7.3 A column that arrives mostly populated and entirely misdated

Every pandas operation preserves the index it was given. That is the behaviour that makes
alignment work, and it is also the behaviour that misfiles a column when the two sides were
never meant to line up.

```
    rows selected for the window         239
    index of that selection runs      23 to 261
    values in the moving average         220

    new frame built from a list, index runs 0 to 238
    index values the two sides share     216
    values that arrived in the column    197 of 239
```

The partial overlap is the dangerous case. A total miss leaves an obviously empty column; a
partial one leaves a column that looks fine:

```
    cells holding a value                       197
    cells whose value belongs to another row    197

    Take report row 42. It is labelled 2024-04-01 and holds 141.8560,
    which is the average computed for 2024-02-28 — the row that carried
    index 42 in the frame the average came from.
```

**Every populated cell holds another date's value.** No exception, no warning.

Three attachments that work, all returning `220 of 239`:

```python
variants = {
    ".to_numpy()": moving_average.to_numpy(),
    ".reset_index(drop=True)": moving_average.reset_index(drop=True),
    ".set_axis(report.index)": moving_average.set_axis(report.index),
}
```

And the tell for the same problem in a concatenation:

```
    concat of a slice and a reindexed slice -> 6 rows, 9 empty cells
    same concat after reset_index on both   -> 3 rows, 0 empty cells
```

Two three-row frames that stack into more than three rows never shared an index in the first
place. **The row count is the check.**

---

## 8. Control limits, and a rule set that is not a larger net

`06_bollinger_and_spc_rules.py`

### 8.1 A band that moves with the series

```python
frame["centre"] = frame["close"].rolling(WINDOW).mean()
frame["spread"] = frame["close"].rolling(WINDOW).std()
frame["upper"]  = frame["centre"] + BAND_SIGMA * frame["spread"]
frame["lower"]  = frame["centre"] - BAND_SIGMA * frame["spread"]
frame["sigmas"] = (frame["close"] - frame["centre"]) / frame["spread"]
```

The centre moves with the series, so the band asks whether today is unusual *against the
recent past* rather than against the whole history. That is what makes it usable on a series
that trends: a price can be at a two-year high and still be ordinary relative to last month.

```
    rows                                 522
    rows with a full window behind them  503
    flagged days     54   above 38   below 16
    that is 10.7% of the days the band could judge
```

Ten percent, not five. Two sigma covers 95% only under a normal distribution, and returns
are not normal — the tails are heavier and the excursions cluster.

### 8.2 A flag reported without its numbers is unreadable

```
    The dearest day flagged as below the band closed at 126.51.
    The cheapest day flagged as above it closed at 96.35.
```

Reported as date and price alone, those two rows contradict each other. With the centre line
beside them they do not — each was judged against its own recent window, and the windows were
at different levels. The script therefore prints all four numbers per flag:

```
    date         side       close   centre    upper    lower  sigmas
    2023-02-27   above      97.67    96.24    97.63    94.85    2.06
    2023-03-29   below      97.48   100.58   103.65    97.51   -2.02
```

**A relative measure reported without its baseline reads as an error.**

### 8.3 Eight rules on a standardised series

The band implements half of one rule. Seven of the eight look at runs rather than single
points, so the series is standardised against its own band and all eight are stated on that
one column:

```
    rule  description                                  days    share
    1     one point beyond 3 sigma                        0     0.0%
    2     nine in a row on one side of centre           230    45.7%
    3     six in a row rising or falling                 48     9.5%
    4     fourteen in a row alternating direction         0     0.0%
    5     two of three beyond 2 sigma, same side         46     9.1%
    6     four of five beyond 1 sigma, same side        173    34.4%
    7     fifteen in a row inside 1 sigma                 4     0.8%
    8     eight in a row all beyond 1 sigma              63    12.5%
    any   flagged by at least one rule                  309    61.4%
```

### 8.4 Reading the share per rule is what makes the baseline visible

Rules 2, 6 and 8 count how long the series stays on one side of centre. They were written for
a process **held at a fixed target**. Here the centre is a 20-day mean that follows the
series, and 61% of days sit above it, so a trend alone keeps those counters running.

> Their high share is a property of the baseline they were given, not of anything unusual in
> the data. Printing per-rule share is what surfaces that; a single "flagged / not flagged"
> column would hide it.

Rule 7 is the one worth remembering in general. Fifteen consecutive points inside one sigma
looks like the best possible outcome, and it is improbable enough to be evidence of
something: roughly `0.68^15`, under half a percent. **"Too good" is a signal too** — a
score that never moves, a test suite that is always green, a metric pinned at 100% all
deserve suspicion of the measurement before celebration of the result.

### 8.5 Days are not events

```
    band rule            54 days ->   22 events, longest run 6 days
    all eight rules     309 days ->   20 events, longest run 58 days
```

One excursion produces a flag on every day it lasts. Counting days answers a question about
rows; counting events answers the question a person asked.

### 8.6 A rule set catches different days, not more days

```
                            caught by the band   caught by the 8 rules
    top 3 single-day moves              1 of 3                  0 of 3
    top 3 three-day moves               3 of 3                  0 of 3
```

The band separates the two lists cleanly — sustained three-day displacement is caught, isolated
one-day jumps mostly are not. The eight run-based rules touch neither, because on all six of
these days the series only crossed the limit on the final day and the runs never accumulate.

> **A rule set is not a strictly larger net than the rule it extends.** It catches different
> days, and here it catches none of the six that the simpler rule was asked about.

---

## 9. A label recovered from one column

`07_label_leakage_and_importance_views.py`

### 9.1 The label a project writes when the outcome has not happened yet

```python
growth = rng.uniform(GROWTH_LOW, GROWTH_HIGH, size=len(frame))
frame["future_aum"] = frame[GENERATING_FEATURE] * growth
frame["label"] = (frame["future_aum"] >= LABEL_THRESHOLD).astype(int)
```

Nothing here is careless in isolation. The growth factor is random, the threshold is a
business rule, and the resulting rate looks reasonable:

```
    rows 10,000, positives 1,464 (0.1464)
```

What makes it unusable is that the only column feeding it is already a feature.

### 9.2 One column and one comparison reproduce it

```
    best single threshold on total_aum   >= 942,865
    accuracy of that one comparison        0.9864
    accuracy of always answering 'no'      0.8536
    AUC of the raw column, no model at all 0.9990
```

The arithmetic is forced. Below `1,000,000 / 1.20` no row can reach the threshold; above
`1,000,000 / 0.95` every row does. Only the band between them is decided by the random
factor, and it is a small fraction of the table. **Whatever a model scores from here is
mostly a measurement of that fact.**

```
    features 12, boosting rounds 200
    held-out AUC       0.9987
    held-out accuracy  0.9840
```

The model scores slightly *below* the raw column.

### 9.3 The result that is easiest to misread

```
    features 11
    held-out AUC       0.9869   (-0.0119)
    held-out accuracy  0.9520   (-0.0320)
```

A drop of one point sounds like a small leak. It is not a measurement of the leak at all:

```
        |corr(deposit_balance, total_aum)| = 0.893
        |corr(monthly_txn_amount, total_aum)| = 0.657
        |corr(fund_balance, total_aum)| = 0.471
        |corr(monthly_txn_count, total_aum)| = 0.460
```

The balance columns were drawn from the same quantity the label was drawn from.

> **Dropping one column cannot undo a label defined by that column while its proxies remain.**
> The only repair is a label that comes from an observed outcome rather than from a feature
> already in the table.

### 9.4 The two lines the script prints, and a third that does not apply here

The script prints two diagnostics before it trains anything:

1. the positive rate;
2. **the score reachable with the single strongest feature and one threshold**.

The second is the cheaper of the two and catches the most. If one column and one comparison
already reach an AUC of 0.99, the model is not the thing being measured.

A third check belongs on this list in general and is deliberately absent here: **the entity
overlap between train and test**. It matters whenever one entity contributes more than one
row — a random row-level split then puts the same entity on both sides, and rows from one
entity are correlated, so the held-out score is not held out. `train_test_split` gives no
warning, and a grouped split is the repair.

It does not apply to this table. `customers.csv` holds one row per customer — 10,000 rows,
10,000 distinct ids — so a row split is an entity split. Which is worth knowing rather than
assuming: **whether the check is needed is itself something to check**, and the answer is one
`nunique()` away.

---

## 10. Four rankings of the same features

`07_label_leakage_and_importance_views.py`, second half

Four measures, one fitted model, one held-out set:

| Measure | What it counts | Measured on |
| :--- | :--- | :--- |
| split | how often the column was cut on | training |
| gain | how much those cuts improved the objective | training |
| permutation | what breaks when the column is shuffled | **held-out** |
| contribution | how far the column moved individual predictions | **held-out** |

```
    feature                 split   gain   perm  contrib     gain value   perm value
    total_aum                   1      1      1        1         30,916      0.30597
    monthly_txn_amount          2      2      3        2            433      0.00016
    deposit_balance             4      3      5        3            383      0.00004
    age                         3      4      4        4            347      0.00013
    ...
    features whose rank is not the same under all four measures: 11 of 12
```

With the generating column removed the disagreement is total, and the four measures no longer
even agree on first place:

```
    features whose rank is not the same under all four measures: 11 of 11
        ranked first by split         monthly_txn_amount
        ranked first by gain          deposit_balance
        ranked first by permutation   deposit_balance
        ranked first by contribution  deposit_balance
```

**Split is the one that dissents, and it dissents for a structural reason.** A continuous
column offers many places to cut and collects splits whether or not they helped; a
low-cardinality column can be cut once, but that one cut can be decisive. Split count is
therefore partly a measure of cardinality.

⇒ Do not present split count as feature importance. Use gain, or one of the two measured on
held-out rows — and note that only those two answer the question a reader thinks the chart
is answering.

Two implementation details worth naming, because both are version-sensitive:

- The permutation measure needs a scikit-learn estimator, so the fitted booster is wrapped in
  a small facade inheriting `ClassifierMixin, BaseEstimator` — the mixin first, so its
  classifier tag wins. Current scikit-learn inspects estimator tags before it will score
  anything, and a bare duck-typed object is rejected.
- The contribution column uses SHAP when it is importable and falls back to the booster's own
  `pred_contrib` output otherwise. Both decompose a prediction into one number per feature
  plus a baseline, so the ranking is the same object either way.

---

## 11. The sample unit decides the answer

`08_association_rules_sample_unit.py`

Support, confidence and lift are proportions. **Of what** is the entire question.

### 11.1 The correct unit: one basket per customer

```python
PRODUCTS = {
    "deposit": "deposit_balance",
    "wealth": "wealth_balance",
    "fund": "fund_balance",
    "insurance": "insurance_balance",
}
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.30
MAX_ITEMSET_SIZE = 3
```

```
    baskets                  10,000
        holds deposit       0.9311
        holds wealth        0.3396
        holds fund          0.3426
        holds insurance     0.1621

    distinct combinations        16  out of 16 possible
```

Every one of the sixteen possible combinations occurs. That is what makes the next section
possible, and it is printed here so the reader sees it coming.

```
        rule                                     support   confidence     lift
        deposit + fund -> wealth                  0.1820       0.5687   1.6748
        fund -> wealth                            0.1947       0.5683   1.6734
        wealth -> fund                            0.1947       0.5733   1.6734
```

### 11.2 One line, and the answer becomes arithmetic

```python
distinct = baskets.drop_duplicates()
```

```
    10,000 rows collapse to 16 rows.
    The table still holds every combination that occurs. What it no longer
    holds is how many customers each combination stands for.
```

```
        itemset                             size     support
        deposit                                1    0.500000
        wealth                                 1    0.500000
        fund                                   1    0.500000
        insurance                              1    0.500000
        deposit + wealth                       2    0.250000
        ...
        rule                                     support   confidence     lift
        deposit -> wealth                         0.2500       0.5000   1.0000
        ...(every rule, 1.0000)
```

**Exactly 0.5, exactly 0.25, exactly 1.0 — and it has to be.** With all sixteen combinations
present once each, every product is in half of them and every pair in a quarter, so
`P(A∩B) = P(A)·P(B)` holds identically and lift is one by construction.

> The deduplicated table is independent **by construction**. That result would be identical
> whatever the customers actually did.

### 11.3 The repair is not "stop deduplicating"

```
    16 rows carrying a customer count each, 10,000 customers in total
    largest group 3,875 customers, smallest 11
    rules identical to the one-row-per-customer result: True
```

Group and keep the count as a weight, and the collapsed table reproduces the full result
exactly. **The unit was never the problem; discarding the multiplicity was.**

### 11.4 Scored against the number the data was built with

```
    P(fund | wealth)      0.5733
    P(fund | not wealth)  0.2240
    lift(wealth -> fund)  1.6734
    the mined rule reports  1.6734   (match: True)
    the deduplicated table reports 1.0000
```

Script 01 prints `lift(wealth -> fund) 1.6734` as the number it drew the data with.
Recovering it is what tells you the mining worked. **Failing to recover it is what the
deduplicated run should have shown, and instead it reported a clean, plausible, entirely
manufactured 1.0000.**

### 11.5 The tell

```
    lift ranges over 0.6716 per customer and 0.0000 per combination
```

> **Supports landing on exact powers of two, or every lift equal to 1.0, are not evidence
> that the data has no structure. Count the rows going in.** One print before mining and one
> after is the whole check.

---

## 12. A cohort is not a time series

`09_cohort_is_not_a_time_series.py`

### 12.1 A series that passes every shape check

```python
customers["cohort"] = customers["account_open_date"].dt.to_period("M")
grouped = customers.groupby("cohort")
```

```
    points 72, running 2019-01 to 2024-12
    period      customers    mean assets
    2019-01           135        547,623
    2019-02           128        434,737
    2019-03           128        567,743
```

Dates on the x axis, one value per month, no gaps. This is the shape a forecasting library
accepts, which is exactly why the question of whether it *should* be forecast never gets
asked.

### 12.2 The question the chart hides

```
    mean share of a month's customers who also appear in the next month: 0.0000
    highest such share across all 71 pairs: 0.0000
```

Against a genuine series:

```
    points 522, running 2023-01-02 to 2024-12-31
    share of one point's subject that appears in the next point: 1.0000
```

A line drawn between two points asserts that something moved from one value to the other.
That assertion needs the two points to be about the same thing. **Zero and one is the whole
argument.**

### 12.3 The shuffle test

The overlap check needs an identity column. When there is not one, there is a test that
needs nothing but the numbers: refit the same model to the same values in random orders.

```python
ARIMA_ORDER = (1, 1, 1)
SHUFFLE_TRIALS = 30
```

```
    series                    error as given   shuffled mean    ratio  shuffles as good
    cohort by join month           55,116.27       54,864.61     1.00          16 of 30
    MRD daily close                     0.83            9.35    11.29           0 of 30
```

If order carries information, destroying it should make the fit worse. On the daily series it
does, by a factor of eleven, and not one of thirty shuffles matches the real ordering. On the
cohort series the shuffled fits land in the same place and **sixteen of thirty do at least as
well as the real one** — the model was never reading time out of it. It was describing the
spread of seventy-two group means, and any permutation shares that spread.

The order is fixed rather than searched deliberately: holding it constant is what makes this a
comparison of the data rather than of two different models.

> `groupby` keys divide into *observation time* and *entity attribute time*. An opening date,
> a birth date, a registration date are attributes. Grouping by them produces cohort analysis;
> forecasting the result extrapolates group means and means nothing.

---

## 13. Two terms with no data under them

`09_cohort_is_not_a_time_series.py`, second half

### 13.1 An additive decomposition on the daily series

The market never trades at the weekend, and script 01 never generated those rows. The weekly
term is still defined there:

```
    day     training rows   weekly term
    Mon               105        1.3361
    Tue               105        1.3462
    Wed               104        1.2234
    Thu               104        1.3154
    Fri               104        1.3871
    Sat                 0       -3.3041
    Sun                 0       -3.3041

    weekend training rows 0, weekend weekly term 6.6082
```

The weekly term is a periodic function fitted at five of seven positions and then evaluated
at all seven. Friday and the following Monday have to be joined up, and the curve between
them takes whatever value smoothness dictates. **−3.3041 is the shape of the curve, not a
property of the data.**

Nothing in the component plot marks the two positions no observation ever constrained. A
reader who takes the chart at face value concludes that weekends behave differently, which is
true only in the sense that they do not exist.

### 13.2 One pass through the calendar cannot identify a yearly term

```
    one year of data     rows   260   complete cycles covered 0.99   term ranges     7.92
    two years of data    rows   522   complete cycles covered 2.00   term ranges    11.27

    correlation between the two yearly terms: +0.4937
```

Both fits succeed. Both print a clean seasonal curve. **The two curves agree at 0.49** — if a
real annual pattern were being recovered, the two estimates would be close.

With one pass through the calendar, the split between "trend" and "season" is not identified:
the same curve can be read as a falling trend with a flat season or a flat trend with a
falling season, and the fitted answer reflects the regulariser rather than the data. The
library says so itself, in a warning the run prints:

```
Yearly seasonality is enabled with less than 730 days (approximately 2 years) of history.
The model may be under-identified, and the trend/seasonality decomposition can be unstable...
```

**The number that separates the two cases is not in the plot. It is the count of complete
cycles the data covers, and it has to be printed on purpose.**

> These two, the misdated column in section 7.3, and the thinning in 7.2 are one shape:
> **the output is complete-looking, and part of it has nothing underneath.**

---

## 14. Two backends, and a limit in the wrong unit

`10_search_backends_and_ui.py`

### 14.1 One corpus, two indexes

Eight short policy documents written into the script — no data files, and every question has
a known correct source:

```python
CHUNK_WORDS = 60
CHUNK_OVERLAP = 15
TOP_K = 3
TOKEN_BUDGET = 220
```

```
    documents 8, chunks 15, window 60 words with 15 overlapping
    chunk length in characters: min 101, max 437, mean 271
    estimated tokens in the whole corpus: 1,011
    keyword index built over 15 chunks
    vector index built with gemini-embedding-001 at 768 dimensions
```

Both indexes are built over the same fifteen chunks, so the only variable is the scoring.

One detail in the vector index that repeats a point from module 02: truncated embeddings are
not unit length, so cosine is taken explicitly rather than read off a dot product.

```python
matrix = np.asarray(vectors, dtype=float)
return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
```

### 14.2 Neither backend is the better one

```
    question                                            kind          keyword   vector
    What does Clause 7.3 cover?                         exact term     rank 1   rank 1
    What is the aggregate limit under Clause 4.1?       exact term     rank 1   rank 2
    Someone stole my suitcase at the airport. Am I cov  paraphrase     missed   rank 1
    I got sick on holiday and had to be flown home. Wh  paraphrase     rank 3   rank 1
    How long does the insurer have to decide on my cla  paraphrase     rank 2   rank 1

    on exact term   questions: keyword 2 of 2 (mean rank 1.0), vector 2 of 2 (mean rank 1.5)
    on paraphrase   questions: keyword 2 of 3 (mean rank 2.5), vector 3 of 3 (mean rank 1.0)
```

The split is clean and it is mechanical. Term-overlap scoring puts an exact clause number
first because a rare token carries most of the weight. Embedding similarity finds the clause
number too — it is scored by what it resembles rather than matched — but is not guaranteed
the top position the way an exact match is. On a question that shares no vocabulary with its
answer, term overlap has nothing to work with and one of the three is missed entirely.

**They fail on different questions. That is the reason a system keeps both rather than
choosing.**

### 14.3 Three chunks is not a limit on anything the model cares about

```
    question                                           fixed count            token budget
                                                chunks      tokens      chunks      tokens
    What does Clause 7.3 cover?                      3         215           3         215
    Someone stole my suitcase at the airpo           3         186           3         186
    I got sick on holiday and had to be fl           3         243           2         209
    How long does the insurer have to deci           3         182           4         213
```

The budget is 220 tokens and the budgeted column stays under it while varying between two and
four chunks. The fixed count never tracks tokens at all — it happens to stay close here only
because every chunk in this corpus is nearly the same length.

> **"Three results" and "220 tokens" are not the same kind of limit.** Three chunks can be
> three fragments or three long sections. The constraint downstream is a context window,
> measured in tokens, so that is the unit the cutoff belongs in.

The budgeting loop is deliberately conservative — it stops *before* the first chunk that
would exceed the budget, and always keeps at least one:

```python
for hit in hits:
    cost = max(1, len(hit["text"]) // CHARS_PER_TOKEN)
    if kept and spent + cost > budget:
        break
    kept.append(hit)
    spent += cost
```

### 14.4 A web interface over the same two backends

`--ui` serves a small Blocks page: a question box, a backend selector, a cutoff selector, and
panes for the retrieved chunks, the context size and the answer. It calls the same
`search` / `by_token_budget` / `answer` functions the command-line path uses, so the interface
is a second front end rather than a second implementation.

Both are worth having for opposite reasons. The command-line run produces a scored table that
can be compared between runs; the interface makes the retrieved chunks visible next to the
answer, which is what turns "the answer is wrong" into "the right chunk was never retrieved".

---

## 15. Peeling a failure back one layer at a time

`10_search_backends_and_ui.py`, final section

When a question fails end to end, the useful first move is not to guess which component is
broken. It is to establish **which layer** the failure lives in.

```
    Taking the keyword backend on: Someone stole my suitcase at the airport. Am I covered?
    Layer 1, the answer      : built from ['property-allrisks', 'medical-abroad', 'medical-abroad']
    Layer 2, the retrieval   : expected baggage-loss, got ['property-allrisks', 'medical-abroad', ...]
    Layer 3, the raw scoring : the expected document's best chunk sits at rank 12 of 15
                               top score 2.3210, expected document's best 0.7815
```

Three layers, three different repairs:

| Layer | If the failure is here | The repair |
| :--- | :--- | :--- |
| **The answer** | The right chunks were retrieved and the reply is still wrong | prompt, schema, model |
| **The retrieval** | The right document exists but did not make the cutoff | scoring, k, budget, a second backend |
| **The raw scoring** | The document is not in the index at all | ingestion, chunking, parsing |

Here the answer was never the problem: the document was in the index the whole time and the
scoring put it at rank 12 of 15. **No amount of prompt work reaches rank 12.** That is a
different repair from anything that could be done to the model's instructions, and knowing
which one you need is worth more than any single fix.

The model's reply on that query is itself the correct behaviour under the circumstances:

```
        keyword   retrieved [property-allrisks, medical-abroad]
                  not in the retrieved context.
```

It declined rather than answering from unrelated clauses. A pipeline that refuses when the
context does not contain the answer is doing the right thing with the wrong input — which is
precisely why the score has to be attributed to retrieval and not to generation.

---

## 16. Routing twice before answering, and checking every citation

`11_answer_routing_and_citation.py`

### 16.1 Page numbers chosen to make invention detectable

```python
# Page bodies elided here; the page numbers are the ones in the script.
REPORTS = {
    "Alderway Foods":       {14: ..., 29: ..., 47: ..., 63: ...},
    "Brightlane Logistics": {11: ..., 38: ..., 52: ..., 71: ...},
    "Coldharbour Energy":   { 9: ..., 26: ..., 44: ..., 58: ...},
}
```

```
    No report is numbered from 1, and no two share a page number.
    Any citation outside [9, 11, 14, 26, 29, 38, 44, 47, 52, 58, 63, 71] was invented rather than read.
```

Sparse, non-contiguous numbering is the point. A model reaching for a plausible citation
reaches for a small round number, and a corpus numbered 1, 2, 3 would hide that.

### 16.2 Two routers, scored separately

```python
def route_to_report(client, model: str, question: str) -> str:
def route_to_type(client, model: str, question: str) -> str:
```

```
    report routing 6 of 6, type routing 6 of 6
```

The two are separate calls and separate scores because they fail separately, and because the
second one changes the instructions the answering call receives:

```python
TYPE_RULES = {
    "number": "answer must be a bare number with no units, no thousands separators "
              "and no words.",
    "boolean": 'answer must be exactly "yes" or "no".',
    "name": "answer must be a single proper name and nothing else.",
    "names": "answer must be the names only, separated by commas, in the order the "
             "source gives them.",
    "string": "answer must be one short sentence.",
}
```

Routing to a type is what lets each answering call carry **one** set of formatting rules
instead of five. More rules in a single request means more chances to break one.

Routing first also means a routing mistake cannot be recovered later: the correct pages are
never put in front of the model that answers. The script therefore answers from **whatever
the router chose**, not from the correct report, so a routing error surfaces as a wrong
answer rather than being silently corrected.

### 16.3 The four fields, in order

```
{"reasoning": "...", "answer": ..., "references": [...], "confidence": 0..1}
```

Reasoning comes first on purpose. The model fills the object in order, so the working is
written before the answer rather than after it — the difference between reasoning and
justifying a conclusion already reached.

The schema also carries the escape hatch that makes the whole thing honest:

> If the pages do not contain the answer, set answer to "N/A", references to an empty list,
> and confidence to 0.

### 16.4 Checking the citations

```python
def validate_references(result: dict, supplied: set) -> dict:
    kept = [number for number in numbers if number in supplied]
    dropped = [number for number in numbers if number not in supplied]
```

A page number the model was never given cannot have been read, whatever the answer says.

```
    Q: What was Alderway Foods' revenue in 2024?
       expected '812.4' from pages [14]
       answer   '812.4'
       cited [14]   valid [14]   invented []
```

**This is the cheapest correctness check in the pipeline, and it needs no judgement about
whether the answer itself is right.**

### 16.5 The question with no answer in the corpus

```
    Q: What dividend per share did Coldharbour Energy declare?
       expected 'N/A' from pages []
       answer   'N/A'
       cited []   valid []   invented []
       schema ok, answer ok, confidence 0
```

Its type is `number` — the *shape* of the question is numeric even though the corpus has no
answer for it. That separates "routed to the right type" from "found an answer", which a
question answerable from the pages cannot do.

The scoring has to treat `N/A` as schema-conformant rather than as a numeric-format failure;
otherwise the conformance figure penalises the one behaviour the instructions ask for. The
check is explicit about it:

```python
if is_not_available(value):
    return not result["references"]
```

An `N/A` with citations attached is still a failure — if there is no answer there is no source.

### 16.6 Comparisons: split, answer, recombine

A comparison spans every report, so the router has nothing to choose. Splitting restores the
property the rest of the pipeline depends on: each sub-question has one report, one set of
pages, one citable source.

```
    Q: Which of the three companies had the highest revenue in 2024?
       Alderway Foods                 812.4   pages [14]
       Brightlane Logistics          1204.7   pages [11]
       Coldharbour Energy             640.1   pages [9]
       combined -> 'Brightlane Logistics'   expected 'Brightlane Logistics'   ok
```

Each sub-answer keeps its own citation, so the comparison **inherits** sources rather than
producing a claim no page supports. The final call is told explicitly to compare figures that
have already been extracted, not to reason about the companies.

### 16.7 Five numbers, because they fail separately

```
    report routing        6 of 6
    answer type routing   6 of 6
    schema conformance    6 of 6
    answers correct       6 of 6
    comparisons correct   2 of 2
    page citations        5 made, 0 of them invented
```

> A wrong answer traced to routing is repaired in the router; one traced to the schema is
> repaired in the prompt; an invented citation is caught **without knowing whether the answer
> was right at all**. Collapsing these into one accuracy figure throws away the only
> information that says what to fix.

As in section 6.5, the model is not deterministic across runs even at temperature 0: type
routing has come back 5 of 6 on other runs, with `names` misrouted. The structure of the
measurement is the deliverable, not any single run's tally.

---

## 17. What all of these have in common

### 17.1 Nine failures, one countermeasure

| Script | The quantity that settles it |
| :--- | :--- |
| 02 | rows per key; a second total computed from an independent column |
| 03 | the recomputed ratio; the sum of the group sizes against the row count |
| 04 | the model's number against a truth table; which groups each digest reaches |
| 05 | distinct x values, not rows; filled cells against correctly-placed cells |
| 06 | the share flagged per rule; events against days |
| 07 | the score from one column and one threshold; four rankings side by side |
| 08 | rows going into the mining, and rows coming out |
| 09 | population overlap between neighbouring points; fit error under shuffling |
| 10 | rank of the expected document; tokens actually spent |
| 11 | cited pages against supplied pages |

Every one of these is one print statement or one small function. None of them requires
understanding the failure in advance — they are checks that make a class of failure visible
whether or not you expected it.

### 17.2 Two errors that only look like the same error

A useful split, because the second class is where the effort belongs:

**Raises.** Missing package, wrong path, port in use, mismatched endpoint name, a
serialisation type the encoder does not know. These announce themselves; the fix is
mechanical.

**Does not raise.** A wrong aggregation, a clamped column, a band that drops a segment, a
digest that spans two subjects, a misdated column, a label recovered from a feature, a lift
of exactly one, a forecast on a cohort. The program completes, the output is well-formed, and
the magnitude is plausible.

The second class is not harder to fix. It is harder to **notice**, and noticing is the entire
problem.

### 17.3 A repair can move an error out of sight

Section 6.4 is the clearest instance: `head(5) + tail(5)` replaced a reply that visibly only
had January data with a reply that spans the full year and is built from two different
instruments. The second is more wrong and much harder to catch.

The same pattern appears in 3.4 (filtering after a join instead of before), and in 4.3 (once
the empty bands are removed, the funnel looks clean and the missing 1,276 customers are still
missing).

> **Judge a fix by whether its input became correct, not by whether its output looks
> correct.**

### 17.4 Ask the data what the categories are

The contrast in section 4.3 is worth stating on its own. When the group list comes from a
query, nobody is lost. When it is written from domain intuition, the categories that exist in
the data but not in the author's head disappear silently, and the categories in the author's
head but not in the data show up as zeroes.

Print `unique()` before writing a band, a label list, or a funnel.

### 17.5 A measure needs its baseline

Two instances, both in section 8:

- A band flag reported as date and price contradicts itself; reported with centre, upper and
  lower it does not.
- A run-based rule firing on 45.7% of days looks alarming until the baseline is named — a
  rolling mean that the series trends away from, with 61% of days above it.

A relative measure printed without the thing it is relative to is not a weak report. It is an
unreadable one.

---

## 18. The eleven scripts

| # | Script | What it establishes | The quantity it prints |
| :--- | :--- | :--- | :--- |
| 01 | `build_project_datasets.py` | Five sources with every later claim's answer written down first | yearly moves, planted lift, three totals, clamped rows |
| 02 | `join_grain_and_aggregation_audit.py` | Grain and aggregation semantics | rows per key, mean drift, 197x total, rank changes |
| 03 | `dashboard_metrics_and_cache.py` | Reported columns, bands, and cache invalidation | clamped rows, band shortfall, cold vs warm scaling |
| 04 | `tool_return_shapes.py` | A tool's return value sets the ceiling on the answer | error against the truth table, groups per digest |
| 05 | `chart_criterion_and_index_alignment.py` | A criterion on the wrong count; index alignment | rows vs distinct x, misplaced cells |
| 06 | `bollinger_and_spc_rules.py` | Control limits, and rule sets that catch different days | per-rule share, events vs days, single vs multi-day |
| 07 | `label_leakage_and_importance_views.py` | A label recovered from a feature; four importance measures | one-column AUC, proxy correlations, rank disagreement |
| 08 | `association_rules_sample_unit.py` | The sample unit decides the answer | rows before and after, support powers of two, lift |
| 09 | `cohort_is_not_a_time_series.py` | Cohorts, and terms with no data under them | population overlap, shuffle ratio, weekend rows, cycles |
| 10 | `search_backends_and_ui.py` | Two backends; a limit in the wrong unit; layer isolation | mean rank per question kind, tokens spent, rank of the miss |
| 11 | `answer_routing_and_citation.py` | Routing, structured answers, citation validation | five separate scores, invented citations |

**Dependencies.** Script 01 writes everything the others read; run it first. Scripts 04, 10
and 11 call a model API. The rest are offline.

**Providers.** One provider per script. Scripts 04 and 11 are text-only and prefer DeepSeek;
script 10 needs embeddings and answers, so it stays entirely on Gemini — one key, one base
URL, one quota to reason about when something fails. All three carry a backoff that retries on
rate-limit responses, because a script that indexes a corpus sends a burst rather than a
trickle.

**Outputs.** `data/` and `outputs/` are both regenerated and both ignored by git. Nothing in
either is source material.

---

## 19. What this module is really about

The techniques in the earlier modules can all be checked by running them: the model answers
or it does not, the loss falls or it does not, the detector finds the object or it does not.

These eleven scripts are about the part that cannot be checked that way. Every one of them
runs cleanly on the first try. The join returns a table. The aggregation returns a number of
the right magnitude. The model answers confidently. The classifier reports an AUC any
reviewer would sign off. The rule mining returns rules. The forecast returns a smooth curve.

What separates the correct version from the incorrect one, in all eleven cases, is a quantity
somewhere in the middle that nobody printed:

> The rows per key. The second total. The distinct x values. The groups the digest reached.
> The share per rule. The score from one column. The rows going into the mining. The overlap
> between neighbouring points. The rank of the document that was missed. The pages that were
> actually supplied.

None of these are sophisticated. All of them are cheap. Each one turns an invisible failure
into a visible one, which is the only part of this that is hard.
