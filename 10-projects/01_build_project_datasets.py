"""Generate the five data sources the rest of this module reads, from known ground truth.

Demonstrates how to build project data whose every later claim can be checked:
    1. Draw two years of daily prices for four instruments from an explicit random walk.
    2. Plant one trading halt and one shock in those prices, and record where they are.
    3. Draw a customer table whose product holdings are correlated on purpose.
    4. Draw a staff table and a review table that stand in a one-to-many relation.
    5. Draw a district table that carries both a daily count and a running total.
    6. Draw a facility table whose reported ratio is capped and whose parts do not sum.
    7. Print the ground truth behind every file, so later scripts can be scored against it.

Module 10: Applied Projects - Dataset Construction.
"""

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
SEED = 20260828

# --- Market data ---------------------------------------------------------------
# Four invented instruments. Nothing here is a real listed company.
INSTRUMENTS = {
    "ARB": {"name": "Arbor Technologies", "start": 142.0, "drift": 0.00042, "vol": 0.0165},
    "CLD": {"name": "Calder Energy", "start": 58.5, "drift": -0.00011, "vol": 0.0231},
    "MRD": {"name": "Meridian Foods", "start": 91.2, "drift": 0.00018, "vol": 0.0104},
    "SVN": {"name": "Severn Logistics", "start": 27.4, "drift": 0.00035, "vol": 0.0192},
}
MARKET_START = date(2023, 1, 2)
MARKET_END = date(2024, 12, 31)

# One instrument stops trading for a stretch. A gap in a daily series is what makes
# "row 1 to row N" and "first date to last date" stop meaning the same thing.
HALT_TICKER = "CLD"
HALT_START = date(2024, 5, 13)
HALT_DAYS = 11

# Each instrument gets one sharp move, so a band drawn from a rolling mean and
# standard deviation has something to actually flag.
SHOCK_DAY_INDEX = {"ARB": 168, "CLD": 402, "MRD": 291, "SVN": 96}
SHOCK_SIZE = {"ARB": 0.091, "CLD": -0.118, "MRD": 0.067, "SVN": -0.083}
SHOCK_LENGTH = 3

# --- Customer data -------------------------------------------------------------
CUSTOMER_ROWS = 10000
CUSTOMER_OPEN_FIRST = date(2019, 1, 1)
CUSTOMER_OPEN_LAST = date(2024, 12, 31)

# Base probability that a customer holds each product at all.
HOLDING_BASE = {"deposit": 0.93, "wealth": 0.34, "fund": 0.22, "insurance": 0.17}
# Holding wealth management makes a fund holding far more likely. This single
# number is the entire signal an association rule can recover.
WEALTH_TO_FUND_MULTIPLIER = 2.6
# Holding insurance is mildly discouraged among fund holders.
FUND_TO_INSURANCE_MULTIPLIER = 0.72

# --- Staff data ----------------------------------------------------------------
STAFF_ROWS = 480
REVIEW_YEARS = (2023, 2024)
REVIEW_QUARTERS = (1, 2, 3, 4)
DEPARTMENTS = ["Claims", "Underwriting", "Operations", "Compliance", "Technology"]
GRADES = ["Associate", "Senior", "Lead", "Principal"]

# --- District data -------------------------------------------------------------
DISTRICT_YEAR = 2024
DISTRICTS = [
    "Ashfield", "Barrowgate", "Clifton Vale", "Dunmore", "Eastmoor", "Fenwick",
    "Granthorpe", "Halstead", "Inverleith", "Kirkburn", "Langmere", "Marchwood",
    "Netherby", "Oakhaven", "Pinecrest", "Quarryside", "Rosslare", "Thornbury",
]

# --- Facility data -------------------------------------------------------------
FACILITY_COUNT = 250
FACILITY_MONTHS = 12
FACILITY_YEAR = 2024
# The reported ratio column is written by an upstream system that clamps at 99.
REPORTED_RATIO_CAP = 99


def business_days(first: date, last: date) -> list:
    """List every weekday between two dates, inclusive."""
    days = []
    cursor = first
    while cursor <= last:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def build_market_table(rng: np.random.Generator) -> pd.DataFrame:
    """Walk four price series forward one weekday at a time, then cut the halt out.

    The walk is multiplicative, so a price never goes negative and a percentage
    move means the same thing at any level. The shock is applied as extra return
    on consecutive days rather than as a single spike, because a one-day spike
    leaves a rolling mean almost untouched and would flag nothing.
    """
    calendar = business_days(MARKET_START, MARKET_END)
    halt_dates = set()
    if HALT_DAYS:
        cursor = HALT_START
        while len(halt_dates) < HALT_DAYS:
            if cursor.weekday() < 5:
                halt_dates.add(cursor)
            cursor += timedelta(days=1)

    frames = []
    for ticker, spec in INSTRUMENTS.items():
        n = len(calendar)
        returns = rng.normal(spec["drift"], spec["vol"], size=n)

        shock_start = SHOCK_DAY_INDEX[ticker]
        per_day = SHOCK_SIZE[ticker] / SHOCK_LENGTH
        returns[shock_start:shock_start + SHOCK_LENGTH] += per_day

        closes = spec["start"] * np.exp(np.cumsum(returns))
        # Intraday range hangs off the close, so high >= max(open, close) always.
        prev_close = np.concatenate([[spec["start"]], closes[:-1]])
        opens = prev_close * (1 + rng.normal(0, spec["vol"] / 3, size=n))
        spread = np.abs(rng.normal(0, spec["vol"] / 2, size=n)) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread
        volume = rng.integers(180_000, 4_200_000, size=n)

        frame = pd.DataFrame({
            "ticker": ticker,
            "instrument": spec["name"],
            "trade_date": [d.isoformat() for d in calendar],
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": volume,
        })
        if ticker == HALT_TICKER:
            keep = [date.fromisoformat(d) not in halt_dates for d in frame["trade_date"]]
            frame = frame[keep].reset_index(drop=True)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def write_market_db(market: pd.DataFrame) -> Path:
    """Write the price table into a SQLite file that later scripts query with SQL.

    The file is deleted first rather than appended to, so a second run of this
    script leaves exactly the same database instead of a doubled one.
    """
    path = DATA / "market.sqlite"
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE daily_price (
                ticker      TEXT    NOT NULL,
                instrument  TEXT    NOT NULL,
                trade_date  TEXT    NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      INTEGER,
                PRIMARY KEY (ticker, trade_date)
            )
        """)
        market.to_sql("daily_price", conn, if_exists="append", index=False)
        conn.execute("CREATE INDEX idx_price_date ON daily_price (trade_date)")
    return path


def build_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Draw a customer base whose product holdings are linked to each other on purpose.

    Assets are drawn lognormal, so a small share of customers sits above the one
    million mark and the rest do not. Holdings are drawn conditionally: the fund
    probability is multiplied when a wealth product is already held. That single
    multiplier is the whole reason an association rule can find anything here.
    """
    n = CUSTOMER_ROWS
    total_aum = np.round(rng.lognormal(mean=12.75, sigma=0.95, size=n), 2)

    span_days = (CUSTOMER_OPEN_LAST - CUSTOMER_OPEN_FIRST).days
    open_offsets = rng.integers(0, span_days + 1, size=n)
    open_dates = [CUSTOMER_OPEN_FIRST + timedelta(days=int(o)) for o in open_offsets]

    # Wealthier customers transact more, but the relation is noisy on purpose.
    aum_scale = np.clip(total_aum / 200_000, 0.15, 12.0)
    monthly_txn_amount = np.round(rng.gamma(2.2, 1400 * aum_scale), 2)
    monthly_txn_count = rng.poisson(np.clip(4 + 3 * np.log1p(aum_scale), 1, 40))
    mobile_login_count = rng.poisson(np.clip(9 + 4 * np.log1p(aum_scale), 1, 90))
    branch_visit_count = rng.poisson(np.clip(2.4 - 0.4 * np.log1p(aum_scale), 0.1, 6))

    holds_deposit = rng.random(n) < HOLDING_BASE["deposit"]
    holds_wealth = rng.random(n) < HOLDING_BASE["wealth"]
    fund_prob = np.where(
        holds_wealth,
        np.clip(HOLDING_BASE["fund"] * WEALTH_TO_FUND_MULTIPLIER, 0, 1),
        HOLDING_BASE["fund"],
    )
    holds_fund = rng.random(n) < fund_prob
    insurance_prob = np.where(
        holds_fund,
        HOLDING_BASE["insurance"] * FUND_TO_INSURANCE_MULTIPLIER,
        HOLDING_BASE["insurance"],
    )
    holds_insurance = rng.random(n) < insurance_prob

    def balance(mask, share):
        drawn = total_aum * share * rng.uniform(0.55, 1.45, size=n)
        return np.round(np.where(mask, drawn, 0.0), 2)

    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "age": rng.integers(22, 76, size=n),
        "city_tier": rng.choice([1, 2, 3], size=n, p=[0.28, 0.44, 0.28]),
        "account_open_date": [d.isoformat() for d in open_dates],
        "total_aum": total_aum,
        "monthly_txn_amount": monthly_txn_amount,
        "monthly_txn_count": monthly_txn_count,
        "mobile_login_count": mobile_login_count,
        "branch_visit_count": branch_visit_count,
        "deposit_balance": balance(holds_deposit, 0.52),
        "wealth_balance": balance(holds_wealth, 0.31),
        "fund_balance": balance(holds_fund, 0.22),
        "insurance_balance": balance(holds_insurance, 0.09),
    })


def build_staff(rng: np.random.Generator) -> tuple:
    """Draw one row per employee, and one review row per employee per quarter worked.

    These two tables exist to make a grain mismatch visible. The master table has
    one row per person; the review table has up to eight. The count is not the same
    for everyone: someone hired halfway through has fewer reviews, and salary rises
    with years of service. Those two facts together are what makes a join at the
    wrong grain shift an average rather than merely duplicate rows.
    """
    hire_year = rng.integers(2012, REVIEW_YEARS[-1] + 1, size=STAFF_ROWS)
    hire_quarter = rng.integers(1, 5, size=STAFF_ROWS)
    years_of_service = (REVIEW_YEARS[-1] + 1) - hire_year
    staff = pd.DataFrame({
        "staff_id": [f"E{i:04d}" for i in range(1, STAFF_ROWS + 1)],
        "department": rng.choice(DEPARTMENTS, size=STAFF_ROWS),
        "grade": rng.choice(GRADES, size=STAFF_ROWS, p=[0.44, 0.31, 0.17, 0.08]),
        "hire_year": hire_year,
        "hire_quarter": hire_quarter,
        "base_salary": np.round(
            rng.normal(58_000 + 1_650 * years_of_service, 8_500, size=STAFF_ROWS), -2
        ),
    })

    rows = []
    for record in staff.itertuples():
        # Each employee has a personal mean, so quarterly scores are not pure noise.
        personal = rng.normal(3.4, 0.42)
        for year in REVIEW_YEARS:
            for quarter in REVIEW_QUARTERS:
                started = (year, quarter) >= (int(record.hire_year), int(record.hire_quarter))
                if not started:
                    continue
                score = float(np.clip(rng.normal(personal, 0.28), 1.0, 5.0))
                rows.append({
                    "staff_id": record.staff_id,
                    "review_year": year,
                    "review_quarter": quarter,
                    "review_score": round(score, 2),
                })
    return staff, pd.DataFrame(rows)


def build_districts(rng: np.random.Generator) -> pd.DataFrame:
    """Draw a per-district daily table holding both a daily count and a running total.

    Both columns are written out because that is what an upstream reporting system
    usually hands over. Only one of them can be added across rows without turning
    into nonsense, and nothing in the column names says which.
    """
    first = date(DISTRICT_YEAR, 1, 1)
    last = date(DISTRICT_YEAR, 12, 31)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]

    frames = []
    for district in DISTRICTS:
        level = rng.uniform(14, 130)
        seasonal = 1 + 0.55 * np.sin(np.linspace(0, 2 * np.pi, len(days)) + rng.uniform(0, 3))
        new_cases = rng.poisson(np.clip(level * seasonal, 1, None))
        cumulative = np.cumsum(new_cases)
        recovered = np.round(cumulative * rng.uniform(0.82, 0.93)).astype(int)
        deaths = np.round(cumulative * rng.uniform(0.004, 0.012)).astype(int)
        frames.append(pd.DataFrame({
            "district": district,
            "report_date": [d.isoformat() for d in days],
            "new_cases": new_cases,
            "cumulative_cases": cumulative,
            "recovered_total": recovered,
            "deaths_total": deaths,
            "active_cases": cumulative - recovered - deaths,
        }))
    return pd.concat(frames, ignore_index=True)


def build_facilities(rng: np.random.Generator) -> pd.DataFrame:
    """Draw a monthly bed table with a clamped ratio column and beds out of service.

    Two properties are planted here. The reported ratio is clamped at 99, so a
    facility running at 104 percent of its staffed beds reads as 99. And occupied
    plus free does not reach the total, because some beds are out of service and
    counted in neither.
    """
    facilities = [f"Facility {i:03d}" for i in range(1, FACILITY_COUNT + 1)]
    departments = ["Cardiology", "General Medicine", "Orthopaedics", "Paediatrics", "Surgery"]

    rows = []
    for facility in facilities:
        department = rng.choice(departments)
        total_beds = int(rng.integers(24, 320))
        pressure = rng.uniform(0.52, 1.12)
        for month in range(1, FACILITY_MONTHS + 1):
            out_of_service = int(rng.integers(0, max(2, total_beds // 14)))
            staffed = total_beds - out_of_service
            occupied = int(np.clip(round(staffed * rng.normal(pressure, 0.09)), 0, staffed))
            free = staffed - occupied
            true_ratio = 100.0 * occupied / total_beds
            rows.append({
                "facility": facility,
                "department": department,
                "report_month": f"{FACILITY_YEAR}-{month:02d}",
                "total_beds": total_beds,
                "occupied_beds": occupied,
                "free_beds": free,
                "out_of_service_beds": out_of_service,
                "reported_utilization_pct": min(REPORTED_RATIO_CAP, round(true_ratio)),
            })
    return pd.DataFrame(rows)


def report_market_truth(market: pd.DataFrame) -> None:
    """Print the yearly move of every instrument, computed from first and last close."""
    print("\n--- 7. Ground truth: yearly move per instrument ---")
    print("A later script asks a model this same question. These are the answers.")
    market = market.copy()
    market["year"] = market["trade_date"].str.slice(0, 4)
    print(f"{'ticker':<8}{'year':<7}{'first date':<13}{'last date':<13}"
          f"{'first':>9}{'last':>9}{'change %':>11}")
    for (ticker, year), group in market.groupby(["ticker", "year"], sort=True):
        group = group.sort_values("trade_date")
        first_row, last_row = group.iloc[0], group.iloc[-1]
        change = 100.0 * (last_row["close"] - first_row["close"]) / first_row["close"]
        print(f"{ticker:<8}{year:<7}{first_row['trade_date']:<13}{last_row['trade_date']:<13}"
              f"{first_row['close']:>9.2f}{last_row['close']:>9.2f}{change:>10.2f}%")

    halted = market[market["ticker"] == HALT_TICKER]["trade_date"]
    print(f"\n{HALT_TICKER} is missing {HALT_DAYS} trading days starting {HALT_START.isoformat()}: "
          f"{len(halted)} rows against {len(market[market['ticker'] == 'ARB'])} for ARB.")
    print("Weekends were never generated, so the number of Saturday and Sunday rows is 0.")


def report_customer_truth(customers: pd.DataFrame) -> None:
    """Print the holding rates and the co-holding structure that was planted."""
    print("\n--- 7. Ground truth: product holdings ---")
    flags = pd.DataFrame({
        "deposit": customers["deposit_balance"] > 0,
        "wealth": customers["wealth_balance"] > 0,
        "fund": customers["fund_balance"] > 0,
        "insurance": customers["insurance_balance"] > 0,
    })
    for product in flags.columns:
        print(f"    holds {product:<10} {flags[product].mean():.4f}")

    with_wealth = flags[flags["wealth"]]["fund"].mean()
    without_wealth = flags[~flags["wealth"]]["fund"].mean()
    support_both = (flags["wealth"] & flags["fund"]).mean()
    lift = support_both / (flags["wealth"].mean() * flags["fund"].mean())
    print(f"\n    P(fund | wealth)     {with_wealth:.4f}")
    print(f"    P(fund | not wealth) {without_wealth:.4f}")
    print(f"    lift(wealth -> fund) {lift:.4f}   <- the number an association rule should recover")

    distinct = flags.drop_duplicates()
    print(f"\n    rows in the table                 {len(flags)}")
    print(f"    distinct holding combinations     {len(distinct)}")
    print("    Those two numbers are what an analysis of this table has to choose between.")

    above_million = (customers["total_aum"] >= 1_000_000).mean()
    print(f"\n    share with total_aum >= 1,000,000 {above_million:.4f}")


def report_district_truth(districts: pd.DataFrame) -> None:
    """Print the total that is correct and the total that adding the wrong column gives."""
    print("\n--- 7. Ground truth: district totals ---")
    true_total = int(districts["new_cases"].sum())
    max_of_cumulative = int(districts.groupby("district")["cumulative_cases"].max().sum())
    sum_of_cumulative = int(districts["cumulative_cases"].sum())
    print(f"    sum of new_cases                        {true_total:>12,}")
    print(f"    sum of per-district max(cumulative)     {max_of_cumulative:>12,}")
    print(f"    sum of cumulative_cases over all rows   {sum_of_cumulative:>12,}"
          f"   ({sum_of_cumulative / true_total:.1f}x the truth)")


def report_facility_truth(facilities: pd.DataFrame) -> None:
    """Print how often the reported ratio is clamped and how often the parts fall short."""
    print("\n--- 7. Ground truth: facility beds ---")
    clamped = (facilities["reported_utilization_pct"] >= REPORTED_RATIO_CAP).sum()
    parts_short = (facilities["occupied_beds"] + facilities["free_beds"]
                   < facilities["total_beds"]).sum()
    print(f"    rows                                  {len(facilities):>8,}")
    print(f"    rows reading exactly {REPORTED_RATIO_CAP}%              {clamped:>8,}")
    print(f"    rows where occupied + free < total    {parts_short:>8,}")
    print("    The second number is out-of-service beds, counted in neither column.")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("--- 1-2. Daily prices for four instruments ---")
    market = build_market_table(rng)
    db_path = write_market_db(market)
    print(f"{len(market):,} rows across {market['ticker'].nunique()} instruments -> {db_path.name}")

    print("\n--- 3. Customer base ---")
    customers = build_customers(rng)
    customers.to_csv(DATA / "customers.csv", index=False)
    print(f"{len(customers):,} rows -> customers.csv")

    print("\n--- 4. Staff master and quarterly reviews ---")
    staff, reviews = build_staff(rng)
    staff.to_csv(DATA / "staff.csv", index=False)
    reviews.to_csv(DATA / "staff_reviews.csv", index=False)
    print(f"{len(staff):,} rows -> staff.csv")
    print(f"{len(reviews):,} rows -> staff_reviews.csv "
          f"({len(reviews) // len(staff)} per employee)")

    print("\n--- 5. District daily counts ---")
    districts = build_districts(rng)
    districts.to_csv(DATA / "district_daily.csv", index=False)
    print(f"{len(districts):,} rows -> district_daily.csv")

    print("\n--- 6. Facility bed occupancy ---")
    facilities = build_facilities(rng)
    facilities.to_csv(DATA / "facility_beds.csv", index=False)
    print(f"{len(facilities):,} rows -> facility_beds.csv")

    report_market_truth(market)
    report_customer_truth(customers)
    report_district_truth(districts)
    report_facility_truth(facilities)

    print(f"\nAll five sources written to {DATA}")
    print(f"The generator is seeded with {SEED}, so rerunning reproduces them exactly.")


if __name__ == "__main__":
    main()
