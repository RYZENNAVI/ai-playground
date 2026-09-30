"""This script generates the five data sources the rest of this module reads as synthetic
data with known ground truth: daily prices for four instruments drawn from a geometric
random walk into SQLite, and customer, staff and review, district and facility tables as
CSV files. Each source has one property planted on purpose, and the script prints it:
    1. Draw two years of weekday prices for four instruments, and plant one trading halt
       and one three-day shock in them.
    2. Draw a customer table in which a wealth product makes a fund holding more likely.
    3. Draw a staff table and a review table that stand in a one-to-many relation.
    4. Draw a district table that carries both a daily count and a running total.
    5. Draw a facility table whose reported ratio is capped and whose parts do not sum.
       The ratio is taken against total beds, so it never exceeds 100 and the cap only
       hides the top half-percent.
    6. Print the ground truth for prices, holdings, districts and facilities, so later
       scripts can be scored against it.
"""

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
SEED = 20260828

# Four invented instruments. Nothing here is a real listed company.
INSTRUMENTS = {
    "ARB": {"name": "Arbor Technologies", "start": 142.0, "drift": 0.00042, "vol": 0.0165},
    "CLD": {"name": "Calder Energy", "start": 58.5, "drift": -0.00011, "vol": 0.0231},
    "MRD": {"name": "Meridian Foods", "start": 91.2, "drift": 0.00018, "vol": 0.0104},
    "SVN": {"name": "Severn Logistics", "start": 27.4, "drift": 0.00035, "vol": 0.0192},
}
MARKET_START = date(2023, 1, 2)
MARKET_END = date(2024, 12, 31)

# One instrument stops trading for eleven weekdays, so the four series do not have the
# same number of rows.
HALT_TICKER = "CLD"
HALT_START = date(2024, 5, 13)
HALT_DAYS = 11

# Each instrument gets one sharp move, so a band drawn from a rolling mean and
# standard deviation has something to flag.
SHOCK_DAY_INDEX = {"ARB": 168, "CLD": 402, "MRD": 291, "SVN": 96}
SHOCK_SIZE = {"ARB": 0.091, "CLD": -0.118, "MRD": 0.067, "SVN": -0.083}
SHOCK_LENGTH = 3

CUSTOMER_ROWS = 10000
CUSTOMER_OPEN_FIRST = date(2019, 1, 1)
CUSTOMER_OPEN_LAST = date(2024, 12, 31)

# Base probability that a customer holds each product at all.
HOLDING_BASE = {"deposit": 0.93, "wealth": 0.34, "fund": 0.22, "insurance": 0.17}
# Holding a wealth product makes a fund holding far more likely. This is the one
# positive link an association rule should recover.
WEALTH_TO_FUND_MULTIPLIER = 2.6
# Holding insurance is mildly discouraged among fund holders.
FUND_TO_INSURANCE_MULTIPLIER = 0.72

STAFF_ROWS = 480
REVIEW_YEARS = (2023, 2024)
REVIEW_QUARTERS = (1, 2, 3, 4)
DEPARTMENTS = ["Claims", "Underwriting", "Operations", "Compliance", "Technology"]
GRADES = ["Associate", "Senior", "Lead", "Principal"]

DISTRICT_YEAR = 2024
DISTRICTS = [
    "Ashfield", "Barrowgate", "Clifton Vale", "Dunmore", "Eastmoor", "Fenwick",
    "Granthorpe", "Halstead", "Inverleith", "Kirkburn", "Langmere", "Marchwood",
    "Netherby", "Oakhaven", "Pinecrest", "Quarryside", "Rosslare", "Thornbury",
]

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
    The walk is multiplicative, so a price never goes negative."""
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
        # The spread is added outside both open and close, so high and low always bracket them.
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
    """Write the prices into SQLite, deleting any old file so a rerun does not double it."""
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
    """Draw customers whose fund holding depends on whether they hold a wealth product.
    Assets are lognormal, so a small share sits above one million."""
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
    Later hires have fewer reviews and lower salaries, so a join at review grain shifts
    the average salary."""
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
    """Draw a per-district daily table holding both a daily count and running totals.
    Only new_cases can be summed across rows."""
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
    """Draw a monthly bed table with a rounded ratio clamped at 99, and beds out of service.
    Those beds are in neither occupied nor free, so the parts fall short of the total."""
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
    print("\n--- 6. Ground truth: yearly move per instrument ---")
    print("Script 04 asks a model for the 2024 moves. These are the answers.")
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
    calendar = business_days(MARKET_START, MARKET_END)
    print(f"Planted shocks, {SHOCK_LENGTH} weekdays each:")
    for ticker, start in SHOCK_DAY_INDEX.items():
        print(f"    {ticker:<8}{calendar[start].isoformat()} to "
              f"{calendar[start + SHOCK_LENGTH - 1].isoformat()}  {SHOCK_SIZE[ticker]:+.1%}")


def report_customer_truth(customers: pd.DataFrame) -> None:
    """Print the holding rates and the co-holding structure that was planted."""
    print("\n--- 6. Ground truth: product holdings ---")
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
    print("    Script 08 counts rules both ways: per customer and per combination.")

    above_million = (customers["total_aum"] >= 1_000_000).mean()
    print(f"\n    share with total_aum >= 1,000,000 {above_million:.4f}")


def report_district_truth(districts: pd.DataFrame) -> None:
    """Print the total that is correct and the total that adding the wrong column gives."""
    print("\n--- 6. Ground truth: district totals ---")
    true_total = int(districts["new_cases"].sum())
    max_of_cumulative = int(districts.groupby("district")["cumulative_cases"].max().sum())
    sum_of_cumulative = int(districts["cumulative_cases"].sum())
    print(f"    sum of new_cases                        {true_total:>12,}")
    print(f"    sum of per-district max(cumulative)     {max_of_cumulative:>12,}")
    print(f"    sum of cumulative_cases over all rows   {sum_of_cumulative:>12,}"
          f"   ({sum_of_cumulative / true_total:.1f}x the truth)")


def report_facility_truth(facilities: pd.DataFrame) -> None:
    """Print how often the reported ratio reads the cap value and how often the parts fall short.
    Reading 99 is not the same as being clamped; script 03 separates the two."""
    print("\n--- 6. Ground truth: facility beds ---")
    rows_at_cap = (facilities["reported_utilization_pct"] >= REPORTED_RATIO_CAP).sum()
    parts_short = (facilities["occupied_beds"] + facilities["free_beds"]
                   < facilities["total_beds"]).sum()
    print(f"    rows                                  {len(facilities):>8,}")
    print(f"    rows reading exactly {REPORTED_RATIO_CAP}%              {rows_at_cap:>8,}")
    gap = facilities["total_beds"] - facilities["occupied_beds"] - facilities["free_beds"]
    explained = bool((gap == facilities["out_of_service_beds"]).all())
    print(f"    rows where occupied + free < total    {parts_short:>8,}")
    print(f"    In every row the shortfall equals out_of_service_beds: {explained}.")
    print("    Those rows hold beds counted in neither column; the figure counts rows, not beds.")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Daily prices
    print("--- 1. Daily prices for four instruments ---")
    market = build_market_table(rng)
    db_path = write_market_db(market)
    print(f"{len(market):,} rows across {market['ticker'].nunique()} instruments -> {db_path.name}")

    # 2. Customer base
    print("\n--- 2. Customer base ---")
    customers = build_customers(rng)
    customers.to_csv(DATA / "customers.csv", index=False)
    print(f"{len(customers):,} rows -> customers.csv")

    # 3. Staff master and quarterly reviews
    print("\n--- 3. Staff master and quarterly reviews ---")
    staff, reviews = build_staff(rng)
    staff.to_csv(DATA / "staff.csv", index=False)
    reviews.to_csv(DATA / "staff_reviews.csv", index=False)
    print(f"{len(staff):,} rows -> staff.csv")
    per_employee = reviews.groupby("staff_id").size()
    print(f"{len(reviews):,} rows -> staff_reviews.csv "
          f"({per_employee.mean():.1f} per employee on average, "
          f"{per_employee.min()} to {per_employee.max()}; "
          f"{int((per_employee == per_employee.max()).sum())} of {len(staff)} have all "
          f"{per_employee.max()})")

    # 4. District daily counts
    print("\n--- 4. District daily counts ---")
    districts = build_districts(rng)
    districts.to_csv(DATA / "district_daily.csv", index=False)
    print(f"{len(districts):,} rows -> district_daily.csv")

    # 5. Facility bed occupancy
    print("\n--- 5. Facility bed occupancy ---")
    facilities = build_facilities(rng)
    facilities.to_csv(DATA / "facility_beds.csv", index=False)
    print(f"{len(facilities):,} rows -> facility_beds.csv")

    # 6. Ground truth
    report_market_truth(market)
    report_customer_truth(customers)
    report_district_truth(districts)
    report_facility_truth(facilities)

    print(f"\nAll five sources written to {DATA}")
    print(f"The generator is seeded with {SEED}, so rerunning reproduces them exactly.")


if __name__ == "__main__":
    main()
