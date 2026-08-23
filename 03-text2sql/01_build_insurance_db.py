"""Build the local SQLite database every other script in this module queries.

Demonstrates how to prepare a schema a language model can actually read:
    1. Define five related tables with typed columns and column comments.
    2. Generate deterministic sample rows so results are reproducible.
    3. Write the database to disk only when it is missing or out of date.
    4. Export the CREATE TABLE statements that later scripts feed to the model.
    5. Print a summary so the data is visible before any model sees it.

Module 03: Text2SQL - Local Database Setup.
"""

import os
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "insurance.db"
SCHEMA_PATH = DATA_DIR / "schema.sql"

# A fixed seed keeps every run identical. Text2SQL is judged by comparing a
# generated query against a known answer, so the data behind that answer has to
# stop moving.
SEED = 20260822

# Column comments live in the schema itself rather than in a side file. The
# model reads the CREATE TABLE text verbatim, so anything it needs to know
# about a column has to survive inside that text.
#
# The status columns store short codes rather than readable words, the way
# production systems usually do. A model asked "which claims were turned down"
# has no way to reach 'DEN' from the column name alone - it has to be told. That
# is what makes the comments load-bearing instead of decorative, and it is what
# script 02 measures.
SCHEMA = """
CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,        -- unique customer number
    name            TEXT    NOT NULL,           -- full name
    gender          TEXT    NOT NULL,           -- one of: Male, Female
    date_of_birth   DATE    NOT NULL,           -- used for age filters
    marital_status  TEXT    NOT NULL,           -- one of: Married, Single, Divorced
    occupation      TEXT    NOT NULL,
    phone_number    TEXT    NOT NULL,
    email           TEXT    NOT NULL,
    city            TEXT    NOT NULL,
    registered_on   DATE    NOT NULL,           -- date the customer signed up
    customer_status TEXT    NOT NULL            -- stored as a code: A = active, L = lapsed, C = closed
);

CREATE TABLE products (
    product_id        INTEGER PRIMARY KEY,      -- unique product number
    product_name      TEXT    NOT NULL,
    product_type      TEXT    NOT NULL,         -- one of: Term Life, Whole Life, Health, Accident, Travel
    coverage_amount   INTEGER NOT NULL,         -- payout ceiling, in whole currency units
    coverage_years    INTEGER NOT NULL,         -- length of cover
    premium           REAL    NOT NULL,         -- price paid per payment period
    payment_frequency TEXT    NOT NULL,         -- one of: Monthly, Quarterly, Annual
    sales_region      TEXT    NOT NULL,         -- one of: North, South, East, West
    product_status    TEXT    NOT NULL          -- one of: Active, Retired
);

CREATE TABLE policies (
    policy_number    INTEGER PRIMARY KEY,       -- unique policy number
    customer_id      INTEGER NOT NULL,          -- joins to customers.customer_id
    product_id       INTEGER NOT NULL,          -- joins to products.product_id
    policy_status    TEXT    NOT NULL,          -- stored as a code: IF = in force, LP = lapsed, TM = terminated
    beneficiary      TEXT    NOT NULL,
    relationship     TEXT    NOT NULL,          -- one of: Spouse, Child, Parent, Sibling
    start_date       DATE    NOT NULL,
    end_date         DATE    NOT NULL,
    payment_status   TEXT    NOT NULL,          -- stored as a code: P = paid, NP = not paid
    payment_method   TEXT    NOT NULL,          -- one of: Card, Transfer, Direct Debit
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id)  REFERENCES products (product_id)
);

CREATE TABLE daily_sales (
    sale_date        DATE    PRIMARY KEY,       -- one row per day
    new_count        INTEGER NOT NULL,          -- policies sold to brand-new customers
    renewal_count    INTEGER NOT NULL,          -- policies renewed by existing customers
    upgrade_count    INTEGER NOT NULL,          -- existing customers moving to a larger plan
    campaign         TEXT    NOT NULL,          -- one of: none, spring, autumn, yearend
    is_month_end     INTEGER NOT NULL,          -- 1 on the last three days of a month
    weekday          TEXT    NOT NULL,          -- Mon .. Sun
    total_premium    REAL    NOT NULL           -- premium collected that day, all segments
);

CREATE TABLE claims (
    claim_number  INTEGER PRIMARY KEY,          -- unique claim number
    policy_number INTEGER NOT NULL,             -- joins to policies.policy_number
    claim_date    DATE    NOT NULL,
    claim_type    TEXT    NOT NULL,             -- one of: Medical, Accident, Death, Property
    claim_amount  REAL    NOT NULL,             -- amount requested
    claim_status  TEXT    NOT NULL,             -- stored as a code: APP = approved, PND = pending, PAY = paid, DEN = denied
    handler       TEXT    NOT NULL,             -- staff member who reviewed the claim
    review_date   DATE,                         -- null while the claim is unreviewed
    denial_reason TEXT,                         -- null unless claim_status is Denied
    FOREIGN KEY (policy_number) REFERENCES policies (policy_number)
);
"""

FIRST_NAMES = [
    "Alice", "Brian", "Carla", "Daniel", "Elena", "Felix", "Grace", "Henry",
    "Iris", "Jonas", "Karen", "Liam", "Maya", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Samuel", "Tessa", "Ulric", "Vera", "Wesley", "Xenia",
]
LAST_NAMES = [
    "Adams", "Bennett", "Clarke", "Doyle", "Evans", "Fisher", "Gordon",
    "Hughes", "Ingram", "Jenkins", "Keller", "Lawson", "Mercer", "Newton",
]
OCCUPATIONS = [
    "Teacher", "Engineer", "Nurse", "Accountant", "Chef", "Driver",
    "Designer", "Analyst", "Electrician", "Pharmacist",
]
CITIES = ["Riverton", "Kingsford", "Ashbury", "Northgate", "Westbrook", "Eastvale"]
HANDLERS = ["M. Doyle", "S. Ingram", "T. Keller", "R. Mercer"]

PRODUCTS = [
    (1, "SecureLife Term 20", "Term Life", 250000, 20, 480.00, "Annual", "North", "Active"),
    (2, "SecureLife Term 30", "Term Life", 400000, 30, 720.00, "Annual", "South", "Active"),
    (3, "Heritage Whole Life", "Whole Life", 300000, 99, 1850.00, "Annual", "East", "Active"),
    (4, "Heritage Legacy Plus", "Whole Life", 500000, 99, 2600.00, "Annual", "West", "Retired"),
    (5, "CareFirst Standard", "Health", 80000, 1, 96.00, "Monthly", "North", "Active"),
    (6, "CareFirst Premier", "Health", 150000, 1, 175.00, "Monthly", "East", "Active"),
    (7, "SafeStep Accident", "Accident", 60000, 1, 45.00, "Quarterly", "South", "Active"),
    (8, "SafeStep Family", "Accident", 120000, 1, 78.00, "Quarterly", "West", "Active"),
    (9, "Voyager Travel Cover", "Travel", 30000, 1, 22.00, "Monthly", "East", "Active"),
    (10, "Voyager Travel Plus", "Travel", 55000, 1, 38.00, "Monthly", "North", "Retired"),
]


def make_customers(rng, count=40):
    """Build customer rows with birth dates spread either side of the age-30 line.

    The spread is deliberate: several benchmark questions filter on age, and a
    dataset where everyone is the same age cannot tell a correct query from a
    query that forgot the filter.
    """
    rows = []
    for i in range(1, count + 1):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        birth = date(1968, 1, 1) + timedelta(days=rng.randint(0, 365 * 34))
        registered = date(2019, 1, 1) + timedelta(days=rng.randint(0, 365 * 6))
        rows.append((
            i,
            f"{first} {last}",
            rng.choice(["Male", "Female"]),
            birth.isoformat(),
            rng.choice(["Married", "Single", "Divorced"]),
            rng.choice(OCCUPATIONS),
            f"555-{rng.randint(1000, 9999)}",
            f"{first.lower()}.{last.lower()}{i}@example.com",
            rng.choice(CITIES),
            registered.isoformat(),
            rng.choices(["A", "L", "C"], weights=[7, 2, 1])[0],
        ))
    return rows


def make_policies(rng, customer_count, count=60):
    """Build policy rows, leaving roughly a fifth unpaid.

    'Find the unpaid policies' is one of the benchmark questions. If every row
    were paid, a query that dropped the WHERE clause would return nothing and
    look correct.
    """
    rows = []
    for i in range(1, count + 1):
        start = date(2020, 1, 1) + timedelta(days=rng.randint(0, 365 * 5))
        product = rng.choice(PRODUCTS)
        rows.append((
            100000 + i,
            rng.randint(1, customer_count),
            product[0],
            rng.choices(["IF", "LP", "TM"], weights=[7, 2, 1])[0],
            f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            rng.choice(["Spouse", "Child", "Parent", "Sibling"]),
            start.isoformat(),
            (start + timedelta(days=365 * product[4] if product[4] < 99 else 365 * 40)).isoformat(),
            rng.choices(["P", "NP"], weights=[8, 2])[0],
            rng.choice(["Card", "Transfer", "Direct Debit"]),
        ))
    return rows


def make_claims(rng, policy_numbers, count=35):
    """Build claim rows spanning the 10000 threshold and every status value.

    Two benchmark questions depend on this spread: one filters on
    claim_amount > 10000, the other on claim_status = 'Pending'.
    """
    rows = []
    for i in range(1, count + 1):
        status = rng.choices(["APP", "PND", "PAY", "DEN"], weights=[3, 3, 3, 1])[0]
        claim_date = date(2022, 1, 1) + timedelta(days=rng.randint(0, 365 * 3))
        reviewed = status != "PND"
        rows.append((
            500000 + i,
            rng.choice(policy_numbers),
            claim_date.isoformat(),
            rng.choice(["Medical", "Accident", "Death", "Property"]),
            round(rng.choice([rng.uniform(500, 9500), rng.uniform(10500, 90000)]), 2),
            status,
            rng.choice(HANDLERS),
            (claim_date + timedelta(days=rng.randint(3, 40))).isoformat() if reviewed else None,
            "Policy lapsed before the incident" if status == "DEN" else None,
        ))
    return rows


# Per-segment premium is not stored anywhere. Only the daily totals are, which
# is the situation the modelling tools in script 06 exist for: the per-segment
# figures have to be recovered from the totals rather than looked up.
SEGMENT_PREMIUM = {"new": 620.0, "renewal": 410.0, "upgrade": 880.0}
CAMPAIGN_LIFT = {"none": 1.0, "spring": 1.15, "autumn": 1.10, "yearend": 1.30}


def make_daily_sales(rng, days=180):
    """Build one row per day, with the total premium implied by the segment mix.

    The totals are generated from fixed per-segment prices plus a campaign lift
    and a little noise. Nothing writes those prices into the table - recovering
    them from the totals is exactly what script 06 has to do.
    """
    rows = []
    start = date(2024, 1, 1)
    for offset in range(days):
        day = start + timedelta(days=offset)
        campaign = rng.choices(
            ["none", "spring", "autumn", "yearend"], weights=[6, 2, 2, 1]
        )[0]
        month_end = 1 if (day + timedelta(days=3)).month != day.month else 0
        new = rng.randint(2, 14)
        renewal = rng.randint(5, 25)
        upgrade = rng.randint(0, 8)
        base = (
            new * SEGMENT_PREMIUM["new"]
            + renewal * SEGMENT_PREMIUM["renewal"]
            + upgrade * SEGMENT_PREMIUM["upgrade"]
        )
        total = base * CAMPAIGN_LIFT[campaign] * rng.uniform(0.97, 1.03)
        rows.append((
            day.isoformat(), new, renewal, upgrade, campaign, month_end,
            day.strftime("%a"), round(total, 2),
        ))
    return rows


def build_database(path):
    """Create the database from scratch and fill it with generated rows.

    The build writes to a temporary file and renames it into place, so an
    interrupted run never leaves a half-populated database that later scripts
    would happily query.
    """
    rng = random.Random(SEED)
    tmp_path = path.with_suffix(".db.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    connection = sqlite3.connect(tmp_path)
    try:
        connection.executescript(SCHEMA)

        customers = make_customers(rng)
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", customers
        )
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", PRODUCTS
        )
        policies = make_policies(rng, len(customers))
        connection.executemany(
            "INSERT INTO policies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", policies
        )
        claims = make_claims(rng, [row[0] for row in policies])
        connection.executemany(
            "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", claims
        )
        connection.executemany(
            "INSERT INTO daily_sales VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            make_daily_sales(rng),
        )
        connection.commit()
    finally:
        connection.close()

    os.replace(tmp_path, path)


def export_schema(path):
    """Write the CREATE TABLE text to a file the other scripts read.

    Later scripts paste this text straight into the prompt. Reading it from the
    database at run time would work too, but a file on disk lets you see exactly
    what the model was shown.
    """
    path.write_text(SCHEMA.strip() + "\n", encoding="utf-8")


def summarise(path):
    """Print row counts and a few sample rows so the data is visible up front."""
    connection = sqlite3.connect(path)
    try:
        print("--- 5. Database summary ---")
        for table in ("customers", "products", "policies", "claims", "daily_sales"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<12} {count:>4} rows")

        print("\n  Sample: three customers")
        for row in connection.execute(
            "SELECT customer_id, name, date_of_birth, marital_status, customer_status "
            "FROM customers ORDER BY customer_id LIMIT 3"
        ):
            print(f"    {row}")

        print("\n  Sample: claims above the 10000 threshold")
        high, total = connection.execute(
            "SELECT SUM(claim_amount > 10000), COUNT(*) FROM claims"
        ).fetchone()
        print(f"    {high} of {total} claims exceed 10000")

        print("\n  Sample: unpaid policies")
        unpaid, total = connection.execute(
            "SELECT SUM(payment_status = 'NP'), COUNT(*) FROM policies"
        ).fetchone()
        print(f"    {unpaid} of {total} policies are unpaid")
    finally:
        connection.close()


def ensure_database(rebuild=False):
    """Return the database path, building it first when it is missing.

    Every other script in this module calls this instead of assuming the file
    exists, which makes the whole module runnable in any order.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if rebuild or not DB_PATH.exists():
        build_database(DB_PATH)
    if rebuild or not SCHEMA_PATH.exists():
        export_schema(SCHEMA_PATH)
    return DB_PATH


def load_schema():
    """Return the CREATE TABLE text, building the database first if needed."""
    ensure_database()
    return SCHEMA_PATH.read_text(encoding="utf-8")


def main():
    print("--- 1. Schema ---")
    print(f"  5 tables defined, {SCHEMA.count('--')} inline column comments")

    print("\n--- 2. Generating rows ---")
    print(f"  seed = {SEED} (fixed, so every run produces the same data)")

    print("\n--- 3. Writing database ---")
    rebuilt = not DB_PATH.exists()
    ensure_database(rebuild=True)
    print(f"  {'created' if rebuilt else 'rebuilt'}: {DB_PATH}")

    print("\n--- 4. Exporting schema ---")
    print(f"  written: {SCHEMA_PATH}")

    print()
    summarise(DB_PATH)


if __name__ == "__main__":
    main()
