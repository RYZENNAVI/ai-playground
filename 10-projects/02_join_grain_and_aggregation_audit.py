"""Join two tables at the wrong grain, add up the wrong column, and catch both by counting.

Demonstrates the two arithmetic mistakes that never raise an error:
    1. Print the grain of each table, meaning how many rows one key owns.
    2. Join on the shared key alone and watch the master table multiply.
    3. Narrow the right-hand table to one row per key first, then join again.
    4. Aggregate to one row per key as the second way to fix the same problem.
    5. Read a table that carries both a daily count and a running total.
    6. Add up each of them three ways and compare the totals against each other.
    7. Rank the districts under each total, and see which ranking moves.

Module 10: Applied Projects - Join Grain and Aggregation.
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"

TARGET_YEAR = 2024
TARGET_QUARTER = 4
TOP_N = 8


def load() -> tuple:
    """Read the three files 01 wrote, and stop early with a clear message if they are missing."""
    needed = ["staff.csv", "staff_reviews.csv", "district_daily.csv"]
    missing = [name for name in needed if not (DATA / name).exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Run 01_build_project_datasets.py first.")
    return (
        pd.read_csv(DATA / "staff.csv"),
        pd.read_csv(DATA / "staff_reviews.csv"),
        pd.read_csv(DATA / "district_daily.csv"),
    )


def describe_grain(frame: pd.DataFrame, key: str, label: str) -> int:
    """Print how many rows share a single key value, which is what grain means.

    A join is only safe when at least one side has exactly one row per key. Reading
    the column names does not tell you that; counting does.
    """
    rows = len(frame)
    keys = frame[key].nunique()
    per_key = rows / keys
    print(f"    {label:<22} {rows:>6,} rows  {keys:>6,} distinct {key}  "
          f"{per_key:>5.1f} rows per key")
    return keys


def naive_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Join on the shared key and nothing else, which is the form that looks correct.

    Nothing about this call is malformed. Pandas does exactly what it was asked:
    it pairs every left row with every matching right row. The damage is that any
    later average over a staff column now weights each employee by their row count.
    """
    return staff.merge(reviews, on="staff_id", how="left")


def narrow_then_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Cut the review table down to one row per employee before joining.

    This is the fix to use when a specific period is what the question asks about.
    The filter has to come first: filtering after the join means the wrong-grain
    table already existed, and any intermediate written out from it is wrong.
    """
    one_quarter = reviews[
        (reviews["review_year"] == TARGET_YEAR)
        & (reviews["review_quarter"] == TARGET_QUARTER)
    ]
    merged = staff.merge(
        one_quarter[["staff_id", "review_score"]], on="staff_id", how="left"
    )
    return merged.rename(columns={"review_score": f"score_{TARGET_YEAR}Q{TARGET_QUARTER}"})


def aggregate_then_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Collapse the review table to one row per employee, then join.

    This is the fix to use when the question is about the whole period rather than
    one slice of it. Collapsing first is what makes the right-hand side one row per
    key, which is the property the join needed all along.
    """
    per_employee = (
        reviews.groupby("staff_id")
        .agg(reviews_counted=("review_score", "size"),
             mean_score=("review_score", "mean"),
             last_score=("review_score", "last"))
        .reset_index()
    )
    return staff.merge(per_employee, on="staff_id", how="left")


def compare_salary_means(staff, naive, narrowed, aggregated) -> None:
    """Take the same average four ways and print the four answers side by side.

    Average base salary is a property of the master table. It cannot depend on how
    many reviews someone happened to receive. When it does, the join is the reason.
    """
    print("\n    mean base_salary, computed on each table:")
    print(f"        staff master only         {staff['base_salary'].mean():>12,.2f}")
    print(f"        after the naive join      {naive['base_salary'].mean():>12,.2f}")
    print(f"        after narrow-then-join    {narrowed['base_salary'].mean():>12,.2f}")
    print(f"        after aggregate-then-join {aggregated['base_salary'].mean():>12,.2f}")
    drift = naive["base_salary"].mean() - staff["base_salary"].mean()
    print(f"\n    The naive figure is off by {drift:+,.2f}, and the sign is not an accident.")
    print("    Each employee appears once per review they received. Longer-serving people")
    print("    have more reviews and earn more, so the join weights the higher salaries up.")


def department_headcount(staff, naive, aggregated) -> None:
    """Count employees per department on each table, where the count has a known answer."""
    print("\n    headcount per department:")
    truth = staff["department"].value_counts().sort_index()
    inflated = naive["department"].value_counts().sort_index()
    fixed = aggregated["department"].value_counts().sort_index()
    print(f"        {'department':<20}{'truth':>8}{'naive join':>13}{'aggregated':>13}")
    for dept in truth.index:
        print(f"        {dept:<20}{truth[dept]:>8}{inflated[dept]:>13}{fixed[dept]:>13}")
    print(f"        {'total':<20}{truth.sum():>8}{inflated.sum():>13}{fixed.sum():>13}")


def aggregation_totals(districts: pd.DataFrame) -> dict:
    """Add up the case counts three ways, only one of which answers the question asked.

    The question is how many cases were recorded in the year. Adding the daily
    column answers it. Taking the last running total per district answers it too,
    and agreeing with the first is what makes both trustworthy. Adding the running
    total across rows answers a question nobody asked: it sums every day's total
    once per day, so a case recorded in January is counted again every day after.
    """
    return {
        "sum of new_cases": int(districts["new_cases"].sum()),
        "sum of per-district max(cumulative_cases)":
            int(districts.groupby("district")["cumulative_cases"].max().sum()),
        "sum of cumulative_cases across all rows": int(districts["cumulative_cases"].sum()),
    }


def per_district_reconciliation(districts: pd.DataFrame) -> pd.DataFrame:
    """Check district by district that the daily column and the running total agree.

    This is the external check. Two columns written by the same upstream system
    should describe the same thing; where they do not, one of them is broken and
    the total built on it is wrong before any chart is drawn.
    """
    grouped = districts.groupby("district")
    table = pd.DataFrame({
        "from_daily": grouped["new_cases"].sum(),
        "from_running_total": grouped["cumulative_cases"].max(),
        "row_sum_of_running_total": grouped["cumulative_cases"].sum(),
    })
    table["daily_vs_running"] = table["from_daily"] - table["from_running_total"]
    return table.reset_index()


def compare_rankings(table: pd.DataFrame) -> None:
    """Rank the districts under the correct total and under the inflated one.

    A wrong total that only changed the scale would still rank the districts the
    same way, and a chart built on it would still point at the right places. This
    one does not: adding a running total across rows weights a district by how
    early its cases arrived, so an early outbreak outranks a larger late one.
    """
    correct = table.sort_values("from_daily", ascending=False).reset_index(drop=True)
    inflated = table.sort_values("row_sum_of_running_total", ascending=False).reset_index(drop=True)
    correct_rank = {name: i + 1 for i, name in enumerate(correct["district"])}

    print(f"\n    top {TOP_N} districts under each total:")
    print(f"        {'#':<4}{'by sum(new_cases)':<20}{'cases':>10}    "
          f"{'by sum(cumulative)':<20}{'true rank':>10}")
    for i in range(TOP_N):
        left = correct.loc[i, "district"]
        right = inflated.loc[i, "district"]
        print(f"        {i + 1:<4}{left:<20}{correct.loc[i, 'from_daily']:>10,}    "
              f"{right:<20}{correct_rank[right]:>10}")

    moved = sum(
        1 for i in range(len(inflated))
        if correct_rank[inflated.loc[i, "district"]] != i + 1
    )
    print(f"\n    {moved} of {len(inflated)} districts sit at a different rank under the two totals.")


def main() -> None:
    staff, reviews, districts = load()

    print("--- 1. Grain of each table ---")
    print("    A join is safe when one side holds exactly one row per key.")
    describe_grain(staff, "staff_id", "staff master")
    describe_grain(reviews, "staff_id", "quarterly reviews")

    print("\n--- 2. Join on the shared key alone ---")
    naive = naive_join(staff, reviews)
    print(f"    {len(staff):,} master rows joined to reviews -> {len(naive):,} rows "
          f"({len(naive) / len(staff):.1f}x)")
    print("    No warning was raised. The result is a valid table, of the wrong thing.")

    print("\n--- 3. Narrow the right-hand table first ---")
    narrowed = narrow_then_join(staff, reviews)
    print(f"    reviews filtered to {TARGET_YEAR}Q{TARGET_QUARTER} -> {len(narrowed):,} rows")

    print("\n--- 4. Aggregate the right-hand table instead ---")
    aggregated = aggregate_then_join(staff, reviews)
    print(f"    reviews collapsed to one row per employee -> {len(aggregated):,} rows")
    print(f"    reviews counted per employee: "
          f"{aggregated['reviews_counted'].min()} to {aggregated['reviews_counted'].max()}")

    compare_salary_means(staff, naive, narrowed, aggregated)
    department_headcount(staff, naive, aggregated)

    print("\n--- 5. A table carrying a daily count and a running total ---")
    print(f"    {len(districts):,} rows, {districts['district'].nunique()} districts, "
          f"{districts['report_date'].nunique()} days")
    print("    columns:", ", ".join(districts.columns))
    print("    Nothing in those names says which one can be added across rows.")

    print("\n--- 6. The same year total, three ways ---")
    totals = aggregation_totals(districts)
    truth = totals["sum of new_cases"]
    for label, value in totals.items():
        marker = "" if value == truth else f"   {value / truth:>6.1f}x the truth"
        print(f"    {label:<44}{value:>14,}{marker}")

    table = per_district_reconciliation(districts)
    mismatches = int((table["daily_vs_running"] != 0).sum())
    print(f"\n    districts where the daily column and the running total disagree: {mismatches}")
    print("    Agreement between two independently written columns is what makes")
    print("    either of them usable. The third total agrees with nothing.")

    print("\n--- 7. Does the wrong total still rank the same? ---")
    compare_rankings(table)


if __name__ == "__main__":
    main()
