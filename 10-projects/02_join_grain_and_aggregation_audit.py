"""This script makes two pandas mistakes that raise no error, a join at the wrong grain
that multiplies the staff table and a sum over a running-total column, and catches both by
counting rows per key and comparing totals. Steps 1 to 5 are about the join, and steps 6
to 8 are about the running total:
    1. Print the grain of each table, meaning how many rows one key owns.
    2. Join on the shared key alone and watch the master table multiply.
    3. Narrow the reviews to one quarter, one row per employee, then join again.
    4. Aggregate the reviews to one row per employee instead, then join.
    5. Take the mean salary and the headcount per department on every table.
    6. Read a table that carries a daily count next to running totals.
    7. Total the year three ways and compare the totals.
    8. Rank the districts under the right total and the wrong one, and see which
       ranking moves.
"""

import sys
from pathlib import Path

import pandas as pd

# Print UTF-8 even when the output is piped or redirected on Windows.
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


def describe_grain(frame: pd.DataFrame, key: str, label: str) -> None:
    """Print how many rows share one key value, which is what grain means."""
    rows = len(frame)
    keys = frame[key].nunique()
    per_key = rows / keys
    print(f"    {label:<22} {rows:>6,} rows  {keys:>6,} distinct {key}  "
          f"{per_key:>5.1f} rows per key")


def naive_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Join on staff_id alone, which pairs each employee with every review row they have."""
    return staff.merge(reviews, on="staff_id", how="left")


def narrow_then_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Cut the reviews to one quarter, one row per employee, before joining.
    Use this when the question asks about one period."""
    one_quarter = reviews[
        (reviews["review_year"] == TARGET_YEAR)
        & (reviews["review_quarter"] == TARGET_QUARTER)
    ]
    merged = staff.merge(
        one_quarter[["staff_id", "review_score"]], on="staff_id", how="left"
    )
    return merged.rename(columns={"review_score": f"score_{TARGET_YEAR}Q{TARGET_QUARTER}"})


def aggregate_then_join(staff: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Collapse the reviews to one row per employee, then join.
    Use this when the question is about the whole period."""
    per_employee = (
        reviews.groupby("staff_id")
        .agg(reviews_counted=("review_score", "size"))
        .reset_index()
    )
    return staff.merge(per_employee, on="staff_id", how="left")


def compare_salary_means(staff, naive, narrowed, aggregated) -> None:
    """Take the mean base salary on each table. It belongs to the master table,
    so a join that moves it has changed the grain."""
    print("\n    mean base_salary, computed on each table:")
    print(f"        staff master only         {staff['base_salary'].mean():>12,.2f}")
    print(f"        after the naive join      {naive['base_salary'].mean():>12,.2f}")
    print(f"        after narrow-then-join    {narrowed['base_salary'].mean():>12,.2f}")
    print(f"        after aggregate-then-join {aggregated['base_salary'].mean():>12,.2f}")
    drift = naive["base_salary"].mean() - staff["base_salary"].mean()
    print(f"\n    The naive figure is off by {drift:+,.2f}. "
          "Each employee appears once per review,")
    print("    so the naive mean is weighted by review count:")
    counted = aggregated["reviews_counted"]
    full = int(counted.max())
    complete = aggregated[counted == full]
    partial = aggregated[counted < full]
    print(f"        {len(complete)} of {len(aggregated)} employees have the full {full} "
          f"reviews, averaging {complete['base_salary'].mean():>10,.2f}")
    print(f"        the other {len(partial)} have fewer, averaging "
          f"{partial['base_salary'].mean():>17,.2f}")
    print(f"    Everyone hired before the {full} quarters began has all {full} reviews. The "
          f"{len(partial)} with")
    print("    fewer are recent hires, who earn less, so the join down-weights them.")


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
    """Total the year's cases three ways. Summing the running total across rows
    counts a January case again on every later day."""
    return {
        "sum of new_cases": int(districts["new_cases"].sum()),
        "sum of per-district max(cumulative_cases)":
            int(districts.groupby("district")["cumulative_cases"].max().sum()),
        "sum of cumulative_cases across all rows": int(districts["cumulative_cases"].sum()),
    }


def district_totals(districts: pd.DataFrame) -> pd.DataFrame:
    """Total each district from the daily column and from the running total.
    Also take the case-weighted mean day of the year, which shows when cases arrived."""
    day = pd.to_datetime(districts["report_date"]).dt.dayofyear
    weighted = districts.assign(weighted_day=day * districts["new_cases"])
    grouped = weighted.groupby("district")
    table = pd.DataFrame({
        "from_daily": grouped["new_cases"].sum(),
        "row_sum_of_running_total": grouped["cumulative_cases"].sum(),
    })
    table["mean_day"] = grouped["weighted_day"].sum() / table["from_daily"]
    return table.reset_index()


def compare_rankings(table: pd.DataFrame) -> None:
    """Rank the districts under the correct total and the row-summed running total.
    The wrong one weights a district by how early its cases arrived."""
    correct = table.sort_values("from_daily", ascending=False).reset_index(drop=True)
    inflated = table.sort_values("row_sum_of_running_total", ascending=False).reset_index(drop=True)
    correct_rank = {name: i + 1 for i, name in enumerate(correct["district"])}

    print(f"\n    top {TOP_N} districts under each total:")
    print(f"        {'#':<4}{'by sum(new_cases)':<20}{'cases':>10}{'day':>6}    "
          f"{'by sum(cumulative)':<20}{'day':>6}{'true rank':>11}")
    for i in range(TOP_N):
        left = correct.loc[i, "district"]
        right = inflated.loc[i, "district"]
        print(f"        {i + 1:<4}{left:<20}{correct.loc[i, 'from_daily']:>10,}"
              f"{correct.loc[i, 'mean_day']:>6.0f}    "
              f"{right:<20}{inflated.loc[i, 'mean_day']:>6.0f}{correct_rank[right]:>11}")
    print("        day is the case-weighted mean day of the year.")

    moved = sum(
        1 for i in range(len(inflated))
        if correct_rank[inflated.loc[i, "district"]] != i + 1
    )
    print(f"\n    {moved} of {len(inflated)} districts sit at a different rank under the two totals.")


def main() -> None:
    staff, reviews, districts = load()

    # 1. Grain of each table
    print("--- 1. Grain of each table ---")
    print("    Joining onto the master keeps one row per employee only if the other")
    print("    table holds at most one row per key; otherwise rows multiply, which is")
    print("    right only when review-level rows are what the question wants.")
    describe_grain(staff, "staff_id", "staff master")
    describe_grain(reviews, "staff_id", "quarterly reviews")

    # 2. Join on the shared key alone
    print("\n--- 2. Join on the shared key alone ---")
    naive = naive_join(staff, reviews)
    print(f"    {len(staff):,} master rows joined to reviews -> {len(naive):,} rows "
          f"({len(naive) / len(staff):.1f}x)")
    print("    No warning was raised. The result is a valid table, of the wrong thing.")

    # 3. Narrow the right-hand table first
    print("\n--- 3. Narrow the right-hand table first ---")
    narrowed = narrow_then_join(staff, reviews)
    print(f"    reviews filtered to {TARGET_YEAR}Q{TARGET_QUARTER} -> {len(narrowed):,} rows")

    # 4. Aggregate the right-hand table instead
    print("\n--- 4. Aggregate the right-hand table instead ---")
    aggregated = aggregate_then_join(staff, reviews)
    print(f"    reviews collapsed to one row per employee -> {len(aggregated):,} rows")
    print(f"    reviews counted per employee: "
          f"{aggregated['reviews_counted'].min()} to {aggregated['reviews_counted'].max()}")

    # 5. Mean salary and headcount on every table
    print("\n--- 5. Mean salary and headcount on every table ---")
    compare_salary_means(staff, naive, narrowed, aggregated)
    department_headcount(staff, naive, aggregated)

    # 6. A daily count next to running totals
    print("\n--- 6. A table carrying a daily count and running totals ---")
    print(f"    {len(districts):,} rows, {districts['district'].nunique()} districts, "
          f"{districts['report_date'].nunique()} days")
    print("    columns:", ", ".join(districts.columns))
    print("    Only new_cases is a daily count. cumulative_cases, recovered_total and")
    print("    deaths_total are running totals, and active_cases is the level on the day.")

    # 7. The year total, three ways
    print("\n--- 7. The same year total, three ways ---")
    totals = aggregation_totals(districts)
    truth = totals["sum of new_cases"]
    for label, value in totals.items():
        marker = "" if value == truth else f"   {value / truth:>6.1f}x the truth"
        print(f"    {label:<44}{value:>14,}{marker}")

    # 8. Ranking under each total
    print("\n--- 8. Does the wrong total still rank the same? ---")
    compare_rankings(district_totals(districts))


if __name__ == "__main__":
    main()
