"""Read one file two ways, get the same shape twice, and only one of them is the data.

Demonstrates why a clean-looking profile is not evidence that the load was correct:
    1. Load the listings with a single-space separator and with a whitespace-run separator.
    2. Compare the two frames on the things people usually check, which agree.
    3. Check a quantity that can be falsified instead, and watch the two disagree.
    4. Show the mechanism on the first affected row, field by field.
    5. Count how much of the file is affected, and what the wrong read does to the missing-value report.
    6. Profile the correct frame, and check the profile against the generator that wrote it.

Module 07: Machine Learning and Deep Learning Foundations - Load Auditing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
LISTINGS = DATA / "vehicle_listings.csv"

# What the file is known to contain, from the generator in script 01. These are
# the checks that can fail; row counts and column counts cannot.
EXPECTED_GEARBOX_VALUES = 2
EXPECTED_DAMAGE_VALUES = 2
EXPECTED_OFFER_TYPE_VALUES = 1
# v_0 to v_4 were paid into the price; v_5 to v_14 are noise columns.
PRICED_LATENTS = 5
TOTAL_LATENTS = 15


def load_both_ways(path):
    """Return the frame read with one-space and with whitespace-run separators."""
    correct = pd.read_csv(path, sep=" ")
    collapsed = pd.read_csv(path, sep=r"\s+", engine="python")
    return correct, collapsed


def compare_the_usual_checks(correct, collapsed):
    """Print the checks that agree, which is exactly why the bug survives."""
    print(f"    shape                 {correct.shape} against {collapsed.shape}")
    print(f"    column names equal    {list(correct.columns) == list(collapsed.columns)}")
    print(f"    first column sum      {correct['listing_id'].sum()} against "
          f"{collapsed['listing_id'].sum()}")
    print(f"    price mean            {correct['price'].mean():.2f} against "
          f"{collapsed['price'].mean():.2f}")
    print("    no exception raised by either read")
    print("    The price means are not identical, but nothing about a 13 unit gap")
    print("    on a 12000 unit average asks to be investigated.")


def compare_a_falsifiable_check(correct, collapsed):
    """Print counts that have a known right answer, where the two reads split."""
    rows = [
        ("gearbox", EXPECTED_GEARBOX_VALUES),
        ("damage_flag", EXPECTED_DAMAGE_VALUES),
        ("offer_type", EXPECTED_OFFER_TYPE_VALUES),
    ]
    print(f"    {'column':<14}{'expected':>10}{'one space':>12}{'whitespace run':>16}")
    for column, expected in rows:
        print(f"    {column:<14}{expected:>10}{correct[column].nunique():>12}"
              f"{collapsed[column].nunique():>16}")


def show_the_mechanism(path, correct, collapsed):
    """Print one affected row under both readings, side by side."""
    holed = correct[correct[["body_type", "fuel_type", "gearbox"]].isna().any(axis=1)]
    index = holed.index[0]
    raw = path.read_text(encoding="utf-8").splitlines()[index + 1]

    print(f"    row {index} in the file reads:")
    print(f"        {raw[:78]}")
    print(f"    it holds {len(raw.split(' '))} fields on one space, "
          f"{len(raw.split())} on a whitespace run\n")

    columns = ["body_type", "fuel_type", "gearbox", "power", "odometer_km",
               "damage_flag", "region_code"]
    header = "    " + "".join(f"{c:>14}" for c in columns)
    print(header)
    print("    " + "".join(f"{str(correct.loc[index, c]):>14}" for c in columns)
          + "   one space")
    print("    " + "".join(f"{str(collapsed.loc[index, c]):>14}" for c in columns)
          + "   whitespace run")
    print("\n    The empty field is not read as missing, it is not read at all.")
    print("    Every column to its right slides one place left, all the way to price.")


def count_the_damage(correct, collapsed):
    """Report how many rows moved, and where the missing values ended up."""
    moved = int((correct["region_code"] != collapsed["region_code"]).sum())
    print(f"    rows whose columns shifted      {moved} of {len(correct)} "
          f"({moved / len(correct):.2%})")

    total_correct = int(correct.isna().sum().sum())
    total_collapsed = int(collapsed.isna().sum().sum())
    print(f"    missing values, one space       {total_correct}")
    print(f"    missing values, whitespace run  {total_collapsed}")
    print("    The totals match, so a missing-value count catches nothing either.")
    print("    What changed is which column the gaps are in:\n")

    print(f"    {'column':<14}{'one space':>12}{'whitespace run':>16}")
    columns = sorted(set(correct.columns[correct.isna().any()])
                     | set(collapsed.columns[collapsed.isna().any()]))
    for column in columns:
        print(f"    {column:<14}{int(correct[column].isna().sum()):>12}"
              f"{int(collapsed[column].isna().sum()):>16}")
    print("\n    The shift runs to the end of the row, so under the wrong read the")
    print("    gaps land in the last column. That column is price, the label.")


def profile_the_correct_frame(correct):
    """Profile the frame that was read properly, and check it against the generator."""
    missing = correct.isna().sum()
    missing = missing[missing > 0]
    print("    columns with gaps:")
    for column, count in missing.items():
        print(f"        {column:<12}{count:>6} rows  ({count / len(correct):.2%})")

    price = correct["price"]
    print(f"\n    price: min {price.min()}, median {price.median():.0f}, "
          f"max {price.max()}, skew {price.skew():.3f}")

    correlations = correct[[f"v_{i}" for i in range(TOTAL_LATENTS)] + ["price"]].corr()["price"]
    ranked = correlations.drop("price").abs().sort_values(ascending=False)
    top = list(ranked.index[:PRICED_LATENTS])
    paid = [f"v_{i}" for i in range(PRICED_LATENTS)]
    print(f"\n    five strongest latent correlations: {top}")
    print(f"    the five that were paid into the price: {paid}")
    print(f"    they match: {sorted(top) == sorted(paid)}")
    print(f"    strongest noise column: {ranked.index[PRICED_LATENTS]} at "
          f"{ranked.iloc[PRICED_LATENTS]:.4f}")


def main():
    if not LISTINGS.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    print("--- 1. The same file, two separators ---")
    correct, collapsed = load_both_ways(LISTINGS)
    print(f"    pd.read_csv(path, sep=' ')            -> {correct.shape}")
    print(f"    pd.read_csv(path, sep=r'\\s+')         -> {collapsed.shape}")

    print("\n--- 2. The checks people run, which agree ---")
    compare_the_usual_checks(correct, collapsed)

    print("\n--- 3. A check that can be falsified, which does not agree ---")
    compare_a_falsifiable_check(correct, collapsed)
    print("\n    gearbox is manual or automatic. A reading that finds hundreds of")
    print("    distinct values for it has not found a surprise in the data.")

    print("\n--- 4. The mechanism, on the first affected row ---")
    show_the_mechanism(LISTINGS, correct, collapsed)

    print("\n--- 5. How much of the file this touches ---")
    count_the_damage(correct, collapsed)

    print("\n--- 6. Profiling the frame that was read properly ---")
    profile_the_correct_frame(correct)

    print("\nThe two reads differ on no shape, no column name and no exception.")
    print("They differ on a count with a known right answer, which is why the")
    print("profile has to include at least one of those before anything is trained.")


if __name__ == "__main__":
    main()
