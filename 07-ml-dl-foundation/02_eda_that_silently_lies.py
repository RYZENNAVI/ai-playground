"""This script reads one space-separated listings file into pandas two ways and runs
exploratory data analysis (EDA) on both. The two frames have the same shape and an
equally clean profile, and only one of them is the data, so a clean profile is no
evidence that the load was correct.

The run prints 6 parts:
    1. The same file, read with a single-space and with a whitespace-run separator.
    2. The checks people run, which agree.
    3. A check that can be falsified, which does not agree.
    4. The mechanism, on the first affected row.
    5. How much of the file this touches, and which columns the gaps move to.
    6. Profiling the frame that was read properly, checked against script 01's formula.
"""

import sys
from pathlib import Path

import pandas as pd

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
LISTINGS = DATA / "vehicle_listings.csv"

# What the file is known to contain, from the generator in script 01. These are
# the checks that can fail; row counts and column counts cannot.
EXPECTED_GEARBOX_VALUES = 2
EXPECTED_UNDAMAGED_VALUES = 2
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
    gap = abs(correct["price"].mean() - collapsed["price"].mean())
    print(f"    The price means are not identical, but nothing about a {gap:.0f} unit gap")
    print(f"    on a {correct['price'].mean():.0f} unit average asks to be investigated.")


def compare_a_falsifiable_check(correct, collapsed):
    """Print counts that have a known right answer, where the two reads split."""
    rows = [
        ("gearbox", EXPECTED_GEARBOX_VALUES),
        ("undamaged_flag", EXPECTED_UNDAMAGED_VALUES),
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
               "undamaged_flag", "region_code"]
    width = max(len(c) for c in columns) + 2
    header = "    " + "".join(f"{c:>{width}}" for c in columns)
    print(header)
    print("    " + "".join(f"{str(correct.loc[index, c]):>{width}}" for c in columns)
          + "   one space")
    print("    " + "".join(f"{str(collapsed.loc[index, c]):>{width}}" for c in columns)
          + "   whitespace run")
    print("\n    The whitespace-run read skips the empty field, so it never becomes a NaN.")
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
    last = correct.columns[-1]
    two_holes = int(total_correct - moved)
    print("\n    The shift runs to the end of the row, so under the wrong read a row")
    print(f"    with one hole loses its {last}, the label. The {two_holes} rows with two")
    print(f"    holes shift twice and lose {correct.columns[-2]} as well, which is why the")
    print("    wrong read puts its gaps in the last two columns.")


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
    weakest = min(paid, key=lambda column: ranked[column])
    print(f"    weakest paid column: {weakest} at {ranked[weakest]:.4f}")
    print(f"    strongest noise column: {ranked.index[PRICED_LATENTS]} at "
          f"{ranked.iloc[PRICED_LATENTS]:.4f}")


def main():
    if not LISTINGS.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    # 1. The same file, two separators

    print("--- 1. The same file, two separators ---")
    correct, collapsed = load_both_ways(LISTINGS)
    print(f"    pd.read_csv(path, sep=' ')            -> {correct.shape}")
    print(f"    pd.read_csv(path, sep=r'\\s+')         -> {collapsed.shape}")

    # 2. The checks people run, which agree

    print("\n--- 2. The checks people run, which agree ---")
    compare_the_usual_checks(correct, collapsed)

    # 3. A check that can be falsified, which does not agree

    print("\n--- 3. A check that can be falsified, which does not agree ---")
    compare_a_falsifiable_check(correct, collapsed)
    print("\n    gearbox is manual or automatic, so a read that finds hundreds of")
    print("    distinct values for it has misread the file.")

    # 4. The mechanism, on the first affected row

    print("\n--- 4. The mechanism, on the first affected row ---")
    show_the_mechanism(LISTINGS, correct, collapsed)

    # 5. How much of the file this touches

    print("\n--- 5. How much of the file this touches ---")
    count_the_damage(correct, collapsed)

    # 6. Profiling the frame that was read properly

    print("\n--- 6. Profiling the frame that was read properly ---")
    profile_the_correct_frame(correct)

    print("\nThe two reads differ on no shape, no column name and no exception.")
    print("They differ on a count with a known right answer, which is why the")
    print("profile has to include at least one of those before anything is trained.")


if __name__ == "__main__":
    main()
