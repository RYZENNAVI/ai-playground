"""Mine the same product holdings three times, changing only what one row is taken to mean.

Demonstrates that an association rule reports on the sample unit, not on the customers:
    1. Turn the customer table into one basket per customer and count what is in them.
    2. Mine frequent itemsets and rules from those baskets with a small implementation.
    3. Drop duplicate baskets, the way a table of combinations is usually prepared.
    4. Mine the deduplicated table and read the supports that come back.
    5. Restore the counts as weights and confirm the first result comes back.
    6. Put the lift of every rule side by side under the three sample units.
    7. Check the strongest rule against the relationship the data was built with.

Module 10: Applied Projects - Sample Units in Association Mining.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"

PRODUCTS = {
    "deposit": "deposit_balance",
    "wealth": "wealth_balance",
    "fund": "fund_balance",
    "insurance": "insurance_balance",
}
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.30
MAX_ITEMSET_SIZE = 3


def build_baskets() -> pd.DataFrame:
    """Turn each customer into one row of true and false, one column per product.

    A basket is a customer here, because the question is which products a person
    holds together. Naming the unit out loud is the whole point of this script: every
    number below is a proportion of something, and this is the something.
    """
    path = DATA / "customers.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path.name}. Run 01_build_project_datasets.py first.")
    customers = pd.read_csv(path)
    return pd.DataFrame(
        {name: customers[column] > 0 for name, column in PRODUCTS.items()}
    )


def frequent_itemsets(baskets: pd.DataFrame, weights: np.ndarray,
                      min_support: float) -> pd.DataFrame:
    """Return every product combination whose weighted support clears the threshold.

    Support is the share of total weight in which every item of the set appears. The
    weights argument is what lets the same routine express all three sample units:
    ones for one row per basket, ones again for one row per distinct combination, and
    the observed counts for the deduplicated table restored to its real proportions.
    """
    total = float(weights.sum())
    columns = list(baskets.columns)
    rows = []
    for size in range(1, MAX_ITEMSET_SIZE + 1):
        for items in combinations(columns, size):
            present = np.ones(len(baskets), dtype=bool)
            for item in items:
                present &= baskets[item].to_numpy()
            support = float(weights[present].sum()) / total
            if support >= min_support:
                rows.append({"items": frozenset(items), "size": size, "support": support})
    return pd.DataFrame(rows)


def association_rules(itemsets: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Split every frequent itemset into antecedent and consequent, and score the split.

    Confidence is how often the consequent shows up among the baskets that already
    hold the antecedent. Lift compares that against how often the consequent shows up
    at all, so lift is the number that says whether the two are related rather than
    merely both common. Lift of one means no relationship was found.
    """
    support_of = dict(zip(itemsets["items"], itemsets["support"]))
    rows = []
    for items, support in support_of.items():
        if len(items) < 2:
            continue
        for size in range(1, len(items)):
            for antecedent in combinations(sorted(items), size):
                antecedent = frozenset(antecedent)
                consequent = items - antecedent
                if antecedent not in support_of or consequent not in support_of:
                    continue
                confidence = support / support_of[antecedent]
                if confidence < min_confidence:
                    continue
                rows.append({
                    "antecedent": " + ".join(sorted(antecedent)),
                    "consequent": " + ".join(sorted(consequent)),
                    "support": support,
                    "confidence": confidence,
                    "lift": confidence / support_of[consequent],
                })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("lift", ascending=False).reset_index(drop=True)


def print_itemsets(itemsets: pd.DataFrame, limit: int = 8) -> None:
    """Print the frequent itemsets, largest support first."""
    ordered = itemsets.sort_values("support", ascending=False)
    print(f"        {'itemset':<34}{'size':>6}{'support':>12}")
    for row in ordered.head(limit).itertuples():
        label = " + ".join(sorted(row.items))
        print(f"        {label:<34}{row.size:>6}{row.support:>12.6f}")
    if len(ordered) > limit:
        print(f"        ... {len(ordered) - limit} more")


def print_rules(rules: pd.DataFrame, limit: int = 6) -> None:
    """Print the strongest rules by lift."""
    if rules.empty:
        print("        no rule cleared the thresholds")
        return
    print(f"        {'rule':<38}{'support':>10}{'confidence':>13}{'lift':>9}")
    for row in rules.head(limit).itertuples():
        label = f"{row.antecedent} -> {row.consequent}"
        print(f"        {label:<38}{row.support:>10.4f}{row.confidence:>13.4f}{row.lift:>9.4f}")
    if len(rules) > limit:
        print(f"        ... {len(rules) - limit} more")


def main() -> None:
    baskets = build_baskets()

    print("--- 1. One basket per customer ---")
    print(f"    baskets                {len(baskets):>8,}")
    print(f"    products               {len(baskets.columns):>8}")
    for product in baskets.columns:
        print(f"        holds {product:<12}{baskets[product].mean():>8.4f}")
    distinct = baskets.drop_duplicates()
    print(f"\n    distinct combinations  {len(distinct):>8}  "
          f"out of {2 ** len(baskets.columns)} possible")

    print("\n--- 2. Mined with one row per customer ---")
    weights_all = np.ones(len(baskets))
    itemsets_all = frequent_itemsets(baskets, weights_all, MIN_SUPPORT)
    rules_all = association_rules(itemsets_all, MIN_CONFIDENCE)
    print_itemsets(itemsets_all)
    print()
    print_rules(rules_all)

    print("\n--- 3-4. Mined after dropping duplicate baskets ---")
    print(f"    {len(baskets):,} rows collapse to {len(distinct)} rows.")
    print("    The table still holds every combination that occurs. What it no longer")
    print("    holds is how many customers each combination stands for.")
    weights_distinct = np.ones(len(distinct))
    itemsets_distinct = frequent_itemsets(distinct, weights_distinct, MIN_SUPPORT)
    rules_distinct = association_rules(itemsets_distinct, MIN_CONFIDENCE)
    print()
    print_itemsets(itemsets_distinct)
    print()
    print_rules(rules_distinct)

    singles = itemsets_distinct[itemsets_distinct["size"] == 1]["support"]
    if len(singles) and np.allclose(singles, 0.5):
        print("\n    Every single-product support is exactly 0.5, and it has to be: with all")
        print(f"    {len(distinct)} combinations present once each, every product is in half of them.")
        print("    Every pair then supports 0.25, which is 0.5 x 0.5, so every lift is")
        print("    exactly 1. The deduplicated table is independent by construction, and")
        print("    that result would be identical whatever the customers actually did.")

    print("\n--- 5. The deduplicated table with its counts restored ---")
    counted = (
        baskets.groupby(list(baskets.columns), as_index=False)
        .size()
        .rename(columns={"size": "customers"})
    )
    weights_counted = counted["customers"].to_numpy(dtype=float)
    itemsets_counted = frequent_itemsets(
        counted[list(baskets.columns)], weights_counted, MIN_SUPPORT
    )
    rules_counted = association_rules(itemsets_counted, MIN_CONFIDENCE)
    print(f"    {len(counted)} rows carrying a customer count each, "
          f"{int(weights_counted.sum()):,} customers in total")
    print(f"    largest group {int(weights_counted.max()):,} customers, "
          f"smallest {int(weights_counted.min()):,}")
    matches = (
        len(rules_counted) == len(rules_all)
        and np.allclose(
            rules_counted.sort_values(["antecedent", "consequent"])["lift"].to_numpy(),
            rules_all.sort_values(["antecedent", "consequent"])["lift"].to_numpy(),
        )
    )
    print(f"    rules identical to the one-row-per-customer result: {matches}")
    print("    The fix is not to avoid deduplicating. It is to carry the count.")

    print("\n--- 6. Lift under the three sample units ---")
    key = ["antecedent", "consequent"]
    merged = (
        rules_all[key + ["lift"]].rename(columns={"lift": "per customer"})
        .merge(rules_distinct[key + ["lift"]].rename(columns={"lift": "per combination"}),
               on=key, how="outer")
        .merge(rules_counted[key + ["lift"]].rename(columns={"lift": "counts restored"}),
               on=key, how="outer")
    )
    merged = merged.sort_values("per customer", ascending=False)
    print(f"    {'rule':<38}{'per customer':>14}{'per combination':>17}{'counts restored':>17}")
    for row in merged.head(8).itertuples():
        label = f"{row.antecedent} -> {row.consequent}"
        values = []
        for value in (row[3], row[4], row[5]):
            values.append(f"{value:.4f}" if pd.notna(value) else "-")
        print(f"    {label:<38}{values[0]:>14}{values[1]:>17}{values[2]:>17}")

    spread_distinct = rules_distinct["lift"].max() - rules_distinct["lift"].min()
    spread_all = rules_all["lift"].max() - rules_all["lift"].min()
    print(f"\n    lift ranges over {spread_all:.4f} per customer and "
          f"{spread_distinct:.4f} per combination")

    print("\n--- 7. Against the relationship the data was built with ---")
    wealth = baskets["wealth"].to_numpy()
    fund = baskets["fund"].to_numpy()
    with_wealth = fund[wealth].mean()
    without_wealth = fund[~wealth].mean()
    measured = (fund & wealth).mean() / (fund.mean() * wealth.mean())
    print(f"    P(fund | wealth)      {with_wealth:.4f}")
    print(f"    P(fund | not wealth)  {without_wealth:.4f}")
    print(f"    lift(wealth -> fund)  {measured:.4f}")
    found = rules_all[
        (rules_all["antecedent"] == "wealth") & (rules_all["consequent"] == "fund")
    ]
    if not found.empty:
        print(f"    the mined rule reports  {found['lift'].iloc[0]:.4f}   (match: "
              f"{np.isclose(found['lift'].iloc[0], measured)})")
    deduped = rules_distinct[
        (rules_distinct["antecedent"] == "wealth") & (rules_distinct["consequent"] == "fund")
    ]
    if not deduped.empty:
        print(f"    the deduplicated table reports {deduped['lift'].iloc[0]:.4f}")
    print("\n    01_build_project_datasets.py prints this same relationship as the number")
    print("    it drew the data with. Recovering it is what tells you the mining worked,")
    print("    and failing to recover it is what the deduplicated run should have shown.")


if __name__ == "__main__":
    main()
