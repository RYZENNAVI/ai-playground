"""This script mines association rules (support, confidence and lift) from the customers'
product holdings three times, by counting every combination of up to three products. The
three runs differ only in what one row stands for: a customer, a distinct combination, or a
distinct combination weighted by its customer count.

    1. Turn the customer table into one basket per customer and count what is in them.
    2. Mine frequent itemsets and rules with one row per customer.
    3. Drop duplicate baskets and mine the rows that are left.
    4. Restore the counts as weights and check that the rules match step 2.
    5. Put the lift of every rule side by side under the three sample units.
    6. Check the wealth-to-fund rule against the probabilities 01 draws it from.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Print UTF-8 even when the output is piped or redirected on Windows.
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
# 01 draws fund with probability 0.22, times 2.6 for customers who hold a wealth product.
FUND_BASE_RATE = 0.22
WEALTH_TO_FUND_MULTIPLIER = 2.6


def build_baskets() -> pd.DataFrame:
    """Turn each customer into one row of true and false, one column per product.

    A basket is a customer, because the question is which products a person holds together.
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

    Support is the share of total weight in rows holding every item of the set.
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

    Lift is confidence over the consequent's own support; 1 means the two are independent.
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
    # 1. One basket per customer
    baskets = build_baskets()

    print("--- 1. One basket per customer ---")
    print(f"    baskets                {len(baskets):>8,}")
    print(f"    products               {len(baskets.columns):>8}")
    for product in baskets.columns:
        print(f"        holds {product:<12}{baskets[product].mean():>8.4f}")
    distinct = baskets.drop_duplicates()
    print(f"\n    distinct combinations  {len(distinct):>8}  "
          f"out of {2 ** len(baskets.columns)} possible")

    # 2. Mined with one row per customer
    print("\n--- 2. Mined with one row per customer ---")
    weights_all = np.ones(len(baskets))
    itemsets_all = frequent_itemsets(baskets, weights_all, MIN_SUPPORT)
    rules_all = association_rules(itemsets_all, MIN_CONFIDENCE)
    print_itemsets(itemsets_all)
    print()
    print_rules(rules_all)

    # 3. Mined after dropping duplicate baskets
    print("\n--- 3. Mined after dropping duplicate baskets ---")
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

    # 4. The deduplicated table with its counts restored
    print("\n--- 4. The deduplicated table with its counts restored ---")
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
    # Identical means the same rules, paired by antecedent and consequent, with the
    # same support, confidence and lift. The same list of lift values is not enough.
    paired = rules_all.merge(rules_counted, on=["antecedent", "consequent"], how="outer",
                             suffixes=("_all", "_counted"), indicator=True)
    matches = bool(
        (paired["_merge"] == "both").all()
        and all(np.allclose(paired[f"{measure}_all"], paired[f"{measure}_counted"])
                for measure in ("support", "confidence", "lift"))
    )
    print(f"    rules identical to the one-row-per-customer result: {matches}")
    if matches:
        print("    The fix is not to avoid deduplicating. It is to carry the count.")

    # 5. Lift under the three sample units
    print("\n--- 5. Lift under the three sample units ---")
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
    print("    - means the rule did not clear the thresholds under that unit")

    spread_distinct = rules_distinct["lift"].max() - rules_distinct["lift"].min()
    spread_all = rules_all["lift"].max() - rules_all["lift"].min()
    print(f"\n    lift ranges over {spread_all:.4f} per customer and "
          f"{spread_distinct:.4f} per combination")

    # 6. Against the probabilities 01 draws fund from
    print("\n--- 6. Against the probabilities 01 draws fund from ---")
    wealth = baskets["wealth"].to_numpy()
    fund = baskets["fund"].to_numpy()
    insurance = baskets["insurance"].to_numpy()
    with_wealth = fund[wealth].mean()
    without_wealth = fund[~wealth].mean()
    measured = (fund & wealth).mean() / (fund.mean() * wealth.mean())
    drawn_with = FUND_BASE_RATE * WEALTH_TO_FUND_MULTIPLIER
    print(f"    P(fund | wealth)      {with_wealth:.4f}   drawn at {drawn_with:.4f}")
    print(f"    P(fund | not wealth)  {without_wealth:.4f}   drawn at {FUND_BASE_RATE:.4f}")
    print(f"    lift(wealth -> fund)  {measured:.4f}")
    found = rules_all[
        (rules_all["antecedent"] == "wealth") & (rules_all["consequent"] == "fund")
    ]
    if found.empty:
        print("    the mined rules: wealth -> fund not mined")
    else:
        print(f"    the mined rule reports  {found['lift'].iloc[0]:.4f}   (match: "
              f"{np.isclose(found['lift'].iloc[0], measured)})")
    deduped = rules_distinct[
        (rules_distinct["antecedent"] == "wealth") & (rules_distinct["consequent"] == "fund")
    ]
    if deduped.empty:
        print("    the deduplicated table: wealth -> fund not mined")
    else:
        print(f"    the deduplicated table reports {deduped['lift'].iloc[0]:.4f}")
    top = rules_all.iloc[0]
    top_has_deposit = "deposit" in f"{top['antecedent']} {top['consequent']}"
    if not found.empty:
        gap = top["lift"] - found["lift"].iloc[0]
        print(f"    highest mined lift: {top['antecedent']} -> {top['consequent']} "
              f"{top['lift']:.4f}, {gap:.4f} above wealth -> fund")
    support_fi = (fund & insurance).mean()
    lift_fi = support_fi / (fund.mean() * insurance.mean())
    if support_fi < MIN_SUPPORT:
        print(f"    fund + insurance: support {support_fi:.4f}, below MIN_SUPPORT {MIN_SUPPORT}, "
              f"lift {lift_fi:.4f}, not mined")

    print("\n    01 sets the two probabilities, not the lift, so the lift is measured on the draw.")
    if top_has_deposit:
        print("    01 draws deposit on its own, so the rules that add deposit to")
        print("    wealth -> fund differ from it only by noise.")
    if support_fi < MIN_SUPPORT:
        print("    01 also lowers insurance among fund holders. A minimum support keeps only")
        print("    common combinations, so this link never reaches the rule table.")


if __name__ == "__main__":
    main()
