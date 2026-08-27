"""Three ways a validation score gets better while the model gets no better at all.

Demonstrates that leakage damages the estimate rather than the model:
    1. Build an honest baseline whose validation score and holdout score agree.
    2. Fit the scaler before the split instead of after, and price the difference.
    3. Encode a high-cardinality column with target statistics taken from every row.
    4. Let the same listing appear on both sides of a random split.
    5. Put the three next to the baseline and read which number moved.
    6. Show the check that catches all three without knowing which one happened.

Module 07: Machine Learning and Deep Learning Foundations - Leakage and Splits.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
LISTINGS = DATA / "vehicle_listings.csv"
HOLDOUT = DATA / "vehicle_holdout.csv"

SEED = 20260824
VALIDATION_FRACTION = 0.2
DUPLICATE_FRACTION = 0.25

# Target encoding is tested at four cardinalities rather than one. How much a
# row's own price flows back into its own feature depends on how many rows share
# its key, so the severity of this leak is a property of the key, not of the
# technique. Each entry is (column, how the key is built).
ENCODING_KEYS = [
    ("brand", lambda f: f["brand"]),
    ("model_code", lambda f: f["model_code"]),
    ("region_code", lambda f: f["region_code"]),
    ("region_code x brand", lambda f: f["region_code"] * 100 + f["brand"]),
]

BASE_FEATURES = ["brand", "model_code", "power", "odometer_km", "damage_flag",
                 "gearbox", "vehicle_age_years"] + [f"v_{i}" for i in range(15)]
SCALE_SENSITIVE_FEATURES = ["power", "odometer_km", "vehicle_age_years",
                            "v_0", "v_1", "v_2", "v_3", "v_4"]

TREE_PARAMS = {"objective": "regression_l1", "metric": "mae", "learning_rate": 0.06,
               "num_leaves": 63, "verbose": -1, "seed": SEED, "num_threads": -1}
TREE_ROUNDS = 400
NEIGHBOURS = 15


def load(path):
    """Load a listings file and derive the few raw features these tests need."""
    frame = pd.read_csv(path, sep=" ")
    registered = pd.to_datetime(frame["reg_date"], format="%Y%m%d", errors="coerce")
    listed = pd.to_datetime(frame["list_date"], format="%Y%m%d", errors="coerce")
    frame["vehicle_age_years"] = (listed - registered).dt.days.clip(lower=0) / 365.0
    for column in ["body_type", "fuel_type", "gearbox"]:
        frame[column] = frame[column].fillna(-1)
    return frame


def mae(actual, predicted):
    """Mean absolute error, the metric every section below reports."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def fit_tree(x_train, y_train, x_eval):
    """Train one gradient boosting model and predict, with no early stopping.

    Early stopping is deliberately off. It would read the evaluation set during
    training, which is a fourth way to leak and would blur the three below.
    """
    import lightgbm as lgb
    model = lgb.train(TREE_PARAMS, lgb.Dataset(x_train, label=y_train),
                      num_boost_round=TREE_ROUNDS)
    return model.predict(x_eval)


def honest_baseline(train, holdout):
    """Split first, fit everything on the training part only."""
    from sklearn.model_selection import train_test_split

    fit, validate = train_test_split(train, test_size=VALIDATION_FRACTION,
                                     random_state=SEED)
    validation = mae(validate["price"],
                     fit_tree(fit[BASE_FEATURES], fit["price"], validate[BASE_FEATURES]))
    holdout_score = mae(holdout["price"],
                        fit_tree(fit[BASE_FEATURES], fit["price"], holdout[BASE_FEATURES]))
    return validation, holdout_score


def scaler_before_split(train, holdout):
    """Compare a scaler fitted on everything against one fitted on the split.

    A nearest-neighbour model is used because a tree does not care about the
    scale of a feature at all, so it cannot show the effect either way.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import MinMaxScaler

    columns = SCALE_SENSITIVE_FEATURES
    results = {}

    for label in ("leaky", "honest"):
        frame = train.copy()
        if label == "leaky":
            # Every row of the holdout is standing right here, contributing its
            # minimum and maximum to the transform that the training rows get.
            scaler = MinMaxScaler().fit(
                pd.concat([frame[columns], holdout[columns]], ignore_index=True))
            scaled_train = pd.DataFrame(scaler.transform(frame[columns]), columns=columns)
            scaled_holdout = pd.DataFrame(scaler.transform(holdout[columns]), columns=columns)
        else:
            scaler = MinMaxScaler().fit(frame[columns])
            scaled_train = pd.DataFrame(scaler.transform(frame[columns]), columns=columns)
            scaled_holdout = pd.DataFrame(scaler.transform(holdout[columns]), columns=columns)

        scaled_train["price"] = frame["price"].to_numpy()
        fit, validate = train_test_split(scaled_train, test_size=VALIDATION_FRACTION,
                                         random_state=SEED)
        model = KNeighborsRegressor(n_neighbors=NEIGHBOURS).fit(fit[columns], fit["price"])
        results[label] = (mae(validate["price"], model.predict(validate[columns])),
                          mae(holdout["price"], model.predict(scaled_holdout[columns])))
    return results


def target_encoding_at(train, holdout, key_builder):
    """Compare an encoding built from every row against one built from the fit rows.

    Returns the two validation and holdout scores plus how many rows share a key,
    which is the number the comparison turns out to depend on.
    """
    from sklearn.model_selection import train_test_split

    train = train.copy()
    holdout = holdout.copy()
    train["_key"] = key_builder(train)
    holdout["_key"] = key_builder(holdout)

    fit, validate = train_test_split(train, test_size=VALIDATION_FRACTION,
                                     random_state=SEED)
    results = {}

    for label in ("leaky", "honest"):
        source = train if label == "leaky" else fit
        means = source.groupby("_key")["price"].mean()
        fallback = source["price"].mean()

        def encode(frame):
            encoded = frame[BASE_FEATURES].copy()
            encoded["key_target"] = frame["_key"].map(means).fillna(fallback).to_numpy()
            return encoded

        results[label] = (
            mae(validate["price"], fit_tree(encode(fit), fit["price"], encode(validate))),
            mae(holdout["price"], fit_tree(encode(fit), fit["price"], encode(holdout))))

    distinct = int(train["_key"].nunique())
    return results, distinct, len(train) / distinct


def duplicated_rows(train, holdout, rng):
    """Copy a quarter of the listings, then split at random and score."""
    from sklearn.model_selection import train_test_split

    copies = train.sample(frac=DUPLICATE_FRACTION, random_state=SEED)
    inflated = pd.concat([train, copies], ignore_index=True)
    inflated = inflated.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    fit, validate = train_test_split(inflated, test_size=VALIDATION_FRACTION,
                                     random_state=SEED)
    shared = int(pd.merge(fit[["listing_id"]].drop_duplicates(),
                          validate[["listing_id"]].drop_duplicates(),
                          on="listing_id").shape[0])

    validation = mae(validate["price"],
                     fit_tree(fit[BASE_FEATURES], fit["price"], validate[BASE_FEATURES]))
    holdout_score = mae(holdout["price"],
                        fit_tree(fit[BASE_FEATURES], fit["price"], holdout[BASE_FEATURES]))
    return validation, holdout_score, shared, len(validate)


def main():
    if not LISTINGS.exists() or not HOLDOUT.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    rng = np.random.default_rng(SEED)
    train = load(LISTINGS)
    holdout = load(HOLDOUT)
    print(f"{len(train)} listings to learn from, {len(holdout)} kept aside and "
          "never touched by any fit below.\n")

    print("--- 1. Honest baseline ---")
    base_validation, base_holdout = honest_baseline(train, holdout)
    print(f"    validation MAE {base_validation:8.2f}")
    print(f"    holdout MAE    {base_holdout:8.2f}")
    print(f"    gap            {base_validation - base_holdout:+8.2f}")
    print("    The two agree, which is what a validation score is for.")

    print("\n--- 2. Scaler fitted before the split ---")
    scaler_results = scaler_before_split(train, holdout)
    print(f"    {'':<10}{'validation':>14}{'holdout':>12}{'gap':>10}")
    for label in ("honest", "leaky"):
        validation, holdout_score = scaler_results[label]
        print(f"    {label:<10}{validation:>14.2f}{holdout_score:>12.2f}"
              f"{validation - holdout_score:>10.2f}")
    delta = scaler_results["honest"][0] - scaler_results["leaky"][0]
    print(f"    the leak is worth {delta:+.2f} MAE on the validation score")
    print("    A minimum and a maximum are two numbers per column. Handing them")
    print("    over leaks almost nothing, and this is the leak most often caught")
    print("    in review while the two below go unnoticed.")

    print("\n--- 3. Target encoding built from every row, at four cardinalities ---")
    print("    Each validation row is averaged into its own key's mean, so the")
    print("    feature carries a share of that row's own price back to it. How big")
    print("    a share is decided by how many rows sit in the key.\n")
    print(f"    {'key':<22}{'keys':>7}{'rows/key':>10}{'honest val':>12}"
          f"{'leaky val':>11}{'leaked':>9}{'leaky holdout':>15}")
    encoding_results = {}
    for name, builder in ENCODING_KEYS:
        results, distinct, per_key = target_encoding_at(train, holdout, builder)
        gain = results["honest"][0] - results["leaky"][0]
        encoding_results[name] = results
        print(f"    {name:<22}{distinct:>7}{per_key:>10.1f}{results['honest'][0]:>12.2f}"
              f"{results['leaky'][0]:>11.2f}{gain:>9.2f}{results['leaky'][1]:>15.2f}")
    print("\n    The last column is the one that never improves. The leak buys")
    print("    validation MAE and nothing else, and it buys more of it the fewer")
    print("    rows share a key. At one and a half rows per key the encoded")
    print("    feature is close to being the label itself.")

    print("\n--- 4. The same listing on both sides of the split ---")
    dup_validation, dup_holdout, shared, validate_rows = duplicated_rows(train, holdout, rng)
    print(f"    listings duplicated: {DUPLICATE_FRACTION:.0%}")
    print(f"    ids present in both halves: {shared} of {validate_rows} validation rows")
    print(f"    validation MAE {dup_validation:8.2f}")
    print(f"    holdout MAE    {dup_holdout:8.2f}")
    print(f"    gap            {dup_validation - dup_holdout:+8.2f}")
    print("    Nothing here is an encoding mistake. The split ran on rows, and the")
    print("    rows were not independent.")

    print("\n--- 5. All four together ---")
    worst_key = "region_code x brand"
    rows = [
        ("honest baseline", base_validation, base_holdout, "boosted trees"),
        ("target encoding leak", encoding_results[worst_key]["leaky"][0],
         encoding_results[worst_key]["leaky"][1], "boosted trees"),
        ("duplicate rows", dup_validation, dup_holdout, "boosted trees"),
        ("scaler leak", scaler_results["leaky"][0], scaler_results["leaky"][1],
         "nearest neighbours"),
    ]
    print(f"    {'setup':<24}{'validation':>12}{'holdout':>10}{'gap':>10}  model")
    for name, validation, holdout_score, model in rows:
        print(f"    {name:<24}{validation:>12.2f}{holdout_score:>10.2f}"
              f"{validation - holdout_score:>10.2f}  {model}")
    print("    The scaler row runs on a different model and is not comparable")
    print("    against the three above it in absolute terms, only in its gap.")

    print("\n--- 6. What survives all three ---")
    print("    For the scaler, for the duplicates and for the two mild encodings,")
    print("    the holdout column hardly moves: the leak did not build a better")
    print("    model, it built a better report of the same model, and the report is")
    print("    what gets acted on.")
    print(f"    The extreme encoding is the exception and is worth stating plainly:")
    print(f"    its holdout went from {base_holdout:.0f} to "
          f"{encoding_results[worst_key]['leaky'][1]:.0f}, so there the model really")
    print("    is worse. Its honest twin scored "
          f"{encoding_results[worst_key]['honest'][0]:.0f} on validation, which says")
    print("    the feature was a bad idea at that cardinality whether or not it")
    print("    leaked. Leakage hid a bad feature behind a good score.")
    print("    None of the three raised an error, printed a warning or produced an")
    print("    implausible number on the way in. What separates them from the")
    print("    baseline is the gap column: a validation score that beats an")
    print("    untouched holdout by a wide margin is describing the split.")


if __name__ == "__main__":
    main()
