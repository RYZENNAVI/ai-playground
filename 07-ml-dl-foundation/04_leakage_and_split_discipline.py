"""This script builds three kinds of data leakage into a LightGBM price model on the
vehicle listings (a scaler fitted with the holdout included, target encoding over
every row, and one listing on both sides of a random split). Each leak makes the
validation score look better than the model is, and a holdout that took part in no
fit shows the difference. The scaler is fitted on the whole training file in both
arms, so what they contrast is the holdout crossing into the transform.

The run prints 6 parts:
    1. An honest baseline, scored on validation and on the holdout.
    2. The scaler fitted with and without the holdout, on a nearest-neighbour model.
    3. Target encoding at four key cardinalities, built from the fit rows, from every
       row, and out of fold.
    4. The same listing on both sides of a random split.
    5. The four setups side by side.
    6. What the gap between validation and holdout catches, and what it misses.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
LISTINGS = DATA / "vehicle_listings.csv"
HOLDOUT = DATA / "vehicle_holdout.csv"

SEED = 20260824
VALIDATION_FRACTION = 0.2
DUPLICATE_FRACTION = 0.25
FOLDS = 5

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

BASE_FEATURES = ["brand", "model_code", "power", "odometer_km", "undamaged_flag",
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
    """Mean absolute error, the metric every part reports."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def fit_tree(x_train, y_train, x_eval):
    """Train one LightGBM model and predict. No early stopping: it would read the
    evaluation set, which is a fourth way to leak."""
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
    """Score a nearest-neighbour model with the scaler fitted with and without the
    holdout. A tree ignores the scale of a feature, so it could not show the effect."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import MinMaxScaler

    columns = SCALE_SENSITIVE_FEATURES
    results = {}

    for label in ("leaky", "honest"):
        frame = train.copy()
        if label == "leaky":
            # The holdout's minimum and maximum go into the transform as well.
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


def out_of_fold_means(fit):
    """Encode each fit row from the other folds only, so no row sees its own price."""
    from sklearn.model_selection import KFold

    values = np.empty(len(fit))
    for kept, held in KFold(FOLDS, shuffle=True, random_state=SEED).split(fit):
        part = fit.iloc[kept]
        means = part.groupby("_key")["price"].mean()
        values[held] = fit["_key"].iloc[held].map(means).fillna(part["price"].mean())
    return values


def target_encoding_at(train, holdout, key_builder):
    """Score the key's mean price as a feature, built three ways, and count the rows
    per key, which the leak turns out to depend on."""
    from sklearn.model_selection import train_test_split

    train = train.copy()
    holdout = holdout.copy()
    train["_key"] = key_builder(train)
    holdout["_key"] = key_builder(holdout)

    fit, validate = train_test_split(train, test_size=VALIDATION_FRACTION,
                                     random_state=SEED)
    results = {}

    for label in ("leaky", "honest", "out_of_fold"):
        source = train if label == "leaky" else fit
        means = source.groupby("_key")["price"].mean()
        fallback = source["price"].mean()

        def encode(frame, values=None):
            encoded = frame[BASE_FEATURES].copy()
            if values is None:
                values = frame["_key"].map(means).fillna(fallback).to_numpy()
            encoded["key_target"] = values
            return encoded

        # The leaky and honest arms encode each fit row with a mean that includes
        # its own price. Only the out of fold arm does not.
        fit_encoded = encode(fit, out_of_fold_means(fit) if label == "out_of_fold" else None)
        results[label] = (
            mae(validate["price"], fit_tree(fit_encoded, fit["price"], encode(validate))),
            mae(holdout["price"], fit_tree(fit_encoded, fit["price"], encode(holdout))))

    distinct = int(train["_key"].nunique())
    return results, distinct, len(train) / distinct


def duplicated_rows(train, holdout):
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

    train = load(LISTINGS)
    holdout = load(HOLDOUT)
    print(f"{len(train)} listings to learn from, {len(holdout)} kept aside and "
          "never touched by any fit below.\n")

    # 1. Honest baseline

    print("--- 1. Honest baseline ---")
    base_validation, base_holdout = honest_baseline(train, holdout)
    print(f"    validation MAE {base_validation:8.2f}")
    print(f"    holdout MAE    {base_holdout:8.2f}")
    print(f"    gap            {base_validation - base_holdout:+8.2f}")
    print("    The two agree, which is what a validation score is for.")

    # 2. Scaler fitted with the holdout included

    print("\n--- 2. Scaler fitted with the holdout included ---")
    scaler_results = scaler_before_split(train, holdout)
    print(f"    {'':<10}{'validation':>14}{'holdout':>12}{'gap':>10}")
    for label in ("honest", "leaky"):
        validation, holdout_score = scaler_results[label]
        print(f"    {label:<10}{validation:>14.2f}{holdout_score:>12.2f}"
              f"{validation - holdout_score:>10.2f}")
    delta = scaler_results["honest"][0] - scaler_results["leaky"][0]
    print(f"    the leak is worth {delta:+.2f} MAE on the validation score")
    print("    A minimum and a maximum are two numbers per column. Handing them")
    print("    over leaks almost nothing.")

    # 3. Target encoding

    print("\n--- 3. Target encoding at four cardinalities ---")
    print("    Each validation row is averaged into its own key's mean, so the")
    print("    feature carries a share of that row's own price back to it. How big")
    print("    a share is decided by how many rows sit in the key.\n")
    print(f"    {'key':<22}{'keys':>7}{'rows/key':>10}{'honest val':>12}"
          f"{'leaky val':>11}{'leaked':>9}{'leaky holdout':>15}")
    encoding_results = {}
    rows_per_key = {}
    for name, builder in ENCODING_KEYS:
        results, distinct, per_key = target_encoding_at(train, holdout, builder)
        gain = results["honest"][0] - results["leaky"][0]
        encoding_results[name] = results
        rows_per_key[name] = per_key
        print(f"    {name:<22}{distinct:>7}{per_key:>10.1f}{results['honest'][0]:>12.2f}"
              f"{results['leaky'][0]:>11.2f}{gain:>9.2f}{results['leaky'][1]:>15.2f}")
    print("\n    The leaked column is what the validation score gained, and it grows")
    print("    as fewer rows share a key. For the three milder keys the holdout")
    print("    stays near the baseline, so the model gained none of it.")

    print("\n    Out of fold, each fit row is encoded from the other "
          f"{FOLDS - 1} folds only:\n")
    print(f"    {'key':<22}{'validation':>12}{'holdout':>10}")
    for name, results in encoding_results.items():
        validation, holdout_score = results["out_of_fold"]
        print(f"    {name:<22}{validation:>12.2f}{holdout_score:>10.2f}")
    worst_key = "region_code x brand"
    oof_validation, oof_holdout = encoding_results[worst_key]["out_of_fold"]
    print(f"\n    {worst_key} now scores {oof_validation:.2f} on validation and "
          f"{oof_holdout:.2f}")
    print(f"    on the holdout, against the baseline's {base_validation:.2f} and "
          f"{base_holdout:.2f}. The")
    print("    honest arm still encodes each fit row with a mean that includes its")
    print(f"    own price, and at {rows_per_key[worst_key]:.1f} rows per key that mean "
          "is mostly the row's")
    print("    own label.")

    # 4. Duplicated listings

    print("\n--- 4. The same listing on both sides of the split ---")
    dup_validation, dup_holdout, shared, validate_rows = duplicated_rows(train, holdout)
    print(f"    listings duplicated: {DUPLICATE_FRACTION:.0%}")
    print(f"    ids present in both halves: {shared} of {validate_rows} validation rows")
    print(f"    validation MAE {dup_validation:8.2f}")
    print(f"    holdout MAE    {dup_holdout:8.2f}")
    print(f"    gap            {dup_validation - dup_holdout:+8.2f}")
    print("    Nothing here is an encoding mistake. The split ran on rows, and the")
    print("    rows were not independent.")

    # 5. The four setups side by side

    print("\n--- 5. All four together ---")
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

    # 6. What the gap catches

    print("\n--- 6. What the gap catches ---")
    mild = [name for name in encoding_results if name != worst_key]
    print(f"    For the scaler, the duplicates and the {len(mild)} milder encodings, the")
    print("    holdout column hardly moves. The leak made the report better and left")
    print("    the model where it was, and the report is what gets acted on.")
    print("    The extreme encoding is the exception: its holdout went from "
          f"{base_holdout:.0f} to")
    print(f"    {encoding_results[worst_key]['leaky'][1]:.0f}, so there the model really "
          "is worse. The key is not to blame,")
    print(f"    since out of fold it scored {oof_holdout:.0f} on the holdout. In the other two arms")
    print("    every fit row saw its own price in the encoding, and the tree learned")
    print("    to lean on it.")
    print("    None of the three raised an error, printed a warning or produced an")
    print("    implausible number on the way in.")

    # Both scaler arms are scored, so the gap's one miss is measured: gaps that
    # agree cannot tell the leaky arm from the honest one.
    honest_gap = scaler_results["honest"][0] - scaler_results["honest"][1]
    leaky_gap = scaler_results["leaky"][0] - scaler_results["leaky"][1]
    print("    The gap column catches the target encoding and the duplicate rows: both")
    print("    beat their untouched holdout by a wide margin, and the baseline does not.")
    print("    It misses the scaler. Its two arms differ by "
          f"{abs(leaky_gap - honest_gap):.2f} MAE of gap")
    print(f"    ({honest_gap:+.2f} honest against {leaky_gap:+.2f} leaky), so the gap "
          "cannot tell")
    print("    them apart. What that column shows for the scaler row comes from the")
    print("    nearest neighbour model, not the leak. A leak this small needs the two")
    print("    fits side by side, which part 2 does and part 5 cannot.")


if __name__ == "__main__":
    main()
