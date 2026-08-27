"""Build seventy features for a gradient boosting model, then ask how many it used.

Demonstrates that the count of engineered features is not the count that matters:
    1. Derive time, ratio and interaction features from the raw listing fields.
    2. Bin vehicle age, and show what the left edge of the first bin does to brand-new cars.
    3. Add the flags that are conventionally added, and check each one for variance first.
    4. Build group statistics from the training rows only, so no label leaks sideways.
    5. Train CatBoost and score it on a holdout the model never touched.
    6. Read the importances back and count how many features the model never once used.
    7. Compare the surviving features against the formula that generated the prices.

Module 07: Machine Learning and Deep Learning Foundations - Feature Engineering Audit.
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
VALIDATION_FRACTION = 0.1

CATEGORICAL = ["brand", "model_code", "body_type", "fuel_type", "gearbox",
               "damage_flag", "region_code", "seller", "offer_type", "age_segment"]
NUMERIC_FOR_FLAGS = (["power", "odometer_km"] + [f"v_{i}" for i in range(15)])

# The bins the age segment is usually cut with. The left edge is the point of
# this section: pandas.cut leaves the leftmost edge open, so a vehicle whose age
# is exactly zero falls outside every bin.
NAIVE_AGE_BINS = [0, 1, 3, 5, 10, 100]
FIXED_AGE_BINS = [-0.01, 1, 3, 5, 10, 100]
AGE_LABELS = ["under_1y", "1_to_3y", "3_to_5y", "5_to_10y", "over_10y"]

# Clipping limits, chosen the way they usually are: from a glance at the
# distribution rather than from the data itself. Section 3 checks what each one
# actually caught.
POWER_LIMIT = 580
ODOMETER_LIMIT = 31.5

ITERATIONS = 1200
LEARNING_RATE = 0.05
DEPTH = 8


def load(path):
    """Load a listings file with the separator script 02 established."""
    return pd.read_csv(path, sep=" ")


def add_time_features(frame):
    """Turn the two date columns into ages, calendar parts and a rate."""
    registered = pd.to_datetime(frame["reg_date"], format="%Y%m%d", errors="coerce")
    listed = pd.to_datetime(frame["list_date"], format="%Y%m%d", errors="coerce")

    age_days = (listed - registered).dt.days
    negative = int((age_days < 0).sum())
    # A listing dated before the registration is not a vehicle with a negative
    # age, it is a data entry error. Clipping is the usual response, and it
    # creates a pile of rows sitting at exactly zero.
    age_days = age_days.clip(lower=0)

    frame["vehicle_age_days"] = age_days
    frame["vehicle_age_years"] = age_days / 365.0
    frame["reg_year"] = registered.dt.year
    frame["reg_month"] = registered.dt.month
    frame["list_year"] = listed.dt.year
    frame["list_month"] = listed.dt.month
    frame["reg_season"] = (registered.dt.month % 12 + 3) // 3
    frame["list_season"] = (listed.dt.month % 12 + 3) // 3
    frame["is_new"] = (frame["vehicle_age_years"] < 1).astype(int)
    frame["km_per_year"] = frame["odometer_km"] / (frame["vehicle_age_years"] + 0.1)
    return frame, negative, int((age_days == 0).sum())


def bin_age(frame, bins):
    """Cut vehicle age into segments and report how many rows fell outside."""
    segment = pd.cut(frame["vehicle_age_years"], bins=bins, labels=AGE_LABELS)
    return segment, int(segment.isna().sum())


def add_ratio_and_interaction_features(frame):
    """Add the ratios and combinations that a domain reading suggests."""
    frame["power_per_year"] = frame["power"] / (frame["vehicle_age_years"] + 0.1)
    frame["power_times_km"] = frame["power"] * frame["odometer_km"]
    frame["brand_model"] = frame["brand"] * 1000 + frame["model_code"]
    frame["latent_mean"] = frame[[f"v_{i}" for i in range(15)]].mean(axis=1)
    frame["latent_spread"] = frame[[f"v_{i}" for i in range(15)]].std(axis=1)
    # A combination with no physical meaning: an engine rating added to a model
    # identifier. It is here so that section 6 can report what the model made of
    # it, next to the features that do mean something.
    frame["power_plus_model_code"] = frame["power"] + frame["model_code"]
    return frame


def add_flags(frame):
    """Add missing markers and outlier markers, and record their variance."""
    created = []
    for column in NUMERIC_FOR_FLAGS:
        name = f"{column}_missing"
        frame[name] = frame[column].isna().astype(int)
        created.append(name)

    frame["power_outlier"] = (frame["power"] > POWER_LIMIT).astype(int)
    frame["odometer_outlier"] = (frame["odometer_km"] > ODOMETER_LIMIT).astype(int)
    created += ["power_outlier", "odometer_outlier"]

    frame["power"] = frame["power"].clip(upper=POWER_LIMIT)
    frame["odometer_km"] = frame["odometer_km"].clip(upper=ODOMETER_LIMIT)

    constant = [name for name in created if frame[name].nunique() == 1]
    return frame, created, constant


def add_group_statistics(train, other_frames):
    """Summarise price by brand on the training rows, then attach to every frame.

    The statistics are computed once, from training rows only. Recomputing them
    on a concatenation of train and holdout would put holdout prices into a
    training feature, which is the leak script 04 measures.
    """
    stats = train.groupby("brand")["price"].agg(
        brand_price_mean="mean", brand_price_median="median",
        brand_price_std="std", brand_price_count="count").reset_index()
    frequency = train["brand"].value_counts(normalize=True).rename("brand_frequency")

    out = []
    for frame in [train] + other_frames:
        merged = frame.merge(stats, on="brand", how="left")
        merged = merged.merge(frequency, left_on="brand", right_index=True, how="left")
        merged["price_to_brand_mean"] = (
            merged["brand_price_mean"] / merged["brand_price_mean"].mean())
        out.append(merged)
    return out, list(stats.columns[1:]) + ["brand_frequency", "price_to_brand_mean"]


def prepare(frame, segment_bins):
    """Run the whole feature pipeline over one frame."""
    frame = frame.copy()
    frame, negative_ages, zero_ages = add_time_features(frame)
    segment, outside = bin_age(frame, segment_bins)
    frame["age_segment"] = segment
    frame = add_ratio_and_interaction_features(frame)
    frame, flags, constant_flags = add_flags(frame)
    return frame, {"negative_ages": negative_ages, "zero_ages": zero_ages,
                   "outside_bins": outside, "flags": flags,
                   "constant_flags": constant_flags}


def to_model_matrix(frame, feature_names):
    """Return features in the layout CatBoost expects, with categoricals as text."""
    matrix = frame[feature_names].copy()
    for column in CATEGORICAL:
        if column in matrix.columns:
            matrix[column] = matrix[column].astype(str).fillna("missing")
    for column in matrix.columns:
        if column not in CATEGORICAL:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype(float)
    return matrix


def main():
    if not LISTINGS.exists() or not HOLDOUT.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    from catboost import CatBoostRegressor, Pool
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    raw_train = load(LISTINGS)
    raw_holdout = load(HOLDOUT)
    print(f"--- 1. Deriving features from {raw_train.shape[1]} raw columns ---")

    train, report = prepare(raw_train, FIXED_AGE_BINS)
    holdout, _ = prepare(raw_holdout, FIXED_AGE_BINS)
    print(f"    listings dated before their own registration: {report['negative_ages']}, "
          "clipped to an age of zero")
    print(f"    listings now sitting at exactly zero days old: {report['zero_ages']}")

    print("\n--- 2. What the left bin edge does to those rows ---")
    _, outside_naive = bin_age(train, NAIVE_AGE_BINS)
    _, outside_fixed = bin_age(train, FIXED_AGE_BINS)
    print(f"    bins={NAIVE_AGE_BINS} leaves {outside_naive} rows with no segment")
    print(f"    bins={FIXED_AGE_BINS} leaves {outside_fixed} rows with no segment")
    print("    pandas.cut excludes the leftmost edge, so an age of exactly 0.0 is")
    print("    outside every interval. The rows that land there are precisely the")
    print("    ones repaired one step earlier: clipping the bad dates to zero")
    print("    created the pile, and the open left edge then dropped it. Two")
    print("    reasonable steps, and no error is raised by either; the column")
    print("    just gains nulls between them.")

    print("\n--- 3. Flags, each checked for variance before it is kept ---")
    print(f"    flags created: {len(report['flags'])}")
    print(f"    flags that are constant: {len(report['constant_flags'])}")
    for name in report["constant_flags"]:
        print(f"        {name}")
    live = [f for f in report["flags"] if f not in report["constant_flags"]]
    for name in live:
        print(f"    {name} marks {int(train[name].sum())} rows")
    print("    A missing marker on a column that is never missing is a column of")
    print("    zeros with a descriptive name. Seventeen of them were built here.")

    print("\n--- 4. Group statistics, built from training rows only ---")
    frames, stat_columns = add_group_statistics(train, [holdout])
    train, holdout = frames
    print(f"    added {len(stat_columns)} columns: {stat_columns}")

    drop = ["price", "listing_id", "reg_date", "list_date"]
    feature_names = [c for c in train.columns if c not in drop]
    print(f"    feature count going into the model: {len(feature_names)}")

    print("\n--- 5. Training CatBoost ---")
    x_train, x_valid, y_train, y_valid = train_test_split(
        to_model_matrix(train, feature_names), train["price"],
        test_size=VALIDATION_FRACTION, random_state=SEED)
    categorical = [c for c in CATEGORICAL if c in feature_names]

    model = CatBoostRegressor(
        iterations=ITERATIONS, learning_rate=LEARNING_RATE, depth=DEPTH,
        loss_function="MAE", eval_metric="MAE", random_seed=SEED,
        od_type="Iter", od_wait=100, verbose=200, thread_count=-1)
    model.fit(Pool(x_train, y_train, cat_features=categorical),
              eval_set=Pool(x_valid, y_valid, cat_features=categorical),
              use_best_model=True)

    x_holdout = to_model_matrix(holdout, feature_names)
    predicted = model.predict(Pool(x_holdout, cat_features=categorical))
    mae = mean_absolute_error(holdout["price"], predicted)
    rmse = float(np.sqrt(np.mean((holdout["price"] - predicted) ** 2)))
    r2 = r2_score(holdout["price"], predicted)
    print(f"\n    best iteration {model.get_best_iteration()} of {ITERATIONS}")
    print(f"    holdout MAE {mae:.2f}, RMSE {rmse:.2f}, R2 {r2:.4f}")
    print(f"    for scale, the noise added to every price has sigma 900, so an")
    print(f"    MAE near 900 x sqrt(2/pi) = {900 * np.sqrt(2 / np.pi):.0f} is the floor")

    print("\n--- 6. How many of those features the model actually used ---")
    importances = pd.Series(model.get_feature_importance(), index=x_train.columns)
    unused = importances[importances == 0.0].sort_index()
    print(f"    features handed to the model: {len(importances)}")
    print(f"    features with importance exactly 0.0: {len(unused)} "
          f"({len(unused) / len(importances):.0%})")
    for name in unused.index:
        print(f"        {name}")
    near_zero = importances[(importances > 0) & (importances < 0.05)]
    print(f"    features under 0.05 importance: {len(near_zero)}")
    print(f"    power_plus_model_code, the combination with no meaning: "
          f"{importances['power_plus_model_code']:.4f}")
    print(f"    features carrying 95% of the total importance: "
          f"{int((importances.sort_values(ascending=False).cumsum() / importances.sum() <= 0.95).sum()) + 1}")

    print("\n--- 7. Checking the survivors against the generating formula ---")
    top = importances.sort_values(ascending=False).head(12)
    print(f"    {'feature':<26}{'importance':>12}")
    for name, value in top.items():
        print(f"    {name:<26}{value:>12.3f}")
    paid = {"v_0", "v_1", "v_2", "v_3", "v_4", "power", "odometer_km",
            "gearbox", "damage_flag", "brand", "vehicle_age_days",
            "vehicle_age_years", "brand_price_mean", "brand_price_median"}
    hits = [name for name in top.index if name in paid]
    print(f"\n    of the twelve strongest, {len(hits)} trace back to a term in the")
    print(f"    price formula or to a direct restatement of one: {hits}")
    noise_latents = [f"v_{i}" for i in range(5, 15)]
    print(f"    the ten latent columns that were never paid into the price rank "
          f"{[int(importances.rank(ascending=False)[n]) for n in noise_latents]}")
    print("\n    The model was given "
          f"{len(importances)} features and decided with far fewer.")
    print("    'I engineered seventy features' is a statement about the pipeline.")
    print("    get_feature_importance() is the statement about the model.")


if __name__ == "__main__":
    main()
