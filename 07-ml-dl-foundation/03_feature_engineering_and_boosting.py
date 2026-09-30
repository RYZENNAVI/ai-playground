"""This script engineers features for a CatBoost gradient boosting model on the
vehicle listings, turning 30 raw columns into 68 model features, then reads the
feature importances back to count how many the model used. The count built is a
fact about the pipeline, and the count used is the fact about the model.

The run prints 7 parts:
    1. Time features, and the rows whose bad dates were clipped to age zero.
    2. What the left edge of the first age bin does to those rows (the model gets
       the fixed bins, and the naive ones are shown for comparison).
    3. The flags, each checked for variance.
    4. Group statistics, built from the rows of the training file only.
    5. CatBoost's holdout score, against the floor the noise sets.
    6. How many features the model never once used.
    7. The strongest features, checked against the formula that generated the prices.
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
VALIDATION_FRACTION = 0.1

CATEGORICAL = ["brand", "model_code", "body_type", "fuel_type", "gearbox",
               "undamaged_flag", "region_code", "seller", "offer_type", "age_segment"]
NUMERIC_FOR_FLAGS = (["power", "odometer_km"] + [f"v_{i}" for i in range(15)])

# The bins the age segment is usually cut with. The left edge is the point of
# part 2: pandas.cut leaves the leftmost edge open, so a vehicle whose age
# is exactly zero falls outside every bin.
NAIVE_AGE_BINS = [0, 1, 3, 5, 10, 100]
FIXED_AGE_BINS = [-0.01, 1, 3, 5, 10, 100]
AGE_LABELS = ["under_1y", "1_to_3y", "3_to_5y", "5_to_10y", "over_10y"]

# Clipping limits set by eye, the way they often are. Part 3 counts what each
# one caught.
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
    # A listing dated before its registration is a data entry error. Clipping it
    # to zero is the usual response, and it builds a pile of rows at exactly zero.
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
    # identifier. It is here so that part 6 can report what the model made of
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
    """Summarise price by brand on the rows of the training file, then attach it to
    every frame. Computing it over the holdout as well would put holdout prices into
    a feature.
    """
    stats = train.groupby("brand")["price"].agg(
        brand_price_mean="mean", brand_price_median="median",
        brand_price_std="std", brand_price_count="count").reset_index()
    frequency = train["brand"].value_counts(normalize=True).rename("brand_frequency")
    # Weighted by rows, the brand means average back to the training price mean.
    overall = train["price"].mean()

    out = []
    for frame in [train] + other_frames:
        merged = frame.merge(stats, on="brand", how="left")
        merged = merged.merge(frequency, left_on="brand", right_index=True, how="left")
        merged["brand_mean_ratio"] = merged["brand_price_mean"] / overall
        out.append(merged)
    return out, list(stats.columns[1:]) + ["brand_frequency", "brand_mean_ratio"]


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
            matrix[column] = matrix[column].astype(str)
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

    # 1. Time features

    raw_train = load(LISTINGS)
    raw_holdout = load(HOLDOUT)
    print(f"--- 1. Deriving features from {raw_train.shape[1]} raw columns ---")

    train, report = prepare(raw_train, FIXED_AGE_BINS)
    holdout, _ = prepare(raw_holdout, FIXED_AGE_BINS)
    print(f"    listings dated before their own registration: {report['negative_ages']}, "
          "clipped to an age of zero")
    print(f"    listings now sitting at exactly zero days old: {report['zero_ages']}")

    # 2. The left edge of the first age bin

    print("\n--- 2. What the left bin edge does to those rows ---")
    _, outside_naive = bin_age(train, NAIVE_AGE_BINS)
    _, outside_fixed = bin_age(train, FIXED_AGE_BINS)
    print(f"    bins={NAIVE_AGE_BINS} leaves {outside_naive} rows with no segment")
    print(f"    bins={FIXED_AGE_BINS} leaves {outside_fixed} rows with no segment")
    print("    pandas.cut excludes the leftmost edge, so an age of exactly 0.0 is")
    print(f"    outside every interval. {report['negative_ages']} of the {outside_naive} "
          "rows that land there are the")
    print("    ones repaired one step earlier: clipping the bad dates to zero built the")
    print("    pile, and the open left edge dropped it. Both steps are reasonable and")
    print("    neither raises an error. The column just gains nulls between them.")

    # 3. Flags

    print("\n--- 3. Flags, each checked for variance before it is kept ---")
    print(f"    flags created: {len(report['flags'])}")
    print(f"    flags that are constant: {len(report['constant_flags'])}")
    for name in report["constant_flags"]:
        print(f"        {name}")
    live = [f for f in report["flags"] if f not in report["constant_flags"]]
    for name in live:
        print(f"    {name} marks {int(train[name].sum())} rows")
    dead_markers = [n for n in report["constant_flags"] if n.endswith("_missing")]
    others = [n for n in report["constant_flags"] if n not in dead_markers]
    print("    A missing marker on a column that is never missing is a column of")
    print(f"    zeros with a descriptive name, and {len(dead_markers)} were built here. "
          f"{', '.join(others)}")
    print("    is constant too: its limit sits above every value in the column.")

    # 4. Group statistics

    print("\n--- 4. Group statistics, built from training rows only ---")
    frames, stat_columns = add_group_statistics(train, [holdout])
    train, holdout = frames
    print(f"    added {len(stat_columns)} columns: {stat_columns}")

    drop = ["price", "listing_id", "reg_date", "list_date"]
    feature_names = [c for c in train.columns if c not in drop]
    print(f"    feature count going into the model: {len(feature_names)}")

    # 5. CatBoost and its holdout score

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
    print("    for scale, the noise added to every price has sigma 900, so an")
    print(f"    MAE near 900 x sqrt(2/pi) = {900 * np.sqrt(2 / np.pi):.0f} is the floor")

    # 6. Features never used

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
    share = importances.sort_values(ascending=False).cumsum() / importances.sum()
    print(f"    features carrying 95% of the total importance: {int((share <= 0.95).sum()) + 1}")

    # 7. The survivors against the formula

    print("\n--- 7. Checking the survivors against the generating formula ---")
    top = importances.sort_values(ascending=False).head(12)
    print(f"    {'feature':<26}{'importance':>12}")
    for name, value in top.items():
        print(f"    {name:<26}{value:>12.3f}")
    # Formula terms, and every feature computed from those terms only.
    paid = {"v_0", "v_1", "v_2", "v_3", "v_4", "power", "odometer_km",
            "gearbox", "undamaged_flag", "brand", "vehicle_age_days",
            "vehicle_age_years", "age_segment", "is_new", "km_per_year",
            "power_per_year", "power_times_km", "power_outlier",
            "brand_price_mean", "brand_price_median", "brand_price_std",
            "brand_price_count", "brand_frequency", "brand_mean_ratio"}
    hits = [name for name in top.index if name in paid]
    print(f"\n    of the twelve strongest, {len(hits)} are a term in the price formula")
    print(f"    or are computed from those terms only: {hits}")
    rank = importances.rank(ascending=False)
    print("    the five latent columns paid into the price rank "
          f"{[int(rank[f'v_{i}']) for i in range(5)]}")
    print("    the ten latent columns that were never paid into the price rank "
          f"{[int(rank[f'v_{i}']) for i in range(5, 15)]}")
    print("\n    The model was given "
          f"{len(importances)} features and decided with far fewer.")
    print(f"    'I engineered {len(importances)} features' is a statement about the pipeline.")
    print("    get_feature_importance() is the statement about the model.")


if __name__ == "__main__":
    main()
