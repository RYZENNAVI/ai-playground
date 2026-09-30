"""This script combines four regressors that price the vehicle listings, by blending
(plain and weighted averages) and by stacking, and puts each result next to how
correlated the members' errors are.

    1. Train three boosted models and one neighbour model, and score each alone.
    2. Correlate their predictions, then their errors, which is the number averaging
       depends on.
    3. Average the three boosters, plainly and weighted by inverse holdout error.
    4. Learn the weights out of fold with a linear meta model, and read them. Rescale
       catboost alone the same way, to see how much of the stack's gain is just that.
    5. Add the neighbour model to both the average and the stack.
    6. Put every result in one table, and check whether the best gain is larger than
       the noise.
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
FOLDS = 4
BOOTSTRAP = 2000

FEATURES = ["brand", "model_code", "power", "odometer_km", "undamaged_flag",
            "gearbox", "body_type", "fuel_type", "vehicle_age_years",
            "km_per_year"] + [f"v_{i}" for i in range(15)]
NEIGHBOUR_FEATURES = ["power", "odometer_km", "vehicle_age_years",
                      "v_0", "v_1", "v_2", "v_3", "v_4"]
NEIGHBOURS = 25


def load(path):
    """Load a listings file and derive the two time features the models use."""
    frame = pd.read_csv(path, sep=" ")
    registered = pd.to_datetime(frame["reg_date"], format="%Y%m%d", errors="coerce")
    listed = pd.to_datetime(frame["list_date"], format="%Y%m%d", errors="coerce")
    frame["vehicle_age_years"] = (listed - registered).dt.days.clip(lower=0) / 365.0
    frame["km_per_year"] = frame["odometer_km"] / (frame["vehicle_age_years"] + 0.1)
    for column in ["body_type", "fuel_type", "gearbox"]:
        frame[column] = frame[column].fillna(-1)
    return frame


def mae(actual, predicted):
    """Mean absolute error, the metric this whole script is scored in."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def make_learners():
    """Return the four learners as fit-and-predict functions, so the fold loop treats
    the scaled neighbour model like the trees."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor

    def tree_learner(build):
        def learner(train, target, evaluate):
            model = build()
            model.fit(train[FEATURES], target)
            return model.predict(evaluate[FEATURES])
        return learner

    def neighbour_learner(train, target, evaluate):
        scaler = MinMaxScaler().fit(train[NEIGHBOUR_FEATURES])
        model = KNeighborsRegressor(n_neighbors=NEIGHBOURS)
        model.fit(scaler.transform(train[NEIGHBOUR_FEATURES]), target)
        return model.predict(scaler.transform(evaluate[NEIGHBOUR_FEATURES]))

    return [
        ("xgboost", tree_learner(lambda: XGBRegressor(
            n_estimators=400, learning_rate=0.08, max_depth=7, subsample=0.9,
            colsample_bytree=0.8, random_state=SEED, verbosity=0, n_jobs=-1))),
        ("lightgbm", tree_learner(lambda: LGBMRegressor(
            n_estimators=400, learning_rate=0.08, num_leaves=63,
            random_state=SEED, verbose=-1, n_jobs=-1))),
        ("catboost", tree_learner(lambda: CatBoostRegressor(
            iterations=400, learning_rate=0.08, depth=7, random_seed=SEED,
            verbose=0, thread_count=-1))),
        ("neighbours", neighbour_learner),
    ]


def out_of_fold_predictions(learners, train, holdout):
    """Return each learner's out-of-fold and holdout predictions. A stack fitted on
    in-sample predictions would favour the model that overfits most."""
    from sklearn.model_selection import KFold

    folds = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = pd.DataFrame(index=train.index, columns=[n for n, _ in learners], dtype=float)
    holdout_predictions = {}

    for name, learner in learners:
        for fit_index, out_index in folds.split(train):
            fit_rows = train.iloc[fit_index]
            out_rows = train.iloc[out_index]
            oof.iloc[out_index, oof.columns.get_loc(name)] = learner(
                fit_rows, fit_rows["price"], out_rows)
        holdout_predictions[name] = learner(train, train["price"], holdout)
        print(f"    {name:<12} out-of-fold MAE {mae(train['price'], oof[name]):8.2f}   "
              f"holdout MAE {mae(holdout['price'], holdout_predictions[name]):8.2f}")

    return oof, pd.DataFrame(holdout_predictions, index=holdout.index)


def main():
    if not LISTINGS.exists() or not HOLDOUT.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    from sklearn.linear_model import LinearRegression

    train = load(LISTINGS)
    holdout = load(HOLDOUT)
    learners = make_learners()
    booster_names = [name for name, _ in learners if name != "neighbours"]

    # 1. Four learners, each scored on its own
    print("--- 1. Four learners, each scored on its own ---")
    print(f"    {len(train)} training rows, {len(holdout)} holdout rows, "
          f"{FOLDS}-fold out-of-fold predictions\n")
    oof, holdout_predictions = out_of_fold_predictions(learners, train, holdout)
    solo = {name: mae(holdout["price"], holdout_predictions[name])
            for name, _ in learners}
    best_solo = min(solo, key=solo.get)
    print(f"\n    best single model: {best_solo} at {solo[best_solo]:.2f}")

    # 2. How much they agree
    print("\n--- 2. How much they agree ---")
    prediction_correlation = holdout_predictions.corr()
    errors = holdout_predictions.sub(holdout["price"], axis=0)
    error_correlation = errors.corr()
    names = list(holdout_predictions.columns)

    print("    correlation between predictions:")
    print("    " + "".join(f"{n:>13}" for n in [""] + names))
    for row in names:
        print(f"    {row:>13}" + "".join(
            f"{prediction_correlation.loc[row, c]:>13.4f}" for c in names))
    print("\n    correlation between errors:")
    print("    " + "".join(f"{n:>13}" for n in [""] + names))
    for row in names:
        print(f"    {row:>13}" + "".join(
            f"{error_correlation.loc[row, c]:>13.4f}" for c in names))
    booster_pairs = [error_correlation.loc[a, b]
                     for i, a in enumerate(booster_names) for b in booster_names[i + 1:]]
    print(f"\n    the three boosters agree on their predictions at "
          f"{prediction_correlation.loc[booster_names, booster_names].values[np.triu_indices(3, 1)].mean():.4f}")
    print(f"    and on their mistakes at {np.mean(booster_pairs):.4f}")
    print("    Predictions correlate because they are all mostly the price. Errors")
    print("    are what is left when the price is taken out, and averaging can only")
    print("    cancel what is left.")

    # 3. Averaging the three boosters
    print("\n--- 3. Averaging the three boosters ---")
    simple = holdout_predictions[booster_names].mean(axis=1)
    inverse = np.array([1.0 / solo[name] for name in booster_names])
    inverse = inverse / inverse.sum()
    weighted = holdout_predictions[booster_names].mul(inverse, axis=1).sum(axis=1)
    weight_text = ", ".join(f"{name} {value:.3f}"
                            for name, value in zip(booster_names, inverse))
    print(f"    simple average               {mae(holdout['price'], simple):8.2f}")
    print(f"    weighted by inverse holdout  {mae(holdout['price'], weighted):8.2f}   "
          f"({weight_text})")
    print(f"    best single model            {solo[best_solo]:8.2f}  [{best_solo}]")
    change = mae(holdout["price"], simple) - solo[best_solo]
    print(f"\n    The average is {change:+.2f} MAE against the best single model,")
    print(f"    which is to say it is {'worse' if change > 0 else 'better'}. "
          "Averaging pulls every member towards")
    print("    the middle, and here the members are not equally good: the best one")
    print(f"    scores {solo[best_solo]:.0f} and the worst "
          f"{max(solo[n] for n in booster_names):.0f}, so an equal vote spends the")
    print("    better model's accuracy on carrying the other two.")
    print("    Weighting by inverse holdout error recovers part of that, but those")
    print("    weights were read off the set being predicted, which is fitting on")
    print("    the holdout with one parameter per model.")

    # 4. Learning the weights out of fold instead
    print("\n--- 4. Learning the weights out of fold instead ---")
    stack = LinearRegression().fit(oof[booster_names], train["price"])
    stacked = stack.predict(holdout_predictions[booster_names])
    print(f"    meta model intercept {stack.intercept_:.2f}")
    for name, weight in zip(booster_names, stack.coef_):
        print(f"    weight for {name:<12}{weight:>8.4f}")
    print(f"    stacked                      {mae(holdout['price'], stacked):8.2f}")
    print("    The weights come from predictions made for unseen rows, so nothing")
    print("    about the holdout took part in choosing them.")
    # The stack both mixes the models and rescales them. Rescaling the best one
    # alone shows how much of the gain is only the rescaling.
    rescale = LinearRegression().fit(oof[[best_solo]], train["price"])
    rescaled = rescale.predict(holdout_predictions[[best_solo]])
    print(f"    {best_solo + ' alone, rescaled':<29}{mae(holdout['price'], rescaled):8.2f}   "
          f"(intercept {rescale.intercept_:.2f}, slope {rescale.coef_[0]:.4f})")

    # 5. Adding a model that is much worse and much more different
    print("\n--- 5. Adding a model that is much worse and much more different ---")
    all_names = names
    simple_four = holdout_predictions[all_names].mean(axis=1)
    stack_four = LinearRegression().fit(oof[all_names], train["price"])
    stacked_four = stack_four.predict(holdout_predictions[all_names])
    print(f"    neighbours alone             {solo['neighbours']:8.2f}   "
          f"({solo['neighbours'] / solo[best_solo]:.1f} times the best model's error)")
    print(f"    its error correlation with the boosters: "
          f"{error_correlation.loc['neighbours', booster_names].mean():.4f}")
    print(f"    simple average of all four   {mae(holdout['price'], simple_four):8.2f}")
    print(f"    stacked over all four        {mae(holdout['price'], stacked_four):8.2f}")
    for name, weight in zip(all_names, stack_four.coef_):
        print(f"    weight for {name:<12}{weight:>8.4f}")
    neighbour_weight = dict(zip(all_names, stack_four.coef_))["neighbours"]
    print("    An average has to take the weak model at full strength. The stack gave it")
    print(f"    a weight of {neighbour_weight:.4f} and moved "
          f"{mae(holdout['price'], stacked_four) - mae(holdout['price'], stacked):+.2f} MAE "
          "from the three-booster stack.")

    # 6. Every result in this run, next to its error correlation
    print("\n--- 6. Every result in this run, next to its error correlation ---")
    booster_correlation = np.mean(booster_pairs)
    all_correlation = float(error_correlation.values[np.triu_indices(4, 1)].mean())
    rows = [
        ("best single model", holdout_predictions[best_solo], None),
        ("average of 3 boosters", simple, booster_correlation),
        ("holdout-weighted average", weighted, booster_correlation),
        ("stack of 3 boosters", stacked, booster_correlation),
        ("average of 4", simple_four, all_correlation),
        ("stack of 4", stacked_four, all_correlation),
    ]
    rows = [(label, prediction, mae(holdout["price"], prediction), correlation)
            for label, prediction, correlation in rows]
    print(f"    {'combination':<28}{'holdout MAE':>13}{'gain':>9}{'error corr':>13}")
    for label, _, score, correlation in rows:
        gain = solo[best_solo] - score
        correlation_text = "-" if correlation is None else f"{correlation:.4f}"
        print(f"    {label:<28}{score:>13.2f}{gain:>9.2f}{correlation_text:>13}")

    # The winner is picked on the holdout, the leak part 3 names, one level up.
    combinations = rows[1:]
    winner_label, winner_prediction, winner_score, _ = min(combinations, key=lambda r: r[2])
    winner_gain = solo[best_solo] - winner_score
    paired = (np.abs(holdout["price"] - holdout_predictions[best_solo])
              - np.abs(holdout["price"] - winner_prediction)).to_numpy()
    rng = np.random.default_rng(SEED)
    draws = [paired[rng.integers(0, len(paired), len(paired))].mean()
             for _ in range(BOOTSTRAP)]
    low, high = np.percentile(draws, [2.5, 97.5])
    if low <= 0 <= high:
        verdict = "so it cannot be told apart from zero"
    elif low > 0:
        verdict = "so it is larger than the noise"
    else:
        verdict = "so it is a real loss"
    rescale_gain = solo[best_solo] - mae(holdout["price"], rescaled)
    worse = [r for r in combinations if r[2] > solo[best_solo]]
    print(f"\n    The best combination here is {winner_label}, with a gain of "
          f"{winner_gain:+.2f} MAE")
    print(f"    ({winner_gain / solo[best_solo]:+.2%}) over the best single model. "
          f"It was chosen by reading {len(combinations)}")
    print("    scores off this holdout and keeping the smallest. Part 4's weights came")
    print("    out of fold; this choice between whole routes did not.")
    print(f"    Resampling the holdout rows {BOOTSTRAP} times puts that gain between "
          f"{low:+.2f} and {high:+.2f},")
    print(f"    {verdict}. Rescaling {best_solo} alone gives {rescale_gain:+.2f} of it.")
    print(f"    {len(worse)} of the {len(combinations)} combinations came out worse "
          "than doing nothing.")
    print("    Two numbers decide whether a model helps a blend: its error, and how much")
    print("    its errors overlap the others'. The neighbour model overlapped least and")
    print(f"    still got a weight of {neighbour_weight:.4f}, because its error was "
          f"{solo['neighbours'] / solo[best_solo]:.1f} times")
    print("    the best model's.")


if __name__ == "__main__":
    main()
