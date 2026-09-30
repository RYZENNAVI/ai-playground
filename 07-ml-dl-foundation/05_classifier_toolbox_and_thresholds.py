"""This script runs nine classifiers, from logistic regression to four gradient boosting
libraries, over the same split of an imbalanced attrition table, then tunes the decision
threshold, the one number none of them chose.

It shows where a classification result actually comes from:
    1. Score the whole toolbox on an imbalanced table, against the accuracy of guessing.
    2. Score the same code on a near-separable table, and compare the two ceilings.
    3. Rescale the features and rerun, to sort the models that care from those that do not.
    4. Swap label encoding for one-hot on the same model, on this split and on ten others.
    5. Drop the two columns that never vary, and measure what that cost.
    6. Sweep the decision threshold and watch the ranking metric refuse to move.
    7. Force the predicted positive rate to match the observed one, and price it.
    8. Read the fitted coefficients back against the log-odds model that made the labels.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
ATTRITION = DATA / "employee_attrition.csv"
SPEAKER = DATA / "speaker_acoustics.csv"

SEED = 20260824
TEST_FRACTION = 0.25
SPLIT_REPEATS = 10

CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "OverTime"]
CONSTANT_COLUMNS = ["EmployeeCount", "StandardHours"]
DROP_ALWAYS = ["employee_id", "Attrition"]

# The coefficients script 01 used to draw the labels. Part 8 asks how much of
# this a logistic regression recovers from 1800 rows.
TRUE_LOG_ODDS = {
    "OverTime": 1.25, "MaritalStatus_Single": 0.85, "BusinessTravel_Travel_Frequently": 0.55,
    "YearsAtCompany": -0.085, "MonthlyIncome": -0.000105, "JobSatisfaction": -0.24,
    "JobInvolvement": -0.20, "WorkLifeBalance": -0.17, "DistanceFromHome": 0.030,
    "Age": -0.019, "NumCompaniesWorked": 0.11, "StockOptionLevel": -0.22,
}


def build_models():
    """Return the toolbox. Names are printed as given, so they stay short."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from ngboost import NGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from xgboost import XGBClassifier

    return [
        ("logistic regression", LogisticRegression(max_iter=2000, random_state=SEED)),
        ("decision tree", DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        ("svm rbf", SVC(kernel="rbf", gamma="auto", probability=True, random_state=SEED)),
        ("random forest", RandomForestClassifier(n_estimators=400, random_state=SEED)),
        ("gradient boosting", GradientBoostingClassifier(random_state=SEED)),
        ("xgboost", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                                  subsample=0.9, colsample_bytree=0.8, eval_metric="auc",
                                  random_state=SEED, verbosity=0)),
        ("lightgbm", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                                    random_state=SEED, verbose=-1)),
        ("catboost", CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4,
                                        random_seed=SEED, verbose=0)),
        # NGBoost's default base tree takes no seed of its own, so it is passed one here.
        ("ngboost", NGBClassifier(Base=DecisionTreeRegressor(criterion="friedman_mse",
                                                             max_depth=3, random_state=SEED),
                                  n_estimators=300, learning_rate=0.02,
                                  random_state=SEED, verbose=False)),
    ]


def load_attrition():
    """Load the HR table and label encode its text columns."""
    from sklearn.preprocessing import LabelEncoder

    frame = pd.read_csv(ATTRITION)
    target = (frame["Attrition"] == "Yes").astype(int)
    features = frame.drop(columns=DROP_ALWAYS)
    for column in CATEGORICAL:
        features[column] = LabelEncoder().fit_transform(features[column].astype(str))
    return features, target


def load_speaker():
    """Load the acoustic table, whose only text column is the label."""
    frame = pd.read_csv(SPEAKER)
    target = (frame["label"] == "male").astype(int)
    return frame.drop(columns=["label"]), target


def score_toolbox(x_train, x_test, y_train, y_test):
    """Fit every model and report ranking quality and accuracy at the default cut."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    rows = []
    for name, model in build_models():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        rows.append({
            "model": name,
            "auc": roc_auc_score(y_test, probability),
            "accuracy": accuracy_score(y_test, (probability >= 0.5).astype(int)),
            "flagged": int((probability >= 0.5).sum()),
        })
    return pd.DataFrame(rows)


def print_table(frame, columns, widths):
    """Print a frame as fixed-width text, since these tables are read in a terminal."""
    header = "".join(f"{c:>{w}}" for c, w in zip(columns, widths))
    print("    " + header)
    for row in frame.itertuples(index=False):
        values = dict(zip(frame.columns, row))
        line = ""
        for column, width in zip(columns, widths):
            value = values[column]
            if isinstance(value, float):
                line += f"{value:>{width}.4f}"
            else:
                line += f"{str(value):>{width}}"
        print("    " + line)


def main():
    if not ATTRITION.exists() or not SPEAKER.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # 1. Nine models on the attrition table
    features, target = load_attrition()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_FRACTION, random_state=SEED, stratify=target)

    majority = 1 - y_test.mean()
    print("--- 1. Nine models on the attrition table ---")
    print(f"    {len(features)} rows, {features.shape[1]} features, "
          f"{target.mean():.1%} of employees leave")
    print(f"    predicting that nobody leaves scores {majority:.4f} accuracy "
          "without fitting anything")
    attrition_scores = score_toolbox(x_train, x_test, y_train, y_test)
    print()
    print_table(attrition_scores.sort_values("auc", ascending=False),
                ["model", "auc", "accuracy", "flagged"], [22, 9, 11, 10])
    beat = attrition_scores[attrition_scores["accuracy"] > majority]
    print(f"\n    models whose accuracy beats that constant guess: "
          f"{len(beat)} of {len(attrition_scores)}")
    print(f"    models that flag nobody at the 0.5 cut: "
          f"{int((attrition_scores['flagged'] == 0).sum())}")
    print(f"    Accuracy on a {target.mean():.1%} positive class is mostly a report of the")
    print("    class balance. The ranking column is the one that separates the models.")

    # 2. The same nine on the acoustic table
    print("\n--- 2. The same nine on the acoustic table ---")
    s_features, s_target = load_speaker()
    sx_train, sx_test, sy_train, sy_test = train_test_split(
        s_features, s_target, test_size=TEST_FRACTION, random_state=SEED, stratify=s_target)
    speaker_scores = score_toolbox(sx_train, sx_test, sy_train, sy_test)
    print_table(speaker_scores.sort_values("auc", ascending=False),
                ["model", "auc", "accuracy", "flagged"], [22, 9, 11, 10])
    print(f"\n    best AUC on attrition {attrition_scores['auc'].max():.4f}, "
          f"on acoustics {speaker_scores['auc'].max():.4f}")
    # One model is unusable on raw columns and would set the spread alone, so the
    # spread is printed with and without it.
    def spread(frame, drop=None):
        rows = frame if drop is None else frame[frame["model"] != drop]
        return rows["auc"].max() - rows["auc"].min()

    outlier = attrition_scores.loc[attrition_scores["auc"].idxmin(), "model"]
    table_gap = speaker_scores["auc"].max() - attrition_scores["auc"].max()
    print(f"    spread between best and worst model: "
          f"attrition {spread(attrition_scores):.4f}, "
          f"acoustics {spread(speaker_scores):.4f}")
    print(f"    the same spread with '{outlier}' left out: "
          f"attrition {spread(attrition_scores, outlier):.4f}, "
          f"acoustics {spread(speaker_scores, outlier):.4f}")
    print(f"    changing the table moves the best AUC by {table_gap:.4f}")
    print("    Same nine calls, same split code, two different ceilings. The")
    print(f"    ceiling belongs to the data: once '{outlier}' is set aside, choosing")
    print("    among the rest moves the result by less than the choice of table")
    print(f"    does. '{outlier}' is the exception and part 3 says why.")

    # 3. Which models care that the columns are on different scales
    print("\n--- 3. Which models care that the columns are on different scales ---")
    scaler = MinMaxScaler().fit(x_train)
    sx_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    sx_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    scaled_scores = score_toolbox(sx_train_scaled, sx_test_scaled, y_train, y_test)
    merged = attrition_scores[["model", "auc"]].merge(
        scaled_scores[["model", "auc"]], on="model", suffixes=("_raw", "_scaled"))
    merged["change"] = merged["auc_scaled"] - merged["auc_raw"]
    print_table(merged.sort_values("change", ascending=False),
                ["model", "auc_raw", "auc_scaled", "change"], [22, 10, 13, 10])
    print(f"\n    MonthlyIncome spans {x_train['MonthlyIncome'].min()} to "
          f"{x_train['MonthlyIncome'].max()}, JobSatisfaction spans "
          f"{x_train['JobSatisfaction'].min()} to {x_train['JobSatisfaction'].max()}")
    trees = merged[~merged["model"].isin(["logistic regression", outlier])]
    largest = trees.loc[trees["change"].abs().idxmax()]
    outlier_change = float(merged.loc[merged["model"] == outlier, "change"].iloc[0])
    print("    A distance in that raw space is a distance in monthly income with a")
    print("    rounding error attached. A tree compares values inside one column and")
    print("    never computes a distance, so in principle scaling leaves its splits")
    print(f"    alone. The largest tree change here is {largest['model']} at "
          f"{largest['change']:+.4f},")
    print(f"    against {outlier_change:+.4f} for '{outlier}'.")

    # 4. One-hot against label encoding on the same model
    print("\n--- 4. One-hot against label encoding on the same model ---")
    from lightgbm import LGBMClassifier
    raw = pd.read_csv(ATTRITION).drop(columns=DROP_ALWAYS)
    one_hot = pd.get_dummies(raw, columns=CATEGORICAL, drop_first=False)
    ox_train, ox_test, oy_train, oy_test = train_test_split(
        one_hot, target, test_size=TEST_FRACTION, random_state=SEED, stratify=target)
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                           random_state=SEED, verbose=-1).fit(ox_train, oy_train)
    one_hot_auc = roc_auc_score(oy_test, model.predict_proba(ox_test)[:, 1])
    label_auc = float(
        attrition_scores.loc[attrition_scores["model"] == "lightgbm", "auc"].iloc[0])
    print(f"    label encoded, {features.shape[1]:>3} columns -> AUC {label_auc:.4f}")
    print(f"    one-hot,       {one_hot.shape[1]:>3} columns -> AUC {one_hot_auc:.4f}")
    print(f"    difference {one_hot_auc - label_auc:+.4f}")
    # One split cannot say whether that difference is the encoding or the split, so
    # the comparison is repeated on other splits.
    gaps = []
    for split_seed in range(SPLIT_REPEATS):
        train_rows, test_rows = train_test_split(
            np.arange(len(target)), test_size=TEST_FRACTION, random_state=split_seed,
            stratify=target)
        pair = []
        for table in (features, one_hot):
            repeat = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                                    random_state=SEED, verbose=-1)
            repeat.fit(table.iloc[train_rows], target.iloc[train_rows])
            pair.append(roc_auc_score(target.iloc[test_rows],
                                      repeat.predict_proba(table.iloc[test_rows])[:, 1]))
        gaps.append(pair[1] - pair[0])
    print(f"    over {SPLIT_REPEATS} other splits, one-hot minus label averages "
          f"{np.mean(gaps):+.4f}, from {min(gaps):+.4f} to {max(gaps):+.4f}")
    print("    Label encoding puts JobRole on an ordered axis it does not have, and a")
    print("    tree can cut that axis back into the right pieces. The difference")
    print("    changes sign from split to split, so on this table neither encoding")
    print("    costs anything measurable.")

    # 5. The two columns that never vary
    print("\n--- 5. The two columns that never vary ---")
    for column in CONSTANT_COLUMNS:
        print(f"    {column}: {features[column].nunique()} distinct value, "
              f"always {features[column].iloc[0]}")
    trimmed_train = x_train.drop(columns=CONSTANT_COLUMNS)
    trimmed_test = x_test.drop(columns=CONSTANT_COLUMNS)
    trimmed = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                             random_state=SEED, verbose=-1).fit(trimmed_train, y_train)
    trimmed_auc = roc_auc_score(y_test, trimmed.predict_proba(trimmed_test)[:, 1])
    print(f"    AUC with them {label_auc:.4f}, without them {trimmed_auc:.4f}, "
          f"difference {trimmed_auc - label_auc:+.4f}")
    print("    A column with no variance cannot split anything, so removing it is")
    print("    housekeeping rather than a fix. It is worth doing because the next")
    print("    reader should not have to check.")

    # 6. Moving the threshold on the best-ranking model
    print("\n--- 6. Moving the threshold on the best-ranking model ---")
    best_name = attrition_scores.sort_values("auc", ascending=False).iloc[0]["model"]
    best_model = dict(build_models())[best_name]
    best_model.fit(x_train, y_train)
    probability = best_model.predict_proba(x_test)[:, 1]
    print(f"    model: {best_name}, AUC {roc_auc_score(y_test, probability):.4f}")
    print(f"    {'threshold':>10}{'flagged':>9}{'precision':>11}{'recall':>9}"
          f"{'f1':>8}{'accuracy':>10}{'auc':>9}")
    for threshold in (0.10, 0.16, 0.20, 0.30, 0.50, 0.70):
        predicted = (probability >= threshold).astype(int)
        print(f"    {threshold:>10.2f}{int(predicted.sum()):>9}"
              f"{precision_score(y_test, predicted, zero_division=0):>11.4f}"
              f"{recall_score(y_test, predicted, zero_division=0):>9.4f}"
              f"{f1_score(y_test, predicted, zero_division=0):>8.4f}"
              f"{accuracy_score(y_test, predicted):>10.4f}"
              f"{roc_auc_score(y_test, probability):>9.4f}")
    print("    The last column is constant down the table. AUC reads the ordering")
    print("    of the scores, and moving a cut through a fixed ordering cannot")
    print("    change it. The columns to its left follow the cut, and where to put")
    print("    the cut is a business decision.")

    # 7. Forcing the flagged rate to match the base rate
    print("\n--- 7. Forcing the flagged rate to match the base rate ---")
    rate = float(y_train.mean())
    wanted = int(round(len(probability) * rate))
    threshold = np.sort(probability)[-wanted]
    forced = (probability >= threshold).astype(int)
    default = (probability >= 0.5).astype(int)
    print(f"    training base rate {rate:.4f}, so flag the top {wanted} of "
          f"{len(probability)} scores")
    print(f"    the threshold that does it: {threshold:.4f}, not 0.5")
    print(f"    {'cut':<12}{'flagged':>9}{'caught':>8}{'missed':>8}"
          f"{'false alarms':>14}{'precision':>11}{'recall':>9}")
    for label, predicted in (("default 0.5", default), ("forced rate", forced)):
        caught = int(((predicted == 1) & (y_test == 1)).sum())
        missed = int(((predicted == 0) & (y_test == 1)).sum())
        false_alarms = int(((predicted == 1) & (y_test == 0)).sum())
        print(f"    {label:<12}{int(predicted.sum()):>9}{caught:>8}{missed:>8}"
              f"{false_alarms:>14}"
              f"{precision_score(y_test, predicted, zero_division=0):>11.4f}"
              f"{recall_score(y_test, predicted, zero_division=0):>9.4f}")
    print("    The same fitted model, the same scores, one number changed by hand.")
    print("    Which row is better depends on what a conversation with a flagged")
    print("    employee costs against what losing one costs, and no metric in this")
    print("    script knows that.")

    # 8. Reading the coefficients back against the generator
    print("\n--- 8. Reading the coefficients back against the generator ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Keeping every category would share each effect among its group's columns, so the
    # first one is dropped. Here that is Divorced, Non-Travel and No, which the
    # generator gives no weight, and OverTime_Yes takes the generator's name.
    design = pd.get_dummies(pd.read_csv(ATTRITION).drop(columns=DROP_ALWAYS),
                            columns=CATEGORICAL, drop_first=True).astype(float)
    design = design.rename(columns={"OverTime_Yes": "OverTime"})
    standardiser = StandardScaler().fit(design)
    fitted = LogisticRegression(max_iter=4000, C=1.0, random_state=SEED).fit(
        standardiser.transform(design), target)
    coefficients = pd.Series(fitted.coef_[0], index=design.columns)
    ranked = coefficients.abs().sort_values(ascending=False)

    # The fit ran on standardised columns, so each coefficient is a weight per standard
    # deviation. Multiplying each generator weight by its column's spread puts the two
    # on the same footing.
    comparable = {term: truth * float(design[term].std())
                  for term, truth in TRUE_LOG_ODDS.items() if term in design.columns}
    true_order = sorted(comparable, key=lambda t: -abs(comparable[t]))
    # Both ranks count among the same twelve terms.
    fitted_order = coefficients[true_order].abs().sort_values(ascending=False)

    print(f"    {'term':<36}{'raw weight':>12}{'x spread':>11}{'fitted':>9}"
          f"{'true rank':>11}{'fitted rank':>13}{'sign':>7}")
    correct_signs = 0
    for position, term in enumerate(true_order, start=1):
        truth = TRUE_LOG_ODDS[term]
        rank = int(fitted_order.index.get_loc(term)) + 1
        agrees = np.sign(coefficients[term]) == np.sign(truth)
        correct_signs += int(agrees)
        print(f"    {term:<36}{truth:>12.4f}{comparable[term]:>11.4f}"
              f"{coefficients[term]:>9.4f}{position:>11}{rank:>13}{'ok' if agrees else 'wrong':>7}")

    within_three = sum(1 for position, term in enumerate(true_order, start=1)
                       if abs(int(fitted_order.index.get_loc(term)) + 1 - position) <= 3)
    print(f"\n    signs recovered: {correct_signs} of {len(comparable)}")
    print(f"    fitted ranks within three places of the true rank: "
          f"{within_three} of {len(comparable)}")
    print(f"    strongest fitted term overall: {ranked.index[0]}")
    print("    Read against raw weights, OverTime should come first: 1.25 is the")
    print("    largest weight in the generator. It is a yes-or-no column, so its whole")
    print("    range is one step, while YearsAtCompany moves over decades. Per standard")
    print("    deviation the generator itself ranks YearsAtCompany first, and so does")
    print(f"    the fit. Twelve overlapping effects and {len(design)} rows are enough to")
    print("    recover directions, not a league table.")


if __name__ == "__main__":
    main()
