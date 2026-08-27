"""Run nine classifiers over the same split, then move the one number none of them chose.

Demonstrates where a classification result actually comes from:
    1. Score the whole toolbox on an imbalanced table, against the accuracy of guessing.
    2. Score the same code on a near-separable table, and compare the two ceilings.
    3. Rescale the features and rerun, to sort the models that care from those that do not.
    4. Swap label encoding for one-hot on the same model, on the same split.
    5. Drop the two columns that never vary, and measure what that cost.
    6. Sweep the decision threshold and watch the ranking metric refuse to move.
    7. Force the predicted positive rate to match the observed one, and price it.
    8. Read the fitted coefficients back against the log-odds model that made the labels.

Module 07: Machine Learning and Deep Learning Foundations - Classifier Toolbox.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
ATTRITION = DATA / "employee_attrition.csv"
SPEAKER = DATA / "speaker_acoustics.csv"

SEED = 20260824
TEST_FRACTION = 0.25

CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "OverTime"]
CONSTANT_COLUMNS = ["EmployeeCount", "StandardHours"]
DROP_ALWAYS = ["employee_id", "Attrition"]

# The coefficients script 01 used to draw the labels. Section 8 asks how much of
# this a logistic regression recovers from 1800 rows.
TRUE_LOG_ODDS = {
    "OverTime": 1.25, "MaritalStatus_Single": 0.85, "BusinessTravel_Travel_Frequently": 0.55,
    "YearsAtCompany": -0.085, "MonthlyIncome": -0.000105, "JobSatisfaction": -0.24,
    "JobInvolvement": -0.20, "WorkLifeBalance": -0.17, "DistanceFromHome": 0.030,
    "Age": -0.019, "NumCompaniesWorked": 0.11, "StockOptionLevel": -0.22,
}


def build_models(scaled):
    """Return the toolbox. Names are printed as given, so they stay short.

    One note on installing this list. NGBoost declares no pandas requirement of
    its own, but it depends on lifelines, and lifelines caps pandas below 3.0.
    Installing NGBoost will therefore downgrade pandas unless it is pinned. The
    cap is precautionary rather than a real incompatibility: lifelines imports
    and NGBClassifier fits and predicts normally on pandas 3.0.5, which is the
    version every script in this module was verified against. Keeping pandas at
    3.x leaves one entry in `pip check` complaining about that declared bound,
    and nothing else.

    The reason to hold pandas at 3.x rather than accept the downgrade is that
    3.0 removed conversions that 2.x only warned about, so a script that runs on
    2.x is not yet known to run on 3.x. Calling float() on a one-element Series
    is the example this file actually tripped over.
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from ngboost import NGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
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
        ("ngboost", NGBClassifier(n_estimators=300, learning_rate=0.02,
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


def score_toolbox(x_train, x_test, y_train, y_test, scaled=False):
    """Fit every model and report ranking quality and accuracy at the default cut."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    rows = []
    for name, model in build_models(scaled):
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
    print("    Accuracy on a 16% positive class is mostly a report of the class")
    print("    balance. The ranking column is the one that separates the models.")

    print("\n--- 2. The same nine on the acoustic table ---")
    s_features, s_target = load_speaker()
    sx_train, sx_test, sy_train, sy_test = train_test_split(
        s_features, s_target, test_size=TEST_FRACTION, random_state=SEED, stratify=s_target)
    speaker_scores = score_toolbox(sx_train, sx_test, sy_train, sy_test)
    print_table(speaker_scores.sort_values("auc", ascending=False),
                ["model", "auc", "accuracy", "flagged"], [22, 9, 11, 10])
    print(f"\n    best AUC on attrition {attrition_scores['auc'].max():.4f}, "
          f"on acoustics {speaker_scores['auc'].max():.4f}")
    print(f"    spread between best and worst model: "
          f"attrition {attrition_scores['auc'].max() - attrition_scores['auc'].min():.4f}, "
          f"acoustics {speaker_scores['auc'].max() - speaker_scores['auc'].min():.4f}")
    print("    Same nine calls, same split code, two different ceilings. The")
    print("    ceiling belongs to the data. Choosing among the nine moves the")
    print("    result by less than the choice of table does.")

    print("\n--- 3. Which models care that the columns are on different scales ---")
    scaler = MinMaxScaler().fit(x_train)
    sx_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    sx_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    scaled_scores = score_toolbox(sx_train_scaled, sx_test_scaled, y_train, y_test, True)
    merged = attrition_scores[["model", "auc"]].merge(
        scaled_scores[["model", "auc"]], on="model", suffixes=("_raw", "_scaled"))
    merged["change"] = merged["auc_scaled"] - merged["auc_raw"]
    print_table(merged.sort_values("change", ascending=False),
                ["model", "auc_raw", "auc_scaled", "change"], [22, 10, 13, 10])
    print(f"\n    MonthlyIncome spans {x_train['MonthlyIncome'].min()} to "
          f"{x_train['MonthlyIncome'].max()}, JobSatisfaction spans "
          f"{x_train['JobSatisfaction'].min()} to {x_train['JobSatisfaction'].max()}")
    print("    A distance in that raw space is a distance in monthly income with a")
    print("    rounding error attached. A tree never computes a distance, so its")
    print("    row of this table is the control group.")

    print("\n--- 4. One-hot against label encoding, same model, same split ---")
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
    print("    Label encoding puts JobRole on an ordered axis it does not have.")
    print("    A deep enough tree can carve that axis back into the right pieces,")
    print("    which is why the difference here is small rather than absent.")

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

    print("\n--- 6. Moving the threshold on the best-ranking model ---")
    best_name = attrition_scores.sort_values("auc", ascending=False).iloc[0]["model"]
    best_model = dict(build_models(False))[best_name]
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
    print("    change it. Everything to its left is a business decision.")

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
    print("    Which row is better depends on what an interview costs against what")
    print("    losing an employee costs, and no metric in this script knows that.")

    print("\n--- 8. Reading the coefficients back against the generator ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    design = pd.get_dummies(pd.read_csv(ATTRITION).drop(columns=DROP_ALWAYS),
                            columns=CATEGORICAL, drop_first=False).astype(float)
    design["OverTime"] = design["OverTime_Yes"]
    standardiser = StandardScaler().fit(design)
    fitted = LogisticRegression(max_iter=4000, C=1.0, random_state=SEED).fit(
        standardiser.transform(design), target)
    coefficients = pd.Series(fitted.coef_[0], index=design.columns)
    ranked = coefficients.abs().sort_values(ascending=False)

    # The fit ran on standardised columns, so each fitted coefficient is a weight
    # per standard deviation of its column, not per unit. Comparing it against a
    # raw log-odds weight compares two different things. The generator's weights
    # are put on the same footing here by multiplying each one by the spread of
    # its own column, which is what makes the two orderings comparable at all.
    comparable = {term: truth * float(design[term].std())
                  for term, truth in TRUE_LOG_ODDS.items() if term in design.columns}
    true_order = sorted(comparable, key=lambda t: -abs(comparable[t]))

    print(f"    {'term':<36}{'raw weight':>12}{'x spread':>11}"
          f"{'true rank':>11}{'fitted rank':>13}{'sign':>7}")
    correct_signs = 0
    for position, term in enumerate(true_order, start=1):
        truth = TRUE_LOG_ODDS[term]
        rank = int(ranked.index.get_loc(term)) + 1
        agrees = np.sign(coefficients[term]) == np.sign(truth)
        correct_signs += int(agrees)
        print(f"    {term:<36}{truth:>12.4f}{comparable[term]:>11.4f}"
              f"{position:>11}{rank:>13}{'ok' if agrees else 'wrong':>7}")

    within_three = sum(1 for position, term in enumerate(true_order, start=1)
                       if abs(int(ranked.index.get_loc(term)) + 1 - position) <= 3)
    print(f"\n    signs recovered: {correct_signs} of {len(comparable)}")
    print(f"    terms whose fitted rank lands within three of the true rank: "
          f"{within_three} of {len(comparable)}")
    print(f"    strongest fitted term overall: {ranked.index[0]}")
    print("    Read against raw weights the ordering looks wrong: OverTime carries")
    print("    the heaviest weight in the generator and does not come out on top.")
    print("    It is a yes-or-no column, so its whole range is one step, while")
    print("    YearsAtCompany moves over decades. Weight per unit and weight per")
    print("    standard deviation are different questions, and a fitted")
    print("    coefficient only ever answers the second one.")
    print("    Even on the same footing the ordering only half survives: every")
    print("    sign is right, the strongest term is right, and most of the middle")
    print("    of the table is shuffled. Twelve overlapping effects and 1800 rows")
    print("    are enough to recover directions, not a league table.")


if __name__ == "__main__":
    main()
