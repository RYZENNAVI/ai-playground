"""This script shows target leakage in a customer table: the label (assets a quarter from now
above one million) is total_aum times a random growth factor, and total_aum is also a feature.
It trains a LightGBM gradient boosted model on that label and ranks the features four ways:
split count, gain, permutation importance and mean absolute SHAP value.

    1. Build the label and print how many rows are positive.
    2. Reproduce the label with one threshold on total_aum, without a model.
    3. Train on all twelve features and score on held-out rows.
    4. Retrain without total_aum and list the columns that still track it.
    5. Rank the features of the step 3 model four ways.
    6. Rank the features of the step 4 model the same way.
"""

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path(__file__).parent / "data"
SEED = 20260828

# The label is defined as "assets a quarter from now clear one million". The future
# is not in the data, so the project simulates it: today's assets multiplied by a
# random growth factor.
GROWTH_LOW = 0.95
GROWTH_HIGH = 1.20
LABEL_THRESHOLD = 1_000_000

FEATURES = [
    "total_aum", "age", "city_tier", "monthly_txn_amount", "monthly_txn_count",
    "mobile_login_count", "branch_visit_count", "product_count",
    "deposit_balance", "wealth_balance", "fund_balance", "insurance_balance",
]
GENERATING_FEATURE = "total_aum"
# 01 draws age on its own, unrelated to assets or the label, so any rank it earns is noise.
INDEPENDENT_FEATURE = "age"
MEASURES = ("split", "gain", "permutation", "contribution")

BOOST_ROUNDS = 200
TEST_SIZE = 0.25


def load_customers() -> pd.DataFrame:
    """Read the customer table and derive the handful of features a project would add."""
    path = DATA / "customers.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path.name}. Run 01_build_project_datasets.py first.")
    frame = pd.read_csv(path)
    frame["product_count"] = (
        (frame["deposit_balance"] > 0).astype(int)
        + (frame["wealth_balance"] > 0).astype(int)
        + (frame["fund_balance"] > 0).astype(int)
        + (frame["insurance_balance"] > 0).astype(int)
    )
    return frame


def attach_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the simulated label: total_aum times a random growth factor, against the threshold."""
    rng = np.random.default_rng(SEED)
    frame = frame.copy()
    growth = rng.uniform(GROWTH_LOW, GROWTH_HIGH, size=len(frame))
    frame["future_aum"] = frame[GENERATING_FEATURE] * growth
    frame["label"] = (frame["future_aum"] >= LABEL_THRESHOLD).astype(int)
    return frame


def recover_the_rule(frame: pd.DataFrame) -> dict:
    """Find the single threshold on the generating feature that best reproduces the label."""
    values = frame[GENERATING_FEATURE].to_numpy()
    labels = frame["label"].to_numpy()
    candidates = np.quantile(values, np.linspace(0.50, 0.999, 400))
    best = {"threshold": None, "accuracy": 0.0}
    for threshold in candidates:
        accuracy = float(((values >= threshold).astype(int) == labels).mean())
        if accuracy > best["accuracy"]:
            best = {"threshold": float(threshold), "accuracy": accuracy}

    best["majority_baseline"] = float((labels == 0).mean())
    best["auc_single_feature"] = float(roc_auc_score(labels, values))
    return best


def train(frame: pd.DataFrame, features: list) -> dict:
    """Train one gradient boosted model on the given features and score it on held-out rows."""
    x_train, x_test, y_train, y_test = train_test_split(
        frame[features], frame["label"],
        test_size=TEST_SIZE, random_state=SEED, stratify=frame["label"],
    )
    model = lgb.train(
        {"objective": "binary", "metric": "auc", "verbosity": -1,
         "seed": SEED, "deterministic": True, "force_col_wise": True},
        lgb.Dataset(x_train, y_train),
        num_boost_round=BOOST_ROUNDS,
    )
    probability = model.predict(x_test)
    return {
        "model": model,
        "features": features,
        "x_test": x_test,
        "y_test": y_test,
        "auc": float(roc_auc_score(y_test, probability)),
        "accuracy": float(accuracy_score(y_test, (probability >= 0.5).astype(int))),
    }


def importance_table(run: dict) -> pd.DataFrame:
    """Collect four rankings of the same features from the same fitted model.
    Split and gain come from training; permutation and contribution from held-out rows."""
    model, features = run["model"], run["features"]
    table = pd.DataFrame({
        "feature": features,
        "split": model.feature_importance(importance_type="split"),
        "gain": model.feature_importance(importance_type="gain"),
    })

    wrapped = _SklearnFacade(model)
    permuted = permutation_importance(
        wrapped, run["x_test"], run["y_test"],
        n_repeats=5, random_state=SEED, scoring="roc_auc",
    )
    table["permutation"] = permuted.importances_mean
    table["permutation_sd"] = permuted.importances_std

    # pred_contrib returns TreeSHAP values plus the baseline as a final column; drop it.
    contributions = model.predict(run["x_test"], pred_contrib=True)[:, :len(features)]
    table["contribution"] = np.abs(contributions).mean(axis=0)
    return table


class _SklearnFacade(ClassifierMixin, BaseEstimator):
    """Wrap a trained booster so permutation_importance can call it like a classifier.
    scikit-learn checks estimator tags first, and the mixin must come first for its tag to win."""

    def __init__(self, booster=None):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def fit(self, x, y):  # noqa: D102; required by the scikit-learn interface, never called
        return self

    def __sklearn_is_fitted__(self):
        return True

    def predict(self, x):
        return (self.booster.predict(x) >= 0.5).astype(int)

    def predict_proba(self, x):
        positive = self.booster.predict(x)
        return np.column_stack([1 - positive, positive])


def print_ranking(table: pd.DataFrame) -> None:
    """Print each feature's rank under all four measures, ordered by gain, and how they agree."""
    ranked = table.copy()
    for column in MEASURES:
        ranked[f"rank_{column}"] = ranked[column].rank(ascending=False, method="min").astype(int)
    ranked = ranked.sort_values("rank_gain")

    print(f"    {'feature':<22}{'split':>7}{'gain':>7}{'perm':>7}{'contrib':>9}"
          f"{'   gain value':>15}{'perm value':>13}{'perm sd':>10}")
    for row in ranked.itertuples():
        print(f"    {row.feature:<22}{row.rank_split:>7}{row.rank_gain:>7}"
              f"{row.rank_permutation:>7}{row.rank_contribution:>9}"
              f"{row.gain:>15,.0f}{row.permutation:>13.5f}{row.permutation_sd:>10.5f}")

    correlation = ranked[[f"rank_{column}" for column in MEASURES]].corr(method="spearman")
    print(f"\n    rank correlation (Spearman){'split':>9}{'gain':>7}{'perm':>7}{'contrib':>9}")
    for column, short in zip(MEASURES, ("split", "gain", "perm", "contrib")):
        values = correlation.loc[f"rank_{column}"].to_numpy()
        print(f"        {short:<23}{values[0]:>9.2f}{values[1]:>7.2f}"
              f"{values[2]:>7.2f}{values[3]:>9.2f}")

    above_noise = int((ranked["permutation"] > 2 * ranked["permutation_sd"]).sum())
    print(f"\n    permutation mean above twice its sd: {above_noise} of {len(ranked)}")
    top_by = {
        column: ranked.loc[ranked[f"rank_{column}"] == 1, "feature"].iloc[0]
        for column in MEASURES
    }
    for column, feature in top_by.items():
        print(f"        ranked first by {column:<14}{feature}")
    independent = ranked.loc[ranked["feature"] == INDEPENDENT_FEATURE].iloc[0]
    print(f"    {INDEPENDENT_FEATURE}, drawn independently of the label in 01: "
          f"split {independent.rank_split}, gain {independent.rank_gain}, "
          f"perm {independent.rank_permutation}, contrib {independent.rank_contribution}")


def main() -> None:
    # 1. The label

    frame = attach_label(load_customers())

    print("--- 1. The label, as the project defines it ---")
    print(f"    future assets = {GENERATING_FEATURE} x uniform({GROWTH_LOW}, {GROWTH_HIGH})")
    print(f"    label         = future assets >= {LABEL_THRESHOLD:,}")
    print(f"    rows {len(frame):,}, positives {int(frame['label'].sum()):,} "
          f"({frame['label'].mean():.4f})")

    # 2. Recover the label from one column

    print("\n--- 2. Recovering the label from one column ---")
    rule = recover_the_rule(frame)
    print(f"    best single threshold on {GENERATING_FEATURE}   >= {rule['threshold']:,.0f}")
    print(f"    accuracy of that one comparison        {rule['accuracy']:.4f}")
    print(f"    accuracy of always answering 'no'      {rule['majority_baseline']:.4f}")
    print(f"    AUC of the raw column, no model at all {rule['auc_single_feature']:.4f}")
    lowest, highest = LABEL_THRESHOLD / GROWTH_HIGH, LABEL_THRESHOLD / GROWTH_LOW
    column = frame[GENERATING_FEATURE]
    undecided = float(((column > lowest) & (column < highest)).mean())
    print(f"    rows the growth factor can still decide {undecided:.4f}   "
          f"({GENERATING_FEATURE} between {lowest:,.0f} and {highest:,.0f})")
    wrong_rule = round((1 - rule["accuracy"]) * len(frame))
    wrong_no = round((1 - rule["majority_baseline"]) * len(frame))
    print(f"\n    One comparison on {GENERATING_FEATURE} gets {wrong_rule:,} of {len(frame):,} "
          f"rows wrong; answering 'no'")
    print(f"    every time gets {wrong_no:,} wrong. A model is measured against that comparison.")

    # 3. A model on all features

    print("\n--- 3. A model on the full feature set ---")
    full = train(frame, FEATURES)
    print(f"    features {len(FEATURES)}, boosting rounds {BOOST_ROUNDS}")
    print(f"    held-out AUC       {full['auc']:.4f}")
    print(f"    held-out accuracy  {full['accuracy']:.4f}")
    raw_held_out = float(roc_auc_score(full["y_test"], full["x_test"][GENERATING_FEATURE]))
    print(f"    raw {GENERATING_FEATURE} AUC on the same held-out rows {raw_held_out:.4f}")

    # 4. The same model without the generating feature

    print(f"\n--- 4. The same model without {GENERATING_FEATURE} ---")
    reduced_features = [name for name in FEATURES if name != GENERATING_FEATURE]
    reduced = train(frame, reduced_features)
    print(f"    features {len(reduced_features)}")
    print(f"    held-out AUC       {reduced['auc']:.4f}   ({reduced['auc'] - full['auc']:+.4f})")
    print(f"    held-out accuracy  {reduced['accuracy']:.4f}   "
          f"({reduced['accuracy'] - full['accuracy']:+.4f})")
    print(f"\n    Removing {GENERATING_FEATURE} costs {full['auc'] - reduced['auc']:.4f} AUC. "
          f"That does not mean the leak is gone:")
    print(f"    01 draws these columns from {GENERATING_FEATURE}, and they stay in the table.")
    correlations = (
        frame[reduced_features + [GENERATING_FEATURE]]
        .corr(numeric_only=True)[GENERATING_FEATURE]
        .drop(GENERATING_FEATURE)
        .abs()
        .sort_values(ascending=False)
    )
    for name, value in correlations.head(4).items():
        print(f"        |corr({name}, {GENERATING_FEATURE})| = {value:.3f}")
    print("\n    Dropping one column cannot undo a label defined by that column while its")
    print("    proxies remain. The only fix is a label that comes from an observed")
    print("    outcome rather than from a feature already in the table.")

    # 5. Four rankings, full model

    print("\n--- 5. Four rankings of the step 3 model ---")
    print_ranking(importance_table(full))

    # 6. Four rankings, generating feature removed

    print(f"\n--- 6. Four rankings of the step 4 model, {GENERATING_FEATURE} removed ---")
    print_ranking(importance_table(reduced))

    print("\n    Split and gain are read from the training data. Contribution is measured on")
    print("    held-out rows, but it follows what the model uses, not whether that helps.")
    print("    Permutation asks what the held-out score loses, and its sd says which of")
    print(f"    those losses are noise. The {INDEPENDENT_FEATURE} line shows the difference on a")
    print("    column with no signal.")


if __name__ == "__main__":
    main()
