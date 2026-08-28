"""Invent a label from one column, score a model that predicts it, then ask four rankings who mattered.

Demonstrates that a strong score can be a property of the label rather than the model:
    1. Build the label the way a project builds one when the outcome has not happened yet.
    2. Recover the rule that produced it using a single feature and one threshold.
    3. Train a gradient boosted model on the full feature set and read its scores.
    4. Retrain with the generating feature removed, and read the same scores again.
    5. Rank the features by split count and by gain, and compare the two orders.
    6. Rank them by permutation on held-out rows, and by mean absolute contribution.
    7. Put the four rankings side by side and read where they disagree.

Module 10: Applied Projects - Label Construction and Feature Attribution.
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

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path(__file__).parent / "data"
SEED = 20260828

# The label is defined as "assets a quarter from now clear one million". The future
# is not in the data, so the project simulates it: today's assets multiplied by a
# random growth factor. That one line is what the rest of this script is about.
GROWTH_LOW = 0.95
GROWTH_HIGH = 1.20
LABEL_THRESHOLD = 1_000_000

FEATURES = [
    "total_aum", "age", "city_tier", "monthly_txn_amount", "monthly_txn_count",
    "mobile_login_count", "branch_visit_count", "product_count",
    "deposit_balance", "wealth_balance", "fund_balance", "insurance_balance",
]
GENERATING_FEATURE = "total_aum"

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
    """Attach the simulated label, exactly as a project short of an outcome column would.

    Nothing here is careless in isolation. The growth factor is random, the
    threshold is a business rule, and the resulting rate looks reasonable. What
    makes it unusable is that the only column feeding it is already a feature.
    """
    rng = np.random.default_rng(SEED)
    frame = frame.copy()
    growth = rng.uniform(GROWTH_LOW, GROWTH_HIGH, size=len(frame))
    frame["future_aum"] = frame[GENERATING_FEATURE] * growth
    frame["label"] = (frame["future_aum"] >= LABEL_THRESHOLD).astype(int)
    return frame


def recover_the_rule(frame: pd.DataFrame) -> dict:
    """Find the single threshold on the generating feature that best reproduces the label.

    If one column and one comparison can reproduce most of the label, then a model
    given that column is not learning about customers. It is learning the line the
    label was drawn along, and its score measures how cleanly that line was drawn.
    """
    values = frame[GENERATING_FEATURE].to_numpy()
    labels = frame["label"].to_numpy()
    candidates = np.quantile(values, np.linspace(0.50, 0.999, 400))
    best = {"threshold": None, "accuracy": 0.0}
    for threshold in candidates:
        accuracy = float(((values >= threshold).astype(int) == labels).mean())
        if accuracy > best["accuracy"]:
            best = {"threshold": float(threshold), "accuracy": accuracy}

    always_zero = float((labels == 0).mean())
    best["positive_rate"] = float(labels.mean())
    best["majority_baseline"] = always_zero
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

    Split counts how often a feature was used to cut the data. Gain adds up how much
    each cut improved the objective. Permutation measures what breaks when the
    column is shuffled on held-out rows. Mean absolute contribution measures how far
    each feature pushed individual predictions. They answer four questions, and only
    the last two are about the model's behaviour on data it did not train on.
    """
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

    table["contribution"] = _mean_absolute_contribution(model, run["x_test"], features)
    return table


class _SklearnFacade(ClassifierMixin, BaseEstimator):
    """Wrap a trained booster so permutation_importance can call it like a classifier.

    The two base classes supply the estimator tags scikit-learn inspects before it
    will score anything, and the mixin has to come first so its classifier tag wins.
    """

    def __init__(self, booster=None):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def fit(self, x, y):  # noqa: D102 - required by the scikit-learn interface, never called
        return self

    def __sklearn_is_fitted__(self):
        return True

    def predict(self, x):
        return (self.booster.predict(x) >= 0.5).astype(int)

    def predict_proba(self, x):
        positive = self.booster.predict(x)
        return np.column_stack([1 - positive, positive])


def _mean_absolute_contribution(model, x_test: pd.DataFrame, features: list) -> np.ndarray:
    """Return each feature's mean absolute contribution to the predictions.

    SHAP values are used when the package is available; LightGBM's own contribution
    output is the fallback. Both decompose a single prediction into one number per
    feature plus a baseline, so the mean of their absolute values ranks features by
    how much they moved individual predictions rather than by how often they were used.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_test)
        if isinstance(values, list):
            values = values[-1]
    except Exception:
        # pred_contrib appends the baseline as a final column; drop it.
        values = model.predict(x_test, pred_contrib=True)[:, :len(features)]
    return np.abs(np.asarray(values)).mean(axis=0)


def print_ranking(table: pd.DataFrame) -> None:
    """Print each feature's rank under all four measures, ordered by gain."""
    ranked = table.copy()
    for column in ("split", "gain", "permutation", "contribution"):
        ranked[f"rank_{column}"] = ranked[column].rank(ascending=False, method="min").astype(int)
    ranked = ranked.sort_values("rank_gain")

    print(f"    {'feature':<22}{'split':>7}{'gain':>7}{'perm':>7}{'contrib':>9}"
          f"{'   gain value':>15}{'perm value':>13}")
    for row in ranked.itertuples():
        print(f"    {row.feature:<22}{row.rank_split:>7}{row.rank_gain:>7}"
              f"{row.rank_permutation:>7}{row.rank_contribution:>9}"
              f"{row.gain:>15,.0f}{row.permutation:>13.5f}")

    disagreements = sum(
        1 for row in ranked.itertuples()
        if len({row.rank_split, row.rank_gain, row.rank_permutation,
                row.rank_contribution}) > 1
    )
    print(f"\n    features whose rank is not the same under all four measures: "
          f"{disagreements} of {len(ranked)}")
    top_by = {
        column: ranked.loc[ranked[f"rank_{column}"] == 1, "feature"].iloc[0]
        for column in ("split", "gain", "permutation", "contribution")
    }
    for column, feature in top_by.items():
        print(f"        ranked first by {column:<14}{feature}")


def main() -> None:
    frame = attach_label(load_customers())

    print("--- 1. The label, as the project defines it ---")
    print(f"    future assets = {GENERATING_FEATURE} x uniform({GROWTH_LOW}, {GROWTH_HIGH})")
    print(f"    label         = future assets >= {LABEL_THRESHOLD:,}")
    print(f"    rows {len(frame):,}, positives {int(frame['label'].sum()):,} "
          f"({frame['label'].mean():.4f})")

    print("\n--- 2. Recovering the label from one column ---")
    rule = recover_the_rule(frame)
    print(f"    best single threshold on {GENERATING_FEATURE}   >= {rule['threshold']:,.0f}")
    print(f"    accuracy of that one comparison        {rule['accuracy']:.4f}")
    print(f"    accuracy of always answering 'no'      {rule['majority_baseline']:.4f}")
    print(f"    AUC of the raw column, no model at all {rule['auc_single_feature']:.4f}")
    print("\n    One column and one threshold already reproduce the label. Whatever a")
    print("    model scores from here is mostly a measurement of that fact.")

    print("\n--- 3. A model on the full feature set ---")
    full = train(frame, FEATURES)
    print(f"    features {len(FEATURES)}, boosting rounds {BOOST_ROUNDS}")
    print(f"    held-out AUC       {full['auc']:.4f}")
    print(f"    held-out accuracy  {full['accuracy']:.4f}")

    print(f"\n--- 4. The same model without {GENERATING_FEATURE} ---")
    reduced_features = [name for name in FEATURES if name != GENERATING_FEATURE]
    reduced = train(frame, reduced_features)
    print(f"    features {len(reduced_features)}")
    print(f"    held-out AUC       {reduced['auc']:.4f}   ({reduced['auc'] - full['auc']:+.4f})")
    print(f"    held-out accuracy  {reduced['accuracy']:.4f}   "
          f"({reduced['accuracy'] - full['accuracy']:+.4f})")
    print("\n    A small drop is the easiest result to misread here. It does not show that")
    print("    there was no leak; it shows the leak survived the removal, because other")
    print("    columns carry the same information:")
    correlations = (
        frame[reduced_features + [GENERATING_FEATURE]]
        .corr(numeric_only=True)[GENERATING_FEATURE]
        .drop(GENERATING_FEATURE)
        .abs()
        .sort_values(ascending=False)
    )
    for name, value in correlations.head(4).items():
        print(f"        |corr({name}, {GENERATING_FEATURE})| = {value:.3f}")
    print(f"\n    Dropping one column cannot undo a label defined by that column while its")
    print(f"    proxies remain. The only fix is a label that comes from an observed")
    print(f"    outcome rather than from a feature already in the table.")

    print("\n--- 5-7. Four rankings of the same features, full model ---")
    print_ranking(importance_table(full))

    print(f"\n--- 5-7. Four rankings again, {GENERATING_FEATURE} removed ---")
    print_ranking(importance_table(reduced))

    print("\n    Split counts how often a column was cut on, and a high-cardinality")
    print("    column collects splits whether or not they helped. Gain counts how much")
    print("    those cuts helped on training data. Permutation and contribution are the")
    print("    two measured on held-out rows, and they are the two that answer the")
    print("    question a reader thinks the chart is answering.")


if __name__ == "__main__":
    main()
