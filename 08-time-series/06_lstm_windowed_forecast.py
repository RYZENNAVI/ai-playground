"""This script turns the cash-flow series into supervised rows with a sliding window, splits
them by time and trains an LSTM on them for a one-step-ahead forecast, keeping the two ends of
the series apart throughout. The run is about the part of sequence forecasting that is not the
network, and prints 6 parts:
    1. Slide a window over the series and read off what one supervised row contains.
    2. Split those rows at random, and count how many observations end up on both sides. This
       part is for comparison only and nothing later uses it.
    3. Split them by time into train, validation and final test, scaling from the train side only.
    4. Train a recurrent model on train, selecting the checkpoint on validation alone.
    5. Open the final test once, and score it against two practical baselines and one oracle.
    6. Score the model on the training and validation rows too, and compare the three numbers.

The forecast is one step ahead on purpose. The step 5 baselines are single-step comparisons, and
with a larger HORIZON they would broadcast against the wider target block instead of failing, so
an assertion pins HORIZON to 1.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 20260827
WINDOW = 14
HORIZON = 1
# The last TEST_DAYS rows are the final test: nothing fits on them, scores them or chooses
# from them before step 5. The VALID_DAYS rows before them are what training may watch.
# Step 2 deals every row at random and step 3 scales them all; neither reaches a choice.
TEST_DAYS = 60
VALID_DAYS = 60
HIDDEN = 48
EPOCHS = 120
LEARNING_RATE = 0.01

assert HORIZON == 1, "the step 5 baselines are written for one-step-ahead forecasting"
assert WINDOW >= 7, "the last-week baseline reads seven positions back in the window"


def series_to_supervised(values: np.ndarray, window: int, horizon: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Turn one long series into rows of (window observations -> next horizon observations).

    Neighbouring rows share window - 1 observations, which is why the split has to be made by time.
    """
    rows_x, rows_y = [], []
    for start in range(len(values) - window - horizon + 1):
        rows_x.append(values[start:start + window])
        rows_y.append(values[start + window:start + window + horizon])
    return np.array(rows_x), np.array(rows_y)


def touched_observations(rows: np.ndarray, window: int, horizon: int,
                         total: int) -> set:
    """Return every observation index the given supervised rows read or predict.

    A row that reaches past total means the rows and the series do not match, so it fails.
    """
    marks: set = set()
    for i in rows:
        end = int(i) + window + horizon
        assert end <= total, f"row {i} reaches observation {end} of a {total}-point series"
        marks.update(range(int(i), end))
    return marks


def shared_observation_count(left_rows: np.ndarray, right_rows: np.ndarray,
                             window: int, horizon: int, total: int) -> int:
    """Count observations of the original series that appear on both sides of a split.

    A row index fixes the stretch a row covers, so the count is exact rather than estimated.
    """
    return len(touched_observations(left_rows, window, horizon, total)
               & touched_observations(right_rows, window, horizon, total))


def input_observations(rows: np.ndarray, window: int, total: int) -> set:
    """Return the observation indices these rows read as inputs, targets excluded."""
    marks: set = set()
    for i in rows:
        end = int(i) + window
        assert end <= total, f"row {i} reads observation {end} of a {total}-point series"
        marks.update(range(int(i), end))
    return marks


def target_observations(rows: np.ndarray, window: int, horizon: int) -> list:
    """Return the observation indices these rows are asked to predict."""
    return [int(i) + window + k for i in rows for k in range(horizon)]


class Forecaster(nn.Module):
    """One recurrent layer reading the window, and one linear layer reading its last state."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(batch)
        return self.head(output[:, -1, :])


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, in the units of the series."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    flow = pd.read_csv(DATA_DIR / "fund_flow_daily.csv",
                       parse_dates=["report_date"], date_format="%Y%m%d")
    flow = flow.set_index("report_date")
    series = flow["total_purchase_amt"].astype(float)
    # Every split below is positional: "the first N rows" only means "the earliest
    # N rows" if the index really is in time order with no repeats, and a target
    # column with a gap in it would put a NaN into a window without raising.
    # These are the three assumptions the whole script rests on, so they are
    # checked rather than assumed.
    assert series.index.is_monotonic_increasing, "the index is not in time order"
    assert not series.index.has_duplicates, "the index repeats a date"
    assert series.notna().all(), "the target column has missing values"
    values = series.to_numpy()

    print("--- 1. Slide a window over the series ---")
    features, targets = series_to_supervised(values, WINDOW, HORIZON)
    print(f"  {len(values)} observations -> {len(features)} rows of "
          f"{WINDOW} inputs and {HORIZON} target")
    print(f"  feature block {features.shape}, target block {targets.shape}")
    print(f"  row 0 covers {series.index[0].date()} .. "
          f"{series.index[WINDOW - 1].date()}, predicting "
          f"{series.index[WINDOW].date()}")
    print(f"  row 1 covers {series.index[1].date()} .. "
          f"{series.index[WINDOW].date()}, predicting "
          f"{series.index[WINDOW + 1].date()}")
    print(f"  the two rows share {WINDOW - 1} of their {WINDOW} inputs, and row 0's "
          f"target is one of row 1's inputs")
    print(f"  a window of {WINDOW} covers two full weeks, so the weekly cycle "
          f"the generator planted is visible inside every single row")

    print("\n--- 2. Split those rows at random ---")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(features))
    cut = len(features) - TEST_DAYS
    random_train_idx, random_test_idx = order[:cut], order[cut:]
    shared = shared_observation_count(random_train_idx, random_test_idx,
                                      WINDOW, HORIZON, len(values))
    test_touched = len(touched_observations(random_test_idx, WINDOW, HORIZON, len(values)))
    print(f"  {len(random_train_idx)} training rows, {len(random_test_idx)} test rows")
    print(f"  observations the test rows touch: {test_touched}")
    print(f"  of those, {shared} also appear in a training row "
          f"({shared / test_touched:.0%})")
    random_train_set = set(random_train_idx.tolist())
    neighbours = sum((i - 1 in random_train_set) or (i + 1 in random_train_set)
                     for i in random_test_idx)
    print(f"  {neighbours} of {len(random_test_idx)} test rows have an immediate "
          f"neighbour in the training set, differing from it by one step")
    # Inputs only: a training row's own target was scored against, not shown to the model.
    train_inputs = input_observations(random_train_idx, WINDOW, len(values))
    test_targets = target_observations(random_test_idx, WINDOW, HORIZON)
    seen_targets = sum(j in train_inputs for j in test_targets)
    print(f"  test targets already present in a training row: {seen_targets} of "
          f"{len(test_targets)} ({seen_targets / len(test_targets):.0%})")
    if seen_targets == len(test_targets):
        print("  that is the direct form of the problem: the value each test row is asked "
              "to predict was already read, as an input, by the rows the weights were "
              "fitted on")
        print("  a score measured on these rows answers how well the model interpolates "
              "inside history it has already read, which is not the question a forecast asks")

    print("\n--- 3. Split by time into train, validation and final test ---")
    valid_start = len(features) - TEST_DAYS - VALID_DAYS
    test_start = len(features) - TEST_DAYS
    train_x_raw, train_y_raw = features[:valid_start], targets[:valid_start]
    valid_x_raw, valid_y_raw = (features[valid_start:test_start],
                                targets[valid_start:test_start])
    test_x_raw, test_y_raw = features[test_start:], targets[test_start:]
    ordered_shared = shared_observation_count(
        np.arange(valid_start), np.arange(test_start, len(features)),
        WINDOW, HORIZON, len(values))
    print(f"  train      {len(train_x_raw):>3} rows, up to "
          f"{series.index[valid_start + WINDOW - 1].date()}")
    print(f"  validation {len(valid_x_raw):>3} rows, watched during training")
    print(f"  final test {len(test_x_raw):>3} rows, not fitted on, scored, or "
          f"selected from before step 5")
    print(f"  observations shared between train and final test: {ordered_shared}")
    boundary = shared_observation_count(
        np.arange(valid_start), np.arange(valid_start, test_start),
        WINDOW, HORIZON, len(values))
    print(f"  between train and validation: {boundary} (the {WINDOW + HORIZON - 1} at the "
          f"boundary, which is unavoidable: the first validation window is made of days "
          f"that had already happened, and using them is what forecasting is)")
    centre = float(train_x_raw.mean())
    spread = float(train_x_raw.std())
    print(f"  scaling centre {centre:,.0f} and spread {spread:,.0f}, computed on the "
          f"training rows only, not on validation or the final test")
    full_centre = float(features.mean())
    print(f"  computing the centre on every row instead would have used "
          f"{full_centre:,.0f}, a {abs(full_centre - centre) / centre:.2%} shift that "
          f"carries information from the later periods into every training row")

    def scale(block: np.ndarray) -> np.ndarray:
        return (block - centre) / spread

    train_x = torch.tensor(scale(train_x_raw), dtype=torch.float32).unsqueeze(-1)
    train_y = torch.tensor(scale(train_y_raw), dtype=torch.float32)
    valid_x = torch.tensor(scale(valid_x_raw), dtype=torch.float32).unsqueeze(-1)
    valid_y = torch.tensor(scale(valid_y_raw), dtype=torch.float32)
    test_x = torch.tensor(scale(test_x_raw), dtype=torch.float32).unsqueeze(-1)
    test_y = torch.tensor(scale(test_y_raw), dtype=torch.float32)

    print("\n--- 4. Train on train, and pick the checkpoint on validation ---")
    model = Forecaster()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} parameters, {EPOCHS} epochs, full batch of "
          f"{len(train_x)} rows")
    print(f"  {'epoch':>6}  {'train loss':>12}  {'validation loss':>16}")
    best_valid, best_epoch, best_state = float("inf"), 0, None
    last_valid, last_train, train_at_best = float("nan"), float("nan"), float("nan")
    printed_valid: dict = {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimiser.zero_grad()
        loss_fn(model(train_x), train_y).backward()
        optimiser.step()
        # Both losses are recomputed after the step, so they belong to the same weights.
        model.eval()
        with torch.no_grad():
            shown_train = loss_fn(model(train_x), train_y).item()
            shown_valid = loss_fn(model(valid_x), valid_y).item()
        last_valid, last_train = shown_valid, shown_train
        if shown_valid < best_valid:
            best_valid, best_epoch, train_at_best = shown_valid, epoch, shown_train
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            printed_valid[epoch] = shown_valid
            print(f"  {epoch:>6}  {shown_train:>12.4f}  {shown_valid:>16.4f}")
    print("  the final test was not evaluated in this loop, and the checkpoint below is "
          "chosen without it")

    # Checkpoint selection, not early stopping: the loop runs all EPOCHS rounds and the
    # best validation weights are put back afterwards.
    model.load_state_dict(best_state)
    print(f"  validation bottomed at epoch {best_epoch} ({best_valid:.4f}) and ended at "
          f"{last_valid:.4f} after {EPOCHS}")
    if best_epoch < EPOCHS and last_train < train_at_best:
        print(f"  over the last {EPOCHS - best_epoch} epochs the training loss fell from "
              f"{train_at_best:.4f} to {last_train:.4f} and the validation loss rose "
              f"{last_valid / best_valid - 1:.0%}: that is what overfitting to this split "
              f"looks like, though one validation window of {len(valid_y_raw)} rows is "
              f"not proof of it")
    print(f"  the weights from epoch {best_epoch} are restored, chosen on validation alone; "
          f"the loop still ran all {EPOCHS} rounds, so this is checkpoint selection, not "
          f"early stopping")
    printed_low = min(printed_valid, key=printed_valid.get)
    if printed_low != best_epoch:
        print(f"  the rows printed every 20 epochs put the low point at epoch "
              f"{printed_low}; the checkpoint needs every epoch, so every epoch is measured")

    print("\n--- 5. Score the final test, for the first time ---")
    model.eval()
    with torch.no_grad():
        predicted = model(test_x).numpy() * spread + centre
    actual = test_y_raw
    persistence = test_x_raw[:, -1:]
    last_week = test_x_raw[:, -7:-6]
    flow_truth = truth["cash_flow"]
    campaign_days = pd.to_datetime(flow_truth["campaign_days"])
    target_dates = series.index[test_start + WINDOW:test_start + WINDOW + len(actual)]
    ordinary = ~target_dates.isin(campaign_days)
    train_dates = series.index[WINDOW:WINDOW + len(train_y_raw)]
    weekday_factor = np.array(flow_truth["purchase_weekday_factor"])
    day_factor = np.array(flow_truth["day_of_month_factor"])

    # The oracle's base level divides each training target by its own date's two factors.
    # The mean of the whole factor arrays would weight every weekday and position equally,
    # which is not how the training dates fall.
    train_scale = (weekday_factor[train_dates.dayofweek.to_numpy()]
                   * day_factor[train_dates.day.to_numpy() - 1])
    base_level = float(np.mean(train_y_raw.ravel() / train_scale))
    factor_pred = (base_level
                   * weekday_factor[target_dates.dayofweek.to_numpy()]
                   * day_factor[target_dates.day.to_numpy() - 1]).reshape(-1, 1)
    print(f"  final test {target_dates[0].date()} .. {target_dates[-1].date()}, "
          f"{len(actual)} days, scored here and nowhere earlier")
    practical = {"yesterday repeated": persistence, "same weekday last week": last_week}
    learned = {"recurrent model": predicted}
    oracle = {"oracle planted factors": factor_pred}
    # Each ratio column names its denominator.
    oracle_rmse = rmse(actual, factor_pred)
    model_rmse = rmse(actual, predicted)
    print(f"    {'route':<24} {'RMSE':>18}  {'ordinary days':>14}  {'/ oracle':>9}  "
          f"{'/ model':>8}")
    for heading, group in (("practical baselines, no training", practical),
                           ("learned model", learned),
                           ("oracle reference, not deployable", oracle)):
        print(f"  {heading}:")
        for label, forecast in group.items():
            score = rmse(actual, forecast)
            print(f"    {label:<24} {score:>18,.0f}  "
                  f"{rmse(actual[ordinary], forecast[ordinary]):>14,.0f}  "
                  f"{score / oracle_rmse:>8.2f}x  {score / model_rmse:>7.2f}x")
    promo = target_dates[~ordinary]
    if len(promo):
        oracle_ordinary = rmse(actual[ordinary], factor_pred[ordinary])
        promo_share = (np.sum((actual - factor_pred)[~ordinary] ** 2)
                       / np.sum((actual - factor_pred) ** 2))
        print(f"  the promotion day {', '.join(str(d.date()) for d in promo)} is known to no "
              f"route and carries {promo_share:.0%} of the oracle's squared error; without "
              f"it the recurrent model is "
              f"{rmse(actual[ordinary], predicted[ordinary]) / oracle_ordinary:.2f}x the "
              f"oracle, not {model_rmse / oracle_rmse:.2f}x")
    for label, forecast in practical.items():
        print(f"  the recurrent model's RMSE is {1 - model_rmse / rmse(actual, forecast):.1%} "
              f"below {label}")
    print(f"  the last row is not a baseline anyone could have built on the day: it reads "
          f"the weekday and month-position factors the generator used, out of "
          f"ground_truth.json, and only its base level is estimated, from the "
          f"{len(train_y_raw)} training targets. It is the score available to something "
          f"that already knew the structure, not a bound anything must beat")
    print(f"  a window of {WINDOW} contains the weekly cycle, so the model can learn "
          f"it; it never sees the calendar, so the month-position effect is only "
          f"available to it through whatever the last two weeks happen to imply")

    print("\n--- 6. Score it on the rows it was trained on ---")
    with torch.no_grad():
        fitted = model(train_x).numpy() * spread + centre
        validated = model(valid_x).numpy() * spread + centre
    train_score = rmse(train_y_raw, fitted)
    valid_score = rmse(valid_y_raw, validated)
    test_score = model_rmse
    print(f"  training rows   RMSE {train_score:>14,.0f}")
    print(f"  validation rows RMSE {valid_score:>14,.0f}  "
          f"({valid_score / train_score:.2f}x the training number)")
    print(f"  final test rows RMSE {test_score:>14,.0f}  "
          f"({test_score / train_score:.2f}x the training number)")
    print("  the first number is what a plot of predictions over the training period shows, "
          "and it is available before any forecast has been made. The other two were both "
          "measured on rows the weights never saw; the validation number was visible while "
          "the run was still being set up, and the final test number was not")
    promo_obs = set(np.flatnonzero(series.index.isin(campaign_days)).tolist())

    def clear_of_promotions(start: int, stop: int) -> np.ndarray:
        return np.array([not touched_observations([i], WINDOW, HORIZON, len(values)) & promo_obs
                         for i in range(start, stop)])

    valid_clear = clear_of_promotions(valid_start, test_start)
    test_clear = clear_of_promotions(test_start, len(features))
    clear_valid = rmse(valid_y_raw[valid_clear], validated[valid_clear])
    clear_test = rmse(test_y_raw[test_clear], predicted[test_clear])
    assert len(valid_y_raw) == len(test_y_raw), "the two held-out windows differ in length"
    print(f"  they also disagree: validation scores {valid_score / test_score:.2f}x the final "
          f"test, on {len(valid_y_raw)} rows each")
    print(f"  without the rows whose window or target holds a promotion day "
          f"({(~valid_clear).sum()} and {(~test_clear).sum()} rows), the two are "
          f"{clear_valid:,.0f} and {clear_test:,.0f} ({clear_valid / clear_test:.2f}x)")
    print("  one day moves a held-out score this much, so one holdout is a sample, not a "
          "verdict, which is what script 07 is for")


if __name__ == "__main__":
    main()
