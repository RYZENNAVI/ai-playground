"""Lay a series out as supervised rows, and keep the two ends of it apart while doing so.

Demonstrates the part of sequence forecasting that is not the network:
    1. Slide a window over the series and read off what one supervised row contains.
    2. Split those rows at random, and count how many observations end up on both sides.
    3. Split them by time into train, validation and final test, scaling from the train side only.
    4. Train a recurrent model, watching train and validation while the final test stays sealed.
    5. Open the final test once, and score it against two practical baselines and one oracle.
    6. Score it again on the rows it was trained on, and compare the two numbers.

This experiment is deliberately one-step-ahead. series_to_supervised takes a
horizon argument, but the three baselines in step 5 are written as single-step
comparisons, and raising HORIZON would leave them broadcasting quietly against a
wider target block rather than failing. The assertion below is what stops that.

Module 08: Time Series Forecasting - Windowing and Split Discipline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 20260827
WINDOW = 14
HORIZON = 1
# The last TEST_DAYS rows are the final test and are not read until step 5. The
# VALID_DAYS rows immediately before them are the validation set, which is what
# the training loop is allowed to watch. Keeping those two separate is the whole
# point: a set that is consulted while the run is still being tuned has become a
# validation set no matter what it is called.
TEST_DAYS = 60
VALID_DAYS = 60
HIDDEN = 48
EPOCHS = 120
LEARNING_RATE = 0.01

# The three baselines in step 5 are one-step comparisons, and the last-week one
# reads position -7 of the window. Neither survives a change to these two
# constants silently, so neither is left to chance.
assert HORIZON == 1, "the step 5 baselines are written for one-step-ahead forecasting"
assert WINDOW >= 7, "the last-week baseline reads seven positions back in the window"


def series_to_supervised(values: np.ndarray, window: int, horizon: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Turn one long series into rows of (window observations -> next horizon observations).

    Every row overlaps the next by window - 1 observations. That overlap is the
    whole reason the split later in this script has to be made by time: two rows
    drawn from neighbouring positions are not two independent examples, they are
    the same stretch of history shifted by one step.
    """
    rows_x, rows_y = [], []
    for start in range(len(values) - window - horizon + 1):
        rows_x.append(values[start:start + window])
        rows_y.append(values[start + window:start + window + horizon])
    return np.array(rows_x), np.array(rows_y)


def touched_observations(rows: np.ndarray, window: int, horizon: int,
                         total: int) -> set:
    """Return every observation index the given supervised rows read or predict.

    total is the length of the underlying series and is checked rather than
    carried: a row index that would reach past the end of the series means the
    rows and the series do not belong to each other, which is worth failing on
    rather than silently counting a stretch that does not exist.
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

    A row index says which stretch of the series a row covers, so membership can
    be counted exactly rather than estimated: mark every observation each side
    touches, and intersect the two sets.
    """
    return len(touched_observations(left_rows, window, horizon, total)
               & touched_observations(right_rows, window, horizon, total))


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
    test_touched = len(set(
        j for i in random_test_idx for j in range(i, i + WINDOW + HORIZON)))
    print(f"  {len(random_train_idx)} training rows, {len(random_test_idx)} test rows")
    print(f"  observations the test rows touch: {test_touched}")
    print(f"  of those, {shared} also appear in a training row "
          f"({shared / test_touched:.0%})")
    random_train_set = set(random_train_idx.tolist())
    neighbours = sum((i - 1 in random_train_set) or (i + 1 in random_train_set)
                     for i in random_test_idx)
    print(f"  {neighbours} of {len(random_test_idx)} test rows have an immediate "
          f"neighbour in the training set, differing from it by one step")
    # Observation overlap says the two sides read some of the same history. The
    # sharper question for a supervised split is narrower: was the value a test
    # row is asked to predict already handed to the model as an input somewhere
    # in training? That is the answer being memorised rather than forecast.
    train_touched = touched_observations(random_train_idx, WINDOW, HORIZON, len(values))
    test_targets = target_observations(random_test_idx, WINDOW, HORIZON)
    seen_targets = sum(j in train_touched for j in test_targets)
    print(f"  test targets already present in a training row: {seen_targets} of "
          f"{len(test_targets)} ({seen_targets / len(test_targets):.0%})")
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
    print(f"  final test {len(test_x_raw):>3} rows, not read until step 5")
    print(f"  observations shared between train and final test: {ordered_shared}")
    boundary = shared_observation_count(
        np.arange(valid_start), np.arange(valid_start, test_start),
        WINDOW, HORIZON, len(values))
    print(f"  between train and validation: {boundary} "
          f"(the {WINDOW + HORIZON - 1} at the boundary, which is unavoidable: the "
          f"first")
    print(f"  validation window is made of days that had already happened, and using "
          f"them is what forecasting is)")
    centre = float(train_x_raw.mean())
    spread = float(train_x_raw.std())
    print(f"  scaling centre {centre:,.0f} and spread {spread:,.0f}, computed on the "
          f"training rows only - not on validation, and not on the final test")
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

    print("\n--- 4. Train, watching validation and leaving the final test sealed ---")
    model = Forecaster()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} parameters, {EPOCHS} epochs, full batch of "
          f"{len(train_x)} rows")
    print(f"  {'epoch':>6}  {'train loss':>12}  {'validation loss':>16}")
    best_valid, best_epoch, best_state = float("inf"), 0, None
    last_valid = float("nan")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimiser.zero_grad()
        loss_fn(model(train_x), train_y).backward()
        optimiser.step()
        # Both numbers are recomputed after the step. The loss that drove the
        # update belongs to the weights as they were before it; printing it
        # beside a validation loss measured after it puts two parameter states on
        # the same line and calls the pair a training curve.
        model.eval()
        with torch.no_grad():
            shown_train = loss_fn(model(train_x), train_y).item()
            shown_valid = loss_fn(model(valid_x), valid_y).item()
        last_valid = shown_valid
        if shown_valid < best_valid:
            best_valid, best_epoch = shown_valid, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {shown_train:>12.4f}  {shown_valid:>16.4f}")
    print("  the final test was not evaluated once in this loop; every decision above "
          "this line was made without it")

    # The training loss can always be driven lower by running longer; the
    # validation loss is the one that says whether the extra rounds bought
    # anything. Keeping the weights from its lowest point is the whole reason a
    # validation set is carved out separately from the final test - a run that
    # only prints the curve has done the measuring and thrown away the answer.
    model.load_state_dict(best_state)
    print(f"  validation bottomed at epoch {best_epoch} ({best_valid:.4f}) and ended at "
          f"{last_valid:.4f} after {EPOCHS}")
    print(f"  the last {EPOCHS - best_epoch} epochs lowered the training loss and raised "
          f"the validation loss by {last_valid / best_valid - 1:.0%}: that is the point "
          f"where the model")
    print(f"  stopped learning the series and started learning this sample of it. The "
          f"weights from")
    print(f"  epoch {best_epoch} are restored, chosen on validation alone, with the final "
          f"test still unread")
    print(f"  note that the printed rows every 20 epochs put the low point at 80; the "
          f"real one is")
    print(f"  epoch {best_epoch}, and the curve is only checked every epoch because "
          f"something acts on it. A")
    print(f"  quantity that is printed for a human to glance at gets sampled; one that "
          f"a decision")
    print(f"  depends on gets measured.")

    print("\n--- 5. Open the final test, once ---")
    model.eval()
    with torch.no_grad():
        predicted = model(test_x).numpy() * spread + centre
    actual = test_y_raw
    persistence = test_x_raw[:, -1:]
    last_week = test_x_raw[:, -7:-6]
    flow_truth = truth["cash_flow"]
    target_dates = series.index[test_start + WINDOW:test_start + WINDOW + len(actual)]
    train_dates = series.index[WINDOW:WINDOW + len(train_y_raw)]
    weekday_factor = np.array(flow_truth["purchase_weekday_factor"])
    day_factor = np.array(flow_truth["day_of_month_factor"])

    # The oracle's base level is estimated from the training targets only, by
    # dividing each one by the two factors its own date carries and averaging.
    # Taking the mean of the whole factor arrays instead would weight every
    # weekday and every month position equally, which is not how the training
    # dates actually fall.
    train_scale = (weekday_factor[train_dates.dayofweek.to_numpy()]
                   * day_factor[train_dates.day.to_numpy() - 1])
    base_level = float(np.mean(train_y_raw.ravel() / train_scale))
    factor_pred = (base_level
                   * weekday_factor[target_dates.dayofweek.to_numpy()]
                   * day_factor[target_dates.day.to_numpy() - 1]).reshape(-1, 1)
    print(f"  final test {target_dates[0].date()} .. {target_dates[-1].date()}, "
          f"{len(actual)} days, scored for the first time here")
    practical = {
        "yesterday repeated": rmse(actual, persistence),
        "same weekday last week": rmse(actual, last_week),
    }
    learned = {"recurrent model": rmse(actual, predicted)}
    oracle = {"oracle planted factors": rmse(actual, factor_pred)}
    best = min({**practical, **learned, **oracle}.values())
    for heading, group in (("practical baselines, no training", practical),
                           ("learned model", learned),
                           ("oracle reference, not deployable", oracle)):
        print(f"  {heading}:")
        for label, score in group.items():
            print(f"    {label:<24} RMSE {score:>14,.0f}  {score / best:>6.2f}x")
    print(f"  the last row is not a baseline anyone could have built on the day: it "
          f"reads the")
    print(f"  weekday and month-position factors the generator used, out of "
          f"ground_truth.json.")
    print(f"  Only its base level is estimated, from the {len(train_y_raw)} training "
          f"targets. It is the score")
    print(f"  available to something that already knew the structure, not a bound "
          f"anything must beat.")
    print(f"  a window of {WINDOW} contains the weekly cycle, so the model can learn "
          f"it; it never sees the calendar, so the month-position effect is only "
          f"available to it through whatever the last two weeks happen to imply")
    scores = {**learned, **practical, **oracle}

    print("\n--- 6. Score it on the rows it was trained on ---")
    with torch.no_grad():
        fitted = model(train_x).numpy() * spread + centre
        validated = model(valid_x).numpy() * spread + centre
    train_score = rmse(train_y_raw, fitted)
    valid_score = rmse(valid_y_raw, validated)
    test_score = scores["recurrent model"]
    print(f"  training rows   RMSE {train_score:>14,.0f}")
    print(f"  validation rows RMSE {valid_score:>14,.0f}  "
          f"({valid_score / train_score:.2f}x the training number)")
    print(f"  final test rows RMSE {test_score:>14,.0f}  "
          f"({test_score / train_score:.2f}x the training number)")
    print("  the first number is what a plot of predictions over the training period")
    print("  shows, and it is available before any forecast has been made. The other two")
    print("  were both measured on rows the weights never saw - the difference between")
    print("  them is that the validation number was visible while the run was still")
    print("  being set up, and the final test number was not.")
    print(f"  they also disagree: validation scores {valid_score / test_score:.2f}x the "
          f"final test here, on")
    print("  sixty rows each. Two held-out windows of the same length, next to each "
          "other in")
    print("  time, and the later one is easier. One holdout is a sample, not a "
          "verdict - which")
    print("  is the question script 07 exists to settle.")


if __name__ == "__main__":
    main()
