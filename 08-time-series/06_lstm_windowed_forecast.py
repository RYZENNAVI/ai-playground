"""Lay a series out as supervised rows, and keep the two ends of it apart while doing so.

Demonstrates the part of sequence forecasting that is not the network:
    1. Slide a window over the series and read off what one supervised row contains.
    2. Split those rows at random, and count how many observations end up on both sides.
    3. Split them by time instead, and scale using only what the earlier side knows.
    4. Train a recurrent model on the time-ordered split and track both curves.
    5. Score it on the held-out tail against two baselines that need no training at all.
    6. Score it again on the rows it was trained on, and compare the two numbers.

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
TEST_DAYS = 60
HIDDEN = 48
EPOCHS = 120
LEARNING_RATE = 0.01


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


def shared_observation_count(left_rows: np.ndarray, right_rows: np.ndarray,
                             window: int, horizon: int, total: int) -> int:
    """Count observations of the original series that appear on both sides of a split.

    A row index says which stretch of the series a row covers, so membership can
    be counted exactly rather than estimated: mark every observation each side
    touches, and intersect the two sets.
    """
    def touched(indices: np.ndarray) -> set[int]:
        marks: set[int] = set()
        for i in indices:
            marks.update(range(i, i + window + horizon))
        return marks
    return len(touched(left_rows) & touched(right_rows))


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
    neighbours = sum(
        (i - 1 in set(random_train_idx.tolist())) or (i + 1 in set(random_train_idx.tolist()))
        for i in random_test_idx)
    print(f"  {neighbours} of {len(random_test_idx)} test rows have an immediate "
          f"neighbour in the training set, differing from it by one step")
    print("  a score measured on these rows answers how well the model interpolates "
          "inside history it has already read, which is not the question a forecast asks")

    print("\n--- 3. Split by time, and scale from the earlier side only ---")
    train_x_raw, test_x_raw = features[:cut], features[cut:]
    train_y_raw, test_y_raw = targets[:cut], targets[cut:]
    ordered_shared = shared_observation_count(
        np.arange(cut), np.arange(cut, len(features)), WINDOW, HORIZON, len(values))
    print(f"  {len(train_x_raw)} training rows up to "
          f"{series.index[cut + WINDOW - 1].date()}, {len(test_x_raw)} test rows after it")
    print(f"  observations shared across the cut: {ordered_shared} "
          f"(the {WINDOW + HORIZON - 1} at the boundary, which is unavoidable)")
    centre = float(train_x_raw.mean())
    spread = float(train_x_raw.std())
    print(f"  scaling centre {centre:,.0f} and spread {spread:,.0f}, "
          f"both computed on training rows only")
    full_centre = float(features.mean())
    print(f"  computing the centre on every row instead would have used "
          f"{full_centre:,.0f}, a {abs(full_centre - centre) / centre:.2%} shift that "
          f"carries information from the test period into every training row")

    def scale(block: np.ndarray) -> np.ndarray:
        return (block - centre) / spread

    train_x = torch.tensor(scale(train_x_raw), dtype=torch.float32).unsqueeze(-1)
    train_y = torch.tensor(scale(train_y_raw), dtype=torch.float32)
    test_x = torch.tensor(scale(test_x_raw), dtype=torch.float32).unsqueeze(-1)
    test_y = torch.tensor(scale(test_y_raw), dtype=torch.float32)

    print("\n--- 4. Train on the time-ordered split ---")
    model = Forecaster()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} parameters, {EPOCHS} epochs, full batch of "
          f"{len(train_x)} rows")
    print(f"  {'epoch':>6}  {'train loss':>12}  {'holdout loss':>13}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimiser.step()
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                held = loss_fn(model(test_x), test_y)
            print(f"  {epoch:>6}  {loss.item():>12.4f}  {held.item():>13.4f}")

    print("\n--- 5. Score the tail against baselines that were never trained ---")
    model.eval()
    with torch.no_grad():
        predicted = model(test_x).numpy() * spread + centre
    actual = test_y_raw
    persistence = test_x_raw[:, -1:]
    last_week = test_x_raw[:, -7:-6]
    flow_truth = truth["cash_flow"]
    target_dates = series.index[cut + WINDOW:cut + WINDOW + len(actual)]
    weekday_factor = np.array(flow_truth["purchase_weekday_factor"])
    day_factor = np.array(flow_truth["day_of_month_factor"])
    factor_pred = (train_y_raw.mean()
                   / (weekday_factor.mean() * day_factor.mean())
                   * weekday_factor[target_dates.dayofweek.to_numpy()]
                   * day_factor[target_dates.day.to_numpy() - 1]).reshape(-1, 1)
    print(f"  holdout {target_dates[0].date()} .. {target_dates[-1].date()}, "
          f"{len(actual)} days")
    scores = {
        "recurrent model": rmse(actual, predicted),
        "yesterday repeated": rmse(actual, persistence),
        "same weekday last week": rmse(actual, last_week),
        "planted periodic factors": rmse(actual, factor_pred),
    }
    best = min(scores.values())
    for label, score in scores.items():
        print(f"  {label:<26} RMSE {score:>14,.0f}  {score / best:>6.2f}x")
    print(f"  a window of {WINDOW} contains the weekly cycle, so the model can learn "
          f"it; it never sees the calendar, so the month-position effect is only "
          f"available to it through whatever the last two weeks happen to imply")

    print("\n--- 6. Score it on the rows it was trained on ---")
    with torch.no_grad():
        fitted = model(train_x).numpy() * spread + centre
    train_score = rmse(train_y_raw, fitted)
    test_score = scores["recurrent model"]
    print(f"  training rows RMSE {train_score:>14,.0f}")
    print(f"  holdout rows  RMSE {test_score:>14,.0f}  "
          f"({test_score / train_score:.2f}x the training number)")
    print("  the first number is what a plot of predictions over the training period "
          "shows, and it is available before any forecast has been made")
    print("  only the second one was measured on data the weights never saw")


if __name__ == "__main__":
    main()
