"""Pick a chart type from the wrong count, then build a column that silently comes back empty.

Demonstrates two rules that hold on the data they were written against and fail elsewhere:
    1. Run three queries that a chart helper would be handed in turn.
    2. Choose a chart type from the row count, the way the rule is usually written.
    3. Choose it from the number of distinct x values instead, and compare.
    4. Draw the disagreeing case both ways and save the two pictures.
    5. Sample ten points from a result the way a helper thins a long series.
    6. Attach a computed column to a fresh frame and count how many values arrived.
    7. Repeat the attachment three ways that work, and stack two frames that share an index.

Module 10: Applied Projects - Chart Criteria and Index Alignment.
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
DB_PATH = DATA / "market.sqlite"

ROW_THRESHOLD = 20
SAMPLE_POINTS = 10
MA_WINDOW = 20

QUERIES = {
    "one instrument, one month": """
        SELECT ticker, trade_date, close FROM daily_price
        WHERE ticker = 'ARB' AND trade_date LIKE '2024-03%'
        ORDER BY trade_date
    """,
    "one instrument, one year": """
        SELECT ticker, trade_date, close FROM daily_price
        WHERE ticker = 'ARB' AND trade_date LIKE '2024%'
        ORDER BY trade_date
    """,
    "four instruments, ten days": """
        SELECT ticker, trade_date, close FROM daily_price
        WHERE trade_date BETWEEN '2024-03-01' AND '2024-03-14'
        ORDER BY trade_date, ticker
    """,
}


def by_row_count(frame: pd.DataFrame) -> str:
    """Choose a chart type from how many rows came back.

    This is the rule as it is normally written, and it is right about the thing it
    was tested on: a long single series should not be drawn as bars. It reads the
    row count as a stand-in for how many positions the x axis needs, which is only
    the same number while every row carries a distinct x value.
    """
    return "line" if len(frame) > ROW_THRESHOLD else "bar"


def by_distinct_x(frame: pd.DataFrame, x_column: str) -> str:
    """Choose a chart type from how many distinct x values came back.

    This asks the question the axis actually poses. When one date carries four
    instruments, the axis still needs ten positions, not forty.
    """
    return "line" if frame[x_column].nunique() > ROW_THRESHOLD else "bar"


def draw(frame: pd.DataFrame, chart_type: str, title: str, path: Path) -> None:
    """Draw one result set as bars or as a line, exactly as the chosen rule dictates.

    The frame is drawn flat, in the order it arrived, with no grouping by series.
    That is what a generic helper does, and it is why an interleaved result turns
    into a zigzag rather than four readable series.
    """
    figure, axes = plt.subplots(figsize=(10, 4))
    positions = range(len(frame))
    if chart_type == "bar":
        axes.bar(positions, frame["close"])
    else:
        axes.plot(positions, frame["close"], marker="o", markersize=3)
    axes.set_title(title)
    axes.set_xlabel("row position in the result")
    axes.set_ylabel("close")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def thin_to_points(frame: pd.DataFrame, points: int) -> pd.DataFrame:
    """Keep evenly spaced rows so a crowded axis stays readable.

    Even spacing over row positions is only even over time when the rows are one
    series in time order. On an interleaved result it walks across instruments.
    """
    if len(frame) <= points:
        return frame
    picked = np.linspace(0, len(frame) - 1, points, dtype=int)
    return frame.iloc[picked]


def compute_moving_average(frame: pd.DataFrame) -> pd.Series:
    """Return a rolling mean of the close column, carrying the frame's own index.

    Every pandas operation preserves the index it was given. That is the behaviour
    that makes alignment work; it is also the behaviour that empties a column when
    the two sides were never meant to line up.
    """
    return frame["close"].rolling(MA_WINDOW).mean()


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Missing {DB_PATH.name}. Run 01_build_project_datasets.py first.")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        results = {name: pd.read_sql_query(sql, connection) for name, sql in QUERIES.items()}

    print("--- 1. Three results a chart helper receives ---")
    print(f"    {'result':<30}{'rows':>7}{'distinct dates':>17}{'instruments':>13}")
    for name, frame in results.items():
        print(f"    {name:<30}{len(frame):>7}{frame['trade_date'].nunique():>17}"
              f"{frame['ticker'].nunique():>13}")

    print("\n--- 2-3. The same three results under two rules ---")
    print(f"    threshold is {ROW_THRESHOLD} either way\n")
    print(f"    {'result':<30}{'by row count':>15}{'by distinct x':>16}{'agree':>8}")
    disagreeing = None
    for name, frame in results.items():
        left = by_row_count(frame)
        right = by_distinct_x(frame, "trade_date")
        agree = "yes" if left == right else "NO"
        if left != right:
            disagreeing = (name, frame, left, right)
        print(f"    {name:<30}{left:>15}{right:>16}{agree:>8}")

    if disagreeing is None:
        print("\n    The two rules agreed on every result here.")
    else:
        name, frame, left, right = disagreeing
        print(f"\n    They disagree on '{name}': {len(frame)} rows but only "
              f"{frame['trade_date'].nunique()} dates.")
        print(f"    The row rule sees {len(frame)} > {ROW_THRESHOLD} and picks a {left}.")
        print(f"    The axis needs {frame['trade_date'].nunique()} positions, so the "
              f"honest answer is a {right}.")

        print("\n--- 4. Both pictures, drawn and saved ---")
        for chart_type, label in ((left, "by-row-count"), (right, "by-distinct-x")):
            path = OUTPUTS / f"chart_{label}_{chart_type}.png"
            draw(frame, chart_type, f"{name} drawn as a {chart_type} ({label})", path)
            print(f"    {path.name}")
        head = frame.head(8)[["trade_date", "ticker", "close"]]
        print("\n    The first rows show why the line zigzags: consecutive positions are")
        print("    different instruments on the same date, not one instrument over time.")
        print(head.to_string(index=False))

    print(f"\n--- 5. Thinning a result to {SAMPLE_POINTS} points ---")
    for name in ("one instrument, one year", "four instruments, ten days"):
        frame = results[name]
        thinned = thin_to_points(frame, SAMPLE_POINTS)
        print(f"    {name:<30} {len(frame):>5} rows -> {len(thinned)} points, "
              f"covering {thinned['ticker'].nunique()} instrument(s) and "
              f"{thinned['trade_date'].nunique()} date(s)")
    print("    Even spacing over row positions is even over time only for one series.")

    print(f"\n--- 6. Attaching a computed column ---")
    year = results["one instrument, one year"]
    windowed = year[year["trade_date"] >= "2024-02-01"]
    moving_average = compute_moving_average(windowed)
    print(f"    rows selected for the window      {len(windowed):>6}")
    print(f"    index of that selection runs      {windowed.index.min()} to "
          f"{windowed.index.max()}")
    print(f"    values in the moving average      {moving_average.notna().sum():>6}")

    report = pd.DataFrame({"trade_date": windowed["trade_date"].tolist()})
    report["moving_average"] = moving_average
    filled = int(report["moving_average"].notna().sum())
    overlap = report.index.intersection(moving_average.index)
    print(f"\n    new frame built from a list, index runs 0 to {len(report) - 1}")
    print(f"    index values the two sides share  {len(overlap):>6}")
    print(f"    values that arrived in the column {filled:>6} of {len(report)}")
    print("    No exception, no warning. Pandas matched the two indexes and filled the")
    print("    column from wherever they happened to overlap.")

    # The partial overlap is the dangerous case: most of the column is populated,
    # so nothing looks broken, and every populated cell holds another date's value.
    correct = moving_average.to_numpy()
    attached = report["moving_average"].to_numpy()
    both_present = ~(pd.isna(correct) | pd.isna(attached))
    misplaced = int((both_present & (correct != attached)).sum())
    print(f"\n    cells holding a value                    {filled:>6}")
    print(f"    cells whose value belongs to another row {misplaced:>6}")

    first_filled = int(np.flatnonzero(~pd.isna(attached))[0])
    shown_date = report["trade_date"].iloc[first_filled]
    source_date = year["trade_date"].iloc[first_filled]
    print(f"\n    Take report row {first_filled}. It is labelled {shown_date} and holds "
          f"{attached[first_filled]:.4f},")
    print(f"    which is the average computed for {source_date} — the row that carried")
    print(f"    index {first_filled} in the frame the average came from. Every filled cell is")
    print(f"    shifted by the {int(windowed.index.min())} rows the window skipped. A column that is")
    print("    mostly populated and entirely misdated is harder to spot than an empty one.")

    print("\n--- 7. Three attachments that work, and one stack that does not ---")
    variants = {
        ".to_numpy()": moving_average.to_numpy(),
        ".reset_index(drop=True)": moving_average.reset_index(drop=True),
        ".set_axis(report.index)": moving_average.set_axis(report.index),
    }
    for label, values in variants.items():
        probe = pd.DataFrame({"trade_date": windowed["trade_date"].tolist()})
        probe["moving_average"] = values
        print(f"    {label:<26}{probe['moving_average'].notna().sum():>6} of {len(probe)} arrived")

    left_frame = windowed.head(3)[["trade_date", "close"]]
    right_frame = windowed.head(3)[["ticker"]].reset_index(drop=True)
    stacked = pd.concat([left_frame, right_frame], axis=1)
    aligned = pd.concat(
        [left_frame.reset_index(drop=True), right_frame], axis=1
    )
    print(f"\n    concat of a slice and a reindexed slice -> {len(stacked)} rows, "
          f"{int(stacked.isna().sum().sum())} empty cells")
    print(f"    same concat after reset_index on both   -> {len(aligned)} rows, "
          f"{int(aligned.isna().sum().sum())} empty cells")
    print("    The row count is the tell. Two three-row frames that stack into more")
    print("    than three rows never shared an index in the first place.")


if __name__ == "__main__":
    main()
