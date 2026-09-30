"""This script flags unusual closing prices with a Bollinger band (a rolling mean plus and
minus two standard deviations) and then with the eight Nelson rules of statistical process
control (SPC), and counts what each rule alone would miss.

An outlier flag needs the numbers it was derived from printed beside it:
    1. Build a rolling centre line and a band two standard deviations wide.
    2. Flag every close that sits outside the band, the way a single rule does.
    3. Report those flags with the four numbers behind them rather than a date alone.
    4. Measure each close in standard deviations from its own centre line, apply the eight
       rules to that one column, and count how many days each rule flags.
    5. Merge consecutive flags into events, since a run is one excursion, not many.
    6. Take the largest one-day and three-day moves on days the band can judge, and check
       which rules fire on that day or in the next three.
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
DB_PATH = DATA / "market.sqlite"

TICKER = "MRD"
WINDOW = 20
BAND_SIGMA = 2.0
EVENT_GAP = 3

# The eight rules are stated on a series measured in standard deviations from its
# own centre line, so every one of them reads as a statement about that one column.
RULE_NAMES = {
    1: "one point beyond 3 sigma",
    2: "nine in a row on one side of centre",
    3: "six in a row rising or falling",
    4: "fourteen in a row alternating direction",
    5: "two of three beyond 2 sigma, same side",
    6: "four of five beyond 1 sigma, same side",
    7: "fifteen in a row inside 1 sigma",
    8: "eight in a row beyond 1 sigma, both sides",
}


def load_series() -> pd.DataFrame:
    """Read one instrument's closes in date order."""
    if not DB_PATH.exists():
        raise SystemExit(f"Missing {DB_PATH.name}. Run 01_build_project_datasets.py first.")
    with sqlite3.connect(DB_PATH) as connection:
        frame = pd.read_sql_query(
            "SELECT trade_date, close FROM daily_price WHERE ticker = ? ORDER BY trade_date",
            connection, params=(TICKER,),
        )
    return frame.reset_index(drop=True)


def add_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the rolling centre, spread, band edges, and the close in sigmas from centre.
    The centre follows the series, so a two-year high can still be ordinary for last month."""
    frame = frame.copy()
    frame["centre"] = frame["close"].rolling(WINDOW).mean()
    frame["spread"] = frame["close"].rolling(WINDOW).std()
    frame["upper"] = frame["centre"] + BAND_SIGMA * frame["spread"]
    frame["lower"] = frame["centre"] - BAND_SIGMA * frame["spread"]
    frame["sigmas"] = (frame["close"] - frame["centre"]) / frame["spread"]
    return frame


def band_breaches(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the days whose close sits outside the band."""
    outside = frame["close"].notna() & frame["upper"].notna() & (
        (frame["close"] > frame["upper"]) | (frame["close"] < frame["lower"])
    )
    breaches = frame[outside].copy()
    breaches["side"] = np.where(breaches["close"] > breaches["upper"], "above", "below")
    return breaches


def print_flag_rows(rows: pd.DataFrame) -> None:
    """Print flagged days with the close, centre, band edges and sigmas behind each flag."""
    print(f"    {'date':<13}{'side':<7}{'close':>9}{'centre':>9}{'upper':>9}"
          f"{'lower':>9}{'sigmas':>8}")
    for row in rows.itertuples():
        print(f"    {row.trade_date:<13}{row.side:<7}{row.close:>9.2f}{row.centre:>9.2f}"
              f"{row.upper:>9.2f}{row.lower:>9.2f}{row.sigmas:>8.2f}")


def runs_on_one_side(sigmas: np.ndarray, length: int) -> np.ndarray:
    """Flag every point that ends a run of `length` consecutive points on one side of centre."""
    flags = np.zeros(len(sigmas), dtype=bool)
    for end in range(length - 1, len(sigmas)):
        window = sigmas[end - length + 1:end + 1]
        if np.isnan(window).any():
            continue
        if np.all(window > 0) or np.all(window < 0):
            flags[end] = True
    return flags


def monotonic_runs(sigmas: np.ndarray, length: int) -> np.ndarray:
    """Flag every point that ends a run of `length` points moving in one direction."""
    flags = np.zeros(len(sigmas), dtype=bool)
    for end in range(length - 1, len(sigmas)):
        window = sigmas[end - length + 1:end + 1]
        if np.isnan(window).any():
            continue
        steps = np.diff(window)
        if np.all(steps > 0) or np.all(steps < 0):
            flags[end] = True
    return flags


def alternating_runs(sigmas: np.ndarray, length: int) -> np.ndarray:
    """Flag every point that ends a run of `length` points that alternate direction."""
    flags = np.zeros(len(sigmas), dtype=bool)
    for end in range(length - 1, len(sigmas)):
        window = sigmas[end - length + 1:end + 1]
        if np.isnan(window).any():
            continue
        steps = np.sign(np.diff(window))
        if np.any(steps == 0):
            continue
        if np.all(steps[1:] != steps[:-1]):
            flags[end] = True
    return flags


def k_of_n_beyond(sigmas: np.ndarray, k: int, n: int, threshold: float) -> np.ndarray:
    """Flag every point ending a window where `k` of `n` sit beyond `threshold` on one side."""
    flags = np.zeros(len(sigmas), dtype=bool)
    for end in range(n - 1, len(sigmas)):
        window = sigmas[end - n + 1:end + 1]
        if np.isnan(window).any():
            continue
        if (window > threshold).sum() >= k or (window < -threshold).sum() >= k:
            flags[end] = True
    return flags


def hugging_centre(sigmas: np.ndarray, length: int) -> np.ndarray:
    """Flag every point ending a run of `length` points inside one sigma (too calm)."""
    flags = np.zeros(len(sigmas), dtype=bool)
    for end in range(length - 1, len(sigmas)):
        window = sigmas[end - length + 1:end + 1]
        if np.isnan(window).any():
            continue
        if np.all(np.abs(window) < 1.0):
            flags[end] = True
    return flags


def all_beyond(sigmas: np.ndarray, length: int) -> np.ndarray:
    """Flag every point ending a run of `length` points all beyond one sigma, on both sides."""
    flags = np.zeros(len(sigmas), dtype=bool)
    for end in range(length - 1, len(sigmas)):
        window = sigmas[end - length + 1:end + 1]
        if np.isnan(window).any():
            continue
        if np.all(np.abs(window) > 1.0) and (window > 0).any() and (window < 0).any():
            flags[end] = True
    return flags


def apply_rules(frame: pd.DataFrame) -> pd.DataFrame:
    """Run all eight rules over the standardised series and return one column each."""
    sigmas = frame["sigmas"].to_numpy(dtype=float)
    return pd.DataFrame({
        1: np.abs(sigmas) > 3.0,
        2: runs_on_one_side(sigmas, 9),
        3: monotonic_runs(sigmas, 6),
        4: alternating_runs(sigmas, 14),
        5: k_of_n_beyond(sigmas, 2, 3, 2.0),
        6: k_of_n_beyond(sigmas, 4, 5, 1.0),
        7: hugging_centre(sigmas, 15),
        8: all_beyond(sigmas, 8),
    }, index=frame.index).fillna(False)


def merge_into_events(positions: list, gap: int) -> list:
    """Group flagged row positions that sit within `gap` of each other into single events.
    One excursion flags every day it lasts, so a count of days overstates how often."""
    events = []
    for position in sorted(positions):
        if events and position - events[-1][-1] <= gap:
            events[-1].append(position)
        else:
            events.append([position])
    return events


def draw(frame: pd.DataFrame, breaches: pd.DataFrame, path: Path) -> None:
    """Plot the close, the centre line, the two band edges and the flagged days."""
    figure, axes = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(frame))
    axes.plot(positions, frame["close"], linewidth=1.0, label="close")
    axes.plot(positions, frame["centre"], linewidth=1.0, label=f"{WINDOW}-day centre")
    axes.plot(positions, frame["upper"], linewidth=0.8, linestyle="--",
              label=f"+{BAND_SIGMA} sigma")
    axes.plot(positions, frame["lower"], linewidth=0.8, linestyle="--",
              label=f"-{BAND_SIGMA} sigma")
    axes.fill_between(positions, frame["lower"], frame["upper"], alpha=0.08)
    axes.scatter(breaches.index, breaches["close"], s=18, zorder=5, label="outside the band")
    ticks = positions[::60]
    axes.set_xticks(ticks)
    axes.set_xticklabels(frame["trade_date"].iloc[ticks], rotation=45, ha="right")
    axes.set_title(f"{TICKER}: close against a {WINDOW}-day band")
    axes.set_ylabel("close")
    axes.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame = add_bands(load_series())

    # 1. Centre line and band

    print(f"--- 1. A {WINDOW}-day centre line and a {BAND_SIGMA} sigma band on {TICKER} ---")
    usable = frame["centre"].notna().sum()
    print(f"    rows                              {len(frame):>6}")
    print(f"    rows with a full window behind them {usable:>4}")
    print(f"    close ranges {frame['close'].min():.2f} to {frame['close'].max():.2f}, "
          f"and the band moves with it")

    # 2. Days outside the band

    print("\n--- 2. Days outside the band ---")
    breaches = band_breaches(frame)
    above = int((breaches["side"] == "above").sum())
    below = int((breaches["side"] == "below").sum())
    print(f"    flagged days   {len(breaches):>4}   above {above}   below {below}")
    print(f"    that is {100 * len(breaches) / usable:.1f}% of the days the band could judge")

    # 3. Flags with the numbers behind them

    print("\n--- 3. The same flags, reported with the numbers behind them ---")
    print_flag_rows(breaches.head(10))
    if len(breaches) > 10:
        print(f"    ... {len(breaches) - 10} more")

    below_rows = breaches[breaches["side"] == "below"]
    above_rows = breaches[breaches["side"] == "above"]
    if len(below_rows) and len(above_rows) and \
            below_rows["close"].max() > above_rows["close"].min():
        print("\n    The dearest day flagged below the band and the cheapest day flagged "
              "above it:")
        print_flag_rows(breaches.loc[[below_rows["close"].idxmax(),
                                      above_rows["close"].idxmin()]])
        print("    Reported as date and price alone, those two rows contradict each other.")
        print("    With the centre line beside them they do not: each was judged against")
        print("    its own recent window, and the windows were at different levels.")
    else:
        print("\n    On this series every below-band close came in under every above-band")
        print("    close, so date and price alone happen not to contradict each other here.")

    # 4. Eight rules over the standardised series

    print("\n--- 4. Eight rules over the standardised series ---")
    flags = apply_rules(frame)
    print(f"    {'rule':<6}{'description':<42}{'days':>7}{'share':>9}")
    for number, name in RULE_NAMES.items():
        days = int(flags[number].sum())
        print(f"    {number:<6}{name:<42}{days:>7}{100 * days / usable:>8.1f}%")
    any_rule = flags.any(axis=1)
    print(f"    {'any':<6}{'flagged by at least one rule':<42}{int(any_rule.sum()):>7}"
          f"{100 * any_rule.sum() / usable:>8.1f}%")
    print(f"\n    the band rule alone flagged {len(breaches)} days; the eight rules together "
          f"flag {int(any_rule.sum())}")
    only_band = int((frame.index.isin(breaches.index) & ~any_rule).sum())
    only_rules = int((any_rule & ~frame.index.isin(breaches.index)).sum())
    print(f"    days the band caught that none of the eight rules did: {only_band}")
    print(f"    days the other rules caught that the band did not: {only_rules}")

    above_centre = float((frame["sigmas"] > 0).sum()) / usable
    print("\n    Rule 2 counts nine days in a row on one side of centre, and rule 6 four of")
    print("    five days more than one sigma out on one side. Both were written for a")
    print(f"    process held at a fixed target. Here the centre is a {WINDOW}-day mean that")
    print(f"    follows the series, and {100 * above_centre:.0f}% of days sit above it, "
          f"so a trend alone")
    print("    keeps those counters running. Their high share is a property of the")
    print("    baseline they were given, not of anything unusual in the data.")

    # 5. Days merged into events

    print(f"\n--- 5. Merging runs into events (gap of {EVENT_GAP} days or less) ---")
    counts = {}
    for label, positions in (("band rule", list(breaches.index)),
                             ("all eight rules", list(frame.index[any_rule]))):
        events = merge_into_events(positions, EVENT_GAP)
        counts[label] = (len(positions), len(events))
        longest = max((len(event) for event in events), default=0)
        mean_run = len(positions) / len(events) if events else 0.0
        print(f"    {label:<18}{len(positions):>5} days -> {len(events):>4} events, "
              f"mean run {mean_run:.1f}, longest {longest} days")
    band_days, band_events = counts["band rule"]
    rule_days, rule_events = counts["all eight rules"]
    if band_days and rule_events <= band_events:
        print(f"    The eight rules flag {rule_days / band_days:.1f}x the band's days but "
              f"{rule_events} events against {band_events}:")
        print("    most of their days sit inside a few long runs.")

    # 6. Largest moves against both rule sets

    print("\n--- 6. One-day jumps against multi-day excursions ---")
    judged = frame["centre"].notna()
    one_day = frame["close"].pct_change().abs()[judged]
    three_day = (frame["close"] / frame["close"].shift(3) - 1).abs()[judged]
    first_later = {}

    def report(label: str, moves: pd.Series) -> tuple:
        by_band = 0
        by_rules = 0
        print(f"\n    largest {label}, on days the band can judge:")
        print(f"        {'date':<13}{'move':>9}{'band':>10}{'rules that day':>17}"
              f"{f'next {EVENT_GAP} days':>14}")
        for position in moves.nlargest(3).index:
            date = frame.loc[position, "trade_date"]
            outside = position in breaches.index
            firing = [str(number) for number in RULE_NAMES if flags.loc[position, number]]
            by_band += 1 if outside else 0
            by_rules += 1 if firing else 0
            later = "-"
            for step in range(1, EVENT_GAP + 1):
                if position + step < len(frame) and any_rule[position + step]:
                    rules = [str(n) for n in RULE_NAMES if flags.loc[position + step, n]]
                    later = f"+{step}: {','.join(rules)}"
                    first_later[position] = step
                    break
            print(f"        {date:<13}{100 * moves[position]:>8.2f}%"
                  f"{'outside' if outside else 'inside':>10}"
                  f"{','.join(firing) if firing else '-':>17}{later:>14}")
        return by_band, by_rules

    band_one, rules_one = report("single-day moves", one_day)
    band_three, rules_three = report("three-day moves", three_day)
    print(f"\n    {'':<22}{'caught by the band':>20}{'caught by the 8 rules':>24}")
    print(f"    {'top 3 single-day moves':<22}{f'{band_one} of 3':>20}"
          f"{f'{rules_one} of 3':>24}")
    print(f"    {'top 3 three-day moves':<22}{f'{band_three} of 3':>20}"
          f"{f'{rules_three} of 3':>24}")
    distinct = set(one_day.nlargest(3).index) | set(three_day.nlargest(3).index)
    same_day = sum(1 for position in distinct if any_rule[position])
    print(f"\n    The band catches {band_one + band_three} of these 6 moves "
          f"({len(distinct)} distinct days), and {6 - band_one - band_three} never crossed it.")
    print("    Rule 1 looks at single points, but at 3 sigma it is stricter than the band,")
    print(f"    and it fired on {int(flags[1].sum())} days of the whole series. "
          f"The other seven count runs or windows.")
    print(f"    The eight rules fire on {same_day} of these {len(distinct)} days.")
    if first_later:
        earliest = min(first_later.values())
        print(f"    After {len(first_later)} of them they fire within {EVENT_GAP} days, "
              f"the earliest {earliest} day{'s' if earliest > 1 else ''} later:")
        print("    a run rule reports an excursion once it has lasted.")

    path = OUTPUTS / f"bands_{TICKER.lower()}.png"
    draw(frame, breaches, path)
    print(f"\n    chart written to outputs/{path.name}")


if __name__ == "__main__":
    main()
