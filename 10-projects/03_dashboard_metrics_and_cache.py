"""Audit the numbers behind a dashboard, then move the work that produced them off the request path.

Demonstrates why a dashboard can be internally consistent and still wrong:
    1. Recompute a ratio the source table already reports, and compare the two.
    2. Check whether the parts of a total actually add up to the total.
    3. Bucket customers into bands and count how many the buckets lost.
    4. Line the band labels up against the band edges they claim to describe.
    5. Compute every tile once, and time how long a cold build takes.
    6. Serve the same tiles from a cache, and time the warm path.
    7. Change the source, and see which invalidation rule notices and which does not.

Module 10: Applied Projects - Dashboard Metrics and Precomputation.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
CACHE_DIR = DATA / "dashboard_cache"
CACHE_FILE = CACHE_DIR / "tiles.pkl"
CACHE_META = CACHE_DIR / "tiles_meta.json"

# The bands a dashboard would show as a funnel. Written the way they usually are:
# a list of edges that reads as if it covers everyone.
AUM_BAND_EDGES = [0, 100_000, 500_000, 1_000_000]
AUM_BAND_LABELS = ["Mass", "Affluent", "High net worth"]

REPORTED_RATIO_CAP = 99


def load_sources() -> tuple:
    """Read the two files this script audits, with a clear message if they are missing."""
    needed = ["facility_beds.csv", "customers.csv"]
    missing = [name for name in needed if not (DATA / name).exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Run 01_build_project_datasets.py first.")
    return pd.read_csv(DATA / "facility_beds.csv"), pd.read_csv(DATA / "customers.csv")


def audit_reported_ratio(beds: pd.DataFrame) -> pd.DataFrame:
    """Recompute the utilization ratio and compare it against the column already provided.

    A dashboard that reads reported_utilization_pct straight through has no way to
    notice that the upstream system clamps it. The clamp is invisible in isolation:
    99 is a legal percentage. It only shows up when the ratio is recomputed from the
    two columns it was supposedly derived from.
    """
    beds = beds.copy()
    beds["recomputed_pct"] = 100.0 * beds["occupied_beds"] / beds["total_beds"]
    beds["gap"] = beds["recomputed_pct"] - beds["reported_utilization_pct"]

    at_cap = beds[beds["reported_utilization_pct"] >= REPORTED_RATIO_CAP]
    disagree = beds[beds["gap"].abs() > 0.5]

    print(f"    rows                                   {len(beds):>8,}")
    print(f"    rows reporting exactly {REPORTED_RATIO_CAP}%             {len(at_cap):>8,}")
    print(f"    rows where the two ratios differ > 0.5 {len(disagree):>8,}")
    if len(at_cap):
        print(f"\n    Among the rows reading {REPORTED_RATIO_CAP}%, the recomputed ratio runs from "
              f"{at_cap['recomputed_pct'].min():.1f}% to {at_cap['recomputed_pct'].max():.1f}%.")
        print("    The reported column cannot go above the cap, so every facility past it")
        print("    lands on the same value and the busiest ones stop being distinguishable.")

    print(f"\n    mean utilization, as reported   {beds['reported_utilization_pct'].mean():>7.2f}%")
    print(f"    mean utilization, recomputed    {beds['recomputed_pct'].mean():>7.2f}%")
    return beds


def audit_parts_and_total(beds: pd.DataFrame) -> None:
    """Check that occupied plus free reaches the total, and report the shortfall.

    A tile reading "free beds" is answering the question "where can a patient go".
    A tile reading "total minus occupied" answers a different question, because beds
    out of service are in the total and available to nobody. The two tiles disagree
    by exactly the out-of-service count, and neither one says so.
    """
    parts = beds["occupied_beds"] + beds["free_beds"]
    shortfall = beds["total_beds"] - parts
    rows_short = int((shortfall > 0).sum())

    print(f"    rows where occupied + free < total     {rows_short:>8,} of {len(beds):,}")
    print(f"    total beds                             {beds['total_beds'].sum():>8,}")
    print(f"    occupied + free                        {parts.sum():>8,}")
    print(f"    difference                             {shortfall.sum():>8,}"
          f"   = out_of_service_beds ({beds['out_of_service_beds'].sum():,})")

    naive_free = beds["total_beds"].sum() - beds["occupied_beds"].sum()
    real_free = beds["free_beds"].sum()
    print(f"\n    'free' tile computed as total - occupied  {naive_free:>8,}")
    print(f"    'free' tile read from the free column     {real_free:>8,}")
    print(f"    A dashboard showing the first number overstates availability by "
          f"{naive_free - real_free:,} beds.")


def audit_banding(customers: pd.DataFrame) -> pd.DataFrame:
    """Bucket customers by assets and count how many fell outside every bucket.

    pandas returns NaN for a value outside the outermost edges, and NaN rows are
    dropped by value_counts without comment. The funnel then adds up to less than
    the customer base, and the shortfall is the segment that matters most.
    """
    banded = customers.copy()
    banded["band"] = pd.cut(
        banded["total_aum"], bins=AUM_BAND_EDGES, labels=AUM_BAND_LABELS
    )
    counts = banded["band"].value_counts().reindex(AUM_BAND_LABELS)

    print(f"    {'band':<20}{'edges':<26}{'customers':>11}")
    for i, label in enumerate(AUM_BAND_LABELS):
        edges = f"({AUM_BAND_EDGES[i]:,} , {AUM_BAND_EDGES[i + 1]:,}]"
        print(f"    {label:<20}{edges:<26}{counts[label]:>11,}")
    print(f"    {'sum of the bands':<20}{'':<26}{counts.sum():>11,}")
    print(f"    {'customers in the file':<20}{'':<26}{len(banded):>11,}")

    lost = banded["band"].isna().sum()
    above = int((banded["total_aum"] > AUM_BAND_EDGES[-1]).sum())
    at_zero = int((banded["total_aum"] <= AUM_BAND_EDGES[0]).sum())
    print(f"\n    customers in no band at all            {lost:>11,}")
    print(f"        above the top edge                 {above:>11,}")
    print(f"        at or below the bottom edge        {at_zero:>11,}")
    print("    Those are the customers a wealth funnel exists to find, and the chart")
    print("    that reads value_counts() never showed that they were missing.")
    return banded


def audit_band_labels(customers: pd.DataFrame) -> None:
    """Line each label up against the range of values that actually landed under it.

    A label is a claim about a range. Once edges are edited and labels are not, the
    two drift apart silently, because nothing checks that "High net worth" contains
    the customers a reader would call high net worth.
    """
    fixed_edges = [0, 100_000, 500_000, 1_000_000, np.inf]
    fixed_labels = ["Mass", "Affluent", "High net worth", "Ultra high net worth"]
    banded = pd.cut(
        customers["total_aum"], bins=fixed_edges, labels=fixed_labels, include_lowest=True
    )
    grouped = customers.groupby(banded, observed=False)["total_aum"]

    print(f"    {'label':<24}{'count':>9}{'min assets':>16}{'max assets':>16}")
    for label in fixed_labels:
        group = grouped.get_group(label)
        if len(group) == 0:
            print(f"    {label:<24}{0:>9}{'-':>16}{'-':>16}")
            continue
        print(f"    {label:<24}{len(group):>9,}{group.min():>16,.0f}{group.max():>16,.0f}")
    print(f"    {'total':<24}{int(banded.notna().sum()):>9,}")
    print("\n    With an open top edge and include_lowest, the bands now hold everyone,")
    print("    and each label's observed range can be read against the name it carries.")


def build_tiles(beds: pd.DataFrame, customers: pd.DataFrame) -> dict:
    """Compute every dashboard tile from the source tables, the slow and honest way.

    This is deliberately the whole computation, not a sample of it. What makes a
    cache worth having is that the work it replaces is real; timing a cheap function
    proves nothing about whether precomputing was the right call.
    """
    beds = beds.copy()
    beds["recomputed_pct"] = 100.0 * beds["occupied_beds"] / beds["total_beds"]
    return {
        "facilities": int(beds["facility"].nunique()),
        "total_beds": int(beds.groupby("facility")["total_beds"].max().sum()),
        "mean_utilization_pct": float(beds["recomputed_pct"].mean()),
        "utilization_by_department": beds.groupby("department")["recomputed_pct"]
                                         .mean().round(2).to_dict(),
        "utilization_by_month": beds.groupby("report_month")["recomputed_pct"]
                                    .mean().round(2).to_dict(),
        "busiest_facilities": beds.groupby("facility")["recomputed_pct"]
                                  .mean().nlargest(5).round(2).to_dict(),
        "customers": int(len(customers)),
        "total_aum": float(customers["total_aum"].sum()),
        "aum_by_city_tier": customers.groupby("city_tier")["total_aum"]
                                     .sum().round(2).to_dict(),
    }


def source_fingerprint(paths: list) -> dict:
    """Describe the sources by size and modification time, which is what a cache keys on."""
    return {
        str(path.name): {"size": path.stat().st_size, "mtime": path.stat().st_mtime}
        for path in paths
    }


def write_cache(tiles: dict, paths: list) -> None:
    """Write the tiles and the fingerprint of the sources they were built from.

    Both files are written together. A cache that stores results without recording
    what produced them can only answer "is there a cache", never "is it still valid".
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump(tiles, handle)
    CACHE_META.write_text(json.dumps(source_fingerprint(paths), indent=2), encoding="utf-8")


def cache_is_fresh(paths: list) -> bool:
    """Decide whether the cached tiles still describe the current sources."""
    if not (CACHE_FILE.exists() and CACHE_META.exists()):
        return False
    stored = json.loads(CACHE_META.read_text(encoding="utf-8"))
    return stored == source_fingerprint(paths)


def timed(label: str, function) -> tuple:
    """Run a function, print how long it took, and hand back the result."""
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    print(f"    {label:<44}{elapsed * 1000:>9.1f} ms")
    return result, elapsed


def main() -> None:
    beds, customers = load_sources()
    sources = [DATA / "facility_beds.csv", DATA / "customers.csv"]

    print("--- 1. A ratio the source table already reports ---")
    beds = audit_reported_ratio(beds)

    print("\n--- 2. Do the parts add up to the total? ---")
    audit_parts_and_total(beds)

    print("\n--- 3. Banding, and the customers it drops ---")
    audit_banding(customers)

    print("\n--- 4. Labels against the ranges they name ---")
    audit_band_labels(customers)

    print("\n--- 5. Cold build: every tile computed from the sources ---")
    for path in (CACHE_FILE, CACHE_META):
        path.unlink(missing_ok=True)
    tiles, cold = timed("compute all tiles from the CSV files",
                        lambda: build_tiles(beds, customers))
    write_cache(tiles, sources)
    print(f"    {len(tiles)} tiles cached to {CACHE_FILE.name}")

    print("\n--- 6. Warm path: the same tiles, read back ---")
    cached, warm = timed("load tiles from the cache",
                         lambda: pickle.load(CACHE_FILE.open("rb")))
    print(f"    cached tiles identical to the freshly built ones: {cached == tiles}")
    print(f"    the cache answers the request {cold / max(warm, 1e-9):.0f}x faster")
    print("    Nothing got faster. The work moved off the request and onto a build step.")

    print("\n    At this size the cold build is already cheap, so the ratio above is a")
    print("    weak argument on its own. What matters is which side grows with the data:")
    for factor in (1, 10, 40):
        bigger = pd.concat([beds] * factor, ignore_index=True)
        started = time.perf_counter()
        build_tiles(bigger, customers)
        grew = (time.perf_counter() - started) * 1000
        print(f"        {len(bigger):>8,} bed rows -> cold build {grew:>8.1f} ms"
              f"   warm read {warm * 1000:>6.1f} ms")
    print("    The cold column tracks the row count. The warm column does not move,")
    print("    because reading a dict of finished numbers does not depend on the source.")

    print("\n--- 7. The source changes ---")
    print(f"    'a cache file exists' says fresh:        {CACHE_FILE.exists()}")
    print(f"    'the fingerprint still matches' says:    {cache_is_fresh(sources)}")

    beds_path = DATA / "facility_beds.csv"
    original_bytes = beds_path.read_bytes()
    changed = pd.read_csv(beds_path)
    first = changed.index[0]
    # Flip the first row between empty and full, so the edit is a real change no
    # matter what the file currently holds. An edit that happens to be a no-op
    # would leave the two figures below identical and prove nothing.
    staffed = int(changed.loc[first, "total_beds"] - changed.loc[first, "out_of_service_beds"])
    was = int(changed.loc[first, "occupied_beds"])
    changed.loc[first, "occupied_beds"] = 0 if was else staffed
    changed.to_csv(beds_path, index=False)

    print("\n    One row edited and written back to facility_beds.csv.")
    print(f"    'a cache file exists' still says fresh:  {CACHE_FILE.exists()}")
    print(f"    'the fingerprint still matches' says:    {cache_is_fresh(sources)}")

    rebuilt = build_tiles(pd.read_csv(DATA / "facility_beds.csv"), customers)
    print(f"\n    mean utilization, stale cache  {cached['mean_utilization_pct']:.4f}%")
    print(f"    mean utilization, rebuilt      {rebuilt['mean_utilization_pct']:.4f}%")
    print("    A dashboard on the first rule would have served the stale figure with")
    print("    no error and no warning, because the cache file was there the whole time.")

    beds_path.write_bytes(original_bytes)
    write_cache(build_tiles(pd.read_csv(beds_path), customers), sources)
    print(f"\n    Source file restored, cache rebuilt against it: {cache_is_fresh(sources)}")


if __name__ == "__main__":
    main()
