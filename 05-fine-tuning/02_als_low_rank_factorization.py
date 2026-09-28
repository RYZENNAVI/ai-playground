"""Alternating least squares (ALS): factor a sparse preference matrix into two thin matrices.

ALS models every cell as the dot product of a user vector and an item vector, both
of length 3. With the item vectors held fixed, each user vector is a small ridge
regression on the cells that user observed, and the other way round. The two fits
alternate. A missing cell still gets a prediction, because each vector is shared
by its whole row or column.

The data is 12 users and 9 items in three groups. Each user touched two of the
three items in their group, so a good fit recommends the third.

The run prints six parts:
    1. The observed matrix. 24 of 108 cells, each a 1.
    2. Two spectra. The complete matrix, with every group filled in, has rank 3.
       The observed one is full rank, with a gap after the third value. Nothing
       later uses this part.
    3. The ALS fit. Over 20 iterations the penalised objective falls every time,
       while the plain RMSE ends higher than it started.
    4. Group agreement after 20 iterations: how often each user's top unseen
       item is in their own group.
    5. The fit stopped after 2 iterations, where the RMSE was lowest, scored the
       same way. Then the same comparison over ten seeds.
    6. A known rank-3 matrix of 300x120 with most cells hidden. The error on the
       hidden cells as the density and the penalty change.
"""

import sys

import numpy as np
from scipy.linalg import svd

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

RANK = 3
REGULARISATION = 0.05
MAX_ITERATIONS = 20
EARLY_STOP = 2
SEED = 3407
SEED_SWEEP = (11, 101, 404, 1234, 2024, 3407, 4242, 8080, 9001, 31337)

# Three groups of users, three items per group, two interactions per user. Every
# user touches only part of their own group, so a correct factorisation has to
# recommend the group item the user has not touched yet.
GROUPS = {
    "A": {"users": (1, 2, 3, 4), "items": (1, 2, 3)},
    "B": {"users": (5, 6, 7, 8, 9), "items": (4, 5, 6)},
    "C": {"users": (10, 11, 12), "items": (7, 8, 9)},
}

INTERACTIONS = (
    (1, 1), (1, 2),
    (2, 1), (2, 3),
    (3, 2), (3, 3),
    (4, 1), (4, 2),
    (5, 4), (5, 5),
    (6, 4), (6, 6),
    (7, 5), (7, 6),
    (8, 4), (8, 5),
    (9, 4), (9, 5),
    (10, 7), (10, 8),
    (11, 8), (11, 9),
    (12, 7), (12, 9),
)


def build_matrix():
    """Part 1. Lay the interactions out as a rating array plus a mask of observed cells.
    Without the mask a missing cell and a real zero look the same to the fit."""
    users = sorted({user for user, _ in INTERACTIONS})
    items = sorted({item for _, item in INTERACTIONS})
    ratings = np.zeros((len(users), len(items)))
    mask = np.zeros_like(ratings, dtype=bool)
    for user, item in INTERACTIONS:
        ratings[users.index(user), items.index(item)] = 1.0
        mask[users.index(user), items.index(item)] = True

    density = mask.sum() / mask.size
    print(f"Matrix shape: {ratings.shape[0]} users x {ratings.shape[1]} items")
    print(f"Observed cells: {mask.sum()} of {mask.size} ({density:.1%} dense)")
    print("\nObserved matrix, dot for missing:")
    header = "        " + " ".join(f"i{item:<2d}" for item in items)
    print(header)
    for row, user in enumerate(users):
        cells = " ".join(" 1 " if mask[row, column] else " . " for column in range(len(items)))
        print(f"  u{user:<3d} {cells}")
    return users, items, ratings, mask


def build_ideal(users, items):
    """The matrix that would exist if every user had touched all of their group.
    Rows inside a group are identical, so each group adds exactly one to the rank."""
    ideal = np.zeros((len(users), len(items)))
    for group in GROUPS.values():
        for user in group["users"]:
            for item in group["items"]:
                ideal[users.index(user), items.index(item)] = 1.0
    return ideal


def inspect_rank(ratings, users, items):
    """Part 2. Compare the spectrum of the observed matrix with the complete one.
    SVD reads the unobserved cells as zeros, so the observed matrix is full rank."""
    ideal = build_ideal(users, items)
    values = svd(ratings, compute_uv=False)
    ideal_values = svd(ideal, compute_uv=False)
    print(f"\nObserved matrix, singular values: {np.round(values, 3)}")
    print(f"Complete matrix, singular values: {np.round(ideal_values, 3)}")
    print(f"Rank: {np.linalg.matrix_rank(ratings)} observed, "
          f"{np.linalg.matrix_rank(ideal)} complete, of {min(ratings.shape)} possible")
    energy = values**2
    cumulative = np.cumsum(energy) / energy.sum()
    print(f"First three terms hold {cumulative[RANK - 1]:.2%} of the energy.")
    print(f"Gap between value {RANK} and value {RANK + 1}: "
          f"{values[RANK - 1]:.3f} vs {values[RANK]:.3f}")


def solve_side(fixed, ratings, mask, regularisation):
    """Solve one side while the other is held fixed: a ridge regression per row,
    on the columns that row observed."""
    rank = fixed.shape[1]
    result = np.zeros((ratings.shape[0], rank))
    eye = np.eye(rank)
    for row in range(ratings.shape[0]):
        observed = mask[row]
        if not observed.any():
            continue
        factors = fixed[observed]
        targets = ratings[row, observed]
        gram = factors.T @ factors + regularisation * eye
        result[row] = np.linalg.solve(gram, factors.T @ targets)
    return result


def masked_rmse(user_factors, item_factors, ratings, mask):
    """Root mean squared error over the observed cells only."""
    predicted = user_factors @ item_factors.T
    errors = (predicted - ratings)[mask]
    return float(np.sqrt(np.mean(errors**2)))


def objective(user_factors, item_factors, ratings, mask, regularisation):
    """The quantity the two ridge fits minimise: squared error on observed cells
    plus the penalty on the factor sizes, which the plain RMSE leaves out."""
    predicted = user_factors @ item_factors.T
    squared = float(((predicted - ratings)[mask] ** 2).sum())
    penalty = regularisation * float((user_factors**2).sum() + (item_factors**2).sum())
    return squared + penalty


def fit(ratings, mask, rank, iterations, regularisation, seed, verbose=True):
    """Part 3. Alternate the two ridge fits and record the error curve.
    The factors are kept at every iteration, so part 5 can score an earlier one."""
    # Every cell is predicted as user_factors[u] . item_factors[i], so fixing both
    # tables at `rank` columns caps the prediction matrix at that rank.
    rng = np.random.default_rng(seed)
    item_factors = rng.normal(0.0, 0.1, (ratings.shape[1], rank))
    user_factors = np.zeros((ratings.shape[0], rank))
    history = []
    objectives = []
    snapshots = {}

    for iteration in range(1, iterations + 1):
        user_factors = solve_side(item_factors, ratings, mask, regularisation)
        item_factors = solve_side(user_factors, ratings.T, mask.T, regularisation)
        error = masked_rmse(user_factors, item_factors, ratings, mask)
        cost = objective(user_factors, item_factors, ratings, mask, regularisation)
        history.append(error)
        objectives.append(cost)
        snapshots[iteration] = (user_factors.copy(), item_factors.copy())
        if verbose:
            print(f"  iteration {iteration:>2d}  objective {cost:>9.6f}  RMSE {error:.6f}")
    return snapshots, history, objectives


def group_of_user(user):
    """Return the group label a user belongs to."""
    for label, group in GROUPS.items():
        if user in group["users"]:
            return label
    raise ValueError(f"user {user} belongs to no group")


def recommend(user_factors, item_factors, users, items, mask, top_n):
    """Rank the unseen items for every user by predicted score."""
    scores = user_factors @ item_factors.T
    output = {}
    for row, user in enumerate(users):
        ranked = [
            (items[column], float(scores[row, column]))
            for column in np.argsort(-scores[row])
            if not mask[row, column]
        ]
        output[user] = ranked[:top_n]
    return output


def score_recommendations(recommendations, label):
    """Parts 4 and 5. Print each user's top unseen items and count how often the
    first one is in the user's own group."""
    hits = 0
    print(f"\n{label}")
    for user, ranked in recommendations.items():
        group = group_of_user(user)
        expected = GROUPS[group]["items"]
        top_item, top_score = ranked[0]
        correct = top_item in expected
        hits += int(correct)
        verdict = "in group" if correct else "WRONG GROUP"
        formatted = ", ".join(f"i{item}:{score:.3f}" for item, score in ranked)
        print(f"  u{user:<3d} group {group}  top i{top_item} [{verdict}]   {formatted}")
    rate = hits / len(recommendations)
    print(f"  top-1 inside the correct group: {hits}/{len(recommendations)} = {rate:.1%}")
    return rate


def group_agreement(user_factors, item_factors, users, items, mask):
    """The share score_recommendations prints, without the listing, for the seed sweep."""
    scores = user_factors @ item_factors.T
    hits = 0
    for row, user in enumerate(users):
        expected = GROUPS[group_of_user(user)]["items"]
        ranked = [items[column] for column in np.argsort(-scores[row])
                  if not mask[row, column]]
        hits += int(ranked[0] in expected)
    return hits / len(users)


def early_stopping_across_seeds(ratings, mask, users, items, seeds):
    """Part 5. For each seed, score the round with the lowest RMSE against the last
    round, since where the RMSE bottoms out depends on the random start."""
    print(f"\n{'seed':>8} {'lowest-RMSE round':>19} {'agreement there':>17} "
          f"{'agreement at end':>18} {'verdict':>9}")
    worse = 0
    full = 0
    for seed in seeds:
        snapshots, history, _ = fit(ratings, mask, RANK, MAX_ITERATIONS,
                                    REGULARISATION, seed, verbose=False)
        best = int(np.argmin(history)) + 1
        early = group_agreement(*snapshots[best], users, items, mask)
        final = group_agreement(*snapshots[MAX_ITERATIONS], users, items, mask)
        worse += int(early < final)
        full += int(final == 1.0)
        verdict = "worse" if early < final else "no cost"
        print(f"{seed:>8} {best:>19d} {early:>16.1%} {final:>17.1%} {verdict:>9}")
    print(f"\nStopping at the lowest training error cost accuracy in {worse} of "
          f"{len(seeds)} seeds.")
    print(f"The converged fit reached 100% in {full} of {len(seeds)} seeds.")
    return worse


def build_synthetic(rank, users, items, density, seed):
    """Generate a matrix of known rank, hide most of it, and mark half the hidden
    cells as held out, so each graded cell has a known answer."""
    rng = np.random.default_rng(seed)
    true_users = rng.normal(0.0, 1.0, (users, rank))
    true_items = rng.normal(0.0, 1.0, (items, rank))
    truth = true_users @ true_items.T
    mask = rng.random(truth.shape) < density
    holdout = (~mask) & (rng.random(truth.shape) < 0.5)
    return truth, mask, holdout


def held_out_error(user_factors, item_factors, truth, holdout):
    """RMSE on cells the fit never saw, and that error as a share of the error of
    answering zero everywhere, so a share above 100% is worse than no model."""
    predicted = user_factors @ item_factors.T
    error = float(np.sqrt(np.mean((predicted - truth)[holdout] ** 2)))
    zero_answer = float(np.sqrt(np.mean(truth[holdout] ** 2)))
    return error, error / zero_answer


def error_by_row_count(user_factors, item_factors, truth, mask, holdout):
    """Held-out RMSE for rows grouped by how many cells they observed."""
    predicted = user_factors @ item_factors.T
    per_row = mask.sum(axis=1)
    groups = []
    for label, low, high in (("0-3", 0, 3), ("4-6", 4, 6), ("7-10", 7, 10), ("11+", 11, None)):
        rows = (per_row >= low) if high is None else (per_row >= low) & (per_row <= high)
        cells = holdout & rows[:, None]
        if cells.any():
            error = float(np.sqrt(np.mean((predicted - truth)[cells] ** 2)))
            groups.append((label, int(rows.sum()), error))
    return groups


def synthetic_recovery(rank, users, items, densities, penalty, seed):
    """Part 6. Hide most of a known rank-r matrix and measure the error on hidden cells.
    Penalty and iterations stay fixed while the density moves."""
    print(f"Ground truth: {users}x{items} matrices built from rank {rank}")
    print(f"Penalty held at {penalty}, iterations held at {MAX_ITERATIONS}.")
    print("share = held-out error / error of answering zero, so above 100% is worse than no model.\n")
    print(f"{'density':>8} {'min obs/row':>12} {'mean obs/row':>13} "
          f"{'train RMSE':>11} {'held-out':>9} {'share':>8}")
    for density in densities:
        truth, mask, holdout = build_synthetic(rank, users, items, density, seed)
        observed = truth * mask
        snapshots, history, _ = fit(observed, mask, rank, MAX_ITERATIONS, penalty,
                                    seed, verbose=False)
        error, share = held_out_error(*snapshots[MAX_ITERATIONS], truth, holdout)
        per_row = mask.sum(axis=1)
        print(f"{density:>8.2f} {per_row.min():>12d} {per_row.mean():>13.1f} "
              f"{history[-1]:>11.4f} {error:>9.4f} {share:>7.0%}")

    print("\nHeld-out error, rows grouped by how many cells they observed:")
    for density in (0.05, 0.08):
        truth, mask, holdout = build_synthetic(rank, users, items, density, seed)
        snapshots, _, _ = fit(truth * mask, mask, rank, MAX_ITERATIONS, penalty,
                              seed, verbose=False)
        groups = error_by_row_count(*snapshots[MAX_ITERATIONS], truth, mask, holdout)
        cells = " | ".join(f"{label}: {count} rows {error:.2f}"
                          for label, count, error in groups)
        print(f"  density {density:.2f}  {cells}")
    print("At 0.05 the rows with 11 or more cells fail as badly as the thinnest ones,")
    print("so the whole matrix lacks data, not a few rows. A stronger penalty softens")
    print("the failure but cannot replace the missing data:")

    truth, mask, holdout = build_synthetic(rank, users, items, 0.05, seed)
    observed = truth * mask
    thinnest = int(mask.sum(axis=1).min())
    cells = "cell" if thinnest == 1 else "cells"
    print(f"\n  at density 0.05, where the thinnest row holds {thinnest} observed {cells}")
    print(f"  {'penalty':>9} {'train RMSE':>11} {'held-out':>9} {'share':>8}")
    for candidate in (0.05, 0.3, 1.0, 3.0):
        snapshots, history, _ = fit(observed, mask, rank, MAX_ITERATIONS, candidate,
                                    seed, verbose=False)
        error, share = held_out_error(*snapshots[MAX_ITERATIONS], truth, holdout)
        print(f"  {candidate:>9.2f} {history[-1]:>11.4f} {error:>9.4f} {share:>7.0%}")


def main():
    print("--- 1. The observed matrix ---")
    users, items, ratings, mask = build_matrix()

    print("\n--- 2. Two spectra ---")
    inspect_rank(ratings, users, items)

    print("\n--- 3. The ALS fit ---")
    snapshots, history, objectives = fit(ratings, mask, RANK, MAX_ITERATIONS,
                                        REGULARISATION, SEED)
    falling = all(later <= earlier + 1e-9
                  for earlier, later in zip(objectives, objectives[1:]))
    squared = [error**2 * mask.sum() for error in history]
    penalties = [cost - part for cost, part in zip(objectives, squared)]
    print(f"\nObjective: {objectives[0]:.6f} -> {objectives[-1]:.6f}, "
          f"decreasing every iteration: {falling}")
    print(f"RMSE:      {history[0]:.6f} -> {history[-1]:.6f}")
    print(f"Squared error {squared[0]:.4f} -> {squared[-1]:.4f}, "
          f"penalty {penalties[0]:.2f} -> {penalties[-1]:.2f}")
    print("Almost all of the fall is the penalty: the factors shrink while the fit")
    print("stays close. Only the objective is minimised, and parts 4 and 5 show")
    print("which of the two numbers tracks the recommendations.")

    print("\n--- 4. Group agreement after 20 iterations ---")
    converged_users, converged_items = snapshots[MAX_ITERATIONS]
    converged = recommend(converged_users, converged_items, users, items, mask, top_n=3)
    converged_rate = score_recommendations(
        converged, f"Recommendations after {MAX_ITERATIONS} iterations:")

    print("\n--- 5. The fit stopped after 2 iterations ---")
    early_users, early_items = snapshots[EARLY_STOP]
    early = recommend(early_users, early_items, users, items, mask, top_n=3)
    early_rate = score_recommendations(
        early, f"Recommendations after {EARLY_STOP} iterations:")
    print(f"\nAfter {EARLY_STOP:>2d} iterations: objective {objectives[EARLY_STOP - 1]:.6f}, "
          f"RMSE {history[EARLY_STOP - 1]:.6f}, group agreement {early_rate:.1%}")
    print(f"After {MAX_ITERATIONS:>2d} iterations: objective {objectives[-1]:.6f}, "
          f"RMSE {history[-1]:.6f}, group agreement {converged_rate:.1%}")
    print("The early-stopped fit has the lower RMSE and the worse recommendations,")
    print("so the training error cannot choose when to stop.")
    early_stopping_across_seeds(ratings, mask, users, items, SEED_SWEEP)

    print("\n--- 6. A known rank-3 matrix with most cells hidden ---")
    synthetic_recovery(RANK, users=300, items=120,
                       densities=(0.03, 0.05, 0.08, 0.15, 0.30),
                       penalty=0.3, seed=SEED)


if __name__ == "__main__":
    main()
