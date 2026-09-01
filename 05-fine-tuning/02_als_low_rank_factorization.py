"""Factor a sparse preference matrix into two thin matrices with alternating least squares.

Demonstrates that the same low-rank idea works when most of the matrix is missing:
    1. Build a 12x9 interaction matrix with three groups baked into it.
    2. Separate the rank of the complete matrix from the rank of the observed one.
    3. Solve for user and item factors by alternating between two least squares fits.
    4. Watch the penalised objective fall while the plain error drifts upward.
    5. Score recommendations by group agreement, a check that cannot be argued with.
    6. Compare an early-stopped fit against a converged one, across ten seeds.
    7. Measure how many observed cells per row a rank-3 recovery actually needs.

Module 05: Fine-Tuning - Alternating Least Squares.
"""

import sys

import numpy as np
from scipy.linalg import svd

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
    """Step 1. Lay the interactions out as a dense matrix plus an observed mask.

    Two arrays are needed, not one. The rating array holds the values and the
    mask records which cells were actually observed. Without the mask a missing
    cell and a genuine zero look identical, and the fit would spend its capacity
    explaining zeros that nobody ever recorded.

    There is no ground truth here, and steps 1 to 6 are shaped by that. The
    data is authored as a list of interactions, not carved out of a complete
    matrix, so the 24 observed cells carry a real value while the other 84 hold
    a placeholder zero that nobody ever measured. Nothing in this file states
    what u1 should score on i3. That is why step 5 has to grade with a yes/no
    question - did the top recommendation land in the right group - instead of
    asking how close a prediction is to a correct number. Step 7 turns this
    around: build_synthetic generates the complete matrix first and hides part
    of it, so every hidden cell has an answer waiting and the accuracy question
    becomes askable. Compare the two before reading the step 7 numbers.
    """
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

    Only used for the comparison in step 2. Rows inside a group become identical
    here, which is what makes each group contribute exactly one to the rank.
    """
    ideal = np.zeros((len(users), len(items)))
    for group in GROUPS.values():
        for user in group["users"]:
            for item in group["items"]:
                ideal[users.index(user), items.index(item)] = 1.0
    return ideal


def inspect_rank(ratings, users, items):
    """Step 2. Compare the spectrum of the observed matrix against the complete one.

    The phrase "the structure is rank 3" is about the complete matrix, not the
    one being factorised. Filling every group in makes the rows of a group
    identical, so each group contributes exactly one to the rank and three
    groups give rank 3 with six singular values at zero. Removing one cell per
    row breaks that equality: the observed matrix printed in step 1 is full
    rank, and the six values that should be zero come back as the noise floor
    below. What survives is not a rank of 3 but an energy concentration in the
    first three directions, which is the weaker property a factorisation can
    still exploit.

    Two cautions about the numbers. The cumulative share is computed with the
    unobserved cells set to zero, so those placeholder zeros inflate the
    denominator and the percentage understates the concentration; the gap
    between the third and fourth value is the reading to trust. And none of
    this validates the choice of RANK. The interactions were authored from
    GROUPS, so a step in third place is the construction showing through rather
    than a discovery, a self-check that the data came out as intended. Real
    data rarely offers a clean step, and the rank has to be chosen by trying
    values and comparing what they produce.
    """
    ideal = build_ideal(users, items)
    values = svd(ratings, compute_uv=False)
    ideal_values = svd(ideal, compute_uv=False)
    print(f"\nObserved matrix, singular values: {np.round(values, 3)}")
    print(f"Complete matrix, singular values: {np.round(ideal_values, 3)}")
    print(f"Rank: {np.linalg.matrix_rank(ratings)} observed, "
          f"{np.linalg.matrix_rank(ideal)} complete, of {min(ratings.shape)} possible")
    energy = values**2
    cumulative = np.cumsum(energy) / energy.sum()
    print(f"First three terms hold {cumulative[RANK - 1]:.2%} of the energy, "
          f"understated by the placeholder zeros in the denominator.")
    print(f"Gap between value {RANK} and value {RANK + 1}: "
          f"{values[RANK - 1]:.3f} vs {values[RANK]:.3f}")


def solve_side(fixed, ratings, mask, regularisation):
    """Solve one half of the problem while the other half is held constant.

    For each row this is an ordinary ridge regression: take only the columns the
    row actually observed, and find the factor vector that best reproduces those
    observations. Holding one side fixed is what turns a hard joint problem into
    two easy ones, and it is the entire trick behind the method's name.
    """
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
    """The quantity the two least squares fits actually minimise.

    Each half-step solves a ridge regression, so what falls every iteration is
    the squared error over observed cells plus the penalty on the factor sizes.
    The plain RMSE leaves the penalty term out, which is why it can drift upward
    while the fit is still improving: the solver is trading a little training
    error for much smaller factors, and that trade is the point of the penalty.
    """
    predicted = user_factors @ item_factors.T
    squared = float(((predicted - ratings)[mask] ** 2).sum())
    penalty = regularisation * float((user_factors**2).sum() + (item_factors**2).sum())
    return squared + penalty


def fit(ratings, mask, rank, iterations, regularisation, seed, verbose=True):
    """Step 3-4. Alternate the two least squares fits and record the error curve.

    The returned snapshots make the early-stopping comparison in step 6 possible:
    the factors are kept at every iteration, so a fit stopped after two rounds
    can be scored against the same data as the converged one.
    """
    # The model is defined here, and nowhere else. Every cell of the matrix,
    # observed or not, is declared to be the dot product of one row of each
    # factor table:
    #
    #     prediction[u, i] = user_factors[u] . item_factors[i]
    #     whole matrix     = user_factors @ item_factors.T
    #                        (users, rank) @ (rank, items) -> (users, items)
    #
    # Fixing both tables at `rank` columns is the same as declaring that the
    # prediction matrix has rank at most `rank`, since a matrix factors into
    # two such tables exactly when its rank fits inside them. That declaration
    # is what makes the unobserved cells computable at all: a user's row of
    # numbers is reused for every item, so fitting the observed cells drags the
    # missing ones along with them. The expression below is written out again
    # in masked_rmse, objective, recommend and held_out_error rather than
    # shared, because each of those reads it for a different purpose.
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
    """Step 5. Count how often the top recommendation stays inside the user's group.

    Group agreement is deterministic: every user has exactly one correct group,
    and the item they have not touched yet is known in advance. A fit that only
    drives the training error down but recommends across groups has not learned
    the structure, and this counter says so without any interpretation.
    """
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
    """Share of users whose top unseen item belongs to their own group.

    The same quantity score_recommendations prints, computed without the
    per-user listing so it can be called once per seed.
    """
    scores = user_factors @ item_factors.T
    hits = 0
    for row, user in enumerate(users):
        expected = GROUPS[group_of_user(user)]["items"]
        ranked = [items[column] for column in np.argsort(-scores[row])
                  if not mask[row, column]]
        hits += int(ranked[0] in expected)
    return hits / len(users)


def early_stopping_across_seeds(ratings, mask, users, items, seeds):
    """Step 6. Repeat the early-stopping comparison over a range of seeds.

    One run shows that the round with the lowest error scored worse than the
    last round. That is a single observation, and the round it lands on depends
    on the random start. Repeating over seeds turns it into a rate: the column
    below records where the error bottomed out, and whether stopping there cost
    anything measurable on the group check.
    """
    print(f"\n{'seed':>8} {'lowest-RMSE round':>19} {'agreement there':>17} "
          f"{'agreement at end':>18} {'verdict':>9}")
    worse = 0
    for seed in seeds:
        snapshots, history, _ = fit(ratings, mask, RANK, MAX_ITERATIONS,
                                    REGULARISATION, seed, verbose=False)
        best = int(np.argmin(history)) + 1
        early = group_agreement(*snapshots[best], users, items, mask)
        final = group_agreement(*snapshots[MAX_ITERATIONS], users, items, mask)
        worse += int(early < final)
        verdict = "worse" if early < final else "no cost"
        print(f"{seed:>8} {best:>19d} {early:>16.1%} {final:>17.1%} {verdict:>9}")
    print(f"\nStopping at the lowest training error cost accuracy in {worse} of "
          f"{len(seeds)} seeds.")
    return worse


def build_synthetic(rank, users, items, density, seed):
    """Generate a matrix of known rank, hide most of it, and keep a held-out part.

    Because the ground truth is generated here, the question stops being "did the
    training error go down" and becomes "did the factors reproduce cells that
    were never shown". Those two questions have different answers.

    This is the first complete matrix in the file, and the difference from
    build_matrix is worth stating plainly. There the interactions came first
    and the matrix was assembled around them, so the unobserved cells never had
    a value to be right or wrong about. Here `truth` is computed for all
    users x items before anything is hidden, so a cell removed by `mask` still
    has a known answer sitting in `truth`, and `holdout` names the subset that
    will be graded. The larger shape is part of the same purpose: 108 cells
    cannot support a density sweep, since sampling 3% of them leaves noise
    rather than a trend, and the min-observations-per-row column needs enough
    rows to range from 1 to 22.
    """
    rng = np.random.default_rng(seed)
    true_users = rng.normal(0.0, 1.0, (users, rank))
    true_items = rng.normal(0.0, 1.0, (items, rank))
    truth = true_users @ true_items.T
    mask = rng.random(truth.shape) < density
    holdout = (~mask) & (rng.random(truth.shape) < 0.5)
    return truth, mask, holdout


def held_out_error(user_factors, item_factors, truth, holdout):
    """RMSE on cells the fit never saw, plus that error as a share of magnitude.

    The share matters more than the raw number. If the error reaches the average
    absolute value of a held-out cell, the factorisation has become worse than
    answering zero for every unseen cell, and a raw RMSE alone never says so.
    """
    predicted = user_factors @ item_factors.T
    error = float(np.sqrt(np.mean((predicted - truth)[holdout] ** 2)))
    magnitude = float(np.abs(truth[holdout]).mean())
    return error, error / magnitude


def synthetic_recovery(rank, users, items, densities, penalty, seed):
    """Step 7. Find out how much data a rank-r factorisation actually needs.

    A row with exactly r observed cells can be fitted perfectly by r free
    parameters, so its training error goes to zero while its predictions carry no
    information at all. Sparsity, not the number of iterations, is what decides
    whether recovery is possible: the sweep below keeps the penalty and the
    iteration count fixed and moves only the density, and the held-out error
    falls by two orders of magnitude across the range.
    """
    print(f"Ground truth: {users}x{items} matrices built from rank {rank}")
    print(f"Penalty held at {penalty}, iterations held at {MAX_ITERATIONS}.\n")
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

    print(f"\nThe rows that break recovery are the ones holding at most {rank} cells,")
    print(f"which is the number of free parameters a rank-{rank} row already has.")
    print("A stronger penalty softens the failure but cannot replace the missing data:")

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
    print("--- 1. Build a sparse interaction matrix ---")
    users, items, ratings, mask = build_matrix()

    print("\n--- 2. Compare the observed spectrum against the complete one ---")
    inspect_rank(ratings, users, items)

    print("\n--- 3-4. Alternate the two least squares fits ---")
    snapshots, history, objectives = fit(ratings, mask, RANK, MAX_ITERATIONS,
                                        REGULARISATION, SEED)
    falling = all(later <= earlier + 1e-9
                  for earlier, later in zip(objectives, objectives[1:]))
    print(f"\nObjective: {objectives[0]:.6f} -> {objectives[-1]:.6f}, "
          f"decreasing every iteration: {falling}")
    print(f"RMSE:      {history[0]:.6f} -> {history[-1]:.6f}")
    print("The RMSE ends higher than it started while the objective falls the whole")
    print("way. Only one of these two numbers is being minimised, and step 5 shows")
    print("which one tracks the quality of the recommendations.")

    print("\n--- 5. Score the converged fit by group agreement ---")
    converged_users, converged_items = snapshots[MAX_ITERATIONS]
    converged = recommend(converged_users, converged_items, users, items, mask, top_n=3)
    converged_rate = score_recommendations(
        converged, f"Recommendations after {MAX_ITERATIONS} iterations:")

    print("\n--- 6. Score a fit that was stopped early on the same check ---")
    early_users, early_items = snapshots[EARLY_STOP]
    early = recommend(early_users, early_items, users, items, mask, top_n=3)
    early_rate = score_recommendations(
        early, f"Recommendations after {EARLY_STOP} iterations:")
    print(f"\nAfter {EARLY_STOP:>2d} iterations: objective {objectives[EARLY_STOP - 1]:.6f}, "
          f"RMSE {history[EARLY_STOP - 1]:.6f}, group agreement {early_rate:.1%}")
    print(f"After {MAX_ITERATIONS:>2d} iterations: objective {objectives[-1]:.6f}, "
          f"RMSE {history[-1]:.6f}, group agreement {converged_rate:.1%}")
    print("The early-stopped fit has the lower RMSE of the two and the worse")
    print("recommendations. Stopping the alternation early still prints a number,")
    print("so the group check decides the outcome and the training error does not.")
    early_stopping_across_seeds(ratings, mask, users, items, SEED_SWEEP)

    print("\n--- 7. Recover a known rank-3 signal from a larger matrix ---")
    synthetic_recovery(RANK, users=300, items=120,
                       densities=(0.03, 0.05, 0.08, 0.15, 0.30),
                       penalty=0.3, seed=SEED)


if __name__ == "__main__":
    main()
