"""Write a network with nothing but arrays, and check every gradient against the definition.

Demonstrates the whole of a small network, with no framework anywhere in it:
    1. Define three activation functions with their derivatives, and tabulate both.
    2. Multiply those derivatives layer after layer, and measure where the gradient goes.
    3. Push one input through a three-layer network by hand, printing every intermediate.
    4. Derive the gradients on paper and check them against a finite difference.
    5. Train on a real table, once with the bias gradients and once without them.
    6. Read the fitted first layer against the coefficients the target was built from.

Module 07: Machine Learning and Deep Learning Foundations - Networks from Scratch.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
PROPERTY = DATA / "property_valuation.csv"

SEED = 20260824
HIDDEN_UNITS = 24
EPOCHS = 4000
LEARNING_RATE = 0.02
TEST_FRACTION = 0.25
GRADIENT_CHECK_EPSILON = 1e-6

# The worked example: two inputs, two hidden layers of three and two units, one
# output. Every number is small enough to verify with a calculator, which is the
# only reason these particular values are here.
EXAMPLE_INPUT = np.array([1.0, 0.5])
EXAMPLE_WEIGHTS = {
    "w1": np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
    "b1": np.array([0.1, 0.2, 0.3]),
    "w2": np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
    "b2": np.array([0.1, 0.2]),
    "w3": np.array([[0.1, 0.3], [0.2, 0.4]]),
    "b3": np.array([0.1, 0.2]),
}


def sigmoid(x):
    """Squash any real number into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Slope of sigmoid, which peaks at 0.25 when x is zero."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x):
    """Squash into (-1, 1), centred on zero."""
    return np.tanh(x)


def tanh_derivative(x):
    """Slope of tanh, which peaks at 1.0 when x is zero."""
    return 1.0 - np.tanh(x) ** 2


def relu(x):
    """Pass positive values through, flatten the rest to zero."""
    return np.maximum(x, 0.0)


def relu_derivative(x):
    """Slope of relu: one where the input was positive, zero elsewhere."""
    return (np.asarray(x) > 0.0).astype(float)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
}


def tabulate_activations():
    """Print each activation and its slope at a handful of inputs."""
    points = [-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0]
    print(f"    {'x':>7}" + "".join(
        f"{name:>11}{'slope':>9}" for name in ACTIVATIONS))
    for x in points:
        line = f"    {x:>7.1f}"
        for function, derivative in ACTIVATIONS.values():
            line += f"{function(x):>11.4f}{derivative(x):>9.4f}"
        print(line)

    grid = np.linspace(-8, 8, 4001)
    print()
    for name, (_, derivative) in ACTIVATIONS.items():
        values = derivative(grid)
        print(f"    {name:<8} largest slope anywhere {values.max():.4f} "
              f"at x = {grid[int(values.argmax())]:+.2f}")


def measure_vanishing(depth):
    """Multiply the largest possible slope of each activation through a stack.

    The product below uses the best case for each activation, the slope at its
    own peak. A real network cannot beat these numbers, so what they show is a
    ceiling rather than a typical run.
    """
    print(f"    {'layers':>7}" + "".join(f"{name:>16}" for name in ACTIVATIONS))
    for layers in depth:
        line = f"    {layers:>7}"
        for name, (_, derivative) in ACTIVATIONS.items():
            best = derivative(np.linspace(-8, 8, 4001)).max()
            line += f"{best ** layers:>16.3e}"
        print(line)


def forward_example():
    """Run one input through the worked network, printing every step."""
    w = EXAMPLE_WEIGHTS
    x = EXAMPLE_INPUT

    z1 = x @ w["w1"] + w["b1"]
    a1 = sigmoid(z1)
    z2 = a1 @ w["w2"] + w["b2"]
    a2 = sigmoid(z2)
    z3 = a2 @ w["w3"] + w["b3"]

    print(f"    input                       {x}")
    print(f"    layer 1 weighted sum        {np.round(z1, 6)}")
    print(f"        first unit by hand:     "
          f"{x[0]} x {w['w1'][0, 0]} + {x[1]} x {w['w1'][1, 0]} + {w['b1'][0]} "
          f"= {x[0] * w['w1'][0, 0] + x[1] * w['w1'][1, 0] + w['b1'][0]:.4f}")
    print(f"    layer 1 after sigmoid       {np.round(a1, 6)}")
    print(f"    layer 2 weighted sum        {np.round(z2, 6)}")
    print(f"    layer 2 after sigmoid       {np.round(a2, 6)}")
    print(f"    output, no activation       {np.round(z3, 6)}")
    print("    Nothing above is approximate. A network at this size is a pair of")
    print("    matrix products with a squash in between, and it can be checked")
    print("    against a calculator line by line.")
    return z3


def initialise(input_units, hidden_units, rng):
    """Draw starting weights with a scale that keeps the first pass in range."""
    return {
        "w1": rng.normal(0, np.sqrt(2.0 / input_units), (input_units, hidden_units)),
        "b1": np.zeros(hidden_units),
        "w2": rng.normal(0, np.sqrt(2.0 / hidden_units), (hidden_units, 1)),
        "b2": np.zeros(1),
    }


def forward(parameters, x):
    """Run the training network forward, keeping what backprop will need."""
    z1 = x @ parameters["w1"] + parameters["b1"]
    a1 = relu(z1)
    prediction = a1 @ parameters["w2"] + parameters["b2"]
    return prediction, {"z1": z1, "a1": a1}


def loss_and_gradients(parameters, x, y):
    """Return mean squared error and the gradient of that same mean.

    Both halves use the mean over the batch. Taking the loss as a mean and its
    gradient as a sum leaves the two off by the batch size, which does not raise
    anything: it silently rescales the learning rate by however many rows are in
    the batch.
    """
    n = x.shape[0]
    prediction, cache = forward(parameters, x)
    residual = prediction - y
    loss = float(np.mean(residual ** 2))

    d_prediction = 2.0 * residual / n
    gradients = {
        "w2": cache["a1"].T @ d_prediction,
        "b2": d_prediction.sum(axis=0),
    }
    d_a1 = d_prediction @ parameters["w2"].T
    d_z1 = d_a1 * relu_derivative(cache["z1"])
    gradients["w1"] = x.T @ d_z1
    gradients["b1"] = d_z1.sum(axis=0)
    return loss, gradients


def gradient_check(parameters, x, y, rng):
    """Compare a few analytic gradients against a finite difference of the loss."""
    _, analytic = loss_and_gradients(parameters, x, y)
    rows = []
    for name in ("w1", "b1", "w2", "b2"):
        flat = parameters[name].reshape(-1)
        picks = rng.choice(flat.size, size=min(3, flat.size), replace=False)
        for index in picks:
            original = flat[index]

            flat[index] = original + GRADIENT_CHECK_EPSILON
            up, _ = loss_and_gradients(parameters, x, y)
            flat[index] = original - GRADIENT_CHECK_EPSILON
            down, _ = loss_and_gradients(parameters, x, y)
            flat[index] = original

            numerical = (up - down) / (2 * GRADIENT_CHECK_EPSILON)
            derived = analytic[name].reshape(-1)[index]
            denominator = max(abs(numerical), abs(derived), 1e-12)
            rows.append((f"{name}[{index}]", derived, numerical,
                         abs(numerical - derived) / denominator))
    return rows


def train(x_train, y_train, x_test, y_test, rng, update_biases=True):
    """Run plain gradient descent, optionally leaving the biases where they started."""
    parameters = initialise(x_train.shape[1], HIDDEN_UNITS, rng)
    history = []
    for _ in range(EPOCHS):
        loss, gradients = loss_and_gradients(parameters, x_train, y_train)
        history.append(loss)
        for name in parameters:
            if not update_biases and name.startswith("b"):
                continue
            parameters[name] -= LEARNING_RATE * gradients[name]
    test_prediction, _ = forward(parameters, x_test)
    test_mae = float(np.mean(np.abs(test_prediction - y_test)))
    return parameters, history, test_mae


def main():
    if not PROPERTY.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("--- 1. Three activations and their slopes ---")
    tabulate_activations()

    print("\n--- 2. What those slopes do when they are multiplied together ---")
    print("    Backpropagation carries the error back through one derivative per")
    print("    layer, multiplying as it goes. Best case per activation:\n")
    measure_vanishing([1, 2, 5, 10, 20, 40])
    print("\n    Sigmoid cannot do better than a quarter per layer, so ten layers")
    print("    of it deliver at most a millionth of the error to the first one.")
    print("    Relu keeps a slope of exactly one wherever its input was positive,")
    print("    which is the whole of why the column on the right is flat.")

    print("\n--- 3. One input, pushed through by hand ---")
    forward_example()

    print("\n--- 4. Checking the derived gradients against the definition ---")
    frame = pd.read_csv(PROPERTY)
    target = frame["market_value"].to_numpy().reshape(-1, 1)
    features = frame.drop(columns=["market_value"]).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_FRACTION, random_state=SEED)
    mean, spread = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - mean) / spread
    x_test = (x_test - mean) / spread

    check_parameters = initialise(x_train.shape[1], HIDDEN_UNITS,
                                  np.random.default_rng(SEED))
    rows = gradient_check(check_parameters, x_train[:64], y_train[:64],
                          np.random.default_rng(SEED))
    print(f"    {'parameter':<12}{'derived':>14}{'finite difference':>20}"
          f"{'relative error':>17}")
    for name, derived, numerical, error in rows:
        print(f"    {name:<12}{derived:>14.8f}{numerical:>20.8f}{error:>17.2e}")
    worst = max(error for *_, error in rows)
    print(f"\n    worst relative error {worst:.2e}")
    print("    A gradient that is wrong still trains, just towards somewhere else.")
    print("    This is the one check that separates the two cases, and it needs no")
    print("    framework: perturb a weight, watch the loss, divide.")

    print("\n--- 5. Training, with and without the bias gradients ---")
    with_bias, history_with, mae_with = train(
        x_train, y_train, x_test, y_test, np.random.default_rng(SEED), True)
    without_bias, history_without, mae_without = train(
        x_train, y_train, x_test, y_test, np.random.default_rng(SEED), False)

    print(f"    target mean {target.mean():.2f}, so the network has to reach a")
    print(f"    non-zero average from standardised inputs that average zero.")
    print(f"    {'run':<22}{'first loss':>13}{'final loss':>13}{'test MAE':>11}")
    print(f"    {'biases updated':<22}{history_with[0]:>13.4f}"
          f"{history_with[-1]:>13.4f}{mae_with:>11.4f}")
    print(f"    {'biases left at zero':<22}{history_without[0]:>13.4f}"
          f"{history_without[-1]:>13.4f}{mae_without:>11.4f}")
    print(f"    final bias of the output unit: {with_bias['b2'][0]:.4f} against "
          f"{without_bias['b2'][0]:.4f}")
    frozen = with_bias["b1"].size + with_bias["b2"].size
    total = sum(value.size for value in with_bias.values())
    print(f"    parameters left frozen: {frozen} of {total} ({frozen / total:.1%})")
    print("    Declaring the biases and never updating them is not a crash and not")
    print("    a warning. It is a small share of the parameters, and it is the")
    print("    share that carries the offset: with the inputs standardised to")
    print("    average zero, the weights alone have nothing to build an average of")
    print(f"    {target.mean():.1f} out of. The loss curve goes down either way.")

    plt.figure(figsize=(9, 5))
    plt.plot(history_with, label="biases updated")
    plt.plot(history_without, label="biases left at zero")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("mean squared error")
    plt.title("Training loss, with and without bias gradients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "training_loss.png", dpi=120)
    plt.close()

    plt.figure(figsize=(9, 5))
    grid = np.linspace(-8, 8, 800)
    for index, (name, (function, derivative)) in enumerate(ACTIVATIONS.items(), start=1):
        plt.subplot(2, 3, index)
        plt.plot(grid, function(grid))
        plt.title(name)
        plt.subplot(2, 3, index + 3)
        plt.plot(grid, derivative(grid))
        plt.title(f"{name} slope")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "activations.png", dpi=120)
    plt.close()
    print(f"    plots written to {OUTPUTS}")

    print("\n--- 6. What the first layer learned, against the generator ---")
    columns = list(frame.drop(columns=["market_value"]).columns)
    influence = pd.Series(
        np.abs(with_bias["w1"]) @ np.abs(with_bias["w2"]).reshape(-1),
        index=columns).sort_values(ascending=False)
    # How much each generator term actually moves the target is the spread of
    # that term over these rows, not its coefficient. Eleven of the twelve terms
    # are linear, so their spread is the coefficient times the column's spread.
    # The twelfth is a tanh and saturates, which caps its swing at roughly twice
    # its coefficient no matter how wide vacancy_rate gets, so it is evaluated
    # rather than approximated.
    linear = {"rooms": 4.9, "pupil_teacher_ratio": -0.79, "school_rating": 0.61,
              "crime_index": -0.55, "distance_to_centre": -0.42, "transit_index": 0.38,
              "noise_level": -6.1, "floor_area": 0.0043, "build_year": 0.021,
              "lot_size": 0.00021, "tax_rate": -0.0072}
    term_spread = {name: abs(coefficient) * float(frame[name].std())
                   for name, coefficient in linear.items()}
    term_spread["vacancy_rate"] = float(
        (-9.5 * np.tanh((frame["vacancy_rate"] - 12.6) / 6.0)).std())
    true_order = sorted(columns, key=lambda n: -term_spread[n])

    print(f"    {'feature':<22}{'term spread':>13}{'true rank':>11}{'network rank':>14}")
    for position, name in enumerate(true_order, start=1):
        print(f"    {name:<22}{term_spread[name]:>13.3f}{position:>11}"
              f"{int(influence.index.get_loc(name)) + 1:>14}")
    top_three = set(true_order[:3]) & set(influence.index[:3])
    print(f"\n    of the three strongest terms in the generator, the network puts "
          f"{len(top_three)} in its own top three")
    print(f"    test MAE {mae_with:.4f} against a target that carries noise of "
          f"sigma 2.20")
    print("    The network was never told there are twelve features or that one of")
    print("    them saturates. It has one hidden layer of relu units and a loop.")


if __name__ == "__main__":
    main()
