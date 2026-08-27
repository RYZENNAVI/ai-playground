"""Build the same network four times, from hand-derived gradients up to one call to fit.

Demonstrates what each rung of the abstraction ladder takes over, and what it does not:
    1. Fix one set of starting weights and one training rule for every rung to share.
    2. Train it in numpy, differentiating the loss by hand.
    3. Train it in PyTorch, letting autograd differentiate the same expression.
    4. Train it in TensorFlow with a gradient tape, one step at a time.
    5. Line the three loss curves up and measure how far apart they ever get.
    6. Train it once more through the one-line fitting interface.
    7. Compile the step into a graph and time it against running it eagerly.
    8. Wrap the model in a data-parallel strategy and report what the machine gave back.

Module 07: Machine Learning and Deep Learning Foundations - Framework Ladder.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
PROPERTY = DATA / "property_valuation.csv"

SEED = 20260824
HIDDEN_UNITS = 10
STEPS = 300
LEARNING_RATE = 0.05
TEST_FRACTION = 0.25
TIMED_STEPS = 200

# Everything below runs in float32. The three rungs are compared digit by digit,
# and numpy defaulting to float64 while the two frameworks default to float32
# would put a gap between them that has nothing to do with the abstraction.
DTYPE = np.float32


def load():
    """Load the property table, standardise it and split it once."""
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(PROPERTY)
    target = frame["market_value"].to_numpy().reshape(-1, 1)
    features = frame.drop(columns=["market_value"]).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_FRACTION, random_state=SEED)
    mean, spread = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = ((x_train - mean) / spread).astype(DTYPE)
    x_test = ((x_test - mean) / spread).astype(DTYPE)
    return (x_train, y_train.astype(DTYPE), x_test, y_test.astype(DTYPE),
            list(frame.drop(columns=["market_value"]).columns))


def starting_weights(input_units):
    """Draw one set of starting weights for every rung to begin from.

    All four runs below start from these exact arrays. Without that the curves
    could not be compared at all, and any difference between the frameworks
    would be a difference between two random draws.
    """
    rng = np.random.default_rng(SEED)
    return {
        "w1": rng.normal(0, np.sqrt(2.0 / input_units),
                         (input_units, HIDDEN_UNITS)).astype(DTYPE),
        "b1": np.zeros(HIDDEN_UNITS, dtype=DTYPE),
        "w2": rng.normal(0, np.sqrt(2.0 / HIDDEN_UNITS),
                         (HIDDEN_UNITS, 1)).astype(DTYPE),
        "b2": np.zeros(1, dtype=DTYPE),
    }


def train_numpy(weights, x, y):
    """Rung one: forward pass, hand-derived gradients, explicit update."""
    w1, b1 = weights["w1"].copy(), weights["b1"].copy()
    w2, b2 = weights["w2"].copy(), weights["b2"].copy()
    n = x.shape[0]
    history = []

    for _ in range(STEPS):
        z1 = x @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        prediction = a1 @ w2 + b2
        residual = prediction - y
        history.append(float(np.mean(residual ** 2)))

        d_prediction = (2.0 / n) * residual
        d_w2 = a1.T @ d_prediction
        d_b2 = d_prediction.sum(axis=0)
        d_z1 = (d_prediction @ w2.T) * (z1 > 0.0)
        d_w1 = x.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        w1 -= LEARNING_RATE * d_w1
        b1 -= LEARNING_RATE * d_b1
        w2 -= LEARNING_RATE * d_w2
        b2 -= LEARNING_RATE * d_b2

    return history, {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def train_torch(weights, x, y):
    """Rung two: the same expression, with autograd supplying the derivatives."""
    import torch

    torch.manual_seed(SEED)
    tensors = {name: torch.tensor(value, requires_grad=True)
               for name, value in weights.items()}
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    history = []

    for _ in range(STEPS):
        prediction = torch.relu(xt @ tensors["w1"] + tensors["b1"]) @ tensors["w2"] \
            + tensors["b2"]
        loss = torch.mean((prediction - yt) ** 2)
        history.append(float(loss.item()))

        for tensor in tensors.values():
            if tensor.grad is not None:
                tensor.grad = None
        loss.backward()
        with torch.no_grad():
            for tensor in tensors.values():
                tensor -= LEARNING_RATE * tensor.grad

    return history, {name: tensor.detach().numpy() for name, tensor in tensors.items()}


def train_tensorflow(weights, x, y):
    """Rung three: the same expression again, recorded on a gradient tape."""
    import tensorflow as tf

    variables = {name: tf.Variable(value) for name, value in weights.items()}
    xt = tf.constant(x)
    yt = tf.constant(y)
    history = []

    for _ in range(STEPS):
        with tf.GradientTape() as tape:
            prediction = tf.nn.relu(xt @ variables["w1"] + variables["b1"]) \
                @ variables["w2"] + variables["b2"]
            loss = tf.reduce_mean((prediction - yt) ** 2)
        history.append(float(loss.numpy()))

        gradients = tape.gradient(loss, list(variables.values()))
        for variable, gradient in zip(variables.values(), gradients):
            variable.assign_sub(LEARNING_RATE * gradient)

    return history, {name: variable.numpy() for name, variable in variables.items()}


def build_keras_model(weights):
    """Rung four: the same network as a two-layer model carrying the same weights."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(weights["w1"].shape[0],)),
        tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.layers[0].set_weights([weights["w1"], weights["b1"]])
    model.layers[1].set_weights([weights["w2"], weights["b2"]])
    return model


def main():
    if not PROPERTY.exists():
        raise SystemExit("Run 01_build_tabular_datasets.py first.")

    import tensorflow as tf

    x_train, y_train, x_test, y_test, columns = load()
    weights = starting_weights(x_train.shape[1])
    parameter_count = sum(value.size for value in weights.values())

    print("--- 1. One network, one starting point, one rule ---")
    print(f"    {x_train.shape[1]} inputs -> {HIDDEN_UNITS} relu units -> 1 output, "
          f"{parameter_count} parameters")
    print(f"    {STEPS} full-batch steps of plain gradient descent at "
          f"{LEARNING_RATE}, no momentum, no shuffling")
    print(f"    {len(x_train)} training rows, {len(x_test)} test rows, everything "
          f"in {np.dtype(DTYPE).name}")

    print("\n--- 2. numpy, gradients derived by hand ---")
    numpy_history, numpy_weights = train_numpy(weights, x_train, y_train)
    print(f"    first loss {numpy_history[0]:.6f}, final loss {numpy_history[-1]:.6f}")
    print("    Five lines of chain rule, written out in the loop above.")

    print("\n--- 3. PyTorch, gradients from autograd ---")
    torch_history, torch_weights = train_torch(weights, x_train, y_train)
    print(f"    first loss {torch_history[0]:.6f}, final loss {torch_history[-1]:.6f}")
    print("    The forward expression is the same. loss.backward() replaces the")
    print("    five lines, and the update is still written out by hand.")

    print("\n--- 4. TensorFlow, gradients from a tape ---")
    tf_history, tf_weights = train_tensorflow(weights, x_train, y_train)
    print(f"    first loss {tf_history[0]:.6f}, final loss {tf_history[-1]:.6f}")
    print("    A tape records the forward pass as it runs, then walks it backwards.")

    print("\n--- 5. How far apart the three ever get ---")
    numpy_curve = np.array(numpy_history)
    torch_curve = np.array(torch_history)
    tf_curve = np.array(tf_history)
    print(f"    {'pair':<22}{'largest gap in loss':>22}{'final gap':>12}")
    pairs = [("numpy vs pytorch", numpy_curve, torch_curve),
             ("numpy vs tensorflow", numpy_curve, tf_curve),
             ("pytorch vs tensorflow", torch_curve, tf_curve)]
    for label, left, right in pairs:
        print(f"    {label:<22}{np.abs(left - right).max():>22.3e}"
              f"{abs(left[-1] - right[-1]):>12.3e}")

    print(f"\n    {'parameter':<12}{'numpy vs pytorch':>20}{'numpy vs tensorflow':>22}")
    for name in ("w1", "b1", "w2", "b2"):
        print(f"    {name:<12}"
              f"{np.abs(numpy_weights[name] - torch_weights[name]).max():>20.3e}"
              f"{np.abs(numpy_weights[name] - tf_weights[name]).max():>22.3e}")
    print("    The three agree to the last few digits a float32 can hold. What")
    print("    autograd removed was the writing of the derivatives, not any part")
    print("    of the arithmetic they stand for.")

    print("\n--- 6. The same run through the one-line interface ---")
    model = build_keras_model(weights)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                  loss="mse")
    fit_history = model.fit(x_train, y_train, epochs=STEPS, batch_size=len(x_train),
                            shuffle=False, verbose=0)
    keras_losses = fit_history.history["loss"]
    print(f"    first loss {keras_losses[0]:.6f}, final loss {keras_losses[-1]:.6f}")
    print(f"    final loss against the hand-written run: "
          f"{abs(keras_losses[-1] - numpy_history[-1]):.3e}")
    print("    Three lines instead of a loop, landing in the same place. What is")
    print("    gone from the file is the update rule and the batching, which are")
    print("    exactly the two things the earlier rungs had to state out loud.")

    predicted = model.predict(x_test, verbose=0)
    print(f"    test MAE {float(np.mean(np.abs(predicted - y_test))):.4f}")

    print("\n--- 7. Running the step eagerly against running it as a graph ---")
    xt = tf.constant(x_train)
    yt = tf.constant(y_train)
    variables = [tf.Variable(weights[name]) for name in ("w1", "b1", "w2", "b2")]

    def step():
        with tf.GradientTape() as tape:
            hidden = tf.nn.relu(xt @ variables[0] + variables[1])
            loss = tf.reduce_mean((hidden @ variables[2] + variables[3] - yt) ** 2)
        gradients = tape.gradient(loss, variables)
        for variable, gradient in zip(variables, gradients):
            variable.assign_sub(LEARNING_RATE * gradient)
        return loss

    compiled = tf.function(step)
    compiled()  # first call traces the graph; timing it would time the compiler

    started = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    eager_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(TIMED_STEPS):
        compiled()
    graph_seconds = time.perf_counter() - started

    print(f"    {TIMED_STEPS} steps eagerly     {eager_seconds:.3f} s")
    print(f"    {TIMED_STEPS} steps as a graph  {graph_seconds:.3f} s")
    print(f"    ratio {eager_seconds / graph_seconds:.2f}x")
    print("    Eager runs each operation as the line is reached. tf.function runs")
    print("    the Python once to record what happened, then replays that record.")
    print("    On a network this small the saving is the Python overhead between")
    print("    operations, which is most of the runtime here and almost none of it")
    print("    on a model large enough to matter.")

    print("\n--- 8. Wrapping the same model in a data-parallel strategy ---")
    strategy = tf.distribute.MirroredStrategy()
    replicas = strategy.num_replicas_in_sync
    devices = [device.device_type for device in tf.config.list_physical_devices()]
    print(f"    visible devices: {devices}")
    print(f"    replicas in sync: {replicas}")

    with strategy.scope():
        parallel = build_keras_model(weights)
        parallel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                         loss="mse")
    parallel_history = parallel.fit(x_train, y_train, epochs=STEPS,
                                    batch_size=len(x_train), shuffle=False, verbose=0)
    parallel_losses = parallel_history.history["loss"]
    print(f"    final loss inside the strategy {parallel_losses[-1]:.6f}, "
          f"outside it {keras_losses[-1]:.6f}")
    print(f"    difference {abs(parallel_losses[-1] - keras_losses[-1]):.3e}")
    print(f"\n    This run had {replicas} replica, so nothing was split and nothing")
    print("    ran in parallel. Reporting it as a speed result would be reporting")
    print("    a measurement that was never taken. What it does show is the")
    print("    programming model: the model, the optimiser and the fit call are")
    print("    unchanged, and the scope is the only new line. Mirrored means each")
    print("    device holds a full copy of the weights and gets a slice of the")
    print("    batch; the gradients are summed across devices before the update,")
    print("    which is why the result does not depend on how many there are.")


if __name__ == "__main__":
    main()
