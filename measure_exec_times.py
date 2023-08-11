import os
import timeit
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax
from pyrcn.base.blocks import InputToNode, NodeToNode
from sklearn.linear_model import Ridge

from chex import Array

from esn import ESN
from utils import chunkify
from gen_mackey_glass import generate_mackey_glass

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--repeat', default=25, type=int)
parser.add_argument('--number', default=100, type=int)
parser.add_argument('--hidden', default=100, type=int)
parser.add_argument('--n_samples', default=1000, type=int)
args = parser.parse_args()

# Constants
PRNGKEY = jax.random.PRNGKey(71)
HIDDEN_NODES = args.hidden
N_SAMPLES = args.n_samples
SPLIT_RATIO = 0.8
HISTORY_LEN = 10
FORECAST_LEN = 1

# Load Mackey-Glass dataset
mackey_data = jnp.squeeze(jnp.array(generate_mackey_glass(N_SAMPLES)))

# Split into train-test
split = int(SPLIT_RATIO * N_SAMPLES)
mackey_train = mackey_data[:split]
mackey_test = mackey_data[split:]

# Split into input and output
X_train, Y_train = chunkify(mackey_train, HISTORY_LEN, FORECAST_LEN)
X_test, Y_test = chunkify(mackey_test, HISTORY_LEN, FORECAST_LEN)

X_train_np, Y_train_np = np.array(X_train), np.array(Y_train)
X_test_np, Y_test_np = np.array(X_test), np.array(Y_test)


def fit_predict_esn_jax(x_train: Array, y_train: Array) -> Array:

    esn_jax = ESN(
        PRNGKEY,
        hidden_nodes=HIDDEN_NODES,
        sparsity_in=1.0,
        sparsity_node=1.0,
        input_activation=lambda x: x,
        node_activation=jax.nn.relu,
        spectral_radius=1.0,
        leakage=0.3156014208642396,
        l2_cost=628.7829947402206
    )
    y_hat = esn_jax.fit(x_train, y_train).predict(x_train)
    
    return y_hat


def fit_predict_esn_pyrcn(x_train: Array, y_train: Array) -> Array:

    input_to_node = InputToNode(
        hidden_layer_size=HIDDEN_NODES,
        k_in=None,
        sparsity=1.0,
        input_activation="identity",
    )
    node_to_node = NodeToNode(
        hidden_layer_size=HIDDEN_NODES,
        sparsity=1.0,
        reservoir_activation="relu",
        spectral_radius=1.0,
        leakage=0.3156014208642396,
        bidirectional=False
    )

    R_i2n = input_to_node.fit_transform(x_train)
    R_n2n = node_to_node.fit_transform(R_i2n)
    ridge_train = Ridge(alpha=628.7829947402206).fit(R_n2n, y_train)
    y_hat = ridge_train.predict(R_n2n)

    return y_hat


def measure_jax_times(repeat: int, number: int) -> Array:

    setup_code = """
from __main__ import fit_predict_esn_jax
from __main__ import X_train, Y_train

fit_predict_esn_jax(X_train, Y_train)
"""

    test_code = """
fit_predict_esn_jax(X_train, Y_train)"""

    jax_times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=repeat, number=number)
    return np.array(jax_times) / number


def measure_pyrcn_times(repeat: int, number: int) -> Array:

    setup_code = """
from __main__ import fit_predict_esn_pyrcn
from __main__ import X_train_np, Y_train_np

fit_predict_esn_pyrcn(X_train_np.T, Y_train_np.T)"""

    test_code = """
fit_predict_esn_pyrcn(X_train_np.T, Y_train_np.T)"""

    pyrcn_times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=repeat, number=number)
    return np.array(pyrcn_times) / number



if __name__ == "__main__":

    # Measure execution times
    print("Measuring JAX execution times...")
    start = timeit.default_timer()
    jax_times = measure_jax_times(repeat=args.repeat, number=args.number)
    print(f"Total time: {timeit.default_timer() - start:.4f}")

    print("Measuring PyRCN execution times...")
    start = timeit.default_timer()
    pyrcn_times = measure_pyrcn_times(repeat=args.repeat, number=args.number)
    print(f"Total time: {timeit.default_timer() - start:.4f}")

    # Convert to dataframe
    save_path_numeric = os.path.join("output", "numeric", f"exec-time_{N_SAMPLES}_{HIDDEN_NODES}.csv")
    frameworks = ['JAX' for _ in range(args.repeat)] + ['PyRCN' for _ in range(args.repeat)]
    df = pd.DataFrame({'Framework': frameworks, 'Time': np.concatenate((jax_times, pyrcn_times))})
    df["N_Samples"] = N_SAMPLES # [N_SAMPLES for _ in range(2 * args.repeat)]
    df["Hidden_Nodes"] = HIDDEN_NODES # [HIDDEN_NODES for _ in range(2 * args.repeat)]
    df.to_csv(save_path_numeric, index=False, header=False)

    # Plot results
    save_path_plot = os.path.join("output", "plots", f"exec-time_{N_SAMPLES}_{HIDDEN_NODES}.pdf")
    df["All"] = ""
    ax = sns.violinplot(x="All", y="Time", hue="Framework", data=df, split=True)
    ax.set_xlabel("")
    ax.set_ylabel("Execution Time (s)")
    plt.title(f"Echo State Networks with different frameworks\nN={N_SAMPLES}, H={HIDDEN_NODES}")
    plt.savefig(save_path_plot, bbox_inches='tight')



