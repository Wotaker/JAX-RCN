import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
import matplotlib.pyplot as plt
import pyrcn
from pyrcn.base.blocks import InputToNode, NodeToNode
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from chex import Array, Scalar
from typing import Tuple

from esn import ESN
from utils import chunkify, plot_predictions

key = jax.random.PRNGKey(43)

N_SAMPLES = 1000
SPLIT_RATIO = 0.8
HISTORY_LEN = 10
FORECAST_LEN = 1

# Load from csv file
mackey_data = jnp.squeeze(jnp.array(pd.read_csv("data/MackeyGlass_1k.csv", index_col=0)))

# Split into train-test
split = int(SPLIT_RATIO * N_SAMPLES)
mackey_train = mackey_data[:split]
mackey_test = mackey_data[split:]

# Split into input and output
X_train, Y_train = chunkify(mackey_train, HISTORY_LEN, FORECAST_LEN)
X_test, Y_test = chunkify(mackey_test, HISTORY_LEN, FORECAST_LEN)

X_train_np, Y_train_np = np.array(X_train), np.array(Y_train)
X_test_np, Y_test_np = np.array(X_test), np.array(Y_test)


def fit_esn_jax(x_train: Array, y_train: Array) -> Array:

    esn_jax = ESN(
        jax.random.PRNGKey(71),
        hidden_nodes=69,
        sparsity_in=1.0,
        sparsity_node=1.0,
        input_activation=lambda x: x,
        node_activation=jax.nn.relu,
        spectral_radius=1.0,
        leakage=0.3156014208642396,
        l2_cost=628.7829947402206
    )
    y_hat_train = esn_jax.fit(X_train, Y_train)
    
    return y_hat_train


