import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from chex import Array, Scalar
from typing import Tuple


def chunkify(data: Array, history_len: Scalar, forecast_len: Scalar, transpose: bool = True) -> Tuple[Array, Array]:

    X_data = []
    Y_data = []
    step_len = history_len + forecast_len
    for ptr in range(step_len, data.shape[0] + 1, 1):
        X_data.append(data[ptr-step_len:ptr-forecast_len])
        Y_data.append(data[ptr-forecast_len:ptr])
    
    if transpose:
        return jnp.array(X_data).T, jnp.array(Y_data).T
    else:
        return jnp.array(X_data), jnp.array(Y_data)


def plot_predictions(x_train: Array, y_train: Array, y_hat_train: Array, history_len: int, forecast_len: int):
    for idx in range(x_train.shape[1]):
        x_sample = x_train[:, idx]
        y_sample = y_train[:, idx]
        pred_sample = y_hat_train[:, idx]

        plt.scatter(range(idx,  idx + history_len), x_sample, s=0.5, color="tab:blue", alpha=0.5)
        plt.scatter(range(idx + history_len, idx + history_len + forecast_len), y_sample, s=0.5, color="tab:orange", alpha=0.5)
        plt.scatter(range(idx + history_len, idx + history_len + forecast_len), pred_sample, s=0.5, color="tab:green", alpha=0.5)

    plt.plot([-100], [0], label="x_train", color="tab:blue")
    plt.plot([-100], [0], label="y_train", color="tab:orange")
    plt.plot([-100], [0], label="y_hat", color="tab:green")
    plt.xlim(0, x_train.shape[1] + history_len + forecast_len)
    plt.legend()
    plt.show()
