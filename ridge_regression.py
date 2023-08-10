import matplotlib.pyplot as plt
import jax.numpy.linalg as jl
import jax.numpy as jnp
import jax

from chex import Array, Scalar

key = jax.random.PRNGKey(43)
key, key_2, key_3 = jax.random.split(key, 3)


# === Define the Data ===
# =======================

# Define the latent transformation (to be discovered)
# 2D -> 1D
W_latent = jnp.array([
    [2.1, -3.7]
])
noise = 1.0
n_samples = 10000
split_ratio = 0.8

# Define the data
X = jax.random.normal(key_2, shape=(2, n_samples))
Y = jnp.squeeze(W_latent @ X)
Y += noise * jax.random.normal(key_3, Y.shape)

# Split to train/test
split = int(split_ratio * n_samples)
X_train, X_test = X[:, :split], X[:, split:]
Y_train, Y_test = Y[:split], Y[split:]

# Present the data
print(f"X train-test shapes: {X_train.shape}\t{X_test.shape}")
print(f"Y train-test shapes: {Y_train.shape}\t{Y_test.shape}")


# === The Ridge Regression ===
# ============================

# define the ridge regression
@jax.jit
def ridge_weights(x_batch: Array, y_batch: Array, l2: Scalar = 0.) -> Array:
    return y_batch @ x_batch.T @ jl.inv((x_batch @ x_batch.T + l2 * jnp.identity(x_batch.shape[0])))

# Predict on test data
W_discovered = ridge_weights(X_train, jnp.expand_dims(Y_train, axis=0))
Y_pred = jnp.squeeze(W_discovered @ X_test)


# ===== Plots =====
# =================

# Create the figure and axes
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot the data
ax1.scatter(X_train[0], X_train[1], Y_train, s=0.1, label="Train")
ax1.scatter(X_test[0], X_test[1], Y_test, s=0.1, label="Test")

ax2.scatter(X_test[0], X_test[1], Y_test, s=0.1, label="Test", color="tab:orange")
ax2.scatter(X_test[0], X_test[1], Y_pred, s=0.1, label="Prediction", color="tab:green")

# Add a title and labels to the axes
ax1.set_title('Train Test Split')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Y')
ax1.legend()

ax2.set_title('Prediction Evaluation')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Y')
ax2.legend()

# Show the plot
# plt.title("Ridge Regression with JAX")
plt.show()
