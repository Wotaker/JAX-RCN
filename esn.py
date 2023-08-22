import jax.numpy.linalg as jl
import jax.numpy as jnp
import jax

from chex import Array, Scalar, PRNGKey, Shape
from typing import Callable, Tuple


def _make_sparse(key: PRNGKey, k_in: int, dense_matrix: Array) -> Array:
    """
    Make a dense weight matrix sparse.

    Parameters
    ----------
    key : PRNGKey
        The key for the jax random number generation.
    k_in : int
        Determines how many inputs are mapped to one neuron.
    dense_matrix : Array, shape=(input_dim, hidden_dim)
        The randomly initialized layer weights.

    Returns
    -------
    sparse_matrix : Array, shape=(input_dim, hidden_dim)
        The sparse layer weights
    """

    input_dim, hidden_dim = dense_matrix.shape
    new_array = jnp.zeros_like(dense_matrix)
    for col in range(hidden_dim):
        key, key_col = jax.random.split(key)
        indices = jax.random.choice(key_col, input_dim, shape=(k_in,), replace=False)
        new_array = new_array.at[indices, col].set(dense_matrix[indices, col])
    
    return new_array


@jax.jit
def append_ones(matrix: Array) -> Array:

    return jnp.concatenate((matrix, jnp.ones((matrix.shape[0], 1), dtype=jnp.float32)), axis=1)


@jax.jit
def normalize_to_spectral_radius(matrix: Array, new_radius: Scalar = 1.0) -> Array:

    spectral_radius = jnp.absolute(jnp.max(jl.eig(matrix)[0]))

    return matrix / spectral_radius * new_radius


@jax.jit
def ridge_weights(x_batch: Array, y_batch: Array, l2: Scalar = 0.) -> Array:

    return jl.inv((x_batch.T @ x_batch + l2 * jnp.identity(x_batch.shape[1]))) @ x_batch.T @ y_batch


@jax.jit
def input_to_node(
        input_batch: Array,
        weights_in: Array,
        activation_fn: Callable[[Array], Array]
) -> Array:
    
    return activation_fn(input_batch @ weights_in)


@jax.jit
def node_to_node(carry: Tuple, r_prime_n: Array):

    r_n_1, weights_res, activation_fn, leakage = carry
    r_n = (1 - leakage) * r_n_1 + leakage * activation_fn(jnp.expand_dims(r_prime_n, 0) + r_n_1 @ weights_res)

    return (r_n, weights_res, activation_fn, leakage), r_n


@jax.jit
def node_to_output(
        reservoir_states: Array,
        weights_out: Array
) -> Array:
    
    return reservoir_states @ weights_out


class ESN():

    def __init__(
            self,
            key: PRNGKey,
            hidden_nodes: int,
            sparsity_in: float,
            sparsity_node: float,
            input_activation: Callable[[Array], Array],
            node_activation: Callable[[Array], Array],
            spectral_radius: float = 1.0,
            leakage: float = 0.9,
            l2_cost: float = 0.0
    ) -> None:
        
        self.key = key
        self.hidden_nodes = hidden_nodes
        self.sparsity_in = jnp.clip(sparsity_in, 0, 1)
        self.sparsity_node = jnp.clip(sparsity_node, 0, 1)
        self.input_activation = jax.tree_util.Partial(input_activation)
        self.node_activation = jax.tree_util.Partial(node_activation)
        self.spectral_radius = jnp.clip(spectral_radius, 0, 1)
        self.leakage = jnp.clip(leakage, 0, 1)
        self.l2_cost = jnp.clip(l2_cost, 0, jnp.inf)

        self.weights_in = None
        self.weights_res = None
        self.weights_out = None
        self.reservoir_state_init = None
    

    def _init_weights(self, x_train_shape: Shape):
        
         # Generate PRNGKeys
        key_1, key_2, key_3, key_4, key_5 = jax.random.split(self.key, 5)

        # Initialize input weights
        weights_in_shape = (x_train_shape[1] + 1, self.hidden_nodes)
        k_in = int(self.sparsity_in * weights_in_shape[0])
        self.weights_in = _make_sparse(key_2, k_in, jax.random.uniform(key_1, weights_in_shape))

        # Initialize reservoir weights
        weights_res_shape = (self.hidden_nodes, self.hidden_nodes)
        k_node = int(self.sparsity_node * weights_res_shape[0])
        self.weights_res = normalize_to_spectral_radius(
            _make_sparse(key_4, k_node, jax.random.uniform(key_3, weights_res_shape)),
            self.spectral_radius
        )

        # Initialize reservoir
        self.reservoir_state = jax.random.uniform(key_5, (1, self.hidden_nodes))
        


    def fit(self, x_train: Array, y_train: Array):
        """
        something

        Parameters
        ----------
        x_train : Array
            training data of shape `n_examples x n_dims`
        """

        # Initialize weights if None
        if self.weights_in is None or \
            self.weights_res is None or \
            self.weights_out is None or \
            self.reservoir_state is None:
            self._init_weights(x_train.shape)

        # Input to Node
        x_train_ones = append_ones(x_train)
        r_prime_batch = input_to_node(x_train_ones, self.weights_in, self.input_activation)

        # Node to Node
        init = (self.reservoir_state, self.weights_res, self.node_activation, self.leakage)
        carry, r_batch = jax.lax.scan(node_to_node, init, r_prime_batch)
        self.reservoir_state = carry[0]

        # Node to Output
        self.reservoir_states = append_ones(jnp.squeeze(r_batch))
        self.weights_out = ridge_weights(self.reservoir_states, y_train, self.l2_cost)
        
        return self
    

    def predict(self, x_batch: Array):

        # Input to Node
        x_batch_ones = append_ones(x_batch)
        r_prime_batch = input_to_node(x_batch_ones, self.weights_in, self.input_activation)

        # Node to Node
        init = (self.reservoir_state, self.weights_res, self.node_activation, self.leakage)
        carry, r_batch = jax.lax.scan(node_to_node, init, r_prime_batch)
        self.reservoir_state = carry[0]

        # Node to Output
        self.reservoir_states = append_ones(jnp.squeeze(r_batch))
        y_hat = node_to_output(self.reservoir_states, self.weights_out)

        return y_hat
    

    def load_weights(self, weights: Tuple[Array, Array, Array, Array]):
            
            self.weights_in, self.weights_res, self.weights_out, self.reservoir_state_init = weights
    

    def save_weights(self):

        return (self.weights_in, self.weights_res, self.weights_out, self.reservoir_state_init)
    
