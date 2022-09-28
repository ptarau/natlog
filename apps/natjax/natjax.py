import jax
from jax import *
import jax.numpy as jnp
from jax.numpy import *
import optax
from sklearn.metrics import accuracy_score

from natlog import Natlog, natprogs

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float32
SEED = jax.random.PRNGKey(0)

shared = dict()


def share(f):
    shared[f.__name__] = f
    return f


def share_syms():
    for n, f in globals().items():
        if n not in {'add'}:
            shared[n] = f
    return shared


# NN design
@jit
def relu(x):
    return jnp.maximum(x, 0)


@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-1 * x))


@jit
def matsum(x, y):
    return jnp.add(x, y)


def init_weights(features, layer_sizes):
    weights = []
    keys = jax.random.split(SEED, len(layer_sizes) + 1)
    for i, units in enumerate(layer_sizes):
        if i == 0:
            w = jax.random.uniform(key=keys[i], shape=(units, features), minval=-1.0, maxval=1.0, dtype=DTYPE)
        else:
            w = jax.random.uniform(key=keys[i], shape=(units, layer_sizes[i - 1]), minval=-1.0, maxval=1.0,
                                   dtype=DTYPE)

        b = jax.random.uniform(key=keys[-1], minval=-1.0, maxval=1.0, shape=(units,), dtype=DTYPE)

        weights.append((w, b))

    return weights


@jit
def linear_layer(weights, input_data):
    w, b = weights
    return jnp.dot(input_data, w.T) + b


@jit
def mlp_forward_pass(weights, input_data):
    layer_out = input_data
    for i in range(len(weights) - 1):
        layer_out = relu(linear_layer(weights[i], layer_out))
    return sigmoid(linear_layer(weights[-1], layer_out))


@jit
def mse_loss(weights, input_data, actual):
    preds = mlp_forward_pass(weights, input_data)
    return ((preds - actual) ** 2).mean()


@jit
def apply_grad(weights, input_data, actual):
    return grad(mse_loss)(weights, input_data, actual)


def init_optimizer(weights, learning_rate):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)
    return optimizer, opt_state


def apply_optimizer(weights, input_data, actual, optimizer, opt_state):
    gradients = apply_grad(weights, input_data, actual)
    updates, opt_state = optimizer.update(gradients, opt_state)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state


def train_model(weights, X, Y, learning_rate, epochs):
    optimizer, opt_state = init_optimizer(weights, learning_rate)

    for i in range(epochs):
        loss = mse_loss(weights, X, Y)
        weights, opt_state = apply_optimizer(weights, X, Y, optimizer, opt_state)

        if i % 50 == 0:
            print("Loss : {:.2f}".format(loss))

    return weights


def test_model(weights, X, Y):
    preds = mlp_forward_pass(weights, X)
    preds = (preds > 0.5).astype(DTYPE)
    loss = mse_loss(weights, X, Y)
    score = accuracy_score(Y, preds)
    return loss, score


# tools exported to natlog

# Natlog activation
def run_natlog():
    share_syms()
    n = Natlog(file_name="natjax.nat",
               with_lib=natprogs() + "lib.nat", callables=shared)
    # n.query("eq Started 'Natlog'.")
    # n.query("`eye 4 M, `matmul M M X, #print X, fail?")
    # n.query('go?'),
    n.repl()


if __name__ == "__main__":
    # test_natjax()
    run_natlog()
