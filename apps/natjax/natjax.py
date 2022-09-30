import sys
import jax
from jax import *
import jax.numpy as jnp
from jax.numpy import *
import optax
from sklearn.metrics import accuracy_score

from natlog import Natlog, natprogs

sys.setrecursionlimit(1 << 14)
jax.config.update("jax_enable_x64", True)

DTYPE = jnp.float32


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


def init_weights(features, layer_sizes,seed):
    KEY = jax.random.PRNGKey(seed)
    weights = []
    keys = jax.random.split(KEY, len(layer_sizes) + 1)
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
    return dot(input_data, w.T) + b


@jit
def mlp_forward_pass(weights, input_data):
    layer_out=input_data
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
    opt_state = jit(optimizer.init)(weights)
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
            print(f"Loss at epoch {i} : \t{loss}")

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


# testers

def xor(x, y):
    return x ^ y


def impl(x, y):
    return max(abs(1 - x), y)


def to_jnp(a):
    return jnp.array(a, dtype=DTYPE)


def split(X, y,seed, test_size=0.1):
    print('SHAPES:',X.shape,y.shape)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def load_dataset(features, op, seed):
    from itertools import product

    n = 2 * features
    layer_sizes = [features, n, n + n, n, features, 1]

    def data_x():
        return jnp.array(list(product([-1.0, 1.0], repeat=features)))

    def data_y():
        m = 2 ** features
        rs = []
        for xs in data_x():
            r = 0
            for x in xs:
                x = int((x + 1) / 2)
                r = op(r, x)
            # rs.append(2*r-1)
            rs.append(r)
        ys = to_jnp(rs).reshape(m, 1)
        print(ys)
        return ys

    data = split(data_x(), data_y(), seed)

    epochs = features ** 2
    if op == xor: epochs *= 8
    return data, layer_sizes, epochs


def test_natjax(features, op, seed):
    learning_rate = jnp.array(0.01)

    (X_train, X_test, Y_train, Y_test), layer_sizes, epochs = load_dataset(features, op, seed)
    _, features = X_train.shape

    weights = init_weights(features, layer_sizes, seed)

    weights = train_model(weights, X_train, Y_train, learning_rate, epochs)

    train_score, train_acc = test_model(weights, X_train, Y_train)
    test_score, test_acc = test_model(weights, X_test, Y_test)

    print("Train Loss Score : {:.2f}".format(train_score))
    print("Test  Loss Score : {:.2f}".format(test_score))

    print("Train Accuracy : {:.2f}".format(train_acc))
    print("Test  Accuracy : {:.2f}".format(test_acc))


if __name__ == "__main__":
    #test_natjax(features=12, op=xor)
    run_natlog()
