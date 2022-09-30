from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
from sklearn.metrics import accuracy_score

jax.config.update("jax_enable_x64", True)

DTYPE = jnp.float32
SEED = 0


def to_jnp(a):
    return jnp.array(a, dtype=DTYPE)


def split(X, y, test_size=0.1):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=SEED)
    return X_train, X_test, y_train, y_test


def xor(x, y):
    return x ^ y


def impl(x, y):
    return max(abs(1 - x), y)


def load_dataset(features, op):
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

    data = split(data_x(), data_y())

    epochs = features ** 2
    if op == xor: epochs *= 4
    return data, layer_sizes, epochs


def InitializeWeights(features, layer_sizes, seed):
    weights = []
    keys = jax.random.split(seed, len(layer_sizes)+1)
    for i, units in enumerate(layer_sizes):
        if i == 0:
            w = jax.random.uniform(key=keys[i], shape=(units, features), minval=-1.0, maxval=1.0, dtype=DTYPE)
        else:
            w = jax.random.uniform(key=keys[i], shape=(units, layer_sizes[i - 1]), minval=-1.0, maxval=1.0,
                                   dtype=DTYPE)

        b = jax.random.uniform(key=keys[-1], minval=-1.0, maxval=1.0, shape=(units,), dtype=DTYPE)

        weights.append([w, b])

    return weights


@jit
def Relu(x):
    return jnp.maximum(x, 0)


@jit
def Sigmoid(x):
    return 1 / (1 + jnp.exp(-1 * x))


@jit
def LinearLayer(weights, input_data):
    w, b = weights
    return jnp.dot(input_data, w.T) + b


@jit
def ForwardPass(weights, input_data):
    layer_out = input_data
    for i in range(len(weights)-1):
        layer_out = Relu(LinearLayer(weights[i], layer_out))
    last = LinearLayer(weights[-1], layer_out)
    #print("LAST:",last)
    return Sigmoid(last)


@jit
def mse(weights, input_data, actual):
    preds = ForwardPass(weights, input_data)
    return ((preds - actual) ** 2).mean()


Loss = mse


@jit
def CalculateGradients(weights, input_data, actual):
    return grad(Loss)(weights, input_data, actual)


def optimize_grads(weights, input_data, actual, optimizer, opt_state):
    gradients = CalculateGradients(weights, input_data, actual)
    updates, opt_state = optimizer.update(gradients, opt_state)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state


def TrainModel(weights, X, Y, learning_rate, epochs):
    optimizer = optax.adam(learning_rate)
    opt_state = jax.jit(optimizer.init)(weights)

    for i in range(epochs):
        loss = Loss(weights, X, Y)
        weights, opt_state = optimize_grads(weights, X, Y, optimizer, opt_state)

        if i % 50 == 0:
            print("Loss : {:.2f}".format(loss))

    return weights


def run(features, op):
    seed = jax.random.PRNGKey(SEED)
    learning_rate = jnp.array(0.01)

    (X_train, X_test, Y_train, Y_test), layer_sizes, epochs = load_dataset(features, op)
    _, features = X_train.shape

    weights = InitializeWeights(features, layer_sizes, seed)

    weights = TrainModel(weights, X_train, Y_train, learning_rate, epochs)

    train_preds = ForwardPass(weights, X_train)
    train_preds = (train_preds > 0.5).astype(DTYPE)

    test_preds = ForwardPass(weights, X_test)
    test_preds = (test_preds > 0.5).astype(DTYPE)

    print("Train Loss Score : {:.2f}".format(Loss(weights, X_train, Y_train)))
    print("Test  Loss Score : {:.2f}".format(Loss(weights, X_test, Y_test)))

    print("Train Accuracy : {:.2f}".format(accuracy_score(Y_train, train_preds)))
    print("Test  Accuracy : {:.2f}".format(accuracy_score(Y_test, test_preds)))


if __name__ == "__main__":
    run(features=12, op=impl)
    run(features=12, op=xor)
