import jax
from jax import *
import jax.numpy as jnp
from jax.numpy import *
from natlog import Natlog, natprogs

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float32
SEED = jax.random.PRNGKey(42)

shared = dict()


def share(f):
    shared[f.__name__] = f
    return f


def share_syms():
    for n, f in globals().items():
        if n not in {'add'} :
           shared[n] = f
    # shared['shared'] = shared
    return shared


@jit
def relu(x):
    return jnp.maximum(x, 0)


@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-1 * x))

@jit
def matsum(x,y):
    return jnp.add(x,y)

def init_weights(features, layer_sizes):
    weights = []
    for i, units in enumerate(layer_sizes):
        if i == 0:
            w = jax.random.uniform(key=SEED, shape=(units, features), minval=-1.0, maxval=1.0, dtype=DTYPE)
        else:
            w = jax.random.uniform(key=SEED, shape=(units, layer_sizes[i - 1]), minval=-1.0, maxval=1.0,
                                   dtype=DTYPE)

        b = jax.random.uniform(key=SEED, minval=-1.0, maxval=1.0, shape=(units,), dtype=DTYPE)

        weights.append((w, b))

    return weights


def crop(a, l1, l2):
    return a[l1:l2]


def run_natlog():
    share_syms()
    n = Natlog(file_name="alt_natjax.nat",
               with_lib=natprogs() + "lib.nat", callables=shared)
    #n.query("eq Started 'Natlog'.")
    #n.query("`eye 4 M, `matmul M M X, #print X, fail?")
    #n.query('go?'),
    n.repl()


def test_natjax():
    m = eye(3)
    mm = matmul(m, m)
    print(mm)
    s = sum(array((m, m)), 0)
    print(s)


if __name__ == "__main__":
    test_natjax()
    run_natlog()
