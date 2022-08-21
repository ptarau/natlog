from .unify import Var, deref


def copy_term(t0):
    def ct(t):
        t = deref(t)
        if isinstance(t, Var):
            return d.setdefault(t, Var())
        elif not isinstance(t, tuple):
            return t
        else:
            return tuple(map(ct, t))

    d = dict()
    # print('CT <<<',t0)
    r = ct(t0)
    # print('CT >>>', r)
    return r


def arg(x, i):
    return x[i]


def to_dict(kvs):
    return dict((kv[0], kv[1]) for kv in kvs)


def from_dict(d):
    return tuple(d.items())


def in_dict(d):
    yield from d.items()

