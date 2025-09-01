from natlog.unify import VarNum, Var, GVar, deref


def copy_term(t0):
    def ct(t):
        t = deref(t)
        if isinstance(t, GVar):
            return t
        if isinstance(t, Var):
            return d.setdefault(t, Var())
        if not isinstance(t, tuple):
            return t
        return tuple(map(ct, t))

    d = dict()
    # print('CT <<<',t0)
    r = ct(t0)
    # print('CT >>>', r)
    return r


def numbervars(t0):

    def nv(t):
        t = deref(t)
        if isinstance(t, GVar):
            return t
        if isinstance(t, Var):
            if t in d:
                return d[t]
            else:
                n = VarNum(len(d) + 1)
                d[t] = n
                return n
        if not isinstance(t, tuple):
            return t
        return tuple(map(nv, t))

    d = dict()

    return nv(t0)


def var_tuple(n):
    return tuple(Var() for i in range(n))


def arg(x, i):
    return x[i]


def setarg(x, i, v):
    x[i] = v


def crop(a, l1, l2):
    return a[l1:l2]


def to_dict(kvs):
    return dict((kv[0], kv[1]) for kv in kvs)


def from_dict(d):
    return tuple(d.items())


def in_dict(d):
    yield from d.items()


def meth_call(o, f, xs):
    m = getattr(o, f)
    return m(*xs)


def write(args):
    print(*args, end=" ")


def nl():
    print()


def writeln(args):
    write(args)
    nl()


def has_type(x, tname):
    t = (eval(tname),)
    return int(isinstance(x, t))


# globals vars for "." and "?" escapes
# refer to them as $eos Dot,$eoq Qest.

eos = "."
eoq = "?"
