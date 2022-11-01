from .scanner import VarNum, Var, GVar, deref


def unify(x, y, trail, occ=False):
    ustack = []
    ustack.append(y)
    ustack.append(x)
    while ustack:
        x1 = deref(ustack.pop())
        x2 = deref(ustack.pop())

        if isinstance(x1, GVar) and isinstance(x2, Var):
            x2.bind(x1, trail)
        elif isinstance(x2, GVar) and isinstance(x1, Var):
            x1.bind(x2, trail)
        elif isinstance(x1, Var):
            if occ and occurs(x1, x2):
                return False
            x1.bind(x2, trail)
        elif isinstance(x2, Var):
            if occ and occurs(x2, x1):
                return False
            x2.bind(x1, trail)
        elif isinstance(x2, tuple) and isinstance(x1, tuple):
            arity = len(x1)
            if len(x2) != arity:
                return False
            for i in range(arity - 1, -1, -1):
                ustack.append(x2[i])
                ustack.append(x1[i])
        elif x1 == x2 and type(x1) == type(x2):
            continue
        else:
            return False
    return True


def new_var(t, d):
    v = d.get(t, None)
    if v is None:
        v = Var()
        d[t] = v
    return v

def lazy_unify(x, y, trail, d):
    ustack = []
    ustack.append(y)
    ustack.append(x)
    while ustack:
        x1 = deref(ustack.pop())
        x2 = deref(ustack.pop())

        if isinstance(x1, GVar) and isinstance(x2, Var):
            x2.bind(x1, trail)
        elif isinstance(x2, GVar) and isinstance(x1, Var):
            x1.bind(x2, trail)
        elif isinstance(x1, Var):
            x1.bind(x2, trail)
        elif isinstance(x2, Var):
            x1 = activate(x1, d)
            x2.bind(x1, trail)
        elif type(x1) != type(x2):
            # this should be before next
            return False
        elif isinstance(x2, tuple):  # and isinstance(x1, tuple):
            arity = len(x2)
            if len(x1) != arity:
                return False
            for i in range(arity - 1, -1, -1):
                ustack.append(x2[i])
                ustack.append(activate(x1[i], d))
        elif isinstance(x1, VarNum):
            # conflating int and VarNum not possible now
            x1 = new_var(x1, d)
            x1.bind(x2, trail)
        elif x1 == x2:
            # not tuples, should be other objects
            continue
        else:
            return False
    return True


def activate(t, d):
    if isinstance(t, VarNum):
        return new_var(t, d)
    elif not isinstance(t, tuple):
        return t
    else:
        return tuple(activate(x, d) for x in t)


def extractTerm(t):
    t = deref(t)
    if isinstance(t, Var):
        return t
    elif not isinstance(t, tuple):
        return t
    else:
        return tuple(map(extractTerm, t))


def occurs(x0, t0):
    def occ(t):
        t = deref(t)
        if x == t:
            return True
        if not isinstance(t, tuple):
            return False
        return any(map(occ, t))

    x = deref(x0)
    return occ(t0)


def path_of_(t):
    def path_of0(t):
        if isinstance(t, Var):
            pass
        elif isinstance(t, tuple):
            for i, x in enumerate(t):
                for ps in path_of0(x):
                    yield i, ps
        else:
            yield t

    return set(path_of0(t))


def path_of(t):
    def path_of0(t):
        if isinstance(t, Var):
            pass
        elif isinstance(t, tuple):
            for i, x in enumerate(t):
                for c, ps in path_of0(x):
                    yield c, (i, ps)
        else:
            yield t, ()

    ps = set(path_of0(t))
    qs = set((c, list2tuple(x)) for (c, x) in ps)
    return qs


def list2tuple(ls):
    # print('!!! LS=',ls)
    def scan(xs):
        while xs != () and isinstance(xs, tuple):
            x, xs = xs
            yield x

    if not isinstance(ls, tuple):
        return ls
    return tuple(scan(ls))


def test_unify():
    a, b, c, d, e, f, g = "abcdefg"
    x, y, z = VarNum(0), VarNum(1), VarNum(2)
    t = (f, a, (g, (b, x, (e, b, c, y)), d))
    for p in path_of(t): print('PATH:', p)

    c = activate(t, dict())

    print('ORIG:', t)
    print('COPY:', c)

    z = activate(z, dict())
    print('Z:', z)

    t1 = (f, z)
    t2 = z

    print('unif occ:', unify(t1, t2, [], occ=True))
    print('unif nocc:', unify(t1, t2, [], occ=False))


if __name__ == "__main__":
    test_unify()
