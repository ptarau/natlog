from .mscanner import VarNum


class Var:
    def __init__(self):
        self.val = None

    def bind(self, val, trail):
        self.val = val
        trail.append(self)

    def unbind(self):
        self.val = None

    def __repr__(self):
        v = deref(self)
        if isinstance(v, Var) and v.val is None:
            return "_" + str(id(v))
        else:
            return repr(v)


def deref(v):
    while isinstance(v, Var):
        if v.val is None:
            return v
        v = v.val
    return v


def unify(x, y, trail):
    ustack = []
    ustack.append(y)
    ustack.append(x)
    while ustack:
        x1 = deref(ustack.pop())
        x2 = deref(ustack.pop())
        if x1 == x2: continue
        if isinstance(x1, Var):
            x1.bind(x2, trail)
        elif isinstance(x2, Var):
            x2.bind(x1, trail)
        elif not (isinstance(x1, tuple) and isinstance(x2, tuple)):
            return False
        else:  # assumed x1,x2 is a tuple
            arity = len(x1)
            if len(x2) != arity:
                return False
            for i in range(arity - 1, -1, -1):
                ustack.append(x2[i])
                ustack.append(x1[i])
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
        if isinstance(x1, VarNum):
            x1 = new_var(x1, d)
            x1.bind(x2, trail)
        elif x1 == x2:
            continue
        elif isinstance(x1, Var):
            x1.bind(x2, trail)
        elif isinstance(x2, Var):
            x1 = activate(x1, d)
            x2.bind(x1, trail)
        elif isinstance(x2, tuple) and isinstance(x1, tuple):
            arity = len(x2)
            if len(x1) != arity:
                return False
            for i in range(arity - 1, -1, -1):
                ustack.append(x2[i])
                ustack.append(activate(x1[i], d))
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
    t=deref(t)
    if isinstance(t, Var):
        return t
    elif not isinstance(t, tuple):
        return t
    else:
        return tuple(map(extractTerm, t))

def copy_term(t0):
    def ct(t):
        t=deref(t)
        if isinstance(t, Var):
            return d.setdefault(t, Var())
        elif not isinstance(t, tuple):
            return t
        else:
            return tuple(map(ct, t))

    d = dict()
    #print('CT <<<',t0)
    r= ct(t0)
    #print('CT >>>', r)
    return r


def arg(x,i) :
    return x[i]

"""
def activate_(t0, d):
    if isinstance(t0, VarNum):
        return new_var(t0, d)
    elif not isinstance(t0, tuple):
        return t0
    top = [None]
    stack = [(top, 0, t0)]

    while stack:
        parent, pos, t = stack.pop()
        c = [None] * len(t)
        for i, x in enumerate(t):
            if isinstance(x, tuple):
                #c[i]=x
                stack.append((c, i, x))
            else:
                if isinstance(x, VarNum):
                    x = new_var(x, d)
                c[i] = x
        parent[pos] = c
    return top[0]

 
class arity(int):
    def __repr__(self):
        return f'$({int(self)})'

def to_postfix(term):
    args = [term]
    stack = []
    while args:
        t = args.pop()
        if not isinstance(t, tuple):
            stack.append(t)
        else:
            stack.append(arity(len(t)))
            for x in reversed(t):
                args.append(x)
    return reversed(stack)

def from_postfix(ws,d):
    stack = []
    for w in ws:
        if not isinstance(w, arity):
            stack.append(w)
        else:
            xs = []
            for _ in range(w):
                x = stack.pop()
                if isinstance(x, VarNum):
                    x= new_var(x, d)
                xs.append(x)
            stack.append(tuple(xs))
    return stack.pop()

def activate(template, d):
    ws=to_postfix(template)
    return from_postfix(ws,d)
"""


"""
def const_of(t):
    def const_of0(t):
        if isinstance(t, Var):
            pass
        elif isinstance(t, tuple):
            for x in t:
                yield from const_of0(x)
        else:
            yield t

    return set(const_of0(t))

def vars_of(t):
    def vars_of0(t):
        if isinstance(t, Var):
            yield t
        elif isinstance(t, tuple):
            for x in t:
                yield from vars_of0(x)
        else:
            pass

    return set(vars_of0(t))
"""


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
                for c,ps in path_of0(x):
                    yield c,(i, ps)
        else:
            yield t,()

    ps=set(path_of0(t))
    qs=set((c,list2tuple(x)) for (c,x) in ps)
    return qs


def list2tuple(ls):
    #print('!!! LS=',ls)
    def scan(xs):
      while xs != () and isinstance(xs,tuple):
        x,xs=xs
        yield x
    if not isinstance(ls,tuple):
        return ls
    return tuple(scan(ls))

def test_unify():
    a, b, c, d, e, f, g, x, y = "abcdefgxy"
    x, y = VarNum(0), VarNum(1)
    t = (f, a, (g, (b, x, (e, b, c, y)), d))
    for p in path_of(t): print('PATH:',p)

    c = activate(t, dict())

    print('ORIG:', t)
    print('COPY:', c)


if __name__ == "__main__":
    test_unify()
