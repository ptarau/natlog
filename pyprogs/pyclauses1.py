from natlog import Natlog, VarNum


def vars(n):
    return map(VarNum, range(n))


def funs(spec):
    return (Fun(x) for x in spec.split())


class Fun:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args):
        # 0-arity functors come out as plain atoms (strings), like Prolog
        if len(args) == 0:
            return Term(self.name)  # keep as Term for DSL (&) but serialize to atom
        return Term(self.name, *args)

    def __repr__(self):
        return self.name


# ---------- DSL core ----------
class Term:
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    # Prolog-ish printing for debugging
    def __repr__(self):
        if not self.args:
            return f"{self.name}"
        return f"{self.name}(" + ", ".join(map(repr, self.args)) + ")"

    # a(X) & b(Y)  ==> Body([a(X), b(Y)])
    def __and__(self, other):
        if isinstance(other, Body):
            return Body([self, *other.terms])
        return Body([self, other])

    # a(X) <= (b(X) & c() & d(Y))  ==> ((..., ...), [ ..., ..., ... ])
    def __le__(self, rhs):
        lits = []
        if isinstance(rhs, Body):
            lits = rhs._to_list()
        elif rhs == True or rhs is None or rhs is ():
            lits = []
        elif isinstance(rhs, list):
            lits = [_as_tuple(t) for t in rhs]
        else:
            lits = [_as_tuple(rhs)]
        return (_as_tuple(self), tuple(lits))

    def as_tuple(self):
        return _as_tuple(self)


class Body:
    def __init__(self, terms):
        # Flatten nested Bodies and normalize to Terms/atoms
        flat = []
        for t in terms:
            if isinstance(t, Body):
                flat.extend(t.terms)
            else:
                flat.append(t)
        self.terms = flat

    def __and__(self, other):
        if isinstance(other, Body):
            return Body(self.terms + other.terms)
        return Body(self.terms + [other])

    def _to_list(self):
        return [_as_tuple(t) for t in self.terms]


def _as_tuple(x):
    """Convert Terms recursively to the nested tuple/atom representation your engine expects."""
    if isinstance(x, Term):
        if not x.args:
            # 0-arity: represent as atom (string), same as your current style
            return x.name
        return (x.name, *(_as_tuple(a) for a in x.args))
    elif isinstance(x, list) or isinstance(x, tuple):
        return [_as_tuple(e) for e in x]
    else:
        # Var, int, str atoms already fine
        return x


def prog():
    a, b, c, d = funs("a b c d")
    X, Y = VarNum(0), VarNum(1)
    clss = (
        a(1) <= True,
        a(2) <= (),
        a(3) <= (),
        a(X, Y) <= b(X) & c(Y) & d(Y),
        b(1) <= (),
        b(2) <= (),
        b(X) <= c(X),
        c(2) <= (),
        c(3) <= (),
        d(X) <= b(X),
        c(b(X), Y) <= a(X) & b(Y),
    )

    print("\nCLAUSES\n")
    for cls in clss:
        print(cls)

    n = Natlog(clauses=clss)

    n.repl()


prog()
