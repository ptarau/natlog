from natlog import Natlog, VarNum


def vars(n):
    return map(VarNum, range(n))


def funs(spec):
    return (Fun(x) for x in spec.split())


class Fun:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args):
        if len(args) == 0:
            return self.name
        return (self.name,) + args

    def __repr__(self):
        return str(self.name)


def pytest1():
    a, b, c = "a b c".split()
    X, Y = vars(2)
    css = [
        ((a, 1), ()),
        ((a, 2), ()),
        ((a, 3), ()),
        ((b, 2), ()),
        ((b, 3), ()),
        ((b, 4), ()),
        ((c, X), ((a, X), (b, X))),
    ]
    n = Natlog(clauses=css)
    for cs in css:
        print(cs)
    n.repl()


def pytest2():
    a, b, c = funs("a b c")
    X, Y = vars(2)
    css = [
        (a(1), ()),
        (a(2), ()),
        (a(3), ()),
        (b(2), ()),
        (b(3), ()),
        (b(4), ()),
        (c(X), (a(X), b(X))),
    ]
    for cs in css:
        print(cs)

    n = Natlog(clauses=css)

    n.repl()


pytest2()
