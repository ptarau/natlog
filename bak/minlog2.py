from mparser import mparse
from mscanner import VarNum


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
        elif not isinstance(x1, tuple):
            return False
        else:
            arity = len(x1)
            if len(x2) != arity:
                return False
            for i in range(arity - 1, -1, -1):
                ustack.append(x2[i])
                ustack.append(x1[i])
    return True


def relocate(t, d):
    if isinstance(t, VarNum):
        v = d.get(t, None)
        if v is None:
            v = Var()
            d[t] = v
        return v
    elif not isinstance(t, tuple):
        return t
    else:
        return tuple(relocate(x, d) for x in t)


def interp(css, goals):
    # vss=list(map(cls_vars,css))
    def step(goals):

        def undo(trail):
            while trail:
                v = trail.pop()
                v.unbind()

        def unfold(g, gs):
            for cs in css:
                h, bs = cs
                d = dict()
                h = relocate(h, d)
                if not unify(h, g, trail):
                    undo(trail)
                    continue  # FAILURE
                else:
                    bs1 = relocate(bs, d)
                    bsgs = gs
                    for b1 in reversed(bs1):
                        bsgs = (b1, bsgs)
                    yield bsgs  # SUCCESS

        if goals == ():
            yield goal
        else:
            trail = []
            g, gs = goals
            for newgoals in unfold(g, gs):
                yield from step(newgoals)
                undo(trail)

    goal = goals[0]
    yield from step((goal))


class MinLog:
    def __init__(self, text=None, file_name=None):
        if file_name:
            with open(file_name, 'r') as f:
                self.text = f.read()
        else:
            self.text = text
        self.css = tuple(mparse(self.text, ground=False, rule=True))

    def solve(self, quest):
        """
         answer generator for given question
        """
        d = dict()
        goals = tuple(mparse(quest, ground=False, rule=False))
        goals = relocate(goals, d)
        yield from interp(self.css, goals)

    def count(self, quest):
        """
        answer counter
        """
        c = 0
        for _ in self.solve(quest):
            c += 1
        return c

    def query(self, quest):
        """
        show answers for given query
        """
        for answer in self.solve(quest):
            print('ANSWER:', *answer)
        print('')

    def repl(self):
        """
        read-eval-print-loop
        """
        print("Type ENTER to quit.")
        while (True):
            q = input('?- ')
            if not q: return
            self.query(q)

    # shows tuples of Nalog rule base
    def __repr__(self):
        xs = [str(cs) + '\n' for cs in self.css]
        return " ".join(xs)


def test_unify1():
    x = Var()
    print(deref(x))

    v = Var()
    u = Var()
    u.bind(42, [])
    v.bind(u, [])
    print(x)
    print(u)
    print(deref(v))


def test_unify():
    X, Y, Z = Var(), Var(), Var()
    f, g, a, b, c = tuple("fgabc")
    t1 = (f, X, (g, a, Y), 10)
    t2 = (f, b, (g, Z, c), 10)
    trail = []

    r = unify(t1, t2, trail)
    print('UNIF:', r)
    print(t1)
    print(t2)
    print(trail)
    t = (f, 0, (g, a, 1), 0)
    d = dict()
    print(relocate(t, d))


def test_minlog():
    n = MinLog(file_name="../natlog/natprogs/tc.nat")
    print(n)
    n.query("tc Who is animal ?")

    n = MinLog(file_name="../natlog/natprogs/queens.nat")
    n.query("goal8 Queens?")


if __name__ == "__main__":
    #test_unify()
    test_minlog()
