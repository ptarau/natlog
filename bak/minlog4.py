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


def new_var(t, d):
    v = d.get(t, None)
    if v is None:
        v = Var()
        d[t] = v
    return v


def activate(t, d):
    if isinstance(t, VarNum):
        return new_var(t, d)
    elif not isinstance(t, tuple):
        return t
    else:
        return tuple(activate(x, d) for x in t)


def unify(x, y, trail, d):
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
        elif isinstance(x2, tuple) and isinstance(x1,tuple):
            arity = len(x2)
            if len(x1) != arity:
                return False
            x1 = activate(x1, d)
            for i in range(arity - 1, -1, -1):
                ustack.append(x2[i])
                ustack.append(x1[i])
        else:
            return False
    return True


def interp(css, goal):
    def step(goals):

        def undo(trail):
            while trail:
                v = trail.pop()
                v.unbind()

        def unfold(g, gs):
            for (h, bs) in css:
                d = dict()
                # h = activate(h, d)
                if not unify(h, g, trail, d):
                    undo(trail)
                    continue  # FAILURE
                else:
                    bs1 = activate(bs, d)
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
        goal_cls = next(mparse(quest, ground=False, rule=False))
        goal = activate(goal_cls, dict())
        yield from interp(self.css, goal)

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
            print('ANSWER:', answer)
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


def test_minlog():
    n = MinLog(file_name="../natlog/natprogs/tc.nat")
    print(n)
    n.query("tc Who is animal ?")

    # n = Natlog(file_name="../natprogs/queens.nat")
    # n.query("goal8 Queens?")

    n = MinLog(file_name="../natlog/natprogs/perm.nat")
    print(n)
    n.query("perm (1 (2 (3 ())))  X ?")


if __name__ == "__main__":
    # test_unify()
    test_minlog()
