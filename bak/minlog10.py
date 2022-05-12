from mparser import mparse
from mscanner import VarNum


# DERIVED FROM minlog3.py

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
        else:  # assumed x1 is a tuple
            arity = len(x1)
            if len(x2) != arity:
                return False
            for i in range(arity - 1, -1, -1):
                ustack.append(x2[i])
                ustack.append(x1[i])
    return True


def activate(t, d):
    if isinstance(t, VarNum):
        v = d.get(t, None)
        if v is None:
            v = Var()
            d[t] = v
        return v
    elif not isinstance(t, tuple):
        return t
    else:
        return tuple(activate(x, d) for x in t)


def to_python(x):
    return x


def from_python(x):
    return x


def extractTerm(t):
    if isinstance(t, Var):
        return deref(t)
    elif not isinstance(t, tuple):
        return t
    else:
        return tuple(map(extractTerm, t))


def interp(css, goals0):
    def step(goals):

        def undo():
            while trail:
                trail.pop().unbind()

        def unfold(g, gs):
            for (h, bs) in css:
                d = dict()
                h = activate(h, d)
                if not unify(h, g, trail):
                    undo()
                    continue  # FAILURE
                else:
                    # NOT TO BE CHANGED !!!
                    bsgs = gs
                    for b in reversed(bs):
                        b = activate(b, d)
                        bsgs = (b, bsgs)
                    yield bsgs  # SUCCESS

        def python_call(g):
            """
            simple call to Python (e.g., print, no return expected)
            """
            f = eval(g[0])
            args = to_python(g[1:])
            f(*args)

        def python_fun(g, goals):
            """
            function call to Python, last arg unified with result
            """
            f = eval(g[0])
            g = g[1:]
            v = g[-1]
            args = to_python(g[:-1])
            r = f(*args)
            r = from_python(r)
            if not unify(v, r, trail=trail):
                undo()
            else:
                yield from step(goals)

            # unifies with last arg yield from a generator
            # and first args, assumed ground, passed to it

        def gen_call(g, goals):
            gen = eval(g[0])
            g = g[1:]
            v = g[-1]
            args = to_python(g[:-1])
            for r in gen(*args):
                r = from_python(r)
                if unify(v, r, trail=trail):
                    yield from step(goals)
                undo()

        def dispatch_call(op, g, goals):
            """
            dispatches several types of calls to Python
            """
            if op == 'not':
                if neg(g):
                    yield from step(goals)
            elif op == '~':  # matches against database of facts
                yield from db_call(g, goals)
            elif op == '^':  # yield g as an answer directly
                yield g
                yield from step(goals)
            elif op == '`':  # function call, last arg unified
                yield from python_fun(g, goals)
            elif op == "``":  # generator call, last arg unified
                yield from gen_call(g, goals)
            else:  # op == '#',  simple call, no return
                python_call(g)
                yield from step(goals)
            undo()

        def neg(g):
            no_sol = object()
            # g = extractTerm(g)
            a = next(step((g, ())), no_sol)
            if a is no_sol:
                return True
            return False

        trail = []
        if goals == ():
            yield extractTerm(goals0)
        else:
            g, goals = goals
            op = g[0]
            if op in {"not", "~", "`", "``", "^", "#"}:
                g = extractTerm(g[1:])
                yield from dispatch_call(op, g, goals)
            else:
                for newgoals in unfold(g, goals):
                    yield from step(newgoals)
                    undo()

    goals0 = activate(goals0, dict())
    yield from step(goals0)


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
        goals = next(mparse(quest, ground=False, rule=False))
        print('<<<<<<', goals)
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
    # print(n)
    n.query("perm (1 (2 (3 ())))  X ?")

    n = MinLog(file_name="../natlog/natprogs/py_call.nat")
    # print(n)
    n.query("goal X?")

    n = MinLog(file_name="../natlog/natprogs/family.nat")
    # print(n)
    n.query("cousin of X C, male C?")
    n.repl()

    n.repl()


if __name__ == "__main__":
    test_minlog()
