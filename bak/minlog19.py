from math import *

from mparser import *
from unify import *  # unify, lazy_unify, activate, extractTerm
from db import Db


def to_python(x):
    return x


def from_python(x):
    return x


def const(x):
    assert len(x) <= 2
    return eval(x)


def interp(css, goals0, db=None):
    def undo(trail):
        while trail:
            trail.pop().unbind()

    def unfold1(g, gs, h, bs, trail):
        d = dict()
        if not lazy_unify(h, g, trail, d):
            undo(trail)
            return None  # FAILURE

        # NOT TO BE CHANGED !!!
        bsgs = gs
        for b in reversed(bs):
            b = activate(b, d)
            bsgs = (b, bsgs)
        return bsgs  # SUCCESS

    def step(goals):

        def dispatch_call(op, g, goals):
            """
            dispatches several types of calls to Python
            """

            # yields facts matching g in Db
            def db_call(g):
                for ok in db.unify_with_fact(g, trail):
                    if not ok:  # FAILURE
                        undo(trail)
                        continue
                    yield from step(goals)  # SUCCESS
                    undo(trail)

            def python_call(g):
                """
                simple call to Python (e.g., print, no return expected)
                """
                f = eval(g[0])
                args = to_python(g[1:])
                f(*args)

            def python_fun(g):
                """
                function call to Python, last arg unified with result
                """
                f = eval(g[0])
                g = g[1:]
                v = g[-1]
                args = to_python(g[:-1])
                r = f(*args)
                r = from_python(r)
                if not unify(v, r, trail):
                    undo(trail)
                else:
                    yield from step(goals)

            def eng(eg0):

                x0, eg0, e = extractTerm(eg0)
                (x, eg) = copy_term((x0, eg0))

                runner = step((eg, ()))

                #next(runner)

                r = (x, runner)

                if not unify(e, r, trail):
                    undo(trail)
                else:
                    yield from step(goals)

            def ask(eg):

                eg = extractTerm(eg)

                engine, answer = eg
                x, e = engine
                a = next(e, None)

                if a is None:
                    r = 'no'
                else:
                    x = copy_term(x)
                    r = ('the', x)
                if not unify(answer, r, trail):
                    undo(trail)
                else:
                    yield from step(goals)

            def gen_call(g):
                """
                  unifies with last arg yield from a generator
                  and first args, assumed ground, passed to it
                """
                gen = eval(g[0])
                g = g[1:]
                v = g[-1]
                args = to_python(g[:-1])
                for r in gen(*args):
                    r = from_python(r)
                    if unify(v, r, trail):
                        yield from step(goals)

            def neg(g):
                """
                negation as failure
                """
                no_sol = object()
                # g = extractTerm(g)
                a = next(step((g, ())), no_sol)
                if a is no_sol:
                    return True
                return False

            def if_op(g):
                cond, yes, no = g
                cond = extractTerm(cond)
                if next(step((cond[0], ())), None) is not None:
                    yield from step((yes, goals))
                else:
                    yield from step((no, goals))

            if op == 'eng':
                # print('!!!', eng)
                yield from eng(g)
            elif op == 'ask':
                yield from ask(g)
            elif op == 'call':
                cg = extractTerm(g)
                yield from step((cg[0], goals))
            elif op == 'not':
                if neg(g):
                    yield from step(goals)
            elif op == 'if':
                yield from if_op(g)

            elif op == '~':  # matches against database of facts
                yield from db_call(g)
            elif op == '^':  # yield g as an answer directly
                yield g
                yield from step(goals)
            elif op == '`':  # function call, last arg unified
                yield from python_fun(g)
            elif op == "``":  # generator call, last arg unified
                yield from gen_call(g)
            else:  # op == '#',  simple call, no return
                python_call(g)
                yield from step(goals)
            undo(trail)

        trail = []
        if goals == ():
            yield extractTerm(goals0)
            undo(trail)
        else:
            g, goals = goals
            op = g[0] if g else None
            if op in {"not", "call", "~", "`", "``", "^", "#", "if", "eng", "ask"}:
                g = extractTerm(g[1:])
                yield from dispatch_call(op, g, goals)
            else:
                for (h, bs) in css:
                    bsgs = unfold1(g, goals, h, bs, trail)
                    if bsgs is not None:
                        yield from step(bsgs)
                        undo(trail)

    yield from step(goals0)  # assumed activated


class MinLog:
    def __init__(self, text=None, file_name=None, db_name=None):
        if file_name:
            with open(file_name, 'r') as f:
                self.text = f.read()
        else:
            self.text = text

        css, ixss = zip(*parse(self.text, ground=False, rule=True))

        self.css = tuple(css)
        self.ixss = tuple(ixss)

        if db_name is not None:
            self.db_init()
            self.db.load(db_name)
        else:
            self.db = None

    def db_init(self):
        """
        overridable database initializer
        sets the type of the database (default or neuro-symbolic)
        """
        self.db = Db()

    def solve(self, quest):
        """
         answer generator for given question
        """
        goals0, ixs = next(parse(quest, ground=False, rule=False))
        vs = dict()
        goals0 = activate(goals0, vs)
        ns = dict(zip(vs, ixs))
        for answer in interp(self.css, goals0, self.db):
            if answer and len(answer) == 1:
                sols = {'_': answer[0]}
            else:
                sols = dict((ns[v], r) for (v, r) in vs.items())
            yield sols

    def count(self, quest):
        """
        answer counter
        """
        c = 0
        for _ in self.solve(quest):
            c += 1
        return floor(c)

    def query(self, quest):
        """
        show answers for given query
        """
        print('QUERY:', quest)
        for answer in self.solve(quest):
            print('ANSWER:', answer)
        print('')

    def repl(self):
        """
        read-eval-print-loop
        """
        print("Type ENTER to quit.")
        while True:
            q = input('?- ')
            if not q: return
            self.query(q)

    # shows tuples of Natlog rule base
    def __repr__(self):
        xs = [str(cs) + '\n' for cs in self.css]
        return " ".join(xs)


# built-ins, callable with ` notation

def numlist(n, m):
    return to_goal(range(n, m))


# tests

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
    # n.repl()

    n = MinLog(file_name="../natlog/natprogs/family.nat")
    # print(n)
    n.query("cousin of X C, male C?")
    # n.repl()

    # n = Natlog(file_name="../natprogs/queens.nat")

    # print(n.count("goal8  X ?"))

    n = MinLog(file_name="../natlog/natprogs/lib.nat")
    print(n)
    n.repl()


if __name__ == "__main__":
    # test_minlog()
    n = MinLog(file_name="../natlog/natprogs/lib.nat")
    # print(n)
    n.query('t6?')
    n.query('t5?')
    # n.repl()
