from math import *
from pathlib import Path

from .parser import *
from .unify import *  # unify, lazy_unify, activate, extractTerm
from .tools import *
from .db import Db


def my_path():
    return str(Path(__file__).parent) + "/"


def natprogs():
    return my_path() + 'natprogs/'


def to_python(x):
    return x


def from_python(x):
    return x

def stop_engine(g):
    E, e, _, _, _, flag = g
    assert E == '$ENG'
    e.close()
    flag[0] = 2


def undo(trail):
    while trail:
        trail.pop().unbind()


def unfold1(g, gs, h, bs, trail):
    d = dict()
    if not lazy_unify(h, g, trail, d):
        undo(trail)
        return None  # FAILURE

    for b in reversed(bs):
        b = activate(b, d)
        gs = (b, gs)
    return gs  # SUCCESS


def interp(css, goals0, db=None, callables=dict()):
    """
    main interpreter
    """

    def to_callable(name):
        """
          associates string names  to callables
        """
        f = callables.get(name, None)
        if f is not None: return f
        return eval(name)

    def dispatch_call(op, g, goals, trail):
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
            f = to_callable(g[0])
            args = to_python(g[1:])
            f(*args)

        def python_fun(g):
            """
            function call to Python, last arg unified with result
            """
            f = to_callable(g[0])
            g = g[1:]
            v = g[-1]
            args = to_python(g[:-1])
            r = f(*args)
            r = from_python(r)
            if not unify(v, r, trail):
                undo(trail)
            else:
                yield from step(goals)

        def python_var(g):
            """
            query value of Python var last arg unified with result
            """
            r = to_callable(g[0])
            v = g[-1]
            r = from_python(r)
            if not unify(v, r, trail):
                undo(trail)
            else:
                yield from step(goals)

        def eng(xge):
            x0, eg0, e = xge
            occ = occurs(x0, eg0)
            (x, eg) = copy_term((x0, eg0))
            g = (eg, ())
            assert isinstance(e, Var)
            # runner = step(g)
            runner = interp(css, g, db=db)
            flag = [0]
            r = ('$ENG', runner, ('the', x), g, occ, flag)

            next(runner,None) # triggers bug in if_ in lib
            e.bind(r, trail)
            # a = next(runner, None)
            # print('DUMMY:',a, flag)
            yield from step(goals)

        def ask(eng_answer):
            eng0, answer = eng_answer
            fun, e, x, g, occ, flag = eng0
            assert fun == '$ENG'
            a = next(e, None)
            # print('REAL:', a, flag)

            if a is None and occ and isinstance(x[1], Var):
                r = 'no'  # bug when true or eq 1 1 is the goal
            elif flag[0] > 0:
                r = 'no'
                e.close()
            else:
                r = copy_term(x)

            if a is None: flag[0] += 1

            if not unify(answer, r, trail):
                undo(trail)
            else:
                yield from step(goals)

        def gen_call(g):
            """
              unifies with last arg yield from a generator
              and first args, assumed ground, passed to it
            """
            gen = to_callable(g[0])
            g = g[1:]
            v = g[-1]
            args = to_python(g[:-1])
            for r in gen(*args):
                r = from_python(r)

                if unify(v, r, trail):
                    yield from step(goals)
                    undo(trail)

        def if_op(g):
            cond, yes, no = g
            cond = extractTerm(cond)

            if next(step((cond, ())), None) is not None:
                yield from step((yes, goals))
            else:
                yield from step((no, goals))

        if op == 'eng':
            yield from eng(g)
        elif op == 'ask':
            yield from ask(g)
        elif op == 'call':
            yield from step((g[0] + g[1:], goals))
        elif op == 'if':
            yield from if_op(g)

        elif op == '~':  # matches against database of facts
            yield from db_call(g)

        elif op == '^':  # yield g as an answer directly
            yield extractTerm(g)
            yield from step(goals)

        elif op == '`':  # function call, last arg unified
            yield from python_fun(g)
        elif op == "``":  # generator call, last arg unified
            yield from gen_call(g)
        elif op == '#':  # simple call, no return
            python_call(g)
            yield from step(goals)
        else:  # op == '$' find value of variable
            yield from python_var(g)

        undo(trail)

    def step(goals):
        """
        recursive inner function
        """
        trail = []
        if goals == ():
            yield extractTerm(goals0)
            undo(trail)
        else:
            g, goals = goals
            op = g[0] if g else None
            if op in {"call", "~", "`", "``", "^", "#","$",  "if", "eng", "ask"}:
                g = extractTerm(g[1:])
                yield from dispatch_call(op, g, goals, trail)
            else:
                for (h, bs) in css:
                    bsgs = unfold1(g, goals, h, bs, trail)
                    if bsgs is not None:
                        yield from step(bsgs)
                        undo(trail)

    done = False
    while not done:
        done = True
        for a in step(goals0):
            if a is not None and len(a) >= 2 and a[0] == 'trust':
                newg = a[1:], ()
                goals0 = newg
                done = False
                break
            yield a


LIB = '../natprogs/lib.nat'


class Natlog:
    def __init__(self, text=None, file_name=None, db_name=None, with_lib=None, callables=dict()):
        if file_name:
            with open(file_name, 'r') as f:
                self.text = f.read()
        else:
            self.text = text

        if with_lib:
            with open(with_lib, 'r') as f:
                lib = f.read()
            self.text = self.text + '\n' + lib

        self.text=clean_comments(self.text)

        self.callables = callables

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
        for answer in interp(self.css, goals0, self.db, self.callables):
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
        success = False
        for answer in self.solve(quest):
            success = True
            print('ANSWER:', answer)
        if not success:
            print('No ANSWER!')
        print('')

    def repl(self):
        """
        read-eval-print-loop
        """
        print("Type ENTER to quit.")
        while True:
            q = input('?- ')
            if not q: return
            try:
                self.query(q)
            except Exception as e:
                print('EXCEPTION:', type(e).__name__, e.args)

    # shows tuples of Natlog rule base
    def __repr__(self):
        xs = [str(cs) + '\n' for cs in self.css]
        return " ".join(xs)





# built-ins, callable with ` notation

def numlist(n, m):
    return to_cons_list(range(n, m + 1))


def consult(natfile=natprogs() + 'family.nat'):
    n = Natlog(file_name=natfile, with_lib=natprogs() + 'lib.nat')
    n.repl()


def load(natfile):
    Natlog(file_name=natprogs() + natfile + ".nat").repl()


# tests

def test_natlog():
    n = Natlog(file_name="natprogs/tc.nat")
    print(n)
    n.query("tc Who is animal ?")

    # n = Natlog(file_name="../natprogs/queens.nat")
    # n.query("goal8 Queens?")

    n = Natlog(file_name="natprogs/perm.nat")
    # print(n)
    n.query("perm (1 (2 (3 ())))  X ?")

    n = Natlog(file_name="natprogs/py_call.nat")
    # print(n)
    n.query("goal X?")
    # n.repl()

    n = Natlog(file_name="natprogs/family.nat")
    # print(n)
    n.query("cousin of X C, male C?")
    # n.repl()

    # n = Natlog(file_name="../natprogs/queens.nat")

    # print(n.count("goal8  X ?"))

    n = Natlog(file_name="natprogs/lib.nat")
    print(n)
    n.repl()
