from math import *
from pathlib import Path

from .parser import *
from .unify import *  # unify, lazy_unify, activate, extractTerm, Var
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


# eng X (between 1 5 X) E,`next E R, #print R, fail?

class Eng:
    def __init__(self, interp, css, g, db, callables):
        self.interp = interp
        self.css = css
        self.db = db
        self.g = g
        self.callables = callables
        self.runner = None
        self.stopped = False

    def start(self):
        if self.runner is None and not self.stopped:
            self.runner = interp(self.css, self.g, db=self.db, callables=self.callables)

    # eng X (between 1 5 X) E, stop E.
    def stop(self):

        if self.runner is not None and not self.stopped:
            self.runner.close()
        self.stopped = True

    def __next__(self):
        if self.stopped: return None
        self.start()
        return next(self.runner)

    def __call__(self):
        self.start()
        if not self.stopped: yield from self.runner

    def __repr__(self):
        mes = ""
        if self.stopped: mes = "stopped_"
        return mes + 'eng_' + str(id(self))


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


nat_builtins = {
    "call", "~", "`", "``", "^", "#", "$",
    "if", "eng", "ask", "unify_with_occurs_check"
}


def interp(css, goals0, db=None, callables=dict()):
    """
    main interpreter
    """

    def to_callable(name):
        """
          associates string names  to callables
        """
        if callable(name): return name
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
            x, g, e = xge
            (x, g) = copy_term((x, g))
            g = (('the', x, g), ())
            assert isinstance(e, Var)
            r = Eng(interp, css, g, db, callables)
            e.bind(r, trail)
            yield from step(goals)

        # eng X (between 1 5 X) E, ask E R, #print R, fail?
        def ask(ex):
            e, x = ex
            a = next(e, None)
            # print('RAW ask next:',a)
            if a is None:
                r = 'no'
                e.stop()
            elif len(a) == 1:  # a ^ operation
                r = ('the', copy_term(a[0]))
            else:
                ((the, r, g), ()) = a
                r = (the, copy_term(r))
            if not unify(x, r, trail):
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

        def unify_with_occurs_check_op(g):
            t1, t2 = g
            if not unify(t1, t2, trail, occ=True):
                undo(trail)
            else:
                yield from step(goals)

        if op == 'eng':
            yield from eng(g)

        elif op == 'ask':
            yield from ask(g)

        elif op == 'call':
            yield from step((g[0] + g[1:], goals))

        elif op == 'if':
            yield from if_op(g)

        elif op == 'unify_with_occurs_check':
            yield from unify_with_occurs_check_op(g)

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
            if op in nat_builtins:
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

        self.callables = callables
        self.gsyms = dict()
        self.gixs = dict()

        css, ixss = zip(*parse(self.text, gsyms=self.gsyms, gixs=self.gixs, ground=False, rule=True))

        self.css = tuple(css)
        self.ixss = tuple(ixss)

        # print('GIXSS in natlog:', self.gixs)

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
        goals0, ixs = next(parse(quest, gsyms=self.gsyms, gixs=self.gixs, ground=False, rule=False))

        vs = dict()
        goals0 = activate(goals0, vs)
        ns = dict(zip(vs, ixs))

        for k, v in self.gixs.items():
            ns[k] = v

        for answer in interp(self.css, goals0, self.db, self.callables):

            if answer and len(answer) == 1:
                sols = {'_': answer[0]}
            else:
                sols = dict((ns[v], deref(r)) for (v, r) in vs.items())
            yield sols

    def count(self, quest):
        """
        answer counter
        """
        c = 0
        for _ in self.solve(quest):
            c += 1
        return floor(c)

    def query(self, quest, in_repl=False):
        """
        show answers for given query
        """
        if not in_repl: print('QUERY:', quest)
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
                self.query(q, in_repl=True)
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


def lconsult(fname):
    fname = natprogs() + fname + ".nat"
    n = Natlog(file_name=fname, with_lib=natprogs() + 'lib.nat')
    n.repl()


def consult(fname):
    fname = natprogs() + fname + ".nat"
    n = Natlog(file_name=fname)
    n.repl()

def dconsult(nname,dname):
    nname = natprogs() + nname + ".nat"
    dname = natprogs() + dname + ".nat"
    n = Natlog(file_name=nname,db_name=dname)
    n.repl()

def tconsult(fname):
    nname = natprogs() + fname + ".nat"
    dname = natprogs() + fname + ".tsv"
    n = Natlog(file_name=nname,db_name=dname)
    n.repl()
