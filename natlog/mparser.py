
from operator import *

from .mscanner import Scanner

trace = 0


# simple LL(1) recursive descent Parser
# supporting parenthesized tuples
# scanned from whitespace separated tokens
class Parser:
    def __init__(self, words):
        words = list(reversed(words))
        self.words = words

    def get(self):
        if self.words:
            w = self.words.pop()
            return w
        else:
            return None

    def peek(self):
        if self.words:
            w = self.words[-1]
            return w
        else:
            return None

    def par(self):
        w = self.get()
        assert w == '('
        return self.pars()

    def pars(self):
        w = self.peek()
        if w == ')':
            self.get()
            return None
        elif w == '(':
            t = self.par()
            ts = self.pars()
            return (t, ts)
        else:
            self.get()
            ts = self.pars()
            return (w, ts)

    def run(self):
        ls = sum(1 for x in self.words if x == '(')
        rs = sum(1 for x in self.words if x == ')')
        assert ls == rs

        t = to_tuple(self.par())
        if trace: print("PARSED", t)
        return t


# extracts a Prolog-like clause made of tuples
def to_clause(xs):
    if ':' not in xs: return (xs, ())
    neck = xs.index(':')
    head = xs[:neck]
    body = xs[neck + 1:]
    if ',' not in xs: return (head, (body,))
    bss = []
    bs = []
    for b in body:
        if b == ',':
            bss.append(tuple(bs))
            bs = []
        else:
            bs.append(b)
    bss.append(tuple(bs))
    return (head, tuple(bss))


# main exported Parser + Scanner
def parse(text, ground=False, rule=False):
    s = Scanner(text, ground=ground)
    for ws in s.run():
        if not rule: ws = ('head_', ':') + ws
        ws = ("(",) + ws + (")",)
        p = Parser(ws)
        r = p.run()
        r = to_clause(r)
        if not rule: r = to_goal(r[1])
        if not rule and ground: r = (r[0],)  # db fact
        yield r, s.names


def mparse(text, ground=False, rule=False):
    for r, ixs in parse(text, ground=ground, rule=rule):
        yield r


# turns cons-like tuples into long tuples
# do not change, deep recursion needed
def to_tuple(xy):
    if xy is None:
        return ()
    elif not isinstance(xy, tuple):
        return xy
    else:
        x, y = xy
        t = to_tuple(x)
        ts = to_tuple(y)
        return (t,) + ts

def to_goal(ts):
    gs = ()
    for g in reversed(ts):
        gs = (g, gs)
    return gs


def from_goal(xs) :
    rs=[]
    while xs:
        x,xs=xs
        rs.append(x)
    return tuple(rs)


def numlist(n, m):
    return to_goal(range(n, m))


# tests

def ptest():
    text = """
       app () Ys Ys. 
       app (X Xs) Ys (X Zs) : 
           app Xs Ys Zs.

       nrev () ().
       nrev (X Xs) Zs : nrev Xs Ys, app Ys (X) Zs.
       """
    for c in mparse(text, ground=True):
        print(c)
    print('')
    for c in mparse(text, ground=False, rule=True):
        print(c)
    print('')
    ptest1()


def ptest1():
    xs = ('a', 0, 1, 2, ':', 'b', 0, ',', 'c', 0, 1, ',', 'd', 1, 2)
    print(to_clause(xs))


def ptest2():
    ws = "( x y ( a ( b ( c 1 2 ) ) d ) ( xx yy ) )".split()
    ws = "( 1 ( 2 3 4 ) 5 6 )".split()

    p = Parser(ws)
    print('WS:',ws)
    r = p.par()
    print('R:',r)
    t=to_tuple(r)
    print('T:',t)
    print('WR:',p.words)
    print('RES:',Parser(ws).run())


if __name__ == '__main__':
    ptest2()
