from operator import *

from .scanner import Scanner, VarNum

trace = 0


def rp(LP):
    return ')' if LP == '(' else ']'


def from_none(LP, w):
    if w is None:
        if LP == '(': return ()
        if LP == '[': return []
    return w


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

    def par(self, LP, RP):
        w = self.get()
        assert w == LP
        return self.pars(LP, RP)

    def pars(self, LP, RP):
        w = self.peek()
        if w == RP:
            self.get()
            return from_none(LP, None)
        elif w == LP:
            t = self.par(LP, RP)
            ts = self.pars(LP, RP)
            ts = from_none(LP, ts)
            return (t, ts) if LP == '(' else [t] + ts
        elif w == '(' or w == '[' and w != LP:
            t = self.par(w, rp(w))
            ts = self.pars(LP, RP)
            ts = from_none(LP, ts)
            return (t, ts) if LP == '(' else [t] + ts
        else:
            self.get()
            ts = self.pars(LP, RP)
            ts = from_none(LP, ts)
            return (w, ts) if LP == '(' else [w] + ts

    def run(self):
        ls = sum(1 for x in self.words if x == '(')
        rs = sum(1 for x in self.words if x == ')')
        assert ls == rs
        ls = sum(1 for x in self.words if x == '[')
        rs = sum(1 for x in self.words if x == ']')
        assert ls == rs
        t = self.par('(', ')')
        t = to_tuple(t)
        if trace: print("PARSED", t)
        return t


# extracts a Prolog-like clause made of tuples
def to_clause(xs):
    if not (':' in xs or '=>' in xs): return xs, ()
    if "=>" in xs:
        sep = '=>'
    else:
        sep = ':'
    neck = xs.index(sep)
    head = xs[:neck]
    body = xs[neck + 1:]

    if sep == ':':
        if ',' not in xs:
            res = head, (body,)
        else:
            bss = []
            bs = []
            for b in body:
                if b == ',':
                    bss.append(tuple(bs))
                    bs = []
                else:
                    bs.append(b)
            bss.append(tuple(bs))

            res = head, tuple(bss)
        return res
    if sep == '=>':
        n0 = 100
        n = n0
        if ',' not in xs:
            vs = (VarNum(n), VarNum(n + 1))
            res = head + vs, (body + vs,)
        else:
            bss = []
            bs = []
            for b in body:
                if b == ',':
                    vs = VarNum(n), VarNum(n + 1)
                    n += 1
                    bs = tuple(bs) + vs
                    bss.append(bs)
                    bs = []
                else:
                    bs.append(b)

            vs = VarNum(n), VarNum(n + 1)
            n += 1
            bs = tuple(bs) + vs
            bss.append(bs)
            head = head + (VarNum(n0), VarNum(n))

            res = head, tuple(bss)
        return res


# main exported Parser + Scanner
def parse(text, gsyms=dict(), gixs=dict(), ground=False, rule=False):
    text = clean_comments(text)
    s = Scanner(text, gsyms=gsyms, gixs=gixs, ground=ground)
    for ws in s.run():
        if not rule: ws = ('head_', ':') + ws
        ws = ("(",) + ws + (")",)
        p = Parser(ws)
        r = p.run()
        r = to_clause(r)
        if not rule: r = to_cons_list(r[1])
        if not rule and ground: r = (r[0],)  # db fact

        yield r, s.names


def mparse(text, ground=False, rule=False):
    for r, ixs in parse(text, ground=ground, rule=rule):
        yield r


# turns cons-like tuples into long tuples
# do not change, deep recursion needed
def to_tuple(xy):
    if xy is None or xy == ():
        return ()
    elif isinstance(xy, list):
        return [to_tuple(x) for x in xy]
    elif not isinstance(xy, tuple):
        return xy
    else:  # tuple
        x, y = xy
        t = to_tuple(x)
        ts = to_tuple(y)
        return (t,) + ts

def from_cons_list_as_tuple(xs):
    return tuple(from_cons_list(xs))


def from_cons_list(xs):
    rs = []
    while xs:
        x, xs = xs
        rs.append(x)
    return rs


def to_cons_list(ts):
    gs = ()
    for g in reversed(ts):
        gs = (g, gs)
    return gs


def q(xs):
    rs = []
    while xs:
        x, xs = xs
        rs.append(x)
    return rs


def numlist(n, m):
    return to_cons_list(range(n, m))


def clean_comments(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        parts = line.split("%")
        if len(parts) > 1:
            line = parts[0]
        cleaned.append(line)
    text = "\n".join(cleaned)
    #print('>>> ???',text)
    return text


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
    ws = "( x y [ a ( b [ c 1 2 ] ) d ] ( xx yy ) )".split()
    # ws = "( 1 [ 2 3 4 ] 5 6 )".split()

    p = Parser(ws)
    print('WS:', ws)
    r = p.par('(', ')')
    print('R:', r)
    # return
    t = to_tuple(r)
    print('T:', t)
    print('WR:', p.words)
    print('RES:', Parser(ws).run())


def ptest3():
    text = """
sent  => a,noun,verb, @on, @a, place.

noun => @cat.
noun => @dog.

verb => @sits.

place => @mat.
place => @bed.
    
@ X (X Xs) Xs.

goal Xs : sent Xs ().

"""

    r = parse(text, ground=False, rule=True)
    print(list(r))


def ptest4():
    r = parse('a [].')
    print(list(r))


def clean_test():
    text = """   
    a b c % d e
        mmm nn pp
    xx yyyy % a % b
    
    % zzz zz z   
    more
   
% aaa

boo.

    """
    print(text)
    print('-----')
    print(clean_comments(text))


if __name__ == '__main__':
    ptest4()
    clean_test()
