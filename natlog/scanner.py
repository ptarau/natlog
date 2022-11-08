import re


# wrapper around int to be used in data fields
# as actual int works as the type of variables
# for simplicity and efficiency


class VarNum(int):
    """
    def __hash__(self):
        return super().__hash__()

    def __eq__(self,other):
       return isinstance(other,VarNum) and int(self)==int(other)
    """

    def __repr__(self):
        return "_" + str(int(self))


class Var:
    def __init__(self):
        self.val = None

    def bind(self, val, trail):
        if self==val: return
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


class GVar(Var):
    def __repr__(self):
        v = deref(self)
        if isinstance(v, GVar) and v.val is None:
            return "&" + str(id(v))
        else:
            return repr(v)


def qtrim(s):
    return s[1:-1]


class Scanner:
    def __init__(self, text, gsyms=dict(), gixs=dict(), ground=True):
        self.text = text
        self.varcount = 0
        self.initsyms()
        self.ground = ground
        self.gsyms = gsyms
        self.gixs = gixs
        self.Scanner = re.Scanner([
            (r"[-+]?\d+\.\d+", lambda sc, tok: ("FLOAT", float(tok))),
            (r"[-+]?\d+", lambda sc, tok: ("INT", int(tok))),
            (r"[a-z]+[\w]*", lambda sc, tok: ("ID", tok)),
            (r"'[\w\s\-\.\/,%=!\+\(\)]+'", lambda sc, tok: ("ID", qtrim(tok))),
            (r"[_]+[\w]*", lambda sc, tok: ("VAR", self.sym(tok + self.ctr()))),
            (r"[A-Z_]+[\w]*", lambda sc, tok: ("VAR", self.sym(tok))),
            (r"[\&]+[\w]*", lambda sc, tok: ("GVAR", self.gsym(tok))),
            (r"[(]", lambda sc, tok: ("LPAR", tok)),
            (r"[)]", lambda sc, tok: ("RPAR", tok)),
            (r"[\[]", lambda sc, tok: ("LPAR_", tok)),
            (r"[\]]", lambda sc, tok: ("RPAR_", tok)),
            (r"[.?]", lambda sc, tok: ("END", self.newsyms())),
            (r":", lambda sc, tok: ("IF", tok)),
            (r"=>", lambda sc, tok: ("REW", tok)),
            (r"[,]", lambda sc, tok: ("AND", tok)),
            (r"~|``|`|\^|\$|#|@|%|;|<=|>=|//|==|\->|\+|\-|\*|/|=|<|>|!", lambda sc, tok: ("OP", tok)),
            #     (r"[;]", lambda sc, tok: ("OR", tok)),
            (r"\s+", None),  # None == skip tok.
        ])

    def ctr(self):
        s = str(self.varcount)
        self.varcount += 1
        return s

    def initsyms(self):
        self.syms = dict()
        self.ixs = []

    def newsyms(self):
        self.names = self.ixs
        self.initsyms()
        return "."

    def sym(self, w):
        if self.ground: return w
        i = self.syms.get(w)
        if i is None:
            i = len(self.syms)
            self.syms[w] = i
            self.ixs.append(w)
        return VarNum(i)

    def gsym(self, w):
        if self.ground: return w
        v = self.gsyms.get(w)
        if v is None:
            v = GVar()
            self.gsyms[w] = v
            self.gixs[v] = w

        return v

    def run(self):
        toks, _ = self.Scanner.scan(self.text)
        ts = []
        for (_, x) in toks:
            if x == '.':
                yield tuple(ts)
                ts = []
            else:
                ts.append(x)


# tests

def stest():
    sent = \
        "(The ~ cat -42) (~ 'sits on' [the mat 0.42]). \n the ` Dog _barks . (` a `` b) and (`b `a) ."
    s = Scanner(sent, ground=False)
    print(list(s.run()))


def gtest():
    sent = """
sent  => a,noun,verb, @ on a,place.

noun => @ cat.
noun => @ dog.

verb => @sits.

place => @ mat.
place => @ bed.
"""
    s = Scanner(sent, ground=False)
    print(list(s.run()))


def ivtest():
    sent = """
node 1 &C1.
node 2 &C2.

edge &C1 &C2.
"""
    s = Scanner(sent, ground=False)
    print(list(s.run()))
    print('GVAR names:', s.gixs)


if __name__ == '__main__':
    # stest()
    # gtest()
    ivtest()
