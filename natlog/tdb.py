from .db import *


def const_of(t):
    def scan(t):
        if isinstance(t, Var):
            pass
        elif isinstance(t, tuple):
            for x in t:
                for c in scan(x):
                    yield c
        else:
            yield t

    qs = set(scan(t))
    return qs


class Tdb(Db):
    """
    specializes to db derived from text
    assumes .txt file with one sentence per line
    ending with '.' or '?' and
    white space separated words
    """

    def __init__(self):
        super().__init__()
        self.index_source = const_of


    def digest(self, text):
        sents=text.split('\n')
        for sent in sents:
            ws=sent.split(' ')
            if len(ws) < 2: continue
            ws=[w for w in ws if w.isalpha()]
            self.add_clause(ws)

    def load_txt(self, fname):
        """ assuming text tokenized, one sentence per line,
         single white space separated
        """
        with open(fname) as f:
            lines = f.read().split('\n')
            for line in lines:
                if len(line) < 2: continue
                line = line.strip()
                ws = line.split(' ')
                ws = [w for w in ws if w.isalpha()]
                self.add_clause(('txt', tuple(ws),))


    def unify_with_fact(self, h, trail):
        _txt, key, val = h
        ms = self.ground_match_of(key)
        for i in ms:
            h0 = self.css[i]
            u = unify(val, h0[1], trail)
            yield u
