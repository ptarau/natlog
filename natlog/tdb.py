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

    def unify_with_fact(self, h, trail):
        _txt, key, val = h
        ms = self.ground_match_of(key)
        for i in ms:
            h0 = self.css[i]
            u = unify(val, h0[1], trail)
            yield u
