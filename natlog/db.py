from collections import defaultdict
import json
import csv

from .unify import unify, activate
from .parser import mparse
from .scanner import Var


def path_of(t):
    def path_of0(t):
        if isinstance(t, Var):
            pass
        elif isinstance(t, tuple):
            for i, x in enumerate(t):
                for c, ps in path_of0(x):
                    yield c, (i, ps)
        else:
            yield t, ()

    ps = set(path_of0(t))
    qs = set((c, list2tuple(x)) for (c, x) in ps)
    return qs


def list2tuple(ls):
    # print('!!! LS=',ls)
    def scan(xs):
        while xs != () and isinstance(xs, tuple):
            x, xs = xs
            yield x

    if not isinstance(ls, tuple):
        return ls
    return tuple(scan(ls))


def make_index():
    return defaultdict(set)


def tuplify(t):
    if isinstance(t, list):
        return tuple(map(tuplify, t))
    if isinstance(t, tuple):
        return tuple(map(tuplify, t))
    else:
        return t


class Db:
    def __init__(self):
        self.index = make_index()  # content --> int index
        self.css = []  # content as ground tuples
        self.index_source = path_of

    # parses text to list of ground tuples
    def digest(self, text):
        for cs in mparse(text, ground=True):
            # print('DIGEST:', cs)
            assert len(cs) == 1
            self.add_clause(cs[0])

    # loads from json list of lists
    def load_json(self, fname):
        with open(fname, 'r') as f:
            ts = json.load(f)
        for t in ts:
            self.add_db_clause(t)

    def load_csv(self, fname, delimiter=','):
        with open(fname) as f:
            wss = csv.reader(f, delimiter=delimiter)
            for ws in wss:
                self.add_db_clause(ws)

    def load_tsv(self, fname):
        self.load_csv(fname, delimiter='\t')

    def load_txt(self, fname):
        """ assuming text tokenized, one sentence per line,
         single white space separated, ending with '.' or '?'
        """
        with open(fname) as f:
            lines = f.read().split('\n')
            for line in lines:
                if len(line) < 2: continue
                line = line.strip()
                assert line[-1] in ".?"
                line = line[0:-1]
                line = line.strip()
                ws = line.split(' ')
                self.add_clause(('txt', tuple(ws),))

    def add_db_clause(self, t):
        # print('####', t)
        if t: self.add_clause(tuplify(t))

    # loads ground facts .nat or .json files
    def load(self, fname):
        if len(fname) > 4 and fname[-4:] == '.nat':
            with open(fname, 'r') as f:
                self.digest(f.read())
        elif len(fname) > 4 and fname[-4:] == '.tsv':
            self.load_tsv(fname)
        elif len(fname) > 4 and fname[-4:] == '.csv':
            self.load_csv(fname)
        elif len(fname) > 4 and fname[-4:] == '.txt':
            self.load_txt(fname)
        else:
            self.load_json(fname)

    def save(self, fname):
        with open(fname, "w") as g:
            json.dump(self.css, g)

    def size(self):
        return len(self.css)

    # adds a clause and indexes it for all constants
    # recursively occurring in it, in any subtuple

    def add_clause(self, cs):
        # add_clause_by_content(self.index, self.css, cs)

        i = len(self.css)
        self.css.append(cs)
        for c in self.index_source(cs):
            self.index[c].add(i)

    def ground_match_of(self, query):
        """
        computes all ground matches of a query term in the Db;
        if a constant occurs in the query, it must also occur in
        a ground term that unifies with it, as the ground term
        has no variables that would match the constant
        """
        # find all paths in query
        paths = self.index_source(query)
        if not paths:
            # match against all clauses css, no help from indexing
            return set(range(len(self.css)))
        # pick a copy of the first set where c occurs
        first_path = next(iter(paths))
        matches = self.index[first_path].copy()
        # shrink it by intersecting with sets  where other paths occur
        for x in paths:
            matches &= self.index[x]
        # these are all possible ground matches - return them
        return matches

    # uses unification to match ground fact
    # with bindining applied to vs and collected on trail
    def unify_with_fact(self, h, trail):
        ms = self.ground_match_of(h)
        for i in ms:
            h0 = self.css[i]
            u = unify(h, h0, trail)
            yield u

    # uses unification to match and return ground fact
    def match_of_(self, h):
        h = activate(h, dict())
        for ok in self.unify_with_fact(h, []):
            if ok: yield h

    def match_of(self, hx):
        h = activate(hx, dict())
        ms = self.ground_match_of(h)
        for i in ms:
            h0 = self.css[i]
            trail = []
            if unify(h, h0, trail):
                yield h0
            for v in trail: v.unbind()

    def search(self, query):
        """
        searches for a matching tuple
        """
        qss = mparse(query, ground=False)
        for qs in qss:
            qs = qs[0]
            # print('SEARCHING:', qs)
            for rs in self.match_of(qs):
                yield rs

    # simple search based on content
    def about(self, c):
        for k, v in self.index.items():
            if k[0] == c:
                for i in v:
                    yield self.css[i]

    def ask_about(self, query):
        print('QUERY:', query)
        for r in self.about(query):
            print('-->', r)
        print('')

    # queries_text the Db directly with a text query
    def ask(self, query):
        print('QUERY:', query)
        for r in self.search(query):
            print('-->', r)
        print('')

    # builds possibly very large string representation
    # of the facts contained in the Db
    def __repr__(self):
        xs = [str(cs) + '\n' for cs in enumerate(self.css)]
        return "".join(xs)


def about_facts():
    prog = """
       quest X Y : ~ (text_term (give X Y)) ?
    """
    db = Db()
    db_name = 'natprogs/facts.nat'
    db.load(db_name)

    print('SIZE:', db.size(), 'LEN:', len(db.css[0]))
    print(42, ':', db.css[42])
    db.ask_about("subgraph")


def test_db():
    pass
    about_facts()


if __name__ == "__main__":
    test_db()
