import re
from natlog.scanner import VarNum

trace = 1

# --- VarNum and symbol table ---

"""
class VarNum(int):
    def __repr__(self):
        return "_" + str(int(self))
"""


def add_sym(syms: dict, nums: list, w: str):
    i = syms.get(w)
    if i is None:
        i = len(syms)
        syms[w] = i
        nums.append(w)
    return VarNum(i)


# --- Tokenizer ---

TOKEN_SPEC = [
    ("BLOCKCOMMENT", r"/\*.*?\*/"),
    ("COMMENT", r"%[^\n]*"),
    ("FLOAT", r"\d+\.\d+"),
    ("NUMBER", r"\d+"),
    ("SQUOTEATOM", r"'([^'\\]|\\.)*'"),
    ("SYMBOLATOM", r"~|``|`|\\\^|\$|#|@|%|;|<=|>=|//|==|->|\+|\-|\*|/|=|<|>|!"),
    ("ATOM", r"[a-z][a-zA-Z0-9_]*"),
    ("VAR", r"[A-Z_][a-zA-Z0-9_]*"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("BAR", r"\|"),
    ("COMMA", r","),
    ("COLONMINUS", r":-"),
    ("DOT", r"\."),
    ("SKIP", r"[ \t\r\n]+"),
    ("MISMATCH", r"."),
]

token_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC)
token_compiled = re.compile(token_regex, re.DOTALL)


def tokenize(code):
    for mo in token_compiled.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        if kind in ("SKIP", "COMMENT", "BLOCKCOMMENT"):
            continue
        elif kind == "SQUOTEATOM":
            yield ("ATOM", eval(value))
        elif kind == "SYMBOLATOM":
            yield ("ATOM", value)
        elif kind == "FLOAT":
            yield ("FLOAT", float(value))
        elif kind == "NUMBER":
            yield ("NUMBER", int(value))
        elif kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character: {value!r}")
        else:
            yield (kind, value)


# --- Infix operators ---

infix_operators = {">=", "<=", "==", ">", "<", "=", "\\=", "+", "-", "*", "/", "//"}

# --- Parser class ---


class Parser:
    def __init__(self, tokens, syms=None, nums=None):
        self.tokens = list(tokens)
        self.pos = 0
        self.syms = syms if syms is not None else {}
        self.nums = nums if nums is not None else []

    def current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", "")

    def match(self, kind):
        if self.current()[0] == kind:
            val = self.current()[1]
            self.pos += 1
            return val
        raise SyntaxError(f"Expected {kind} but got {self.current()}")

    def parse_term(self):
        left = self._parse_atomic_term()
        while self.current()[0] == "ATOM" and self.current()[1] in infix_operators:
            op = self.match("ATOM")
            right = self._parse_atomic_term()
            left = (op, left, right)
        return left

    def _parse_atomic_term(self):
        tok, val = self.current()
        if tok == "ATOM":
            self.pos += 1
            if self.current()[0] == "LPAREN":
                self.match("LPAREN")
                args = self.parse_args()
                self.match("RPAREN")
                return tuple([val] + args)
            else:
                return val
        elif tok == "VAR":
            self.pos += 1
            return add_sym(self.syms, self.nums, val)
        elif tok in ("NUMBER", "FLOAT"):
            self.pos += 1
            return val
        elif tok == "LBRACK":
            return self.parse_list()
        else:
            raise SyntaxError(f"Unexpected token: {self.current()}")

    def parse_args(self):
        args = [self.parse_term()]
        while self.current()[0] == "COMMA":
            self.match("COMMA")
            args.append(self.parse_term())
        return args

    def parse_list(self):
        self.match("LBRACK")
        if self.current()[0] == "RBRACK":
            self.match("RBRACK")
            return ()
        head = self.parse_term()
        if self.current()[0] == "BAR":
            self.match("BAR")
            tail = self.parse_term()
            self.match("RBRACK")
            return (head, tail)
        lst = [head]
        while self.current()[0] == "COMMA":
            self.match("COMMA")
            lst.append(self.parse_term())
        self.match("RBRACK")
        return self.to_cons(lst)

    def to_cons(self, lst):
        if not lst:
            return ()
        else:
            return (lst[0], self.to_cons(lst[1:]))

    def parse_clause(self):
        head = self.parse_term()
        if not isinstance(head, tuple):
            head = (head,)
        if self.current()[0] == "COLONMINUS":
            self.match("COLONMINUS")
            body = self.parse_body()
            self.match("DOT")
            return (head, body)
        else:
            self.match("DOT")
            return (head, ())

    def parse_body(self):
        goals = [self.parse_term()]
        while self.current()[0] == "COMMA":
            self.match("COMMA")
            g = self.parse_term()
            goals.append(g)

        goals = [(g if isinstance(g, tuple) else (g,)) for g in goals]
        return tuple(goals)  # if len(goals) > 1 else goals[0]


# --- Public API ---


def parse_prolog_clause(text, syms=None, nums=None):
    tokens = tokenize(text)
    parser = Parser(tokens, syms, nums)
    return parser.parse_clause()


def parse_prolog_program(text):
    tokens = list(tokenize(text))
    pos = 0
    clauses = []
    while pos < len(tokens):
        syms, nums = {}, []
        parser = Parser(tokens[pos:], syms, nums)
        clause = parser.parse_clause()
        clauses.append(clause)  # , syms.copy(), nums[:]))
        pos += parser.pos
    if trace:
        print("Parsed clauses:")
        for c in clauses:
            print(c)
    return tuple(clauses)


def parse_prolog_file(filename):
    with open(filename, "r") as f:
        return parse_prolog_program(f.read())


# TESTS


def test_prolog_parser():
    assert parse_prolog_clause("f(a,b,c).") == (("f", "a", "b", "c"), ())
    assert parse_prolog_clause("p(X):- q(X), r(X).") == (
        ("p", Var("X")),
        ("and", ("q", Var("X")), ("r", Var("X"))),
    )
    assert parse_prolog_clause("'>='(X, 3.14).") == (
        (">=", Var("X"), 3.14),
        (),
    )
    assert parse_prolog_clause("'op'.") == (("op",), ())
    assert parse_prolog_clause("q(X,Y) :- p(X),p(Y).") == (
        ("q", Var("X"), Var("Y")),
        ("and", ("p", Var("X")), ("p", Var("Y"))),
    )
    print(parse_prolog_clause("distance('New York 42', 12.34)."))
    # →   (('distance', 'New York 42', 12.34), ())

    print(parse_prolog_clause("rule(X):- '>='(X,3.14), +('op')."))
    # → (('rule', Var('X')), ('and', ('>=', Var('X'), 3.14), ('+', 'op')))

    program = """
      % example Prolog program
      f(a,b,c).
      p(X):-
       q(X),
       r(X).
     '>='(X, 3.14).
     '+'('op').
    q(X,Y) :- p(X),p(Y).
    """

    for clause in parse_prolog_program(program):
        print(clause)


if __name__ == "__main__":
    test_prolog_parser()
