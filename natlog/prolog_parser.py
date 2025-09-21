import re
import ast
from natlog.scanner import VarNum

# --- VarNum and symbol table ---


def add_sym(syms: dict, nums: list, w: str):
    """
    Ensure variable name `w` has a stable index in this clause.
    NOTE: Anonymous '_' is handled separately so each occurrence is fresh.
    """
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
    # Multi-char operators before single-char ones; includes \=, backticks, &, @, ^ ~
    ("SYMBOLATOM", r"<=|>=|//|==|\\=|->|``|~|`|\\\^|\$|#|@|&|%|;|\+|\-|\*|/|=|<|>|!|~"),
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


def tokenize(code: str):
    for mo in token_compiled.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        if kind in ("SKIP", "COMMENT", "BLOCKCOMMENT"):
            continue
        elif kind == "SQUOTEATOM":
            # decode quoted atom (handles escapes) safely
            yield ("ATOM", ast.literal_eval(value))
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


# --- Operator sets ---

# Minimal infix support (left-associative chaining at a single precedence)
infix_operators = {">=", "<=", "==", ">", "<", "=", "\\=", "+", "-", "*", "/", "//"}

# Prefix (fx) operators (apply to the following term): #, `, ``, &, @, ^
prefix_operators = {"#", "`", "``", "&", "@", "^", "~"}

# --- Parser ---


class Parser:
    def __init__(self, tokens, syms=None, nums=None):
        self.tokens = list(tokens)
        self.pos = 0
        # syms: name -> index (only for named variables; NOT for '_' which is always fresh)
        self.syms = syms if syms is not None else {}
        # nums: allocation order of variables (includes each '_' occurrence)
        self.nums = nums if nums is not None else []
        # next unique variable index within this clause/goal
        self._next_var_index = 0

    def current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", "")

    def match(self, kind):
        if self.current()[0] == kind:
            val = self.current()[1]
            self.pos += 1
            return val
        raise SyntaxError(f"Expected {kind} but got {self.current()}")

    # Allocate/retrieve VarNum for a variable name
    def _var_for(self, name: str) -> VarNum:
        """
        - '_' (anonymous): always allocate a fresh VarNum and record '_' in nums.
        - Named vars (incl. those starting with '_', e.g. '_X'): reuse same VarNum on repeats.
        """
        if name == "_":
            vid = self._next_var_index
            self._next_var_index += 1
            self.nums.append("_")
            return VarNum(vid)
        if name in self.syms:
            return VarNum(self.syms[name])
        vid = self._next_var_index
        self._next_var_index += 1
        self.syms[name] = vid
        self.nums.append(name)
        return VarNum(vid)

    def parse_term(self):
        """
        term := (prefix_op)* atomic_term (infix_op (prefix_op)* atomic_term)*
        Prefix operators (#, `, ``, &, @, ^, ~) are applied as fx:
          - If operand is a tuple ('f', a, b) => ('#', 'f', a, b)
          - Else => ('#', operand)
        Multiple prefixes stack: "# @ f(X)" => ('#', '@', 'f', X)
        """
        left = self._parse_prefix_then_atomic()
        # minimal infix: A OP B -> (OP, A, B)
        while self.current()[0] == "ATOM" and self.current()[1] in infix_operators:
            op = self.match("ATOM")
            right = self._parse_prefix_then_atomic()
            left = (op, left, right)
        return left

    def _parse_prefix_then_atomic(self):
        # collect zero or more prefix operators
        ops = []
        while self.current()[0] == "ATOM" and self.current()[1] in prefix_operators:
            ops.append(self.match("ATOM"))
        # parse the atomic term they apply to
        node = self._parse_atomic_term()
        # apply prefixes inside-out: last prefix read applies outermost
        for op in reversed(ops):
            if isinstance(node, tuple):
                node = tuple([op] + list(node))
            else:
                node = (op, node)
        return node

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
            return self._var_for(val)
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
        # [e1, e2, ..., en | tail]  OR  [e1, e2, ..., en]
        self.match("LBRACK")
        if self.current()[0] == "RBRACK":
            self.match("RBRACK")
            return ()  # []

        items = [self.parse_term()]
        while self.current()[0] == "COMMA":
            self.match("COMMA")
            items.append(self.parse_term())

        if self.current()[0] == "BAR":
            self.match("BAR")
            tail = self.parse_term()
        else:
            tail = ()  # proper list

        self.match("RBRACK")
        return self.to_cons_with_tail(items, tail)

    def to_cons(self, lst):
        return self.to_cons_with_tail(lst, ())

    def to_cons_with_tail(self, items, tail):
        """Build nested cons cells ending in 'tail' (support improper lists)."""
        result = tail
        for h in reversed(items):
            result = (h, result)
        return result

    def parse_clause(self):
        head = self.parse_head_terms()
        if self.current()[0] == "COLONMINUS":
            self.match("COLONMINUS")
            body = self.parse_body()
            self.match("DOT")
            return (head, body)
        else:
            self.match("DOT")
            return (head, [])

    def parse_head_terms(self):
        head = self.parse_term()
        return [head if isinstance(head, tuple) else (head,)]

    def parse_body(self):
        goals = [self.parse_term()]
        while self.current()[0] == "COMMA":
            self.match("COMMA")
            goals.append(self.parse_term())
        # body is always a list of 1-tuples
        return [(g if isinstance(g, tuple) else (g,)) for g in goals]


# --- Public API ---


def parse_prolog_clause(text, syms=None, nums=None):
    tokens = tokenize(text)
    parser = Parser(
        tokens, syms if syms is not None else {}, nums if nums is not None else []
    )
    return parser.parse_clause()


def parse_prolog_program(text):
    tokens = list(tokenize(text))
    pos = 0
    clauses = []
    while pos < len(tokens):
        # variables are local to each clause; fresh maps/counter
        syms, nums = {}, []
        parser = Parser(tokens[pos:], syms, nums)
        clause = parser.parse_clause()
        clauses.append((clause, syms.copy(), nums[:]))
        pos += parser.pos
    return clauses


def parse_prolog_file(filename):
    with open(filename, "r") as f:
        return parse_prolog_program(f.read())


# --- Convenience APIs returning just the nums list (allocation order of variables) ---


def parse_clause_with_varnames(text):
    """
    Returns: (clause, nums)
      - clause: normalized clause structure
      - nums: list of variable names in allocation order (each '_' occurrence included)
    """
    syms, nums = {}, []
    clause = parse_prolog_clause(text, syms, nums)
    return clause, nums


def parse_program_with_varnames(text):
    """
    Returns a list of (clause, nums) pairs for each clause in the program.
    'nums' is the allocation-order list of variable names for that clause.
    """
    results = []
    for clause, syms, nums in parse_prolog_program(text):
        results.append((clause, nums))
    return results


def parse_goal(text):
    """
    Parse a goal-only body (e.g., 'q(X), r(_), # f(X)') with or without trailing '.'
    Reuses Parser.parse_body().
    Returns: (body_as_list_of_1_tuples, nums)
      - nums is the allocation-order list of variable names (each '_' occurrence included)
    """
    syms, nums = {}, []
    tokens = list(tokenize(text))
    p = Parser(tokens, syms, nums)

    # empty input -> empty goal list
    if p.current()[0] == "EOF":
        return [], []

    # Reuse the body parser
    body = p.parse_body()

    # Optional trailing dot
    if p.current()[0] == "DOT":
        p.match("DOT")

    # Must end at EOF
    if p.current()[0] != "EOF":
        raise SyntaxError(f"Unexpected token after goal: {p.current()}")

    return body, nums


# --- Simple line-editable REPL ---


def repl():
    """
    Type a single Prolog clause per line, ending with a '.', or 'quit.' to exit.
    Uses Python's input() so you can edit the line (readline where available).
    """
    print("Prolog clause REPL. Type a clause ending with '.'  (type 'quit.' to exit)\n")
    while True:
        try:
            line = input("?- ").strip()
        except EOFError:
            print("\nbye.")
            break
        if not line:
            continue
        if line.lower() == "quit.":
            print("bye.")
            break
        if not line.endswith("."):
            print("Please end the clause with a '.'")
            continue
        try:
            clause, nums = parse_clause_with_varnames(line)
            print("Clause:", clause)
            print("Vars (allocation order):", nums)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    repl()
