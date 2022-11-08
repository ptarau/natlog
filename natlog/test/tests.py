from natlog.natlog import *
from natlog.unify import *
from natlog.neural_natlog import *

NATPROGS = natprogs()

my_text = """
    app () Ys Ys. 
    app (X Xs) Ys (X Zs) : 
        app Xs Ys Zs.

    nrev () ().
    nrev (X Xs) Zs : nrev Xs Ys, app Ys (X ()) Zs.

    goal N N :
      `numlist 0 N Xs,
      nrev Xs Ys.
    """


def test_generators():
    prog = """
  good 'l'.
  good 'o'.
  goal X : ``iter hello X, good X.
  goal X : ``range 1000 1005 X.
  """
    n = Natlog(text=prog)
    for answer in n.solve("goal R?"):
        print(answer)


def test_answer_stream():
    prog = """
  perm () ().
  perm (X Xs) Zs : perm Xs Ys, ins X Ys Zs.

  ins X Xs (X Xs).
  ins X (Y Xs) (Y Ys) : ins X Xs Ys.
  """
    n = Natlog(text=prog)
    for answer in n.solve("perm (a (b (c ()))) P?"):
        print(answer)


def yield_test():
    prog = """
    worm : ^o, worm.
  """
    n = Natlog(text=prog)
    for i, answer in enumerate(n.solve("worm ?")):
        print(answer[0], end='')
        if i > 42: break
    print('')


# testing with string text
def t1():
    n = Natlog(text=my_text)
    n.query("nrev  (a (b (c (d ())))) R ?")
    n.query("goal 10 L?")


# testing with some .nat files

def t2():
    n = Natlog(file_name=NATPROGS + "tc.nat")
    print(n)
    n.query("tc Who is animal ?")
    # n.query("tc Who is What ?")


def t4():
    n = Natlog(file_name=NATPROGS + "perm.nat")
    n.query("perm (1 (2 (3 ()))) Ps?")


def t3():
    n = Natlog(file_name=NATPROGS + "arith.nat")
    print(n)
    n.query("goal R ?")


# longer output: 8 queens
def t5():
    n = Natlog(file_name=NATPROGS + "queens.nat")
    # print(n)
    n.query("goal8 Queens?")
    # n.repl()


def t6():
    n = Natlog(file_name=NATPROGS + "family.nat", with_lib=NATPROGS + "lib.nat")
    # print(n)
    n.query("grand parent of 'Adam' GP ?")


def t7():
    n = Natlog(file_name=NATPROGS + "family.nat", with_lib=NATPROGS + "lib.nat")
    n.query("cousin of X B?")


def t8():
    n = Natlog(file_name=NATPROGS + "lib.nat")
    n.query('`numlist 1 5 Xs, findall X (member X Xs) Ys.')
    # n.repl()


def t9():
    n = Natlog(file_name=NATPROGS + "pro.nat")
    n.repl()


def t10():
    print('it takes a while to start')
    n = Natlog(file_name=NATPROGS + "sudoku4.nat", with_lib=NATPROGS + "lib.nat")
    # n.repl()
    n.query("goal Xss, nl, member Xs Xss, tuple Xs T, writeln T, fail?")


def fam_repl():
    n = Natlog(file_name=NATPROGS + "family.nat", with_lib=LIB)
    print('Enter some queries!')
    n.repl()


def lib():
    n = Natlog(file_name=NATPROGS + "lib.nat")
    n.repl()


def loop():
    n = Natlog(file_name=NATPROGS + "loop.nat")
    print(n)
    n.query("goal X?")


def db_test():
    nd = Natlog(
        file_name=NATPROGS + "dbtc.nat",
        db_name=NATPROGS + "db.nat")
    print('RULES')
    print(nd)
    print('DB FACTS')
    print(nd.db)
    print('QUERY:')
    nd.query("tc Who is_a animal ?")
    # nd.repl()


def ndb_test():
    nd = NeuralNatlog(file_name=NATPROGS + "dbtc.nat", db_name=NATPROGS + "db.nat")
    print('RULES')
    print(nd)
    print('DB FACTS')
    print(nd.db)
    nd.query("tc Who is_a animal ?")


def db_chem():
    nd = Natlog(
        file_name=NATPROGS + "elements.nat",
        db_name=NATPROGS + "elements.tsv"
    )
    print('RULES')
    print(nd)
    # print('DB FACTS');print(nd.db)
    print('SIZE:', nd.db.size(), 'LEN:', len(nd.db.css[0]))
    nd.query("an_el Num Element ?")
    nd.query("gases Num Element ?")


def ndb_chem():
    nd = NeuralNatlog(
        file_name=NATPROGS + "elements.nat",
        db_name=NATPROGS + "elements.tsv"
    )
    print('RULES')
    print(nd)
    print('DB FACTS')
    print(nd.db)
    nd.query("gases Num Element ?")


def py_test():
    nd = Natlog(file_name=NATPROGS + "py_call.nat")
    print('RULES')
    # print(nd)
    nd.query("goal X?")


def py_test1():
    nd = Natlog(file_name=NATPROGS + "py_call1.nat")
    print('RULES')
    # print(nd)
    nd.query("goal X?")


def dtest1():
    c1 = ('a', 1, 'car', 'a')
    c2 = ('a', 2, 'horse', 'aa')
    c3 = ('b', 1, 'horse', 'b')
    c4 = ('b', 2, 'car', 'bb')

    g1 = ('a', Var(), Var(), Var())
    g2 = (Var(), Var(), 'car', Var())
    g3 = (Var(), Var(), Var(), Var())

    print(c1, '\n<-const:', list(path_of(c1)))
    d = Db()
    for cs in [c1, c2, c3, c4]:
        d.add_clause(cs)
    print('\nindex')
    for xv in d.index.items():
        print(xv)

    print('\ncss')
    for cs in d.css:
        print(cs)
    print('Gmatch', g1, list(d.ground_match_of(g1)))
    print('Vmatch', g1, list(d.match_of(g1)))
    print('Gmatch', g2, list(d.ground_match_of(g2)))
    print('Vmatch', g2, list(d.match_of(g2)))
    print('Gmatch', g3, list(d.ground_match_of(g3)))
    print('Vmatch', g3, list(d.match_of(g3)))


# Db built form text
def dtest():
    text = """
   John has (a car).
   Mary has (a bike).
   Mary is (a student).
   John is (a pilot).
   """
    print(text)
    d = Db()
    d.digest(text)
    print(d)
    print('')
    query = "Who has (a What)?"
    d.ask(query)

    query = "Who is (a pilot)?"
    d.ask(query)

    query = "'Mary' is What?"
    d.ask(query)

    query = "'John' is (a What)?"
    d.ask(query)

    query = "Who is What?"
    d.ask(query)


# Db from a .nat file
def dtestf():
    fname = NATPROGS + 'db.tsv'
    d = Db()
    d.load(fname)
    print(d)
    print('LOADED:', fname)
    d.ask("Who is mammal?")


# Db from a json file
def dtestj():
    fname = NATPROGS + 'db'
    jname = fname + '.json'
    nname = fname + '.nat'
    d = Db()
    d.load(nname)
    d.save(jname)
    d = Db()
    d.load(jname)
    # print(d)
    print('LOADED:', jname)
    print("")
    query = "Who is What?"
    d.ask(query)


def big_db():
    prog = """
       quest X Y : ~ (text_term (give X Y)) ?
    """
    n = Natlog(text=prog, db_name=NATPROGS + 'facts.nat')
    # print(n)
    print('SIZE:', n.db.size(), 'LEN:', len(n.db.css[0]))
    # print(n.db.css[0])
    n.query("quest X Y?")
    # n.repl()


def big_ndb():
    prog = """
       quest X Y : ~ (text_term (give X Y)) ?
    """
    n = NeuralNatlog(text=prog, db_name=NATPROGS + 'facts.nat')
    # print(n)
    print('SIZE:', n.db.size(), 'LEN:', len(n.db.css[0]))
    # print(n.db.css[0])
    n.query("quest X Y?")
    # n.repl()


def libtest():
    n = Natlog(file_name=NATPROGS + 'emu.nat', with_lib=NATPROGS + "lib.nat")
    n.repl()


def gramtest():
    n = Natlog(file_name=NATPROGS + 'dall_e.nat')
    print(n)
    n.query("go.")
    # n.repl()


def meta_test():
    n = Natlog(file_name=NATPROGS + 'meta.nat')
    n.query("metaint ((goal R) ()) ?")
    # n.repl()

def ivtest():
    n = Natlog(file_name=NATPROGS + 'gcol.nat',with_lib=NATPROGS + "lib.nat")
    #print(n)
    n.query("go Colors?")
    #n.repl()


def go():
    ts = [dtest1,
          dtest,
          dtestf,
          test_generators,
          test_answer_stream,
          t1,
          t2,
          t3,
          t4,
          t5,
          t6,
          t7,
          db_test, db_chem, py_test, py_test1,
          big_db,
          gramtest,
          meta_test,
          ivtest
          ]
    for t in ts:
        print('\n\n', '*' * 20, t.__name__, '*' * 20, '\n')
        t()


if __name__ == "__main__":
    go()
    #ndb_test()
    #libtest()
    # gramtest()
    # meta_test()
    #ivtest()
