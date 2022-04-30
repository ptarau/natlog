from minlog import *
from unify import *

my_text = """
    app () Ys Ys. 
    app (X Xs) Ys (X Zs) : 
        app Xs Ys Zs.

    nrev () ().
    nrev (X Xs) Zs : nrev Xs Ys, app Ys (X ()) Zs.

    goal N N :
      `genList N Xs,
      nrev Xs Ys.
    """


def test_generators():
    prog = """
  good 'l'.
  good 'o'.
  goal X : ``iter hello X, good X.
  goal X : ``range 1000 1005 X.
  """
    n = MinLog(text=prog)
    for answer in n.solve("goal R?"):
        print(answer[1])


def test_answer_stream():
    prog = """
  perm () ().
  perm (X Xs) Zs : perm Xs Ys, ins X Ys Zs.

  ins X Xs (X Xs).
  ins X (Y Xs) (Y Ys) : ins X Xs Ys.
  """
    n = MinLog(text=prog)
    for answer in n.solve("perm (a (b (c ()))) P?"):
        print(answer[2])


def yield_test():
    prog = """
    worm : ^o, worm.
  """
    n = MinLog(text=prog)
    for i, answer in enumerate(n.solve("worm ?")):
        print(answer[0], end='')
        if i > 42: break
    print('')


# testing with string text
def t1():
    n = MinLog(text=my_text)
    n.query("nrev  (a (b (c (d ())))) R ?")
    n.query("goal 10 L?")


# testing with some .nat files

def t2():
    n = MinLog(file_name="../natprogs/tc.nat")
    print(n)
    n.query("tc Who is animal ?")
    # n.query("tc Who is What ?")


def t4():
    n = MinLog(file_name="../natprogs/perm.nat")
    n.query("perm (1 (2 (3 ()))) Ps?")


def t3():
    n = MinLog(file_name="../natprogs/arith.nat")
    print(n)
    n.query("goal R ?")


# longer output: 8 queens
def t5():
    n = MinLog(file_name="../natprogs/queens.nat")
    print(n)
    n.query("goal8 Queens?")


def t6():
    n = MinLog(file_name="../natprogs/family.nat")
    print(n)
    n.query("grand_parent_of 'Joe' GP ?")


def t7():
    n = MinLog(file_name="../natprogs/family.nat")
    n.query("cousin of X B?")


def t8():
    n = MinLog(file_name="../natprogs/family.nat")
    print('Enter some queries!')
    n.repl()


def loop():
    n = MinLog(file_name="../natprogs/loop.nat")
    print(n)
    n.query("goal X?")


def db_test():
    nd = MinLog(
        file_name="../natprogs/dbtc.nat",
        db_name="../natprogs/Db.nat")
    print('RULES')
    print(nd)
    print('DB FACTS')
    print(nd.db)
    print('QUERY:')
    nd.query("tc Who is_a animal ?")
    nd.repl()


def ndb_test():
    nd = NeuralMinLog(file_name="../natprogs/dbtc.nat", db_name="../natprogs/Db.nat")
    print('RULES')
    print(nd)
    print('DB FACTS')
    print(nd.db)
    print('QUERY:')
    nd.query("tc Who is_a animal ?")


def db_chem():
    nd = MinLog(
        file_name="../natprogs/elements.nat",
        db_name="../natprogs/elements.tsv"
    )
    print('RULES')
    print(nd)
    print('DB FACTS')
    print(nd.db)
    nd.query("an_el Num Element ?")
    nd.query("gases Num Element ?")


def ndb_chem():
    nd = NeuralMinLog(
        file_name="../natprogs/elements.nat",
        db_name="../natprogs/elements.tsv"
    )
    print('RULES')
    print(nd)
    print('DB FACTS')
    print(nd.db)
    nd.query("gases Num Element ?")


def py_test():
    nd = MinLog(file_name="../natprogs/py_call.nat")
    print('RULES')
    # print(nd)
    nd.query("goal X?")


def py_test1():
    nd = MinLog(file_name="../natprogs/py_call1.nat")
    print('RULES')
    # print(nd)
    nd.query("goal X?")


def go():
    t1()
    t2()
    t3()
    t4()
    t5()
    t6()


# tests

c1 = ('a', 1, 'car', 'a')
c2 = ('a', 2, 'horse', 'aa')
c3 = ('b', 1, 'horse', 'b')
c4 = ('b', 2, 'car', 'bb')

g1 = ('a', 0, 1, 2)
g2 = (0, 1, 'car', 2)
g3 = (0, 1, 2, 0)


def dtest1():
    print(c1, '<-const:', list(const_of(c1)))
    print(c3, '<-vars:', list(vars_of(c3)))
    d = Db()
    for cs in [c1, c2, c3, c4]:
        d.add_clause(cs)
    print('index', d.index)
    print('css', d.css)
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
    fname = 'natprogs/Db.nat'
    d = Db()
    d.load(fname)
    print(d)
    print('LOADED:', fname)
    d.ask("Who is mammal?")


# Db from a json file
def dtestj():
    fname = 'natprogs/Db'
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


if __name__ == "__main__":
    """
    uncomment any
    db_test()
    py_test()
    test_generators()
    test_answer_stream()
    yield_test()
    dtestj()
    t5()
    t7()
    ndb_test()  # tests transitive closure with learner
    
    ndb_chem()  # tests query about chemical elements
    """
    pass
    db_test()
    # t7()
    ##t8()
    go()
    py_test()
    # t3()
