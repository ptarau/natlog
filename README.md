## A lightweight Prolog-like system designed to smoothly interoperate with deep learning tools

We closely follow Einstein's *"Everything should be made as simple as possible, but no simpler."*

At this point, we rely on Python's natural error checking, without doing much to warn about syntactic or semantic errors. This can be added, but this is meant as an *executable specification* of an otherwise simple and natural logic language that we hereby name **Natlog**.


#### NEW: 
at https://github.com/ptarau/natlog/tree/main/natlog/doc there are a few papers explaining Natlog and its applications (including Natlog chatting with GPT3)


#### DEMOS:

- demo apps in folder ```apps/natgpt``` showing Natlog's chats with GPT3 and DALL.E
- demo apps in folder ```apps/deepnat``` interfacing Natlog with JAX and TORCH based neural nets
- a demo app in folder ```apps/nat3d``` combining Natlog and vpython to build (quite easily!) 3D objects and animations



###  **Natlog** : a succinct overview

* Terms are represented as nested tuples.

* A parser and scanner for a simplified Prolog term syntax is used
to turn terms into nested Python tuples.

Surface syntax of facts, as read from strings, is just whitespace separated words 
(with tuples parenthesized) and
sentences ended with ```.``` or ```?```.
Like in Prolog, variables are capitalized, unless quoted. Example programs are in folder ```natprogs```, for instance ```tc.nat```:

```
cat is feline.
tiger is feline.
mouse is rodent.
feline is mammal.
rodent is mammal.
snake is reptile.
mammal is animal.
reptile is animal.

tc A Rel B : A Rel B.
tc A Rel C : A Rel B, tc B Rel C.
```

After 

```pip3 install -U natlog```



To query it, try:

``` python3 -i

>>> from natlog import Natlog, natprogs
>>> n=Natlog(file_name=natprogs()+"tc.nat")
>>> n.query("tc Who is animal ?")
```

It will return answers based on the the transitive closure of the ```is``` relation.

```
QUERY: tc Who is animal ?
ANSWER: {'Who': 'cat'}
ANSWER: {'Who': 'tiger'}
ANSWER: {'Who': 'mouse'}
ANSWER: {'Who': 'feline'}
ANSWER: {'Who': 'rodent'}
ANSWER: {'Who': 'snake'}
ANSWER: {'Who': 'mammal'}
ANSWER: {'Who': 'reptile'}
```

If you are in the folder where the file `tc.nat` is located you could also say

```
python3 -m natlog tc.nat
```
and then once the interactive REPL starts type:

```
?- tc Who is animal ?
```
resulting in the same output, with a chance to enter more queries.

List processing is also supported as in:

```
app () Ys Ys. 
app (X Xs) Ys (X Zs) : app Xs Ys Zs.
```

The interpreter supports a ```yield``` mechanism, similar to Python's own. Something like 
``` ^ my_answer X ```
resulting in my_answer X to be yield as an answer.

The interpreter has also been extended to handle simple function and generator calls to Python  using the same prefix operator syntax:

- ``` `f A B .. Z  R```, resulting in Python function ```f(A,B,,..,Z)``` being called and ```R``` unified with its result
- ``` ``f A B .. Z  R```, resulting in Python generator ```f(A,B,..,Z)``` being called and ```R``` unified with its multiple yields, one a time
- ``` ~R A B .. Z ``` for unifying  ``` ~ R A B .. Z ``` with matching facts in the term store
- ``` # f A B .. Z```, resulting in ```f(A,B,C,..,Z)``` being called with no result returned
- ``` $ V X```, resulting in value of variable named V being unified with X
- ``` eng X G E```,resulting in first class natlog engine with answer pattern X and goal G being bound to E
- ``` ask G A```, resulting in next answer of engine E being unified to A

Take a look at ```natprogs/lib.nat``` for examples of built-ins obtained by extending this interface, mostly at source level.

Take a look at ```natprogs/emu.nat``` for emulation of built-ins in terms of First Class Logic Engines.

### A nested tuple store for unification-based tuple mining

An indexer in combination with the unification algorithm is used to retrieve ground terms matching terms containing logic variables.

Indexing is on all constants occurring in 
ground facts placed in a database. 

As facts are ground,
unification has occurs check and trailing turned off when searching
for a match.

To try it out, do:

```python3 -i ```

```
>>> from natlog.test.tests import *
>>> dtest()

```

It gives, after digesting a text and then querying it:

```
QUERY: Who has (a What)?
--> ('John', 'has', ('a', 'car'))
--> ('Mary', 'has', ('a', 'bike'))

QUERY: Who is (a pilot)?
--> ('John', 'is', ('a', 'pilot'))

QUERY: 'Mary' is What?
--> ('Mary', 'is', ('a', 'student'))

QUERY: 'John' is (a What)?
--> ('John', 'is', ('a', 'pilot'))

QUERY: Who is What?
--> ('Mary', 'is', ('a', 'student'))
--> ('John', 'is', ('a', 'pilot'))
```

### Neuro-symbolic tuple database

As an extension to the nested tuple store the neuro-symbolic tuple database uses a machine learning algorithm instead of its indexer.Thus it offers the same interface as the tuple store that it extends. The learner is trained upon loading the database file (from a .nat,  .csv or .tsv file) and its inference mechanism is triggered when facts from the database are queried. The stream of tuples returned from the query is then filtered via unification (and possibly, more general integrity constraints, expressed via logic programming constructs).

#### Example of usage (see more at https://github.com/ptarau/minlog/blob/main/natlog/test/tests.py )
```
def ndb_test() :
  nd = neural_natlog(file_name=natprogs()+"dbtc.nat",db_name=natprogs()+"db.nat")
  print('RULES')
  print(nd)
  print('DB FACTS')
  print(nd.db)
  nd.query("tc Who is_a animal ?")
```
The output will show the ```X``` and ```y``` numpy arrays used to fit the sklearn learner and then the logic program's rules and the facts from which the arrays were extracted when the facts were loaded.

```
X:
 [[1 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 1]]

y:
 [[1 0 1 0 0 0 0 0 0 0]
 [1 1 1 1 1 1 1 1 1 1]
 [1 0 0 0 0 0 0 0 0 0]
 [0 1 0 1 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 1 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 1 0 1 0 0 0]
 [0 0 0 0 0 1 1 0 0 1]
 [0 0 0 0 0 0 0 1 1 1]
 [0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 1 0]] 

RULES
(('cat', 'is_a', 'feline'), ())
 ((_0, 'is_a', _1), (('~', _0, 'is', _1),))
 (('tc', _0, _1, _2), ((_0, _1, _3), ('tc1', _3, _1, _2)))
 (('tc1', _0, _1, _0), ())
 (('tc1', _0, _1, _2), (('tc', _0, _1, _2),))

DB FACTS
(0, ('tiger', 'is', 'feline'))
(1, ('mouse', 'is', 'rodent'))
(2, ('feline', 'is', 'mammal'))
(3, ('rodent', 'is', 'mammal'))
(4, ('snake', 'is', 'reptile'))
(5, ('mammal', 'is', 'animal'))
(6, ('reptile', 'is', 'animal'))
(7, ('bee', 'is', 'insect'))
(8, ('ant', 'is', 'insect'))
(9, ('insect', 'is', 'animal'))

QUERY: tc Who is_a animal ?
ANSWER: {'Who': 'cat'}
ANSWER: {'Who': 'tiger'}
ANSWER: {'Who': 'mouse'}
ANSWER: {'Who': 'feline'}
ANSWER: {'Who': 'rodent'}
ANSWER: {'Who': 'snake'}
ANSWER: {'Who': 'mammal'}
ANSWER: {'Who': 'reptile'}
ANSWER: {'Who': 'bee'}
ANSWER: {'Who': 'ant'}
ANSWER: {'Who': 'insect'}


```
