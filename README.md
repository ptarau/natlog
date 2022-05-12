## A lightweight Prolog-like interpreter with a natural-language style syntax and deeply indexed tuple database interface

We closely follow Einstein's *"Everything should be made as simple as possible, but no simpler."*

At this point, we rely on Python's natural error checking, without doing much to warn about syntactic or semantic errors. This can be added, but this is meant as an *executable specification* of an otherwise simple and natural logic language that we hereby name **Natlog**.

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

To query it, try:

``` python3 -i natlog.py

>>> n=natlog(file_name="natprogs/tc.nat")
>>> n.query("tc Who is animal ?")
```

It will return the transitive closure of the ```is``` relation.

```
GOAL PARSED: (('tc', 0, 'is', 'animal'),)
ANSWER: ('tc', 'cat', 'is', 'animal')
ANSWER: ('tc', 'tiger', 'is', 'animal')
ANSWER: ('tc', 'mouse', 'is', 'animal')
ANSWER: ('tc', 'feline', 'is', 'animal')
ANSWER: ('tc', 'rodent', 'is', 'animal')
ANSWER: ('tc', 'snake', 'is', 'animal')
ANSWER: ('tc', 'mammal', 'is', 'animal')
ANSWER: ('tc', 'reptile', 'is', 'animal')
```

List processing is also supported as in:

```
app () Ys Ys. 
app (X Xs) Ys (X Zs) : app Xs Ys Zs.
```

The interpreter supports a ```yield``` mechanism, similar to Python's own. Something like 
``` ^ my_answer X ```
resulting in my_answer X to be yield as an answer.

The interpreter has also been extended to handle simple function and generator calls to Python  using the same prefix operator syntax:

- ``` `f A B .. Z  R```, resulting in Python function ```f(A,B,C)``` being called and R unified with its result
-  ``` ``f A B .. Z  R```, resulting in Python generator ```f(A,B,C)``` being called and R unified with its multiple yields, one a time
- ``` ~R A B .. Z ``` for unifying  ``` ~ R A B .. Z ``` with matching facts in the term store
- ``` # f A B .. Z```, resulting in ```f(A,B,C,..,Z)``` being called with no result returned


### A nested tuple store for unification-based tuple mining

An indexer in combination with the unification algorithm is used to retrieve ground terms matching terms containing logic variables.

Indexing is on all constants occurring in 
ground facts placed in a database. 

As facts are ground,
unification has occurs check and trailing turned off when searching
for a match.

To try it out, do:

```python3 -i tests.py```

`````>>> dtest()`````

It gives, after digesting a text and then querying it:

```
   John has (a car).
   Mary has (a bike).
   Mary is (a student).
   John is (a pilot).
   
('John', 'has', ('a', 'car'))
('Mary', 'has', ('a', 'bike'))
('Mary', 'is', ('a', 'student'))
('John', 'is', ('a', 'pilot'))


Who has (a What)?
--> ('John', 'has', ('a', 'car'))
--> ('Mary', 'has', ('a', 'bike'))

Who is (a pilot)?
--> ('John', 'is', ('a', 'pilot'))

'Mary' is What?
--> ('Mary', 'is', ('a', 'student'))

'John' is (a What)?
--> ('John', 'is', ('a', 'pilot'))

Who is What?
--> ('Mary', 'is', ('a', 'student'))
--> ('John', 'is', ('a', 'pilot'))

```

