bug :- #(print,bug).

buggy(X) :- true,true.

nobug(X) :- eq(X,ok).

proc(ok) :- #(print,hello, you).

fun(R) :- `(len, [1, 2, 3], R).

gen(R) :- ``(range, 1, 5, R).

fib(Xs) :-eng(X,slide_fibo(1,1),E),take(10,E,Xs).

p:-q,r.
q:-r.
r.