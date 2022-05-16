c:-make.

go:-pro2nat('sudoku4'),shell('cat sudoku4.nat').

pro2nat(F):-
 pl2nl(F),
 writeln(done).


pl2nl(F):-
  atomic_list_concat(['prolog_progs/',F,'.pro'],PL),
  atom_concat(F,'.nat',NL),
  pl2nl(PL,NL).

pl2nl(PL,NL):-
  see(PL),
  tell(NL),
  repeat,
   clause2sent(EOF),
   EOF==yes,
   !,
  seen,
  told.

read_with_names(T,T0):-
  read_term(T0,[variable_names(Es)]),
  copy_term(T0,T),
  maplist(call,Es).

writes(':'):-!,write((':')),nl,write('  ').
writes(','):-!,write(','),nl,write('  ').
writes(('.')):-!,write(('.')),nl.
writes(W):-write(W),write(' ').

clause2sent(EOF):-
   read_with_names(T,T0),
   (
     T==end_of_file->EOF=yes
   ;
     EOF=no,
     cls2nat(T,Ns),
     T=T0,
     Ns=Xs,
     maplist(writes,Xs),nl
  ).

cls2nat(C,Es):-
    cls2eqs(C,Es).

cls2eqs(C,Rs):-
  (C=(H:-Bs)->true;C=H,Bs=true),
  cls2list(H,Bs,Ts),
  maplist(term2list,Ts,[Hs|Bss]),
  add_commas(Bss,Cs),
  (Cs=[]->Neck=[];Neck=[':']),
  append([Hs,Neck,Cs,['.']],Rs).

add_commas([],[]).
add_commas([Xs],Xs):-!.
add_commas([Xs|Xss],Rs):-add_commas(Xss,Rs1),append(Xs,[','|Rs1],Rs).

cls2list(H,Bs,Cs):-
  body2list((H,Bs),Cs).

body2list(B,R):-var(B),!,R=[call(B)].
body2list((B,Cs),[B|Bs]):-!,body2list(Cs,Bs).
body2list(true,[]):-!.
body2list(C,[C]).


term2list(T,Zs):-term2list(T,Xs,[]),Xs=[_|Ys],append(Zs,[_],Ys),!.

term2list(A)-->{var(A)},!,[A].
term2list([])-->!,['()'].
term2list(A)-->{atomic(A)},!,[A].
term2list([X|Xs])-->!,['('],term2list(X),term2list(Xs),[')'].
term2list(T)-->{T=..[F|Xs]},['(',F],term2tuple(Xs),[')'].

term2tuple([])-->[].
term2tuple([X|Xs])-->term2list(X),term2tuple(Xs).



