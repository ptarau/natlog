import timeit
import sys
from natlog.natlog import *

sys.setrecursionlimit(1 << 28)

def time_of(f, x, times=1):
    res = None
    start_time = timeit.default_timer()
    for i in range(times):
        res = f(x)
        if i == times - 1: print(x)
    end_time = timeit.default_timer()
    print(x, '==>', 'res = ', res)
    total_time=end_time - start_time
    print('time = ', total_time)
    print('')
    return total_time


my_text = """
    app () Ys Ys. 
    app (X Xs) Ys (X Zs) : 
        app Xs Ys Zs.

    nrev () ().
    nrev (X Xs) Zs : nrev Xs Ys, app Ys (X ()) Zs.

    goal N Ys :
      `numlist 0 N Xs,
      nrev Xs Ys.
    """


def bm1():
    n = Natlog(text=my_text)
    print('NREV STARTING:')
    n.query("goal 10 L?")
    time_of(n.count, "goal 16 L?", times=512)
    time_of(n.count, "goal 32 L?", times=256)
    time_of(n.count, "goal 64 L?", times=64)
    t=time_of(n.count, "goal 128 L?", times=32)
    lips = 128*129//2*32/t
    print('LIPS:',lips)
    #time_of(n.count, "goal 256 L?", times=1)
    #time_of(n.count, "goal 512 L?", times=1)
    #time_of(n.count, "goal 1024 L?", times=1)
    print('')


def bm():
    print('N-QUEENS STARTING:')
    n = Natlog(file_name=natprogs()+"queens.nat")
    time_of(n.count, "goal8 Queens?", times=9)
    time_of(n.count, "goal9 Queens?")
    time_of(n.count, "goal10 Queens?")
    # return # runs, but quite a bit longer
    time_of(n.count, "goal11 Queens?")
    time_of(n.count, "goal12 Queens?")


def prof():
    import cProfile
    p = cProfile.Profile()

    n = Natlog(file_name=natprogs()+"queens.nat")

    def fun():
        n.count('goal10 L?')

    print('PROFILING STARTED')
    p.runcall(fun)
    p.print_stats(sort=1)


def run_all():
    bm1()
    bm()
    prof()

if __name__ == "__main__":
    run_all()


