from vp import *


def leg():
    s = sphere()
    size(s, 1, 2, 1)
    col(s, 'blue')
    down(s)
    return s


def head():
    h = sphere()
    size(h, 2, 3, 2)
    col(h, 'white')
    up(h)
    up(h)
    up(h)
    return h


def eye():
    s = sphere()
    size(s, 0.5, 0.4, 0.4)
    col(s, 'red')
    closer(s)
    closer(s)
    up(s)
    up(s)
    up(s)
    up(s)
    rotz(s)
    return s


def humpty_dumpty():
    x = eye()
    left(x)
    y = eye()
    right(y)
    h = head()
    ll = leg()
    left(ll)
    rl = leg()
    right(rl)
    return compound((x, y, h, ll, rl))


def tumble(c):
    for _ in range(6):
        left(c)
        sleep(0.5)
        rotx(c)
        sleep(0.5)
        roty(c)
        sleep(0.5)
        rotz(c)
        right(c)


def go():
    start()
    c=humpty_dumpty()
    tumble(c)


if __name__ == "__main__":
    go()
