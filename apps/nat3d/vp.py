from vpython import *
from natlog import Natlog, natprogs

scene=canvas(width=1200, height=800, title='Natlog 3D DEMO, zoom in!')

shared = dict()


def share(f):
    # print('SHARING:', f.__name__)
    shared[f.__name__] = f
    return f


def share_primitives():
    for f in (
        sleep, box, cylinder, cone, sphere, ellipsoid, pyramid, compound, vector, color, helix, ring,
        compound
    ):
        share(f)
    shared['scene'] = scene
    return shared


@share
def start():
    scene.caption = """To rotate "camera", drag with right button or Ctrl-drag.
    To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
      On a two-button mouse, middle is left + right.
    To pan left/right and up/down, Shift-drag.
    Touch screen: pinch/extend to zoom, swipe or two-finger rotate."""
    return scene


@share
def objects():
    return tuple(shared.keys())


delta = 0.5


@share
def desc(o):
    try:
        d = o.__dict__
    except Exception:
        d = dict()
    print(d)


@share
def step(val):
    global delta
    delta = val


@share
def show(x):
    x.visible = True


@share
def hide(x):
    x.visible = False


@share
def left(x):
    x.pos = x.pos + vector(-delta, 0, 0)


@share
def right(x):
    x.pos = x.pos + vector(delta, 0, 0)


@share
def up(x):
    x.pos = x.pos + vector(0, delta, 0)


@share
def down(x):
    x.pos = x.pos + vector(0, -delta, 0)


@share
def closer(x):
    x.pos = x.pos + vector(0, 0, delta)


@share
def farther(x):
    x.pos = x.pos + vector(0, 0, -delta)


@share
def rotx(o, angle=90):
    o.rotate(angle=angle, axis=vector(1, 0, 0))


@share
def roty(o, angle=90):
    o.rotate(angle=angle, axis=vector(0, 1, 0))


@share
def rotz(o, angle=90):
    o.rotate(angle=angle, axis=vector(0, 0, 1))


@share
def col(x, c):
    if isinstance(c, str):
        c = getattr(color, c)
    x.color = c


@share
def size(o, x, y, z):
    v = vector(x, y, z)
    o.size = v


@share
def resize(o, k):
    o.size = o.size * float(k)


def normalize(o):
    v=o.size
    k = 3/(v.x + v.y + v.z)
    o.pos = vector(0, 0, 0)
    resize(o, k)


# tests

def rtest(i=20):
    b = box(size=vector(5, 2, 1))
    b.color = color.blue

    def w(t=2):
        sleep(t)
        print(b.pos, b.axis)

    for _ in range(i):
        # rotx(b)
        w()
        roty(b)
        w()
        rotz(b)


def ttest(i=10):
    s = cone(radius=3)

    def w(t=0.3):
        sleep(t)
        print(s.pos)

    for _ in range(i):

        w()
        s.color = color.blue
        w()
        for i in range(10): farther(s)
        w()
        left(s)
        w()
        up(s)
        w()
        s.color = color.red
        for _ in range(10): closer(s)
        w()
        right(s)
        w()
        down(s)
        w()


def from_stl(fname):  # specify file
    with open(fname, mode='rb') as fd:
        fd.seek(0)
        vs = []
        ts = []  # list of triangles to compound
        for line in fd.readlines():
            ps = line.split()
            if ps[0] == b'facet':
                N = vec(float(ps[2]), float(ps[3]), float(ps[4]))
            elif ps[0] == b'vertex':
                vs.append(vertex(pos=vec(float(ps[1]), float(ps[2]),
                                         float(ps[3])), normal=N,
                                 color=color.white))
                if len(vs) == 3:
                    ts.append(triangle(vs=vs))
                    vs = []

        return compound(ts)

@share
def follow(o):
    scene.camera.follow(o)

@share
def camera(x,y,z):
    scene.camera.pos=vector(x,y,z)

@share
def bot():
    b = from_stl('bot.stl')
    normalize(b)
    return b


def nats():
    share_primitives()
    start()
    n = Natlog(file_name="vp.nat",
               with_lib=natprogs() + "lib.nat", callables=shared)
    # n.query("go.")
    n.query("bot dance.")
    n.repl()


if __name__ == "__main__":
    # ttest()
    # rtest()
    #bot()
    nats()
