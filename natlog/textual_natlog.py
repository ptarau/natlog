from .natlog import Natlog, natprogs
from .tdb import *


class TextualNatlog(Natlog):
    """
    overrrides Natlog's database constructor
    to use an indexed text seen as a set of ground db facts
    """

    def db_init(self):
        self.db = Tdb()


def xconsult(fname):
    nname = natprogs() + fname + ".nat"
    dname = natprogs() + fname + ".txt"
    print('consulted:', nname, dname)
    n = TextualNatlog(file_name=nname, db_name=dname)
    n.repl()
