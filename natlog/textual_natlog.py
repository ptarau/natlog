from .natlog import Natlog, natprogs
from .tdb import *


class TextualNatlog(Natlog):
    """
    overrrides Natlog's database constructor
    to use a neurally indexed nd instead of Db
    """

    def db_init(self):
        self.db = Tdb()


def xconsult(fname):
    nname = natprogs() + fname + ".nat"
    dname = natprogs() + fname + ".txt"
    print('consulted:', nname, dname)
    n = TextualNatlog(file_name=nname, db_name=dname)
    n.repl()
