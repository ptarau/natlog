from .natlog import Natlog,natprogs
from .ndb import *


class NeuralNatlog(Natlog):
    """
    overrrides Natlog's database constructor
    to use a neurally indexed nd instead of Db
    """

    def db_init(self):
        self.db = Ndb()


def nconsult(fname):
    nname = natprogs() + fname + ".nat"
    dname = natprogs() + fname + ".tsv"
    print('consulted:',nname,dname)
    n = NeuralNatlog(file_name=nname,db_name=dname)

    n.repl()
