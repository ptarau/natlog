from .natlog import Natlog
from .ndb import *


class NeuralNatlog(Natlog):
    """
    overrrides Natlog's database constructor
    to use a neurally indexed nd instead of Db
    """

    def db_init(self):
        self.db = Ndb()
