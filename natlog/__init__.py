__all__ = ("Natlog", "interp", "natlog")

from natlog.natlog import *
from natlog.db import Db

__version__ = "2.2.2"


def get_version():
    return __version__
