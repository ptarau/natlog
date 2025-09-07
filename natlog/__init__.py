__all__ = ("Natlog", "interp", "natlog")

from natlog.natlog import *
from natlog.db import Db

__version__ = "2.1.0"


def get_version():
    return __version__
