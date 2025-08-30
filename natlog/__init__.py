__all__ = ("Natlog", "interp", "natlog")

from natlog.natlog import *
from natlog.db import Db

__version__ = "1.9.8"


def get_version():
    return __version__
