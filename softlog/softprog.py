from natlog.natlog import Natlog, natprogs
from softlog import SoftNatlog


def start():
    nname = "softprog.nat"
    dname = '../docs/quotes.txt'

    n = SoftNatlog(file_name=nname, db_name=dname)
    n.repl()


if __name__ == "__main__":
    start()

# ?- ~'Who barks out there'?
