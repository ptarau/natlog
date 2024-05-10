from natlog.natlog import Natlog, natprogs
from softdb import *


class SoftNatlog(Natlog):
    """
    overrrides Natlog's database constructor
    to use semantic similarity when unifying with facts
    """

    def db_init(self):
        self.db = SoftDB()

    def repl(self):
        """
        read-eval-print-loop
        """
        print("Type ENTER to quit.")
        while True:
            q = input('?- ')
            if not q: return
            try:
                self.query(q, in_repl=True)
                print('ABDUCED CLAUSES')
                for (h,b,r) in self.db.abduced_clauses:
                    print(f"{h} : {b} % distance={r}")
                print()

            except Exception as e:
                print('EXCEPTION:', type(e).__name__, e.args)
                raise e


def xconsult(fname):
    nname = natprogs() + fname + ".nat"
    dname = natprogs() + fname + ".txt"
    print('consulted:', nname, dname)
    n = SoftNatlog(file_name=nname, db_name=dname)
    n.repl()
