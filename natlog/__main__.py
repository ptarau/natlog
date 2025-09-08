import sys
from natlog import get_version
from natlog.natlog import natlog

print("Natlog: ", get_version())
print(
    """Usage: python3 -m <file_name> <goal>

    <file_name>, if present, must end with .nat, .pro or .pl

    <goal> if present, is a string representing a Natlog goal to query <file_name>

    if both absent, the REPL starts, with the only the library loaded.
    """
)
k = len(sys.argv)
file_name, goal = None, None
syntax = "natlog"
if k > 1:
    file_name = sys.argv[1]
    if file_name.endswith(".pl") or file_name.endswith(".pl"):
        syntax = "prolog"

    if k > 2:
        goal = sys.argv[2:]
        print("CML GOAL:", len(sys.argv), goal)
else:
    file_name = None

natlog(file_name=file_name, goal=goal, syntax=syntax)
