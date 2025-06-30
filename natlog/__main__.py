import sys
from natlog.natlog import natlog

print(
    """usage: python3 -m <file_name> <goal>
    <file_name> if present, is ending with .nat, .pro or .pl
    <goal> if present, is a string representing a Natlog goal to query <file_name>
    if both absent, the REPL starts, with the only the library loaded.
    """
)
k = len(sys.argv)
file_name, goal = None, None
if k > 1:
    file_name = sys.argv[1]
    if k > 2:
        goal = sys.argv[2]
else:
    file_name = None
natlog(file_name=file_name, goal=goal)
