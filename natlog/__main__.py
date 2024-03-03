import sys
from natlog.natlog import *
print('usage: python3 -m natlog <file_name>.nat\n')
if len(sys.argv)>1:
    file_name=sys.argv[1]
    n = Natlog(file_name=file_name, with_lib=natprogs() + 'lib.nat', callables=globals())
else:
    n= Natlog(text='',with_lib=natprogs() + 'lib.nat', callables=globals())
n.repl()
