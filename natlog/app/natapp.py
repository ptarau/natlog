import os

import streamlit as st

from natlog.natlog import *

print('Running Natlog as a streamlit app!')

st.set_page_config(layout="wide")

st.sidebar.title('Streamlit-based [NatLog](https://github.com/ptarau/natlog) Client')


def ppp(*args):
    args=[str(x) for x in args]
    st.write(*args)


upload_dir = "UPLOADS/"

suf = '.nat'


def handle_uploaded(uploaded_file):
    if uploaded_file is not None:
        fname, prog = save_uploaded_file(uploaded_file)
        suf0 = '.' + fname.split('.')[-1]
        if suf0 == suf:
            return fname, prog
        else:
            ppp(f'Please chose a {suf} file!')
    else:
        ppp(f'You can also edit your code here!')
    return None, ""


def save_uploaded_file(uploaded_file):
    name = uploaded_file.name
    fname = os.path.join(upload_dir, name)
    # if exists_file(fname): return fname,file2string(fname)
    ensure_path(upload_dir)
    bs = uploaded_file.getbuffer()
    prog = str(bs, 'utf-8')

    with open(fname, "wb") as f:
        f.write(bs)
    return fname, prog


def ensure_path(fname):
    folder, _ = os.path.split(fname)
    os.makedirs(folder, exist_ok=True)


def exists_file(fname):
    return os.path.exists(fname)


def file2string(fname):
    with open(fname, 'r') as f:
        return f.read()


fname, prog = handle_uploaded(st.sidebar.file_uploader('Select a File', type=[suf]))
print(f'fname={fname} chars:',len(prog))

editor = st.text_area('Program', prog, height=320)  # pixels

print('editor chars:',len(editor))

with st.sidebar:
    question = st.text_area('Query?')
    query_it = st.button('Submit your question!')


def do_query():
    if fname is not None:
        lib = natprogs() + "lib.nat"
        if fname != lib:
            with_lib = lib
        else:
            with_lib = None
        nat = Natlog(text=editor, with_lib=with_lib)
    else:
        print('running with code in editor, chars:',len(editor))
        nat = Natlog(text=editor, with_lib=natprogs() + "lib.nat")

    ppp('?- ' + question)

    success = False
    ppp('ANSWERS:')
    for answer in nat.solve(question):
        success = True
        ppp(answer)
    if not success:
        ppp('No ANSWER!')
    ppp('')

if query_it:
    do_query()
