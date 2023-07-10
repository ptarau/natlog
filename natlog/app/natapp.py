import os

import streamlit as st

from natlog.natlog import *

print('Running Natlog as a streamlit app!')

st.set_page_config(layout="wide")

st.title('Streamlit-based [NatLog](https://github.com/ptarau/natlog) Client')


def ppp(*args):
    st.write(*args)


upload_dir = natprogs()

suf = '.nat'


def handle_uploaded(uploaded_file):
    if uploaded_file is not None:
        fname,prog = save_uploaded_file(uploaded_file)
        suf0 = '.' + fname.split('.')[-1]
        if suf0 == suf:
            return fname,prog
        else:
            ppp(f'Please chose a {suf} file!')
    else:
        ppp(f'Please upload your {suf} file!')
    return None,""

def save_uploaded_file(uploaded_file):
    name = uploaded_file.name
    fname = os.path.join(upload_dir, name)
    if exists_file(fname): return fname,file2string(fname)
    ensure_path(upload_dir)
    bs = uploaded_file.getbuffer()
    prog=str(bs)

    with open(fname, "wb") as f:
        f.write(bs)
    return fname,prog


def ensure_path(fname):
    folder, _ = os.path.split(fname)
    os.makedirs(folder, exist_ok=True)


def exists_file(fname):
    return os.path.exists(fname)

def file2string(fname):
    with open(fname,'r') as f:
        return f.read()

fname,prog = handle_uploaded(st.sidebar.file_uploader('Select a File', type=[suf]))
print(f'fname={fname}:')

editor = st.text_area('Program',prog, height=320) # pixels

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

        nat = Natlog(file_name=fname, with_lib=with_lib)

    else:
        print('running with lib.nat')
        nat = Natlog(file_name=natprogs() + "lib.nat")

    ppp('?- ' + question)

    answers = nat.solve(question)

    if not answers:
        ppp("I do not know.")
    else:
        for a in answers:
            ppp(a)


if query_it:
    do_query()
else:
    st.write('Please upload a .nat file, then query it!')
