import os

import streamlit as st

from natlog.natlog import *

print('Running Natlog as a streamlit app!')

st.set_page_config(layout="wide")

st.title('Streamlit-based NatLog Client')


def ppp(*args):
    st.write(*args)


upload_dir = natprogs()

suf = '.nat'


def handle_uploaded(uploaded_file):
    if uploaded_file is not None:
        fname = save_uploaded_file(uploaded_file)
        suf0 = '.' + fname.split('.')[-1]
        if suf0 == suf:
            return fname
        else:
            ppp(f'Please chose a {suf} file!')
    else:
        ppp(f'Please upload your {suf} file!')


def save_uploaded_file(uploaded_file):
    name = uploaded_file.name
    fname = os.path.join(upload_dir, name)
    if exists_file(fname): return fname
    ensure_path(upload_dir)
    with open(fname, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return fname


def ensure_path(fname):
    folder, _ = os.path.split(fname)
    os.makedirs(folder, exist_ok=True)


def exists_file(fname):
    return os.path.exists(fname)


fname = handle_uploaded(st.sidebar.file_uploader('Select a File', type=[suf]))
print(f'fname={fname}:')

with st.sidebar:
    question = st.text_area('Query?')
    query_it = st.button('Submit your question!')


def do_query():
    if fname is not None:
        lib = natprogs() + "lib.nat"
        if fname != lib:
            with_lib=lib
        else:
            with_lib=None
        nat = Natlog(file_name=fname, with_lib=lib)

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
