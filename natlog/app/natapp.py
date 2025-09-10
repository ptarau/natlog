import os
from pathlib import Path

import streamlit as st

from natlog.natlog import *
from natlog import get_version

print("Running Natlog as a streamlit app!")

# ----- Page config (only once) -----
st.set_page_config(
    page_title="Natlog",
    page_icon=":lips:",
    layout="wide",
)

st.sidebar.title("[NatLog](https://github.com/ptarau/natlog) app " + get_version())

# ----- Config / state -----


if "program_text" not in st.session_state:
    st.session_state.program_text = ""
if "edit_area" not in st.session_state:
    st.session_state.edit_area = ""


def ppp(*args):
    args = [str(x) for x in args]
    st.write(*args)


# ----- Syntax & allowed extensions -----
# Expected ext with and without dot
expected_ext_no_dot = ["nat", "pro", "pl"]


def syntax():  # ext is extension
    if uploaded:
        assert ext in expected_ext_no_dot, expected_ext_no_dot
        return "natlog" if ext == "nat" else "prolog"
    else:
        return edit_area_syntax


# ----- Helpers -----


def save_uploaded_file(uploaded_file) -> str:
    """
    Decodes as UTF-8 with replacement to avoid crashes on odd bytes.
    """
    raw: bytes = uploaded_file.read()  # read once; returns bytes
    text = raw.decode("utf-8", errors="replace")
    return text


# ----- Upload UI -----
uploaded = st.sidebar.file_uploader(
    "Select a file", type=expected_ext_no_dot  # must be extension without dot
)

# If a new file is uploaded this run, validate & save it.
if uploaded is not None:  # ext will be defined
    _, ext = os.path.splitext(uploaded.name)
    if ext.startswith("."):
        ext = ext[1:]
    # print("!!! UPLOADED:", ext)
    if ext.lower() not in expected_ext_no_dot:
        st.sidebar.error(f"Please choose a file with suffix in {expected_ext_no_dot}!")
    else:  # ext is good
        try:
            text = save_uploaded_file(uploaded)
            st.session_state.program_text = text
        except Exception as ex:
            st.sidebar.error(f"Loading to edit area failed: {ex}")
else:
    ext = "nat"  # default


# ----- Editor -----


def clear_text_area():
    st.session_state.edit_area = ""
    st.session_state.program_text = ""
    uploaded = None


if uploaded is None:
    edit_area_syntax = st.radio("Editor syntax", ["natlog", "prolog"], horizontal=True)
    clear_text_area()


st.session_state.program_text = st.text_area(
    "Program text:", st.session_state.program_text, height=320, key="edit_area"
)

st.button("🧹 Clear edit area", on_click=clear_text_area)

# ----- Query UI -----

with st.sidebar:

    question = st.text_area(f"Enter your **{syntax()}** query!")
    query_it = st.button("Submit query!")


def do_query():

    print("running with code in edit area, chars:", len(st.session_state.program_text))
    print("syntax:", syntax())
    if not st.session_state.program_text:
        text = "hi."  # hi." # Natlog unhappy with empty program
    else:
        text = st.session_state.program_text

    if query_it and not question:
        st.error("Please submit query!")
        return

    nat = Natlog(text=text, syntax=syntax(), with_lib=natprogs() + "lib.nat")

    ppp("?- " + question)
    success = False
    ppp("ANSWERS:")
    for answer in nat.solve(question):
        success = True
        ppp(answer)
    if not success:
        ppp("No ANSWER!")
    ppp("")


if query_it:
    do_query()
