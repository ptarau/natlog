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
# Single source of truth for the editor content
if "program_text" not in st.session_state:
    st.session_state.program_text = ""

# Track the last uploaded file to avoid re-loading every rerun
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# Default editor syntax when not deduced from a file
if "edit_area_syntax" not in st.session_state:
    st.session_state.edit_area_syntax = "natlog"  # or "prolog"

# Track the last uploaded extension so syntax() can be stable across reruns
if "last_uploaded_ext" not in st.session_state:
    st.session_state.last_uploaded_ext = "nat"


def ppp(*args):
    args = [str(x) for x in args]
    st.write(*args)


# ----- Syntax & allowed extensions -----
expected_ext_no_dot = ["nat", "pro", "pl"]  # without dot


def syntax():
    """
    Decide syntax: if a file has been uploaded (remembered via last_uploaded_ext),
    use its extension; otherwise use the radio selection.
    """
    ext = st.session_state.get("last_uploaded_ext", "nat")
    if st.session_state.last_uploaded_name:
        assert ext in expected_ext_no_dot, expected_ext_no_dot
        return "natlog" if ext == "nat" else "prolog"
    else:
        return st.session_state.edit_area_syntax


# ----- Helpers -----
def save_uploaded_file(uploaded_file) -> str:
    """
    Decodes as UTF-8 with replacement to avoid crashes on odd bytes.
    """
    raw: bytes = uploaded_file.read()  # read once; returns bytes
    text = raw.decode("utf-8", errors="replace")
    return text


def clear_text_area():
    st.session_state.program_text = ""
    # Do NOT try to mutate the upload widget by setting a local variable.
    # Optionally, also forget the last uploaded name/ext:
    st.session_state.last_uploaded_name = None
    st.session_state.last_uploaded_ext = "nat"


# ----- Upload UI -----
uploaded = st.sidebar.file_uploader(
    "Select a file", type=expected_ext_no_dot  # extension without dot
)

# If a (new) file is uploaded this run, validate & load it once.
if uploaded is not None:
    filename = uploaded.name
    _, ext = os.path.splitext(filename)
    if ext.startswith("."):
        ext = ext[1:]
    if ext.lower() not in expected_ext_no_dot:
        st.sidebar.error(f"Please choose a file with suffix in {expected_ext_no_dot}!")
    else:
        # Only (re)load if a different file (or first time)
        if filename != st.session_state.last_uploaded_name:
            try:
                text = save_uploaded_file(uploaded)
                st.session_state.program_text = text
                st.session_state.last_uploaded_name = filename
                st.session_state.last_uploaded_ext = ext.lower()
                # Also set syntax to match the uploaded file
                st.session_state.edit_area_syntax = (
                    "natlog"
                    if st.session_state.last_uploaded_ext == "nat"
                    else "prolog"
                )
            except Exception as ex:
                st.sidebar.error(f"Loading to edit area failed: {ex}")
else:
    # No new upload this run; keep whatever we had
    pass

# ----- Editor -----
# Only show a syntax picker when there is no remembered upload
if not st.session_state.last_uploaded_name:
    st.session_state.edit_area_syntax = st.radio(
        "Editor syntax",
        ["natlog", "prolog"],
        horizontal=True,
        index=0 if st.session_state.edit_area_syntax == "natlog" else 1,
    )

st.text_area("Program text:", key="program_text", height=320)
st.button("🧹 Clear edit area", on_click=clear_text_area)

# ----- Query UI -----
with st.sidebar:
    question = st.text_area(f"Enter your **{syntax()}** query!", key="query_text")
    query_it = st.button("Submit query!", key="submit_query")


def do_query():
    prog_text = (
        st.session_state.program_text or "hi."
    )  # Natlog unhappy with empty program

    if query_it and not question:
        st.error("Please submit query!")
        return

    nat = Natlog(text=prog_text, syntax=syntax(), with_lib=natprogs() + "lib.nat")

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
