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
UPLOAD_DIR = Path("UPLOADS")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "program_text" not in st.session_state:
    st.session_state.program_text = ""


def ppp(*args):
    args = [str(x) for x in args]
    st.write(*args)


# ----- Syntax & allowed extensions -----
syntax = st.radio("Program syntax", ["natlog", "prolog"], horizontal=True)
# Expected ext with and without dot
expected_ext_no_dot = "nat" if syntax == "natlog" else "pro"
expected_ext_dot = f".{expected_ext_no_dot}"


# ----- Helpers -----
def ensure_unique_path(dirpath: Path, name: str) -> Path:
    """Return a unique path inside dirpath for filename name."""
    base = Path(name).stem
    ext = Path(name).suffix
    candidate = dirpath / f"{base}{ext}"
    i = 1
    while candidate.exists():
        candidate = dirpath / f"{base}_{i}{ext}"
        i += 1
    return candidate


def save_uploaded_file(uploaded_file) -> tuple[str, str]:
    """
    Save the uploaded file to UPLOADS and return (absolute_path_str, text).
    Decodes as UTF-8 with replacement to avoid crashes on odd bytes.
    """
    raw: bytes = uploaded_file.read()  # read once; returns bytes
    # Persist to disk with a unique name (preserve original base + ext)
    target_path = ensure_unique_path(UPLOAD_DIR, uploaded_file.name)
    target_path.write_bytes(raw)
    # Decode to text for the editor (best-effort)
    text = raw.decode("utf-8", errors="replace")
    return str(target_path.resolve()), text


def load_file_text(path_str: str) -> str:
    try:
        return Path(path_str).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        st.error(f"Failed to read saved file: {e}")
        return ""


# ----- Upload UI -----
uploaded = st.sidebar.file_uploader(
    "Select a file",
    type=[expected_ext_no_dot],  # must be extension without dot
)

# If a new file is uploaded this run, validate & save it.
if uploaded is not None:
    # Double-check extension (defensive; streamlit already filters by 'type=')
    _, ext = os.path.splitext(uploaded.name)
    if ext.lower() != expected_ext_dot:
        st.sidebar.error(
            f"Please choose a {expected_ext_dot} file to match '{syntax}' syntax."
        )
    else:
        try:
            saved_path, text = save_uploaded_file(uploaded)
            st.session_state.uploaded_path = saved_path
            st.session_state.program_text = text
            st.sidebar.success(f"Saved to {saved_path}")
        except Exception as ex:
            st.sidebar.error(f"Saving failed: {ex}")

# Convenience: show current file info
if st.session_state.uploaded_path:
    st.sidebar.caption(f"Current file: `{st.session_state.uploaded_path}`")

# ----- Editor -----
# If there is a saved file but no program_text (e.g., fresh run), reload it
if st.session_state.uploaded_path and not st.session_state.program_text:
    st.session_state.program_text = load_file_text(st.session_state.uploaded_path)

editor = st.text_area("Program text:", st.session_state.program_text, height=320)

# Allow saving edits back to the same file (optional but handy)
col_save, col_clear = st.columns([1, 1])
with col_save:
    if st.button("💾 Save edits to file"):
        if st.session_state.uploaded_path:
            try:
                Path(st.session_state.uploaded_path).write_text(
                    editor, encoding="utf-8", errors="strict"
                )
                st.success("Edits saved.")
                st.session_state.program_text = editor
            except Exception as ex:
                st.error(f"Could not save edits: {ex}")
        else:
            st.warning("No uploaded file yet. Upload a file first.")
with col_clear:
    if st.button("🧹 Clear uploaded file (keep editor)"):
        st.session_state.uploaded_path = None

# ----- Query UI -----
with st.sidebar:
    question = st.text_area("Query?")
    query_it = st.button("Submit your question!")


def do_query():
    use_file = st.session_state.uploaded_path is not None
    if use_file:
        lib = natprogs() + "lib.nat"
        with_lib = None if st.session_state.uploaded_path == lib else lib
        nat = Natlog(
            file_name=st.session_state.uploaded_path, syntax=syntax, with_lib=with_lib
        )
    else:
        print("running with code in editor, chars:", len(editor), "syntax:", syntax)
        nat = Natlog(text=editor, syntax=syntax, with_lib=natprogs() + "lib.nat")

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

# Debug prints (safe)
print("fname=", st.session_state.uploaded_path, "chars:", len(editor))
