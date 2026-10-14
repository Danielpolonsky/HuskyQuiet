"""HuskyQuiet web application. Run with ``python -m streamlit run streamlit_app.py``."""

import sys
from pathlib import Path

import streamlit as st

# Community Cloud installs requirements.txt but not the src layout. Importing from
# this checkout keeps Paths() rooted at the repository, not site-packages.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="HuskyQuiet",
    page_icon=":material/map:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

page = st.navigation(
    [
        st.Page("app_pages/explore.py", title="Explore", icon=":material/map:", default=True),
        st.Page("app_pages/compare.py", title="Compare locations", icon=":material/compare_arrows:"),
        st.Page("app_pages/how_it_works.py", title="How it works", icon=":material/menu_book:"),
    ],
    position="top",
)

with st.container(width=760, gap=None):
    st.subheader("HuskyQuiet · Seattle pending-permit explorer", anchor=False)
    st.caption(
        "See pending development permits near a Seattle address and their likelihood "
        "of being issued within 90 days."
    )
page.run()
