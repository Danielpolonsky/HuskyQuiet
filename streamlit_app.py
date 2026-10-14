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
        st.Page("app_pages/explore.py", title="Explore map", icon=":material/map:", default=True),
        st.Page("app_pages/analyze.py", title="Analyze address", icon=":material/location_on:"),
        st.Page("app_pages/compare.py", title="Compare apartments", icon=":material/compare_arrows:"),
        st.Page("app_pages/performance.py", title="Model performance", icon=":material/monitoring:"),
        st.Page("app_pages/about.py", title="About", icon=":material/menu_book:"),
    ],
    position="top",
)

st.markdown("**HuskyQuiet**")
st.caption(
    "Understand development activity near your next home. "
    "Forecasts are the probability a pending permit is issued within 90 days, not construction activity or noise."
)
page.run()
