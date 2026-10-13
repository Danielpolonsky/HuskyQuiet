"""HuskyQuiet web application. Run with ``python -m streamlit run streamlit_app.py``."""

import streamlit as st

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
