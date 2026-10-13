import streamlit as st

from huskyquiet.analysis import ISSUANCE_NOT_NOISE, comparison_text, nearby_summary, resolve_pin
from huskyquiet.charts import category_bars, probability_histogram
from huskyquiet.geo import with_distance
from huskyquiet.geocode import GeocodeError
from huskyquiet.mapdeck import location_deck
from huskyquiet.ui import geocoder, load_active, parse_coord, render_summary_metrics, show_figure, status_caption

RADIUS = {"250 m": 250, "500 m": 500, "1 km": 1000}


def _short(label: str) -> str:
    return label if len(label) <= 42 else label[:39] + "..."

st.header("Compare apartments")
st.caption(
    "The same radius, the same model, and two Seattle addresses. "
    "The paragraph under the charts is computed from the counts. It does not judge which home is quieter."
)
active = load_active()
status_caption(len(active))

with st.form("compare-addresses"):
    radius_label = st.segmented_control("Search radius", list(RADIUS), default="500 m", key="compare_radius")
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**First address**")
        left_address = st.text_input(
            "First address", placeholder="4500 University Way NE", label_visibility="collapsed"
        )
        left_manual = st.checkbox("Use coordinates for the first address")
        left_lat = st.text_input("First latitude", placeholder="47.66150")
        left_lon = st.text_input("First longitude", placeholder="-122.31300")
    with right_col:
        st.markdown("**Second address**")
        right_address = st.text_input(
            "Second address", placeholder="2100 N Northlake Way", label_visibility="collapsed"
        )
        right_manual = st.checkbox("Use coordinates for the second address")
        right_lat = st.text_input("Second latitude", placeholder="47.64700")
        right_lon = st.text_input("Second longitude", placeholder="-122.33600")
    submitted = st.form_submit_button("Compare", type="primary")

if submitted:
    try:
        finder = geocoder()
        left_pin = resolve_pin(
            left_address, parse_coord(left_lat), parse_coord(left_lon), left_manual, finder
        )
        right_pin = resolve_pin(
            right_address, parse_coord(right_lat), parse_coord(right_lon), right_manual, finder
        )
    except ValueError as exc:
        message = str(exc)
        st.error("Coordinates must be decimal numbers." if "could not convert" in message else message)
    except GeocodeError as exc:
        st.error(str(exc))
    else:
        st.session_state["compare_pins"] = (left_pin, right_pin)
        st.session_state["compare_radius_m"] = RADIUS[radius_label or "500 m"]

pins = st.session_state.get("compare_pins")
if pins:
    left_pin, right_pin = pins
    radius_m = int(st.session_state.get("compare_radius_m", 500))
    left_nearby = with_distance(active, left_pin.latitude, left_pin.longitude, radius_m)
    right_nearby = with_distance(active, right_pin.latitude, right_pin.longitude, radius_m)
    left_summary = nearby_summary(active, left_pin.latitude, left_pin.longitude, radius_m, left_pin.label)
    right_summary = nearby_summary(active, right_pin.latitude, right_pin.longitude, radius_m, right_pin.label)
    columns = st.columns(2)
    decks = (
        (left_pin, left_summary, left_nearby),
        (right_pin, right_summary, right_nearby),
    )
    for column, (pin, summary, _nearby) in zip(columns, decks, strict=True):
        with column:
            st.markdown(f"**{pin.label}**")
            st.caption(f"{pin.latitude:.5f}, {pin.longitude:.5f} · {pin.source}")
            render_summary_metrics(summary)
            st.pydeck_chart(
                location_deck(active, pin.latitude, pin.longitude, radius_m),
                height=420,
                alt=f"Map of pending permits within {radius_m} meters of {pin.label}",
            )
    st.markdown(comparison_text(left_summary, right_summary))
    if left_summary.n_permits or right_summary.n_permits:
        show_figure(
            probability_histogram({
                _short(left_pin.label): left_nearby["probability_issued_90d"],
                _short(right_pin.label): right_nearby["probability_issued_90d"],
            }),
            "Issuance-probability distributions for the two addresses",
        )
        show_figure(
            category_bars(
                _short(left_pin.label), left_summary.by_description,
                _short(right_pin.label), right_summary.by_description,
            ),
            "Permit description categories near the two addresses",
        )
    st.caption(ISSUANCE_NOT_NOISE)
