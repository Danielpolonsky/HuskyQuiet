import streamlit as st

from huskyquiet.analysis import ISSUANCE_NOT_NOISE, format_cost, format_probability, nearby_summary, resolve_pin
from huskyquiet.charts import probability_histogram
from huskyquiet.geo import with_distance
from huskyquiet.geocode import GeocodeError
from huskyquiet.mapdeck import location_deck
from huskyquiet.ui import (
    geocoder,
    load_active,
    parse_coord,
    permit_table,
    render_summary_metrics,
    show_figure,
    status_caption,
)

RADIUS = {"250 m": 250, "500 m": 500, "1 km": 1000}

st.header("Analyze address")
st.caption(
    "Find pending permit applications near a Seattle address. "
    "Results are issuance forecasts from the production model, not a construction or noise forecast."
)
active = load_active()
status_caption(len(active))

with st.form("analyze-address"):
    address = st.text_input("Seattle address", placeholder="4500 University Way NE")
    radius_label = st.segmented_control("Search radius", list(RADIUS), default="500 m")
    manual = st.checkbox("Enter coordinates instead of geocoding")
    coord_cols = st.columns(2)
    with coord_cols[0]:
        lat_text = st.text_input("Latitude", placeholder="47.66150")
    with coord_cols[1]:
        lon_text = st.text_input("Longitude", placeholder="-122.31300")
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    try:
        latitude = parse_coord(lat_text)
        longitude = parse_coord(lon_text)
        pin = resolve_pin(address, latitude, longitude, manual, geocoder())
    except ValueError as exc:
        message = str(exc)
        st.error("Coordinates must be decimal numbers." if "could not convert" in message else message)
    except GeocodeError as exc:
        st.error(str(exc))
    else:
        st.session_state["analysis_pin"] = pin
        st.session_state["analysis_radius"] = RADIUS[radius_label or "500 m"]

pin = st.session_state.get("analysis_pin")
if pin is not None:
    radius_m = int(st.session_state.get("analysis_radius", 500))
    nearby = with_distance(active, pin.latitude, pin.longitude, radius_m)
    summary = nearby_summary(active, pin.latitude, pin.longitude, radius_m, pin.label)
    st.markdown(f"**{pin.label}**")
    st.caption(f"Located by {pin.source} at {pin.latitude:.5f}, {pin.longitude:.5f}. Radius {radius_m:,} m.")
    render_summary_metrics(summary)
    st.pydeck_chart(
        location_deck(active, pin.latitude, pin.longitude, radius_m),
        height=520,
        alt="Map of the selected address, search radius, and nearby pending permits",
    )
    if summary.n_permits == 0:
        st.info("No pending applications with usable coordinates fall inside this radius.")
    else:
        show_figure(
            probability_histogram({"This address": nearby["probability_issued_90d"]}),
            "Distribution of predicted 90-day issuance probabilities near the address",
        )
        st.subheader("Highest predicted issuance probabilities")
        for row in summary.top:
            distance = "distance unknown" if row["distance_m"] is None else f"{row['distance_m']:.0f} m away"
            st.markdown(
                f"**{format_probability(row['probability_issued_90d'])}** · {row['address']} · "
                f"{row['permit_type_desc']} · {distance} · {format_cost(row['est_project_cost'])}"
            )
            st.text(row["description"])
        with st.expander("All permits in the radius"):
            permit_table(nearby, key="analyze-table", alt="Pending permits within the selected radius")
    st.caption(ISSUANCE_NOT_NOISE)
