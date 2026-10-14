import streamlit as st

from huskyquiet.analysis import ISSUANCE_NOTE, summarize_permits
from huskyquiet.areas import area_by_label, area_scope, load_areas, permits_in_area
from huskyquiet.config import UW_LAT, UW_LON
from huskyquiet.geo import valid_coordinates, with_distance
from huskyquiet.mapdeck import area_deck, location_deck
from huskyquiet.search import RADII_M, LocationInput, current, submit
from huskyquiet.ui import (
    data_as_of,
    data_caption,
    filter_caption,
    geocoder,
    load_active,
    located,
    long_date,
    permit_filters,
    permit_table,
    probability_legend,
    prose,
    remember_selection,
    render_details,
    render_permit_detail,
    render_summary_metrics,
    selection_prompt,
)

RESULT_KEY = "explore_result"
ALL_SEATTLE = "All of Seattle"

st.header("Explore", anchor=False)
with prose():
    st.markdown(
        "Start from an address to see pending permits within a set distance, "
        "or browse an area of Seattle."
    )
    st.caption(ISSUANCE_NOTE)

active = load_active()
as_of = data_as_of()
data_caption(len(active))
CLASSES = sorted(active["permit_class"].dropna().unique())

mode = st.segmented_control(
    "Start from",
    ["An address", "An area"],
    default="An address",
    required=True,
    key="explore_mode",
)


def show_results(frame, scope: str, file_stem: str, *, show_nearest: bool, deck_for, map_alt: str) -> None:
    """Shared result layout: filters, metrics, map, table, selected permit, details."""
    filtered, filtering = permit_filters(frame, CLASSES, "explore")
    filter_caption(len(filtered), len(frame), scope, filtering)
    summary = summarize_permits(
        filtered, "", scope, distance_col="distance_m" if show_nearest else None
    )
    render_summary_metrics(summary, show_nearest=show_nearest)
    map_state = st.pydeck_chart(
        deck_for(filtered),
        height=520,
        on_select="rerun",
        selection_mode="single-object",
        key=f"explore_map_{file_stem}",
        alt=map_alt,
    )
    probability_legend()
    if filtered.empty:
        if filtering:
            st.info("No permits match these filters. Loosen the filters above to see more.")
        else:
            st.info(f"No pending permits with usable coordinates {scope}.")
        return
    ordered = filtered.sort_values(
        ["probability_issued_90d", "permit_num"], ascending=[False, True]
    ).reset_index(drop=True)
    st.subheader("Permits", anchor=False)
    table_state = permit_table(
        ordered,
        key=f"explore_table_{file_stem}",
        file_stem=file_stem,
        alt=f"Pending permits {scope}, highest chance of issuance first",
    )
    chosen = remember_selection(f"explore_{file_stem}", ordered, map_state, table_state)
    if chosen is None:
        selection_prompt()
    else:
        render_permit_detail(ordered.loc[ordered["permit_num"] == chosen].iloc[0])
    render_details(summary)


if mode == "An address":
    with st.container(width=640):
        address = st.text_input(
            "Seattle address",
            key="explore_address",
            placeholder="4500 University Way NE",
            max_chars=100,
        )
        radius_label = st.segmented_control(
            "Search radius", list(RADII_M), default="500 m", required=True, key="explore_radius"
        )
        with st.expander("Enter coordinates"):
            lat_col, lon_col = st.columns(2)
            with lat_col:
                lat_text = st.text_input("Latitude", key="explore_lat", placeholder="47.66150")
            with lon_col:
                lon_text = st.text_input("Longitude", key="explore_lon", placeholder="-122.31300")
            st.caption("When both are filled in, they are used instead of the address.")
        clicked = st.button("Search", type="primary", key="explore_search")

    radius_m = RADII_M[radius_label]
    entry = LocationInput(address, lat_text, lon_text)
    if clicked:
        spinner = "Finding the address…" if not entry.uses_coordinates else "Checking the coordinates…"
        with st.spinner(spinner):
            submit(st.session_state, RESULT_KEY, [entry], radius_m, as_of, geocoder())

    outcome = current(st.session_state, RESULT_KEY, as_of)
    if outcome is None:
        st.info("Enter a Seattle address and choose **Search** to see pending permits nearby.")
    elif not outcome.ok:
        st.error(outcome.error)
    else:
        pin = outcome.pins[0]
        if not outcome.matches([entry], radius_m):
            st.warning(
                "You changed the search but have not run it yet. "
                "The results below are still for the previous search. Choose **Search** to update them."
            )
        scope = f"within {outcome.radius_m:,} m"
        st.subheader(f"Pending permits {scope} of {pin.label}", anchor=False)
        st.caption(f"{located(pin)} Permit data through {long_date(outcome.data_as_of)}.")
        nearby = with_distance(active, pin.latitude, pin.longitude, outcome.radius_m)
        show_results(
            nearby,
            scope,
            "huskyquiet-nearby-permits",
            show_nearest=True,
            deck_for=lambda frame: location_deck(frame, pin.latitude, pin.longitude, outcome.radius_m),
            map_alt=f"Map of pending permits {scope} of the searched location",
        )
else:
    areas = load_areas()
    labels = [ALL_SEATTLE, *(area.label for area in areas)]
    choice = st.pills("Area", labels, default="University District", required=True, key="explore_area")
    mapped = active.loc[valid_coordinates(active)]
    if choice == ALL_SEATTLE:
        frame, scope, outline = mapped, "in Seattle", None
        latitude, longitude, zoom = UW_LAT, UW_LON, 11.5
        unmapped = len(active) - len(mapped)
        if unmapped == 1:
            st.caption("1 pending application has no usable coordinates and is left off the map.")
        elif unmapped:
            st.caption(f"{unmapped:,} pending applications have no usable coordinates and are left off the map.")
    else:
        area = area_by_label(choice)
        frame, scope, outline = permits_in_area(mapped, area), area_scope(area), area.geometry
        latitude, longitude, zoom = area.view
        st.caption(area.scope_note)
    st.subheader(f"Pending permits {scope}", anchor=False)
    show_results(
        frame,
        scope,
        "huskyquiet-area-permits",
        show_nearest=False,
        deck_for=lambda rows: area_deck(rows, latitude=latitude, longitude=longitude, zoom=zoom, outline=outline),
        map_alt=f"Map of pending permits {scope}",
    )
