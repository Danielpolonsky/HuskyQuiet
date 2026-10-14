import pandas as pd
import streamlit as st

from huskyquiet.analysis import difference_points, summarize_permits
from huskyquiet.areas import area_by_label, area_scope, load_areas, permits_in_area
from huskyquiet.charts import probability_histogram
from huskyquiet.geo import valid_coordinates, with_distance
from huskyquiet.mapdeck import area_deck, location_deck
from huskyquiet.search import RADII_M, LocationInput, current, same_location, submit
from huskyquiet.ui import (
    data_as_of,
    data_caption,
    geocoder,
    load_active,
    located,
    long_date,
    probability_legend,
    prose,
    render_summary_metrics,
    show_figure,
)

RESULT_KEY = "compare_result"
NAMES = ("Location A", "Location B")

st.header("Compare locations", anchor=False)
with prose():
    st.markdown(
        "Put two places side by side. Both use the same radius or the same kind of area, "
        "so the numbers can be compared directly."
    )
    st.caption(
        "Differences are counts and predicted chances of permit issuance. "
        "They do not say which place is quieter or better to live in."
    )

active = load_active()
as_of = data_as_of()
data_caption(len(active))

mode = st.segmented_control(
    "Compare", ["Two addresses", "Two areas"], default="Two addresses", required=True, key="compare_mode"
)


def panels(summaries, decks, titles, captions, *, show_nearest: bool) -> None:
    columns = st.columns(2, gap="medium")
    for column, summary, deck, title, caption, index in zip(
        columns, summaries, decks, titles, captions, range(2), strict=True
    ):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            render_summary_metrics(summary, show_nearest=show_nearest)
            st.pydeck_chart(deck, height=340, key=f"compare_map_{index}", alt=f"Map for {title}")
            st.caption(caption)
    probability_legend()


def differences(left, right, frames, identical: bool, names=NAMES) -> None:
    st.subheader("What differs", anchor=False)
    with prose():
        for point in difference_points(left, right, names[0], names[1], identical=identical):
            st.markdown(f"- {point}")
    with st.expander("Distributions and categories"):
        if left.n_permits or right.n_permits:
            show_figure(
                probability_histogram(
                    {NAMES[0]: frames[0]["probability_issued_90d"], NAMES[1]: frames[1]["probability_issued_90d"]},
                    "Pending permits by predicted chance of issuance within 90 days",
                ),
                "Grouped histogram of predicted issuance chances for both locations",
            )
        categories = sorted(set(left.by_description) | set(right.by_description))
        table = pd.DataFrame({
            "Category": categories,
            NAMES[0]: [left.by_description.get(name, 0) for name in categories],
            NAMES[1]: [right.by_description.get(name, 0) for name in categories],
        }).sort_values([NAMES[0], NAMES[1]], ascending=False)
        st.dataframe(table, hide_index=True, width="stretch", alt="Pending permits by category in both locations")


if mode == "Two addresses":
    radius_label = st.segmented_control(
        "Search radius for both", list(RADII_M), default="500 m", required=True, key="compare_radius"
    )
    entries = []
    for column, name, index in zip(st.columns(2, gap="medium"), NAMES, range(2), strict=True):
        with column, st.container(border=True):
            address = st.text_input(
                f"{name} address",
                key=f"compare_address_{index}",
                placeholder="4500 University Way NE" if index == 0 else "1200 NE 45th St",
                max_chars=100,
            )
            with st.expander("Enter coordinates"):
                lat_col, lon_col = st.columns(2)
                with lat_col:
                    lat_text = st.text_input("Latitude", key=f"compare_lat_{index}", placeholder="47.66150")
                with lon_col:
                    lon_text = st.text_input("Longitude", key=f"compare_lon_{index}", placeholder="-122.31300")
                st.caption("When both are filled in, they are used instead of the address.")
            entries.append(LocationInput(address, lat_text, lon_text))
    clicked = st.button("Compare", type="primary", key="compare_submit")

    radius_m = RADII_M[radius_label]
    if clicked:
        with st.spinner("Finding both locations…"):
            submit(st.session_state, RESULT_KEY, entries, radius_m, as_of, geocoder(), names=NAMES)

    outcome = current(st.session_state, RESULT_KEY, as_of)
    if outcome is None:
        st.info("Enter two Seattle addresses and choose **Compare**.")
    elif not outcome.ok:
        st.error(outcome.error)
    else:
        if not outcome.matches(entries, radius_m):
            st.warning(
                "You changed the inputs but have not compared them yet. "
                "The results below are still for the previous comparison. Choose **Compare** to update them."
            )
        scope = f"within {outcome.radius_m:,} m"
        st.subheader(f"Pending permits {scope} of each location", anchor=False)
        st.caption(f"Permit data through {long_date(outcome.data_as_of)}. No permit filters apply here.")
        frames = [with_distance(active, pin.latitude, pin.longitude, outcome.radius_m) for pin in outcome.pins]
        summaries = [
            summarize_permits(frame, pin.label, scope) for frame, pin in zip(frames, outcome.pins, strict=True)
        ]
        identical = same_location(*outcome.pins)
        if identical:
            st.info("Both entries point to the same place, so the two panels are identical.")
        panels(
            summaries,
            [
                location_deck(frame, pin.latitude, pin.longitude, outcome.radius_m, zoom_offset=-0.5)
                for frame, pin in zip(frames, outcome.pins, strict=True)
            ],
            [f"{name}: {pin.label}" for name, pin in zip(NAMES, outcome.pins, strict=True)],
            [located(pin) for pin in outcome.pins],
            show_nearest=True,
        )
        differences(summaries[0], summaries[1], frames, identical)
else:
    areas = load_areas()
    labels = [area.label for area in areas]
    columns = st.columns(2, gap="medium")
    choices = []
    for column, name, default in zip(columns, NAMES, ("University District", "Wallingford"), strict=True):
        with column:
            choices.append(st.selectbox(name, labels, index=labels.index(default), key=f"compare_area_{name}"))
    chosen = [area_by_label(label) for label in choices]
    mapped = active.loc[valid_coordinates(active)]
    frames = [permits_in_area(mapped, area) for area in chosen]
    summaries = [
        summarize_permits(frame, area.label, area_scope(area), distance_col=None)
        for frame, area in zip(frames, chosen, strict=True)
    ]
    st.caption(f"Permit data through {long_date(as_of)}. No permit filters apply here: counts include every class.")
    if {area.name for area in chosen} == {"University Village", "Ravenna"}:
        st.caption(
            "University Village lies inside the Ravenna extent, so permits inside the shopping center "
            "are counted on both sides."
        )
    identical = chosen[0].name == chosen[1].name
    if identical:
        st.info("You picked the same area twice, so the two panels are identical.")
    panels(
        summaries,
        [
            area_deck(frame, latitude=area.view[0], longitude=area.view[1], zoom=area.view[2], outline=area.geometry)
            for frame, area in zip(frames, chosen, strict=True)
        ],
        [f"{name}: {area.label}" for name, area in zip(NAMES, chosen, strict=True)],
        [area.scope_note for area in chosen],
        show_nearest=False,
    )
    differences(summaries[0], summaries[1], frames, identical, names=NAMES if identical else tuple(choices))
