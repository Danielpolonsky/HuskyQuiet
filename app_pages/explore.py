import streamlit as st

from huskyquiet.analysis import ISSUANCE_NOT_NOISE, comparison_text, summarize_permits
from huskyquiet.areas import area_by_name, area_scope, load_areas, permits_in_area
from huskyquiet.charts import category_bars, legend_figure
from huskyquiet.config import UW_LAT, UW_LON
from huskyquiet.geo import valid_coordinates
from huskyquiet.mapdeck import city_deck
from huskyquiet.ui import (
    load_active,
    permit_num_from_pydeck,
    permit_table,
    render_permit_detail,
    render_summary_metrics,
    selected_row_index,
    show_figure,
    status_caption,
)

st.header("Explore map")
st.caption(
    "Each point is a real pending Seattle permit at its recorded coordinate. "
    "Color is the production model's predicted probability that the City issues it within 90 days. "
    "Darker teal means a higher issuance probability. It is not a noise scale and not a construction forecast."
)

active = load_active()
status_caption(len(active))
names = [area.name for area in load_areas()]
area_name = st.pills(
    "Area",
    ["All mapped", *names],
    default="University District",
    key="explore_area",
    width="stretch",
    wrap=True,
)
area_name = area_name or "University District"
classes = sorted(active["permit_class"].dropna().unique())
chosen_classes = st.multiselect("Permit class", classes, default=classes, placeholder="Choose classes")
minimum = st.slider("Minimum issuance probability", 0.0, 1.0, 0.0, 0.05)

view = active
if area_name != "All mapped":
    area = area_by_name(area_name)
    view = permits_in_area(view, area)
    latitude, longitude, zoom = area.view
    st.caption(area.note)
else:
    latitude, longitude, zoom = UW_LAT, UW_LON, 12
view = view.loc[view["permit_class"].isin(chosen_classes)] if chosen_classes else view.iloc[0:0]
view = view.loc[view["probability_issued_90d"] >= minimum]
mapped = view.loc[valid_coordinates(view)]
unmapped = int((~valid_coordinates(active)).sum())

summary = summarize_permits(
    mapped,
    label=area_name,
    scope="on this map",
    distance_col=None,
)
render_summary_metrics(summary)
if unmapped == 1:
    st.caption("1 active application has no usable Seattle coordinates, so it is not on the map.")
elif unmapped:
    st.caption(
        f"{unmapped:,} active applications have no usable Seattle coordinates, so they are not on the map."
    )

if mapped.empty:
    st.info("No mapped permits match these filters.")
else:
    selection = st.pydeck_chart(
        city_deck(mapped, latitude=latitude, longitude=longitude, zoom=zoom),
        height=680,
        on_select="rerun",
        selection_mode="single-object",
        key="city-map",
        alt="Map of pending Seattle permits colored by 90-day issuance probability",
    )
    show_figure(legend_figure(), "Legend for issuance-probability colors")
    map_id = permit_num_from_pydeck(selection)
    if map_id:
        st.session_state["selected_permit"] = map_id

    st.subheader("Permits in view")
    table_state = permit_table(
        mapped.reset_index(drop=True),
        key="explore-table",
        alt="Pending permits in the current map view",
        selectable=True,
    )
    row_index = selected_row_index(table_state)
    if row_index is not None:
        st.session_state["selected_permit"] = mapped.reset_index(drop=True).iloc[row_index]["permit_num"]

selected_id = st.session_state.get("selected_permit")
if selected_id:
    matches = active.loc[active["permit_num"] == selected_id]
    if not matches.empty:
        render_permit_detail(matches.iloc[0])

st.subheader("Compare two areas")
st.caption(
    "Counts below are pending applications inside each extent. "
    "They are not neighborhood risk scores."
)
left_name, right_name = st.columns(2)
with left_name:
    left_choice = st.selectbox("First area", names, index=names.index("University District"), key="area_left")
with right_name:
    right_choice = st.selectbox("Second area", names, index=names.index("Wallingford"), key="area_right")
left_area, right_area = area_by_name(left_choice), area_by_name(right_choice)
left_frame = permits_in_area(active, left_area)
right_frame = permits_in_area(active, right_area)
left_summary = summarize_permits(left_frame, left_area.name, area_scope(left_area), distance_col=None)
right_summary = summarize_permits(right_frame, right_area.name, area_scope(right_area), distance_col=None)
metric_cols = st.columns(2)
with metric_cols[0]:
    st.markdown(f"**{left_area.name}**")
    render_summary_metrics(left_summary)
with metric_cols[1]:
    st.markdown(f"**{right_area.name}**")
    render_summary_metrics(right_summary)
st.markdown(comparison_text(left_summary, right_summary))
if left_summary.n_permits or right_summary.n_permits:
    show_figure(
        category_bars(left_area.name, left_summary.by_description, right_area.name, right_summary.by_description),
        "Permit description categories in the two areas",
    )
st.caption(ISSUANCE_NOT_NOISE)
