"""Streamlit helpers shared by the app pages. Model math stays outside this module."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from huskyquiet.analysis import (
    EXPECTED_HELP,
    PROBABILITY_COLUMN,
    RAMP_HEX,
    LocationSummary,
    apply_filters,
    explanatory_rows,
    format_cost,
    format_distance,
    format_probability,
    permit_rows,
    permits_csv,
)
from huskyquiet.bundle import ensure_runtime
from huskyquiet.config import get_paths
from huskyquiet.predict import IssuancePredictor

_PLOTLY = {"displayModeBar": False}
PROSE_WIDTH = 760
TABLE_COLUMNS = ["Address", "Category", "Class", PROBABILITY_COLUMN, "Distance (m)", "Age (days)", "Permit number"]


def setting(name: str) -> str | None:
    """Environment variable first, then a top-level Streamlit secret."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        secret = st.secrets.get(name)
    except Exception:
        return None
    return None if secret in (None, "") else str(secret)


def prose():
    """A container narrow enough for comfortable reading."""
    return st.container(width=PROSE_WIDTH)


def show_figure(figure, alt: str) -> None:
    st.plotly_chart(figure, width="stretch", config=_PLOTLY, alt=alt)


def model_metadata() -> dict | None:
    path = get_paths().model_dir / "production" / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def data_as_of() -> str:
    meta = model_metadata() or {}
    return str(meta.get("data_as_of", "unknown"))


def long_date(text: str, days: int = 0) -> str:
    try:
        day = date.fromisoformat(text) + timedelta(days=days)
    except ValueError:
        return text
    return f"{day:%b} {day.day}, {day.year}"


def located(pin) -> str:
    where = f"{pin.latitude:.5f}, {pin.longitude:.5f}"
    if pin.matched:
        matched = pin.matched
        if len(matched) > 60:
            matched = ", ".join(matched.split(", ")[:3]) + "…"
        return f"Matched to “{matched}” by {pin.source}, at {where}."
    return f"{pin.source}: {where}."


def data_caption(n_active: int) -> None:
    as_of = data_as_of()
    st.caption(
        f"Permit data through {long_date(as_of)}. "
        f"{n_active:,} applications were still pending at the start of {long_date(as_of, 1)}."
    )


def load_active() -> pd.DataFrame:
    """Scored active permits, cached for the life of the Streamlit process."""
    try:
        model_mtime, clean_mtime = _mtimes()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    try:
        return _cached_active(model_mtime, clean_mtime)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Active permits could not be scored: {exc}")
        st.stop()


def load_metrics() -> dict:
    from huskyquiet.reporting import load_metrics as _load

    try:
        return _load()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(str(exc))
        st.stop()


def geocoder():
    return _cached_geocoder(
        setting("HUSKYQUIET_GEOCODERS") or "",
        setting("HUSKYQUIET_NOMINATIM_URL") or "",
        setting("HUSKYQUIET_GEOCODER_EMAIL") or "",
    )


def probability_legend() -> None:
    bands = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
        f'border:1px solid #9AA5A9;display:inline-block"></span>{band}</span>'
        for band, color in zip(bands, RAMP_HEX[:5], strict=True)
    )
    st.markdown(
        f'<div style="font-size:0.85rem;color:#3E4A50;line-height:1.8">'
        f"<b>Chance issued within 90 days</b> (darker is higher): {swatches}"
        '<span style="display:inline-flex;align-items:center;gap:4px">'
        '<span style="width:12px;height:12px;border-radius:50%;background:#C4A35A;display:inline-block">'
        "</span>Searched location or UW</span></div>",
        unsafe_allow_html=True,
    )


def render_summary_metrics(summary: LocationSummary, *, show_nearest: bool) -> None:
    with st.container(horizontal=True):
        st.metric("Pending permits", f"{summary.n_permits:,}", border=True)
        st.metric(
            "Expected permits issued within 90 days",
            f"{summary.expected_issuances_90d:.1f}",
            help=EXPECTED_HELP,
            border=True,
        )
        if show_nearest:
            st.metric("Nearest permit", format_distance(summary.nearest_m), border=True)


def render_details(summary: LocationSummary) -> None:
    """Secondary statistics, kept out of the main reading path."""
    with st.expander("More statistics"):
        if summary.n_permits == 0:
            st.caption("No permits in this selection.")
            return
        median = "—" if summary.median_probability is None else format_probability(summary.median_probability)
        st.markdown(
            f"- Median chance of issuance within 90 days: **{median}**\n"
            f"- Permits with at least a 50% chance: **{summary.n_likely:,} of {summary.n_permits:,}**"
        )
        left, right = st.columns(2)
        with left:
            st.dataframe(_count_table(summary.by_class, "Class"), hide_index=True, width="stretch",
                         alt="Pending permits by class")
        with right:
            st.dataframe(_count_table(summary.by_description, "Category"), hide_index=True, width="stretch",
                         alt="Pending permits by category")


def permit_filters(frame: pd.DataFrame, classes: list[str], key: str) -> tuple[pd.DataFrame, bool]:
    """Filter controls. Returns the filtered rows and whether any filter is active."""
    with st.expander("Filter permits"):
        chosen = st.multiselect(
            "Permit class",
            classes,
            default=classes,
            key=f"{key}_classes",
            placeholder="Choose at least one class",
        )
        minimum = st.slider(
            "Minimum chance of issuance within 90 days",
            0,
            100,
            0,
            5,
            format="%d%%",
            key=f"{key}_minimum",
        )
    all_classes = set(chosen) >= set(classes)
    active = not all_classes or minimum > 0
    filtered = apply_filters(frame, None if all_classes else chosen, minimum / 100)
    return filtered, active


def filter_caption(shown: int, total: int, scope: str, active: bool) -> None:
    if active:
        st.caption(
            f"Filters are on: showing {shown:,} of {total:,} pending permits {scope}. "
            "Every number, the map, the table, and the download use this filtered set."
        )


def permit_table(frame: pd.DataFrame, *, key: str, file_stem: str, alt: str):
    """Short sortable table of ``frame`` plus a CSV of the same rows."""
    rows = permit_rows(frame)
    columns = [column for column in TABLE_COLUMNS if column in rows.columns]
    state = st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * max(len(rows), 1), 390),
        key=key,
        column_order=columns,
        column_config={
            PROBABILITY_COLUMN: st.column_config.NumberColumn("Chance issued in 90 days", format="%.0f%%"),
            "Distance (m)": st.column_config.NumberColumn(format="%d m"),
            "Age (days)": st.column_config.NumberColumn(format="%d"),
        },
        on_select="rerun",
        selection_mode="single-row",
        alt=alt,
    )
    st.download_button(
        f"Download these {len(rows):,} permits (CSV)",
        data=permits_csv(frame),
        file_name=f"{file_stem}.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
        key=f"{key}_csv",
    )
    return state


def remember_selection(key: str, frame: pd.DataFrame, map_state, table_state) -> str | None:
    """The permit most recently picked on the map or in the table, if it is still in ``frame``."""
    store = st.session_state.setdefault(f"{key}_picks", {"map": None, "table": None, "chosen": None})
    map_id = permit_num_from_pydeck(map_state)
    row = selected_row_index(table_state)
    table_id = None
    if row is not None and 0 <= row < len(frame):
        table_id = frame.iloc[row]["permit_num"]
    if map_id != store["map"]:
        store["map"] = map_id
        if map_id:
            store["chosen"] = map_id
    if table_id != store["table"]:
        store["table"] = table_id
        if table_id:
            store["chosen"] = table_id
    chosen = store["chosen"]
    if chosen is None or chosen not in set(frame["permit_num"]):
        return None
    return chosen


def permit_num_from_pydeck(state) -> str | None:
    selection = _selection(state)
    if selection is None:
        return None
    objects = selection.get("objects") if isinstance(selection, dict) else getattr(selection, "objects", None)
    if not isinstance(objects, dict):
        return None
    rows = objects.get("permits") or []
    if not rows:
        return None
    first = rows[0]
    if isinstance(first, dict):
        return first.get("permit_num")
    return None


def selected_row_index(state) -> int | None:
    selection = _selection(state)
    if selection is None:
        return None
    rows = selection.get("rows") if isinstance(selection, dict) else getattr(selection, "rows", None)
    if not rows:
        return None
    return int(rows[0])


def render_permit_detail(row: pd.Series) -> None:
    with st.container(border=True):
        st.markdown(f"**{row.get('address_key') or 'Address not recorded'}**")
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.markdown(
                f"{row['permit_num']} · {row['permit_type_mapped']} · {row['permit_class']}"
                f" · {row.get('permit_type_desc') or 'Category not recorded'}"
            )
            applied = pd.Timestamp(row["applied_date"]).date()
            st.caption(
                f"Applied {applied:%b} {applied.day}, {applied.year} · {int(row['age_days'])} days pending · "
                f"estimated cost {format_cost(row.get('est_project_cost'))}"
            )
            st.metric(
                "Chance issued within 90 days",
                format_probability(float(row["probability_issued_90d"])),
                border=True,
            )
            description = row.get("description")
            if description is None or pd.isna(description) or str(description).strip() == "":
                st.text("No description on file.")
            else:
                st.text(str(description))
        with right:
            st.markdown("**What the model looked at**")
            st.dataframe(
                pd.DataFrame(explanatory_rows(row), columns=["Input", "Value"]),
                hide_index=True,
                width="stretch",
                alt="Model inputs for the selected permit",
            )


def selection_prompt() -> None:
    st.caption("Select a permit on the map or in the table to see its details.")


def _count_table(counts: dict[str, int], label: str) -> pd.DataFrame:
    return pd.DataFrame(sorted(counts.items(), key=lambda item: (-item[1], item[0])), columns=[label, "Permits"])


@st.cache_resource(show_spinner="Loading the issuance model…")
def _cached_predictor(model_mtime: float, clean_mtime: float) -> IssuancePredictor:
    del model_mtime, clean_mtime
    ensure_runtime()
    return IssuancePredictor()


@st.cache_data(show_spinner="Scoring pending permit applications. The model is not retrained.")
def _cached_active(model_mtime: float, clean_mtime: float) -> pd.DataFrame:
    predictor = _cached_predictor(model_mtime, clean_mtime)
    return predictor.score_active_permits(return_features=True)


@st.cache_resource
def _cached_geocoder(providers: str, nominatim_url: str, email: str):
    from huskyquiet.geocode import Geocoder, providers_from_settings

    values = {
        "HUSKYQUIET_GEOCODERS": providers,
        "HUSKYQUIET_NOMINATIM_URL": nominatim_url,
        "HUSKYQUIET_GEOCODER_EMAIL": email,
    }
    return Geocoder(providers_from_settings(lambda name: values.get(name) or None))


def _mtimes() -> tuple[float, float]:
    paths = get_paths()
    model = paths.model_dir / "production" / "model.joblib"
    if not model.exists() or not paths.clean_permits.exists():
        ensure_runtime(paths)
    if not model.exists() or not paths.clean_permits.exists():
        raise FileNotFoundError(
            "The production model or cleaned permits are still missing after the runtime check."
        )
    return model.stat().st_mtime, paths.clean_permits.stat().st_mtime


def _selection(state):
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get("selection")
    return getattr(state, "selection", None)
