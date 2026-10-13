"""Streamlit helpers shared by the app pages. Model math stays outside this module."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from huskyquiet.bundle import ensure_runtime
from huskyquiet.config import get_paths
from huskyquiet.predict import IssuancePredictor

_PLOTLY = {"displayModeBar": False}


def show_figure(figure, alt: str) -> None:
    st.plotly_chart(figure, width="stretch", config=_PLOTLY, alt=alt)


def status_caption(n_active: int | None = None) -> None:
    path = get_paths().model_dir / "production" / "metadata.json"
    if not path.exists():
        st.caption("Production model metadata is not on disk.")
        return
    meta = json.loads(path.read_text(encoding="utf-8"))
    parts = [
        f"Data through {meta.get('data_as_of', 'unknown')}",
        f"Production model {meta.get('model', 'unknown')} · {meta.get('calibration', 'unknown')} calibration",
        f"Horizon {meta.get('horizon_days', 90)} days",
    ]
    if n_active is not None:
        parts.append(f"{n_active:,} active applications scored")
    st.caption(" · ".join(parts))
    as_of = meta.get("data_as_of")
    if as_of:
        st.caption(
            f"Active applications are those still pending at the start of the day after {as_of}. "
            "Issuance after that date is not in this extract. Opening this app does not retrain the model "
            "or redownload the dataset."
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
    return _cached_geocoder()


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


def permit_table(frame: pd.DataFrame, *, key: str, alt: str, selectable: bool = False):
    show = pd.DataFrame({
        "Permit": frame["permit_num"].to_numpy(),
        "Address": frame["address_key"].fillna("Address not recorded").to_numpy()
        if "address_key" in frame.columns else "",
        "Class": frame["permit_class"].to_numpy(),
        "Category": frame["permit_type_desc"].fillna("Not recorded").to_numpy(),
        "Age (days)": frame["age_days"].to_numpy(),
        "Estimated cost": frame["est_project_cost"].to_numpy(),
        "P(issued in 90 days)": frame["probability_issued_90d"].to_numpy(),
    })
    if "distance_m" in frame.columns:
        show.insert(6, "Distance (m)", frame["distance_m"].round(0).to_numpy())
    kwargs = {"on_select": "rerun", "selection_mode": "single-row"} if selectable else {}
    return st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        key=key,
        column_config={
            "Estimated cost": st.column_config.NumberColumn(format="$%,.0f"),
            "P(issued in 90 days)": st.column_config.NumberColumn(format="%.1%", min_value=0, max_value=1),
            "Distance (m)": st.column_config.NumberColumn(format="%.0f"),
        },
        alt=alt,
        **kwargs,
    )


def render_summary_metrics(summary) -> None:
    with st.container(horizontal=True):
        st.metric(
            "Pending permits",
            f"{summary.n_permits:,}",
            help="Applications with usable coordinates in this scope. Permits without coordinates are omitted.",
            border=True,
        )
        st.metric(
            "Expected issuances in 90 days",
            f"{summary.expected_issuances_90d:.1f}",
            help="Sum of predicted issuance probabilities. An expected count of issuances, not of construction starts.",
            border=True,
        )
        if summary.nearest_m is not None:
            st.metric("Nearest permit", f"{summary.nearest_m:,.0f} m", border=True)
        median = "—" if summary.median_probability is None else f"{summary.median_probability:.1%}"
        st.metric("Median issuance probability", median, border=True)


def render_permit_detail(row: pd.Series) -> None:
    from huskyquiet.analysis import ISSUANCE_NOT_NOISE, explanatory_rows, format_cost, format_probability

    st.subheader(str(row.get("address_key") or "Address not recorded"))
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f"**{row['permit_num']}** · {row['permit_type_mapped']} · {row['permit_class']}"
            f" · {row.get('permit_type_desc') or 'Category not recorded'}"
        )
        applied = pd.Timestamp(row["applied_date"]).date()
        st.markdown(f"Applied {applied} · {int(row['age_days'])} days old · {format_cost(row.get('est_project_cost'))}")
        st.metric(
            "P(issued within 90 days)",
            format_probability(float(row["probability_issued_90d"])),
            help="Production-model probability for an application that is still pending.",
            border=True,
        )
        description = row.get("description")
        if description is None or pd.isna(description) or str(description).strip() == "":
            st.text("No description on file.")
        else:
            st.text(str(description))
        st.caption(
            "The description above is the City's text. The model does not read it directly. "
            "Administrative notes are stripped, and only a fixed list of scope keywords is used."
        )
    with right:
        st.markdown("**Model inputs behind this score**")
        explained = explanatory_rows(row)
        st.dataframe(
            pd.DataFrame(explained, columns=["Input", "Value"]),
            hide_index=True,
            width="stretch",
            alt="Model inputs for the selected permit",
        )
    st.caption(ISSUANCE_NOT_NOISE)


def parse_coord(text: str) -> float | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    return float(cleaned)


@st.cache_resource(show_spinner="Loading the issuance model…")
def _cached_predictor(model_mtime: float, clean_mtime: float) -> IssuancePredictor:
    del model_mtime, clean_mtime
    ensure_runtime()
    return IssuancePredictor()


@st.cache_data(show_spinner="Scoring active permit applications. The model is not retrained.")
def _cached_active(model_mtime: float, clean_mtime: float) -> pd.DataFrame:
    predictor = _cached_predictor(model_mtime, clean_mtime)
    return predictor.score_active_permits(return_features=True)


@st.cache_resource
def _cached_geocoder():
    from huskyquiet.geocode import NominatimGeocoder

    return NominatimGeocoder()


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
