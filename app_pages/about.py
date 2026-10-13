import streamlit as st

from huskyquiet import __version__
from huskyquiet.areas import load_areas
from huskyquiet.ui import status_caption

st.header("About")
status_caption()
st.markdown(
    f"""
**HuskyQuiet** (version {__version__}) is a permit-issuance forecast for people comparing housing near the
University of Washington. It answers a narrow question: if a Seattle building-permit application is still
pending, what is the probability the City issues it within 90 days?

Issuance is a paperwork milestone. It is not the day construction starts, and it is not a measurement of
noise. The map, the address search, and the comparison all use that same probability. Nearby "expected
issuances" are the sum of those probabilities. They are a proxy for how much permitted work is likely to
clear review, not a forecast that a residence will be disturbed.

The data is the City of Seattle's Building Permits dataset
([`76t5-zqzr`](https://data.seattle.gov/Built-Environment/Building-Permits/76t5-zqzr)).
The modeling write-up is in `docs/modeling_methodology.md` and the field-by-field audit is in
`docs/data_audit.md`.
"""
)

st.subheader("Areas on the map")
st.markdown(
    "Neighborhood extents come from the Seattle City Clerk Neighborhood Map Atlas, which the City describes "
    "as an unofficial indexing boundary. University Village is the OpenStreetMap footprint of the shopping "
    "center. Neither is a risk score. Filtering to an area only counts the pending applications whose "
    "recorded coordinates fall inside it."
)
for area in load_areas():
    st.markdown(f"**{area.name}.** {area.note} Source: {area.source}.")

st.subheader("Refresh the data and the scores")
st.markdown(
    """
From the project directory, with the virtual environment active:

```
python -m huskyquiet all
```

That downloads the current portal extract, cleans it, rebuilds snapshots, and retrains. To keep the raw
file you already have and refresh everything after it:

```
python -m huskyquiet all --skip-download
```

Then restart this app, or use the button below. Restarting is what drops the in-memory score cache.
The app will not download or retrain on its own.

A lighter check of month-by-month calibration, using the already trained evaluation model:

```
python -m huskyquiet report-drift
```

Deployment, including how to ship the git-ignored model without committing it, is in `docs/deployment.md`.
"""
)
if st.button("Reload scored permits", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.subheader("What can fail")
st.markdown(
    """
- If the model files are missing, the map pages stop with the command that builds them.
  They do not invent permits or scores.
- If geocoding is down, rate-limited, or cannot find the address, the form asks for coordinates.
  A failed lookup is not replaced with a guessed point.
- Permits without coordinates stay in the scored count and stay off the map.
- An address outside the Seattle bounding box is rejected.
"""
)
