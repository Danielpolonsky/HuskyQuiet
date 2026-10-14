# Running and deploying HuskyQuiet

The app reads three local inputs and does not train anything when it starts:

| Input | Path | In git? |
|---|---|---|
| Production model | `artifacts/models/production/model.joblib` | No. Joblib pickles are git-ignored. Load only a bundle you built. |
| Cleaned permits | `data/interim/permits_clean.parquet` | No. Raw and derived tables are git-ignored. |
| Evaluation reports | `reports/metrics.json`, `reports/test_month_calibration.json`, `reports/figures/reliability_test.png` | Yes. These are small and contain no credentials. |

Model metadata (`artifacts/models/production/metadata.json`) is committed so the app can show the data date and horizon even before you inspect the pickle. The pickle itself is required to score.

## Refresh locally

```powershell
python -m huskyquiet all                  # download, clean, audit, snapshots, train
python -m huskyquiet all --skip-download  # reuse the raw extract
python -m huskyquiet report-drift         # month-level calibration from the saved evaluation model
python -m streamlit run streamlit_app.py
```

Restart the app after a refresh. That drops the score cache. A normal browser refresh does not retrain and does not redownload.

## Address lookup

The app geocodes only when someone chooses **Search** or **Compare**. Reruns, filter changes, and page switches never send a lookup. Providers are tried in order:

1. **U.S. Census Geocoder** (`geocoding.geo.census.gov`). Free, no key, no account. It matches street addresses only, not place names. This is the default first provider because the public Nominatim server answered HTTP 429 to requests from Streamlit Community Cloud's shared network.
2. **OpenStreetMap Nominatim** (`nominatim.openstreetmap.org`). Free, no key. Its [usage policy](https://operations.osmfoundation.org/policies/nominatim/) allows at most one request per second per application, requires an identifying User-Agent, and expects results to be cached. It is used when Census finds no Seattle match, for example for a place name.

How requests are limited:

* One limiter per provider is shared by every session in the Streamlit process, with at least 1.0 s (Census) or 1.1 s (Nominatim) between requests. Concurrent lookups of the same address make one request.
* An HTTP 429 pauses that provider for the `Retry-After` time, or 60 s if none is given. The request is retried once only if the wait is 3 s or less. Server errors and dropped connections get one short retry. Timeouts (8 s) are not retried.
* Successful lookups are cached in memory and in `data/cache/geocode.json` (git-ignored). "Not found" answers are remembered for 10 minutes.
* Each failure says which kind it was: not found, rate limited, timed out, service unavailable, or unreadable response. Old results are cleared when a new search fails. Coordinates can always be typed under **Enter coordinates**. Nothing is substituted for a failed lookup.

None of this removes the providers' own limits. Both are shared public services and can still refuse or slow requests.

Optional settings, as environment variables or top-level Streamlit secrets:

| Setting | Default | Purpose |
|---|---|---|
| `HUSKYQUIET_GEOCODERS` | `census,nominatim` | Provider order. Use `census` or `nominatim` alone to pin one. |
| `HUSKYQUIET_NOMINATIM_URL` | `https://nominatim.openstreetmap.org/search` | Another Nominatim-compatible `/search` endpoint over HTTPS, such as your own Nominatim server. |
| `HUSKYQUIET_GEOCODER_EMAIL` | none | Contact address sent to Nominatim, as its policy recommends for heavier use. |

No key is needed for the defaults. If both free services prove unreliable for your traffic, run your own Nominatim instance (or use a Nominatim-compatible host that does not need a key in the URL) and set `HUSKYQUIET_NOMINATIM_URL` to it, with `HUSKYQUIET_GEOCODERS = "nominatim"`. Providers that need an API key are not supported, so no key is ever written into the source.

On Streamlit Community Cloud: open the app's **Settings → Secrets**, add lines such as

```toml
HUSKYQUIET_GEOCODERS = "census,nominatim"
HUSKYQUIET_GEOCODER_EMAIL = "you@example.org"
```

and save. The app restarts with the new settings.

## What a deployment host needs

Streamlit Community Cloud (or any other free host) builds from git, so it will not have the model or the permit table. Do not commit them. Two workable options:

1. **Build on the host only if you accept a long first boot.** `python -m huskyquiet all` downloads about 194,000 rows and trains for roughly 20 minutes. That exceeds a typical free-tier boot timeout. It is the right command for a machine you control, not for Community Cloud's first start.
2. **Ship a runtime zip you built yourself.** This is the path the app already supports.

```powershell
python -m huskyquiet bundle
```

That writes `dist/huskyquiet-runtime.zip` and `dist/huskyquiet-runtime.sha256`. The zip contains the production pickle, a column-subset of model-eligible permits, and the report files the How it works page reads. `dist/` is git-ignored.

Publish the zip as a GitHub Release asset, or any other HTTPS URL that does not require a secret baked into the source. On the host, set:

| Variable | Purpose |
|---|---|
| `HUSKYQUIET_BUNDLE_URL` | HTTPS URL of the zip |
| `HUSKYQUIET_BUNDLE_SHA256` | Hex digest from the `.sha256` file. Optional, but set it. A mismatch aborts the download. |

On startup the app downloads the zip only when the model or the cleaned table is missing, checks the digest, and extracts files that are not already present. It will not overwrite a model that is already on disk.

Community Cloud settings, when you are ready to deploy:

* Repository: this project, main branch.
* Main file: `streamlit_app.py`.
* Python: 3.11 or newer. The verified environment is 3.13.
* Secrets: `HUSKYQUIET_BUNDLE_URL` and `HUSKYQUIET_BUNDLE_SHA256` in the app's secret settings, not in git. There is no `.streamlit/secrets.toml` in the repository.
* Requirements: `requirements.txt`.

The public deployment at <https://huskyquiet.streamlit.app> uses this procedure, with the bundle URL and digest read from `deployment/bundle_source.json`.

## What not to commit

* `data/raw/`, `data/interim/`, `data/processed/`, and `*.parquet`
* `artifacts/**/*.joblib`
* `data/cache/` and `dist/`
* `.streamlit/secrets.toml`, `.env`, or a Socrata app token

The Socrata token is optional and only raises the open-data rate limit. Neither geocoder uses a key.
