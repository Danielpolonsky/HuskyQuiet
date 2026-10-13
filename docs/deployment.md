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

Restart the app after a refresh, or use **Reload scored permits** on the About page. Either one drops the score cache. A normal browser refresh does not retrain and does not redownload.

Geocoding uses the public Nominatim service. Results are cached in `data/cache/geocode.json` (git-ignored). The client waits at least 1.1 seconds between network calls and sends an identifying User-Agent. If Nominatim fails, the address forms accept coordinates instead. Nothing is substituted for a failed lookup.

## What a deployment host needs

Streamlit Community Cloud (or any other free host) builds from git, so it will not have the model or the permit table. Do not commit them. Two workable options:

1. **Build on the host only if you accept a long first boot.** `python -m huskyquiet all` downloads about 194,000 rows and trains for roughly 20 minutes. That exceeds a typical free-tier boot timeout. It is the right command for a machine you control, not for Community Cloud's first start.
2. **Ship a runtime zip you built yourself.** This is the path the app already supports.

```powershell
python -m huskyquiet bundle
```

That writes `dist/huskyquiet-runtime.zip` and `dist/huskyquiet-runtime.sha256`. The zip contains the production pickle, a column-subset of model-eligible permits, and the report files the Model performance page reads. `dist/` is git-ignored.

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

This document does not mean the app is deployed. It is the procedure for deploying it.

## What not to commit

* `data/raw/`, `data/interim/`, `data/processed/`, and `*.parquet`
* `artifacts/**/*.joblib`
* `data/cache/` and `dist/`
* `.streamlit/secrets.toml`, `.env`, or a Socrata app token

The Socrata token is optional and only raises the open-data rate limit. Nominatim does not use a key.
