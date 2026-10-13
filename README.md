# HuskyQuiet

Predictive analytics for University of Washington students and Seattle renters who want
to know what construction-permit activity is pending near an apartment.

**Milestone 1 (this repository state):** a reproducible data pipeline over the official
[Seattle Building Permits](https://data.seattle.gov/Built-Environment/Building-Permits/76t5-zqzr)
dataset (`76t5-zqzr`), a leakage-audited prediction problem, calibrated baseline and ML
models, and a validated Python prediction interface. The dashboard/map is Milestone 2.

> **What the model predicts.** The probability that the City *issues* a pending permit
> within the next 90 days. Permit issuance is not construction start and is not noise.
> HuskyQuiet never presents these probabilities as noise predictions.

## Quick start (Windows PowerShell)

```powershell
cd C:\path\to\HuskyQuiet
py -3.13 -m venv .venv            # Python 3.11+ works; 3.13.2 was used for verification
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt   # exact pinned versions
pip install -e . --no-deps        # installs the `huskyquiet` package from src/

python -m huskyquiet all          # download -> clean -> audit -> snapshots -> train (~20 min)
python -m pytest                  # 66 tests (integration tests use the built artifacts)
python -m pytest -m "not integration"   # 61 tests, no model files required
```

Individual stages:

```powershell
python -m huskyquiet download     # ~1.5 min, 193k rows from the SODA API -> data/raw/*.parquet
python -m huskyquiet clean        # typing, validation, dedup -> data/interim/permits_clean.parquet
python -m huskyquiet audit        # data-quality + leakage diagnostics -> reports/data_audit_stats.*
python -m huskyquiet snapshots    # forecast snapshots + features -> data/processed/snapshots_h90.parquet
python -m huskyquiet train        # model selection, calibration, test eval, backtest, production refit
python -m huskyquiet --log-file reports\pipeline_log.txt all --skip-download   # reuse raw data
```

An optional free Socrata app token (`$env:SOCRATA_APP_TOKEN = "..."`) raises API rate
limits. No paid services are used.

## Web application

```powershell
python -m streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The app scores the permits
already on disk. It does not retrain or redownload. Pages:

* **Explore map** — pending applications at their recorded coordinates, colored by predicted issuance probability, with University District, University Village, Roosevelt, Wallingford, and Ravenna filters.
* **Analyze address** — geocode a Seattle address (or type coordinates) and summarize permits within 250 m, 500 m, or 1 km.
* **Compare apartments** — the same summary for two addresses, with a paragraph computed from the counts.
* **Model performance** — ROC-AUC, PR-AUC, Brier, calibration, backtest, and feature importance read from `reports/metrics.json`.
* **About** — what the number means, how to refresh the data, and where the area boundaries come from.

"Expected issuances" is the sum of predicted issuance probabilities. It is not a noise forecast and not a count of construction starts. Refresh and deployment are documented in [`docs/deployment.md`](docs/deployment.md).

## Using the model

```python
from datetime import date
from huskyquiet.predict import IssuancePredictor, PermitApplication

predictor = IssuancePredictor()          # artifacts/models/production + cleaned history
result = predictor.predict(
    PermitApplication(
        applied_date=date(2026, 8, 20),
        permit_type_mapped="Building",
        permit_class="Multifamily",
        permit_type_desc="New",
        description="Construct new 6-story apartment building with ground floor retail, per plan.",
        est_project_cost=12_500_000,
        latitude=47.6612, longitude=-122.3134,
        address="4500 UNIVERSITY WAY NE",
    ),
    as_of=date(2026, 10, 10),
)
print(result.probability, result.warnings, result.interpretation)

active = predictor.score_active_permits()   # every pending application, scored (for the map)
```

Inputs are validated with pydantic. Outcome fields are rejected, coordinates must be in
Seattle, and costs must be non-negative. Out-of-scope requests raise `PredictionInputError`:
`as_of` on or before the application date, applications older than 730 days, already-issued
permits, unknown permit types, or permit data older than 14 days.

## Repository layout

```
src/huskyquiet/
  config.py                 paths, dataset ids, horizon, age cap, seeds
  cli.py, __main__.py       `python -m huskyquiet <stage>`
  data/schema.py            verified API schema + per-field availability classification
  data/ingest.py            paginated, validated SODA download -> Parquet + metadata
  data/clean.py             typing, sentinel/invalid handling, dedup, quality flags
  data/audit.py             reproducible data-quality & leakage diagnostics
  features/snapshots.py     risk sets, labels, forecast-date grid (DuckDB)
  features/engineering.py   point-in-time features shared by training and inference
  features/text.py          description sanitization + scope keyword flags
  models/baseline.py        historical-rate and recent-rate baselines
  models/pipelines.py       logistic regression and HistGradientBoosting pipelines
  models/calibration.py     Platt / isotonic calibration with grouped-CV selection
  models/evaluate.py        metrics, ECE, reliability, clustered bootstrap, permutation importance
  models/train.py           temporal split, selection, evaluation, backtest, persistence
  models/registry.py        model bundle save/load
  predict.py                validated prediction interface
tests/                      pytest suite (leakage property tests, boundary tests, API tests)
docs/data_audit.md          what the data contains and what is/isn't usable
docs/modeling_methodology.md problem definition, splits, features, models, results
reports/                    generated metrics, audit stats, figures (committed; small)
artifacts/models/           model bundles (*.joblib, git-ignored) + metadata.json
data/                       raw/interim/processed Parquet (git-ignored)
```

## Results

Test year 2025-08 … 2026-07 (32,617 forecast snapshots, 6,952 applications, 30.0% issued
within 90 days). The data extract is as of 2026-10-10:

| Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|
| Historical-rate baseline (type × class × age) | 0.686 | 0.437 | 0.1909 | 0.042 |
| Logistic regression + isotonic | 0.712 | 0.475 | 0.1863 | 0.025 |
| HistGradientBoosting + Platt (production) | 0.765 | 0.562 | 0.1720 | 0.011 |

The bootstrap 95% CI for HGB's ROC-AUC gain over the baseline is [0.072, 0.087]. Details,
slices, ablations, the rolling backtest, and caveats are in
[`docs/modeling_methodology.md`](docs/modeling_methodology.md#5-results-from-the-current-run).
Full machine-readable results are in `reports/metrics.json`, with a readable summary in `reports/model_report.md`.

## Documentation

* [`docs/data_audit.md`](docs/data_audit.md): schema, coverage, quality issues, and the
  evidence behind every included or excluded field.
* [`docs/modeling_methodology.md`](docs/modeling_methodology.md): prediction target,
  snapshot construction, leakage controls, temporal split with embargo, models,
  calibration, evaluation, and results.

## Notes

* Raw and derived datasets are never committed. They are reproducible from the API, and the
  raw extract's SHA-256 is recorded in `data/raw/download_metadata.json`. The portal reloads
  daily, so a fresh download is a newer snapshot and the metrics will shift slightly.
* Model bundles are joblib pickles. Load only artifacts you built yourself.
* Keep the project outside OneDrive or other synced folders. Syncing `.venv/`, `data/` and
  `artifacts/` slows training substantially and can interrupt long runs.
