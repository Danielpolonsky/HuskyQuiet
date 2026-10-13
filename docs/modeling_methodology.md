# Modeling methodology

## 1. Prediction problem

> Given an application that is **active at the start of forecast date _t_**, and only
> information available before _t_, estimate
> **P(the permit is issued in [t, t + 90 days))**.

This is a permit-**issuance** model. Issuance is a regulatory milestone. It does not mean
construction has started, that it will start soon, or that it will be loud. HuskyQuiet
never presents this probability as a noise prediction.

### Unit of analysis: forecast snapshots

A snapshot is one `(permit, t)` pair. Forecast dates are the first day of every month
from 2008-01-01 to the last date with a fully observed label window
(`t + 90 ≤ data_as_of`; with the current extract, `data_as_of` = 2026-10-10 and the
last forecast date is 2026-07-01). That gives 223 forecast dates.

| Rule | Definition | Rationale |
|---|---|---|
| Entered the risk set | `applied_date < t` | Events dated on day *t* are not yet known at the start of *t* |
| Not yet issued | `issued_date IS NULL OR issued_date ≥ t` | Only pending applications are scored |
| Age cap | `t − applied_date ≤ 730 days` | 98.6% of issuances happen within 730 days. Without a cap, abandoned applications would stay "active" forever |
| Label | `1[t ≤ issued_date < t + 90]` | Never-issued applications (withdrawn, cancelled, still pending) are negatives |
| Complete window | `t + 90 ≤ data_as_of` | No right-censored labels |
| Eligibility | non-null `applied_date`, and `issued_date ≥ applied_date` when present | See `data_audit.md` |

**Why not use status?** `statuscurrent` is today's value, and the dataset never records when a
withdrawal or cancellation happened. Using status to decide who was active in 2015 would
leak 2026 knowledge into the past. The age cap is the leakage-free alternative. Its cost is
that an application withdrawn early stays in the risk set as a negative until it reaches 730
days. That is a correct negative label for each 90-day window, but it inflates the number of
negatives among old applications.

**Same-day issuance.** 31.6% of permits (mostly "subject to field inspection") are issued
the day they are applied for. They never appear in a risk set, which is correct: they were never
pending at the start of any day.

### Repeated observations

A permit pending for _k_ months contributes up to _k_ snapshots (6.4 on average). The design
accounts for this in four ways:

* **Splits are by time, not by row.** A permit can appear in both the training and test
  periods, but its training labels end at or before the training cutoff (see the embargo below).
  So the training data only "knows" that the permit was still pending at the cutoff,
  which was genuinely known then.
* **Uncertainty uses a permit-clustered bootstrap.** Whole permits are resampled with
  replacement (200 replicates), so the confidence intervals reflect within-permit correlation.
  `run_training` passes `permit_num` into `cluster_bootstrap`, which draws permits from a
  multinomial and weights every snapshot of a drawn permit by that permit's draw count.
  The intervals in `reports/metrics.json` are this clustered bootstrap. A later check confirmed
  the implementation; the intervals were not recomputed and the metrics are unchanged.
* **Metrics are reported two ways.** Over all test snapshots (the operational view: every
  forecast the product would make), and over each permit's **first** test-period snapshot
  only (one row per permit, so long-pending permits do not dominate). Weighting each permit by
  `1/k` was rejected because *k*, the number of snapshots, depends on the outcome: a permit
  issued quickly has fewer snapshots. That would down-weight exactly the slow, mostly negative permits.
* **Calibration selection uses grouped folds.** `GroupKFold` by permit inside the calibration period.

## 2. Temporal split with embargo

```
train:       t + 90 ≤ 2024-08-01          (forecast dates 2008-01 … 2024-05)
validation:  2024-08-01 ≤ t, t + 90 ≤ 2025-08-01   (2024-08 … 2025-05)
test:        2025-08-01 ≤ t ≤ 2026-07-01           (12 forecast dates)
```

Every stored snapshot, including the test period, satisfies `t + 90 days ≤ data_as_of`
(2026-10-10). The latest test forecast is 2026-07-01, so its label window ends on 2026-09-29.
No held-out label is right-censored.

The 90-day embargo before each boundary ensures that no training label depends on events
inside the validation period, and no validation label depends on events inside the test period. The split
boundaries are derived from the last labelled forecast date (12-month test period, 12-month
validation period), so they move forward automatically when the data is refreshed. They are
recorded in `reports/metrics.json`.

* **Validation is used for:** the logistic-regression C grid, the HGB hyperparameter grid, the
  calibration method choice (none / Platt / isotonic, by permit-grouped CV log loss),
  fitting the calibrator, and choosing the model family.
* **Test is used for:** a single scoring of each final model, plus post-hoc analyses
  (slices, ablations, permutation importance) that feed no decision.

### Production model

The deployed model reruns the same procedure shifted forward one period. It is fit on all
snapshots with `t + 90 ≤ test_start`, and its calibrator (same method) is fit on the
test-period snapshots. Its expected performance is the evaluated test performance of the
procedure. The evaluation-run models are saved too, under `artifacts/models/evaluation/`.

## 3. Features

All features are computed by `huskyquiet.features.engineering.build_feature_frame`. The
same function builds the training snapshots and scores live requests, so there is no
train/serve skew.

**Application-time attributes:**
* Categorical: `permit_class`, `permit_type_mapped`, `permit_type_desc`, the permit-number suffix
  (`CN`, `DM`, `BK`, `GR`, `CC`, `PH`), and `zip`.
* Numeric: `log(1 + est_project_cost)` (clipped at $5B), a cost-missing flag, a has-related-MUP flag, latitude and longitude, distance to UW, and the
  application month.
* 15 project-scope keyword flags from the **sanitized** description, plus a
  description-missing flag. The raw text, tokens, and length are never used.

**Time-varying (as of t):** application age in days (raw and log), and the forecast month.

**Point-in-time context**, computed only from events dated before *t*:

| Feature | Definition |
|---|---|
| `ctx_backlog_all`, `ctx_backlog_grp` | Size of the risk set at *t*, city-wide and within the permit type × class group |
| `ctx_applied_90d_grp`, `ctx_issued_90d_grp` | Applications and issuances in the group during [t−90, t) |
| `ctx_issue_rate_lag_all/_grp/_age` | Share of the risk set at *t − 90* that was issued in [t−90, t): city-wide, by group, and by age bucket |
| `ctx_median_days_to_issue_365d_grp` | Median applied-to-issued days for group permits issued in [t−365, t) |
| `addr_prior_apps`, `addr_prior_issued_365d` | Other applications at the same address before *t*, and issuances there in [t−365, t) |

The lagged-rate features describe a window that ends exactly at *t*. That window is fully
observed at *t*, so this is not target leakage.

**Excluded**, with the evidence in `data_audit.md` §4:
* The issued, expires, and completed dates, and `statuscurrent`.
* All review-cycle metrics and review dates, including `readytoissuedate`.
* `housingunits`, `housingunitsadded`, and `housingunitsremoved`. They are filled in during plan review: for
  recent new single-family applications, the field is null for 49.7% of pending ones vs 9.3% of issued ones.
* The contractor name, `parentpermitnum`, `housingcategory`, `zoning`, `dwellingunittype`,
  `standardplan`, `development_site`, and `dependentbuilding`.
* Raw description text.
* Application year: a trend term would extrapolate poorly, and recent drift is handled by
  the context features and recalibration.

### Leakage tests

* `tests/test_leakage.py` perturbs every event after *t*: it shifts or erases future
  issuances, deletes future applications, and inserts new ones. It then asserts that
  every feature at *t* is unchanged. A companion test confirms that perturbing *past* events does
  change features, so the invariance test is not vacuous.
* `tests/test_schema.py` asserts that no outcome or current-state field is a model input.
* `tests/test_snapshots.py` pins the risk-set and label boundary rules (issued exactly on
  *t*, on *t + 89*, and on *t + 90*; applied on *t*; same-day issuance; the age cap).

## 4. Models

| Model | Details |
|---|---|
| Climatology | Constant training-period positive rate (the reference for Brier skill) |
| Recent-rate baseline | `ctx_issue_rate_lag_age`: last quarter's issuance rate for applications of the same age (persistence) |
| **Historical-rate baseline** | Training-period rate by type × class × age bucket, shrunk toward its parent cell (pseudo-count 50; hierarchy: type×class×age → type×age → age → global) |
| Logistic regression | One-hot categoricals (rare levels pooled), cubic spline on log-age, log1p for counts, median imputation with missing indicators, standardization. C ∈ {0.01, 0.1, 1} |
| HistGradientBoosting | Native categorical splits, NaN-aware. Grid over learning rate, leaves, and iterations (6 configurations). No internal random early-stopping split, which would mix snapshots of the same permit |

**Calibration.** Probabilities are recalibrated on the validation period with Platt scaling (a
logistic fit on the logit) or isotonic regression. The method is chosen by permit-grouped
5-fold CV log loss within validation. Recalibrating on a recent period matters here because
the base rate drifts: the yearly positive rate was about 46% in 2012–13, about 26% in 2018–24,
and about 30% in 2025–26.

**Metrics.** ROC-AUC; PR-AUC (average precision); Brier score; log loss; expected calibration
error (10 equal-mass bins); calibration slope (from a logistic recalibration, where 1 is ideal);
Brier skill score against the historical-rate baseline and against climatology; reliability
tables and a plot (`reports/figures/reliability_test.png`). The paired bootstrap gives
intervals for the difference between each model and the historical-rate baseline.

**Robustness:**
* A rolling-origin backtest (test years starting 2020-08, 2021-08, 2022-08, 2023-08; models
  retrained on everything before each origin with the selected hyperparameters; uncalibrated).
* Ablations without the description keywords, without the context features, and without the
  cost features (cost is the main remaining field that could have been revised during review).
* A training-window sensitivity check (training on forecast dates from 2016 on only).

## 5. Results

See §6 for the numbers from the current run. They are generated in `reports/metrics.json`
and `reports/model_report.md`.

## 6. Results from the current run

Run of 2026-10-12, data as of 2026-10-10 (193,610 records, 508,292 snapshots). All numbers
are copied from `reports/metrics.json`; rerunning `huskyquiet train` regenerates them.

**Sample sizes.** Train has 438,240 snapshots from 69,466 permits (positive rate 31.9%). Validation has 26,972
snapshots from 6,008 permits (28.7%). Test (forecast dates 2025-08-01 to 2026-07-01) has 32,617
snapshots from 6,952 permits (30.0%). Only 444 test permits also appear in train.

**Selection on validation.**
* Logistic regression uses C = 0.01. Its validation log loss was 0.5273, and the grid was flat.
* HGB uses a learning rate of 0.05, 63 leaves, and 600 iterations, for a validation log loss of 0.4874 (the grid ranged from 0.4874 to 0.4909).
* Calibration, chosen by grouped-CV log loss: isotonic for logistic regression and for the baseline, and Platt for HGB. For HGB the gain was negligible (0.48741 → 0.48738), meaning it was already well calibrated.

**Test period, each snapshot weighted equally:**

| Model | ROC-AUC | PR-AUC | Brier | ECE | Cal. slope | Brier skill vs hist. rate |
|---|---|---|---|---|---|---|
| Climatology (train rate) | 0.500 | 0.300 | 0.2102 | 0.020 | — | −0.101 |
| Recent-rate baseline | 0.664 | 0.403 | 0.1932 | 0.021 | 0.90 | −0.012 |
| Historical-rate baseline | 0.686 | 0.437 | 0.1909 | 0.042 | 0.93 | 0 |
| Historical-rate baseline, isotonic | 0.688 | 0.432 | 0.1889 | 0.018 | 0.93 | 0.011 |
| Logistic, isotonic | 0.712 | 0.475 | 0.1863 | 0.025 | 0.90 | 0.024 |
| **HGB, Platt (selected)** | **0.765** | **0.562** | **0.1720** | **0.011** | **0.97** | **0.099** |

The permit-clustered bootstrap (200 resamples) gives HGB minus the historical-rate baseline:
ROC-AUC +0.079 [0.072, 0.087], PR-AUC +0.125 [0.114, 0.136], and Brier −0.019 [−0.021, −0.017]. Logistic
regression is also significantly better than the baseline, but only modestly so (ROC-AUC +0.027 [0.019, 0.033]).

**First test snapshot per permit** (n = 6,952, prevalence 38.2%): HGB has ROC-AUC 0.791 and Brier 0.179,
against 0.631 and 0.225 for the baseline. A permit with many snapshots therefore does not inflate the headline result.

**Slices (HGB vs baseline, ROC-AUC).**
* By permit type: Building 0.774 vs 0.690. Demolition 0.655 vs 0.667, the one slice where HGB does not beat the baseline.
* By permit class: Commercial 0.830, Single Family/Duplex 0.755, Multifamily 0.739.
* By application age: 0–30 days 0.782, and more than 365 days 0.669.
* Within 2 km of UW (n = 1,848): 0.721 vs 0.662.

**Robustness.**
* The ablations change ROC-AUC by at most 0.012:
  * no description keywords: 0.762
  * no context features: 0.753
  * no cost: 0.762
  * training from 2016 only: 0.761
* No single family of features carries the model, and cost (the main residual leakage risk) adds only 0.003 AUC.
* In the rolling-origin backtest, HGB beats the historical-rate baseline in every year: 0.767 vs 0.689 (2020), 0.755 vs 0.665
  (2021), 0.748 vs 0.671 (2022), and 0.774 vs 0.727 (2023).

**Permutation importance.** Application age dominates (AUC drop 0.155). Next come
`permit_type_desc` (0.043), `log_est_cost` (0.031), and the lagged age-specific issuance rate (0.011).

**Production model.** HGB with the same hyperparameters is retrained on forecast dates with
t + 90 ≤ 2025-08-01 (470,479 snapshots) and recalibrated on the test period (isotonic, re-selected
there; 32,617 snapshots). It lives in `artifacts/models/production/`. No held-out data remains
for it, so the test numbers above are its expected performance, assuming the next year
resembles the last one.

**Note on an earlier run.** An earlier run that used `housingunits*` reached a test ROC-AUC of 0.787. That
figure was inflated by the leakage described in `data_audit.md` §4 and is superseded.
