import pandas as pd
import streamlit as st

from huskyquiet import __version__
from huskyquiet.areas import load_areas
from huskyquiet.charts import backtest_figure, importance_figure, month_gap_figure, reliability_figure
from huskyquiet.config import get_paths
from huskyquiet.geocode import DEFAULT_PROVIDERS
from huskyquiet.reporting import data_counts, evaluation_headline, model_name, model_timeline
from huskyquiet.ui import load_metrics, long_date, model_metadata, prose, setting, show_figure

CHART_WIDTH = 900

st.header("How it works", anchor=False)

metrics = load_metrics()
meta = model_metadata()
counts = data_counts()
headline = evaluation_headline(metrics)
split = metrics["split"]
first_forecast = str(counts.get("first_forecast_date", "2008-01-01"))
eval_key = str(headline["model_key"])
eval_name = model_name(eval_key)
test_period = f"forecast dates {long_date(split['test_start'])} to {long_date(split['last_forecast'])}"
data_through = str(counts.get("data_as_of") or (meta or {}).get("data_as_of", ""))

overview, evaluation, limits = st.tabs(["What it shows", "How well it predicts", "Data and limits"])

with overview, prose():
    st.markdown(
        f"""
HuskyQuiet looks at Seattle building-permit applications that are still waiting for approval and
estimates, for each one, **the chance the City issues the permit within the next 90 days**.

**What issuance means.** Issuance is the City approving the permit paperwork. Work can start weeks or
months later, or never. HuskyQuiet does not predict construction start dates, noise, or disruption.

**What the numbers on the map mean.**

- **Pending permits** counts applications with a usable location inside your radius or area.
- **Expected permits issued within 90 days** adds up every nearby permit's chance of issuance.
  Ten permits at a 30% chance each add up to 3.0. It counts permits, not unique projects:
  one building often has several permits.
- **Chance issued within 90 days** on a single permit is a calibrated probability. In the held-out
  test, permits given about a 30% chance were issued within 90 days about 30% of the time.

**Where the data comes from.** The City of Seattle's
[Building Permits dataset](https://data.seattle.gov/Built-Environment/Building-Permits/76t5-zqzr),
with records through {long_date(data_through)}. Addresses are located with free public geocoders
(see *Data and limits*).
"""
    )
    st.caption(f"HuskyQuiet version {__version__}.")

with evaluation:
    with prose():
        st.markdown(
            "The model was trained on older months and tested on later months it had never seen. "
            "Every number here is read from the saved evaluation reports."
        )
        st.subheader("Which model was tested, on which dates", anchor=False)
        rows = ["| Model | Step | Forecast dates | Notes |", "| --- | --- | --- | --- |"]
        for period in model_timeline(metrics, meta, first_forecast):
            dates = f"{period.start} to {period.end}" if period.end else period.start
            rows.append(f"| {period.model} | {period.role} | {dates} | {period.note} |")
        st.markdown("\n".join(rows))
        st.markdown(
            f"""
The **evaluation copy** ({eval_name.lower()}) is the model the numbers below describe. The **deployed model**
on this site repeats the same recipe one year later: it trains on data through the test period's start
and is then calibrated on the test period itself. That makes it more current, but it also means the test
period is no longer unseen for it, so the scores below are an estimate of how the deployed model behaves,
not a measurement of it.
"""
        )

        st.subheader("Held-out test results", anchor=False)
        st.caption(
            f"Evaluation copy, {test_period}. {headline['n']:,} monthly snapshots of "
            f"{metrics['sizes']['test']['permits']:,} permits; "
            f"{headline['prevalence']:.0%} of snapshots were issued within 90 days."
        )
    with st.container(horizontal=True, width=CHART_WIDTH):
        st.metric(
            "ROC-AUC",
            f"{headline['roc_auc']:.2f}",
            help="How often an issued permit outranks one that wasn't issued. 0.5 is random, 1.0 perfect.",
            border=True,
        )
        st.metric("Baseline ROC-AUC", f"{headline['baseline_roc_auc']:.2f}",
                  help="Training-period issuance rate for the permit's type, class, and age band.", border=True)
        st.metric("Brier score", f"{headline['brier']:.3f}",
                  help="Average squared error of the probabilities. Lower is better. "
                  f"Baseline {headline['baseline_brier']:.3f}.",
                  border=True)
        st.metric("Calibration error", f"{headline['ece']:.1%}",
                  help="Average gap between predicted and observed rates across 10 groups. Lower is better.",
                  border=True)
    with prose():
        bootstrap = metrics.get("test_bootstrap_95ci", {}).get(eval_key, {})
        auc_ci = bootstrap.get("roc_auc")
        gain_ci = bootstrap.get("diff_vs_historical_rate_baseline_roc_auc")
        first = metrics.get("test_first_snapshot_per_permit", {})
        lines = [
            "**ROC-AUC is not accuracy.** It measures ranking: given one permit that was issued within 90 days "
            f"and one that was not, the model gives the issued one the higher chance about "
            f"{headline['roc_auc']:.0%} of the time.",
        ]
        if auc_ci and gain_ci:
            lines.append(
                f"Resampling whole permits gives a 95% interval of {auc_ci['ci_low']:.3f} to {auc_ci['ci_high']:.3f} "
                f"for ROC-AUC, and {gain_ci['ci_low']:.3f} to {gain_ci['ci_high']:.3f} for the gain over the baseline."
            )
        if eval_key in first and "historical_rate_baseline" in first:
            lines.append(
                f"Counting each permit once (its first month in the test), ROC-AUC is "
                f"{first[eval_key]['roc_auc']:.2f} versus {first['historical_rate_baseline']['roc_auc']:.2f} "
                f"for the baseline, over {first[eval_key]['n']:,} permits."
            )
        st.markdown("\n\n".join(lines))

    reliability = {
        key: metrics["test_reliability"][key]
        for key in (eval_key, "historical_rate_baseline")
        if key in metrics.get("test_reliability", {})
    }
    charts = st.container(width=CHART_WIDTH)
    if reliability:
        with charts:
            show_figure(reliability_figure(reliability), "Predicted versus observed issuance on the held-out test")
    month_path = get_paths().reports / "test_month_calibration.json"
    if month_path.exists():
        month_table = pd.read_json(month_path, dtype={"month": str})
        gaps = (month_table["predicted"] - month_table["observed"]).abs()
        worst = month_table.loc[gaps.idxmax()]
        with charts:
            show_figure(month_gap_figure(month_table), "Predicted and observed issuance by test month")
            st.caption(
                f"Largest monthly gap: {worst['month']}, predicted {float(worst['predicted']):.0%} "
                f"versus observed {float(worst['observed']):.0%}. Scored with the evaluation copy."
            )
    with st.expander("Earlier test years and the most important inputs", width=CHART_WIDTH):
        if metrics.get("backtest"):
            show_figure(backtest_figure(metrics["backtest"]), "ROC-AUC in earlier one-year test windows")
        importance = metrics.get("permutation_importance_test_sample") or []
        if importance:
            show_figure(importance_figure(importance), "Inputs ranked by permutation importance")

with limits, prose():
    last_forecast = str(counts.get("last_forecast_date", split["last_forecast"]))
    raw = counts.get("n_raw")
    eligible = counts.get("model_eligible")
    if raw and eligible:
        st.markdown(
            f"The extract has {int(raw):,} permit records, and {int(eligible):,} have the dates the model needs. "
            f"They become {int(counts.get('n_snapshots', 0)):,} monthly snapshots of "
            f"{int(counts.get('n_permits', 0)):,} pending applications, with forecast dates from "
            f"{long_date(first_forecast)} to {long_date(last_forecast)}."
        )
    st.subheader("Limitations", anchor=False)
    st.markdown(
        """
- **Fields change without a history.** The City publishes current values only. Cost and description
  can be edited during review without leaving a trace, so older snapshots may use later values.
- **Repeated snapshots.** A permit pending for six months appears in six monthly snapshots.
  Intervals resample whole permits so those repeats are not treated as independent.
- **Survivorship.** Applications that were never issued are rarer in older years, likely because some
  records were removed. The deployed model is calibrated on the most recent year for that reason.
- **Withdrawals.** The file does not say when an application was withdrawn, so withdrawn applications
  stay in the training data until they are 730 days old.
- **Drift.** Review times change with staffing, policy, and volume. Monthly gaps above show how far
  predictions can move from what happened.
- **Locations.** Permits without usable coordinates are scored but not mapped. Address lookup can
  place a point at the wrong spot on a long block.
"""
    )
    st.subheader("Areas", anchor=False)
    st.markdown(
        "Neighborhood shapes are City Clerk Neighborhood Map Atlas extents, an unofficial indexing boundary. "
        "University Village is the shopping center's footprint from OpenStreetMap, not a neighborhood, and "
        "it lies inside Ravenna, so its permits count toward both."
    )
    for area in load_areas():
        st.markdown(f"- **{area.label}.** Source: {area.source}.")
    st.subheader("Address lookup", anchor=False)
    configured = setting("HUSKYQUIET_GEOCODERS") or DEFAULT_PROVIDERS
    st.markdown(
        f"This site tries these free services in order: `{configured}`. Both are public and shared, so "
        "they can be slow or limit requests. Successful lookups are cached and every visitor shares one "
        "request queue. If a lookup fails, the error says why, and you can enter coordinates instead."
    )
    with st.expander("Refreshing the data (for maintainers)"):
        st.markdown(
            """
Run this from the project directory to download a new extract, rebuild snapshots, and retrain:

```
python -m huskyquiet all
```

Then restart the app. Opening the app never downloads data or retrains the model.
See `docs/deployment.md` for how the model bundle is published.
"""
        )
