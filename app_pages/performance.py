import pandas as pd
import streamlit as st

from huskyquiet.charts import backtest_figure, importance_figure, metric_bars, month_gap_figure, reliability_figure
from huskyquiet.config import get_paths
from huskyquiet.reporting import evaluation_headline, load_model_metadata
from huskyquiet.ui import load_metrics, show_figure, status_caption

st.header("Model performance")
st.caption(
    "Every number on this page is read from the saved evaluation reports. "
    "The target is P(a pending application is issued within 90 days). "
    "It is not a model of construction starts or noise."
)

metrics = load_metrics()
headline = evaluation_headline(metrics)
split = metrics["split"]
status_caption()

try:
    meta = load_model_metadata()
except (FileNotFoundError, ValueError) as exc:
    meta = None
    st.warning(str(exc))

st.subheader("Held-out test period")
st.markdown(
    f"Forecast dates {split['test_start']} through {split['last_forecast']}, "
    f"with a {split['horizon_days']}-day horizon. "
    f"Training labels stop 90 days before {split['val_start']}, and validation labels stop 90 days before "
    f"{split['test_start']}, so no label window crosses into the next period. "
    f"The test set has {headline['n']:,} snapshots and an issuance rate of {headline['prevalence']:.1%}."
)
st.markdown(
    "Chronological validation matters because a random split would train on later outcomes of the same "
    "permit and on citywide conditions that had not happened yet. The model also refuses outcome fields "
    "(issuance date, current status, review times) and drops fields that the audit showed are filled in "
    "during review, including housing-unit counts."
)

with st.container(horizontal=True):
    st.metric("Production-procedure ROC-AUC", f"{headline['roc_auc']:.3f}", border=True)
    st.metric("Baseline ROC-AUC", f"{headline['baseline_roc_auc']:.3f}", border=True)
    st.metric(
        "PR-AUC", f"{headline['pr_auc']:.3f}",
        help=f"Baseline PR-AUC {headline['baseline_pr_auc']:.3f}", border=True,
    )
    st.metric(
        "Brier score", f"{headline['brier']:.3f}",
        help=f"Baseline Brier {headline['baseline_brier']:.3f}. Lower is better.", border=True,
    )
    st.metric(
        "Calibration error", f"{headline['ece']:.3f}",
        help="Expected calibration error, 10 equal-mass bins. Lower is better.", border=True,
    )

selected = metrics["selection"]
st.caption(
    f"These figures are `{headline['model_key']}`, the evaluation model selected on the validation period "
    f"({selected['calibration_method']['hgb']} calibration, "
    f"{selected['hgb_best_params']['max_leaf_nodes']} leaves). "
    "The production bundle repeats that procedure one period later and calibrates on this test window, "
    "so quoting this ROC-AUC as the production artifact's score on the same window would be biased. "
    + (
        f"The loaded production bundle uses {meta.get('calibration', 'unknown')} calibration and was "
        f"{meta.get('trained_on', 'fit on earlier data')}; its calibrator was "
        f"{meta.get('calibrated_on', 'fit on a later window')}."
        if meta else ""
    )
)

auc_keys = [
    ("historical_rate_baseline", "Historical rate"),
    ("recent_rate_baseline", "Recent rate"),
    ("logistic_calibrated_isotonic", "Logistic"),
    (str(headline["model_key"]), "Gradient boosting"),
]
show_figure(
    metric_bars(
        [label for key, label in auc_keys if key in metrics["test"]],
        [float(metrics["test"][key]["roc_auc"]) for key, label in auc_keys if key in metrics["test"]],
        "Test ROC-AUC",
        "ROC-AUC",
    ),
    "Test-period ROC-AUC by model",
)

reliability = {
    name: metrics["test_reliability"][name]
    for name in ("historical_rate_baseline", str(headline["model_key"]))
    if name in metrics["test_reliability"]
}
if reliability:
    show_figure(reliability_figure(reliability), "Reliability curves for the test period")

if metrics.get("backtest"):
    show_figure(backtest_figure(metrics["backtest"]), "Rolling-origin backtest ROC-AUC")

importance = metrics.get("permutation_importance_test_sample") or []
if importance:
    show_figure(importance_figure(importance), "Permutation importance on the test sample")

month_path = get_paths().reports / "test_month_calibration.json"
if month_path.exists():
    month_table = pd.read_json(month_path)
    show_figure(month_gap_figure(month_table), "Observed and predicted issuance rates by test month")
    gaps = (month_table["predicted"] - month_table["observed"]).abs()
    worst = month_table.loc[gaps.idxmax()]
    st.markdown(
        f"The largest absolute month-level gap in this file is {worst['month']}: "
        f"predicted {float(worst['predicted']):.3f}, observed {float(worst['observed']):.3f}. "
        "Aggregate calibration can look good while individual months drift. "
        "That series was scored with the evaluation bundle, which did not train on these labels."
    )
else:
    st.info("Monthly calibration has not been written. Run `python -m huskyquiet report-drift`.")

st.subheader("Limitations")
st.markdown(
    """
- The City publishes current values, not a history of edits. Cost and description category
  can change during review without leaving a trace in this extract.
- Withdrawn applications stay in the historical risk set until they are 730 days old,
  because the file does not say when the withdrawal happened.
- Never-issued applications are rarer in older years. Some of that is likely record
  survivorship, which is why the calibrator is fit on a recent window.
- Description cleaning is a keyword and date filter. An administrative note phrased some other way can remain.
- Ranking inside a single age band is weaker, because application age carries much of the signal.
- A new download of the daily portal reload will move these numbers. They are tied to the extract named above.
"""
)
bootstrap = metrics.get("test_bootstrap_95ci", {}).get(str(headline["model_key"]), {})
diff = bootstrap.get("diff_vs_historical_rate_baseline_roc_auc")
if diff:
    st.caption(
        "The ROC-AUC gain over the historical-rate baseline has a permit-clustered bootstrap 95% interval of "
        f"[{diff['ci_low']:.3f}, {diff['ci_high']:.3f}]. "
        "Whole permits are resampled, so repeated monthly snapshots of one application are not treated as independent."
    )
