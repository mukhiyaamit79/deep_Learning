"""CNN vs RNN vs LSTM — shows the results exported by amit_eda.ipynb (NN-39). Nothing is trained here."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.results_service import best_model, load_results
from ui.auth_ui import require_teacher
from ui.components import NOTEBOOK_EXPORT_STEPS, empty_state, page_header
from ui.state import artifact_status, get_paths
from ui.theme import model_color, style_figure
from utils.config import FIGURE_TITLES

require_teacher()
page_header("CNN vs RNN vs LSTM", "Comparison on the same face images and the same train/validation/test split. "
            "Every number comes from the notebook's exported results.", kicker="TECH 405 · Model comparison")

paths = get_paths()
results = load_results(paths)
metadata = artifact_status().metadata or {}
for err in results.errors:
    st.error(err, icon=":material/error:")

QUALITY = [("Accuracy", "Accuracy"), ("Precision", "Macro precision"), ("Recall", "Macro recall"), ("F1-score", "Macro F1")]


def explanation(comparison: pd.DataFrame | None) -> None:
    st.subheader("Why the architecture matters for face images")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**CNN — spatial filters**")
        st.caption("Convolution kernels look at small 2-D neighbourhoods (eyes, nose, jaw line) with shared weights, "
                   "and pooling tolerates small shifts. This matches the structure of a face photo.")
    with c2, st.container(border=True):
        st.markdown("**SimpleRNN — rows as a sequence**")
        st.caption("The 128×128 image is read as 128 rows of 128 pixels. Vertical relationships must survive in one "
                   "hidden state across many steps, and plain RNNs suffer from vanishing gradients.")
    with c3, st.container(border=True):
        st.markdown("**LSTM — gated memory**")
        st.caption("Gates keep information across the 128 rows, but there is still no built-in notion of 2-D "
                   "locality. Image rows are not really a time series.")
    if comparison is None or comparison.empty:
        return
    leaders = best_model(comparison, "Accuracy")
    test_size = (metadata.get("split_sizes") or {}).get("test")
    names = " and ".join(m for m, _ in leaders)
    text = f"**Measured result:** the highest test accuracy ({leaders[0][1]:.1%}) was reached by **{names}**."
    if len(leaders) > 1:
        text += " It is a tie, so the test set alone cannot separate these models."
    if test_size:
        text += (f" The test set has {test_size} images, so one image changes accuracy by "
                 f"{100 / int(test_size):.1f} percentage points — differences this small are not conclusive.")
    text += (" The CNN is the deployed model because the CNN family was fixed in advance and its configuration was "
             "selected on validation data only; the test set was not used for that choice.")
    st.markdown(text)


if not results.available:
    st.warning("**No comparison results found.** Run the notebook so NN-39 exports them — no placeholder numbers "
               "are shown.", icon=":material/hourglass_empty:")
    if results.missing:
        st.caption("Missing in results/: " + ", ".join(results.missing))
    st.markdown(NOTEBOOK_EXPORT_STEPS)
    explanation(None)
    st.stop()

comparison = results.comparison.copy()
comparison["Model"] = comparison["Model"].astype(str)
summary = results.summary or {}

# ---- deployment decision and run information ------------------------------------------------------
decision = summary.get("deployment_decision") or (metadata.get("selection") or {})
if decision:
    with st.container(border=True):
        st.markdown(f"**:material/rocket_launch: Deployed model: {decision.get('architecture', 'CNN')} — "
                    f"“{decision.get('configuration', '?')}” configuration**")
        if decision.get("selection_rule"):
            st.caption(f"Selection rule: {decision['selection_rule']}.")
        parts = []
        if decision.get("validation_accuracy") is not None:
            parts.append(f"validation accuracy {float(decision['validation_accuracy']):.1%}")
        if decision.get("validation_loss") is not None:
            parts.append(f"validation loss {float(decision['validation_loss']):.4f}")
        if decision.get("parameters"):
            parts.append(f"{int(decision['parameters']):,} parameters")
        parts.append("test set used for selection: " + ("yes" if decision.get("test_used_for_selection") else "no"))
        st.caption(" · ".join(parts))
hardware = summary.get("hardware") or metadata.get("hardware") or {}
info = [f"split: {summary.get('split_protocol') or metadata.get('split_protocol', '?')}",
        f"seed {summary.get('seed', metadata.get('random_seed', '?'))}"]
if metadata.get("split_sizes"):
    s = metadata["split_sizes"]
    info.append(f"train/val/test = {s.get('train')}/{s.get('validation')}/{s.get('test')} images")
if hardware:
    info.append(f"trained on {hardware.get('gpu') or hardware.get('device', '?')} · PyTorch {hardware.get('pytorch', '?')}")
st.caption("Experiment: " + " · ".join(info))

# ---- summary table --------------------------------------------------------------------------------
st.subheader("Test-set results")
table = comparison.copy()
baseline = summary.get("majority_baseline")
if isinstance(baseline, dict):
    table = pd.concat([table, pd.DataFrame([baseline])], ignore_index=True)
pct = ["Accuracy", "Precision", "Recall", "F1-score", "Weighted Precision", "Weighted Recall", "Weighted F1-score"]
for col in pct:
    if col in table.columns:
        table[col] = pd.to_numeric(table[col], errors="coerce") * 100
config = {col: st.column_config.NumberColumn(col if col != "Precision" and col != "Recall" and col != "F1-score"
                                             else f"Macro {col.lower()}", format="%.1f%%") for col in pct}
config.update({"Test Loss": st.column_config.NumberColumn("Test loss", format="%.4f"),
               "Parameters": st.column_config.NumberColumn("Parameters", format="%d"),
               "Training Time (s)": st.column_config.NumberColumn("Training time (s)", format="%.1f"),
               "Inference Time (ms/image)": st.column_config.NumberColumn("Latency (ms/image)", format="%.2f"),
               "Batch Time (ms)": st.column_config.NumberColumn("Batch time (ms)", format="%.2f"),
               "Batch Throughput (images/s)": st.column_config.NumberColumn("Throughput (img/s)", format="%.0f")})
st.dataframe(table, hide_index=True, column_config=config)
st.caption("The last row is the majority-class baseline for reference. Timings depend on the GPU used for training.")

# ---- charts ----------------------------------------------------------------------------------------
fig = go.Figure()
for _, row in comparison.iterrows():
    values = [float(row[col]) for col, _ in QUALITY]
    fig.add_bar(x=[label for _, label in QUALITY], y=values, name=row["Model"], marker_color=model_color(row["Model"]),
                text=[f"{v:.1%}" for v in values], textposition="outside",
                hovertemplate=f"{row['Model']} · %{{x}}: %{{y:.3f}}<extra></extra>")
fig.update_layout(barmode="group", title=dict(text="Classification quality (test set)", font_size=14))
fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
st.plotly_chart(style_figure(fig, height=360), key="cmp_quality")


def metric_bar(column: str, title: str, fmt: str, key: str) -> None:
    if column not in comparison.columns:
        return
    values = pd.to_numeric(comparison[column], errors="coerce")
    figure = go.Figure(go.Bar(x=comparison["Model"], y=values, marker_color=[model_color(m) for m in comparison["Model"]],
                              text=[fmt.format(v) if pd.notna(v) else "n/a" for v in values], textposition="outside",
                              hovertemplate="%{x}: %{y}<extra></extra>"))
    figure.update_layout(title=dict(text=title, font_size=14))
    figure.update_yaxes(rangemode="tozero")
    figure.update_xaxes(type="category")
    st.plotly_chart(style_figure(figure, height=290, legend=False), key=key)


c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_bar("Training Time (s)", "Training time (s)", "{:.1f}", "cmp_time")
with c2:
    metric_bar("Parameters", "Parameters", "{:,.0f}", "cmp_params")
with c3:
    metric_bar("Inference Time (ms/image)", "Latency per image (ms)", "{:.2f}", "cmp_latency")
with c4:
    metric_bar("Batch Throughput (images/s)", "Throughput (images/s)", "{:,.0f}", "cmp_throughput")
st.caption("Separate charts because the quantities have different units.")

# ---- per student --------------------------------------------------------------------------------------
st.subheader("Per-student precision, recall and F1")
if results.per_class is None:
    empty_state("Per-class metrics not exported", "Run notebook NN-39.", ":material/table_rows:")
else:
    per_class = results.per_class.copy()
    metric = st.segmented_control("Metric", ["F1-score", "Precision", "Recall"], default="F1-score", key="cmp_metric") \
        or "F1-score"
    fig = go.Figure()
    for model in comparison["Model"]:
        part = per_class[per_class["Model"].astype(str) == model]
        fig.add_bar(x=part["Student ID"].astype(str), y=part[metric], name=model, marker_color=model_color(model),
                    hovertemplate=f"{model} · %{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_layout(barmode="group", title=dict(text=f"Per-student {metric}", font_size=14))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(type="category")
    st.plotly_chart(style_figure(fig, height=340), key="cmp_per_class")
    with st.expander("Per-student table"):
        st.dataframe(per_class, hide_index=True)

# ---- notebook figures ---------------------------------------------------------------------------------
st.subheader("Training curves and confusion matrices")
if not results.figures:
    st.caption("No figures found in results/figures/ (exported by the notebook's comparison cells).")
else:
    groups = {
        "Training curves": [n for n in results.figures if n.startswith("history_")],
        "Confusion matrices": [n for n in results.figures if n.startswith("confusion_")],
        "Metric charts": [n for n in results.figures if n.startswith("test_")],
        "Efficiency": [n for n in results.figures if n in ("training_time.png", "parameters.png", "inference_latency.png",
                                                           "batch_throughput.png")],
    }
    tabs = st.tabs([g for g, names in groups.items() if names])
    for tab, (group, names) in zip(tabs, [(g, n) for g, n in groups.items() if n]):
        with tab:
            cols = st.columns(2)
            for i, name in enumerate(names):
                with cols[i % 2]:
                    st.image(str(results.figures[name]), caption=FIGURE_TITLES.get(name, name), width="stretch")

# ---- experiments and reports ----------------------------------------------------------------------------
with st.expander("All training runs (CNN hyper-parameter experiments, RNN and LSTM)"):
    if results.experiments is not None:
        st.dataframe(results.experiments, hide_index=True)
    if results.validation is not None:
        st.markdown("**Selected configuration per architecture (validation only)**")
        st.dataframe(results.validation, hide_index=True)
reports = summary.get("classification_reports") or {}
if reports:
    with st.expander("Classification reports (test set)"):
        for model, report in reports.items():
            if isinstance(report, dict):
                rows = {k: v for k, v in report.items() if isinstance(v, dict)}
                st.markdown(f"**{model}** — accuracy {float(report.get('accuracy', float('nan'))):.1%}")
                st.dataframe(pd.DataFrame(rows).T.round(4), width="stretch")

explanation(comparison)
