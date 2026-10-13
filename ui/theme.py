"""Colours, light CSS polish and Plotly helpers.

Model identity uses fixed categorical colours (CNN, RNN, LSTM always keep theirs). Attendance
states use a reserved status palette and are ALWAYS shown with a text label too. Charts keep
Streamlit's native Plotly theme, so they work in light and dark mode.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "SMART ATTENDANCE SYSTEM"
APP_SUBTITLE = "Deep Learning-Based Face Recognition and Attendance Management"
COURSE = "TECH 405"

_CATEGORICAL = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
}
MODEL_SLOTS = {"CNN": 0, "RNN": 1, "LSTM": 2}

STATUS_COLORS = {"present": "#0ca30c", "late": "#fab219", "absent": "#d03b3b", "excused": "#898781",
                 "not marked": "#c3c2b7"}
STATUS_LABELS = {"present": "Present", "late": "Late", "absent": "Absent", "excused": "Excused",
                 "not marked": "Not yet marked"}
SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def theme_mode() -> str:
    try:
        mode = getattr(getattr(st.context, "theme", None), "type", None)
    except Exception:  # noqa: BLE001
        mode = None
    return "dark" if mode == "dark" else "light"


def categorical(index: int) -> str:
    palette = _CATEGORICAL[theme_mode()]
    return palette[min(max(index, 0), len(palette) - 1)]


def model_color(model: str) -> str:
    return categorical(MODEL_SLOTS.get(str(model).upper(), 6))


def status_badge(status: str | None, detail: str | None = None) -> str:
    """HTML pill with a text label (colour is never the only signal)."""
    key = status if status in STATUS_LABELS else "not marked"
    text = STATUS_LABELS[key] + (f" · {detail}" if detail else "")
    return f'<span class="sas-badge sas-{key.replace(" ", "-")}">{html.escape(text)}</span>'


def style_figure(fig: go.Figure, *, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(namelength=-1), bargap=0.3, bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    fig.update_traces(selector=dict(type="scatter"), line_width=2, marker_size=8)
    return fig


_CSS = """
<style>
.sas-header { padding: 0.15rem 0 0.7rem 0; border-bottom: 1px solid rgba(128,128,128,0.25); margin-bottom: 1rem; }
.sas-header .sas-kicker { font-size: 0.74rem; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.7; font-weight: 600; }
.sas-header h1 { font-size: 1.7rem; line-height: 1.2; margin: 0.15rem 0 0.25rem 0; padding: 0; }
.sas-header p { margin: 0; opacity: 0.8; }
.sas-brand { font-weight: 800; letter-spacing: 0.06em; font-size: 0.92rem; line-height: 1.2; }
.sas-brand-sub { font-size: 0.74rem; opacity: 0.75; line-height: 1.3; margin-top: 0.15rem; }
.sas-badge { display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px; font-size: 0.78rem;
             font-weight: 600; color: #ffffff; white-space: nowrap; }
.sas-present { background: #107c10; }
.sas-late { background: #9a6700; }
.sas-absent { background: #b42318; }
.sas-excused { background: #5f5e5a; }
.sas-not-marked { background: #6b7280; }
.sas-name { font-weight: 700; font-size: 0.98rem; line-height: 1.25; margin-top: 0.3rem; }
.sas-muted { opacity: 0.7; font-size: 0.82rem; }
.sas-identity { font-size: 1.45rem; font-weight: 800; margin: 0.1rem 0; }
.sas-avatar { width: 100%; aspect-ratio: 1 / 1; border-radius: 10px; display: flex; align-items: center;
              justify-content: center; font-size: 2.2rem; font-weight: 800; color: #ffffff; background: #256abf; }
.sas-login-title { text-align: center; margin-bottom: 0.2rem; }
.sas-login-title h1 { font-size: 1.6rem; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
.sas-login-title p { opacity: 0.75; margin: 0; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
