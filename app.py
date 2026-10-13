"""SMART ATTENDANCE SYSTEM — Streamlit web app (entry point).

Recommended start (applies the notebook's native-library setting first):

    python run_app.py

Alternative:

    python -m streamlit run app.py

Sign-in decides which pages exist: teachers get the full app, students get their own attendance.
Pages live in app_pages/ (not pages/) because navigation is defined here with st.navigation.
"""

from __future__ import annotations

from attendance_runtime import configure_runtime

configure_runtime()  # the notebook's MKL setting, before NumPy / PyTorch are used

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Smart Attendance System", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

from ui.auth_ui import current_user, render_user_sidebar  # noqa: E402
from ui.state import get_database  # noqa: E402
from ui.theme import APP_SUBTITLE, APP_TITLE, inject_css  # noqa: E402

inject_css()

# Create / migrate the database and the demo accounts before anyone signs in.
try:
    get_database()
except Exception as exc:  # noqa: BLE001 - show the problem instead of a stack trace
    st.error(f"**The local database could not be prepared:** {exc}", icon=":material/database:")
    st.stop()

user = current_user()

if user is None:
    navigation = st.navigation([st.Page("app_pages/login.py", title="Sign in", icon=":material/login:")],
                               position="hidden")
elif user.is_teacher:
    navigation = st.navigation(
        {
            "Overview": [
                st.Page("app_pages/teacher_dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
            ],
            "Attendance": [
                st.Page("app_pages/live_attendance.py", title="Live Attendance", icon=":material/videocam:"),
                st.Page("app_pages/attendance_records.py", title="Attendance Records", icon=":material/fact_check:"),
                st.Page("app_pages/reports.py", title="Reports", icon=":material/insights:"),
            ],
            "People": [
                st.Page("app_pages/students.py", title="Students", icon=":material/groups:"),
            ],
            "Models": [
                st.Page("app_pages/model_comparison.py", title="CNN vs RNN vs LSTM", icon=":material/compare:"),
            ],
            "System": [
                st.Page("app_pages/settings.py", title="Settings", icon=":material/settings:"),
                st.Page("app_pages/account.py", title="My account", icon=":material/manage_accounts:"),
                st.Page("app_pages/about.py", title="About", icon=":material/info:"),
            ],
        }
    )
else:
    navigation = st.navigation(
        [
            st.Page("app_pages/student_dashboard.py", title="My attendance", icon=":material/person:", default=True),
            st.Page("app_pages/account.py", title="My account", icon=":material/manage_accounts:"),
            st.Page("app_pages/about.py", title="About", icon=":material/info:"),
        ]
    )

if user is not None:
    render_user_sidebar(user)
    with st.sidebar:
        st.divider()
        st.markdown(f'<div class="sas-brand">{APP_TITLE}</div><div class="sas-brand-sub">{APP_SUBTITLE}</div>',
                    unsafe_allow_html=True)
        st.caption("Academic prototype · runs locally · nothing is uploaded")

navigation.run()
