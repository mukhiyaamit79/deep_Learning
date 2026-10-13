"""Sign-in page with separate Teacher and Student tabs (demo accounts)."""

from __future__ import annotations

import time

import streamlit as st

from services.auth_service import AuthService
from ui.auth_ui import sign_in
from ui.state import require_database
from ui.theme import APP_SUBTITLE, APP_TITLE, COURSE
from utils.config import DEMO_STUDENT_PASSWORD, DEMO_TEACHER_PASSWORD, DEMO_TEACHER_USERNAME

FAILS_KEY = "login_failures"
LOCK_KEY = "login_locked_until"
MAX_FAILURES = 5
LOCK_SECONDS = 20

db = require_database()
auth = AuthService(db)

_, middle, _ = st.columns([1, 1.25, 1])
with middle:
    st.write("")
    st.markdown(f'<div class="sas-login-title"><div class="sas-muted">{COURSE}</div><h1>🎓 {APP_TITLE}</h1>'
                f"<p>{APP_SUBTITLE}</p></div>", unsafe_allow_html=True)
    st.write("")

    def login_form(role: str) -> None:
        locked_until = float(st.session_state.get(LOCK_KEY, 0.0))
        if time.time() < locked_until:
            st.warning(f"Too many failed attempts. Try again in {int(locked_until - time.time()) + 1} s.")
            return
        label = "Teacher username" if role == "teacher" else "Student ID"
        with st.form(f"login_{role}", clear_on_submit=False):
            username = st.text_input(label, key=f"login_user_{role}",
                                     placeholder="teacher" if role == "teacher" else "e.g. Amit")
            password = st.text_input("Password", type="password", key=f"login_pass_{role}")
            submitted = st.form_submit_button("Sign in", type="primary", icon=":material/login:")
        if not submitted:
            return
        result = auth.authenticate(username, password, role)
        if result.user is None:
            failures = int(st.session_state.get(FAILS_KEY, 0)) + 1
            st.session_state[FAILS_KEY] = failures
            if failures >= MAX_FAILURES:
                st.session_state[LOCK_KEY] = time.time() + LOCK_SECONDS
                st.session_state[FAILS_KEY] = 0
            st.error(result.message, icon=":material/error:")
            return
        st.session_state[FAILS_KEY] = 0
        sign_in(result.user)
        st.rerun()

    with st.container(border=True):
        teacher_tab, student_tab = st.tabs([":material/school: Teacher", ":material/person: Student"])
        with teacher_tab:
            st.caption("Run live attendance, manage students and view all reports.")
            login_form("teacher")
        with student_tab:
            st.caption("See your own attendance history and attendance rate.")
            login_form("student")

    with st.expander("Demo login details", icon=":material/key:", expanded=True):
        st.markdown(
            f"- **Teacher:** username `{DEMO_TEACHER_USERNAME}` · password `{DEMO_TEACHER_PASSWORD}`\n"
            f"- **Students:** username = Student ID (`Amit`, `Shreejal`, `Susman`, `Udhav`, `Uttam`) · "
            f"password `{DEMO_STUDENT_PASSWORD}`"
        )
        st.caption("Dummy accounts for the classroom demo only. Passwords are stored hashed and can be "
                   "changed under “My account”.")
    st.caption("Academic prototype — runs only on this computer; camera frames are never stored or uploaded.")
