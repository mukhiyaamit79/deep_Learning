"""Login state for the demo teacher / student accounts."""

from __future__ import annotations

import html

import streamlit as st

from services.auth_service import CurrentUser

USER_KEY = "sas_user"
#: Session-state keys removed on sign-out (live camera objects, cached results, flash messages).
_CLEAR_PREFIXES = ("live_", "still_", "sas_flash", "dash_", "rec_", "rep_", "stu_")


def current_user() -> CurrentUser | None:
    user = st.session_state.get(USER_KEY)
    return user if isinstance(user, CurrentUser) else None


def sign_in(user: CurrentUser) -> None:
    st.session_state[USER_KEY] = user


def sign_out() -> None:
    for key in list(st.session_state.keys()):
        if key == USER_KEY or str(key).startswith(_CLEAR_PREFIXES):
            del st.session_state[key]


def require_teacher() -> CurrentUser:
    user = current_user()
    if user is None or not user.is_teacher:
        st.error("This page is for teachers only.", icon=":material/lock:")
        st.stop()
    return user


def require_student() -> CurrentUser:
    user = current_user()
    if user is None or user.role != "student" or user.student_pk is None:
        st.error("This page is for students only.", icon=":material/lock:")
        st.stop()
    return user


def render_user_sidebar(user: CurrentUser) -> None:
    with st.sidebar:
        role = "Teacher" if user.is_teacher else "Student"
        icon = ":material/school:" if user.is_teacher else ":material/person:"
        st.markdown(f"{icon} **{html.escape(user.display_name)}**  \n"
                    f"<span class='sas-muted'>{role} · {html.escape(user.username)}</span>", unsafe_allow_html=True)
        if st.button("Sign out", icon=":material/logout:", key="sas_sign_out", width="stretch"):
            sign_out()
            st.rerun()
