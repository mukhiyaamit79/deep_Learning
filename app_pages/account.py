"""My account: change the password of the signed-in teacher or student."""

from __future__ import annotations

import streamlit as st

from services.auth_service import AuthService
from ui.auth_ui import current_user
from ui.components import page_header, show_validation_error
from ui.state import require_database
from utils.validators import ValidationError

user = current_user()
if user is None:
    st.stop()
page_header("My account", f"Signed in as {user.display_name} ({'teacher' if user.is_teacher else 'student'}).")
db = require_database()

with st.container(border=True):
    st.markdown(f"**Username:** `{user.username}`  \n**Role:** {'Teacher' if user.is_teacher else 'Student'}")

st.markdown("##### Change password")
st.caption("The demo passwords are public in the README — change them if you keep using the app.")
with st.form("change_password", clear_on_submit=True):
    current = st.text_input("Current password", type="password")
    new = st.text_input("New password (min. 6 characters)", type="password")
    confirm = st.text_input("Repeat new password", type="password")
    submitted = st.form_submit_button("Change password", type="primary", icon=":material/key:")
if submitted:
    try:
        AuthService(db).change_password(user.id, current, new, confirm)
    except ValidationError as exc:
        show_validation_error(exc)
    else:
        st.success("Password changed.", icon=":material/check_circle:")
