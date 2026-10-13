"""Student management: directory with photos, registration, editing, consent, logins and model-class links."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from database.db import DatabaseError
from services.auth_service import AuthService
from services.student_service import ELIGIBILITY_TEXT, RecognitionEligibility, StudentForm, StudentService, eligibility
from ui.auth_ui import require_teacher
from ui.components import avatar, download_csv, empty_state, metric_card, page_header, show_validation_error
from ui.state import artifact_status, get_paths, require_database
from utils.config import DEFAULT_CLASS_NAME, DEMO_STUDENT_PASSWORD
from utils.validators import ValidationError

NO_LINK = "— Not in trained model —"
FLASH_KEY = "stu_flash"

user = require_teacher()
page_header("Students", "Registered students, their photos, login accounts and link to the trained CNN.")

db = require_database()
paths = get_paths()
service = StudentService(db, paths.dataset_dir)
auth = AuthService(db)
status = artifact_status()
model_labels = status.label_set
model_classes = status.class_names

if FLASH_KEY in st.session_state:
    st.success(st.session_state.pop(FLASH_KEY), icon=":material/check_circle:")


def text(value: object) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


tab_dir, tab_reg, tab_edit, tab_map = st.tabs([":material/groups: Directory", ":material/person_add: Register",
                                              ":material/edit: Edit / deactivate", ":material/link: Model links & logins"])

# =====================================================================================
# Directory
# =====================================================================================
with tab_dir:
    f1, f2, f3 = st.columns([2, 1.2, 1])
    search = f1.text_input("Search", placeholder="ID, name or model class", key="stu_search")
    classes = service.class_names()
    class_filter = f2.selectbox("Class", ["All classes", *classes], key="stu_class")
    active_choice = f3.selectbox("Status", ["Active", "Inactive", "All"], key="stu_active")
    df = service.directory(model_labels=model_labels, search=search or None,
                           class_name=None if class_filter == "All classes" else class_filter,
                           active={"Active": True, "Inactive": False, "All": None}[active_choice])

    recognisable = ELIGIBILITY_TEXT[RecognitionEligibility.RECOGNIZABLE]
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Students shown", len(df))
    with m2:
        metric_card("Recognisable by the camera", int((df["recognition_status"] == recognisable).sum()))
    with m3:
        metric_card("Need manual marking", int((df["recognition_status"] != recognisable).sum()),
                    help="Not in the trained model, no consent, or inactive.")

    if df.empty:
        empty_state("No students found", "Register a student or change the filters.", ":material/person_search:")
    else:
        rows = df.to_dict("records")
        per_row = 5
        for start in range(0, len(rows), per_row):
            cols = st.columns(per_row)
            for col, row in zip(cols, rows[start : start + per_row]):
                with col, st.container(border=True):
                    avatar(service.photo(row), text(row["full_name"]))
                    st.markdown(f'<div class="sas-name">{html.escape(text(row["full_name"]))}</div>'
                                f'<div class="sas-muted">{html.escape(text(row["student_id"]))} · '
                                f'{html.escape(text(row["class_name"]))}</div>', unsafe_allow_html=True)
                    ok = row["recognition_status"] == recognisable
                    st.caption((":green[:material/verified:] " if ok else ":orange[:material/info:] ")
                               + row["recognition_status"])
        with st.expander("Table view and export"):
            view = df.copy()
            view["consent"] = view["consent_recorded_at"].map(lambda v: f"Recorded {str(v)[:10]}" if text(v) else "Not recorded")
            cols = ["student_id", "full_name", "class_name", "section", "model_class_label", "recognition_status",
                    "consent", "is_active"]
            st.dataframe(view[cols], hide_index=True,
                         column_config={"student_id": "Student ID", "full_name": "Full name", "class_name": "Class",
                                        "section": "Section", "model_class_label": "Model class",
                                        "recognition_status": "Recognition status", "consent": "Consent",
                                        "is_active": st.column_config.CheckboxColumn("Active")})
            download_csv(view[cols], "students.csv", "Download student list (CSV)", key="stu_csv")

# =====================================================================================
# Register
# =====================================================================================
with tab_reg:
    st.info("**Registering does not teach the CNN a new face.** Only the trained classes "
            f"({', '.join(model_classes) if model_classes else 'none found'}) can be recognised. A new student without "
            "a model class is marked manually until the notebook is re-run with their photos.", icon=":material/info:")
    free = service.unlinked_labels(model_classes)
    with st.form("register_student", clear_on_submit=True):
        c1, c2 = st.columns(2)
        sid = c1.text_input("Student ID *", max_chars=32, help="Also the student's login username.")
        name = c2.text_input("Full name *", max_chars=120)
        c3, c4, c5 = st.columns([2, 1, 2])
        cls = c3.text_input("Class *", value=DEFAULT_CLASS_NAME, max_chars=50)
        sec = c4.text_input("Section", max_chars=20)
        mail = c5.text_input("Email (optional)", max_chars=120)
        link = st.selectbox("Trained model class", [NO_LINK, *free], disabled=not free,
                            help="Only classes not yet linked to another student are listed.")
        consent = st.checkbox("The student has given informed consent to face recognition for this demonstration.")
        submitted = st.form_submit_button("Register student", type="primary", icon=":material/person_add:")
    if submitted:
        form = StudentForm(sid, name, cls, sec, mail, None if link == NO_LINK else link)
        try:
            pk = service.register(form, consent=consent, model_labels=model_labels)
        except (ValidationError, DatabaseError) as exc:
            show_validation_error(exc)
        else:
            record = service.get(pk) or {}
            st.session_state[FLASH_KEY] = (f"Registered {form.full_name.strip()}. Login: username "
                                           f"'{form.student_id.strip()}', password '{DEMO_STUDENT_PASSWORD}'. "
                                           f"Status: {ELIGIBILITY_TEXT[eligibility(record, model_labels)]}.")
            st.rerun()

# =====================================================================================
# Edit / deactivate
# =====================================================================================


@st.dialog("Change enrolment status")
def confirm_active(pk: int, name: str, activate: bool) -> None:
    verb = "Reactivate" if activate else "Deactivate"
    st.write(f"{verb} **{name}**?" + ("" if activate else " Deactivated students cannot log in, are not recognised "
                                                          "and are left out of new sessions. History is kept."))
    c1, c2 = st.columns(2)
    if c1.button(verb, type="primary", key="dlg_active_yes"):
        service.set_active(pk, activate)
        st.session_state[FLASH_KEY] = f"{name} {'reactivated' if activate else 'deactivated'}."
        st.rerun()
    if c2.button("Cancel", key="dlg_active_no"):
        st.rerun()


@st.dialog("Face-recognition consent")
def confirm_consent(pk: int, name: str, record: bool) -> None:
    st.write(f"{'Record' if record else 'Withdraw'} face-recognition consent for **{name}**?"
             + ("" if record else " The camera will no longer mark this student; manual marking still works."))
    c1, c2 = st.columns(2)
    if c1.button("Record consent" if record else "Withdraw consent", type="primary", key="dlg_consent_yes"):
        service.set_consent(pk, record)
        st.session_state[FLASH_KEY] = f"Consent {'recorded' if record else 'withdrawn'} for {name}."
        st.rerun()
    if c2.button("Cancel", key="dlg_consent_no"):
        st.rerun()


with tab_edit:
    everyone = service.directory(model_labels=model_labels, active=None)
    if everyone.empty:
        empty_state("No students yet", "Register a student first.")
    else:
        rows = {int(r["id"]): r for r in everyone.to_dict("records")}
        pk = st.selectbox("Student", list(rows), key="stu_edit_pick",
                          format_func=lambda k: f"{rows[k]['student_id']} · {rows[k]['full_name']}"
                          + ("" if rows[k]["is_active"] else " (inactive)"))
        current = rows[pk]
        left, right = st.columns([1, 3], gap="large")
        with left:
            avatar(service.photo(current), text(current["full_name"]))
            st.caption(current["recognition_status"])
        with right:
            current_link = text(current.get("model_class_label"))
            options = [NO_LINK, *service.unlinked_labels(model_classes)]
            if current_link and current_link not in options:
                options.insert(1, current_link)
            with st.form(f"edit_student_{pk}"):
                c1, c2 = st.columns(2)
                e_sid = c1.text_input("Student ID *", value=text(current["student_id"]), max_chars=32)
                e_name = c2.text_input("Full name *", value=text(current["full_name"]), max_chars=120)
                c3, c4, c5 = st.columns([2, 1, 2])
                e_cls = c3.text_input("Class *", value=text(current["class_name"]), max_chars=50)
                e_sec = c4.text_input("Section", value=text(current.get("section")), max_chars=20)
                e_mail = c5.text_input("Email", value=text(current.get("email")), max_chars=120)
                e_link = st.selectbox("Trained model class", options, index=options.index(current_link or NO_LINK))
                save = st.form_submit_button("Save changes", type="primary", icon=":material/save:")
            if save:
                form = StudentForm(e_sid, e_name, e_cls, e_sec, e_mail, None if e_link == NO_LINK else e_link)
                try:
                    service.update(pk, form, model_labels=model_labels)
                except (ValidationError, DatabaseError, ValueError) as exc:
                    show_validation_error(exc)
                else:
                    st.session_state[FLASH_KEY] = f"Saved changes for {form.full_name.strip()}."
                    st.rerun()

            b1, b2, b3 = st.columns(3)
            has_consent = bool(text(current.get("consent_recorded_at")))
            if b1.button("Withdraw consent" if has_consent else "Record consent", icon=":material/how_to_reg:",
                         key=f"consent_{pk}"):
                confirm_consent(pk, text(current["full_name"]), not has_consent)
            active = bool(current["is_active"])
            if b2.button("Deactivate" if active else "Reactivate",
                         icon=":material/person_off:" if active else ":material/person_check:", key=f"active_{pk}"):
                confirm_active(pk, text(current["full_name"]), not active)
            account = auth.account_for_student(pk)
            if account and b3.button("Reset login password", icon=":material/lock_reset:", key=f"reset_{pk}"):
                auth.reset_password(int(account["id"]))
                st.session_state[FLASH_KEY] = (f"Password for '{account['username']}' reset to '{DEMO_STUDENT_PASSWORD}'.")
                st.rerun()
            st.caption(f"Login username: **{account['username']}**" if account else "This student has no login account.")

# =====================================================================================
# Model links and logins
# =====================================================================================
with tab_map:
    if model_classes is None:
        st.warning("models/class_names.json was not found, so the trained classes are unknown. Run notebook NN-36.",
                   icon=":material/warning:")
    else:
        st.markdown(f"The trained CNN has **{len(model_classes)} classes**. Each class can be linked to one student "
                    "(enforced by a UNIQUE index in SQLite).")
        st.dataframe(service.model_class_overview(model_classes), hide_index=True,
                     column_config={"model_index": st.column_config.NumberColumn("Output", format="%d"),
                                    "model_class_label": "Model class", "student_id": "Student ID",
                                    "full_name": "Student", "status": "Status"})
        stale = service.stale_mappings(model_classes)
        if stale:
            st.warning("Linked to classes the current model does not contain: "
                       + ", ".join(f"{s['student_id']} → {s['model_class_label']}" for s in stale))
        if st.button("Link classes to students with the same ID", icon=":material/link:", key="stu_autolink"):
            created = service.auto_link_matching_ids(model_classes)
            if created:
                st.session_state[FLASH_KEY] = "Linked: " + ", ".join(f"{c} → {s}" for c, s in created)
                st.rerun()
            st.info("Nothing to link: every class is linked or no unlinked student ID matches.")

    st.markdown("##### Login accounts")
    accounts = pd.DataFrame(auth.list_accounts())
    if not accounts.empty:
        accounts["is_active"] = accounts["is_active"].astype(bool)
        st.dataframe(accounts[["username", "display_name", "role", "student_id", "is_active", "last_login_at"]],
                     hide_index=True, column_config={"username": "Username", "display_name": "Name", "role": "Role",
                                                     "student_id": "Student ID",
                                                     "is_active": st.column_config.CheckboxColumn("Enabled"),
                                                     "last_login_at": "Last sign-in"})
    st.caption(f"Demo passwords: students '{DEMO_STUDENT_PASSWORD}' (until changed). Students sign in with their "
               "Student ID and only see their own attendance.")
