"""Live face recognition that marks attendance.

For each student (one at a time):
    camera frame -> OpenCV face detection -> crop -> notebook preprocessing (attendance_utils)
    -> CNN (attendance_cnn.pt) -> score >= threshold -> same student in N consecutive frames
    -> AUTO-MARK Present in SQLite (or teacher clicks Confirm when auto-mark is off) -> Undo available

Threading: the video callback runs in a streamlit-webrtc worker thread and only touches the
lock-protected LiveRecognitionSession. The side panel is an auto-refreshing fragment in the normal
Streamlit thread; every SQLite write happens there.
"""

from __future__ import annotations

import hashlib
import html
import time
from typing import Any

import cv2
import pandas as pd
import streamlit as st

from database.db import Database, DatabaseError
from database.repositories import SessionRepository
from services.attendance_service import AttendanceService, IdentityResolution, IdentityStatus, suggested_status
from services.face_recognition import FaceRecognizer, FrameAnalysis, FrameStatus, ModelArtifactError, StillAnalysis
from services.live_video import LiveRecognitionSession
from services.settings_service import live_config, load_settings, recognition_thresholds
from services.student_service import StudentService
from ui.auth_ui import require_teacher
from ui.components import (
    artifact_problem_box,
    avatar,
    clock,
    format_confidence,
    page_header,
    session_create_form,
    session_label,
)
from ui.state import artifact_status, get_paths, get_recognizer, require_database
from utils.config import parse_resolution
from utils.image_utils import (
    COLOR_CANDIDATE,
    COLOR_RECOGNIZED,
    COLOR_REJECTED,
    COLOR_UNCERTAIN,
    InvalidImageError,
    decode_image_bytes,
    draw_face_box,
)

LIVE_OBJ_KEY = "live_recognition_session"
PROCESSED_KEY = "live_processed_events"
RESULT_KEY = "live_last_result"
STILL_KEY = "still_analysis_cache"
WAS_PLAYING_KEY = "live_was_playing"
PREFERRED_KEY = "live_session_preferred"
FRAME_TIMEOUT_S = 3.0
RESULT_DISPLAY_S = 20.0
STUN = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

user = require_teacher()
page_header("Live Attendance",
            "Show one face at a time to the camera — recognised students are marked Present automatically.")

db = require_database()
paths = get_paths()
settings = load_settings(db)
log_events = bool(settings["privacy.log_recognition_events"])
service = AttendanceService(db, log_events=log_events)
students = StudentService(db, paths.dataset_dir)
auto_mark = bool(settings["recognition.auto_mark"])
as_percent = bool(settings["display.confidence_as_percent"])

# =====================================================================================
# Helpers
# =====================================================================================


def set_result(kind: str, message: str) -> None:
    st.session_state[RESULT_KEY] = {"kind": kind, "message": message, "at": time.time(), "new": True}


def show_result() -> None:
    result = st.session_state.get(RESULT_KEY)
    if not result or time.time() - result["at"] > RESULT_DISPLAY_S:
        return
    icon = {"success": ":material/check_circle:", "info": ":material/info:", "warning": ":material/warning:",
            "error": ":material/error:"}[result["kind"]]
    getattr(st, result["kind"])(result["message"], icon=icon)
    if result.get("new"):
        st.toast(result["message"], icon=icon)
        result["new"] = False


def current_status(session: dict[str, Any] | None) -> str:
    return suggested_status(session, mark_late=bool(settings["session.mark_late"]),
                            late_after_minutes=int(settings["session.late_after_minutes"]))


def identity_card(resolution: IdentityResolution, confidence: float | None, label: str) -> None:
    """Name / ID / photo only for an active, consenting, linked student."""
    if resolution.may_show_identity and resolution.student is not None:
        student = resolution.student
        pic, info = st.columns([1, 2])
        with pic:
            avatar(students.photo(student), student["full_name"])
        with info:
            st.markdown(f'<div class="sas-identity">{html.escape(student["full_name"])}</div>', unsafe_allow_html=True)
            st.caption(f"Student ID **{student['student_id']}** · {student['class_name']}")
            st.metric(label, format_confidence(confidence, as_percent))
    else:
        st.markdown("#### :material/help: Unknown / not enrolled")
        st.caption("No name is shown: the face does not match an active, consenting, enrolled student.")
    if resolution.status is IdentityStatus.ALREADY_RECORDED and resolution.existing:
        st.success(f"{resolution.student['full_name']} is marked **{resolution.existing['status']}** "
                   f"at {clock(resolution.existing['recorded_at'])}.", icon=":material/check_circle:")
    elif resolution.status is IdentityStatus.ELIGIBLE:
        st.info("Recognised — ready to mark Present.", icon=":material/how_to_reg:")
    else:
        st.warning(resolution.message, icon=":material/warning:")


def top_scores(analysis: FrameAnalysis | None) -> None:
    if not settings["display.show_top_predictions"] or analysis is None or analysis.prediction is None:
        return
    rows = [{"Model class": lab, "Score": round(p * 100, 1)} for lab, p in analysis.prediction.top_k]
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"Score": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Softmax scores of the CNN for this frame (not calibrated probabilities).")


def _undo(*, attendance_id: int, database: Database, log: bool) -> None:
    result = AttendanceService(database, log_events=log).undo_mark(attendance_id)
    set_result("info" if result.ok else "error", result.message)
    live: LiveRecognitionSession | None = st.session_state.get(LIVE_OBJ_KEY)
    if live is not None:
        live.reset_stability()


def session_roster_panel(session_id: int | None) -> None:
    """Who is already marked in this session (with Undo for camera marks) and who is still missing."""
    if session_id is None:
        return
    roster = SessionRepository(db).roster(session_id)
    present = sorted((r for r in roster if r["status"] in ("present", "late")),
                     key=lambda r: str(r["recorded_at"]), reverse=True)
    waiting = [r["full_name"] for r in roster if not r["status"]]
    st.markdown(f"**This session: {len(present)} / {len(roster)} present**")
    for row in present[:10]:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        source = "camera" if row["method"] == "face_recognition" else "manual"
        score = f" · {float(row['confidence']):.0%}" if row["confidence"] is not None else ""
        c1.markdown(f":green[:material/check_circle:] **{html.escape(row['full_name'])}** · "
                    f"{row['status']} {clock(row['recorded_at'])} · {source}{score}")
        if row["method"] == "face_recognition":
            c2.button("Undo", key=f"undo_{row['attendance_id']}", on_click=_undo,
                      kwargs={"attendance_id": int(row["attendance_id"]), "database": db, "log": log_events},
                      help="Remove this camera mark (e.g. a wrong match).")
    if waiting:
        st.caption("Waiting for: " + ", ".join(waiting))


# =====================================================================================
# Session bar
# =====================================================================================


@st.dialog("End attendance session")
def end_session_dialog(session: dict[str, Any], operator: str) -> None:
    st.write(f"**{session_label(session)}**")
    st.write(f"{session['not_marked_count']} student(s) without a mark will be recorded as **absent**. "
             "The camera can no longer mark this session; records can still be corrected afterwards.")
    c1, c2 = st.columns(2)
    if c1.button("End session", type="primary", key="dlg_end_yes"):
        try:
            inserted = service.close_session(int(session["id"]), operator=operator)
        except DatabaseError as exc:
            st.error(str(exc))
            return
        set_result("success", f"Session ended — {inserted} absence record(s) created.")
        st.rerun()
    if c2.button("Cancel", key="dlg_end_no"):
        st.rerun()


open_sessions = service.open_sessions()
default_class = str(settings["session.default_class_name"])
class_options = students.class_names() or [default_class]
session_id: int | None = None
session: dict[str, Any] | None = None

if not open_sessions:
    with st.container(border=True):
        st.markdown("#### :material/event_available: No attendance session is running")
        st.caption("Start a session so the camera can mark students present. Students without a mark become "
                   "absent only when you end the session.")
        c1, c2, c3 = st.columns([2, 1.2, 1.2], vertical_alignment="bottom")
        quick_class = c1.selectbox("Class", class_options,
                                   index=class_options.index(default_class) if default_class in class_options else 0,
                                   key="live_quick_class")
        if c2.button("Start today's session", type="primary", icon=":material/play_arrow:", key="live_quick_start"):
            try:
                new_id = service.start_quick_session(class_name=quick_class, operator=user.username)
            except DatabaseError as exc:
                st.error(str(exc))
            else:
                st.session_state[PREFERRED_KEY] = new_id
                set_result("success", "Session started. Press START under the video.")
                st.rerun()
        with c3.popover(":material/tune: Custom session"):
            created = session_create_form(service, user.username, key="live_custom_session",
                                          class_options=class_options, default_class=default_class)
            if created:
                st.session_state[PREFERRED_KEY] = created
                st.rerun()
else:
    by_id = {int(s["id"]): s for s in open_sessions}
    ids = list(by_id)
    preferred = st.session_state.get(PREFERRED_KEY, int(settings["session.active_session_id"]))
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    session_id = c1.selectbox("Running session", ids, index=ids.index(preferred) if preferred in by_id else 0,
                              format_func=lambda k: session_label(by_id[k]))
    st.session_state[PREFERRED_KEY] = session_id
    session = by_id[session_id]
    with c2.popover(":material/add: New session"):
        created = session_create_form(service, user.username, key="live_new_session", class_options=class_options,
                                      default_class=default_class)
        if created:
            st.session_state[PREFERRED_KEY] = created
            st.rerun()
    if c3.button("End session", icon=":material/stop_circle:", key="live_end_session"):
        end_session_dialog(session, user.username)

# =====================================================================================
# Model
# =====================================================================================
status = artifact_status()
if not status.ready:
    artifact_problem_box(status)
    st.stop()
try:
    recognizer: FaceRecognizer = get_recognizer(status)
except ModelArtifactError as exc:
    st.error(f"**The trained model could not be loaded.**\n\n{exc}", icon=":material/error:")
    st.stop()
except Exception as exc:  # noqa: BLE001 - e.g. a broken PyTorch installation
    st.error(f"**PyTorch failed while loading the model:** {exc}", icon=":material/error:")
    st.caption("Check that the packages in requirements.txt are installed in the Python you used to start the app.")
    st.stop()

overlay_names = students.overlay_names(recognizer.class_names)
if not overlay_names:
    st.warning("No student is currently recognisable — on the **Students** page, link model classes to students "
               "and record consent.", icon=":material/link_off:")
thresholds = recognition_thresholds(settings)

m1, m2 = st.columns([3, 2], vertical_alignment="center")
mode = m1.radio("Camera mode", ["Live video", "Still photo", "Upload a photo"], horizontal=True, key="live_mode",
                help="Live video = continuous recognition. Still photo / upload = fallback when live video does not work.")
m2.caption(f"{'**Auto-mark ON**: recognised students are marked Present automatically.' if auto_mark else '**Auto-mark OFF**: click Confirm for each student.'} "
           f"Model on **{recognizer.bundle.device}** · threshold {thresholds.confidence_threshold:.0%} · "
           f"{settings['recognition.stability_frames']} frames. Change in Settings.")

# =====================================================================================
# Live video
# =====================================================================================


def _confirm_live(*, event_id: int, label: str, confidence: float, session_id: int, status_key: str,
                  operator: str, database: Database, log: bool) -> None:
    live: LiveRecognitionSession | None = st.session_state.get(LIVE_OBJ_KEY)
    if live is None:
        set_result("error", "The live camera session has ended. Press START again.")
        return
    snap = live.snapshot().stability
    if not (snap.is_stable and snap.event_id == event_id and snap.label == label):
        set_result("warning", "The face changed before confirmation. Hold still and try again.")
        return
    result = AttendanceService(database, log_events=log).mark_recognised(
        label=label, confidence=confidence, session_id=session_id, status=st.session_state.get(status_key, "present"),
        operator=operator, source="live_video", automatic=False)
    set_result("success" if result.ok else ("info" if result.duplicate else "error"), result.message)
    live.reset_stability()


def _reject_live(*, label: str, confidence: float | None, session_id: int | None, database: Database, log: bool) -> None:
    AttendanceService(database, log_events=log).reject(label=label, confidence=confidence, session_id=session_id,
                                                       source="live_video")
    live: LiveRecognitionSession | None = st.session_state.get(LIVE_OBJ_KEY)
    if live is not None:
        live.reset_stability()
    set_result("info", "Rejected — nothing was recorded.")


def _render_live_panel(playing: bool, session_id: int | None) -> None:
    show_result()
    live: LiveRecognitionSession | None = st.session_state.get(LIVE_OBJ_KEY)
    if live is None or not playing:
        st.info("Press **START** under the video and allow the camera in your browser. "
                "Then let each student look at the camera, one at a time.", icon=":material/videocam:")
        session_roster_panel(session_id)
        return

    snap = live.snapshot()
    since = snap.seconds_since_last_frame()
    if since is None:
        st.info("Waiting for the first camera frame…", icon=":material/hourglass_top:")
        return
    if since > FRAME_TIMEOUT_S:
        st.error(f"No video for {since:.0f} s — the camera may be disconnected or used by another app. "
                 "Press STOP, check the camera, then START.", icon=":material/videocam_off:")
        return

    analysis, stability = snap.analysis, snap.stability
    if analysis is None:
        st.caption("Analysing…")
    elif analysis.status is FrameStatus.NO_FACE:
        st.info(analysis.message, icon=":material/face:")
    elif analysis.status is FrameStatus.MULTIPLE_FACES:
        st.warning(analysis.message, icon=":material/groups:")
    elif analysis.status is FrameStatus.UNCERTAIN:
        st.warning(analysis.message, icon=":material/help:")
    elif analysis.status in (FrameStatus.ERROR, FrameStatus.INVALID_IMAGE):
        st.error(analysis.message, icon=":material/error:")
    elif not stability.is_stable:
        st.caption(":material/pending: Face recognised in this frame — verifying over several frames…")
    st.progress(stability.progress, text=f"Verifying: {stability.consecutive}/{stability.required} frames")
    timing = f"{snap.processing_ms:.0f} ms per analysed frame · " if snap.processing_ms else ""
    st.caption(f"{timing}{snap.frames_analyzed} analysed / {snap.frames_received} frames")
    top_scores(analysis)

    if stability.is_stable and stability.label is not None:
        resolution = service.resolve_identity(stability.label, session_id)
        processed: set = st.session_state.setdefault(PROCESSED_KEY, set())
        event_key = (session_id, stability.event_id)
        if auto_mark and resolution.can_mark and session_id is not None and event_key not in processed:
            processed.add(event_key)
            result = service.mark_recognised(
                label=stability.label, confidence=float(stability.mean_confidence or 0.0), session_id=session_id,
                status=current_status(session), operator=user.username, source="live_video", automatic=True)
            set_result("success" if result.ok else ("info" if result.duplicate else "error"), result.message)
            resolution = service.resolve_identity(stability.label, session_id)
        with st.container(border=True):
            identity_card(resolution, stability.mean_confidence, "Average score (verified frames)")
            if not auto_mark and resolution.can_mark and session_id is not None:
                status_key = f"live_status_{stability.event_id}"
                default = current_status(session)
                st.radio("Record as", ["present", "late"], index=0 if default == "present" else 1, horizontal=True,
                         key=status_key, format_func=str.capitalize)
                c1, c2 = st.columns(2)
                common = {"database": db, "log": log_events}
                c1.button("Confirm attendance", type="primary", icon=":material/check:", width="stretch",
                          key=f"live_confirm_{stability.event_id}", on_click=_confirm_live,
                          kwargs={"event_id": stability.event_id, "label": stability.label,
                                  "confidence": float(stability.mean_confidence or 0.0), "session_id": session_id,
                                  "status_key": status_key, "operator": user.username, **common})
                c2.button("Not this person", icon=":material/close:", width="stretch",
                          key=f"live_reject_{stability.event_id}", on_click=_reject_live,
                          kwargs={"label": stability.label, "confidence": stability.mean_confidence,
                                  "session_id": session_id, **common})
    st.divider()
    session_roster_panel(session_id)


@st.fragment(run_every=0.5)
def live_panel_running(playing: bool, session_id: int | None) -> None:
    _render_live_panel(playing, session_id)


@st.fragment
def live_panel_idle(playing: bool, session_id: int | None) -> None:
    _render_live_panel(playing, session_id)


def render_live_mode() -> None:
    try:
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("**Live video needs the `streamlit-webrtc` package**, which is not installed. Run "
                 "`python project_setup.py` (or install requirements-app.txt), or use the Still photo mode.",
                 icon=":material/videocam_off:")
        return

    config = live_config(settings, overlay_names)
    live: LiveRecognitionSession | None = st.session_state.get(LIVE_OBJ_KEY)
    if live is None or live.recognizer is not recognizer:
        live = LiveRecognitionSession(recognizer, config)
        st.session_state[LIVE_OBJ_KEY] = live
        st.session_state[PROCESSED_KEY] = set()
    else:
        live.configure(config)

    def video_frame_callback(frame: "av.VideoFrame") -> "av.VideoFrame":
        # WebRTC worker thread: no Streamlit calls and no database access here.
        annotated = live.process_frame(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    width, height = parse_resolution(settings["camera.resolution"])
    video_col, panel_col = st.columns([3, 2], gap="large")
    with video_col:
        ctx = webrtc_streamer(
            key="sas-live-attendance",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=STUN if settings["camera.use_stun_server"] else None,
            media_stream_constraints={
                "video": {"width": {"ideal": width}, "height": {"ideal": height},
                          "frameRate": {"ideal": int(settings["camera.frame_rate"])}},
                "audio": False,
            },
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
        with st.expander("Camera not working?", icon=":material/build:"):
            st.markdown(
                "- **Permission denied:** click the camera icon in the address bar → *Allow* for `localhost` → START.\n"
                "- **Camera busy:** close Zoom / Teams / the Windows Camera app.\n"
                "- **Wrong camera:** use *SELECT DEVICE* under the video.\n"
                "- **Stuck on connecting:** Settings → Camera → turn ON *Use a public STUN server*, then reload.\n"
                "- **Still failing:** use the *Still photo* mode above.\n"
                "- **Recognition:** good front lighting, face the camera, about 50–80 cm away, one person in view."
            )
    playing = bool(ctx.state.playing)
    if st.session_state.get(WAS_PLAYING_KEY) and not playing:
        live.reset_all()
    st.session_state[WAS_PLAYING_KEY] = playing
    with panel_col:
        if playing:
            live_panel_running(playing, session_id)
        else:
            live_panel_idle(playing, session_id)


# =====================================================================================
# Still photo / upload (fallback)
# =====================================================================================


def _annotate(image, still: StillAnalysis):
    out = image.copy()
    color = {FrameStatus.CANDIDATE: COLOR_RECOGNIZED, FrameStatus.UNCERTAIN: COLOR_UNCERTAIN,
             FrameStatus.MULTIPLE_FACES: COLOR_REJECTED}.get(still.analysis.status, COLOR_CANDIDATE)
    for box in still.analysis.faces:
        draw_face_box(out, box, None, color, thickness=max(2, image.shape[1] // 320))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _confirm_still(*, digest: str, label: str, confidence: float, session_id: int, status_key: str, operator: str,
                   database: Database, log: bool) -> None:
    cache = st.session_state.get(STILL_KEY)
    if not cache or cache["digest"] != digest:
        set_result("warning", "The photo changed — nothing was recorded.")
        return
    result = AttendanceService(database, log_events=log).mark_recognised(
        label=label, confidence=confidence, session_id=session_id, status=st.session_state.get(status_key, "present"),
        operator=operator, source="still_image", automatic=False)
    cache["done"] = True
    set_result("success" if result.ok else ("info" if result.duplicate else "error"), result.message)


def render_still_mode(source: str) -> None:
    st.caption(":material/photo_camera: **Still-photo recognition** — a single image instead of live video. The face "
               "is classified under several slightly shifted crops and all must agree. Photos are never saved.")
    if source == "camera":
        photo = st.camera_input("Take a photo (one student, facing the camera)", key="still_camera")
    else:
        photo = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "bmp", "webp"], key="still_upload")
    show_result()
    if photo is None:
        session_roster_panel(session_id)
        return
    data = photo.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    variants = int(settings["recognition.still_image_variants"])
    cache = st.session_state.get(STILL_KEY)
    if not cache or cache["digest"] != digest or cache.get("settings") != (thresholds, variants):
        try:
            image = decode_image_bytes(data)
        except InvalidImageError as exc:
            st.error(str(exc), icon=":material/broken_image:")
            return
        with st.spinner("Detecting and recognising…"):
            still = recognizer.analyze_still(image, thresholds, variants)
        cache = {"digest": digest, "still": still, "rgb": _annotate(image, still), "settings": (thresholds, variants),
                 "done": False}
        st.session_state[STILL_KEY] = cache
        if still.analysis.status in (FrameStatus.UNCERTAIN, FrameStatus.NO_FACE):
            pred = still.analysis.prediction
            service.log_still_outcome(status="uncertain" if pred else "unknown", label=pred.label if pred else None,
                                      confidence=pred.confidence if pred else None, session_id=session_id)

    still: StillAnalysis = cache["still"]
    analysis = still.analysis
    img_col, res_col = st.columns([3, 2], gap="large")
    img_col.image(cache["rgb"], caption="Analysed photo (not stored)", width="stretch")
    with res_col:
        if analysis.status is FrameStatus.NO_FACE:
            st.info(analysis.message, icon=":material/face:")
            return
        if analysis.status is FrameStatus.MULTIPLE_FACES:
            st.warning(analysis.message + " Retake the photo with one student.", icon=":material/groups:")
            return
        if analysis.status in (FrameStatus.ERROR, FrameStatus.INVALID_IMAGE):
            st.error(analysis.message, icon=":material/error:")
            return
        st.progress(still.agreement, text=f"Crop agreement: {still.agreement:.0%}")
        top_scores(analysis)
        if analysis.status is FrameStatus.UNCERTAIN or analysis.prediction is None:
            st.warning(analysis.message, icon=":material/help:")
            return
        label, score = analysis.prediction.label, analysis.prediction.confidence
        resolution = service.resolve_identity(label, session_id)
        if auto_mark and resolution.can_mark and session_id is not None and not cache.get("done"):
            cache["done"] = True
            result = service.mark_recognised(label=label, confidence=score, session_id=session_id,
                                             status=current_status(session), operator=user.username,
                                             source="still_image", automatic=True)
            set_result("success" if result.ok else ("info" if result.duplicate else "error"), result.message)
            show_result()
            resolution = service.resolve_identity(label, session_id)
        with st.container(border=True):
            identity_card(resolution, score, "Score (single photo)")
            if not auto_mark and resolution.can_mark and session_id is not None and not cache.get("done"):
                status_key = f"still_status_{digest}"
                default = current_status(session)
                st.radio("Record as", ["present", "late"], index=0 if default == "present" else 1, horizontal=True,
                         key=status_key, format_func=str.capitalize)
                st.button("Confirm attendance", type="primary", icon=":material/check:", width="stretch",
                          key=f"still_confirm_{digest}", on_click=_confirm_still,
                          kwargs={"digest": digest, "label": label, "confidence": float(score),
                                  "session_id": session_id, "status_key": status_key, "operator": user.username,
                                  "database": db, "log": log_events})
    session_roster_panel(session_id)


# =====================================================================================
# Dispatch
# =====================================================================================
if mode == "Live video":
    render_live_mode()
elif mode == "Still photo":
    render_still_mode("camera")
else:
    render_still_mode("upload")

with st.expander("How the camera decides", icon=":material/rule:"):
    st.markdown(
        f"""
1. **Detect** the face with OpenCV's Haar cascade (faces smaller than {settings['recognition.min_face_size_px']} px are
   ignored). More than one face → nothing is recognised (one student at a time).
2. **Prepare** the face exactly like the notebook: `attendance_utils.preprocess_image` → grayscale, 128×128, ÷255.
3. **Classify** with the trained CNN (`models/attendance_cnn.pt`) on **{recognizer.bundle.device}**.
4. **Accept** a frame only if the top score ≥ **{thresholds.confidence_threshold:.0%}**; otherwise it shows *Unknown*.
5. **Verify**: the same student in **{settings['recognition.stability_frames']} consecutive** analysed frames, with the face
   staying in roughly the same place.
6. **Mark**: {'automatically **Present** (teacher can Undo)' if auto_mark else 'after the teacher clicks **Confirm**'}.
   SQLite's UNIQUE(student, session) rule makes double marking impossible.

**Limitations:** the CNN only knows the {len(recognizer.class_names)} trained students, so a stranger is still matched to
one of them (the threshold and frame check reduce this). There is no liveness detection — a printed photo could be
recognised. Use Undo for any wrong mark.
"""
    )
