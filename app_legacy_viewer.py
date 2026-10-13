"""Optional local presentation interface; no continuous camera or attendance writes."""
from attendance_runtime import configure_runtime
configure_runtime()

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

from attendance_utils import PROJECT_ROOT, load_bundle, predict_face

st.set_page_config(page_title="TECH 405 Smart Attendance", layout="wide")
st.title("Smart Attendance System")
st.caption("TECH 405 · Amit · CNN, SimpleRNN and LSTM comparison")
st.write("An academic comparison of enrolled-student face classifiers, with an optional local "
         "attendance demonstration. CNN uses spatial filters; recurrent baselines interpret spatial image rows.")
st.warning("Softmax scores are not calibrated identity probabilities. Unknown people and presentation "
           "attacks may receive high scores. This interface never writes attendance.")


@st.cache_resource
def cached_bundle(directory, fingerprint):
    return load_bundle(directory)


tabs = st.tabs(["Face prediction", "Model comparison", "Attendance records"])
with tabs[0]:
    model_dir = PROJECT_ROOT / "models"
    try:
        bundle_files = [model_dir / name for name in
                        ["attendance_cnn.pt", "class_names.json", "preprocessing_config.json", "model_metadata.json"]]
        fingerprint = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in bundle_files)
        model, names, config = cached_bundle(str(model_dir), fingerprint)
        st.caption(f"PyTorch inference device: {next(model.parameters()).device}")
        consent = st.checkbox("I am the enrolled student, or have their explicit permission to process this image locally.")
        source = st.radio("Image source", ["Upload face crop", "Camera still photo"], horizontal=True)
        uploaded = (st.file_uploader("Upload one tightly cropped face", type=["jpg", "jpeg", "png", "webp"])
                    if source == "Upload face crop" else st.camera_input("Take a still photo; keep one face close to the camera"))
        st.caption("Use a single face crop matching training. A full camera frame can introduce a cropping mismatch. "
                   "Continuous detection and confirmed attendance are in live_attendance.py.")
        if uploaded is not None and consent:
            with Image.open(uploaded) as opened:
                face = opened.convert("RGB")
            st.image(face, caption="Provided image", width=320)
            if st.button("Predict face identity"):
                result = predict_face(model, names, config, face)
                st.subheader(result["student_id"])
                st.write(f"Softmax score: {result['confidence']:.3f}; threshold: {config['confidence_threshold']:.2f}")
                st.dataframe(pd.DataFrame(result["top_classes"], columns=["Student ID", "Softmax score"]),
                             hide_index=True)
                st.info("No attendance was written. Use the local webcam script for opt-in, confirmed attendance.")
        elif uploaded is not None:
            st.info("Consent is required before processing the image.")
    except (OSError, ValueError, RuntimeError, UnidentifiedImageError) as exc:
        st.info(f"Model or image unavailable: {exc}. Run notebook NN-36 to generate the model bundle.")

with tabs[1]:
    st.write("CNN: repeated convolution/pooling/dropout blocks and global average pooling. "
             "SimpleRNN: a recurrent hidden state over spatial rows. LSTM: gated memory over the same rows. "
             "Validation chooses configurations before final testing; CNN is the predefined demonstration family.")
    results_dir = PROJECT_ROOT / "results"
    comparison_file = results_dir / "model_comparison.csv"
    if comparison_file.is_file():
        try:
            comparison = pd.read_csv(comparison_file)
            st.dataframe(comparison, hide_index=True)
            st.bar_chart(comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]])
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"Cannot read comparison: {exc}")
    else:
        st.info("Run the notebook through NN-39 to export real comparison results.")
    for path in sorted((results_dir / "figures").glob("*.png")):
        with st.expander(path.stem.replace("_", " ").title()):
            st.image(str(path))

with tabs[2]:
    st.caption("Private local viewer only. Do not expose this Streamlit server publicly.")
    if st.checkbox("I am authorized to view these student attendance records"):
        files = sorted((PROJECT_ROOT / "attendance").glob("attendance_*.csv"), reverse=True)
        if files:
            selected = st.selectbox("Attendance date", files, format_func=lambda p: p.stem)
            try:
                st.dataframe(pd.read_csv(selected, dtype={"Student ID": str}), hide_index=True)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Cannot read attendance file: {exc}")
        else:
            st.info("No confirmed attendance records yet.")
