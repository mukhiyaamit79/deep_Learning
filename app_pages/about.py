"""Project information, architectures, methodology, limitations and privacy notice."""

from __future__ import annotations

import streamlit as st

from ui.components import page_header
from ui.state import artifact_status
from ui.theme import APP_SUBTITLE, APP_TITLE, COURSE

page_header(APP_TITLE, APP_SUBTITLE, kicker=f"{COURSE} · About this project")
status = artifact_status()
metadata = status.metadata or {}

st.markdown("**Smart Attendance System Using Deep Learning: Comparative Evaluation of CNN, RNN, and LSTM** — a "
            f"{COURSE} project. The notebook `amit_eda.ipynb` explores the face dataset, trains and compares three "
            "PyTorch architectures, and exports the selected CNN, which this web app uses to mark attendance.")
st.warning("**Academic prototype.** Not a validated or secure attendance system. Do not use it for real decisions "
           "about students.", icon=":material/school:")

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("##### :material/build: Technologies")
    st.markdown("- Python, **PyTorch** (training on CUDA in the notebook; CUDA or CPU in the app)\n"
                "- OpenCV (Haar-cascade face detection), Pillow (exact training preprocessing)\n"
                "- Streamlit + streamlit-webrtc (local web app, browser webcam)\n"
                "- SQLite (Python `sqlite3`) · pandas · scikit-learn · Plotly")
with c2, st.container(border=True):
    st.markdown("##### :material/photo_library: Dataset")
    sizes = metadata.get("split_sizes") or {}
    st.markdown(f"- {metadata.get('number_of_classes', '?')} students "
                f"({', '.join(status.class_names) if status.class_names else '?'}), "
                f"{metadata.get('dataset_images', '?')} grayscale face crops (originally 200×200)\n"
                f"- Split ({metadata.get('split_protocol', '?')}): train {sizes.get('train', '?')} / "
                f"validation {sizes.get('validation', '?')} / test {sizes.get('test', '?')}\n"
                "- Model input: Pillow grayscale, bilinear resize to 128×128, ÷255\n"
                "- Layout: `dataset/<student_id>/<image>.jpg`")

st.subheader("Architectures")
arch = metadata.get("architecture_config") or {}
filters = arch.get("filters") or [32, 64, 128]
a1, a2, a3 = st.columns(3)
with a1, st.container(border=True):
    st.markdown("##### CNN (deployed)")
    st.caption(f"Input 1×128×128 → three blocks of [Conv 3×3 → ReLU → Conv 3×3 → ReLU → MaxPool → Dropout] with "
               f"{' / '.join(map(str, filters))} filters → global average pooling → Dense {arch.get('dense_units', 128)} "
               f"(ReLU, dropout {arch.get('dropout', 0.3)}) → {arch.get('num_classes', len(status.class_names or []))} "
               "logits → softmax.")
with a2, st.container(border=True):
    st.markdown("##### SimpleRNN")
    st.caption("Input 128 rows × 128 pixels → `torch.nn.RNN` (64 units) → last hidden state → Dropout → Dense 128 (ReLU) "
               "→ softmax. Reads the face top-to-bottom, one row per step.")
with a3, st.container(border=True):
    st.markdown("##### LSTM")
    st.caption("Same row-sequence input → `torch.nn.LSTM` (64 units) → Dropout → Dense 128 (ReLU) → softmax. Gates help "
               "keep information across the 128 rows.")

st.subheader("Methodology")
st.markdown(
    "1. **EDA** — dataset card, quality audit, class balance, image sizes, sample grids, intensity histograms, mean faces.\n"
    "2. **Preprocessing** shared by notebook and app (`attendance_utils.preprocess_image`).\n"
    "3. **One stratified 70/15/15 split** (seed 42) for all three architectures; augmentation on training data only.\n"
    "4. **Five CNN experiments** (baseline, lower LR, higher dropout, larger model, larger batch); configuration chosen on "
    "validation data.\n"
    "5. **RNN and LSTM** trained with the same controls; early stopping restores the best validation weights.\n"
    "6. **Held-out test evaluation**: accuracy, macro/weighted precision, recall, F1, per-student metrics, confusion "
    "matrices, training time, parameters, latency.\n"
    "7. **Deployment**: the CNN bundle is saved (NN-36) and loaded by this app, which adds face detection, multi-frame "
    "verification, automatic Present marking, SQLite storage, reports and teacher/student logins."
)

st.subheader("Known limitations")
st.markdown(
    "- **Closed set:** the CNN only knows the trained students; a stranger is matched to the most similar one. The 90% "
    "threshold and the 8-frame check reduce, but do not prevent, false matches — the teacher can Undo.\n"
    "- **New students need retraining:** add their photos to `dataset/<id>/` and re-run the notebook.\n"
    "- **Small dataset:** 50 images per student and a 38-image test set; one image ≈ 2.6 percentage points.\n"
    "- **Domain shift:** webcam lighting, pose and the Haar crop differ from the training photos.\n"
    "- **No liveness detection:** a printed photo of a student could be recognised.\n"
    "- **Demo logins:** the passwords are public; this is not production authentication."
)

st.subheader("Privacy and consent")
st.info(
    "- Only students with recorded consent are marked by the camera; everyone else is marked manually. The five project "
    "members in `dataset/` are seeded with consent because they supplied their photos for the project — the teacher "
    "can withdraw it on the Students page.\n"
    "- **No camera frames or photos are stored.** The recognition log keeps only the label, score and outcome.\n"
    "- **Everything stays on this computer** (model, database, processing). The app listens on localhost only.\n"
    "- Students can see only their own attendance. Keep the dataset and database out of public repositories.",
    icon=":material/privacy_tip:",
)
