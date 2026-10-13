# Smart Attendance System: CNN vs RNN vs LSTM

TECH 405 · Amit · Comparative face-image classification and an opt-in local attendance demonstration.

This project extends `amit_eda.ipynb`; `calculus.ipynb` is unrelated and unchanged. The original Week 2 EDA, sample grids, intensity histograms, mean faces and five CNN experiments are preserved. The neural-network section now uses PyTorch with CUDA. New sections add genuine `nn.RNN` and `nn.LSTM` baselines, shared-split evaluation, efficiency measurements, saved-model inference and confirmed webcam attendance.

## Quick start — Smart Attendance web app

The notebook has already trained and exported the CNN (`models/`) and the comparison results (`results/`).
The web app uses them directly:

```powershell
python project_setup.py      # Anaconda base: installs the missing app packages, keeps CUDA PyTorch
python run_app.py            # opens http://localhost:8501
```

Demo logins: teacher `teacher` / `teacher123`; students `Amit`, `Shreejal`, `Susman`, `Udhav`, `Uttam` /
`student123`. **Step-by-step instructions for VS Code on Windows: [HOW_TO_RUN.md](HOW_TO_RUN.md).**

| Web app page | What it does |
|---|---|
| Sign in | Separate Teacher / Student tabs (dummy accounts, hashed passwords in SQLite) |
| Dashboard (teacher) | Today's class as photo cards with Present / Not yet / Absent, KPIs, 30-day trend, rate per student, recent activity, system status |
| Live Attendance | Browser webcam (streamlit-webrtc) → OpenCV face detection → your CNN → 8 stable frames ≥ 0.90 → **automatically marked Present** in SQLite, with Undo; still-photo and upload fallback |
| Attendance Records | Filters, search, CSV export, sessions (start / end → absences), manual entry, audited corrections, recognition log |
| Reports | Daily, weekly, monthly, per student, per class, per session — charts + CSV |
| Students | Photo directory, register / edit / deactivate, consent, link to model classes, login accounts |
| CNN vs RNN vs LSTM | Reads `results/` (NN-39): metrics, per-student F1, timing, figures, deployment decision |
| Settings / My account / About | Thresholds, auto-mark, camera; passwords; methodology, limitations, privacy |
| My attendance (student) | The signed-in student's own rate, history and monthly chart |

The app reuses the notebook's code: `attendance_models.AttendanceCNN`, `attendance_utils.preprocess_image`
and the same model-file checks as `attendance_utils.load_bundle`. Unlike the notebook, the app falls back to
the **CPU** when CUDA is unavailable (inference only; training still requires CUDA).

**Status:** source prepared for your friend's Run All workflow. On the development computer, the existing CUDA PyTorch build performed a GPU tensor calculation after the MKL runtime setting was applied; missing base dependencies were installed. Nothing has been installed or tested on your friend's computer. No notebook cells or model training were run during this setup work. Existing EDA outputs are historical; training results must come from your friend's actual execution.

## Dataset and privacy

Use only consented local images in `dataset/student_id/image.jpg`. Runtime discovery supports JPG/JPEG/PNG/WEBP. The historical collection had 250 grayscale images across five identities, 50 per student, originally 200 × 200; current counts and properties are recomputed. Training uses Pillow grayscale conversion, bilinear resize to 128 × 128 and float32 division by 255. The ordered label mapping comes from one `LabelEncoder`.

Biometric enrollment must be voluntary and authorized before collection/training. Explain local storage, purpose, access and retention/deletion arrangements; offer manual attendance. Demo opt-in is additional, not a substitute for enrollment consent. The webcam saves no frames. CSV files and models remain private local artifacts. Closing the demo does not delete earlier records or model files.

**Do not upload the face dataset or identifiable records to GitHub.** Existing notebook outputs contain private face images/IDs: preserve the local notebook, but clear all outputs in a separate public copy and review source text, metadata and attachments for identifiers before publishing. `.gitignore` cannot protect data already tracked by Git, embedded in notebooks, or committed in history. No Git commit or push is part of this implementation.

## Files

```text
amit_eda.ipynb              EDA, CNN experiments, recurrent training, comparison and export
project_setup.py           First-cell dependency installation in Anaconda base
attendance_runtime.py       Native-library settings applied before NumPy/torch
attendance_models.py       Shared PyTorch CNN, RNN and LSTM architectures
attendance_training.py     CUDA training, augmentation, evaluation and timing
attendance_utils.py        Shared exact preprocessing and PyTorch model loading
live_attendance.py         Local continuous webcam, opt-in and confirmed CSV logging (OpenCV window)
app_legacy_viewer.py       The earlier simple Streamlit viewer (renamed from app.py; kept unchanged)
app.py                     NEW web app entry point: sign-in and role-based navigation
run_app.py                 NEW launcher: python run_app.py (applies attendance_runtime first)
app_pages/                 NEW web app pages (login, dashboards, live attendance, records, reports, ...)
services/                  NEW recognition (PyTorch), attendance, students, logins, reports, results
database/                  NEW SQLite schema, migrations and all SQL
ui/  utils/                NEW Streamlit helpers; config, image helpers, validation, password hashing
tests/                     NEW unit tests (pytest)
database_storage/          GENERATED by the web app: attendance.db (+ backups/)
requirements-cuda.txt      CUDA wheel source used automatically when torch is missing
requirements.txt           All dependencies (now incl. streamlit-webrtc, plotly); not verified through installation
requirements-app.txt       NEW web-app-only dependencies (CPU PyTorch for computers without NVIDIA)
README.md
HOW_TO_RUN.md              NEW step-by-step guide (VS Code, Windows)
DEMO_GUIDE.md
.streamlit/config.toml     NEW localhost-only server, no usage statistics
.vscode/                   NEW launch configuration "Smart Attendance: run web app"
.gitignore
dataset/                   PRIVATE input: student_id/image.jpg
  capture_sessions.csv     Optional metadata: filepath,session_id
models/                    GENERATED by notebook NN-36
  attendance_cnn.pt
  class_names.json
  preprocessing_config.json
  model_metadata.json
results/                   GENERATED by notebook execution
  model_comparison.csv
  per_class_metrics.csv
  experiment_configurations.csv
  validation_comparison.csv
  summary.json
  figures/
attendance/                GENERATED only after confirmed webcam attendance
  attendance_YYYY-MM-DD.csv
```

## Architectures and experiment controls

| Architecture | Representation | Implemented network |
|---|---|---|
| CNN | N × 1 × 128 × 128 (NCHW) | Three two-convolution ReLU blocks, max pooling/dropout, global average pooling, dense ReLU/dropout, softmax |
| SimpleRNN | 128 rows × 128 features | Genuine `torch.nn.RNN`, default 64 units, dropout, dense ReLU, softmax |
| LSTM | 128 rows × 128 features | Genuine `torch.nn.LSTM`, default 64 units, dropout, dense ReLU, softmax |

The models return logits for numerically stable cross-entropy; softmax is applied during inference. Source NumPy arrays remain NHWC for EDA, while loaders convert to PyTorch NCHW and recurrent NHW. GPU augmentation combines rotation/translation/zoom in one bilinear warp with border padding, flips horizontally and adds clipped brightness variation. Framework initializers, recurrent parameterization and scheduler/augmentation implementations differ from the original Keras code: regenerate all results in PyTorch instead of mixing frameworks.

CNN's local 2D filters, shared weights and hierarchical composition are a natural fit for spatial images. RNN compresses successive spatial rows into a hidden state. LSTM adds gated memory with greater complexity. Image rows are not temporal observations. These rationales do not imply CNN necessarily wins the measured comparison.

All architectures use one dataset, label mapping and split indices. The original approximately 70/15/15 stratified protocol and seed 42 remain the default. Training augmentation is identical in policy, with a dedicated seeded augmentation generator per experiment; recurrent inputs are squeezed only after augmentation. Validation and test images remain clean. Manual early stopping restores minimum-validation-loss weights, which are then evaluated on validation data. Adam and `CrossEntropyLoss` on logits, a 120-epoch ceiling, early-stopping patience 25 and ReduceLROnPlateau patience 10 are shared.

The five original CNN runs are Baseline, Lower LR, Higher Dropout, Larger Model and Larger Batch. CNN configuration selection uses restored validation accuracy, then validation loss; exact residual ties prefer fewer parameters, then experiment name. The recurrent baselines use that selected CNN's batch size, learning rate, head dropout and dense width. `EXTRA_RECURRENT_CONFIGS` is empty by default; optional variations must be specified before testing. CNN's five-run search versus one recurrent baseline per family is explicitly disclosed; this is not an equal-search-budget or equal-parameter-count architecture experiment.

RNN/LSTM training and model selection occur at **NN-10A–D, before NN-11 tests the CNN**. NN-29 reuses the CNN predictions and evaluates each selected recurrent model. Do not retune against these results. Metrics include test loss, accuracy, macro/weighted precision/recall/F1, per-class reports, common-order confusion matrices and a training-majority baseline. Training time covers the selected run; total search cost is reported separately.

Inference timing uses validation images, three warm-ups and twenty repetitions on the same device: median synchronized eager-call batch-one latency and common-batch throughput. Forward/softmax calls are bracketed by CUDA synchronization. Input transfers are excluded, as are detection, image preprocessing, I/O and UI time. Hardware, versions and actual batch size are exported. Seed control does not guarantee identical numbers across devices.

CNN is the predefined deployment family; its configuration is selected from validation results. Parameters and measured validation-input latency document suitability for the demo. The model with the highest observed **test** accuracy is reported separately, including ties. No test result changes the saved model.

### Optional session-aware evaluation

Create `dataset/capture_sessions.csv` with `filepath` relative to `dataset/` and a `session_id` for every model image. Each identity needs at least three distinct sessions. Set `USE_SESSION_SPLIT = True` in NN-2 **before** training, restart the kernel and rerun all models. The helper reserves at least one session for each split; proportions may differ from 70/15/15. Never fabricate session IDs from arbitrary image positions. This alternative is not enabled automatically. Near-duplicate frames within one capture burst must remain together.

## On your friend's computer: Anaconda base + Run All

Requirements: Windows, Anaconda **base** with Python 3.10–3.14, a CUDA-compatible NVIDIA GPU/driver, internet access for missing downloads, and permission to install packages into base. The setup checks the actual computer; no username, GPU model or Anaconda installation path is hardcoded.

1. Copy the notebook, all project Python files, both requirements files and the consented local dataset to your friend's computer. Keep the project files together.
2. Open `amit_eda.ipynb` in VS Code or Jupyter and select the **Anaconda base** Python interpreter/kernel. Its path should point to the root Anaconda installation, not `envs/...` or a standalone Python.
3. **Restart the kernel, then click Run All.**
4. The first Setup cell checks base, downloads only missing/incompatible dependencies with that kernel's exact Python interpreter, verifies CUDA in an isolated subprocess, and continues into EDA/training.
5. NN-0 prints the actual NVIDIA GPU, CUDA/PyTorch versions and training device. It must say `cuda:0`.

An existing compatible CUDA-enabled torch installation is reused. Missing torch comes from the [official PyTorch CUDA wheel index](https://pytorch.org/get-started/locally/), configured in `requirements-cuda.txt`. A CPU-only torch build is replaced with the CUDA build of the same public version. First-time downloads can be large. Already installed packages are not redownloaded just because Run All was pressed.

**No CPU fallback:** neural-network training, validation, prediction and benchmarking require CUDA. CPU image loading, plotting, CSV handling and checkpoint storage are still normal; completed models are stored on CPU between experiments to conserve VRAM, then moved back to CUDA for computation.

`attendance_runtime.py` configures the supported `MKL_THREADING_LAYER=SEQUENTIAL` setting before numerical imports to avoid the duplicate OpenMP-runtime issue observed with Anaconda/PyTorch. This changes CPU BLAS threading, not CUDA execution; it does not use `KMP_DUPLICATE_LIB_OK`. See [Intel's MKL threading guidance](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/call-onemkl-functions-from-multi-threaded-apps.html).

If setup updates a package already loaded in the kernel, it stops and asks for one restart rather than continuing with a stale native library. Internet outages, incompatible/missing NVIDIA drivers, insufficient VRAM, unavailable wheels, installation permissions or invalid datasets can still require attention. No software can guarantee error-free training on an uninspected computer. The setup will not reinstall drivers, replace Python, download private face data or switch to CPU.

Optional manual setup from **Anaconda Prompt with base active**:

```text
python project_setup.py
```

This installs/checks dependencies only; it does not train models. There is no need for a separate virtual environment.

## Run the notebook top to bottom

1. Open `amit_eda.ipynb`, select **Anaconda base**, restart the kernel, and press Run All. Setup installs missing dependencies; NN-0 must report `cuda:0` and your friend's GPU.
2. Keep the notebook, all helper modules and dataset in this project folder. Setup searches the current folder and its parents; set `PROJECT_ROOT_OVERRIDE` in Setup if your kernel starts elsewhere.
3. Review collection consent and EDA, resolve corrupt files or exact duplicates, and inspect image crops. The loader stops rather than silently dropping problematic files.
4. Choose the split protocol and optional recurrent variants before examining new test results. Keep seed 42. Restart the kernel, then run from the top in order.
5. NN-7–8 train all five CNN experiments. NN-10 selects the CNN; NN-10A–D train/select RNN/LSTM and freeze the deployment policy. Training duration depends on hardware and early stopping; no fixed runtime is promised.
6. NN-11–21 retain the original CNN analysis. NN-22–35 produce the comparative analysis. Graph cells save figures when `SAVE_FIGURES = True`.
7. NN-36 saves and reloads the final CNN bundle. Do not run it before training. NN-37 accepts `FACE_IMAGE_PATH` for a single face prediction without logging attendance.
8. NN-39 exports real CSV/JSON results. NN-40–42 provide academic discussion, computed findings and report guidance.

The default path processes supplied face images without adding face detection to training. The webcam's Haar crop can therefore cause a distribution shift. Compare capture framing with independent consented validation images; do not silently change preprocessing or tune on the held-out test set.

## Save and load the model

Execute **NN-36** after the preceding cells. It writes a CPU-portable `.pt` tensor state dictionary, the exact CNN constructor settings, ordered labels, exact image dimensions/grayscale/resize/normalization, score policy, seed/split indices, versions and model metadata. Re-executing replaces the generated bundle. Keep all files together when moving from Colab to your local machine.

```python
from attendance_utils import load_bundle, predict_face
model, names, config = load_bundle()  # defaults relative to attendance_utils.py
result = predict_face(model, names, config, "path/to/consented_face_crop.jpg")
print(result)
```

Only load a trusted bundle trained by you. The loader reconstructs `AttendanceCNN` from `model_metadata.json`, uses `torch.load(..., weights_only=True)` and validates class count, state-dictionary shapes and the preprocessing contract. Existing `.keras` files are not converted or loaded; retrain the three architectures and regenerate the full PyTorch bundle. Existing model files are not deleted during migration.

## Local webcam demo

After NN-36 saves the bundle, run locally from Anaconda Prompt with base active. CUDA is mandatory, including when the optional `--device cuda` flag is omitted:

```powershell
python live_attendance.py --device cuda
```

Optional flags:

```text
python live_attendance.py --device cuda --camera 0 --threshold 0.90 --consistent-frames 8 --timezone Asia/Katmandu
python live_attendance.py --model-dir "C:\path\to\models" --attendance-dir "C:\private\attendance"
```

The enrolled student enters their own ID and explicitly opts in before camera processing. One face is required. Haar detection locates the face; the CNN recognizes the identity. Eight consecutive predictions above the heuristic threshold, matching the opted-in ID and spatially continuous detection, enable **C**. Scores below threshold show **Unknown**; multiple faces/no face/reset conditions disable recording. Softmax confidence is not calibrated probability and high-scoring unknown people remain possible.

Press **C** when the identity is stable, inspect the frozen preview, and type `CONFIRM <student_id>` in the terminal within 30 seconds. A changed local date invalidates confirmation. Only then is the record written. **N** switches to another consenting student. **Q** or closing the preview releases the camera and exits. Camera errors also release resources. A lock file prevents simultaneous writers in the same attendance folder; after a crash, verify no demo is running before manually removing a stale `.attendance.lock`.

Attendance CSV columns:

```text
Student ID,Date,Time,Confidence,Status
```

Files are named `attendance/attendance_YYYY-MM-DD.csv`. Dates use the selected timezone, default Asia/Katmandu. Time includes its UTC offset. Confidence is the final accepted frame's softmax score. Status is `Present (confirmed)`. Existing student/date entries are checked before appending, including across restarts. Preserve IDs as text when opening CSVs in a spreadsheet. Run with ordinary student IDs, and do not treat imported CSV values as spreadsheet formulas.

Google Colab's `VideoCapture(0)` generally cannot access your local webcam directly. Download all four model files, install dependencies locally and run the separate script. Nothing launches the webcam from the notebook.

## Smart Attendance web app (Streamlit + SQLite)

```powershell
python run_app.py
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for installation, sign-in and the classroom flow. Design notes:

* **Recognition flow:** browser webcam (streamlit-webrtc, video stays on this computer) → Haar face detection
  (exactly one face) → crop → `attendance_utils.preprocess_image` → `AttendanceCNN` → softmax ≥ threshold
  (default 0.90 from `preprocessing_config.json`) → same student in N consecutive frames (default 8) with a
  continuous face box (IoU ≥ 0.25, as in `live_attendance.py`) → **auto-mark Present** (or teacher Confirm when
  auto-mark is off) → Undo available while the session runs.
* **Threading:** the video callback runs in a worker thread and only updates a lock-protected recognition state;
  all SQLite writes happen in the Streamlit thread.
* **Database (`database_storage/attendance.db`):** `students` (unique ID, model class, photo, consent),
  `users` (teacher/student logins, PBKDF2 hashes), `attendance_sessions`, `attendance`
  (**UNIQUE(student_id, session_id)** — no duplicates), `attendance_corrections` (audit), `recognition_logs`
  (events only, never images), `app_settings`, `schema_migrations`. Foreign keys on, parameterised SQL,
  transactions, schema versioning with automatic backup before upgrades; never deleted on startup.
* **Absences** are written only when the teacher ends a session; until then students are "not yet marked".
  Attendance rate = (present + late) / (present + late + absent) over ended sessions.
* **First start:** a teacher account and one student per class in `class_names.json` (linked to that class,
  first dataset photo as profile picture, consent recorded, login `<ID>` / `student123`).
* **Demo logins are dummy accounts** for a local classroom demo — not production security.

The earlier viewer is kept as `app_legacy_viewer.py`, and the OpenCV window script `live_attendance.py` (CSV
records in `attendance/`) still works as before.

## Results, screenshots and presentation

No numerical findings are provided until you execute the notebook. Capture the EDA, shared split, three model summaries, CNN experiment table, four learning curves, metric tables, confusion matrices, per-class report, efficiency charts, saved-model reload, single-image result, webcam confirmation and a redacted attendance CSV. See [DEMO_GUIDE.md](DEMO_GUIDE.md) for the recommended sequence and narration. Use actual outputs; do not stage invented accuracy or false attendance. Keep face/identity screenshots private unless sharing is explicitly authorized.

## Limitations and academic use

This is an academic prototype, not production-ready attendance software. Small datasets and webcam bursts can inflate image-level test results. Different days, rooms, lighting, cameras, backgrounds and detectors may change performance. Training can overfit background or clothing. The classifier is closed-set; heuristic threshold rejection is not validated open-set recognition. There is no anti-spoofing/liveness system and no independent field evaluation. A printout or replayed video may pass. Human confirmation reduces accidental logging but does not prove identity. Keep a manual attendance option and avoid consequential decisions based on these predictions.

Runtime issues still to check yourself: package resolution on your OS, notebook execution, accelerator availability, training convergence, saved-model reload, webcam permissions/detection/framing, actual inference speed, consent/confirmation cancellation, no/multiple-face handling and duplicate CSV behavior. Installation and implementation references: [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/), [saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
