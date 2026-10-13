# TECH 405 presentation guide

## Before presenting

On your friend's Windows/NVIDIA computer, select **Anaconda base**, restart the kernel and press **Run All**. The first Setup cell installs missing dependencies in base automatically. Show NN-0 reporting `cuda:0`, the actual NVIDIA GPU and the Anaconda interpreter path. Wait for all five CNN experiments and the RNN/LSTM baselines. Run NN-36 to save/reload `attendance_cnn.pt` and its three JSON files and NN-39 to export real results. Check the local webcam on your own equipment with a consenting enrolled participant. Dependency installation and a CUDA tensor operation were verified on the development computer only. Your friend's notebook, model training and demonstration still need to run on their machine.

Keep the terminal beside the webcam preview: typed opt-in and record confirmation happen there. Know the enrolled student's ID. Use a local Python environment; a remote Colab runtime will not normally see the local camera. Prepare a manual attendance alternative. Keep sensitive screenshots and records out of the public repository.

## Web app demonstration (recommended)

Start it before the presentation with `python run_app.py` (see HOW_TO_RUN.md) and open Live Attendance once so
the model is already loaded. Checklist: camera allowed in the browser for `localhost`, Zoom/Teams closed, front
lighting, the five enrolled students present (or a subset), demo passwords at hand.

| Step | Where | What to show and say |
|---|---|---|
| 1 | Sign-in page | “Two roles with dummy demo accounts: teacher and student. Passwords are hashed in SQLite.” Sign in as `teacher` / `teacher123`. |
| 2 | Dashboard | KPIs, photo cards, trend, rate per student. “Absences only count after a session ends — never for days without a class.” |
| 3 | CNN vs RNN vs LSTM | Real exported metrics: CNN and RNN tied at 100 % test accuracy, LSTM 97.4 % (one of 38 test images). “With 38 test images one image ≈ 2.6 points, so the tie is not conclusive. The CNN was chosen in advance and its configuration on validation data only.” Show training curves and confusion matrices. |
| 4 | Students | Photo directory, model links (each trained class ↔ one student), consent, login accounts. “Registering a student does not retrain the CNN.” |
| 5 | Live Attendance | **Start today's session → START.** One student at a time: blue box “verifying k/8”, then green and the toast “Amit marked Present”. Explain detection (Haar) vs. identification (CNN) vs. stability (8 frames ≥ 90 %). |
| 6 | Duplicate | The same student again → “already marked present at hh:mm” (UNIQUE constraint in SQLite, not just a Python check). |
| 7 | Edge cases | Two faces → “one student at a time”; covered face / poor light → “Unknown”. Use **Undo** to show how a wrong match is corrected. |
| 8 | End session | **End session** → remaining students recorded Absent. Dashboard cards update. |
| 9 | Records & Reports | Filters, CSV export, a correction with reason (audit trail), daily / per-student reports. |
| 10 | Student view | Sign out → sign in as `Amit` / `student123` → own rate, history and monthly chart only. |
| 11 | Limitations | Closed-set classifier (a stranger is matched to the most similar student), no liveness, small dataset, demo-only logins. |

Fallback if live video fails: Live Attendance → **Still photo** (one snapshot, crop-variant agreement) or
**Upload a photo**; or mark manually on Attendance Records.

## Suggested sequence and narration (notebook)

| Step | Show | What to explain |
|---|---|---|
| 1 | Dataset card and sample grid | “These are self-collected, consented face images. Counts are calculated from the current files.” |
| 2 | Quality audit, histograms and mean images | “Equal counts do not guarantee independent or diverse images. Similar burst frames can inflate evaluation.” |
| 3 | NN-1/2 shapes and shared indices | “All three models use exactly the same identities, normalization and train/validation/test samples.” |
| 4 | Augmentation examples | “Only training images are augmented; clean validation and test images remain held out.” |
| 5 | Three summaries at NN-10 | “CNN learns local spatial filters. SimpleRNN reads spatial image rows. LSTM adds gated memory; these are not video timesteps.” |
| 6 | CNN experiments and recurrent settings | “Five CNN configurations are compared using validation. Recurrent baselines share the selected CNN training controls. Search effort is unequal and disclosed.” |
| 7 | Four curves | “Each curve ends at its actual stopping epoch. The minimum-validation-loss weights are restored.” |
| 8 | NN-29/30 tables and majority baseline | Read your computed accuracy and macro precision/recall/F1. Explain weighted metrics and support; never claim a score before running. |
| 9 | Confusion matrices and reports | Explain a true/predicted identity pair if an error exists. If none exists, say so without inventing errors. |
| 10 | Efficiency charts | “This is selected-run training time, parameter count and synchronized batch-one model latency. It excludes detection/UI time.” |
| 11 | NN-34/35 | Name actual test leader(s), then the independently predefined CNN deployment candidate. Do not claim CNN won if it did not. |
| 12 | NN-36/37 | Show saving/reloading and a consented cropped-image prediction. Explain that softmax scores are not calibrated identity probabilities. |
| 13 | Webcam flow below | Explain separate detection, classification, stable-ID matching and explicit attendance confirmation. |
| 14 | Dated CSV | Show one confirmed record, then repeat confirmation to demonstrate duplicate prevention. |
| 15 | Limitations/conclusion | Discuss correlated capture frames, unseen students, cropping shifts, privacy, spoofing and independent evaluation still needed. |

## Webcam flow

```text
python live_attendance.py --device cuda
```

1. The student types their enrolled ID and answers `yes` to opt in. This covers the demonstration; dataset collection/training consent must already exist.
2. Show one face. Explain that the rectangle comes from a Haar detector, not the identity classifier.
3. Wait for eight stable predictions above the default 0.90 heuristic threshold that match the opted-in ID. The threshold is not a guarantee of correctness.
4. If the displayed identity is correct, the student presses **C**, reviews the frozen preview, and types `CONFIRM <student_id>` within 30 seconds. Anything else cancels. A false or uncertain identity must never be confirmed.
5. Show the green feedback and open `attendance/attendance_YYYY-MM-DD.csv`. Read the student/date/time/score/status columns. Keep the ID column as text.
6. Repeat stable confirmation to show “Already marked today.” No duplicate row should appear.
7. Demonstrate no-face and multiple-face states without recording. If a low-confidence case occurs, show Unknown; do not promise every stranger will be rejected or manufacture a score.
8. Press **N** for another student's opt-in or **Q** to quit and release the camera.

Do not present a photo/replay as a successful liveness check. The demo cannot reliably detect such attacks. If the camera fails, show the saved-image prediction and explain the local hardware issue honestly.

The earlier simple viewer is kept as `app_legacy_viewer.py` (`python -m streamlit run app_legacy_viewer.py`); the
full web app described at the top of this guide replaces it for the demonstration.

## Screenshot checklist

- EDA dataset card, consented sample grid, class balance, intensity histogram and mean faces.
- Preprocessing shapes, split distribution and train-only augmentation.
- NN-0 CUDA/GPU output, then CNN (`nn.Conv2d`), RNN (`nn.RNN`) and LSTM (`nn.LSTM`) summaries with parameter counts.
- Five-CNN configuration table and recurrent training settings.
- Training accuracy, validation accuracy, training loss and validation loss curves.
- Headline macro table, weighted metrics and majority baseline.
- Three count/normalized confusion matrices and per-class report.
- Parameter, training-time and inference-time charts with timing/hardware notes.
- Validation-based deployment decision and separate observed test leader(s).
- Saved-model reload and single-image prediction with top scores.
- Webcam stable confirmation, no-face/multiple-face/Unknown behavior when observed.
- Confirmed attendance feedback and redacted CSV with duplicate prevention.
- Academic limitations and conclusion generated from your real run.
- Web app: sign-in page, teacher dashboard with photo cards, Live Attendance with a green recognised box and the
  “marked Present” message, the “already marked” duplicate message, Unknown / two-faces states, Undo, ended
  session with absences, Attendance Records with a correction, a report chart + exported CSV, the student view,
  the CNN vs RNN vs LSTM page and Settings.

Save screenshots outside the public repository or under ignored `screenshots/`. Obtain explicit permission for any identifiable face screenshot. Redact IDs and other personal information. The original notebook already embeds face outputs: clear outputs in a separate public copy and inspect its source/metadata before sharing; preserve your private original.
