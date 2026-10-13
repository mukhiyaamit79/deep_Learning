# How to run the Smart Attendance System (Windows + VS Code)

Your notebook has **already trained** the models: `models/attendance_cnn.pt` and the `results/` files are in
this folder. To use the web app you only need to **install three extra packages and start it** — no retraining.

> Nothing in this folder was executed while the web app was written. If a step fails, the
> Troubleshooting table at the end covers the usual causes.

---

## 1. Put the folder somewhere simple

Unzip `Smart_Attendance_System.zip` to a short path **without special characters**, for example
`C:\Projects\Smart_Attendance_System`. (Very long paths or non-English characters can stop OpenCV from loading
its face detector.)

The folder already contains your `dataset/` (Amit, Shreejal, Susman, Udhav, Uttam — 50 photos each),
`amit_eda.ipynb`, `models/` and `results/`.

## 2. Open it in VS Code

1. **File → Open Folder…** → choose `Smart_Attendance_System` (the folder itself).
2. Accept the recommended extensions (**Python**, **Python Debugger**, **Jupyter**).
3. **Ctrl + Shift + P → “Python: Select Interpreter”** → choose the one that says **`('base')`** and points to
   `...\anaconda3\python.exe` (the same Anaconda base that trained the notebook).
4. **Terminal → New Terminal**. The prompt should start with `(base)`. Check:

   ```powershell
   python --version
   ```
   It should print Python 3.12.x (3.10–3.13 also work).

   If `python` is not recognised or `(base)` is missing, open **Anaconda Prompt** from the Start menu instead and
   go to the folder: `cd C:\Projects\Smart_Attendance_System`.

## 3. Install what the web app needs

### A) On the laptop with Anaconda + NVIDIA GPU (the one that trained the notebook) — recommended

```powershell
python project_setup.py
```

This is your project's own setup script. It reads `requirements.txt` and installs only what is missing —
here **streamlit-webrtc** (live camera), **plotly** (charts) and a newer **streamlit** — and keeps your CUDA
build of PyTorch.

### B) On a laptop without an NVIDIA GPU (only to run the app)

Install Anaconda (or 64-bit Python 3.12 from python.org), then in the project folder:

```powershell
python -m pip install -r requirements-app.txt
```

This installs the normal CPU build of PyTorch. The trained CNN is small, so the app runs fine on the CPU.
(Training in the notebook still requires CUDA.)

## 4. Start the web app

```powershell
python run_app.py
```

or in VS Code: **Run and Debug** (Ctrl + Shift + D) → **Smart Attendance: run web app** → ▶.

Your browser opens **http://localhost:8501**. The first start creates the database
`database_storage/attendance.db`, the demo accounts and the five students (with their first dataset photo as profile
picture). Stop the app with **Ctrl + C** in the terminal.

> Use `python run_app.py` rather than `streamlit run app.py`: it applies your project's
> `MKL_THREADING_LAYER` fix before NumPy/PyTorch load (avoids the Anaconda “OMP Error #15” crash).

## 5. Sign in (demo accounts)

| Role | Username | Password | Sees |
|---|---|---|---|
| Teacher | `teacher` | `teacher123` | everything: dashboard, live camera, students, records, reports, model comparison, settings |
| Student | `Amit`, `Shreejal`, `Susman`, `Udhav` or `Uttam` | `student123` | only their own attendance |

These are dummy accounts for the classroom demo. Passwords are stored hashed in SQLite; change them on
**My account**.

## 6. Take attendance with the live camera

1. Sign in as **teacher** → **Live Attendance**.
2. Click **Start today's session** (class *TECH 405*).
3. Under the video click **START** and **allow the camera** in the browser.
4. Let each student look at the camera **one at a time**, about 50–80 cm away, face well lit.
5. The box turns blue (“verifying 3/8”), then green. After **8 consecutive matching frames with ≥ 90 % score**
   the student is **marked Present automatically** — a green message appears and the name moves to
   “This session … present”.
6. A wrong match? Click **Undo** next to the name.
7. When everyone has passed, click **End session** → students without a mark are recorded **Absent**.
8. See it on the **Dashboard** (photo cards with Present / Absent), **Attendance Records** (CSV export,
   corrections) and **Reports** (daily / weekly / monthly / per student).
9. Sign out and sign in as a student (e.g. `Amit` / `student123`) to show the student view.

If the live video does not work, switch the camera mode to **Still photo** (takes one picture) or
**Upload a photo**.

Want the teacher to confirm each student instead of automatic marking? **Settings → Recognition →
Auto-mark** OFF.

## 7. (Optional) Re-run the notebook

Not needed for the app — results already exist. If you retrain: open `amit_eda.ipynb`, select the
**Anaconda base** kernel, **Restart → Run All**. NN-36 rewrites `models/`, NN-39 rewrites `results/`. The app
notices the new files and reloads the model by itself.

## 8. Adding a new student (needs retraining)

The CNN only knows the five trained students. For a sixth student:

1. Create `dataset/<StudentID>/` with about 50 consented, tightly cropped face photos (like the others).
2. Re-run the notebook (step 7).
3. In the app: **Students → Register** with that Student ID, then **Model links & logins → Link classes to
   students with the same ID**. A login (`<StudentID>` / `student123`) is created automatically.

Students can be registered without photos too — they are then marked manually
(**Attendance Records → Manual entry**).

## 9. Optional: unit tests

```powershell
python -m pip install pytest
python -m pytest -q
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Select the Anaconda base interpreter (step 2.3) or use **Anaconda Prompt**. |
| `No module named 'streamlit_webrtc'` / `plotly` | Run step 3 again in the **same** Python you start the app with. |
| `OMP: Error #15 … libiomp5md.dll already initialized` | Start with `python run_app.py`, not `streamlit run app.py`. |
| Login page says the database could not be prepared | Make sure the folder is writable (not inside a zip, not read-only OneDrive). |
| Browser blocks the camera | Click the camera icon in the address bar → *Allow* for `localhost` → press START again. |
| “No video frames received” | Close Zoom / Teams / the Windows Camera app (only one app can use the camera). Use *SELECT DEVICE* for another camera. |
| Video stays on “Connecting…” | **Settings → Camera → Use a public STUN server** ON, save, reload the page. Or use *Still photo*. |
| Always “Unknown” | Light from the front, face straight at the camera, closer. If needed lower **Confidence threshold** to 0.80 in Settings (more risk of wrong matches). |
| Wrong student marked | Click **Undo**. Raise **Consecutive frames required** (e.g. 12) in Settings. |
| “2 faces detected” | Only one person in view at a time. |
| Port 8501 already in use | `python run_app.py --server.port 8502` |
| Start again with a fresh demo database | Stop the app, delete `database_storage\attendance.db` (and `.db-wal`, `.db-shm`) yourself, start again. The app never deletes it on its own. |
| Forgot the teacher password | Another teacher account can't reset it; use the fresh-database step above (this also deletes attendance records). |
