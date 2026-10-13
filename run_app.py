"""Start the Smart Attendance web app.

    python run_app.py

Why a launcher? The notebook sets MKL_THREADING_LAYER=SEQUENTIAL *before* NumPy/PyTorch load,
to avoid Anaconda's duplicate-OpenMP crash (see attendance_runtime.py). Running Streamlit through
this file applies the same setting before Streamlit imports NumPy. It also makes the project
folder the working directory, so .streamlit/config.toml (localhost only) is used.
Extra arguments are passed to Streamlit, e.g.  python run_app.py --server.port 8502
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from attendance_runtime import configure_runtime

configure_runtime()

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    os.chdir(PROJECT_ROOT)
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(PROJECT_ROOT / "app.py"), "--server.address", "localhost", *sys.argv[1:]]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
