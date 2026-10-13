"""Install missing project dependencies in Anaconda base; never train on import."""
import importlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

from attendance_runtime import configure_runtime


def _pip(*arguments):
    subprocess.run([sys.executable, "-s", "-m", "pip", "install", "--disable-pip-version-check",
                    *arguments], check=True)
    importlib.invalidate_caches()


def _cuda_probe():
    """Probe native imports outside the kernel so a broken DLL cannot kill Run All."""
    code = (
        "import json, torch; "
        "x = torch.ones(1, device='cuda') + 1 if torch.cuda.is_available() else None; "
        "torch.cuda.synchronize() if torch.cuda.is_available() else None; "
        "print(json.dumps({'torch':str(torch.__version__),'runtime':torch.version.cuda,"
        "'available':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    result = subprocess.run([sys.executable, "-s", "-c", code], text=True,
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError("PyTorch could not initialize in Anaconda base:\n" + result.stderr[-5000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def ensure_environment(project_root):
    """Called by the first notebook cell before numerical-library imports.

    Download only missing/incompatible packages. Preserve an existing working CUDA
    torch build. A clean kernel is needed if an already-imported package is replaced.
    """
    configure_runtime()
    project_root = Path(project_root).resolve()
    prefix = Path(sys.prefix).resolve()
    if not ((prefix / "conda-meta").is_dir() and (prefix / "condabin" / "conda.bat").is_file()):
        raise RuntimeError(
            f"Select your Anaconda base Python kernel and restart, then Run All. "
            f"Current interpreter is {sys.executable}; no packages were installed here.")
    print("Anaconda base:", sys.executable, flush=True)
    if not (3, 10) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("This setup needs Python 3.10–3.14 in Anaconda base. "
                           "Update Anaconda/Python first; the notebook will not replace your Python installation.")
    try:
        from packaging.requirements import Requirement
    except ImportError:
        _pip("packaging>=24")
        from packaging.requirements import Requirement

    requirements = []
    for line in (project_root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(Requirement(line))
    torch_requirement = next(req for req in requirements if req.name == "torch")
    def installed_version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    before_versions = {req.name: installed_version(req.name) for req in requirements}
    try:
        installed_torch = metadata.version("torch")
    except metadata.PackageNotFoundError:
        installed_torch = None
    replaced = []
    if installed_torch is None or installed_torch not in torch_requirement.specifier:
        _pip("-r", str(project_root / "requirements-cuda.txt"))
        replaced.append("torch")
    probe = _cuda_probe()
    if probe["runtime"] is None:
        # Replace a CPU-only wheel, preserving the public version and companion-package compatibility.
        public_version = probe["torch"].split("+")[0]
        _pip("--force-reinstall", f"torch=={public_version}",
             "--index-url", "https://download.pytorch.org/whl/cu126")
        replaced.append("torch")
        probe = _cuda_probe()
    if not probe["available"]:
        raise RuntimeError("CUDA-enabled torch is installed, but the NVIDIA GPU is unavailable. "
                           "Check the NVIDIA driver and restart the kernel. CPU fallback is disabled.")

    missing = []
    for requirement in requirements:
        if requirement.name == "torch":
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed is None or installed not in requirement.specifier:
            missing.append(str(requirement))
            replaced.append(requirement.name)
    if missing:
        print("Installing missing/incompatible dependencies:", ", ".join(missing), flush=True)
        _pip(*missing)
    replaced.extend(name for name, old_version in before_versions.items()
                    if old_version != installed_version(name) and name not in replaced)
    # Never continue with a stale extension module after pip changed it on disk.
    import_names = {"scikit-learn": "sklearn", "Pillow": "PIL", "opencv-python": "cv2"}
    already_loaded = [name for name in replaced if import_names.get(name, name) in sys.modules]
    if already_loaded:
        raise RuntimeError("Updated already-loaded packages: " + ", ".join(already_loaded)
                           + ". Restart the kernel once and Run All again; installation is complete.")
    print(f"Ready: PyTorch {probe['torch']} | CUDA {probe['runtime']} | {probe['gpu']}", flush=True)
    print("Dependencies ready. Continuing with EDA and CUDA training; no CPU fallback.", flush=True)
    return probe


if __name__ == "__main__":
    ensure_environment(Path(__file__).resolve().parent)
