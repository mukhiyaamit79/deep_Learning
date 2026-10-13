"""Configure Anaconda's native libraries before importing NumPy or PyTorch."""
import os


def configure_runtime():
    # Anaconda MKL and torch ship competing OpenMP runtimes on this machine.
    # Use MKL's supported sequential layer; never suppress duplicate-runtime errors.
    # This affects CPU BLAS only. Models and augmentation still run on CUDA.
    os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


configure_runtime()
