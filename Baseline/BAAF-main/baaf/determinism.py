"""Process-wide determinism for BAAF and the bundled detectors.

Call ``apply_runtime_env()`` before importing torch when possible, then
``configure_determinism(seed)`` before any RNG use.

These flags make a *single machine + pinned wheels* re-run match itself.
They do **not** make GPU kernels bit-identical across CUDA builds, GPU
architectures, or OS BLAS libraries.
"""
from __future__ import annotations

import os
import random
import sys

# Must be set before cuBLAS initializes (i.e. before the first CUDA op).
# PYTHONHASHSEED only takes effect if set before the interpreter starts.
RUNTIME_ENV = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def apply_runtime_env():
    """Fill in determinism-related environment variables if unset."""
    for key, value in RUNTIME_ENV.items():
        os.environ.setdefault(key, value)


def configure_determinism(seed, warn_only=True, reseeds=("python", "numpy", "torch")):
    """Seed RNGs we control and turn on PyTorch deterministic kernels.

    ``warn_only=True`` (default) keeps EfficientAD runnable: some CUDA ops it
    uses (bilinear upsample backward, ``torch.quantile``) have no deterministic
    implementation. Tests of CPU-only BAAF pieces can pass ``warn_only=False``.

    ``reseeds`` selects which generators to reset. Detector wrappers that already
    received a fold seed should typically reseed torch (and maybe numpy) without
    touching Python ``random``, so later bag splits stay on the BAAF stream.
    """
    apply_runtime_env()
    import numpy as np
    import torch

    seed = int(seed)
    reseeds = set(reseeds)

    if "python" in reseeds:
        random.seed(seed)
    if "numpy" in reseeds:
        np.random.seed(seed)
    if "torch" in reseeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        for name, enabled in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            fn = getattr(torch.backends.cuda, name, None)
            if callable(fn):
                fn(enabled)

    try:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
    except TypeError:
        torch.use_deterministic_algorithms(True)

    try:
        torch.set_float32_matmul_precision("highest")
    except (AttributeError, RuntimeError):
        pass

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Interop thread count can only be set once per process.
        pass

    return seed


def seed_worker(worker_id):
    """``DataLoader(worker_init_fn=...)`` so workers do not inherit a stale RNG."""
    import numpy as np
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed):
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def environment_report():
    """Versions and device identity, stored next to metrics so mismatches are obvious."""
    apply_runtime_env()
    report = {
        "python": sys.version.replace("\n", " "),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        import numpy as np

        report["numpy"] = np.__version__
    except ImportError:
        report["numpy"] = None
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_built"] = str(torch.version.cuda)
        report["cudnn"] = (
            str(torch.backends.cudnn.version())
            if torch.cuda.is_available() and torch.backends.cudnn.is_available()
            else None
        )
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0)
            )
        else:
            report["gpu_name"] = None
            report["gpu_capability"] = None
    except ImportError:
        report["torch"] = None
    try:
        import faiss

        report["faiss"] = getattr(faiss, "__version__", "unknown")
    except ImportError:
        report["faiss"] = None
    try:
        import torchvision

        report["torchvision"] = torchvision.__version__
    except ImportError:
        report["torchvision"] = None
    try:
        import PIL

        report["pillow"] = PIL.__version__
    except ImportError:
        report["pillow"] = None
    return report
