"""BAAF: Bootstrap Aggregation Anomaly Filtering."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baaf.determinism import apply_runtime_env, configure_determinism  # noqa: E402

apply_runtime_env()

from baaf.algorithm import run_baaf, super_seed  # noqa: E402

__all__ = ["run_baaf", "super_seed", "configure_determinism"]
