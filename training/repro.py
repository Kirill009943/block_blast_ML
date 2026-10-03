"""Reproducibility metadata for experiments and training runs.

``collect_run_metadata`` captures the full software/hardware environment
plus the training configuration; ``write_config_json`` stores it as
``config.json`` next to the run's artifacts.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_run_metadata(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Software/hardware/config snapshot for one run."""
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch

    cuda_available = torch.cuda.is_available()
    metadata: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "stable_baselines3_version": stable_baselines3.__version__,
        "sb3_contrib_version": sb3_contrib.__version__,
        "gymnasium_version": gymnasium.__version__,
        "git_commit": _git_commit(),
    }
    if config:
        metadata["config"] = config
    return metadata


def write_config_json(path: Path, config: Dict[str, Any]) -> Path:
    """Write metadata + config dict to ``path`` (pretty JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(collect_run_metadata(config), f, indent=2)
    return path
