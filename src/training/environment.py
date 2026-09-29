"""Check the shared installation and report available CUDA hardware."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "environments"
    / "runtime.json"
)
RUNTIME_KEYS = (
    "python",
    "pip",
    "torch",
    "torch_cuda",
    "cudnn",
    "numpy",
    "numpy_blas",
    "h5py",
    "hdf5",
    "pyyaml",
    "lightning",
    "wandb",
)
PLATFORM_KEYS = ("os", "architecture")


def collect_runtime() -> dict[str, str | int | None]:
    """Collect the pinned runtime versions."""

    numpy_build = np.show_config(mode="dicts")
    numpy_blas = numpy_build["Build Dependencies"]["blas"]["name"]
    return {
        "python": platform.python_version(),
        "pip": version("pip"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "numpy_blas": numpy_blas,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "pyyaml": version("PyYAML"),
        "lightning": version("lightning"),
        "wandb": version("wandb"),
    }


def collect_platform() -> dict[str, str]:
    """Collect stable host fields required by the release environment."""

    return {
        "os": platform.system(),
        "architecture": platform.machine(),
    }


def collect_nvidia_driver() -> str | None:
    """Return the visible NVIDIA driver version, when ``nvidia-smi`` works."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        return None
    return versions.pop()


def compare_runtime(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return every missing or unequal runtime field."""

    mismatches: dict[str, dict[str, Any]] = {}
    for key in RUNTIME_KEYS:
        if expected.get(key) != actual.get(key):
            mismatches[key] = {
                "expected": expected.get(key),
                "actual": actual.get(key),
            }
    return mismatches


def validate_environment(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Compare pinned software versions and optionally require CUDA."""

    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["runtime"]
    actual = collect_runtime()
    mismatches = compare_runtime(expected, actual)

    expected_platform = manifest["platform"]
    actual_platform = collect_platform()
    for key in PLATFORM_KEYS:
        if expected_platform.get(key) != actual_platform.get(key):
            mismatches[f"platform.{key}"] = {
                "expected": expected_platform.get(key),
                "actual": actual_platform.get(key),
            }

    hardware: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if hardware["cuda_available"]:
        hardware["gpu"] = torch.cuda.get_device_name(0)
        hardware["driver"] = collect_nvidia_driver()
    if require_cuda and not hardware["cuda_available"]:
        mismatches["cuda_available"] = {"expected": True, "actual": False}

    return {
        "manifest": str(manifest_path),
        "environment_file": manifest["release_environment"]["file"],
        "actual": actual,
        "platform": actual_platform,
        "hardware": hardware,
        "mismatches": mismatches,
        "compatible": not mismatches,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="also require an available CUDA device",
    )
    arguments = parser.parse_args(argv)
    result = validate_environment(
        arguments.manifest,
        require_cuda=arguments.require_cuda,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
