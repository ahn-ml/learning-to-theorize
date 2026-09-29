"""Small distributed-runtime primitives shared by released experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch
from torch import distributed as dist


WandbMode = Literal["online", "offline", "disabled"]


@dataclass(frozen=True, slots=True)
class GitSourceState:
    """Auditable Git state attached to every run directory."""

    repository_root: str | None
    commit: str | None
    branch: str | None
    remote_url: str | None
    clean: bool
    pushed: bool
    working_tree_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """One process's resolved torchrun topology."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    owns_process_group: bool = False

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_global_zero(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True, slots=True)
class ExperimentTrackingConfig:
    """Optional W&B destination for a run that always logs locally."""

    project: str
    name: str
    entity: str | None = None
    group: str | None = None
    mode: WandbMode = "online"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.project.strip() or not self.name.strip():
            raise ValueError("tracking project and run name must not be empty")
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError("tracking mode must be online, offline, or disabled")


class MetricLog:
    """Write JSONL locally and mirror records to W&B when configured."""

    def __init__(
        self,
        path: Path,
        tracking: ExperimentTrackingConfig,
        resolved_config: Mapping[str, Any],
    ) -> None:
        self._file = path.open("x", encoding="utf-8")
        self._wandb: Any | None = None
        self._wandb_run: Any | None = None
        try:
            if tracking.mode != "disabled":
                try:
                    import wandb
                except ImportError as error:
                    raise RuntimeError(
                        "W&B tracking requires `pip install -e '.[train]'`"
                    ) from error
                self._wandb = wandb
                self._wandb_run = wandb.init(
                    project=tracking.project,
                    entity=tracking.entity,
                    group=tracking.group,
                    name=tracking.name,
                    mode=tracking.mode,
                    dir=str(path.parent),
                    config=dict(resolved_config),
                    tags=list(tracking.tags),
                )
                if self._wandb_run is None:
                    raise RuntimeError("wandb.init did not return a run")
        except BaseException:
            if self._wandb is not None:
                try:
                    self._wandb.finish(exit_code=1)
                except Exception:
                    pass
            self._file.close()
            raise

    @property
    def tracker_url(self) -> str | None:
        return (
            getattr(self._wandb_run, "url", None)
            if self._wandb_run is not None
            else None
        )

    def log(self, metrics: Mapping[str, int | float]) -> None:
        record = dict(metrics)
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(record)

    def finish(self, *, exit_code: int) -> None:
        try:
            if self._wandb is not None:
                self._wandb.finish(exit_code=exit_code)
        finally:
            self._file.close()


def capture_git_source(path: str | Path) -> GitSourceState:
    """Capture commit, cleanliness, remote reachability, and a dirty fingerprint."""

    start = Path(path)
    root_result = _run_git(start, "rev-parse", "--show-toplevel")
    if root_result is None:
        return GitSourceState(None, None, None, None, False, False, None)
    root = Path(root_result)
    commit = _required_git(root, "rev-parse", "HEAD")
    branch_value = _required_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    branch = None if branch_value == "HEAD" else branch_value
    status = _required_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    clean = status == ""
    remote_url = _run_git(root, "remote", "get-url", "origin")
    remote_url = _sanitize_remote_url(remote_url) if remote_url else None
    containing = _run_git(
        root,
        "for-each-ref",
        "--contains",
        commit,
        "--format=%(refname:short)",
        "refs/remotes/origin",
    )
    pushed = bool(containing and containing.strip())
    fingerprint = None if clean else _working_tree_fingerprint(root)
    return GitSourceState(
        repository_root=str(root),
        commit=commit,
        branch=branch,
        remote_url=remote_url,
        clean=clean,
        pushed=pushed,
        working_tree_fingerprint=fingerprint,
    )


def initialize_distributed(
    *,
    backend: str = "nccl",
    timeout_minutes: int = 30,
    force_cpu: bool = False,
) -> DistributedContext:
    """Resolve torchrun environment variables and initialize a process group."""

    rank = environment_integer("RANK", 0)
    local_rank = environment_integer("LOCAL_RANK", 0)
    world_size = environment_integer("WORLD_SIZE", 1)
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid RANK, LOCAL_RANK, or WORLD_SIZE environment")

    if world_size > 1:
        if force_cpu and backend == "nccl":
            raise ValueError("NCCL cannot be combined with force_cpu")
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA")
        if torch.cuda.is_available() and not force_cpu:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} exceeds "
                    f"{torch.cuda.device_count()} visible GPUs"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        owns_process_group = False
        if not dist.is_initialized():
            device_argument = {"device_id": device} if device.type == "cuda" else {}
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=timeout_minutes),
                **device_argument,
            )
            owns_process_group = True
        elif dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError("existing process group disagrees with torchrun environment")
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            owns_process_group=owns_process_group,
        )

    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available() and not force_cpu
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistributedContext(rank, local_rank, world_size, device, False)


def create_run_directory(
    output_root: str | Path,
    run_name: str,
    source: GitSourceState,
    context: DistributedContext,
) -> Path:
    """Create one unique rank-zero directory and share its path with all ranks."""

    payload: list[Any] = [None]
    if context.is_global_zero:
        try:
            parent = Path(output_root).expanduser().resolve() / run_name
            parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            revision = source.commit[:8] if source.commit else "unknown"
            output = parent / f"{timestamp}-{revision}"
            output.mkdir(exist_ok=False)
            payload[0] = {"value": str(output)}
        except BaseException as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    broadcast_object(payload, context)
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("rank zero did not provide an output directory")
    if "error" in result:
        raise RuntimeError(f"output setup failed on rank zero: {result['error']}")
    return Path(result["value"])


def seed_everything(seed: int) -> None:
    """Set the deterministic Python, NumPy, CPU, CUDA, cuDNN, and cuBLAS state."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def bfloat16_autocast(device: torch.device) -> Any:
    """Use CUDA bf16 mixed precision and a no-op context on CPU."""

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def barrier(context: DistributedContext) -> None:
    if context.is_distributed:
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        dist.barrier(device_ids=device_ids)


def broadcast_object(payload: list[Any], context: DistributedContext) -> None:
    if context.is_distributed:
        dist.broadcast_object_list(payload, src=0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_integer(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return default if value is None else int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}-{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _working_tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest.update(b"tracked\0")
    digest.update(diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        if candidate.is_symlink():
            digest.update(os.readlink(candidate).encode("utf-8"))
        elif candidate.is_file():
            with candidate.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"non-regular")
    return digest.hexdigest()


def _run_git(path: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _required_git(path: Path, *arguments: str) -> str:
    result = _run_git(path, *arguments)
    if result is None:
        raise RuntimeError(f"git {' '.join(arguments)} failed")
    return result


def _sanitize_remote_url(value: str) -> str:
    if "://" not in value:
        return value
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
