"""Exact identities of the HDF5 artifacts consumed by the GridWorld paper."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tasks.gridworld.data.profiles import (
    GridWorldProfile,
    SplitName,
    available_profiles,
    get_profile,
)


@dataclass(frozen=True, slots=True)
class GridWorldArtifactSpec:
    """Expected filename, content identity, and paper use of one data split."""

    profile: str
    split: SplitName
    experiment_usage: str
    consumed_by_model: bool
    filename: str
    episodes: int
    num_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.num_bytes < 1:
            raise ValueError("artifact episodes and num_bytes must be positive")
        if len(self.sha256) != 64:
            raise ValueError("artifact SHA-256 must contain 64 hexadecimal characters")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("artifact SHA-256 must be hexadecimal") from error


@dataclass(frozen=True, slots=True)
class GridWorldArtifactEvidence:
    """Observed identity after an artifact matches its exact paper contract."""

    profile: str
    split: SplitName
    path: str
    filename: str
    episodes: int
    bytes: int
    sha256: str
    exact_paper_artifact: bool = True

    def as_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


def _artifact(
    profile_name: str,
    split_name: SplitName,
    experiment_usage: str,
    consumed_by_model: bool,
    num_bytes: int,
    sha256: str,
) -> GridWorldArtifactSpec:
    profile = get_profile(profile_name)
    split = next(
        (candidate for candidate in profile.splits if candidate.name == split_name),
        None,
    )
    if split is None or split.num_episodes < 1:
        raise ValueError(f"profile {profile_name!r} has no generated split {split_name!r}")
    return GridWorldArtifactSpec(
        profile=profile_name,
        split=split_name,
        experiment_usage=experiment_usage,
        consumed_by_model=consumed_by_model,
        filename=profile.artifact_filename(split),
        episodes=split.num_episodes,
        num_bytes=num_bytes,
        sha256=sha256,
    )


_ARTIFACTS = (
    _artifact(
        "paper-alpha-0.33", "practice", "theorizer_training", True,
        194_313_464,
        "fc9dd6fc69473fc3a43df62aeca25abbe80608fcab03eb2d651d1f49338cb060",
    ),
    _artifact(
        "paper-alpha-0.33", "exam", "in_distribution_evaluation", True,
        19_378_616,
        "c0492bbb49c44cb5d6c948d15b8213c23262ba868d0e9aedce3dd22999888273",
    ),
    _artifact(
        "paper-alpha-0.33", "ood_test", "compositional_ood_evaluation", True,
        19_370_568,
        "c2bae4494e3b20d3a52ac277124b99c51d6865c5925ccf33b11aeb524e8be30a",
    ),
    _artifact(
        "paper-alpha-0.66", "practice", "theorizer_training", True,
        194_264_936,
        "bd2edb4485e5e7884163d37468b393b96b41009a2324e1544dc78a3942280022",
    ),
    _artifact(
        "paper-alpha-0.66", "exam", "in_distribution_evaluation", True,
        19_374_136,
        "a29a603a7319bb17d3a6334b8069af619a6d710ea914e60ecf6512b91db28f87",
    ),
    _artifact(
        "paper-alpha-0.66", "ood_test", "compositional_ood_evaluation", True,
        19_374_472,
        "4903c117831670588af5b3c3eaecf035fe4d01b87c2a9699d37573655ff94291",
    ),
    _artifact(
        "paper-alpha-1.00", "practice", "theorizer_training", True,
        194_256_456,
        "5415c8ebd67ca16e44fefa4a419894d7fc978c39579087c62c596ee7e6318b82",
    ),
    _artifact(
        "paper-alpha-1.00", "exam", "in_distribution_evaluation", True,
        19_374_904,
        "e8295a5d4dd12ab268e8b71806f279561cc4c0972a06343f6cf0214d535e3414",
    ),
    _artifact(
        "paper-length-4-8", "practice", "generation_only_rng_prefix", False,
        395_632,
        "bd7400ce151044c00d090cd4346b41627879b156a22ad7535023a94a1f8cc497",
    ),
    _artifact(
        "paper-length-4-8", "exam", "length_ood_evaluation", True,
        39_257_256,
        "7b949d0d23be138e08f16a0b26f6f6ef4e42baf3c315ad22d4721f31bd57e2e4",
    ),
    _artifact(
        "vae-pretraining", "practice", "observation_model_pretraining", True,
        1_939_749_952,
        "c09869f635bb2c482e83cca4caf8cf2afa056f17b3607606a59d97ddf7d5e3dd",
    ),
    _artifact(
        "vae-pretraining", "exam", "observation_model_evaluation", True,
        19_374_232,
        "0e82ea78bc7f988869f0e112bdcabc6c013d01fb5331198dadcfbe3db4f6c70b",
    ),
)
_ARTIFACT_BY_SPLIT = {
    (artifact.profile, artifact.split): artifact for artifact in _ARTIFACTS
}
if len(_ARTIFACT_BY_SPLIT) != len(_ARTIFACTS):  # pragma: no cover
    raise RuntimeError("duplicate GridWorld artifact specification")


def paper_artifacts(
    profiles: Iterable[str] | None = None,
) -> tuple[GridWorldArtifactSpec, ...]:
    """Return paper artifacts in deterministic profile and split order."""

    selected = available_profiles() if profiles is None else tuple(profiles)
    unknown = sorted(set(selected) - set(available_profiles()))
    if unknown:
        raise ValueError(f"unknown GridWorld profiles: {unknown}")
    selected_set = set(selected)
    return tuple(artifact for artifact in _ARTIFACTS if artifact.profile in selected_set)


def get_artifact_spec(profile: str, split: SplitName) -> GridWorldArtifactSpec:
    """Resolve the exact paper artifact for one profile split."""

    try:
        return _ARTIFACT_BY_SPLIT[(profile, split)]
    except KeyError as error:
        raise ValueError(f"no paper artifact for profile={profile!r}, split={split!r}") from error


def verify_artifact(
    path: str | Path,
    expected: GridWorldArtifactSpec,
) -> GridWorldArtifactEvidence:
    """Require exact filename, byte count, and SHA-256 identity."""

    artifact_path = Path(path).expanduser().resolve(strict=True)
    if artifact_path.name != expected.filename:
        raise ValueError(
            f"artifact filename mismatch: expected {expected.filename}, "
            f"got {artifact_path.name}"
        )
    num_bytes = artifact_path.stat().st_size
    if num_bytes != expected.num_bytes:
        raise ValueError(
            f"{expected.filename} byte-size mismatch: "
            f"expected {expected.num_bytes}, got {num_bytes}"
        )
    digest = _sha256(artifact_path)
    if digest != expected.sha256:
        raise ValueError(
            f"{expected.filename} SHA-256 mismatch: "
            f"expected {expected.sha256}, got {digest}"
        )
    return GridWorldArtifactEvidence(
        profile=expected.profile,
        split=expected.split,
        path=str(artifact_path),
        filename=expected.filename,
        episodes=expected.episodes,
        bytes=num_bytes,
        sha256=digest,
    )


def verify_profile_artifacts(
    profile: str | GridWorldProfile,
    data_directory: str | Path,
) -> tuple[GridWorldArtifactEvidence, ...]:
    """Verify every sequential artifact belonging to one paper profile."""

    profile_name = profile.name if isinstance(profile, GridWorldProfile) else profile
    root = Path(data_directory).expanduser().resolve()
    return tuple(
        verify_artifact(root / expected.filename, expected)
        for expected in paper_artifacts((profile_name,))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "GridWorldArtifactEvidence",
    "GridWorldArtifactSpec",
    "get_artifact_spec",
    "paper_artifacts",
    "verify_artifact",
    "verify_profile_artifacts",
]
