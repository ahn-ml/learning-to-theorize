"""Small domain-independent contracts for support/query tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


ObservationT = TypeVar("ObservationT")
MetadataT = TypeVar("MetadataT")


@dataclass(frozen=True, slots=True)
class Example(Generic[ObservationT]):
    """One before/after observation pair."""

    input: ObservationT
    output: ObservationT


@dataclass(frozen=True, slots=True)
class Episode(Generic[ObservationT, MetadataT]):
    """A task containing demonstrations and held-out queries.

    The contract deliberately makes no assumptions about observation shape,
    tensor library, or domain metadata. Domain adapters own those choices.
    """

    support: tuple[Example[ObservationT], ...]
    query: tuple[Example[ObservationT], ...]
    metadata: MetadataT | None = None

    def __post_init__(self) -> None:
        if not self.support:
            raise ValueError("an episode must contain at least one support example")
        if not self.query:
            raise ValueError("an episode must contain at least one query example")
