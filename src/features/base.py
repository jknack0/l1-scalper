"""Abstract Feature base class for all L1 features."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schemas import L1Record


class Feature(ABC):
    """Base class for all L1 features.

    Each feature processes one L1Record at a time and returns a float
    (or None if the feature isn't ready yet, e.g. during warmup).
    """

    name: str

    @abstractmethod
    def update(self, record: L1Record) -> float | None:
        """Process one record, return feature value or None if not ready."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state for a new session."""
        ...
