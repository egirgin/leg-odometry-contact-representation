"""Dataset ABC: PyTorch-style indexing over whole trajectories."""

from __future__ import annotations

from abc import ABC, abstractmethod

from leg_odom.datasets.types import LegOdometrySequence


class BaseLegOdometryDataset(ABC):
    """One index = one prepared sequence."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> LegOdometrySequence: ...
