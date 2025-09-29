"""Generator abstractions for candidate design proposals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Callable, Iterable
from copy import deepcopy
import numpy as np

from ..mutation import MutationProtocol
from ..system import System


@dataclass
class BatchSpec:
    base_system: System
    batch_size: int = 1
    metadata: dict[str, object] | None = None

    def with_batch_size(self, batch_size: int) -> 'BatchSpec':
        return replace(self, batch_size=batch_size)


class Generator(ABC):
    """Abstract generator interface returning batches of :class:`System` instances."""

    @abstractmethod
    def generate(self, batch_spec: BatchSpec) -> list[System]:
        raise NotImplementedError


class MonteCarloGenerator(Generator):
    """Generator that reuses mutation protocols without performing acceptance tests."""

    def __init__(self, mutation_protocol: MutationProtocol, random_seed: int | None = None) -> None:
        self.mutation_protocol = mutation_protocol
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def _mutate_once(self, system: System) -> System:
        working_system = deepcopy(system)
        reference_system = deepcopy(system)
        mutated, _ = self.mutation_protocol.one_step(working_system, reference_system)
        return mutated

    def generate(self, batch_spec: BatchSpec) -> list[System]:
        base_system = batch_spec.base_system
        systems = [self._mutate_once(base_system) for _ in range(batch_spec.batch_size)]
        for index, candidate in enumerate(systems):
            if candidate.name is None:
                candidate.name = f'{base_system.name or "design"}_mc_{index}'
        return systems


class CallableGenerator(Generator):
    """Wrap an arbitrary callable into the :class:`Generator` interface."""

    def __init__(self, func: Callable[[BatchSpec], Iterable[System]]) -> None:
        self.func = func

    def generate(self, batch_spec: BatchSpec) -> list[System]:
        return list(self.func(batch_spec))


class AF2MPNNGenerator(CallableGenerator):
    """Generator that can coordinate AF2 ↔ MPNN round-trips via a user callback."""

    def __init__(self, func: Callable[[BatchSpec], Iterable[System]] | None = None) -> None:
        if func is None:
            func = lambda batch_spec: [deepcopy(batch_spec.base_system)]
        super().__init__(func)


class RFDiffusionGenerator(CallableGenerator):
    """Generator that delegates to an RFdiffusion sampling callback."""

    def __init__(self, func: Callable[[BatchSpec], Iterable[System]] | None = None) -> None:
        if func is None:
            func = lambda batch_spec: [deepcopy(batch_spec.base_system)]
        super().__init__(func)


__all__ = [
    'BatchSpec',
    'Generator',
    'MonteCarloGenerator',
    'CallableGenerator',
    'AF2MPNNGenerator',
    'RFDiffusionGenerator',
]
