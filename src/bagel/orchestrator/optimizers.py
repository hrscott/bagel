"""Adapters that connect BAGEL generators with optimisation libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

from ..system import System
from .design_spec import DesignSpec
from .evaluation import EvaluationRecord, evaluate_candidates
from .generators import BatchSpec, Generator
from .tracking import RunLogger

logger = logging.getLogger(__name__)


@dataclass
class NSGAIIAdapter:
    """Light-weight adapter around ``pymoo``'s NSGA-II interface."""

    generator: Generator
    evaluate_hook: Any = evaluate_candidates

    def run(
        self,
        design_spec: DesignSpec,
        initial_system: System,
        generations: int,
        population_size: int,
        tracker: RunLogger | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        current_system = initial_system
        for generation in range(generations):
            batch_spec = BatchSpec(
                base_system=current_system,
                batch_size=population_size,
                metadata={'generation': generation},
            )
            population = self.generator.generate(batch_spec)
            if not population:
                logger.warning('Generator returned no candidates for generation %s', generation)
                break
            generation_records = self.evaluate_hook(
                design_spec=design_spec,
                candidates=population,
                generator_name='nsga2',
                tracker=tracker,
                design_id_prefix=f'gen{generation}',
                provenance={**(provenance or {}), 'generation': generation},
            )
            records.extend(generation_records)
            current_system = population[0]
        return records


@dataclass
class AxBoTorchAdapter:
    """Adapter emulating Ax/BoTorch qEHVI loops for expensive evaluations."""

    generator: Generator
    evaluate_hook: Any = evaluate_candidates
    batch_size: int = 1

    def _is_feasible(self, record: EvaluationRecord, design_spec: DesignSpec) -> bool:
        for objective in design_spec.objectives:
            constraint = objective.constraint or {}
            metric_value = record.metrics.get(objective.metric)
            if metric_value is None:
                continue
            if 'max' in constraint and metric_value > constraint['max']:
                return False
            if 'min' in constraint and metric_value < constraint['min']:
                return False
        return True

    def run(
        self,
        design_spec: DesignSpec,
        initial_system: System,
        iterations: int,
        tracker: RunLogger | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        current_system = initial_system
        for iteration in range(iterations):
            batch_spec = BatchSpec(
                base_system=current_system,
                batch_size=self.batch_size,
                metadata={'iteration': iteration},
            )
            candidates = self.generator.generate(batch_spec)
            if not candidates:
                logger.warning('Generator returned no candidates for iteration %s', iteration)
                break
            for candidate_index, candidate in enumerate(candidates):
                iteration_records = self.evaluate_hook(
                    design_spec=design_spec,
                    candidates=[candidate],
                    generator_name='qehvi',
                    tracker=tracker,
                    design_id_prefix=f'iter{iteration}-cand{candidate_index}',
                    provenance={**(provenance or {}), 'iteration': iteration},
                )
                for record in iteration_records:
                    if self._is_feasible(record, design_spec):
                        records.append(record)
                current_system = candidate
        return records


__all__ = ['NSGAIIAdapter', 'AxBoTorchAdapter']
