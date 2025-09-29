"""Evaluation helpers for orchestrator workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from uuid import uuid4
import numpy as np

from ..system import System, EvaluationResult
from .design_spec import DesignSpec, ObjectiveSpec
from .tracking import RunLogger


@dataclass
class EvaluationRecord:
    design_id: str
    metrics: dict[str, float]
    vector_metrics: dict[str, np.ndarray] = field(default_factory=dict)
    generator: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    system: System | None = None

    def jsonable(self) -> dict[str, Any]:
        return {
            'design_id': self.design_id,
            'generator': self.generator,
            'metrics': self.metrics,
            'vector_metrics': {key: value.tolist() for key, value in self.vector_metrics.items()},
            'provenance': self.provenance,
        }

    def parquet_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            'design_id': self.design_id,
            'generator': self.generator,
        }
        row.update({f'metric::{key}': value for key, value in self.metrics.items()})
        for key, value in self.vector_metrics.items():
            row[f'vector::{key}'] = list(map(float, np.asarray(value).ravel()))
        for key, value in self.provenance.items():
            row[f'provenance::{key}'] = value
        return row


def _normalise_objectives(objectives: Sequence[str | ObjectiveSpec] | None) -> list[str] | None:
    if objectives is None:
        return None
    normalised: list[str] = []
    for objective in objectives:
        if isinstance(objective, str):
            normalised.append(objective)
        else:
            normalised.append(objective.metric)
    return normalised


def evaluate_candidates(
    design_spec: DesignSpec,
    candidates: Iterable[System],
    generator_name: str,
    tracker: RunLogger | None = None,
    design_id_prefix: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> list[EvaluationRecord]:
    """Evaluate a batch of systems and optionally persist the results."""

    records: list[EvaluationRecord] = []
    allowed_metrics = _normalise_objectives(design_spec.objectives)
    for index, system in enumerate(candidates):
        evaluation: EvaluationResult = system.evaluate(allowed_metrics)
        design_id = (
            f"{design_id_prefix}-{index}" if design_id_prefix is not None else f'{system.name or "design"}-{uuid4().hex[:8]}'
        )
        record = EvaluationRecord(
            design_id=design_id,
            metrics=evaluation.scalars,
            vector_metrics=evaluation.vector_metrics,
            generator=generator_name,
            provenance=dict(provenance or {}),
            system=system,
        )
        if tracker is not None:
            tracker.log(record)
        records.append(record)

    if tracker is not None:
        tracker.flush()

    return records


__all__ = ['EvaluationRecord', 'evaluate_candidates']
