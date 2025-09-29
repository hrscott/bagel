"""Objective and constraint specifications for BAGEL energy aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

AggregationRule = Literal['sum', 'max', 'delta']


def aggregate_values(values: Iterable[float], rule: AggregationRule) -> float:
    """Aggregate ``values`` according to ``rule``.

    Parameters
    ----------
    values:
        Iterable of numeric values to aggregate.
    rule:
        Aggregation rule to apply. ``'sum'`` performs an arithmetic sum, ``'max'``
        returns the maximum value, while ``'delta'`` computes a signed
        difference between the first element and the sum of the remaining
        elements. Empty collections return ``0.0`` for all rules.
    """

    collected = list(values)
    if not collected:
        return 0.0

    if rule == 'sum':
        return float(sum(collected))
    if rule == 'max':
        return float(max(collected))
    if rule == 'delta':
        head, *tail = collected
        return float(head - sum(tail))
    raise ValueError(f'Unsupported aggregation rule: {rule}')


@dataclass(frozen=True)
class ConstraintSpec:
    """Defines how a constraint metric should be aggregated and weighted."""

    constraint_id: str
    aggregation: AggregationRule = 'delta'
    weight: float = 1.0
    states: Sequence[str] | None = None


@dataclass(frozen=True)
class ObjectiveSpec:
    """Configuration for aggregating an objective across energy terms/states."""

    objective_id: str
    aggregation: AggregationRule = 'sum'
    weight: float = 1.0
    constraints: Sequence[ConstraintSpec] = field(default_factory=tuple)


DEFAULT_OBJECTIVE_ID = 'total_energy'


def default_objective_spec() -> ObjectiveSpec:
    """Return a fresh :class:`ObjectiveSpec` for the legacy total energy."""

    return ObjectiveSpec(objective_id=DEFAULT_OBJECTIVE_ID, aggregation='sum', weight=1.0)


def default_objective_specs() -> dict[str, ObjectiveSpec]:
    """Return the default mapping of objective specifications."""

    spec = default_objective_spec()
    return {spec.objective_id: spec}

