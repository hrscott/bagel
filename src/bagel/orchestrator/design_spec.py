"""Utilities for loading and representing design specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class TargetSpec:
    name: str
    states: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSpec:
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveSpec:
    name: str
    metric: str
    direction: str = 'min'
    weight: float = 1.0
    constraint: dict[str, Any] | None = None


@dataclass
class DesignSpec:
    targets: list[TargetSpec]
    states: dict[str, StateSpec]
    objectives: list[ObjectiveSpec]

    def get_objective_metrics(self) -> list[str]:
        return [objective.metric for objective in self.objectives]


class DesignSpecLoader:
    """Load :class:`DesignSpec` instances from dictionaries or files."""

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> DesignSpec:
        if 'states' not in data:
            raise ValueError('Design spec must define "states" section')
        if 'targets' not in data:
            raise ValueError('Design spec must define "targets" section')
        if 'objectives' not in data:
            raise ValueError('Design spec must define "objectives" section')

        states = {
            state_data['name']: StateSpec(
                name=state_data['name'],
                metadata={key: value for key, value in state_data.items() if key != 'name'},
            )
            for state_data in data['states']
        }

        targets = [
            TargetSpec(
                name=target_data['name'],
                states=list(target_data.get('states', [])),
                metadata={key: value for key, value in target_data.items() if key not in {'name', 'states'}},
            )
            for target_data in data['targets']
        ]

        objectives = [
            ObjectiveSpec(
                name=objective_data['name'],
                metric=objective_data.get('metric', objective_data['name']),
                direction=objective_data.get('direction', 'min'),
                weight=float(objective_data.get('weight', 1.0)),
                constraint=objective_data.get('constraint'),
            )
            for objective_data in data['objectives']
        ]

        return DesignSpec(targets=targets, states=states, objectives=objectives)

    @staticmethod
    def from_file(path: str | Path) -> DesignSpec:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        if file_path.suffix.lower() in {'.yaml', '.yml'}:
            if yaml is None:  # pragma: no cover - optional dependency
                raise RuntimeError('pyyaml is required to load YAML design specs')
            with file_path.open('r') as handle:
                data = yaml.safe_load(handle)
        else:
            with file_path.open('r') as handle:
                data = json.load(handle)

        if not isinstance(data, Mapping):
            raise TypeError('Design spec file must contain a mapping at the top level')

        return DesignSpecLoader.from_dict(data)


__all__ = ['DesignSpec', 'DesignSpecLoader', 'ObjectiveSpec', 'StateSpec', 'TargetSpec']
