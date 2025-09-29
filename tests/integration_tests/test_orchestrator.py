from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import bagel as bg
from bagel.orchestrator import (
    BatchSpec,
    DesignSpecLoader,
    MonteCarloGenerator,
    NSGAIIAdapter,
    RunLogger,
    evaluate_candidates,
)
from bagel.orchestrator.design_spec import DesignSpec
from bagel.oracles.base import Oracle, OracleResult


class DummyOracleResult(OracleResult):
    scalar: float
    vector: list[float]

    def save_attributes(self, filepath: Path) -> None:  # pragma: no cover - unused in tests
        filepath.write_text('dummy')

    def vector_metrics(self) -> dict[str, np.ndarray]:
        return {'dummy_vector': np.asarray(self.vector, dtype=float)}


class DummyOracle(Oracle):
    result_class = DummyOracleResult

    def predict(self, chains: list[bg.Chain]) -> DummyOracleResult:
        sequence = ''.join(chain.sequence for chain in chains)
        vector = [float(ord(char)) for char in sequence]
        return DummyOracleResult(input_chains=chains, scalar=float(len(sequence)), vector=vector)


class SequenceDiversityEnergy(bg.energies.EnergyTerm):
    def __init__(self, oracle: Oracle) -> None:
        super().__init__(name='diversity', oracle=oracle, inheritable=False, weight=1.0)

    def compute(self, oracles_result: bg.oracles.OraclesResultDict) -> tuple[float, float]:
        oracle_result = oracles_result[self.oracle]
        sequence = ''.join(chain.sequence for chain in oracle_result.input_chains)
        value = float(len(set(sequence)))
        return value, value


@pytest.fixture()
def toy_design_spec() -> DesignSpec:
    spec = {
        'targets': [{'name': 'toy', 'states': ['toy']}],
        'states': [{'name': 'toy'}],
        'objectives': [
            {'name': 'system_energy'},
            {'name': 'diversity', 'metric': 'toy:diversity', 'direction': 'max'},
            {'name': 'dummy_vector', 'metric': 'toy:dummy_vector'},
        ],
    }
    return DesignSpecLoader.from_dict(spec)


@pytest.fixture()
def toy_system() -> bg.System:
    chain = bg.Chain(
        residues=[
            bg.Residue(name='A', chain_ID='A', index=0, mutable=True),
            bg.Residue(name='C', chain_ID='A', index=1, mutable=True),
        ]
    )
    state = bg.State(name='toy', chains=[chain], energy_terms=[])
    oracle = DummyOracle()
    state.energy_terms.append(SequenceDiversityEnergy(oracle=oracle))
    return bg.System(states=[state], name='toy')


def test_evaluate_candidates_with_tracking(tmp_path: Path, toy_design_spec: DesignSpec, toy_system: bg.System) -> None:
    generator = MonteCarloGenerator(bg.mutation.Canonical(n_mutations=1), random_seed=42)
    batch_spec = BatchSpec(base_system=toy_system, batch_size=2)
    candidates = generator.generate(batch_spec)
    tracker = RunLogger(tmp_path)

    records = evaluate_candidates(
        design_spec=toy_design_spec,
        candidates=candidates,
        generator_name='monte-carlo',
        tracker=tracker,
        design_id_prefix='mc',
    )

    assert len(records) == 2
    for record in records:
        assert 'system_energy' in record.metrics
        assert 'toy:diversity' in record.metrics
        assert 'toy:dummy_vector' in record.vector_metrics

    tracker.flush()
    jsonl_contents = list(tmp_path.joinpath('evaluations.jsonl').read_text().strip().splitlines())
    assert len(jsonl_contents) == 2
    decoded = json.loads(jsonl_contents[0])
    assert decoded['generator'] == 'monte-carlo'


def run_monte_carlo_loop(
    generator: MonteCarloGenerator,
    design_spec: DesignSpec,
    system: bg.System,
    steps: int,
    tracker: RunLogger,
) -> list[float]:
    energies: list[float] = []
    current = system
    for step in range(steps):
        batch = generator.generate(BatchSpec(base_system=current, batch_size=1, metadata={'step': step}))
        records = evaluate_candidates(
            design_spec=design_spec,
            candidates=batch,
            generator_name='monte-carlo',
            tracker=tracker,
            design_id_prefix=f'mc-{step}',
        )
        energies.extend(record.metrics['system_energy'] for record in records)
        current = batch[0]
    return energies


def test_design_spec_reused_between_monte_carlo_and_nsga(
    tmp_path: Path,
    toy_design_spec: DesignSpec,
    toy_system: bg.System,
) -> None:
    tracker_mc = RunLogger(tmp_path / 'mc')
    generator = MonteCarloGenerator(bg.mutation.Canonical(n_mutations=1), random_seed=1)
    mc_energies = run_monte_carlo_loop(generator, toy_design_spec, toy_system, steps=2, tracker=tracker_mc)

    tracker_nsga = RunLogger(tmp_path / 'nsga')
    nsga = NSGAIIAdapter(generator=generator)
    nsga_records = nsga.run(
        design_spec=toy_design_spec,
        initial_system=toy_system,
        generations=2,
        population_size=2,
        tracker=tracker_nsga,
    )

    assert len(mc_energies) == 2
    assert len(nsga_records) == 4
    assert all('system_energy' in record.metrics for record in nsga_records)
    assert all(record.generator == 'nsga2' for record in nsga_records)

    jsonl_mc = (tmp_path / 'mc' / 'evaluations.jsonl').read_text().strip().splitlines()
    jsonl_nsga = (tmp_path / 'nsga' / 'evaluations.jsonl').read_text().strip().splitlines()
    assert jsonl_mc and jsonl_nsga
    mc_design_ids = {json.loads(entry)['design_id'] for entry in jsonl_mc}
    nsga_design_ids = {json.loads(entry)['design_id'] for entry in jsonl_nsga}
    assert all(design_id.startswith('mc-') for design_id in mc_design_ids)
    assert all(design_id.startswith('gen') for design_id in nsga_design_ids)
