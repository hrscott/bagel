import bagel as bg
import pandas as pd
import pathlib as pl
import shutil
from typing import Any, Callable
from bagel.objectives import DEFAULT_OBJECTIVE_ID, ConstraintSpec, ObjectiveSpec, default_objective_spec
from biotite.sequence.io.fasta import FastaFile
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np
from unittest.mock import Mock


def test_system_dump_config_file_is_correct(mixed_system: bg.System) -> None:
    mock_output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / mixed_system.name
    mock_experiment = test_system_dump_config_file_is_correct.__name__

    experiment_folder = mock_output_folder / mock_experiment
    if experiment_folder.exists():  # clean data folder if it exists
        shutil.rmtree(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)

    mixed_system.dump_config(experiment_folder)

    file = mock_output_folder / mock_experiment / 'config.csv'
    assert file.exists(), 'config file not present in expected location'

    # Check the version.txt file
    version_file = mock_output_folder / mock_experiment / 'version.txt'
    assert version_file.exists(), 'version.txt file not present in expected location'
    with open(version_file, 'r') as vfile:
        version_line = vfile.readline().strip()
    assert version_line == str(bg.__version__), f'version.txt does not match version: {version_line}'

    # Now read the CSV
    config = pd.read_csv(file)

    assert all(config.columns == ['state', 'energy', 'weight']), 'config does not contain correct columns'
    correct_config = pd.DataFrame(
        {
            'state': ['small', 'small', 'mixed', 'mixed'],
            'energy': [
                'pTM',
                'selective_surface_area',
                'local_pLDDT',
                'cross_PAE',
                #'pTM',
                #'normalized_globular',
            ],
            'weight': [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert all(config == correct_config), 'data within config file is incorrect'  # checks non index cols

    shutil.rmtree(experiment_folder)


def test_system_dump_logs_folder_is_correct(mixed_system: bg.System) -> None:
    mock_step = 0
    mock_output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / mixed_system.name
    mock_experiment = test_system_dump_logs_folder_is_correct.__name__

    experiment_folder = mock_output_folder / mock_experiment
    if experiment_folder.exists():  # clean data folder if it exists
        shutil.rmtree(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # Ensure system energy is calculated
    assert mixed_system.total_energy is not None, 'System energy should be calculated before dumping logs'

    # Call dump_logs with the parameters aligned with the current implementation
    mixed_system.dump_logs(step=mock_step, path=experiment_folder, save_structure=True)

    sequences = {
        header: sequence
        for header, sequence in FastaFile.read_iter(file=experiment_folder / f'{mixed_system.states[1].name}.fasta')
    }
    assert sequences == {'0': 'G:VV:GVVV'}, 'incorrect sequence information saved'

    masks = {
        header: mask
        for header, mask in FastaFile.read_iter(file=experiment_folder / f'{mixed_system.states[1].name}.mask.fasta')
    }
    assert masks == {'0': 'M:MM:MIIM'}, 'incorrect mutability mask information saved'

    oracle = mixed_system.states[0].oracles_list[0]
    oracle_name = type(oracle).__name__
    assert oracle_name == 'ESMFold', 'incorrect oracle information saved'
    assert oracle == mixed_system.states[1].oracles_list[0], 'inconsistent oracles between states'

    structures = {
        'small': get_structure(
            CIFFile().read(file=experiment_folder / 'structures' / f'small_{oracle_name}_{mock_step}.cif')
        )[0],
        'mixed': get_structure(
            CIFFile().read(file=experiment_folder / 'structures' / f'mixed_{oracle_name}_{mock_step}.cif')
        )[0],
    }
    correct_structures = {
        'small': mixed_system.states[0]._oracles_result[oracle].structure,
        'mixed': mixed_system.states[1]._oracles_result[oracle].structure,
    }

    energies = pd.read_csv(experiment_folder / 'energies.csv')
    correct_energies = pd.DataFrame(
        {
            'step': [mock_step],
            'small:pTM:objectives:total_energy': [-0.7],
            'small:selective_surface_area:objectives:total_energy': [0.2],
            'mixed:local_pLDDT:objectives:total_energy': [-0.4],
            'mixed:cross_PAE:objectives:total_energy': [0.5],
            'small:objective:total_energy': [-0.5],
            'small:objective_raw:total_energy': [-0.5],
            'mixed:objective:total_energy': [0.1],
            'mixed:objective_raw:total_energy': [0.1],
            'small:state_energy': [-0.5],
            'mixed:state_energy': [0.1],
            'system:objective:total_energy': [-0.4],
            'system_energy': [-0.4],
        }
    )

    assert structures == correct_structures, 'incorrect structure information saved'

    # Check column names match
    assert set(energies.columns) == set(correct_energies.columns), "DataFrame columns don't match"

    # Sort columns to ensure same order and compare values only
    assert np.array_equal(energies.sort_index(axis=1).values, correct_energies.sort_index(axis=1).values), (
        'incorrect energy information saved'
    )

    # load the pae and plddt files
    small_pae = np.loadtxt(experiment_folder / 'structures' / f'small_{oracle_name}_{mock_step}.pae')
    small_plddt = np.loadtxt(experiment_folder / 'structures' / f'small_{oracle_name}_{mock_step}.plddt')
    mixed_pae = np.loadtxt(experiment_folder / 'structures' / f'mixed_{oracle_name}_{mock_step}.pae')
    mixed_plddt = np.loadtxt(experiment_folder / 'structures' / f'mixed_{oracle_name}_{mock_step}.plddt')
    assert np.array_equal(small_pae, mixed_system.states[0]._oracles_result[oracle].pae[0]), (
        'incorrect pae information saved'
    )
    assert np.array_equal(small_plddt, mixed_system.states[0]._oracles_result[oracle].local_plddt[0]), (
        'incorrect plddt information saved'
    )
    assert np.array_equal(mixed_pae, mixed_system.states[1]._oracles_result[oracle].pae[0]), (
        'incorrect pae information saved'
    )
    assert np.array_equal(mixed_plddt, mixed_system.states[1]._oracles_result[oracle].local_plddt[0]), (
        'incorrect plddt information saved'
    )

    shutil.rmtree(experiment_folder)


def test_copied_system_is_independant_of_original_system(mixed_system: bg.System) -> None:
    copied_system = mixed_system.__copy__()
    mixed_system.states[0].chains[0].add_residue(amino_acid='A', index=0)
    assert mixed_system.states[0].chains[0] != copied_system.states[0].chains[0]


def test_system_states_still_reference_shared_chain_object_after_copy_method(shared_chain_system: bg.System) -> None:
    copied_system = shared_chain_system.__copy__()
    copied_system.states[0].chains[0].add_residue(amino_acid='A', index=0)
    assert copied_system.states[0].chains[0] == copied_system.states[1].chains[0]


def test_system_evaluate_gives_correct_output(mixed_system: bg.System) -> None:
    contributions = [-0.5, 0.1]

    for state, contribution in zip(mixed_system.states, contributions):
        def make_side_effect(value: float, target_state: bg.State) -> Callable[[dict[str, Any]], float]:
            def side_effect(_: dict[str, Any]) -> float:
                target_state._objective_metrics_weighted = {DEFAULT_OBJECTIVE_ID: value}
                return value

            return side_effect

        state.get_energy = Mock(side_effect=make_side_effect(contribution, state))

    totals = mixed_system.evaluate([DEFAULT_OBJECTIVE_ID])

    assert np.isclose(totals[DEFAULT_OBJECTIVE_ID], sum(contributions))
    assert np.isclose(mixed_system.total_energy, sum(contributions))


def test_system_multi_objective_delta_constraint() -> None:
    class DummyOracle(bg.oracles.base.Oracle):
        def predict(self, chains):
            return self.result_class()

    dummy_oracle = DummyOracle()

    class MockEnergy(bg.energies.EnergyTerm):
        def __init__(self, name: str, metrics: dict[str, float]) -> None:
            super().__init__(name=name, oracle=dummy_oracle, inheritable=False, weight=1.0)
            self._metrics = metrics

        def compute(self, oracles_result: bg.oracles.OraclesResultDict) -> bg.energies.EnergyTermResult:
            return bg.energies.EnergyTermResult(
                objectives=self._metrics,
                constraints={'binding_margin': self._metrics['binding']},
            )

    def make_state(value_total: float, value_binding: float, name: str) -> bg.State:
        term = MockEnergy(name=f'{name}_energy', metrics={DEFAULT_OBJECTIVE_ID: value_total, 'binding': value_binding})
        state = bg.State(name=name, chains=[], energy_terms=[term])
        state._oracles_result = bg.oracles.OraclesResultDict({dummy_oracle: dummy_oracle.result_class()})
        return state

    state_a = make_state(1.0, -2.0, 'state_a')
    state_b = make_state(0.5, -1.0, 'state_b')

    system = bg.System(states=[state_a, state_b])
    binding_spec = ObjectiveSpec(
        objective_id='binding',
        aggregation='delta',
        weight=1.0,
        constraints=(ConstraintSpec(constraint_id='binding_margin', aggregation='delta', weight=1.0),),
    )
    total_spec = default_objective_spec()
    system.objective_specs = {total_spec.objective_id: total_spec, binding_spec.objective_id: binding_spec}

    totals = system.evaluate([DEFAULT_OBJECTIVE_ID, 'binding'])

    assert pytest.approx(totals[DEFAULT_OBJECTIVE_ID]) == 1.5
    assert pytest.approx(totals['binding']) == -1.0
    assert pytest.approx(system.states[0]._constraint_metrics_weighted['binding_margin']) == -2.0
    assert pytest.approx(system.states[1]._constraint_metrics_weighted['binding_margin']) == -1.0
