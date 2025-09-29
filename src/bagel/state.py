"""
Standard object to encode the tertiary structure, losses, and folding logic for a chain or complex of chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from biotite.structure.io.pdbx import CIFFile, set_structure

from .chain import Chain
from .energies import EnergyTerm, EnergyTermResult
from .objectives import (
    DEFAULT_OBJECTIVE_ID,
    ObjectiveSpec,
    aggregate_values,
    default_objective_specs,
)
from .oracles import FoldingOracle, Oracle, OraclesResultDict

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    A State is a multimeric collection of :class:`.Chain` objects with associated :class:`.EnergyTerm` objects.
    Chains can be independent of other States, or be shared between multiple States.

    Parameters
    ----------
    name : str
        Unique identifier for this State.
    chains : List[:class:`.Chain`]
        List of single monomeric Chains in this State.
    energy_terms : List[:class:`.EnergyTerm`]
        Collection of EnergyTerms that define the State.

    Attributes
    ----------
    _energy : Optional[float]
        Cached total (weighted) energy value for the State.
    _oracles_result : dict[Oracle, OracleResult]
        Results of different oracles, e.g., folding, embedding, etc.
    _energy_terms_value : dict[str, dict[str, dict[str, float]]]
        Cached raw objective and constraint metrics for individual
        :class:`.EnergyTerm` objects.
    objective_specs : dict[str, :class:`.ObjectiveSpec`]
        Per-objective aggregation rules applied within this state.
    _objective_metrics_weighted : dict[str, float]
        Aggregated objective values after applying energy term weights and
        objective weights.
    _objective_metrics_raw : dict[str, float]
        Aggregated objective values before applying the objective weights.
    _constraint_metrics_weighted : dict[str, float]
        Aggregated constraint values after weighting.
    _constraint_metrics_raw : dict[str, float]
        Aggregated constraint values before weighting.
    """

    name: str
    chains: List[Chain]
    energy_terms: List[EnergyTerm]
    _energy: Optional[float] = field(default=None, init=False)
    _oracles_result: OraclesResultDict = field(default_factory=lambda: OraclesResultDict(), init=False)
    _energy_terms_value: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict, init=False)
    objective_specs: dict[str, ObjectiveSpec] | None = field(default=None)
    _objective_metrics_weighted: dict[str, float] = field(default_factory=dict, init=False)
    _objective_metrics_raw: dict[str, float] = field(default_factory=dict, init=False)
    _constraint_metrics_weighted: dict[str, float] = field(default_factory=dict, init=False)
    _constraint_metrics_raw: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Sanity check."""
        if self.objective_specs is None:
            self.objective_specs = default_objective_specs()

    def __copy__(self) -> Any:
        """Copy the state object, setting the structure and energy to None."""
        return deepcopy(self)

    @property
    def oracles_list(self) -> list[Oracle]:
        return list(set([term.oracle for term in self.energy_terms]))

    @property
    def total_sequence(self) -> List[str]:
        return [chain.sequence for chain in self.chains]

    def get_energy(self, objective_specs: dict[str, ObjectiveSpec] | None = None) -> float:
        """Calculate energy of state using energy terms."""
        if not self._oracles_result:
            for oracle in self.oracles_list:
                if oracle not in self._oracles_result:
                    self._oracles_result[oracle] = oracle.predict(chains=self.chains)

        specs = objective_specs or self.objective_specs or default_objective_specs()
        self.objective_specs = specs

        energy_term_names = [term.name for term in self.energy_terms]
        assert len(energy_term_names) == len(set(energy_term_names)), (
            f"Energy term names must be unique. Found duplicates: {energy_term_names}. Please rename using 'name'."
        )

        per_objective_raw: dict[str, list[float]] = defaultdict(list)
        per_objective_weighted: dict[str, list[float]] = defaultdict(list)
        per_constraint_raw: dict[str, list[float]] = defaultdict(list)
        per_constraint_weighted: dict[str, list[float]] = defaultdict(list)

        self._energy_terms_value = {}

        for term in self.energy_terms:
            result: EnergyTermResult = term.compute(oracles_result=self._oracles_result)
            term_metrics: dict[str, dict[str, float]] = {'objectives': dict(result.objectives)}
            if result.constraints:
                term_metrics['constraints'] = dict(result.constraints)
            self._energy_terms_value[term.name] = term_metrics

            for objective_id, value in result.objectives.items():
                per_objective_raw[objective_id].append(float(value))
                per_objective_weighted[objective_id].append(float(value) * term.weight)
            for constraint_id, value in result.constraints.items():
                per_constraint_raw[constraint_id].append(float(value))
                per_constraint_weighted[constraint_id].append(float(value) * term.weight)
            logger.debug(f'Energy term {term.name} objectives: {result.objectives}')

        aggregated_weighted: dict[str, float] = {}
        aggregated_raw: dict[str, float] = {}
        constraint_weighted: dict[str, float] = {}
        constraint_raw: dict[str, float] = {}

        for objective_id, spec in specs.items():
            raw_values = per_objective_raw.get(objective_id, [])
            weighted_values = per_objective_weighted.get(objective_id, [])
            raw_value = aggregate_values(raw_values, spec.aggregation)
            weighted_value = aggregate_values(weighted_values, spec.aggregation) * spec.weight
            aggregated_raw[objective_id] = raw_value
            aggregated_weighted[objective_id] = weighted_value

            for constraint_spec in spec.constraints:
                raw_constraint_values = per_constraint_raw.get(constraint_spec.constraint_id, [])
                weighted_constraint_values = per_constraint_weighted.get(constraint_spec.constraint_id, [])
                raw_constraint = aggregate_values(raw_constraint_values, constraint_spec.aggregation)
                weighted_constraint = (
                    aggregate_values(weighted_constraint_values, constraint_spec.aggregation) * constraint_spec.weight
                )
                constraint_raw[constraint_spec.constraint_id] = raw_constraint
                constraint_weighted[constraint_spec.constraint_id] = weighted_constraint

        self._objective_metrics_weighted = aggregated_weighted
        self._objective_metrics_raw = aggregated_raw
        self._constraint_metrics_weighted = constraint_weighted
        self._constraint_metrics_raw = constraint_raw

        self._energy = aggregated_weighted.get(DEFAULT_OBJECTIVE_ID, 0.0)

        logger.debug(f'**Weighted** objective metrics for state {self.name}: {aggregated_weighted}')

        return self._energy

    def to_cif(self, oracle: FoldingOracle, filepath: Path) -> bool:
        """
        Write the state to a CIF file of a specific FoldingOracle.

        Parameters
        ----------
        filepath : Path
            Path to the file to write the CIF structure to.

        Returns
        -------
        bool
            True if the file was written successfully, False otherwise.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        structure_file = CIFFile()
        set_structure(structure_file, self._oracles_result.get_structure(oracle))
        logger.debug(f'Writing CIF structure of {self.name} from {type(oracle).__name__} to {filepath}')
        structure_file.write(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'Structure file {filepath} was not created')
        else:
            return True

    def total_residues(self) -> int:
        return sum([len(chain.residues) for chain in self.chains])

    def remove_residue_from_all_energy_terms(self, chain_ID: str, residue_index: int) -> None:
        """Remove the residue from the energy terms associated to it in the current state."""
        for term in self.energy_terms:
            # The order of these two operations is important. FIRST you remove the residue, THEN you shift the indices
            # If you do the opposite, you will remove the wrong residue
            term.remove_residue(chain_ID, residue_index)
            # ensuring residue indexes in energy terms are updated to reflect a change in chain length
            term.shift_residues_indices_after_removal(chain_ID, residue_index)

    def add_residue_to_all_energy_terms(self, chain_ID: str, residue_index: int) -> None:
        """
        You look within the same chain and the same state and you add the residue to the same energy terms the
        neighbours are part of. You actually look left and right, and randomly decide between the two. If the residue is
        at the beginning or at the end of the chain, you just look at one of them. You do it for all
        terms that are inheritable.
        """

        # Get the chain that needs to be checked to inherit the energy terms from the neighbours
        chains = self.chains
        chain = None
        for i in range(len(chains)):
            if chains[i].chain_ID == chain_ID:
                chain = chains[i]
                break

        if chain is None:
            # This is ok, it can happen if a residue is added to a chain that is not in one of the states
            return

        # Remember the following selection is done AFTER the residue has been added to the Chain object via chain.add_residue
        left_residue = chain.residues[residue_index - 1] if residue_index > 0 else None
        right_residue = chain.residues[residue_index + 1] if residue_index < len(chain.residues) - 1 else None
        # Now choose randomly between the left and the right residue, if they exist
        assert left_residue is not None or right_residue is not None, (
            'This should not be possible unless a whole chain has disappeared but was still picked for mutation'
        )
        if left_residue is None:
            parent_residue = right_residue
        elif right_residue is None:
            parent_residue = left_residue
        else:
            parent_residue = np.random.choice([left_residue, right_residue])  # type: ignore

        assert parent_residue is not None, 'The parent residue is None, should not happen!'
        # Now add the residue to the energy terms associated to the parent residue
        for term in self.energy_terms:
            # The order of these two operations is important.
            # **Opposite** to what you do when you remove, you FIRST shift indices, and only THEN add one in the 'hole'
            # created.

            # MUST be called BEFORE add_residue. In this way, the residue of the parent index in the
            # residue_group attributed of the energy term is correct and updated to the same value of residue.index
            term.shift_residues_indices_before_addition(chain_ID, residue_index)
            # Add the residue to the energy term if the parent residue is part of it and the term is inheritable
            # The function automatically checks if the parent is also in it, or not.
            if term.inheritable:
                # Just a double check here before proceeding
                parent_index = parent_residue.index
                assert parent_residue.chain_ID == chain_ID, (
                    'The parent residue is not in the same chain, should not happen!'
                )
                term.add_residue(chain_ID, residue_index, parent_index)
