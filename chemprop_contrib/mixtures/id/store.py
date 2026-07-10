from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, InitVar, field
from datetime import datetime, timezone
from pathlib import Path
import pickle
import platform
from typing import Any, TypeAlias

import numpy as np
from rdkit import Chem, rdBase

import chemprop
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer


SMILES: TypeAlias = str
MoleculeIndex: TypeAlias = int
SMILESDict: TypeAlias = dict[MoleculeIndex, SMILES]
IndexDict: TypeAlias = dict[SMILES, MoleculeIndex]
MolDict: TypeAlias = dict[MoleculeIndex, Chem.Mol]
MolGraphDict: TypeAlias = dict[MoleculeIndex, MolGraph]
AtomCountDict: TypeAlias = dict[MoleculeIndex, int]
BondCountDict: TypeAlias = dict[MoleculeIndex, int]
MakeMolFunc: TypeAlias = Callable[[SMILES], Chem.Mol]
MakeMolGraphFunc: TypeAlias = Callable[[Chem.Mol], MolGraph]


def molgraph_to_dict(mg: MolGraph) -> dict[str, np.ndarray]:
    return {
        "V": mg.V,
        "E": mg.E,
        "edge_index": mg.edge_index,
        "rev_edge_index": mg.rev_edge_index,
    }


def dict_to_molgraph(d: dict[str, np.ndarray]) -> MolGraph:
    return MolGraph(d["V"], d["E"], d["edge_index"], d["rev_edge_index"])


@dataclass
class MolGraphStore:
    """A persistent store mapping unique SMILES strings to integer IDs and their precomputed graphs.

    A ``MolGraphStore`` deduplicates a collection of SMILES strings, assigns each unique string an
    integer ID, and then builds both the RDKit Chem.Mol's and the chemprop :class:`MolGraph`
    objects. The entire store can be pickled to and loaded from disk.

    Parameters
    ----------
    smiles_strings : Iterable[str] | None, default=None
        an iterable of SMILES strings to populate the store. Duplicates are removed. If ``None``,
        the other dictionary args are required (used internally by :meth:`load`).
    make_mol_func : MakeMolFunc | None, default=None
        an optional callable that converts SMILES strings to ``Chem.Mol``. If ``None``, chemprop's
        ``make_mol`` is used.
    make_molgraph_func : MakeMolGraphFunc | None, default=None
        an optional callable that converts ``Chem.Mol`` to ``MolGraph``. If ``None``, chemprop's
        ``SimpleMoleculeMolGraphFeaturizer`` is used.
    id_to_smiles : SMILESDict
        mapping from molecule ID to SMILES string, populated automatically
    smiles_to_id : IndexDict
        mapping from SMILES string to molecule ID, populated automatically
    id_to_mol : MolDict
        mapping from molecule ID to RDKit molecule, populated automatically
    id_to_atom_count : AtomCountDict
        mapping from molecule ID to number of atoms, populated automatically
    id_to_bond_count : BondCountDict
        mapping from molecule ID to number of bonds, populated automatically
    id_to_graph : MolGraphDict
        mapping from molecule ID to its :class:`MolGraph`, populated automatically
    metadata : dict[str, Any]
        provenance information (creation time, and Python, chemprop, and RDKit versions)
    """

    smiles_strings: InitVar[Iterable[str] | None] = None
    make_mol_func: InitVar[MakeMolFunc | None] = None
    make_molgraph_func: InitVar[MakeMolGraphFunc | None] = None

    id_to_smiles: SMILESDict = field(default_factory=dict)
    smiles_to_id: IndexDict = field(default_factory=dict)
    id_to_mol: MolDict = field(default_factory=dict)
    id_to_atom_count: AtomCountDict = field(default_factory=dict)
    id_to_bond_count: BondCountDict = field(default_factory=dict)
    id_to_graph: MolGraphDict = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(
        self,
        smiles_strings: Iterable[str] | None,
        make_mol_func: Callable[[SMILESDict], MolDict] | None,
        make_molgraph_func: Callable[[MolDict], MolGraphDict] | None,
    ) -> None:
        if smiles_strings is None:  # Skip creation when self.load is called
            return

        unique = list(dict.fromkeys(smiles_strings))

        self.id_to_smiles = dict(enumerate(unique))
        self.smiles_to_id = {ident: idx for idx, ident in enumerate(unique)}

        make_mol_func = make_mol_func or chemprop.utils.make_mol
        self.id_to_mol = {idx: make_mol_func(smiles) for idx, smiles in self.id_to_smiles.items()}
        self.id_to_atom_count = {idx: mol.GetNumAtoms() for idx, mol in self.id_to_mol.items()}
        self.id_to_bond_count = {idx: mol.GetNumBonds() for idx, mol in self.id_to_mol.items()}

        make_molgraph_func = make_molgraph_func or SimpleMoleculeMolGraphFeaturizer()
        self.id_to_graph = {idx: make_molgraph_func(mol) for idx, mol in self.id_to_mol.items()}

        self.metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "chemprop_version": chemprop.__version__,
            "rdkit_version": rdBase.rdkitVersion,
        }

    def __getitem__(self, mol_id: int) -> MolGraph:
        return self.id_to_graph[mol_id]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": self.metadata,
            "id_to_smiles": self.id_to_smiles,
            "smiles_to_id": self.smiles_to_id,
            "id_to_mol": self.id_to_mol,
            "id_to_atom_count": self.id_to_atom_count,
            "id_to_bond_count": self.id_to_bond_count,
            "id_to_graph": {k: molgraph_to_dict(v) for k, v in self.id_to_graph.items()},
        }

        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> MolGraphStore:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)

        return cls(
            id_to_smiles=payload["id_to_smiles"],
            smiles_to_id=payload["smiles_to_id"],
            id_to_mol=payload["id_to_mol"],
            id_to_atom_count=payload["id_to_atom_count"],
            id_to_bond_count=payload["id_to_bond_count"],
            id_to_graph={k: dict_to_molgraph(v) for k, v in payload["id_to_graph"].items()},
            metadata=payload["metadata"],
        )


class IDMolGraphCache:
    """A callable that retrieves precomputed :class:`MolGraph`\\ s by ID and optionally augments the
    node and edge features.

    Parameters
    ----------
    mg_store : MolGraphStore
        the store providing base molecular graphs keyed by molecule ID
    """

    def __init__(self, mg_store: MolGraphStore) -> None:
        self.mg_store = mg_store

    @staticmethod
    def augment_molgraph(
        base: MolGraph,
        atom_features_extra: np.ndarray | None,
        bond_features_extra: np.ndarray | None,
    ) -> MolGraph:
        if atom_features_extra is not None:
            V = np.hstack((base.V, np.asarray(atom_features_extra, dtype=np.float32)))
        else:
            V = base.V
        if bond_features_extra is not None:
            E = np.hstack((base.E, np.repeat(bond_features_extra, repeats=2, axis=0)))
        else:
            E = base.E
        return MolGraph(V, E, base.edge_index, base.rev_edge_index)

    def get_molgraph(self, mol_id: int) -> MolGraph:
        return self.mg_store[mol_id]

    def __call__(
        self,
        mol_ids: list,
        V_fs: list[np.ndarray | None],
        E_fs: list[np.ndarray | None],
    ) -> list[MolGraph]:
        """Retrieve and augment a batch of :class:`MolGraph`.

        Parameters
        ----------
        mol_ids : list
            the molecule IDs to look up. list[int] for molecules and list[list[int]] for mixtures.
        V_fs : list[np.ndarray | None]
            extra per-atom descriptors for each molecule or mixture
        E_fs : list[np.ndarray | None]
            extra per-bond descriptors for each molecule or mixture

        Returns
        -------
        list[MolGraph]
            the retrieved graphs, each augmented with its corresponding extra features
        """
        return [
            self.augment_molgraph(self.get_molgraph(mol_id), V_f, E_f)
            for mol_id, V_f, E_f in zip(mol_ids, V_fs, E_fs)
        ]


class IDMixtureMolGraphCache(IDMolGraphCache):
    """A callable that retrieves precomputed :class:`MolGraph`\\ s by ID, merges those corresponding
    to molecules in a mixture into a single disconnected graph for each mixture, and optionally
    augments the node and edge features.

    Parameters
    ----------
    mg_store : MolGraphStore
        the store providing base molecular graphs keyed by molecule ID
    """

    def __init__(self, mg_store: MolGraphStore) -> None:
        super().__init__(mg_store)
        self._combined_molgraph_cache: dict[tuple[int, ...], MolGraph] = {}

    @staticmethod
    def combine_molgraphs(graphs: list[MolGraph]) -> MolGraph:
        V = np.concatenate([g.V for g in graphs], axis=0)
        E = np.concatenate([g.E for g in graphs], axis=0)

        edge_parts = []
        rev_parts = []
        atom_offset = 0
        edge_offset = 0
        for g in graphs:
            edge_parts.append(g.edge_index + atom_offset)
            rev_parts.append(g.rev_edge_index + edge_offset)
            atom_offset += g.V.shape[0]
            edge_offset += g.E.shape[0]

        edge_index = np.concatenate(edge_parts, axis=1)
        rev_edge_index = np.concatenate(rev_parts)

        return MolGraph(V, E, edge_index, rev_edge_index)

    def get_molgraph(self, mol_id: list[int]) -> MolGraph:
        if len(mol_id) == 1:
            return self.mg_store[mol_id[0]]
        key = tuple(mol_id)
        mg = self._combined_molgraph_cache.get(key)
        if mg is None:
            mg = self.combine_molgraphs([self.mg_store[i] for i in mol_id])
            self._combined_molgraph_cache[key] = mg
        return mg
