from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from chemprop.data.datasets import MoleculeDataset

from chemprop_contrib.mixtures.data.datasets import MixtureDataset
from chemprop_contrib.mixtures.id.datapoints import IDMixtureDatapoint, IDMoleculeDatapoint
from chemprop_contrib.mixtures.id.store import (
    IDMixtureMolGraphCache,
    IDMolGraphCache,
    MolGraphStore,
)


@dataclass
class IDMoleculeDataset(MoleculeDataset):
    """A :class:`MoleculeDataset` of :class:`IDMoleculeDatapoint`\\ s.

    Molecular graphs are retrieved by ID from a :class:`MolGraphStore` (via an
    :class:`IDMolGraphCache`).

    Parameters
    ----------
    data : list[IDMoleculeDatapoint]
        the datapoints comprising the dataset
    mg_store : MolGraphStore | None, default=None
        the store providing precomputed molecular graphs. Required; a default of ``None`` is only
        used because it must follow the inherited :class:`MoleculeDataset` fields.
    """

    data: list[IDMoleculeDatapoint]
    mg_store: MolGraphStore | None = None

    def __post_init__(self):
        if self.mg_store is None:
            raise ValueError("`IDMoleculeDataset` requires argument `mg_store`")
        super().__post_init__()

    def _init_cache(self):
        cache = IDMolGraphCache(mg_store=self.mg_store)
        self.mg_cache = cache(self.mol_ids, self.V_fs, self.E_fs)

    @property
    def smiles(self) -> list[str]:
        return [self.mg_store.id_to_smiles[d.mol_id] for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [self.mg_store.id_to_mol[d.mol_id] for d in self.data]

    @property
    def mol_ids(self) -> list[int]:
        return [d.mol_id for d in self.data]


@dataclass
class IDMixtureDataset(MixtureDataset):
    """A :class:`MixtureDataset` of :class:`IDMixtureDatapoint`\\ s.

    Molecular graphs are retrieved by ID from a :class:`MolGraphStore` (via an
    :class:`IDMixtureMolGraphCache`).

    Parameters
    ----------
    data : list[IDMixtureDatapoint]
        the datapoints comprising the dataset
    mg_store : MolGraphStore | None, default=None
        the store providing precomputed molecular graphs. Required; a default of ``None`` is only
        used because it must follow the inherited :class:`MixtureDataset` fields.
    """

    data: list[IDMixtureDatapoint]
    mg_store: MolGraphStore | None = None

    def __post_init__(self):
        if self.mg_store is None:
            raise ValueError("`IDMixtureDataset` requires argument `mg_store`")
        super().__post_init__()

    def _init_cache(self):
        cache = IDMixtureMolGraphCache(mg_store=self.mg_store)
        self.mg_cache = cache(self.mol_idss, self.V_fs, self.E_fs)

    def get_molecule_sizes(self, idx: int) -> np.ndarray:
        return np.array(
            [self.mg_store.id_to_atom_count[mol_id] for mol_id in self.data[idx].mol_ids]
        )

    @property
    def molss(self) -> list[list[Chem.Mol]]:
        return [[self.mg_store.id_to_mol[mol_id] for mol_id in d.mol_ids] for d in self.data]

    @property
    def mol_idss(self) -> list[list[int]]:
        return [d.mol_ids for d in self.data]

    # Define self.mols and self.smiles for API consistency.
    @property
    def mols(self) -> list[Chem.Mol]:
        _mols = []
        molss = self.molss
        for idx in range(len(self.data)):
            mixture_mol = Chem.RWMol()
            for m in molss[idx]:
                mixture_mol.InsertMol(m)
            mol = mixture_mol.GetMol()
            mol.SetProp(
                "molecule_sizes", ",".join(self.get_molecule_sizes(idx).astype(str).tolist())
            )
            _mols.append(mol)
        return _mols

    @property
    def smiles(self) -> list[str]:
        return [Chem.MolToSmiles(mol) for mol in self.mols]
