from dataclasses import dataclass, field
from enum import auto
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom, Bond
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.base import Featurizer, GraphFeaturizer, VectorFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from chemprop.utils.utils import EnumMapping, parallel_execute

from chemprop_contrib.mixtures.data.molgraph import ComponentMolGraph, MixtureGraph


class _EmptyMolFeaturizer(VectorFeaturizer[Mol]):
    """mol featurizer that returns an empty feature vector"""

    def __call__(self, mol: Mol) -> np.ndarray:
        return np.empty((0,))

    def __len__(self) -> int:
        return 0


@dataclass
class ComponentMolGraphFeaturizer(Featurizer[Mol | None, ComponentMolGraph | None]):
    """A :class:`ComponentMolGraphFeaturizer` produces :class:`ComponentMolGraph`s.

    Parameters
    ----------
    atom_featurizer : VectorFeaturizer[Atom], default=MultiHotAtomFeaturizer.v2()
    bond_featurizer : VectorFeaturizer[Bond], default=MultiHotBondFeaturizer()
    mol_featurizer : VectorFeaturizer[Mol], default=_EmptyMolFeaturizer()
        the featurizer with which to calculate feature representations of a given molecule
    extra_atom_fdim : int, default=0
    extra_bond_fdim : int, default=0
    extra_mol_fdim : int, default=0
            the dimension of the additional features that will be concatenated onto the calculated
            features of each mol
    """

    atom_featurizer: VectorFeaturizer[Atom] = field(default_factory=MultiHotAtomFeaturizer.v2)
    bond_featurizer: VectorFeaturizer[Bond] = field(default_factory=MultiHotBondFeaturizer)
    mol_featurizer: VectorFeaturizer[Mol] = field(default_factory=_EmptyMolFeaturizer)
    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0
    extra_mol_fdim: int = 0

    def __post_init__(self):
        self.atom_fdim = len(self.atom_featurizer) + self.extra_atom_fdim
        self.bond_fdim = len(self.bond_featurizer) + self.extra_bond_fdim
        self.graph_fdim = len(self.mol_featurizer) + self.extra_mol_fdim

    @property
    def shape(self) -> tuple[int, int, int]:
        """the feature dimension of atoms, bonds, and graph features, respectively"""
        return self.atom_fdim, self.bond_fdim, self.graph_fdim

    def __call__(
        self,
        mol: Chem.Mol | None,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
        mol_features_extra: np.ndarray | None = None,
        w_fp: float = 1.0,
    ) -> ComponentMolGraph | None:
        if mol is None:
            return None

        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )
        if mol_features_extra is not None and mol_features_extra.ndim != 1:
            raise ValueError(
                "`graph_features_extra` must be a 1-D array! "
                f"got: {mol_features_extra.ndim}-D array with shape {mol_features_extra.shape}"
            )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

            E[i : i + 2] = x_e

            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        G = self.mol_featurizer(mol)
        if mol_features_extra is not None:
            G = np.concatenate([G, mol_features_extra], dtype=np.single)

        return ComponentMolGraph(V, E, edge_index, rev_edge_index, G, w_fp)


class ComponentMolGraphCache:
    """
    A :class:`ComponentMolGraphCache` precomputes the corresponding
    :class:`~chemprop_contrib.mixture.data.molgraph.ComponentMolGraph`\s and caches them in memory.
    """

    def __init__(
        self,
        mols: Iterable[Mol],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        G_d: Iterable[np.ndarray | None],
        w_fps: np.ndarray,
        featurizer: Featurizer[Mol, ComponentMolGraph],
        n_workers: int = 0,
    ):
        self._mgs = parallel_execute(featurizer, zip(mols, V_fs, E_fs, G_d, w_fps), n_workers=n_workers)

    def __len__(self) -> int:
        return len(self._mgs)

    def __getitem__(self, index: int) -> ComponentMolGraph:
        return self._mgs[index]


class ComponentMolGraphCacheOnTheFly:
    """
    A :class:`ComponentMolGraphCacheOnTheFly` computes the corresponding
    :class:`~chemprop_contrib.mixture.data.molgraph.ComponentMolGraph`\s as they are requested.
    """

    def __init__(
        self,
        mols: Iterable[Mol],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        G_d: Iterable[np.ndarray | None],
        w_fps: np.ndarray,
        featurizer: Featurizer[Mol, ComponentMolGraph],
    ):
        self._mols = list(mols)
        self._V_fs = list(V_fs)
        self._E_fs = list(E_fs)
        self._G_d = list(G_d)
        self.w_fps = w_fps
        self._featurizer = featurizer

    def __len__(self) -> int:
        return len(self._mols)

    def __getitem__(self, index: int) -> ComponentMolGraph:
        return self._featurizer(self._mols[index], self._V_fs[index], self._E_fs[index], self._G_d[index], self.w_fps[index])


class MultiHotMolInteractionFeaturizer(VectorFeaturizer[Sequence[Mol]]):
    """A :class:`MultiHotMolInteractionFeaturizer` uses a multi-hot encoding to featurize
    interactions in mixture graphs.

    The generated interaction features are ordered as follows:
    * hydrogen bonding

    Parameters
    ----------
    h_bonds : Sequence[int]
        the numbers of hydrogen bonds to include a bit for
    """

    def __init__(self, h_bonds: Sequence[int]):
        self.h_bonds = {i: i for i in h_bonds}

        self._subfeats: list[dict] = [self.h_bonds]
        subfeat_sizes = [1 + len(self.h_bonds)]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, mols: Sequence[Mol] | None) -> np.ndarray:
        x = np.zeros(self.__size)

        if mols is None:
            return x
        if len(mols) == 1:
            feats = [self._descriptor_intra_HB(mols[0])]
        elif len(mols) > 1:
            feats = [self._descriptor_inter_HB(mols)]
        else:
            raise ValueError(
                "Tried to featurize molecular interactions but empty list of mols was provided."
            )

        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1

        return x

    def _descriptor_intra_HB(self, mol):
        # From https://github.com/edgarsmdn/GH-GNN/blob/main/scr/models/utilities/mol2graph.py
        # Intra hydrogen-bond acidity and basicity
        return min(Chem.rdMolDescriptors.CalcNumHBA(mol), Chem.rdMolDescriptors.CalcNumHBD(mol))

    def _descriptor_inter_HB(self, mol_list):
        # From https://github.com/edgarsmdn/GH-GNN/blob/main/scr/models/utilities/mol2graph.py
        # Inter hydrogen-bond acidity and basicity
        if len(mol_list) != 2:
            raise ValueError(
                "Interaction can only be calculated between two molecules. "
                f"{len(mol_list)} molecules are given."
            )
        mol_1, mol_2 = mol_list[0], mol_list[1]
        return min(
            Chem.rdMolDescriptors.CalcNumHBA(mol_1),
            Chem.rdMolDescriptors.CalcNumHBD(mol_2),
        ) + min(
            Chem.rdMolDescriptors.CalcNumHBA(mol_2),
            Chem.rdMolDescriptors.CalcNumHBD(mol_1),
        )

    @classmethod
    def hb(cls, max_hbond_num: int = 3):
        """The implementation of molecular interactions based on hydrogen bonding (hb) used in [1].

        Parameters
        ----------
        max_hbond_num : int, default=3
            Include a bit for all hydrogen bond numbers in the interval :math:`[0, \mathtt{max\_hbond\_num}]`

        References
        -----------
        .. [1] Medina, E. I. S., Linke, S., Stoll, M., & Sundmacher, K. (2023). Gibbs–Helmholtz
        graph neural network: capturing the temperature dependency of activity coefficients at
        infinite dilution. Digital Discovery, 2(3), 781-798. https://doi.org/10.1039/D2DD00142J
        """

        return cls(h_bonds=list(range(0, max_hbond_num + 1)))


class MolInteractionFeatureMode(EnumMapping):
    """The mode of an atom is used for featurization into a `MolGraph`"""

    HB = auto()


@dataclass
class _MixtureGraphFeaturizerMixin:
    # TODO: we could also featurize mols and thereby provide the option to omit the MPNN on
    # individual molecules.
    # mol_featurizer: VectorFeaturizer[Mol] = field(default_factory=)
    interaction_featurizer: VectorFeaturizer[list[Mol]] = field(
        default_factory=MultiHotMolInteractionFeaturizer.hb
    )

    def __post_init__(self):
        self.mol_fdim = 0  # len(self.mol_featurizer), TODO
        self.interaction_fdim = len(self.interaction_featurizer)

    @property
    def shape(self) -> tuple[int, int]:
        """the feature dimension of the molecules and interactions, respectively, of `MixtureGraph`s
        generated by this featurizer"""
        return self.mol_fdim, self.interaction_fdim


@dataclass
class SimpleMixtureGraphFeaturizer(_MixtureGraphFeaturizerMixin, GraphFeaturizer[Sequence[Mol]]):
    """
    Parameters
    ----------
    interaction_featurizer : InteractionFeaturizer, default=MultiHotMolInteractionFeaturizer()
        the featurizer with which to calculate feature representations of the molecular interactions
        in a given mixture
    extra_interaction_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each interaction
    """

    extra_mol_fdim: int = 0
    extra_interaction_fdim: int = 0

    def __post_init__(self):
        super().__post_init__()

        self.mol_fdim += self.extra_mol_fdim
        self.interaction_fdim += self.extra_interaction_fdim

    def __call__(
        self,
        mols: list[Chem.Mol],
        mol_features_extra: np.ndarray | None = None,
        interaction_features_extra: np.ndarray | None = None,
    ) -> MixtureGraph:
        n_mols = len(mols)
        n_interactions = int((n_mols - 1) * n_mols / 2 + n_mols)

        if mol_features_extra is not None and len(mol_features_extra) != n_mols:
            raise ValueError(
                "Input mixture must have same number of molecules as `len(mol_features_extra)`! "
                f"got: {n_mols} and {len(mol_features_extra)}, respectively"
            )

        if (
            interaction_features_extra is not None
            and len(interaction_features_extra) != n_interactions
        ):
            raise ValueError(
                "Input mixture must have same number of interactions (n_mols * (n_mols / 2 + 1)) as"
                " `len(interaction_features_extra)`! "
                f"got: {n_interactions} and {len(interaction_features_extra)}, respectively"
            )

        if n_mols == 0:
            V = np.zeros((1, self.mol_fdim), dtype=np.single)
        elif mol_features_extra is not None:
            V = mol_features_extra
        else:
            V = np.zeros((n_mols, self.mol_fdim), dtype=np.single)

        E = np.empty((2 * n_interactions, self.interaction_fdim))
        edge_index = [[], []]
        i = 0
        k = 0
        for mol1_idx, mol1 in enumerate(mols):
            for mol2_idx, mol2 in enumerate(mols):
                # avoid duplicate interactions
                if mol2_idx < mol1_idx:
                    continue
                # self-interaction
                if mol1_idx == mol2_idx:
                    x_e = self.interaction_featurizer([mol1])
                # intermolecular interaction
                else:
                    x_e = self.interaction_featurizer([mol1, mol2])
                if interaction_features_extra is not None:
                    x_e = np.concatenate(
                        (x_e, interaction_features_extra[k]),
                        dtype=np.single,
                    )

                E[i : i + 2] = x_e

                edge_index[0].extend([mol1_idx, mol2_idx])
                edge_index[1].extend([mol2_idx, mol1_idx])

                i += 2
                k += 1

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MixtureGraph(V, E, edge_index, rev_edge_index)
