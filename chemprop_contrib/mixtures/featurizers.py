from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from chemprop.data.molgraph import MolGraph

from chemprop.featurizers.base import GraphFeaturizer, VectorFeaturizer


class EmptyVectorFeaturizer(VectorFeaturizer[Mol]):
    """featurizer that returns an empty feature vector"""

    def __call__(self, x: Any) -> np.ndarray:
        return np.empty((0,))

    def __len__(self) -> int:
        return 0


class HydrogenBondFeaturizer(VectorFeaturizer[Sequence[Mol]]):
    """One-hot featurizer for hydrogen-bonding interactions in mixture graphs, based on the
    implementation of molecular interactions based on hydrogen bonding (hb) used in [1]. Adapted
    from https://github.com/edgarsmdn/GH-GNN/blob/main/scr/models/utilities/mol2graph.py

    For a single molecule, computes intra-molecular H-bonding as ``min(HBA, HBD)``. For a pair of
    molecules, computes inter-molecular H-bonding as ``min(HBA_1, HBD_2) + min(HBA_2, HBD_1)``. The
    resulting integer is one-hot encoded over ``[0, max_hbond_num]``, with an extra bit for values
    exceeding ``max_hbond_num``.

    Note that this featurizer uses `rdMolDescriptors.CalcNumHBA` and `rdMolDescriptors.CalcNumHBA`,
    which gives `0` for water.

    Parameters
    ----------
    max_hbond_num : int, default=3
        Include a bit for each H-bond count in ``[0, max_hbond_num]``, plus one overflow bit.

    References
    -----------
    .. [1] Medina, E. I. S., Linke, S., Stoll, M., & Sundmacher, K. (2023). Gibbs–Helmholtz graph
    neural network: capturing the temperature dependency of activity coefficients at infinite
    dilution. Digital Discovery, 2(3), 781-798. https://doi.org/10.1039/D2DD00142J
    """

    def __init__(self, max_hbond_num: int = 3):
        self.max_hbond_num = max_hbond_num
        self._size = max_hbond_num + 2

    def __len__(self) -> int:
        return self._size

    def __call__(self, mols: Sequence[Mol] | None) -> np.ndarray:
        x = np.zeros(self._size)
        if mols is None:
            return x

        if len(mols) == 1:
            mol = mols[0]
            n = min(
                Chem.rdMolDescriptors.CalcNumHBA(mol),
                Chem.rdMolDescriptors.CalcNumHBD(mol),
            )
        elif len(mols) == 2:
            m1, m2 = mols
            n = min(
                Chem.rdMolDescriptors.CalcNumHBA(m1),
                Chem.rdMolDescriptors.CalcNumHBD(m2),
            ) + min(
                Chem.rdMolDescriptors.CalcNumHBA(m2),
                Chem.rdMolDescriptors.CalcNumHBD(m1),
            )
        else:
            raise ValueError(f"Expected 1 or 2 molecules, got {len(mols)}.")

        x[n if n <= self.max_hbond_num else self.max_hbond_num + 1] = 1
        return x


@dataclass
class CompleteInteractionGraphFeaturizer(GraphFeaturizer[Sequence[Mol]]):
    """A :class:`CompleteInteractionGraphFeaturizer` creates a complete graph of interactions (every
    molecule interacts with every other molecule).

    Parameters
    ----------
    mol_featurizer : VectorFeaturizer[Mol], default=EmptyVectorFeaturizer()
        the featurizer with which to calculate feature representations of the molecules in the graph
    interaction_featurizer : VectorFeaturizer[list[Mol]], default=EmptyVectorFeaturizer()
        the featurizer with which to calculate feature representations of the interactions in the
        graph
    extra_mol_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each molecule
    extra_interaction_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each interaction
    self_loops : bool, default=False
        whether to include molecule self-interactions in the graph
    interact_only_mixture : bool, default=False
        whether to only include molecules in the first mixture in the interaction graph,
        i.e., exclude molecules not considered part of the mixture
    """

    mol_featurizer: VectorFeaturizer[Mol] = field(default_factory=EmptyVectorFeaturizer)
    interaction_featurizer: VectorFeaturizer[list[Mol]] = field(
        default_factory=EmptyVectorFeaturizer
    )
    extra_mol_fdim: int = 0
    extra_interaction_fdim: int = 0
    self_loops: bool = False
    interact_only_mixture: bool = False

    def __post_init__(self):
        self.mol_fdim = len(self.mol_featurizer) + self.extra_mol_fdim
        self.interaction_fdim = len(self.interaction_featurizer) + self.extra_interaction_fdim

    @property
    def shape(self) -> tuple[int, int]:
        return self.mol_fdim, self.interaction_fdim

    def __call__(
        self,
        mols: list[Chem.Mol | list[Chem.Mol]],
        mol_features_extra: np.ndarray | None = None,
        interaction_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        if self.interact_only_mixture:
            nodes = next(item for item in mols if isinstance(item, list))
        else:
            nodes = [mol for item in mols for mol in (item if isinstance(item, list) else [item])]

        V = np.array([self.mol_featurizer(mol) for mol in nodes], dtype=np.single)
        if mol_features_extra is not None:
            V = np.hstack((V, mol_features_extra))

        # Upper triangle indices creates complete graph (all molecules paired).
        # Including the diagonal (`k=0`) adds self pairs.
        i, j = np.triu_indices(len(nodes), k=int(not self.self_loops))
        n_pairs = len(i)
        n_edges = 2 * n_pairs
        edge_index = np.empty((2, n_edges), dtype=int)
        edge_index[0, 0::2], edge_index[1, 0::2] = i, j
        edge_index[0, 1::2], edge_index[1, 1::2] = j, i
        rev_edge_index = np.arange(n_edges).reshape(-1, 2)[:, ::-1].ravel()

        E = np.zeros((n_edges, self.interaction_fdim - self.extra_interaction_fdim))
        for k, (u, v) in enumerate(zip(edge_index[0, 0::2], edge_index[1, 0::2])):
            x_e = (
                self.interaction_featurizer([nodes[u]])
                if u == v
                else self.interaction_featurizer([nodes[u], nodes[v]])
            )
            E[2 * k] = x_e
            E[2 * k + 1] = x_e

        if interaction_features_extra is not None:
            extras = np.repeat(interaction_features_extra, 2, axis=0)
            E = np.hstack((E, extras))

        return MolGraph(V, E, edge_index, rev_edge_index)

    @classmethod
    def with_self_loops_and_hbonds(cls, **kwargs):
        return cls(
            interaction_featurizer=HydrogenBondFeaturizer(),
            self_loops=True,
            **kwargs,
        )
