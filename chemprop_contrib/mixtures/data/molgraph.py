from __future__ import annotations

from typing import NamedTuple

import numpy as np

from chemprop.data import MolGraph


class MixtureMolGraph(NamedTuple):
    """A combination of the MolGraphs of the molecules in a mixture"""

    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    molecule_sizes: np.ndarray
    """the number of atoms in each molecule in the combined mixture graph"""
    w_fps: np.ndarray
    """the weights of each molecules' fingerprint for molecule-to-mixture aggregation"""

    @classmethod
    def from_molgraph(
        cls, mg: MolGraph, molecule_sizes: np.ndarray, w_fp: np.ndarray
    ) -> MixtureMolGraph:
        return cls(mg.V, mg.E, mg.edge_index, mg.rev_edge_index, molecule_sizes, w_fp)


class InteractionGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    sub_mgs: list[MolGraph | MixtureMolGraph]
    sub_mgs_V_ds: list[np.ndarray | None]

    @classmethod
    def from_molgraph(
        cls,
        mg: MolGraph,
        sub_mgs: list[MolGraph | MixtureMolGraph],
        sub_mgs_V_ds: list[np.ndarray | None],
    ) -> InteractionGraph:
        return cls(mg.V, mg.E, mg.edge_index, mg.rev_edge_index, sub_mgs, sub_mgs_V_ds)
