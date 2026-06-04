from typing import NamedTuple

import numpy as np


# See also chemprop.data.molgraph.MolGraph
class ComponentMolGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    G: np.ndarray = np.empty((0,))
    """graph level features to append after atom-to-molecule aggregation but before component
    message passing or mixture aggregation"""
    w_fp: float = 1.0
    """the weight of the component's fingerprint when combining components in the mixture"""


# See also chemprop.data.molgraph.MolGraph
class MixtureGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
