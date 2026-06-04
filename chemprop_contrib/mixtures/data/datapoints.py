from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from chemprop.data.datapoints import MoleculeDatapoint, _DatapointMixin
from chemprop.utils import make_mol
from rdkit import Chem


@dataclass
class ComponentDatapoint(MoleculeDatapoint):
    G_d : np.ndarray | None = None
    """A numpy array of shape ``1 x d_gd``, where ``d_gd`` is the number of additional descriptors 
    that will be concatenated to the learned graph representation after atom-to-molecule 
    aggregation, but before component-to-mixture aggregation"""
    w_fp: float = 1.0
    """the weight of the molecule's learned fingerprint when averaging in the mixture"""


@dataclass
class _MixtureDatapointMixin:
    mols: list[Chem.Mol]
    """the mixture associated with this datapoint"""

    @classmethod
    def from_smis(
        cls,
        smis: Sequence[str],
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        **kwargs,
    ) -> _MixtureDatapointMixin:
        mols = [make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms) for smi in smis if smi]

        kwargs["name"] = "|".join(smis) if "name" not in kwargs else kwargs["name"]

        return cls(mols, *args, **kwargs)


@dataclass
class MixtureDatapoint(_DatapointMixin, _MixtureDatapointMixin):
    V_f: np.ndarray | None = None
    """a numpy array of shape ``V x d_vf``, where ``V`` is the number of molecules in the mixture, and
    ``d_vf`` is the number of additional features that will be concatenated to molecule-level features
    *before* message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the number of interactions in the mixture, and
    ``d_ef`` is the number of additional features containing additional features that will be
    concatenated to interaction-level features *before* message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the number of molecules in the mixture, and
    ``d_vd`` is the number of additional descriptors that will be concatenated to molecule-level
    descriptors *after* message passing"""

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

        super().__post_init__()

    def __len__(self) -> int:
        return 1
