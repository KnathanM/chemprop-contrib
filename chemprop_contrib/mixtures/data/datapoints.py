from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from chemprop.data.datapoints import MoleculeDatapoint, _DatapointMixin
from chemprop.utils import make_mol
from rdkit import Chem


@dataclass
class _MixtureDatapointMixin:
    mols: list[Chem.Mol]

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
class ComponentDatapoint(_DatapointMixin, _MixtureDatapointMixin):
    V_fs: list[np.ndarray] | None = None
    """A list of optional V_f numpy arrays, one for each molecule in the mixture"""
    E_fs: list[np.ndarray] | None = None
    """A list of optional E_f numpy arrays, one for each molecule in the mixture"""
    V_ds: list[np.ndarray] | None = None
    """A list of optional V_d numpy arrays, one for each molecule in the mixture"""
    G_ds: list[np.ndarray] | None = None
    """A list of numpy arrays of shape ``1 x d_gd``, where ``d_gd`` is the number of additional  
    descriptors that will be concatenated to the learned graph representation after atom-to-molecule 
    aggregation, but before component-to-mixture aggregation"""
    w_fps: list[float] | np.ndarray | float = 1.0
    """The predetermined weights of the molecule's learned fingerprints when averaging in the mixture. If a numpy array, it should be 1D with length equal to the number of molecules in the mixture. If a single float is given, it is applied to all molecules."""

    def __post_init__(self):
        super().__post_init__()
        n = len(self.mols)

        self.w_fps = np.asarray(self.w_fps, dtype=float)
        if self.w_fps.ndim == 0:
            self.w_fps = np.full(n, self.w_fps.item())

        for name in ("w_fps", "V_fs", "E_fs", "V_ds", "G_ds"):
            val = getattr(self, name)
            if val is not None and len(val) != n:
                raise ValueError(f"{name} has length {len(val)}, expected {n}")


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
