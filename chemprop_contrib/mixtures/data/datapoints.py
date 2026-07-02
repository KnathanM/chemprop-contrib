from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import Chem

from chemprop.data.datapoints import _DatapointMixin
from chemprop.utils import make_mol


@dataclass
class _MixtureDatapointMixin:
    mols: list[Chem.Mol]
    """the molecules associated with this mixture"""

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
        mols = [make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms) for smi in smis]

        kwargs["name"] = "|".join(smis) if "name" not in kwargs else kwargs["name"]

        return cls(mols, *args, **kwargs)


@dataclass
class MixtureDatapoint(_DatapointMixin, _MixtureDatapointMixin):
    """A :class:`MixtureDatapoint` represents a mixture of molecules, i.e. where the order of input
    molecules does not matter and the input can have any number of molecules."""

    V_f: np.ndarray | None = None
    """A numpy array of shape ``V x d_vf``, where ``V`` is the sum of the number of atoms in each
    molecule in the mixture, and ``d_vf`` is the number of additional features that will be
    concatenated to atom-level features *before* message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the sum of the number of bonds in each
    molecule in the mixture, and ``d_ef`` is the number of additional features  containing
    additional features that will be concatenated to bond-level features *before* message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the sum of the number of atoms in each
    molecule in the mixture, and ``d_vd`` is the number of additional descriptors that will be
    concatenated to atom-level descriptors *after* message passing"""
    w_fps: list[float] | np.ndarray | float = 1.0
    """The predetermined weights of each molecule's learned fingerprints when aggregating them into
    a mixture representation. If a numpy array, it should be 1D with length equal to the number of
    molecules in the mixture. If a single float, it is applied to all molecules."""

    def __post_init__(self):
        # Combine the molecules into a single Chem.Mol to give to the MolGraph featurizer.
        mixture_mol = Chem.RWMol()
        for m in self.mols:
            mixture_mol.InsertMol(m)
        self.mol = mixture_mol.GetMol()
        self.mol.SetProp("molecule_sizes", ",".join(str(m.GetNumAtoms()) for m in self.mols))

        self.w_fps = np.asarray(self.w_fps, dtype=float)
        if self.w_fps.ndim == 0:
            self.w_fps = np.full(len(self.mols), self.w_fps.item())
        elif len(self.w_fps) != len(self.mols):
            raise ValueError(f"w_fps has length {len(self.w_fps)}, expected {len(self.mols)}")

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


@dataclass
class InteractionDatapoint(_DatapointMixin):
    """An :class:`InteractionDatapoint` stores extra molecule (node) and interaction (edge)
    features/descriptors for an interaction graph between molecules in a datapoint. While it also
    holds training target information, it is not intended to be used alone. When combined with
    and mixture datapoints in an :class:`InteractionDataset`, the featurizer of that dataset will
    determine the graph topology including which molecules are included in the graph.

    This class can also be used to supply precomputed extra molecule descriptors to the molecules in
    the mixture before molecule-to-mixture aggregation (without the need to use interaction message
    passing).
    """

    V_f: np.ndarray | None = None
    """a numpy array of shape ``V x d_vf``, where ``V`` is the number of molecules in the mixture,
    and ``d_vf`` is the number of additional features that will be concatenated to molecule-level
    features *before* interaction message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the number of interactions in the
    mixture, and ``d_ef`` is the number of additional features containing additional features that
    will be concatenated to interaction-level features *before* interaction message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the number of molecules in the mixture,
    and ``d_vd`` is the number of additional descriptors that will be concatenated to molecule-level
    descriptors *after* interaction message passing"""

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
