from dataclasses import dataclass, field

import numpy as np

from chemprop.data.datapoints import _DatapointMixin

from chemprop_contrib.mixtures.data.datapoints import _V_f_E_f_V_d


@dataclass
class _IDMoleculeDatapointMixin:
    mol_id: int


@dataclass
class IDMoleculeDatapoint(_V_f_E_f_V_d, _DatapointMixin, _IDMoleculeDatapointMixin):
    """A single-molecule datapoint that references its molecule by integer ID. Instead of storing a
    :class:`Chem.Mol`, this datapoint holds a ``mol_id`` used to look up the precomputed graph in a
    :class:`MolGraphStore`.
    """

    def __len__(self) -> int:
        return 1


@dataclass
class _IDMixtureDatapointMixin:
    mol_ids: list[int] = field(default_factory=list)


@dataclass
class IDMixtureDatapoint(_V_f_E_f_V_d, _DatapointMixin, _IDMixtureDatapointMixin):
    """A mixture datapoint that references its component molecules by integer IDs. Instead of
    storing :class:`Chem.Mol` objects, this datapoint holds a list of ``mol_id`` used to look up the
    precomputed graph in a :class:`MolGraphStore`.
    """

    w_fps: list[float] | np.ndarray | float = 1.0

    def __post_init__(self):
        self.w_fps = np.asarray(self.w_fps, dtype=float)
        if self.w_fps.ndim == 0:
            self.w_fps = np.full(len(self.mol_ids), self.w_fps.item())
        elif len(self.w_fps) != len(self.mol_ids):
            raise ValueError(f"w_fps has length {len(self.w_fps)}, expected {len(self.mol_ids)}")
        super().__post_init__()

    def __len__(self) -> int:
        return 1
