from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from chemprop.data.datasets import Datum, MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import Featurizer
from rdkit import Chem
from torch.utils.data import Dataset

from chemprop_contrib.mixtures.data.datapoints import ComponentDatapoint, MixtureDatapoint
from chemprop_contrib.mixtures.data.molgraph import ComponentMolGraph, MixtureGraph
from chemprop_contrib.mixtures.featurizers import (
    ComponentMolGraphCache,
    ComponentMolGraphCacheOnTheFly,
    ComponentMolGraphFeaturizer,
    SimpleMixtureGraphFeaturizer,
)


# See also chemprop.data.datasets.Datum
class ComponentDatum(NamedTuple):
    mg: ComponentMolGraph | None
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


@dataclass
class ComponentDataset(MoleculeDataset, Dataset[ComponentMolGraph]):
    data: list[ComponentDatapoint]
    featurizer: Featurizer[Chem.Mol | None, ComponentMolGraph | None] = field(
        default_factory=ComponentMolGraphFeaturizer
    )

    def __getitem__(self, idx: int) -> ComponentDatum:
        d = self.data[idx]
        mg = self.mg_cache[idx]
        return ComponentDatum(
            mg,
            self.V_ds[idx],
            self.X_d[idx],
            self.Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    def _init_cache(self):
        """initialize the cache"""
        if self.cache:
            self.mg_cache = ComponentMolGraphCache(
                self.mols, self.V_fs, self.E_fs, self.G_d, self.w_fps, self.featurizer, n_workers=self.n_workers
            )
        else:
            self.mg_cache = ComponentMolGraphCacheOnTheFly(self.mols, self.V_fs, self.E_fs, self.G_d, self.w_fps, self.featurizer)

    @cached_property
    def _G_d(self) -> np.ndarray:
        """the raw extra component descriptors of the dataset"""
        return np.array([d.G_d for d in self.data])

    @property
    def G_d(self) -> np.ndarray:
        """the (scaled) extra component descriptors of the dataset"""
        return self.__G_d

    @G_d.setter
    def G_d(self, G_d: ArrayLike):
        self._validate_attribute(G_d, "extra component descriptors")

        self.__G_d = np.array(G_d)

    @property
    def w_fps(self) -> np.ndarray:
        return np.array([d.w_fp for d in self.data])
    
    def reset(self):
        """ Reset the extra component descriptors of each datapoint to their initial, unnormalized
        values"""
        super().reset()
        self.__G_d = self._G_d


# See also chemprop.data.datasets.Datum
class MixtureDatum(NamedTuple):
    mg: MixtureGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


@dataclass
class MixtureGraphDataset(MoleculeDataset, Dataset[MixtureGraph]):
    data: list[MixtureDatapoint]
    featurizer: Featurizer[list[Chem.Mol], MixtureGraph] = field(
        default_factory=SimpleMixtureGraphFeaturizer
    )

    def __getitem__(self, idx: int) -> MixtureDatum:
        d = self.data[idx]
        mg = self.mg_cache[idx]
        return MixtureDatum(
            mg,
            self.V_ds[idx],
            self.X_d[idx],
            self.Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    @property
    def smiles(self) -> list[list[str]]:
        """the SMILES strings associated with the dataset"""
        return [[Chem.MolToSmiles(mol) for mol in d.mols] for d in self.data]

    @property
    def mols(self) -> list[list[Chem.Mol]]:
        """the molecules associated with the dataset"""
        return [[mol for mol in d.mols] for d in self.data]


@dataclass(repr=False, eq=False)
class MixtureDataset(MulticomponentDataset):
    datasets: list[MoleculeDataset | ReactionDataset | ComponentDataset | MixtureGraphDataset]

    def __post_init__(self):
        super().__post_init__()
        if any(isinstance(dataset, MixtureGraphDataset) for dataset in self.datasets[:-1]):
            raise ValueError("The MixtureGraphDataset must be the final entry of arg: `datasets`")

    def __getitem__(self, idx: int) -> list[Datum | ComponentDatum | MixtureDatum]:
        return [dset[idx] for dset in self.datasets]
