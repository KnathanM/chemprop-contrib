from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import ArrayLike
from chemprop.data.datasets import Datum, MoleculeDataset, MulticomponentDataset, ReactionDataset, _MolGraphDatasetMixin
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import Featurizer
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
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
    mgs: list[ComponentMolGraph]
    V_ds: list[np.ndarray] | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


@dataclass
class ComponentDataset(_MolGraphDatasetMixin, Dataset[ComponentDatum]):
    data: list[ComponentDatapoint]
    featurizer: Featurizer[list[Chem.Mol], list[ComponentMolGraph]] = field(
        default_factory=ComponentMolGraphFeaturizer
    )
    n_workers: int = 0

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data cannot be None!")

        self.reset()
        self.cache = False

    def __getitem__(self, idx: int) -> ComponentDatum:
        d = self.data[idx]
        mgs = self.mg_cache[idx]
        return ComponentDatum(
            mgs,
            self.V_dss[idx],
            self.X_d[idx],
            self.Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )
    
    @property
    def cache(self) -> bool:
        return self.__cache

    @cache.setter
    def cache(self, cache: bool = False):
        self.__cache = cache
        self._init_cache()

    def _init_cache(self):
        """initialize the cache"""
        if self.cache:
            self.mg_cache = ComponentMolGraphCache(
                self.molss, self.V_fss, self.E_fss, self.G_dss, self.w_fpss, self.featurizer, n_workers=self.n_workers
            )
        else:
            self.mg_cache = ComponentMolGraphCacheOnTheFly(self.molss, self.V_fss, self.E_fss, self.G_dss, self.w_fpss, self.featurizer)

    @property
    def smiless(self) -> list[list[str]]:
        """the SMILES strings associated with the dataset"""
        return [[Chem.MolToSmiles(mol) for mol in d.mols] for d in self.data]

    @property
    def molss(self) -> list[list[Chem.Mol]]:
        """the molecules associated with the dataset"""
        return [d.mols for d in self.data]

    @property
    def _V_fss(self) -> list[list[np.ndarray] | None]:
        """the raw atom features of the dataset"""
        return [d.V_fs for d in self.data]

    @property
    def V_fss(self) -> list[list[np.ndarray] | None]:
        """the (scaled) atom descriptors of the dataset"""
        return self.__V_fss

    @V_fss.setter
    def V_fss(self, V_fss: list[list[np.ndarray] | None]):
        """the (scaled) atom features of the dataset"""
        self.__V_fss = V_fss
        self._init_cache()

    @property
    def _E_fss(self) -> list[list[np.ndarray] | None]:
        """the raw bond features of the dataset"""
        return [d.E_fs for d in self.data]

    @property
    def E_fss(self) -> list[list[np.ndarray] | None]:
        """the (scaled) bond features of the dataset"""
        return self.__E_fss

    @E_fss.setter
    def E_fs(self, E_fss: list[list[np.ndarray] | None]):
        self.__E_fss = E_fss
        self._init_cache()

    @property
    def _V_dss(self) -> list[list[np.ndarray] | None]:
        """the raw atom descriptors of the dataset"""
        return [d.V_ds for d in self.data]

    @property
    def V_dss(self) -> list[list[np.ndarray] | None]:
        """the (scaled) atom descriptors of the dataset"""
        return self.__V_dss

    @V_dss.setter
    def V_dss(self, V_dss: list[list[np.ndarray] | None]):
        self.__V_dss = V_dss

    @cached_property
    def _G_dss(self) -> list[list[np.ndarray] | None]:
        """the raw extra molecule descriptors of the dataset"""
        return [np.array([d.G_d for d in self.data])]

    @property
    def G_dss(self) -> list[list[np.ndarray] | None]:
        """the (scaled) extra molecule descriptors of the dataset"""
        return self.__G_dss

    @G_dss.setter
    def G_dss(self, G_dss: list[list[np.ndarray] | None]):
        self.__G_dss = np.array(G_dss)

    @property
    def d_vf(self) -> int:
        """the extra atom feature dimension, if any"""
        return 0 if self.V_fss[0][0] is None else self.V_fss[0][0].shape[1]

    @property
    def d_ef(self) -> int:
        """the extra bond feature dimension, if any"""
        return 0 if self.E_fss[0][0] is None else self.E_fss[0][0].shape[1]

    @property
    def d_vd(self) -> int:
        """the extra atom descriptor dimension, if any"""
        return 0 if self.V_dss[0][0] is None else self.V_dss[0][0].shape[1]
    
    @property
    def d_gd(self) -> int:
        """the extra molecule descriptor dimension, if any"""
        return 0 if self.G_dss[0][0] is None else self.G_dss[0][0].shape[1]

    def normalize_inputs(
        self, key: str = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_d", "V_f", "E_f", "V_d", "G_d"}

        match key:
            case "X_d":
                X = None if self.d_xd == 0 else self._X_d
            case "V_f":
                X = None if self.d_vf == 0 else np.concatenate([V_f for V_fs in self._V_fss for V_f in V_fs], axis=0)
            case "E_f":
                X = None if self.d_ef == 0 else np.concatenate([E_f for E_fs in self._E_fss for E_f in E_fs], axis=0)
            case "V_d":
                X = None if self.d_vd == 0 else np.concatenate([V_d for V_ds in self._V_dss for V_d in V_ds], axis=0)
            case "G_d":
                X = None if self.d_gd == 0 else np.concatenate([G_d for G_ds in self._G_dss for G_d in G_ds], axis=0)
            case _:
                raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)

        match key:
            case "X_d":
                self.X_d = scaler.transform(X)
            case "V_f":
                self.V_fs = [[scaler.transform(V_f) if V_f.size > 0 else V_f for V_f in V_fs] for V_fs in self._V_fss]
            case "E_f":
                self.E_fs = [[scaler.transform(E_f) if E_f.size > 0 else E_f for E_f in E_fs] for E_fs in self._E_fss]
            case "V_d":
                self.V_ds = [[scaler.transform(V_d) if V_d.size > 0 else V_d for V_d in V_ds] for V_ds in self._V_dss]
            case "G_d":
                self.G_ds = [[scaler.transform(G_d) if G_d.size > 0 else G_d for G_d in G_ds] for G_ds in self._G_dss]
            case _:
                raise RuntimeError("unreachable code reached!")

        return scaler

    @property
    def w_fpss(self) -> list[np.ndarray]:
        return [d.w_fps for d in self.data]

    def reset(self):
        super().reset()
        self.__V_fss = self._V_fss
        self.__E_fss = self._E_fss
        self.__V_dss = self._V_dss
        self.__G_dss = self._G_dss


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
class MixtureDataset(_MolGraphDatasetMixin, Dataset):
    datasets: list[MoleculeDataset | ReactionDataset | ComponentDataset | MixtureGraphDataset]

    def __post_init__(self):
        super().__post_init__()
        if any(isinstance(dataset, MixtureGraphDataset) for dataset in self.datasets[:-1]):
            raise ValueError("The MixtureGraphDataset must be the final entry of arg: `datasets`")

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> list[Datum | ComponentDatum | MixtureDatum]:
        return [dset[idx] for dset in self.datasets]

