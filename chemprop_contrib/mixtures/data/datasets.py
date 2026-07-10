from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from chemprop.data.datasets import MoleculeDataset
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import Featurizer

from chemprop_contrib.mixtures.data.datapoints import (
    InteractionDatapoint,
    MixtureDatapoint,
)
from chemprop_contrib.mixtures.data.molgraph import InteractionGraph, MixtureMolGraph
from chemprop_contrib.mixtures.featurizers import CompleteInteractionGraphFeaturizer


class MixtureDatum(NamedTuple):
    mg: MixtureMolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


@dataclass
class MixtureDataset(MoleculeDataset):
    data: list[MixtureDatapoint]

    def __getitem__(self, idx: int) -> MixtureMolGraph:
        d = self.data[idx]
        mg = self.mg_cache[idx]
        molecule_sizes = self.get_molecule_sizes(idx)
        mg = MixtureMolGraph.from_molgraph(mg, molecule_sizes, d.w_fps)
        return MixtureDatum(
            mg,
            self.V_ds[idx],
            self.X_d[idx],
            self.Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    def get_molecule_sizes(self, idx: int) -> np.ndarray:
        d = self.data[idx]
        return np.fromstring(d.mol.GetProp("molecule_sizes"), sep=",", dtype=int)

    @property
    def molss(self) -> list[list[Chem.Mol]]:
        # Used when making an InteractionGraph.
        return [d.mols for d in self.data]


class InteractionDatum(NamedTuple):
    ig: InteractionGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


@dataclass(repr=False, eq=False)
class InteractionDataset(MoleculeDataset, Dataset[InteractionDatum]):
    data: list[InteractionDatapoint]
    featurizer: Featurizer[list[Chem.Mol | list[Chem.Mol]], MolGraph] = field(
        default_factory=CompleteInteractionGraphFeaturizer.with_self_loops_and_hbonds
    )
    n_workers: int = 0
    subgraph_datasets: list[MoleculeDataset | MixtureDataset] | None = None

    def __post_init__(self):
        if self.subgraph_datasets is None:
            raise ValueError("When using `InteractionDataset`, you must supply subgraph_datasets!")

        sizes = [len(dset) for dset in self.subgraph_datasets] + [len(self.data)]
        if len(set(sizes)) != 1:
            raise ValueError(f"Datasets must have all same length! got: {sizes}")

        super().__post_init__()

    def __getitem__(self, idx: int) -> InteractionDatum:
        d = self.data[idx]
        mg = self.mg_cache[idx]
        sub_mgs = []
        for dset in self.subgraph_datasets:
            if isinstance(dset, MixtureDataset):
                molecule_sizes = dset.get_molecule_sizes(idx)
                sub_mgs.append(
                    MixtureMolGraph.from_molgraph(
                        dset.mg_cache[idx], molecule_sizes, dset.data[idx].w_fps
                    )
                )
            else:
                sub_mgs.append(dset.mg_cache[idx])
        sub_mgs_V_ds = [dset.V_ds[idx] for dset in self.subgraph_datasets]
        ig = InteractionGraph.from_molgraph(mg, sub_mgs, sub_mgs_V_ds)
        return InteractionDatum(
            ig,
            self.V_ds[idx],
            self.X_d[idx],
            self.Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    @property
    def smiles(self) -> list[tuple[str, ...]]:
        return list(zip(*[dset.smiles for dset in self.subgraph_datasets]))

    @property
    def mols(self) -> list[list[Chem.Mol | list[Chem.Mol]]]:
        # self.mols is passed to the featurizer to populate self.mg_cache
        mols = []
        for dset in self.subgraph_datasets:
            if isinstance(dset, MixtureDataset):
                mols.append(dset.molss)
            else:
                mols.append(dset.mols)
        return list(zip(*mols))

    def normalize_inputs_subgraph_datasets(
        self, key: str = "X_d", scaler: list[StandardScaler] | None = None
    ) -> list[StandardScaler]:
        match scaler:
            case None:
                return [dset.normalize_inputs(key) for dset in self.subgraph_datasets]
            case _:
                assert len(scaler) == len(
                    self.subgraph_datasets
                ), "Number of scalers must match number of datasets!"

                return [
                    dset.normalize_inputs(key, s) for dset, s in zip(self.subgraph_datasets, scaler)
                ]

    def reset(self):
        super().reset()
        for dset in self.subgraph_datasets:
            dset.reset()
