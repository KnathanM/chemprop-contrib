from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.collate import BatchMolGraph, collate_batch
from chemprop.data.datasets import Datum
from chemprop_contrib.mixtures.data.datasets import InteractionDatum, MixtureDatum
from chemprop_contrib.mixtures.data.molgraph import InteractionGraph, MixtureMolGraph


@dataclass(repr=False, eq=False)
class BatchMixtureMolGraph(BatchMolGraph):
    mgs: InitVar[Sequence[MixtureMolGraph]]
    w_fps: Tensor = field(init=False)
    """the predetermined weights of each molecule's learned fingerpint in the batch"""
    batch_mixture: Tensor = field(init=False)
    """the index of the parent mixture for each molecule in the batched mixture graph"""

    def __post_init__(self, mgs: Sequence[MixtureMolGraph]):
        super().__post_init__(mgs)

        w_fps, batch, batch_mixture = [], [], []
        total_mols = 0
        for i, mg in enumerate(mgs):
            n_mols = len(mg.molecule_sizes)
            w_fps.append(mg.w_fps)
            batch.append(np.repeat(np.arange(n_mols) + total_mols, mg.molecule_sizes))
            batch_mixture.append(np.full(n_mols, i))
            total_mols += n_mols

        self.w_fps = torch.from_numpy(np.concatenate(w_fps)).float()
        self.batch = torch.from_numpy(np.concatenate(batch)).long()
        self.batch_mixture = torch.from_numpy(np.concatenate(batch_mixture)).long()

    def to(self, device: str | torch.device):
        super().to(device)
        self.w_fps = self.w_fps.to(device)
        self.batch_mixture = self.batch_mixture.to(device)


class MixtureTrainingBatch(NamedTuple):
    bmg: BatchMixtureMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_mixture_batch(batch: Iterable[MixtureDatum]) -> MixtureTrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return MixtureTrainingBatch(
        BatchMixtureMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


class MixtureMulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph | BatchMixtureMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_multicomponent_with_mixture(
    batches: Iterable[Iterable[Datum | MixtureDatum]],
) -> MixtureMulticomponentTrainingBatch:
    tbs = []
    for batch in zip(*batches):
        if isinstance(batch[0], Datum):
            tbs.append(collate_batch(batch))
        elif isinstance(batch[0], MixtureDatum):
            tbs.append(collate_mixture_batch(batch))

    return MixtureMulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )


@dataclass(repr=False, eq=False)
class BatchInteractionGraph(BatchMolGraph):
    mgs: InitVar[Sequence[InteractionGraph]]
    sub_bmgs: list[BatchMolGraph | BatchMixtureMolGraph] = field(init=False)
    """the batched subgraphs that get aggregated into the InteractionGraph node embeddings"""
    sub_bmgs_V_ds: list[Tensor | None] = field(init=False)
    """the optional batched V_ds for each subgraph"""

    def __post_init__(self, mgs: Sequence[InteractionGraph]):
        super().__post_init__(mgs)

        self.sub_bmgs = []
        for subs in zip(*(ig.sub_mgs for ig in mgs)):
            if isinstance(subs[0], MixtureMolGraph):
                self.sub_bmgs.append(BatchMixtureMolGraph(subs))
            else:
                self.sub_bmgs.append(BatchMolGraph(subs))

        self.sub_bmgs_V_ds = []
        for V_ds in zip(*(ig.sub_mgs_V_ds for ig in mgs)):
            self.sub_bmgs_V_ds.append(
                None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float()
            )

    def to(self, device: str | torch.device):
        super().to(device)
        for sub_bmg in self.sub_bmgs:
            sub_bmg.to(device)
        self.sub_bmgs_V_ds = [
            V_ds.to(device) if V_ds is not None else None for V_ds in self.sub_bmgs_V_ds
        ]


class InteractionTrainingBatch(NamedTuple):
    big: BatchInteractionGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_interaction_batch(
    batch: Iterable[InteractionDatum],
) -> InteractionTrainingBatch:
    igs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return InteractionTrainingBatch(
        BatchInteractionGraph(igs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )
