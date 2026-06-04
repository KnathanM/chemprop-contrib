from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from chemprop.data.collate import BatchMolGraph, collate_batch
from chemprop.data.datasets import Datum
from torch import Tensor

from chemprop_contrib.mixtures.data.datasets import ComponentDatum, MixtureDatum
from chemprop_contrib.mixtures.data.molgraph import ComponentMolGraph, MixtureGraph


# See also chemprop.data.collate.BatchMolGraph
@dataclass(repr=False, eq=False, slots=True)
class BatchComponentMolGraph(BatchMolGraph):
    mgs: InitVar[Sequence[ComponentMolGraph | None]]
    G: Tensor = field(init=False)
    w_fps: Tensor = field(init=False)

    def __post_init__(self, mgs):
        self._BatchMolGraph__size = len(mgs)

        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        Gs = []
        w_fps = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            if mg is None:
                continue
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            Gs.append(mg.G)
            w_fps.append(mg.w_fp)

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()
        self.G = torch.from_numpy(np.vstack(Gs)).float()
        self.w_fps = torch.from_numpy(np.array(w_fps)).float()

    def to(self, device: str | torch.device):
        super(BatchComponentMolGraph, self).to(device)
        self.G = self.G.to(device)
        self.w_fps = self.w_fps.to(device)


# See also chemprop.data.collate.TrainingBatch
class BatchComponentDatum(NamedTuple):
    bmg: BatchComponentMolGraph | None
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


# See also chemprop.data.collate.collate_batch
def collate_component(batch: Iterable[ComponentDatum]) -> BatchComponentDatum:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return BatchComponentDatum(
        BatchComponentMolGraph(mgs) if any(mg is not None for mg in mgs) else None,
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


class BatchMixtureGraph(BatchMolGraph):
    """A :class:`BatchMixtureGraph` represents a batch of individual :class:`MixtureGraph`\s.

    It has all the attributes of a ``BatchMolGraph``.
    """

    mgs: InitVar[Sequence[MixtureGraph]]  # just updating the type hint


# See also chemprop.data.collate.TrainingBatch
class BatchMixtureDatum(NamedTuple):
    bmg: BatchMixtureGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_mixturegraph(batch: Iterable[MixtureDatum]) -> BatchMixtureDatum:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return BatchMixtureDatum(
        BatchMixtureGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


# See also chemprop.data.collate.MulticomponentTrainingBatch
class MixtureBatch(NamedTuple):
    bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


# See also chemprop.data.collate.collate_multicomponent
def collate_mixture(
    batches: Iterable[Iterable[Datum | ComponentDatum | MixtureDatum]],
) -> MixtureBatch:
    tbs = []
    for batch in zip(*batches):
        if isinstance(batch[0], Datum):
            tbs.append(collate_batch(batch))
        elif isinstance(batch[0], ComponentDatum):
            tbs.append(collate_component(batch))
        elif isinstance(batch[0], MixtureDatum):
            tbs.append(collate_mixturegraph(batch))

    return MixtureBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )


class BatchNodesOnly(NamedTuple):
    """A reduced version of :class:`BatchMolGraph` that only has nodes and a
    mapping from nodes to subgraphs.
    """

    V: Tensor
    """the node feature matrix"""
    batch: Tensor
    """the index of the parent graph in the batched graph"""
