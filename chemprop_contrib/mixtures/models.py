import copy
import logging
from typing import Iterable

import torch
from torch import Tensor

from chemprop.data.collate import BatchMolGraph
from chemprop.models import MulticomponentMPNN
from chemprop.nn.agg import Aggregation
from chemprop.nn.message_passing import MessagePassing, MulticomponentMessagePassing
from chemprop.nn.metrics import ChempropMetric
from chemprop.nn.predictors import Predictor
from chemprop.nn.transforms import ScaleTransform

from chemprop_contrib.mixtures.data.collate import (
    BatchInteractionGraph,
    BatchMixtureMolGraph,
    InteractionTrainingBatch,
)
from chemprop_contrib.mixtures.nn.agg import MixtureAggregation


logger = logging.getLogger(__name__)


class MixtureMPNN(MulticomponentMPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        mixture_agg: MixtureAggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
        )
        self.hparams["mixture_agg"] = mixture_agg.hparams
        self.mixture_agg = mixture_agg

    def fingerprint(
        self,
        bmgs: Iterable[BatchMolGraph | BatchMixtureMolGraph],
        V_ds: Iterable[Tensor | None],
        X_d: Tensor | None = None,
    ) -> Tensor:
        H_vs = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]
        Hs = [
            self.mixture_agg(H, bmg.batch_mixture, bmg.w_fps)
            if isinstance(bmg, BatchMixtureMolGraph)
            else H
            for H, bmg in zip(Hs, bmgs)
        ]
        H = torch.cat(Hs, 1)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    @classmethod
    def _load(cls, path, map_location, **submodules):
        submodules, state_dict, hparams = super()._load(path, map_location, **submodules)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("mixture_agg",)
            if key not in submodules
        }
        return submodules, state_dict, hparams


class InteractionMPNN(MulticomponentMPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        interaction_mp: MessagePassing,
        mixture_agg: MixtureAggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
        interact_only_mixture: bool = False,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
        )
        self.hparams["interaction_mp"] = interaction_mp.hparams
        self.hparams["mixture_agg"] = mixture_agg.hparams
        self.hparams["interact_only_mixture"] = interact_only_mixture

        self.interaction_mp = interaction_mp
        self.mixture_agg = mixture_agg
        self.interact_only_mixture = interact_only_mixture

    def fingerprint(
        self,
        big: BatchInteractionGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> Tensor:
        bmgs = big.sub_bmgs
        V_ds = big.sub_bmgs_V_ds
        H_vs = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]

        # Copy to not mutate input. Shallow copy to not copy tensors.
        big_copy = copy.copy(big)
        if self.interact_only_mixture:
            mixture_idx = next(
                i for i, bmg in enumerate(bmgs) if isinstance(bmg, BatchMixtureMolGraph)
            )
            V = Hs[mixture_idx]
        else:
            big_copy, V_d = self._reorder_big_V(big_copy, V_d)
            V = torch.cat(Hs)

        big_copy.V = torch.concat((V, big.V), dim=1)
        V = self.interaction_mp(big_copy, V_d)

        if self.interact_only_mixture:
            Hs[mixture_idx] = V
        else:
            sizes = [
                len(bmg.batch_mixture if isinstance(bmg, BatchMixtureMolGraph) else bmg)
                for bmg in bmgs
            ]
            Hs = list(torch.split(V, sizes))

        Hs = [
            self.mixture_agg(H, bmg.batch_mixture, bmg.w_fps)
            if isinstance(bmg, BatchMixtureMolGraph)
            else H
            for H, bmg in zip(Hs, bmgs)
        ]
        H = torch.cat(Hs, 1)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    @staticmethod
    def _reorder_big_V(big: BatchInteractionGraph, V_d: Tensor | None) -> tuple[BatchInteractionGraph, Tensor | None]:
        """Interaction graph nodes are ordered by sub_bmg and then batched together by datapoint.
        E.g. big.V = [node_from_MolGraph1,
                      nodes_from_MixtureMolGraph1,
                      node_from_MolGraph2,
                      node_from_MixtureMolGraph2]
        But the sub_bmgs are batched by datapoint and then concatenated after MolGraph aggregation.
        E.g. H = [node_from_MolGraph1,
                  node_from_MolGraph2,
                  nodes_from_MixtureMolGraph1,
                  node_from_MixtureMolGraph2]
        This code adjusts big to match the order in H. To do so, it computes three things:
        1. The index in V where each datapoint starts. `i_dp_start_in_V`
        2. The index in each datapoint where each graph starts. `i_sub_bmg_starts_per_dp_per_sub_bmg`
        3. The index in each graph where each molecule starts. `mol_i_in_mg`
        """

        def cumsum_exclude_current(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            return torch.cumsum(tensor, dim=dim) - tensor

        B = len(big)

        # Get a count of how many molecules are in each batched graph for each datapoint.
        # One for normal BatchMolGraph's, but more for mixtures.
        n_mol_per_dp_per_sub_bmg = torch.zeros(B, len(big.sub_bmgs), dtype=torch.long)
        for sub_bmg_idx, sub_bmg in enumerate(big.sub_bmgs):
            if isinstance(sub_bmg, BatchMixtureMolGraph):
                n_mol_per_dp_per_sub_bmg[:, sub_bmg_idx] = torch.bincount(
                    sub_bmg.batch_mixture, minlength=B
                )
            else:  # BatchMolGraph
                n_mol_per_dp_per_sub_bmg[:, sub_bmg_idx] = 1

        total_mol_per_dp = n_mol_per_dp_per_sub_bmg.sum(dim=1)
        i_dp_start_in_V = cumsum_exclude_current(total_mol_per_dp, dim=0)

        i_sub_bmg_starts_per_dp_per_sub_bmg = cumsum_exclude_current(
            n_mol_per_dp_per_sub_bmg, dim=1
        )

        sub_bmg_i_to_V_i_per_sub_bmg = []
        for sub_bmg_idx, sub_bmg in enumerate(big.sub_bmgs):
            if isinstance(sub_bmg, BatchMixtureMolGraph):
                i_mol_to_i_dp = sub_bmg.batch_mixture
                n_mol_in_mg_per_dp = n_mol_per_dp_per_sub_bmg[:, sub_bmg_idx]
                i_dp_starts_in_bmg = cumsum_exclude_current(n_mol_in_mg_per_dp, dim=0)
                # The line below is a faster version of
                # torch.concat([torch.arange(n) for n in n_mol_in_mg_per_dp])
                mol_i_in_mg = (
                    torch.arange(i_mol_to_i_dp.numel()) - i_dp_starts_in_bmg[i_mol_to_i_dp]
                )
            else:
                i_mol_to_i_dp = torch.arange(B)
                mol_i_in_mg = torch.zeros(B, dtype=torch.long)

            sub_bmg_i_to_V_i = (
                i_dp_start_in_V[i_mol_to_i_dp]
                + i_sub_bmg_starts_per_dp_per_sub_bmg[i_mol_to_i_dp, sub_bmg_idx]
                + mol_i_in_mg
            )
            sub_bmg_i_to_V_i_per_sub_bmg.append(sub_bmg_i_to_V_i)

        H_idx_to_V_idx = torch.cat(sub_bmg_i_to_V_i_per_sub_bmg)
        V_idx_to_H_idx = torch.argsort(H_idx_to_V_idx)

        # We want to order V like H, so for each row in H with index i, pick the row in V at index
        # H_idx_to_V_idx[i]
        big.V = big.V[H_idx_to_V_idx]
        V_d = V_d[H_idx_to_V_idx] if V_d is not None else None
        # When the index map is used as *values* instead of indices, the map inverts
        V_idx_to_H_idx = V_idx_to_H_idx.to(big.edge_index.device)
        big.edge_index = V_idx_to_H_idx[big.edge_index]
        return big, V_d

    def get_batch_size(self, batch: InteractionTrainingBatch) -> int:
        return len(batch[0])

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        self.interaction_mp.V_d_transform.train()
        self.interaction_mp.graph_transform.train()

    @classmethod
    def _load(cls, path, map_location, **submodules):
        submodules, state_dict, hparams = super()._load(path, map_location, **submodules)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("interaction_mp", "mixture_agg")
            if key not in submodules
        }
        return submodules, state_dict, hparams
