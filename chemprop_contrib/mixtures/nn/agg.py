from abc import abstractmethod
import copy
from typing import Sequence

import torch
from chemprop.data import BatchMolGraph
from chemprop.nn.agg import Aggregation
from chemprop.nn.hparams import HasHParams
from torch import Tensor, nn

from chemprop_contrib.mixtures.data import BatchComponentMolGraph, BatchMixtureGraph, BatchNodesOnly
from chemprop_contrib.mixtures.nn.message_passing import (
    InteractionMessagePassing,
    MixtureMessagePassing,
    MolecularMessagePassing,
)


class MixtureAggregation(nn.Module, HasHParams):
    """A `MixtureAggregation` first aggregates node embeddings into a graph embedding for each
    molecule in a datapoint. Then it optionally performs message passing between the molecule
    embeddings. Finally it aggregates the molecule embeddings into a single datapoint embedding. 

    Parameters
    ----------
    graph_agg: Aggregation
        an instance of a chemprop Aggregation for the node to graph aggregation
    groups: Sequence[Sequence[int]]
        the indices of the molecules/components split into groups, e.g. [[0],[1,2]] for solute in
        binary solvent
    fp_dims: Sequence[int]
        the dimensions of the final molecule embeddings, used in some mixture aggregation schemes
        and for padding missing components with zeros. If `mixmp` is not None, it is the output
        dimensions of that module. Otherwise it is the output dimensions of the 
        `MixtureMulticomponentMessagePassing` module plus the length of any molecule features that
        get concatenated to the learned representation, either via a mol_featurizer in 
        `ComponentMolGraphFeaturizer` or extra mol features given via the datapoints.
    mixmp: MixtureMessagePassing | MolecularMessagePassing | InteractionMessagePassing | None = None
        the optional message passing block for passing messages between molecule embeddings
    """
    output_dim: int

    def __init__(
        self,
        graph_agg: Aggregation,
        groups: Sequence[Sequence[int]],
        fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing
        | MolecularMessagePassing
        | InteractionMessagePassing
        | None = None,
    ):
        if mixmp is not None:
            if len(set(fp_dims)) > 1:
                raise ValueError(
                    "If using mixmp, the fp_dim for each component in a datapoint must be the same."
                    )
            if fp_dims[0] != mixmp.output_dim:
                raise ValueError(
                    "If using mixmp, the fp_dim for each component must be equal to mixmp.output_dim."
                )

        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "groups": groups,
            "fp_dims": fp_dims,
            "graph_agg": graph_agg.hparams,
            "mixmp": mixmp.hparams if mixmp is not None else None,
        }
        self.graph_agg = graph_agg
        self.groups = groups
        self.fp_dims = fp_dims
        self.mixmp = mixmp

    def _aggregate_components(
        self,
        H_vs: list[Tensor | None],
        bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Aggregate all atom representations into per-component mixture representation"""
        # Atom-to-component aggregation
        Hs, w_fps, Hs_batch = [], [], []

        for H_v, bmg in zip(H_vs, bmgs):
            if isinstance(bmg, BatchMixtureGraph):
                continue
            # If all datapoints in a batch are missing a component, skip the bmg for that component
            if bmg is None:
                continue

            H_batch, batch_contiguous = torch.unique(bmg.batch, return_inverse=True)
            H = self.graph_agg(H_v, batch_contiguous)
            if isinstance(bmg, BatchComponentMolGraph):
                H = torch.concat([H, bmg.G], dim=1)
            Hs.append(H)
            # The i-th element in `Hs_batch` says which datapoints have an i-th component.
            Hs_batch.append(H_batch)
            w_fps.append(getattr(bmg, "w_fps", None))

        # Message passing between all molecules/reactions in datapoint
        # TODO: Allow mixmp to act on groups separately
        if self.mixmp is not None:
            # Components are vertexes and mixtures are sub-graphs
            V = torch.cat(Hs)
            batch = torch.cat(Hs_batch)

            if isinstance(self.mixmp, MixtureMessagePassing):
                bmg = BatchNodesOnly(V=V, batch=batch)
            elif isinstance(self.mixmp, (MolecularMessagePassing, InteractionMessagePassing)):
                if not isinstance(bmgs[-1], BatchMixtureGraph):
                    raise ValueError(
                        f"{type(self.mixmp).__name__} requires a MixtureGraphDataset as the final"
                        f" entry MixtureDataset.datasets."
                    )

                # bmg.V is ordered by mixture, then by component.
                # V is ordered by component, then by mixture.
                component_order_to_mixture_order = torch.argsort(batch, stable=True)
                mixture_order_to_component_order = torch.argsort(component_order_to_mixture_order)
                # Shallow copy to not delete original tensors of bmg on overwrite, in case forward
                # called twice on a batch
                bmg = copy.copy(bmgs[-1])
                bmg.V = torch.concat([bmg.V[mixture_order_to_component_order], V], dim=1)
                # When the index map is used as *values* instead of indices, the map inverts
                bmg.edge_index = component_order_to_mixture_order[bmg.edge_index]
            else:
                raise ValueError(
                    f"Mixture message passing block of type {type(self.mixmp)} is not supported"
                )

            Hs = self.mixmp(bmg)
            sizes = [b.shape[0] for b in Hs_batch]
            Hs = list(torch.split(Hs, sizes))

        # Pad missing components with zeros
        def reinsert_nones(xs, mask):
            it = iter(xs)
            return [None if m else next(it) for m in mask]

        bmg_is_None = [bmg is None for bmg in bmgs if not isinstance(bmg, BatchMixtureGraph)]
        Hs = reinsert_nones(Hs, bmg_is_None)
        Hs_batch = reinsert_nones(Hs_batch, bmg_is_None)
        w_fps = reinsert_nones(w_fps, bmg_is_None)

        batch_size = next(len(bmg) for bmg in bmgs if bmg is not None)
        device = next(H for H in Hs if H is not None).device

        def pad_with_zeros(vals: Tensor | None, idx: Tensor | None, dim: int | None) -> Tensor:
            out = torch.zeros(
                (batch_size, dim) if dim is not None else (batch_size,), device=device
            )
            if vals is not None and idx is not None:
                out[idx] = vals
            return out

        Hs = [
            pad_with_zeros(H, H_batch, dim) for H, H_batch, dim in zip(Hs, Hs_batch, self.fp_dims)
        ]
        w_fps = [pad_with_zeros(w_fp, H_batch, None) for w_fp, H_batch in zip(w_fps, Hs_batch)]

        # Molecule-to-mixture aggregation: implemented in subclasses

        return Hs, w_fps, Hs_batch

    @abstractmethod
    def forward(
        self,
        H_vs: list[Tensor | None],
        bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
    ) -> Tensor:
        """Aggregate component representations into a single mixture representation.

        Start with `Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)`.
        """


class ConcatAggregation(MixtureAggregation):
    r"""Concatenate aggregation of the graph-level representation:

    .. math::
        \mathbf h_g = \text{concat}_c \mathbf h_c

        \mathbf h = \text{concat}(\text{concat}_g \mathbf h_g, \text{concat}_c w_c)
    """

    @property
    def components_in_mixture(self) -> set[int]:
        return {idx for group in self.groups if len(group) > 1 for idx in group}

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims) + len(self.components_in_mixture)

    def forward(
        self,
        H_vs: list[Tensor],
        bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
    ) -> Tensor:
        Hs, w_fps, _ = self._aggregate_components(H_vs, bmgs)

        w_fps = torch.stack([w_fps[idx] for idx in self.components_in_mixture], dim=1)
        return torch.cat(Hs + [w_fps], 1)


class WeightedSumAggregation(MixtureAggregation):
    r"""Weighted sum (MolPool) aggregation of the graph-level representation:

    .. math::
        \mathbf h = \sum_{c \in C} w_c \mathbf h_c
    """

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self,
        H_vs: list[Tensor],
        bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
    ) -> Tensor:
        Hs, w_fps, _ = self._aggregate_components(H_vs, bmgs)

        combined_Hs = []
        for group in self.groups:
            if len(group) == 1:
                combined_Hs.append(Hs[group[0]])
                continue
            group_Hs = torch.stack([Hs[idx] for idx in group])  # n x b x d
            group_w_fps = torch.stack([w_fps[idx] for idx in group])  # n x b
            # n: num. components in group, b: num. datapoints in batch, d: output dim of message passing
            combined_H = torch.einsum("nb,nbd->bd", group_w_fps, group_Hs)
            combined_Hs.append(combined_H)
        return torch.cat(combined_Hs, 1)


class DeepsetsAggregation(MixtureAggregation):
    r"""Deep sets aggregation of the graph-level representation:

    .. math::
        \mathbf h_g = \begin{cases}
        \mathrm{MLP_g}(\mathbf h_c) & \text{if } |C| = 1 \\
        \mathrm{MLP_g}\left(\sum_{c \in C} \mathrm{MLP_l}(w_c \cdot \mathbf h_c)\right) & \text{if } |C| > 1
        \end{cases}

        \mathbf h = \text{concat}_g \mathbf h_g
    """

    def __init__(
        self,
        graph_agg: Aggregation,
        groups: Sequence[Sequence[int]],
        fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing
        | MolecularMessagePassing
        | InteractionMessagePassing
        | None = None,
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp)

        self.MLPs_local = nn.ModuleList([])
        self.MLPs_global = nn.ModuleList([])
        for group in groups:
            # TODO: allow to set hparams for MLP by kwargs (e.g., hidden_dim, n_layers)
            hidden_dim = self.fp_dims[group[0]]
            if len(group) > 1:
                self.MLPs_local.append(
                    nn.Sequential(
                        nn.Linear(self.fp_dims[group[0]], hidden_dim, bias=False),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim, bias=False),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.fp_dims[group[0]], bias=False),
                    )
                )
            self.MLPs_global.append(
                nn.Sequential(
                    nn.Linear(self.fp_dims[group[0]], hidden_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.fp_dims[group[0]], bias=False),
                )
            )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self,
        H_vs: list[Tensor | None],
        bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
    ) -> Tensor:
        Hs, w_fps, _ = self._aggregate_components(H_vs, bmgs)

        local_mlps = iter(self.MLPs_local)
        combined_Hs = []
        for g_idx, group in enumerate(self.groups):
            if len(group) == 1:
                # local MLP would just nest into global MLP
                combined_H = Hs[group[0]]
            else:
                local_mlp = next(local_mlps)
                group_w_Hs = torch.stack(
                    [local_mlp(w_fps[idx].unsqueeze(1) * Hs[idx]) for idx in group]
                )  # n x b x d
                combined_H = torch.sum(group_w_Hs, dim=0)
            combined_Hs.append(self.MLPs_global[g_idx](combined_H))
        return torch.cat(combined_Hs, 1)


class AttentiveAggregation(MixtureAggregation):
    r"""Attentive aggregation of the graph-level representation:

    .. math::
        \mathbf h_g = \sum_{c \in C} \alpha_c (w_c \cdot \mathbf h_c)

        \alpha_c = \mathrm{softmax}(\mathbf{W}_a (w_c \cdot \mathbf h_c))

        \mathbf h = \text{concat}_g \mathbf h_g
    """

    def __init__(
        self,
        graph_agg: Aggregation,
        groups: Sequence[Sequence[int]],
        fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing
        | MolecularMessagePassing
        | InteractionMessagePassing
        | None = None,
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp)

        self.Ws_a = nn.ModuleList(
            [nn.Linear(self.fp_dims[group[0]], 1, bias=False) for group in groups if len(group) > 1]
        )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self,
        H_vs: list[Tensor | None],
        bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)

        attn_layers = iter(self.Ws_a)
        combined_Hs = []
        for group in self.groups:
            if len(group) == 1:
                combined_Hs.append(Hs[group[0]])
                continue

            W_a = next(attn_layers)
            w_Hs = torch.stack([w_fps[idx].unsqueeze(1) * Hs[idx] for idx in group])  # n x b x d
            logits = W_a(w_Hs).squeeze(-1)  # n x b

            # Mask out batch entries with missing components (so they don't contribute to softmax)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i, idx in enumerate(group):
                if Hs_batch[idx] is not None:
                    mask[i, Hs_batch[idx]] = True

            logits = logits.masked_fill(~mask, float("-inf"))
            alphas = torch.softmax(logits, dim=0)
            combined_H = torch.sum(alphas.unsqueeze(-1) * w_Hs, dim=0)
            combined_Hs.append(combined_H)

        return torch.cat(combined_Hs, 1)


class Set2SetAggregation(MixtureAggregation):
    r"""Set2Set aggregation of the graph-level representation:

    For each group with :math:`|C| > 1`

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{c,t} &= \mathrm{softmax}(w_c \mathbf{h}_c \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_c \alpha_{c,t} w_c \mathbf{h}_c

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf h_g = \mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input. Groups with a single component pass through unchanged, i.e.
    :math:`\mathbf h_g = \mathbf h_c`. The final mixture representation concatenates over groups:

    .. math::
        \mathbf h = \text{concat}_g \mathbf h_g

    Note: This implementation follows PyTorch Geometric
    (cf. https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/aggr/set2set.html#Set2Set)
    and is based on `"Order Matters: Sequence to sequence for Sets" <https://arxiv.org/abs/1511.06391>`_ paper.
    """

    def __init__(
        self,
        graph_agg: Aggregation,
        groups: Sequence[Sequence[int]],
        fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing
        | MolecularMessagePassing
        | InteractionMessagePassing
        | None = None,
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp)

        # TODO: allow to set hparams for Set2Set by kwargs (e.g., processing steps)
        self.processing_steps = 3
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(self.fp_dims[group[0]] * 2, self.fp_dims[group[0]])
                for group in groups
                if len(group) > 1
            ]
        )

    @property
    def output_dim(self) -> int:
        return sum(
            self.fp_dims[group[0]] * 2 if len(group) > 1 else self.fp_dims[group[0]]
            for group in self.groups
        )

    def forward(
        self,
        H_vs: list[Tensor | None],
        bmgs: list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)

        lstms = iter(self.lstms)
        combined_Hs = []
        for group in self.groups:
            if len(group) == 1:
                combined_Hs.append(Hs[group[0]])
                continue

            lstm = next(lstms)
            w_Hs = torch.stack([w_fps[idx].unsqueeze(1) * Hs[idx] for idx in group])  # n x b x d
            w_Hs = torch.transpose(w_Hs, 0, 1)  # b x n x d
            b_dim, n_dim, d_dim = w_Hs.shape

            mask = torch.zeros((b_dim, n_dim), dtype=torch.bool, device=w_Hs.device)
            for i, idx in enumerate(group):
                if Hs_batch[idx] is not None:
                    mask[Hs_batch[idx], i] = True

            h = (
                w_Hs.new_zeros((lstm.num_layers, b_dim, d_dim)),
                w_Hs.new_zeros((lstm.num_layers, b_dim, d_dim)),
            )
            q_star = w_Hs.new_zeros(b_dim, d_dim * 2)

            for _ in range(self.processing_steps):
                q, h = lstm(q_star.unsqueeze(0), h)
                q = q.squeeze(0)  # b x d

                logits = (w_Hs * q.unsqueeze(1)).sum(dim=2)  # b x n
                logits = logits.masked_fill(~mask, float("-inf"))
                alphas = torch.softmax(logits, dim=1)  # b x n

                r = torch.sum(w_Hs * alphas.unsqueeze(2), dim=1)  # b x d
                q_star = torch.cat([q, r], dim=1)  # b x 2*d
            combined_Hs.append(q_star)
        return torch.cat(combined_Hs, 1)
