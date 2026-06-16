from abc import abstractmethod
import copy
from typing import Sequence

import torch
from chemprop.data import BatchMolGraph
from chemprop.nn.agg import Aggregation
from chemprop.nn.hparams import HasHParams
from chemprop.nn.transforms import ScaleTransform
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
    graph_agg : Aggregation
        an instance of a chemprop Aggregation for the node to graph aggregation
    groups : Sequence[Sequence[int]]
        the indices of the molecules/components split into groups, e.g. [[0],[1,2]] for solute in
        binary solvent
    fp_dims : Sequence[int]
        the dimensions of the final group embeddings
    mixmp : MixtureMessagePassing | MolecularMessagePassing | InteractionMessagePassing | None = None
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
        G_d_transform: ScaleTransform | None = None,
    ):
        if mixmp is not None:
            if len(set(fp_dims)) > 1:
                raise ValueError(
                    "If using mixmp, the fp_dim for each group must be the same."
                    )
            if fp_dims[0] != mixmp.output_dim:
                raise ValueError(
                    "If using mixmp, the fp_dim must be equal to mixmp.output_dim."
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
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
    ) -> tuple[list[list[Tensor]], list[list[Tensor]], list[list[Tensor]]]:
        """Aggregate all atom representations into per-component mixture representation"""
        # flatten because groups don't matter for the operations in this function
        H_vs = [item for sublist in H_vs for item in sublist]
        bmgs = [item for sublist in bmgs for item in sublist]

        # Atom-to-component aggregation
        Hs, w_fps, Hs_batch = [], [], []

        for H_v, bmg in zip(H_vs, bmgs):
            if isinstance(bmg, BatchMixtureGraph):
                continue
            if bmg is None:
                continue  # Component missing in this batch

            # `H_batch` says which datapoints contribute to the bmg of this i-th component.
            H_batch, batch_contiguous = torch.unique(bmg.batch, return_inverse=True)
            H = self.graph_agg(H_v, batch_contiguous)
            if isinstance(bmg, BatchComponentMolGraph):
                H = torch.concat([H, bmg.G], dim=1)

            Hs.append(H)
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

        # Molecule-to-mixture aggregation: implemented in subclasses

        return Hs, w_fps, Hs_batch
    
    @staticmethod
    def _infer_batch_info(
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
    ) -> tuple[int, torch.device, list[list[bool]]]:
        """returns batch size, tensor device, and mask for present bmg's"""
        batch_size = next(len(bmg) for group in bmgs for bmg in group if bmg is not None)
        device = next(H_v for group in H_vs for H_v in group if H_v is not None).device
        present_components_mask = [[bmg is not None for bmg in group] for group in bmgs]
        return batch_size, device, present_components_mask

    @abstractmethod
    def forward(
        self,
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
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
    def output_dim(self) -> int:
        return sum(
            len(group) * (self.fp_dims[g_idx] + int(len(group) > 1))
            for g_idx, group in enumerate(self.groups)
        )

    def forward(
        self,
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)
        batch_size, device, present_components_mask = self._infer_batch_info(H_vs, bmgs)

        def pad_with_zeros(vals: Tensor | None, idx: Tensor | None, dim: int | None) -> Tensor:
            shape = (batch_size, dim) if dim is not None else (batch_size,)
            out = torch.zeros(shape, device=device)
            if vals is not None and idx is not None:
                out[idx] = vals
            return out

        iter_agg_data = iter(zip(Hs, w_fps, Hs_batch))
        collected_Hs = []
        collected_w_fps = []
        for dim, mask in zip(self.fp_dims, present_components_mask):
            if len(mask) == 1:
                H, _, b = next(iter_agg_data)
                collected_Hs.append(pad_with_zeros(H, b, dim))
            else:
                for present in mask:
                    if present:
                        H, w, b = next(iter_agg_data)
                        collected_Hs.append(pad_with_zeros(H, b, dim))
                        collected_w_fps.append(pad_with_zeros(w, b, None))
                    else:
                        collected_Hs.append(pad_with_zeros(None, None, dim))
                        collected_w_fps.append(pad_with_zeros(None, None, None))


        w_fps = torch.stack(collected_w_fps, dim=1)
        return torch.cat(collected_Hs + [w_fps], dim=1)


class WeightedSumAggregation(MixtureAggregation):
    r"""Weighted sum (MolPool) aggregation of the graph-level representation:

    .. math::
        \mathbf h = \sum_{c \in C} w_c \mathbf h_c
    """

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims)

    def forward(
        self,
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)
        batch_size, device, present_components_mask = self._infer_batch_info(H_vs, bmgs)

        iter_agg_data = iter(zip(Hs, w_fps, Hs_batch))
        combined_Hs = []
        for dim, mask in zip(self.fp_dims, present_components_mask):
            out = torch.zeros((batch_size, dim), device=device)
            if len(mask) == 1:
                H, w, b = next(iter_agg_data)
                out[b] = H
            else:
                for present in mask:
                    if present:
                        H, w, b = next(iter_agg_data)
                        out.index_add_(0, b, w.unsqueeze(1) * H)
            combined_Hs.append(out)
        return torch.cat(combined_Hs, dim=1)


class DeepsetsAggregation(MixtureAggregation):
    r"""Deep sets aggregation of the graph-level representation:

    .. math::
        \mathbf h_g = \begin{cases}
        \mathrm{MLP_g}(\mathbf h_c) & \text{if } |C| = 1 \\
        \mathrm{MLP_g}\left(\sum_{c \in C} \mathrm{MLP_l}(w_c \cdot \mathbf h_c)\right) & \text{if } |C| > 1
        \end{cases}

        \mathbf h = \text{concat}_g \mathbf h_g
    """

    def _make_mlp(self, dim: int, hidden_dim: int | None = None) -> nn.Sequential:
        hidden_dim = hidden_dim or dim
        return nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

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

        # TODO: allow to set hparams for MLP by kwargs (e.g., hidden_dim, n_layers)
        self.MLPs_local = nn.ModuleList(
            [self._make_mlp(self.fp_dims[g_idx])
            for g_idx, group in enumerate(groups) if len(group) > 1]
        )
        self.MLPs_global = nn.ModuleList(
            [self._make_mlp(self.fp_dims[g_idx]) for g_idx in range(len(groups))]
        )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims)

    def forward(
        self,
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)
        batch_size, device, present_components_mask = self._infer_batch_info(H_vs, bmgs)

        iter_agg_data = iter(zip(Hs, w_fps, Hs_batch))
        local_mlps = iter(self.MLPs_local)
        combined_Hs = []
        for g_idx, (dim, mask) in enumerate(zip(self.fp_dims, present_components_mask)):
            local_out = torch.zeros((batch_size, dim), device=device)
            if len(mask) == 1:
                H, w, b = next(iter_agg_data)
                local_out[b] = H
            else:
                local_mlp = next(local_mlps)
                for present in mask:
                    if present:
                        H, w, b = next(iter_agg_data)
                        local_out.index_add_(0, b, local_mlp(w.unsqueeze(1) * H))
            combined_Hs.append(self.MLPs_global[g_idx](local_out))
        return torch.cat(combined_Hs, dim=1)


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
            [nn.Linear(self.fp_dims[g_idx], 1, bias=False) for g_idx, group in enumerate(groups) if len(group) > 1]
        )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims)

    def forward(
        self,
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)
        batch_size, device, present_components_mask = self._infer_batch_info(H_vs, bmgs)

        iter_agg_data = iter(zip(Hs, w_fps, Hs_batch))
        attn_layers = iter(self.Ws_a)
        combined_Hs = []
        for dim, mask in zip(self.fp_dims, present_components_mask):
            out = torch.zeros((batch_size, dim), device=device)
            if len(mask) == 1:
                H, w, b = next(iter_agg_data)
                out[b] = H
                combined_Hs.append(out)
                continue

            W_a = next(attn_layers)
            w_Hs_list, b_list = [], []
            for present in mask:
                if present:
                    H, w, b = next(iter_agg_data)
                    w_Hs_list.append(w.unsqueeze(1) * H)
                    b_list.append(b)

            w_H_flat = torch.cat(w_Hs_list, dim=0)
            b_flat = torch.cat(b_list, dim=0)
            logits = W_a(w_H_flat).squeeze(-1)

            # Subtract logits by max for stability
            # Alternatively, we could use torch.softmax, but that requires zero padding missing
            # components, which is expensive if the batch is sparse.
            # Alternatively, we could use something like 
            # `torch_scatter.scatter_softmax(logits, b_flat, dim=0, dim_size=batch_size)`, but that
            # is another dependency.
            max_per_b = torch.full((batch_size,), float("-inf"), device=device)
            max_per_b.scatter_reduce_(0, b_flat, logits, reduce="amax", include_self=True)
            exps = torch.exp(logits - max_per_b[b_flat])

            sum_per_b = torch.zeros(batch_size, device=device)
            sum_per_b.index_add_(0, b_flat, exps)
            alphas = exps / sum_per_b[b_flat]

            out.index_add_(0, b_flat, alphas.unsqueeze(-1) * w_H_flat)
            combined_Hs.append(out)
        return torch.cat(combined_Hs, dim=1)


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
                nn.LSTM(self.fp_dims[g_idx] * 2, self.fp_dims[g_idx])
                for g_idx, group in enumerate(self.groups)
                if len(group) > 1
            ]
        )

    @property
    def output_dim(self) -> int:
        return sum(
            self.fp_dims[g_idx] * 2 if len(group) > 1 else self.fp_dims[g_idx]
            for g_idx, group in enumerate(self.groups)
        )

    def forward(
        self,
        H_vs: list[list[Tensor | None]],
        bmgs: list[list[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self._aggregate_components(H_vs, bmgs)
        batch_size, device, present_components_mask = self._infer_batch_info(H_vs, bmgs)

        iter_agg_data = iter(zip(Hs, w_fps, Hs_batch))
        lstms = iter(self.lstms)
        combined_Hs = []
        for dim, mask in zip(self.fp_dims, present_components_mask):
            if len(mask) == 1:
                out = torch.zeros((batch_size, dim), device=device)
                H, w, b = next(iter_agg_data)
                out[b] = H
                combined_Hs.append(out)
                continue

            lstm = next(lstms)
            w_Hs_list, b_list = [], []
            for present in mask:
                if present:
                    H, w, b = next(iter_agg_data)
                    w_Hs_list.append(w.unsqueeze(1) * H)
                    b_list.append(b)

            w_H_flat = torch.cat(w_Hs_list, dim=0)
            b_flat = torch.cat(b_list, dim=0)

            h = (
                w_H_flat.new_zeros((lstm.num_layers, batch_size, dim)),
                w_H_flat.new_zeros((lstm.num_layers, batch_size, dim)),
            )
            q_star = w_H_flat.new_zeros(batch_size, dim * 2)

            for _ in range(self.processing_steps):
                q, h = lstm(q_star.unsqueeze(0), h)
                q = q.squeeze(0)

                logits = (w_H_flat * q[b_flat]).sum(dim=1)

                # See note about manual softmax in `AttentiveAggregation.forward`
                max_per_b = torch.full((batch_size,), float("-inf"), device=device)
                max_per_b.scatter_reduce_(0, b_flat, logits, reduce="amax", include_self=True)
                exps = torch.exp(logits - max_per_b[b_flat])

                sum_per_b = torch.zeros(batch_size, device=device)
                sum_per_b.index_add_(0, b_flat, exps)
                alphas = exps / sum_per_b[b_flat]

                r = torch.zeros((batch_size, dim), device=device)
                r.index_add_(0, b_flat, alphas.unsqueeze(-1) * w_H_flat)
                q_star = torch.cat([q, r], dim=1)

            combined_Hs.append(q_star)
        return torch.cat(combined_Hs, dim=1)
