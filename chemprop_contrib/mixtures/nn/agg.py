from abc import abstractmethod

import torch
from torch import Tensor, nn

from chemprop.nn.hparams import HasHParams
from chemprop.nn.utils import get_activation_function


class MixtureAggregation(nn.Module, HasHParams):
    """A `MixtureAggregation` aggregates molecule learned fingerprints into fixed-length embedding.

    Parameters
    ----------
    fp_dim : int
        the dimension of the learned fingerpint of the molecules in the mixture. Use to calculate
        self.output_dim. Also used by some aggregation schemes.
    """

    def __init__(self, fp_dim: int):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "fp_dim": fp_dim,
        }
        self.fp_dim = fp_dim

    @property
    def output_dim(self) -> int:
        return self.fp_dim

    @abstractmethod
    def forward(self, H: Tensor, batch: Tensor, w_fps: Tensor) -> Tensor:
        """Combine the learned fingerprints of molecules in a mixture into the respective mixture
        representation.

        Parameters
        ----------
        H : Tensor
            a tensor of shape ``N x d`` containing the learned fingerprints, where ``N`` is the
            total number of molecules in the batch of mixture data, and ``d`` is the length of the
            learned fingerprint
        batch : Tensor
            a tensor of shape ``N`` containing the index of the mixture a given learned fingerprint
            corresponds to
        w_fps : Tensor
            a tensor of shape ``N`` containing the predetermined weights for each learned
            fingerprint
        Returns
        -------
        Tensor
            a tensor of shape ``b x d`` containing the mixture-level representations, where ``b`` is
            the number of mixtures in the batch
        """


class ConcatAggregation(MixtureAggregation):
    r"""
    .. math::
        mathbf h = \text{concat}(\text{concat}_c \mathbf h_c, \text{concat}_c w_c)

    Parameters
    ----------
    fp_dim : int
        the dimension of the learned fingerpint of the molecules in the mixture.
    max_components : int
        the maximum number of components allowed in a mixture, affects zero-padding
    randomize_pad_position : bool, default=False
        whether to randomize the position of the zero-pad during training. False corresponds to
        always zero-padding on the right.
    randomize_component_order : bool, default=False
        whether to randomize the order of the components when concatenating during training. False
        corresponds to keeping them in the order given.
    """

    def __init__(
        self,
        fp_dim: int,
        max_components: int,
        randomize_pad_position: bool = False,
        randomize_component_order: bool = False,
    ):
        super().__init__(fp_dim)
        self.hparams.update(
            {
                "max_components": max_components,
                "randomize_pad_position": randomize_pad_position,
                "randomize_component_order": randomize_component_order,
            }
        )
        self.max_components = max_components
        self.randomize_pad_position = randomize_pad_position
        self.randomize_component_order = randomize_component_order

    @property
    def output_dim(self) -> int:
        return (self.fp_dim + 1) * self.max_components

    def _compute_concat_order(
        self,
        batch_mixture: Tensor,
        i_mol_in_mix: Tensor,
        n_mol_per_mix: Tensor,
        B: int,
        K: int,
        device: torch.device,
    ) -> Tensor:
        # A bit of magic to get a tensor which says which column in the output each row in H should
        # go. Placing the rows of H in a different columns, depending on the row, is equivalent to
        # concatenating them in different orders.
        rand_pad = self.randomize_pad_position and self.training
        rand_ord = self.randomize_component_order and self.training
        if not (rand_pad or rand_ord):
            return i_mol_in_mix

        perm = torch.rand(B, K, device=device).argsort(dim=1)

        if rand_pad and rand_ord:
            return perm[batch_mixture, i_mol_in_mix]

        col = torch.arange(K, device=device).unsqueeze(0)
        mask = col < n_mol_per_mix.unsqueeze(1)
        masked_perm = perm.masked_fill(~mask, K + 1)

        if rand_pad:
            sorted_subset = masked_perm.sort(dim=1).values
            return sorted_subset[batch_mixture, i_mol_in_mix]

        ranks = masked_perm.argsort(dim=1).argsort(dim=1)
        return ranks[batch_mixture, i_mol_in_mix]

    def forward(self, H: Tensor, batch_mixture: Tensor, w: Tensor) -> Tensor:
        B = int(batch_mixture.max().item()) + 1
        N, d = H.shape
        K = self.max_components
        device = H.device

        # Number of molecules in each mixture
        n_mol_per_mix = torch.bincount(batch_mixture, minlength=B)
        if (n_mol_per_mix > K).any():
            raise ValueError(
                f"A mixture has {int(n_mol_per_mix.max())} components, exceeding max_components={K}."
            )

        # The indices where each mixture's molecules start in the batched H
        i_mol_start = torch.cumsum(n_mol_per_mix, 0) - n_mol_per_mix
        # The indices of each molecule in their respective mixture
        # Faster equivalent of torch.concat([torch.arange(n) for n in n_mol_per_mix])
        i_mol_in_mix = torch.arange(N, device=device) - i_mol_start[batch_mixture]
        concat_order = self._compute_concat_order(
            batch_mixture, i_mol_in_mix, n_mol_per_mix, B, K, device
        )

        out_H = H.new_zeros((B, K, d))
        out_w = w.new_zeros((B, K))
        out_H[batch_mixture, concat_order] = H
        out_w[batch_mixture, concat_order] = w
        return torch.cat([out_H.reshape(B, K * d), out_w], dim=1)


class WeightedSumAggregation(MixtureAggregation):
    r"""
    .. math::
        \mathbf h = \sum_{c \in C} w_c \mathbf h_c
    """

    def forward(self, H: Tensor, batch_mixture: Tensor, w: Tensor) -> Tensor:
        B = int(batch_mixture.max().item()) + 1
        return torch.zeros((B, H.shape[1]), device=H.device, dtype=H.dtype).index_add_(
            0, batch_mixture, w.unsqueeze(1) * H
        )


class DeepsetsAggregation(MixtureAggregation):
    r"""
    .. math::
        \mathbf h = \mathrm{MLP_g}\!\left(\sum_{c \in C} \mathrm{MLP_l}(w_c \mathbf h_c)\right)

    Parameters
    ----------
    fp_dim : int
        the dimension of the learned fingerpint of the molecules in the mixture
    hidden_dim : int | None, default=None
        an optional hidden dimension for the MLPs, defaults to fp_dim if None
    n_hidden_layers : int, default=2
        number of hidden layers in the MLPs
    bias : bool, default=False
        whether to include a bias in the MLP layers
    activation : nn.Module, default=nn.ReLU()
        the non-linear activation function to use between layers in the MLPs
    """

    def __init__(self, fp_dim: int, **mlp_args):
        super().__init__(fp_dim)
        self.hparams.update(mlp_args)
        self.MLP_local = self._make_mlp(fp_dim, **mlp_args)
        self.MLP_global = self._make_mlp(fp_dim, **mlp_args)

    @staticmethod
    def _make_mlp(
        dim: int,
        hidden_dim: int | None = None,
        n_hidden_layers: int = 2,
        bias: bool = False,
        activation: str | nn.Module = "relu",
    ) -> nn.Sequential:
        if n_hidden_layers == 0:
            return nn.Sequential(nn.Linear(dim, dim, bias=bias))

        activation = get_activation_function(activation)
        hidden_dim = hidden_dim or dim
        layers = [nn.Linear(dim, hidden_dim, bias=bias), activation]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=bias), activation]
        layers.append(nn.Linear(hidden_dim, dim, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, H: Tensor, batch_mixture: Tensor, w: Tensor) -> Tensor:
        B = int(batch_mixture.max().item()) + 1
        pooled = torch.zeros((B, H.shape[1]), device=H.device, dtype=H.dtype).index_add_(
            0, batch_mixture, self.MLP_local(w.unsqueeze(1) * H)
        )
        return self.MLP_global(pooled)


class AttentiveAggregation(MixtureAggregation):
    r"""
    .. math::
        \mathbf h = \sum_{c \in C} \alpha_c (w_c \mathbf h_c),\quad
        \alpha_c = \mathrm{softmax}_C\!\left(\mathbf{W}_a (w_c \mathbf h_c)\right)
    """

    def __init__(self, fp_dim: int):
        super().__init__(fp_dim)
        self.W_a = nn.Linear(fp_dim, 1, bias=False)

    def forward(self, H: Tensor, batch_mixture: Tensor, w: Tensor) -> Tensor:
        B = int(batch_mixture.max().item()) + 1
        device = H.device

        w_H = w.unsqueeze(1) * H
        logits = self.W_a(w_H).squeeze(-1)

        # Subtract logits by max for stability
        # Alternatively, we could use torch.softmax, but that requires zero padding so all mixtures
        # have the same number of inputs.
        # Alternatively, we could use something like
        # `torch_scatter.scatter_softmax(logits, batch_mixture, dim=0, dim_size=B)`, but that is
        # another dependency.
        max_per_b = torch.full((B,), float("-inf"), device=device)
        max_per_b.scatter_reduce_(0, batch_mixture, logits, reduce="amax", include_self=True)
        exps = torch.exp(logits - max_per_b[batch_mixture])
        sum_per_b = torch.zeros(B, device=device).index_add_(0, batch_mixture, exps)
        alphas = exps / sum_per_b[batch_mixture]

        return torch.zeros((B, H.shape[1]), device=device, dtype=H.dtype).index_add_(
            0, batch_mixture, alphas.unsqueeze(-1) * w_H
        )


class Set2SetAggregation(MixtureAggregation):
    r"""
    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{c,t} &= \mathrm{softmax}_C(w_c \mathbf{h}_c \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{c \in C} \alpha_{c,t} w_c \mathbf{h}_c

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t

    where :math:`\mathbf h = \mathbf{q}^{*}_T` is the output of the layer with twice
    the dimensionality of the input.

    Note: This implementation follows PyTorch Geometric
    (cf. https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/aggr/set2set.html#Set2Set)
    and is based on `"Order Matters: Sequence to sequence for Sets" <https://arxiv.org/abs/1511.06391>`_ paper.
    """

    def __init__(self, fp_dim: int, processing_steps: int = 3):
        super().__init__(fp_dim)
        self.hparams["processing_steps"] = processing_steps
        self.processing_steps = processing_steps
        self.lstm = nn.LSTM(fp_dim * 2, fp_dim)

    @property
    def output_dim(self) -> int:
        return self.fp_dim * 2

    def forward(self, H: Tensor, batch_mixture: Tensor, w: Tensor) -> Tensor:
        B = int(batch_mixture.max().item()) + 1
        device = H.device
        dim = H.shape[1]

        w_H = w.unsqueeze(1) * H

        h = (
            w_H.new_zeros((self.lstm.num_layers, B, dim)),
            w_H.new_zeros((self.lstm.num_layers, B, dim)),
        )
        q_star = w_H.new_zeros(B, dim * 2)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.squeeze(0)

            logits = (w_H * q[batch_mixture]).sum(dim=1)

            # See note about manual softmax in `AttentiveAggregation.forward`
            max_per_b = torch.full((B,), float("-inf"), device=device)
            max_per_b.scatter_reduce_(0, batch_mixture, logits, reduce="amax", include_self=True)
            exps = torch.exp(logits - max_per_b[batch_mixture])
            sum_per_b = torch.zeros(B, device=device).index_add_(0, batch_mixture, exps)
            alphas = exps / sum_per_b[batch_mixture]

            r = torch.zeros((B, dim), device=device, dtype=H.dtype).index_add_(
                0, batch_mixture, alphas.unsqueeze(-1) * w_H
            )
            q_star = torch.cat([q, r], dim=1)

        return q_star
