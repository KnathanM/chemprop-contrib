import logging
from typing import Iterable, Sequence

import torch
from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.data.collate import BatchMolGraph
from chemprop.nn.hparams import HasHParams
from chemprop.nn.message_passing import AtomMessagePassing, BondMessagePassing, MessagePassing
from chemprop.nn.utils import Activation, get_activation_function
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import Tensor, nn

from chemprop_contrib.mixtures.data import BatchComponentMolGraph, BatchMixtureGraph, BatchNodesOnly

logger = logging.getLogger(__name__)


class MixtureMulticomponentMessagePassing(nn.Module, HasHParams):
    """A `MixtureMulticomponentMessagePassing` performs message-passing on each individual input in
    a multicomponent input, while accounting for shared message-passing blocks for a group of
    components in a mixture.

    Parameters
    ----------
    blocks : Sequence[MessagePassing]
        the invidual message-passing blocks for each group
    groups: Sequence[Sequence[int]]
        the indices of the molecules/components split into groups, e.g. [[0],[1,2]] for solute in
        binary solvent
    shared : bool, default=False
        whether one block will be shared among all groups
    """

    def __init__(
        self,
        blocks: Sequence[MessagePassing],
        groups: Sequence[Sequence[int]],
        shared: bool = False,
    ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "blocks": [block.hparams for block in blocks],
            "groups": groups,
            "shared": shared,
        }

        if shared and len(blocks) > 1:
            logger.warning(
                "More than 1 block was supplied but 'shared' was True! Using only the 0th block..."
            )
        elif not shared and len(blocks) != len(groups):
            raise ValueError(
                "arg 'len(groups)' must be equal to `len(blocks)` if 'shared' is False! "
                f"got: {len(groups)} and {len(blocks)}, respectively."
            )

        self.groups = groups
        self.shared = shared
        self.blocks = nn.ModuleList()
        if shared:
            self.blocks.extend([blocks[0]] * sum(len(g) for g in groups))
        else:
            for g_idx, g in enumerate(groups):
                self.blocks.extend([blocks[g_idx]] * len(g))

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def output_dims(self) -> list[int]:
        return [block.output_dim for block in self.blocks]

    def forward(
        self,
        bmgs: Iterable[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None],
        V_ds: Iterable[Tensor | None],
    ) -> list[Tensor | None]:
        # If the final element in bmgs is a BatchMixtureGraph, then len(bmgs) = len(self.blocks) - 1
        # The BatchMixtureGraph is used in agg.mixmp, not here.
        return [
            block(bmg, V_d) if bmg is not None else None
            for block, bmg, V_d in zip(self.blocks, bmgs, V_ds)
        ]


class MixtureMessagePassing(nn.Module, HyperparametersMixin, HasHParams):
    r"""A :class:`MixtureMessagePassing` updates encodings of components in a mixture by passing
    messages between them in a fully connected graph (no self connections).

    It implements the following operation:

    .. math::

        h_v^{(0)} &= \tau \left( \mathbf{W}_i\, x_v \right) \\
        m_v^{(t)} &= \sum_{w \in \mathcal{V} \setminus \{v\}} h_w^{(t-1)} \\
        h_v^{(t)} &= \tau\left( \mathbf{W}_i\, x_v + \mathbf{W}_h\, m_v^{(t)} \right)

    where :math:`\tau` is the activation function; :math:`\mathbf{W}_i` and :math:`\mathbf{W}_h`
    are learned weight matrices; :math:`x_v` is the feature vector of component :math:`v`;
    :math:`\mathcal{V}` denotes the set of components in the same mixture as :math:`v`
    (the mixture graph is assumed fully connected); :math:`h_v^{(t)}` is the hidden
    representation of component :math:`v` at iteration :math:`t`; :math:`m_v^{(t)}` is the
    message received by component :math:`v` at iteration :math:`t`; and
    :math:`t \in \{1, \dots, T\}` indexes the message-passing iterations.
    """

    def __init__(
        self,
        d_v: int = DEFAULT_HIDDEN_DIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 1,
        activation: str | Activation = Activation.RELU,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.depth = depth
        self.tau = get_activation_function(activation)

        self.W_i = nn.Linear(d_v, d_h, bias)
        self.W_h = nn.Linear(d_h, d_h, bias)

    @property
    def output_dim(self) -> int:
        return self.W_h.out_features

    def initialize(self, bmg: BatchNodesOnly) -> Tensor:
        return self.W_i(bmg.V)

    def message(self, H: Tensor, bmg: BatchNodesOnly):
        batch_size = int(bmg.batch.max().item()) + 1
        M = torch.zeros(batch_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, bmg.batch.unsqueeze(1).expand_as(H), H, reduce="sum", include_self=False
        )[bmg.batch]
        return M - H  # exclude self

    def update(self, M_t: Tensor, H_0: Tensor):
        H_t = self.W_h(M_t)
        H_t = self.tau(H_0 + H_t)
        return H_t

    def forward(self, bmg: BatchNodesOnly) -> Tensor:
        H_0 = self.initialize(bmg)
        H = self.tau(H_0)
        for _ in range(self.depth):
            M = self.message(H, bmg)
            H = self.update(M, H_0)
        return H


class InteractionMessagePassing(BondMessagePassing):
    r"""Same as BondMessagePassing, except bonds are replaced with intermolecular interactions and
    the args for `forward` are as follows:

    Parameters
    ----------
    bmg: BatchMixtureGraph
        a batch of :class:`MixtureGraph`s to encode
    V_d : Tensor | None, default=None
        an optional tensor of shape ``V x d_vd`` containing additional descriptors for each molecule

    Returns
    -------
    Tensor
        a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
        molecule in the batch, depending on whether additional molecular descriptors were provided
    """


class MolecularMessagePassing(AtomMessagePassing):
    r"""Same as AtomMessagePassing, except atoms are replaced with molecules  and args for `forward`
    are as follows:

    Parameters
    ----------
    bmg: BatchMixtureGraph
        a batch of :class:`MixtureGraph`s to encode
    V_d : Tensor | None, default=None
        an optional tensor of shape ``V x d_vd`` containing additional descriptors for each molecule

    Returns
    -------
    Tensor
        a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
        molecule in the batch, depending on whether additional molecular descriptors were provided
    """
