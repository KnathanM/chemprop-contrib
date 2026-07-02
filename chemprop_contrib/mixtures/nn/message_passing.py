import logging

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn

from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.nn.hparams import HasHParams
from chemprop.nn.message_passing import AtomMessagePassing, BondMessagePassing, MessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function

from chemprop_contrib.mixtures.data.collate import BatchInteractionGraph

logger = logging.getLogger(__name__)


class MixtureMessagePassing(nn.Module, HyperparametersMixin, HasHParams):
    r"""A :class:`MixtureMessagePassing` performs simple message passing between connected nodes. No
    edge information is used.

    It implements the following operation:

    .. math::

        h_v^{(0)} &= \tau \left( \mathbf{W}_i\, x_v \right) \\
        m_v^{(t)} &= \sum_{w \in \mathcal{N}(v)} h_w^{(t-1)} \\
        h_v^{(t)} &= \tau\left( \mathbf{W}_i\, x_v + \mathbf{W}_h\, m_v^{(t)} \right)

    where :math:`\tau` is the activation function; :math:`\mathbf{W}_i` and :math:`\mathbf{W}_h`
    are learned weight matrices; :math:`x_v` is the feature vector of node :math:`v`;
    :math:`\mathcal{N}(v)` is the set of neighbors of :math:`v` as given by the graph's
    ``edge_index``; :math:`h_v^{(t)}` is the hidden representation of node :math:`v` at iteration
    :math:`t`; :math:`m_v^{(t)}` is the message received by node :math:`v` at iteration :math:`t`;
    and :math:`t \in \{1, \dots, T\}` indexes the message-passing iterations.
    """

    def __init__(
        self,
        d_v: int = DEFAULT_HIDDEN_DIM,
        d_e: None = None,  # Only here for signature parity
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 1,
        activation: str | Activation = Activation.RELU,
        undirected: None = None,  # Only here for signature parity
        d_vd: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__()
        ignore_list = ["V_d_transform", "graph_transform"]
        if isinstance(activation, nn.Module):
            ignore_list.append("activation")
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["V_d_transform"] = V_d_transform
        self.hparams["graph_transform"] = graph_transform
        if isinstance(activation, nn.Module):
            self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__

        self.depth = depth
        self.tau = get_activation_function(activation)

        self.W_i = nn.Linear(d_v, d_h, bias)
        self.W_h = nn.Linear(d_h, d_h, bias)
        self.W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None

        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self.W_d.out_features if self.W_d is not None else self.W_h.out_features

    def initialize(self, big) -> Tensor:
        return self.W_i(big.V)

    def message(self, H: Tensor, big) -> Tensor:
        src, dst = big.edge_index[0], big.edge_index[1]
        M = torch.zeros_like(H)
        M.index_add_(0, dst, H[src])
        return M

    def update(self, M_t: Tensor, H_0: Tensor):
        H_t = self.W_h(M_t)
        H_t = self.tau(H_0 + H_t)
        return H_t

    def forward(self, big: BatchInteractionGraph, V_d: Tensor | None = None) -> Tensor:
        big = self.graph_transform(big)
        H_0 = self.initialize(big)
        H = self.tau(H_0)
        for _ in range(self.depth):
            M = self.message(H, big)
            H = self.update(M, H_0)

        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            H = self.W_d(torch.cat((H, V_d), dim=1))  # V x (d_o + d_vd)

        return H


class InteractionMessagePassing(BondMessagePassing):
    r"""Same as BondMessagePassing except nodes are molecules, edges are interactions, and the args
    for `forward` are as follows:

    Parameters
    ----------
    big: BatchInteractionGraph
    V_d : Tensor | None, default=None
        an optional tensor of shape ``V x d_vd`` containing additional descriptors for each molecule

    Returns
    -------
    Tensor
        a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
        molecule in the batch, depending on whether additional molecular descriptors were provided
    """


class MolecularMessagePassing(AtomMessagePassing):
    r"""Same as AtomMessagePassing except nodes are molecules, edges are interactions, and the args
    for `forward` are as follows:

    Parameters
    ----------
    big: BatchInteractionGraph
    V_d : Tensor | None, default=None
        an optional tensor of shape ``V x d_vd`` containing additional descriptors for each molecule

    Returns
    -------
    Tensor
        a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
        molecule in the batch, depending on whether additional molecular descriptors were provided
    """


class NoMessagePassing(HyperparametersMixin, MessagePassing):
    """The :class:`NoMessagePassing` only performs (optional) transforms on the vertex embeddings"""

    def __init__(
        self,
        d_v: int = DEFAULT_HIDDEN_DIM,
        d_e: None = None,
        d_h: None = None,
        bias: None = None,
        depth: None = None,
        dropout: None = None,
        activation: None = None,
        undirected: None = None,
        d_vd: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["V_d_transform", "graph_transform"])
        self.hparams.update(
            {
                "cls": self.__class__,
                "V_d_transform": V_d_transform,
                "graph_transform": graph_transform,
            }
        )
        self.d_v = d_v
        self.d_vd = d_vd
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self.d_v + self.d_vd if self.d_vd is not None else self.d_v

    def forward(self, big: BatchInteractionGraph, V_d: Tensor | None = None) -> Tensor:
        big = self.graph_transform(big)
        H = big.V
        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            H = torch.cat((H, V_d), dim=1)
        return H
