from torch import nn

from chemprop.nn.transforms import ScaleTransform

from chemprop_contrib.mixtures.data.collate import BatchComponentMolGraph


class GraphTransform(nn.Module):
    def __init__(self, V_transform: ScaleTransform, E_transform: ScaleTransform, G_transform: ScaleTransform):
        super().__init__()

        self.V_transform = V_transform
        self.E_transform = E_transform
        self.G_transform = G_transform

    def forward(self, bmg: BatchComponentMolGraph) -> BatchComponentMolGraph:
        if self.training:
            return bmg

        bmg.V = self.V_transform(bmg.V)
        bmg.E = self.E_transform(bmg.E)
        bmg.G = self.G_transform(bmg.G)

        return bmg