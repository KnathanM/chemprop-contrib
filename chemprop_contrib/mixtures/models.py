import logging
from typing import Iterable

import torch
from chemprop.data.collate import BatchMolGraph
from chemprop.models import MulticomponentMPNN
from chemprop.nn.metrics import ChempropMetric
from chemprop.nn.predictors import Predictor
from chemprop.nn.transforms import ScaleTransform
from torch import Tensor

from chemprop_contrib.mixtures.data.collate import BatchComponentMolGraph, BatchMixtureGraph
from chemprop_contrib.mixtures.nn.agg import MixtureAggregation
from chemprop_contrib.mixtures.nn.message_passing import MixtureMulticomponentMessagePassing

logger = logging.getLogger(__name__)


class MixtureMPNN(MulticomponentMPNN):
    def __init__(
        self,
        message_passing: MixtureMulticomponentMessagePassing,
        agg: MixtureAggregation,
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

    def fingerprint(
        self,
        bmgs: Iterable[Iterable[BatchMolGraph | BatchComponentMolGraph | BatchMixtureGraph | None]],
        V_ds: Iterable[Iterable[Tensor]],
        X_d: Tensor | None = None,
    ) -> Tensor:
        H_vs = self.message_passing(bmgs, V_ds)
        H = self.agg(H_vs, bmgs)
        H = self.bn(H)
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location, weights_only=False)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults")
                or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]

        if hparams["predictor"]["criterion"] is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (
                not torch.cuda.is_available() and metric.device.type != "cpu"
            ):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

        hparams["message_passing"]["blocks"] = [
            block_hparams.pop("cls")(**block_hparams)
            for block_hparams in hparams["message_passing"]["blocks"]
        ]

        graph_agg_hparams = hparams["agg"]["graph_agg"]
        hparams["agg"]["graph_agg"] = graph_agg_hparams.pop("cls")(**graph_agg_hparams)

        if hparams["agg"]["mixmp"] is not None:
            mixmp_hparams = hparams["agg"]["mixmp"]
            hparams["agg"]["mixmp"] = mixmp_hparams.pop("cls")(**mixmp_hparams)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults")
                or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]

        return submodules, state_dict, hparams
