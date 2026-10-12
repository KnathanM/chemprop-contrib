from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop_contrib.polymers import (
    PolymerDatapoint,
    PolymerDataset,
    WeightedBondMessagePassing,
    build_dataloader,
    collate_polymer_batch,
)
from chemprop_contrib.polymers.models import _weighted_mean, wMPNN


@pytest.fixture
def data(polymer_regression_data):
    smis, Y = polymer_regression_data

    return [PolymerDatapoint.from_smi(smi, y=y) for smi, y in zip(smis, Y)]


@pytest.fixture
def dataloader(data):
    dset = PolymerDataset(data)
    dset.normalize_targets()

    return DataLoader(dset, 32, collate_fn=collate_polymer_batch)


@pytest.mark.parametrize(
    "wmpnn",
    [
        (WeightedBondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.SumAggregation()),
        (WeightedBondMessagePassing(), nn.NormAggregation()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_quick(wmpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(wmpnn, dataloader, None)


@pytest.mark.parametrize(
    "wmpnn",
    [
        (WeightedBondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.MeanAggregation()),
        (WeightedBondMessagePassing(), nn.NormAggregation()),
        (WeightedBondMessagePassing(), nn.SumAggregation()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_overfit(wmpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=50,
        overfit_batches=1.00,
    )
    trainer.fit(wmpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = wmpnn(bmg)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.05


@pytest.mark.parametrize(
    "regression_wmpnn_mve",
    [WeightedBondMessagePassing(), nn.AtomMessagePassing()],
    indirect=True,
)
@pytest.mark.integration
def test_mve_quick(regression_wmpnn_mve, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_wmpnn_mve, dataloader, None)


@pytest.mark.parametrize(
    "regression_wmpnn_evidential",
    [WeightedBondMessagePassing(), nn.AtomMessagePassing()],
    indirect=True,
)
@pytest.mark.integration
def test_evidential_quick(regression_wmpnn_evidential, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_wmpnn_evidential, dataloader, None)


def test_weighted_mean_equals_mean_for_unit_weights():
    H = torch.randn(10, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    V_w = torch.ones(10)

    torch.testing.assert_close(_weighted_mean(H, batch, V_w), nn.MeanAggregation()(H, batch))


def test_weighted_mean_uses_atom_weights(data):
    """The wMPNN fingerprint with mean aggregation divides by the sum of the atom weights"""
    mp = WeightedBondMessagePassing()
    model = wMPNN(mp, nn.MeanAggregation(), nn.RegressionFFN())
    bmg, *_ = next(iter(build_dataloader(PolymerDataset(data[:4]), shuffle=False)))

    model.eval()
    with torch.no_grad():
        H_v = mp(bmg)
        n_atoms = torch.bincount(bmg.batch).unsqueeze(1)
        sum_V_w = torch.zeros(len(bmg)).index_add_(0, bmg.batch, bmg.V_w).unsqueeze(1)
        expected = nn.MeanAggregation()(H_v, bmg.batch) * n_atoms / sum_V_w
        expected = expected * bmg.degree_of_poly.unsqueeze(1)

        torch.testing.assert_close(model.fingerprint(bmg), expected)
