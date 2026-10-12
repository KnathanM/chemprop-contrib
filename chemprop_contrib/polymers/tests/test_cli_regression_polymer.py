"""This tests the CLI functionality of training and predicting a regression model on a single polymer.
"""

from argparse import ArgumentError
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from chemprop_contrib.polymers import PolymerDatapoint, PolymerDataset, build_dataloader, wMPNN
from chemprop_contrib.polymers.cli import patches
from chemprop_contrib.polymers.cli.main import PolymerTrainSubcommand as TrainSubcommand
from chemprop_contrib.polymers.cli.main import main

pytestmark = pytest.mark.CLI

_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def train_dir(monkeypatch, tmp_path):
    # keep runs without --save-dir out of the working directory
    monkeypatch.setattr("chemprop.cli.train.CHEMPROP_TRAIN_DIR", tmp_path / "chemprop_training")


@pytest.fixture
def data_path(tmp_path):
    # predict and fingerprint write their outputs next to the input file by default
    input_path = tmp_path / "dataset-poly_chemprop_small.csv"
    shutil.copy(_DATA_DIR / "dataset-poly_chemprop_small.csv", input_path)
    return (str(input_path),)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """The original PR stored a pretrained checkpoint, which can't be loaded here, so a model is
    trained once per session from the same config with the same target."""
    save_dir = tmp_path_factory.mktemp("polymer_model")
    args = [
        "chemprop-polymer",
        "train",
        "--config-path",
        str(_DATA_DIR / "config.toml"),
        "--epochs",
        "3",
        "--save-dir",
        str(save_dir),
        "--target-columns",
        "EA vs SHE (eV)",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with pytest.MonkeyPatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    return str(save_dir / "model_0" / "checkpoints" / "last.ckpt")


@pytest.fixture
def config_path():
    return str(_DATA_DIR / "config.toml")


def test_train_quick(monkeypatch, data_path):
    input_path, *_ = data_path

    args = [
        "chemprop-polymer",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--show-individual-scores",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_config(monkeypatch, config_path, tmp_path):
    args = [
        "chemprop-polymer",
        "train",
        "--config-path",
        config_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    new_config_path = tmp_path / "config.toml"
    parser = TrainSubcommand.parser

    new_args = parser.parse_args(["--config-path", str(new_config_path)])
    old_args = parser.parse_args(["--config-path", str(config_path)])

    for key, value in old_args.__dict__.items():
        if key not in ["config_path", "output_dir", "epochs"]:
            assert getattr(new_args, key) == value

    assert new_args.epochs == 3


def test_predict_quick(monkeypatch, data_path, model_path):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize("ffn_block_index", ["0", "1"])
def test_fingerprint_quick(monkeypatch, data_path, model_path, ffn_block_index):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "fingerprint",
        "-i",
        input_path,
        "--model-path",
        model_path,
        "--ffn-block-index",
        ffn_block_index,
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_output_structure(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()
    assert (tmp_path / "model_0" / "test_predictions.csv").exists()


def test_train_output_structure_replicate_ensemble(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--split-type",
        "random",
        "--num-replicates",
        "3",
        "--ensemble-size",
        "2",
        "--metrics",
        "mse",
        "rmse",
        "--molecule-featurizers",
        "rdkit_2d",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "replicate_2" / "model_1" / "best.pt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "replicate_2" / "train_smiles.csv").exists()


def test_predict_output_structure(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        model_path,
        "--polymer-columns",
        "poly_chemprop_input",
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "preds_individual.csv").exists()


def test_train_outputs(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path

    args = [
        "chemprop-polymer",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = wMPNN.load_from_checkpoint(checkpoint_path)
    assert model is not None


def test_checkpoint_model(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--checkpoint",
        model_path,
        "--polymer-columns",
        "poly_chemprop_input",
        "--target-columns",
        "EA vs SHE (eV)",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = wMPNN.load_from_checkpoint(checkpoint_path)
    assert model is not None


# Tests for chemprop-contrib's CLI wrapper, not part of the original PR


def test_predict_matches_python(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        "--polymer-columns",
        "poly_chemprop_input",
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    cli_preds = pd.read_csv(tmp_path / "preds.csv")["pred_0"].to_numpy()

    model = wMPNN.load_from_file(model_path, map_location="cpu")
    model.eval()
    smis = pd.read_csv(input_path)["poly_chemprop_input"]
    dset = PolymerDataset([PolymerDatapoint.from_smi(smi) for smi in smis])
    with torch.no_grad():
        preds = [model(bmg, V_d, X_d) for bmg, V_d, X_d, *_ in build_dataloader(dset, shuffle=False)]
    preds = torch.cat(preds).squeeze(1).numpy()

    np.testing.assert_allclose(cli_preds, preds, rtol=1e-5, atol=1e-5)


def test_patches_restored(monkeypatch, data_path, tmp_path):
    def current():
        return [getattr(module, name) for module, name, _ in patches._targets()]

    # importing chemprop_contrib.polymers does not patch chemprop
    originals = current()
    assert all(old is not new for old, (*_, new) in zip(originals, patches._targets()))

    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "predict",
        "-i",
        input_path,
        "--model-path",
        str(tmp_path / "missing.pt"),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(FileNotFoundError):
            main()

    assert current() == originals


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--reaction-columns", "poly_chemprop_input"],
        ["--mol-target-columns", "EA vs SHE (eV)"],
        ["--use-cuikmolmaker-featurization"],
        ["--from-foundation", "CHEMELEON"],
        ["--polymer-columns", "poly_chemprop_input", "poly_chemprop_input"],
    ],
)
def test_unsupported_args(monkeypatch, data_path, extra_args):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--polymer-columns",
        "poly_chemprop_input",
        *extra_args,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError):
            main()
