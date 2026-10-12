from pathlib import Path

import pandas as pd
import pytest

from chemprop import nn
from chemprop_contrib.polymers import PolymerDatapoint, wMPNN

_DATA_DIR = Path(__file__).parent / "data"
_POLY = pd.read_csv(_DATA_DIR / "test_polymer.csv")


@pytest.fixture
def data_dir():
    return _DATA_DIR


@pytest.fixture(params=_POLY.smiles.sample(5))
def polymer(request):
    return PolymerDatapoint.from_smi(request.param)


@pytest.fixture
def polymer_regression_data():
    smis = _POLY["smiles"].to_list()
    Y = _POLY["EA vs SHE (eV)"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture(scope="session")
def wmpnn(request):
    message_passing, agg, *act = request.param
    ffn = nn.RegressionFFN(activation=act[0] if len(act) > 0 else "RELU")
    return wMPNN(message_passing, agg, ffn, True)


@pytest.fixture(scope="session")
def regression_wmpnn_mve(request):
    agg = nn.SumAggregation()
    ffn = nn.MveFFN()

    return wMPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def regression_wmpnn_evidential(request):
    agg = nn.SumAggregation()
    ffn = nn.EvidentialFFN()

    return wMPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_wmpnn_dirichlet(request):
    agg = nn.SumAggregation()
    ffn = nn.BinaryDirichletFFN()

    return wMPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_wmpnn(request):
    agg = nn.SumAggregation()
    ffn = nn.BinaryClassificationFFN()

    return wMPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_wmpnn_multiclass(request):
    agg = nn.SumAggregation()
    ffn = nn.MulticlassClassificationFFN(n_classes=3)

    return wMPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_wmpnn_multiclass_dirichlet(request):
    agg = nn.SumAggregation()
    ffn = nn.MulticlassDirichletFFN(n_classes=3)

    return wMPNN(request.param, agg, ffn, True)
