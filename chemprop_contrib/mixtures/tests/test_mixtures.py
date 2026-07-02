from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

from lightning import pytorch as pl
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
import torch
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset, MolGraph, MulticomponentDataset
from chemprop.featurizers.molecule import ChargeFeaturizer
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.message_passing import BondMessagePassing, MulticomponentMessagePassing
from chemprop.nn.predictors import RegressionFFN
from chemprop.nn.transforms import GraphTransform, ScaleTransform, UnscaleTransform

from chemprop_contrib.mixtures.data import (
    BatchInteractionGraph,
    InteractionDatapoint,
    InteractionDataset,
    MixtureDatapoint,
    MixtureDataset,
    MixtureMolGraph,
    collate_interaction_batch,
    collate_multicomponent_with_mixture,
)
from chemprop_contrib.mixtures.featurizers import (
    CompleteInteractionGraphFeaturizer,
    EmptyVectorFeaturizer,
    HydrogenBondFeaturizer,
)
from chemprop_contrib.mixtures.models import InteractionMPNN, MixtureMPNN
from chemprop_contrib.mixtures.nn import (
    AttentiveAggregation,
    ConcatAggregation,
    DeepsetsAggregation,
    InteractionMessagePassing,
    MixtureMessagePassing,
    MolecularMessagePassing,
    NoMessagePassing,
    Set2SetAggregation,
    WeightedSumAggregation,
)

from dataclasses import fields


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def glycol():
    return Chem.MolFromSmiles("OCCO")


@pytest.fixture
def ethanol():
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def benzene():
    return Chem.MolFromSmiles("c1ccccc1")


class TestMultiHotMolInteractionFeaturizer:
    @pytest.fixture
    def feat(self):
        return HydrogenBondFeaturizer(max_hbond_num=3)

    def test_len(self, feat):
        assert len(feat) == 5

    def test_none_returns_zeros(self, feat):
        x = feat(None)
        assert x.shape == (5,)
        assert np.all(x == 0)

    def test_single_mol_uses_intra_hb(self, feat, ethanol):
        x = feat([ethanol])
        assert x[1] == 1
        assert x.sum() == 1

    def test_two_mols_uses_inter_hb(self, feat, glycol, ethanol):
        x = feat([glycol, ethanol])
        assert x[2] == 1
        assert x.sum() == 1

    def test_out_of_range_bin(self, feat):
        polyol = Chem.MolFromSmiles("OCC(O)C(O)C(O)CO")
        x = feat([polyol])
        assert x[-1] == 1

    def test_empty_list_raises(self, feat):
        with pytest.raises(ValueError):
            feat([])

    def test_inter_hb_three_mols_raises(self, feat, glycol, ethanol, benzene):
        with pytest.raises(ValueError, match="Expected 1 or 2 molecules, got 3."):
            feat([glycol, ethanol, benzene])


class TestCompleteInteractionGraphFeaturizer:
    @pytest.fixture
    def mixture_featurizer(self):
        return CompleteInteractionGraphFeaturizer.with_self_loops_and_hbonds()

    def test_shape_property(self, mixture_featurizer):
        assert mixture_featurizer.shape == (0, 5)

    def test_single_component(self, mixture_featurizer, ethanol):
        mg = mixture_featurizer([ethanol])
        # 1 mol: only the diagonal self-interaction; n_interactions = 1
        assert mg.V.shape == (1, mixture_featurizer.mol_fdim)
        assert mg.E.shape == (2, mixture_featurizer.interaction_fdim)
        np.testing.assert_array_equal(mg.edge_index, np.array([[0, 0], [0, 0]]))

    def test_two_components_edge_count(self, mixture_featurizer, glycol, ethanol):
        mg = mixture_featurizer([glycol, ethanol])
        # n_interactions = 2*(2+1)/2 = 3 (two self + one cross)
        # 2 directional edges per interaction = 6
        assert mg.E.shape[0] == 6
        assert mg.edge_index.shape == (2, 6)

    def test_three_components_edge_count(self, mixture_featurizer, glycol, ethanol, benzene):
        mg = mixture_featurizer([glycol, ethanol, benzene])
        # n_interactions = 3*4/2 = 6; directional edges = 12
        assert mg.E.shape[0] == 12

    def test_rev_edge_index_is_swap(self, mixture_featurizer, glycol, ethanol, benzene):
        mg = mixture_featurizer([glycol, ethanol, benzene])
        # rev_edge_index pairs i<->i+1 for even i
        for i in range(0, len(mg.rev_edge_index), 2):
            assert mg.rev_edge_index[i] == i + 1
            assert mg.rev_edge_index[i + 1] == i


def test_empty_vector_featurizer():
    e = EmptyVectorFeaturizer()
    assert len(e) == 0
    assert e(glycol).shape == (0,)


class TestMixtureDatapoint:
    def test_default_w_fps(self):
        d = MixtureDatapoint.from_smis(["CCO"], y=np.array([1.0]))
        np.testing.assert_array_equal(d.w_fps, np.array([1.0]))

    def test_custom_w_fp(self):
        d = MixtureDatapoint.from_smis(["CCO"], y=np.array([1.0]), w_fps=0.5)
        np.testing.assert_array_equal(d.w_fps, np.array([0.5]))

    def test_from_smis(self):
        d = MixtureDatapoint.from_smis(["O", "CCO"])
        assert len(d.mols) == 2
        assert all(isinstance(m, Chem.Mol) for m in d.mols)
        assert d.name == "O|CCO"

    def test_from_smis_custom_name(self):
        d = MixtureDatapoint.from_smis(["O", "CCO"], y=np.array([1.0]), name="custom")
        assert d.name == "custom"

    def test_nan_replaced_in_features(self):
        V_f = np.array([[1.0, np.nan], [3.0, 4.0]])
        d = MixtureDatapoint(
            mols=[Chem.MolFromSmiles("O"), Chem.MolFromSmiles("CCO")],
            y=np.array([1.0]),
            V_f=V_f,
        )
        assert not np.isnan(d.V_f).any()
        assert d.V_f[0, 1] == 0

    def test_mixture_wfps_scalar_broadcast(self, glycol, ethanol):
        dp = MixtureDatapoint([glycol, ethanol])  # default w_fps=1.0
        assert isinstance(dp.w_fps, np.ndarray)
        assert dp.w_fps.shape == (2,)
        assert np.allclose(dp.w_fps, [1.0, 1.0])
        assert len(dp) == 1

    def test_mixture_wfps_array_preserved(self, glycol, ethanol):
        dp = MixtureDatapoint([glycol, ethanol], w_fps=[0.3, 0.7])
        assert np.allclose(dp.w_fps, [0.3, 0.7])

    def test_mixture_wfps_wrong_length_raises(self, glycol, ethanol):
        with pytest.raises(ValueError):
            MixtureDatapoint([glycol, ethanol], w_fps=[0.3, 0.3, 0.4])

    def test_mixture_molecule_sizes_and_combined_mol(self, glycol, ethanol):
        dp = MixtureDatapoint([glycol, ethanol])
        assert dp.mol.GetProp("molecule_sizes") == "4,3"
        assert dp.mol.GetNumAtoms() == 7


class TestInteractionDatapoint:
    def test_length(self):
        dp = InteractionDatapoint()
        assert len(dp) == 1


# Claude Opus 4.8 tests
# --------------------------------------------------------------------------------------
# Mixture aggregations
# --------------------------------------------------------------------------------------
def test_concat_output_dim():
    agg = ConcatAggregation(fp_dim=4, max_components=2)
    assert agg.output_dim == (4 + 1) * 2


def test_concat_forward_shape_and_padding():
    agg = ConcatAggregation(fp_dim=2, max_components=2).eval()
    H = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    batch = torch.tensor([0, 0, 1])
    w = torch.tensor([0.5, 0.5, 2.0])
    out = agg(H, batch, w)

    assert out.shape == (2, 6)  # K*d (=4) + K weights (=2)
    # mixture 0: [h0, h1, w0, w1]
    assert torch.allclose(out[0], torch.tensor([1.0, 1.0, 2.0, 2.0, 0.5, 0.5]))
    # mixture 1: single component -> second slot (features and weight) is zero-padded
    assert torch.allclose(out[1], torch.tensor([3.0, 3.0, 0.0, 0.0, 2.0, 0.0]))


def test_concat_exceeds_max_raises():
    agg = ConcatAggregation(fp_dim=2, max_components=1)
    H = torch.randn(2, 2)
    batch = torch.tensor([0, 0])  # 2 components but max is 1
    w = torch.ones(2)
    with pytest.raises(ValueError):
        agg(H, batch, w)


def test_weighted_sum_correctness():
    agg = WeightedSumAggregation(fp_dim=2)
    H = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    batch = torch.tensor([0, 0, 1])
    w = torch.tensor([0.5, 0.5, 2.0])
    out = agg(H, batch, w)
    assert torch.allclose(out, torch.tensor([[1.5, 1.5], [6.0, 6.0]]))


def test_deepsets_output_dim_and_zero_hidden_layers():
    assert DeepsetsAggregation(fp_dim=4).output_dim == 4
    mlp = DeepsetsAggregation._make_mlp(4, n_hidden_layers=0)
    assert len(mlp) == 1
    assert isinstance(mlp[0], torch.nn.Linear)
    assert mlp[0].in_features == 4 and mlp[0].out_features == 4


def test_attentive_single_component_equals_weighted():
    # With one component per mixture, softmax over a single logit => alpha == 1,
    # so the output must equal w * H regardless of the learned attention weights.
    agg = AttentiveAggregation(fp_dim=2)
    H = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = torch.tensor([0, 1])
    w = torch.tensor([2.0, 3.0])
    out = agg(H, batch, w)
    assert torch.allclose(out, torch.tensor([[2.0, 4.0], [9.0, 12.0]]))


def test_set2set_output_dim_and_shape():
    agg = Set2SetAggregation(fp_dim=3)
    assert agg.output_dim == 6
    H = torch.randn(3, 3)
    batch = torch.tensor([0, 0, 1])
    w = torch.ones(3)
    assert agg(H, batch, w).shape == (2, 6)


# --------------------------------------------------------------------------------------
# Message passing
# --------------------------------------------------------------------------------------
def test_mixture_mp_output_dim():
    assert MixtureMessagePassing(d_v=6, d_h=5).output_dim == 5
    assert MixtureMessagePassing(d_v=6, d_h=5, d_vd=3).output_dim == 8


def test_mixture_mp_message_aggregates_neighbors():
    # Graph 0<->1: message(dst) sums the hidden states of its src neighbors.
    mp = MixtureMessagePassing(d_v=2, d_h=2)
    big = SimpleNamespace(edge_index=torch.tensor([[0, 1], [1, 0]]))
    H = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    M = mp.message(H, big)
    assert torch.allclose(M, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))


def test_no_message_passing_forward_and_output_dim():
    nmp = NoMessagePassing(d_v=4)
    assert nmp.output_dim == 4
    big = SimpleNamespace(V=torch.arange(8.0).reshape(2, 4))
    assert torch.allclose(nmp(big), big.V)

    nmp2 = NoMessagePassing(d_v=4, d_vd=2)
    assert nmp2.output_dim == 6
    big2 = SimpleNamespace(V=torch.randn(3, 4))
    out = nmp2(big2, torch.randn(3, 2))
    assert out.shape == (3, 6)


# --------------------------------------------------------------------------------------
# MolGraph wrappers
# --------------------------------------------------------------------------------------
def test_mixture_molgraph_from_molgraph():
    V = np.zeros((3, 2))
    E = np.zeros((0, 1))
    edge_index = np.zeros((2, 0), dtype=int)
    rev = np.zeros((0,), dtype=int)
    mg = MolGraph(V, E, edge_index, rev)

    sizes = np.array([2, 1])
    w = np.array([0.5, 0.5])
    mmg = MixtureMolGraph.from_molgraph(mg, sizes, w)

    assert np.array_equal(mmg.V, V)
    assert np.array_equal(mmg.edge_index, edge_index)
    assert np.array_equal(mmg.molecule_sizes, sizes)
    assert np.array_equal(mmg.w_fps, w)


# Integration tests by human, some formatting by Opus.

N_EXTRA_DATAPOINT_DESCRIPTORS = 1
N_EXTRA_ATOM_FEATURES = 2
N_EXTRA_BOND_FEATURES = 3
N_EXTRA_ATOM_DESCRIPTORS = 4
N_EXTRA_MOLECULE_FEATURES = 5
N_EXTRA_INTERACTION_FEATURES = 6
N_EXTRA_MOLECULE_DESCRIPTORS = 7

SOLVENT_MP_OUTPUT_DIM = 300 + N_EXTRA_ATOM_DESCRIPTORS

DATA_CSV = Path(__file__).parent / "test_data.csv"

MODEL_CONFIGURATIONS = [
    ("c_agg", None),
    ("c_agg_rand", "mixmp"),
    ("c_agg_rand_pad", None),
    ("c_agg_rand_order", None),
    ("ws_agg", None),
    ("ds_agg", "imp"),
    ("ds_agg_custom", None),
    ("a_agg", "molmp"),
    ("s2s_agg", None),
    ("s2s_agg_custom", "nomp"),
]


def load_data():
    df = pd.read_csv(DATA_CSV).head(18)
    ys = df[["Gsolv (kcal/mol)"]].to_numpy(dtype=float)
    solute_mols = df["inchi_solute"].map(Chem.MolFromInchi)
    solvent_mols, w_fpss = [], []

    for inchi1, inchi2, frac in zip(
        df["inchi_solvent1"], df["inchi_solvent2"], df["frac_solvent1"]
    ):
        if frac == 0:
            solvent_mols.append([Chem.MolFromInchi(inchi2)])
            w_fpss.append([1.0])
        elif frac == 1:
            solvent_mols.append([Chem.MolFromInchi(inchi1)])
            w_fpss.append([1.0])
        else:
            solvent_mols.append([Chem.MolFromInchi(inchi1), Chem.MolFromInchi(inchi2)])
            w_fpss.append([frac, 1 - frac])

    return ys, solute_mols, solvent_mols, w_fpss


def make_datapoints():
    ys, solute_mols, solvent_mols, w_fpss = load_data()

    extra_datapoint_descriptors = [
        np.random.rand(N_EXTRA_DATAPOINT_DESCRIPTORS) * 2 for _ in solute_mols
    ]
    extra_atom_features = [
        np.random.rand(sum(m.GetNumAtoms() for m in mols), N_EXTRA_ATOM_FEATURES) * 3
        for mols in solvent_mols
    ]
    extra_bond_features = [
        np.random.rand(sum(m.GetNumBonds() for m in mols), N_EXTRA_BOND_FEATURES) * 4
        for mols in solvent_mols
    ]
    extra_atom_descriptors = [
        np.random.rand(sum(m.GetNumAtoms() for m in mols), N_EXTRA_ATOM_DESCRIPTORS) * 5 + 5
        for mols in solvent_mols
    ]
    extra_molecule_features = [
        np.random.rand((len(mols) + 1), N_EXTRA_MOLECULE_FEATURES) * 6 for mols in solvent_mols
    ]
    extra_interactions_features = [
        np.random.rand(((len(mols) + 1) * (len(mols) + 2)) // 2, N_EXTRA_INTERACTION_FEATURES) * 7
        + 7
        for mols in solvent_mols
    ]
    extra_molecule_descriptors = [
        np.random.rand((len(mols) + 1), N_EXTRA_MOLECULE_DESCRIPTORS) for mols in solvent_mols
    ]

    dp_solutes = [
        MoleculeDatapoint(mol, y=y, x_d=x_d)
        for mol, y, x_d in zip(solute_mols, ys, extra_datapoint_descriptors)
    ]
    dp_solvents = [
        MixtureDatapoint(mols, w_fps=w_fps, y=y, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d)
        for mols, w_fps, y, x_d, V_f, E_f, V_d in zip(
            solvent_mols,
            w_fpss,
            ys,
            extra_datapoint_descriptors,
            extra_atom_features,
            extra_bond_features,
            extra_atom_descriptors,
        )
    ]
    dp_interaction = [
        InteractionDatapoint(y=y, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d)
        for y, x_d, V_f, E_f, V_d in zip(
            ys,
            extra_datapoint_descriptors,
            extra_molecule_features,
            extra_interactions_features,
            extra_molecule_descriptors,
        )
    ]

    train_dp = (dp_solutes[0:6], dp_solvents[0:6], dp_interaction[0:6])
    val_dp = (dp_solutes[6:12], dp_solvents[6:12], dp_interaction[6:12])
    test_dp = (dp_solutes[12:18], dp_solvents[12:18], dp_interaction[12:18])
    return train_dp, val_dp, test_dp


def make_datasets(train_dp, val_dp, test_dp):
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=N_EXTRA_ATOM_FEATURES, extra_bond_fdim=N_EXTRA_BOND_FEATURES
    )
    train_ds = [MoleculeDataset(train_dp[0]), MixtureDataset(train_dp[1], featurizer)]
    val_ds = [MoleculeDataset(val_dp[0]), MixtureDataset(val_dp[1], featurizer)]
    test_ds = [MoleculeDataset(test_dp[0]), MixtureDataset(test_dp[1], featurizer)]
    return train_ds, val_ds, test_ds


def make_mc_datasets(train_ds, val_ds, test_ds):
    train_mc = MulticomponentDataset(train_ds)
    val_mc = MulticomponentDataset(val_ds)
    test_mc = MulticomponentDataset(test_ds)

    scaler = train_mc.normalize_targets()
    val_mc.normalize_targets(scaler)
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    scalers = train_mc.normalize_inputs("X_d")
    val_mc.normalize_inputs("X_d", scalers)
    X_d_transform = ScaleTransform.from_standard_scaler(scalers[1])

    featurizer = train_ds[1].featurizer
    scalers = train_mc.normalize_inputs("V_f")
    val_mc.normalize_inputs("V_f", scalers)
    V_f_transform = ScaleTransform.from_standard_scaler(
        scalers[1], pad=featurizer.atom_fdim - featurizer.extra_atom_fdim
    )

    scalers = train_mc.normalize_inputs("E_f")
    val_mc.normalize_inputs("E_f", scalers)
    E_f_transform = ScaleTransform.from_standard_scaler(
        scalers[1], pad=featurizer.bond_fdim - featurizer.extra_bond_fdim
    )
    graph_transform = GraphTransform(V_f_transform, E_f_transform)

    scalers = train_mc.normalize_inputs("V_d")
    val_mc.normalize_inputs("V_d", scalers)
    V_d_transform = ScaleTransform.from_standard_scaler(scalers[1])

    transforms = [output_transform, X_d_transform, graph_transform, V_d_transform]
    return train_mc, val_mc, test_mc, transforms


def make_interaction_datasets(train_ds, val_ds, test_ds, train_dp, val_dp, test_dp):
    featurizer = CompleteInteractionGraphFeaturizer.with_self_loops_and_hbonds(
        extra_mol_fdim=N_EXTRA_MOLECULE_FEATURES,
        extra_interaction_fdim=N_EXTRA_INTERACTION_FEATURES,
    )
    train_ids = InteractionDataset(
        subgraph_datasets=train_ds, data=train_dp[2], featurizer=featurizer
    )
    val_ids = InteractionDataset(subgraph_datasets=val_ds, data=val_dp[2], featurizer=featurizer)
    test_ids = InteractionDataset(subgraph_datasets=test_ds, data=test_dp[2], featurizer=featurizer)

    scaler = train_ids.normalize_targets()
    val_ids.normalize_targets(scaler)
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    scaler = train_ids.normalize_inputs("X_d")
    val_ids.normalize_inputs("X_d", scaler)
    X_d_transform = ScaleTransform.from_standard_scaler(scaler)

    scaler = train_ids.normalize_inputs("V_f")
    val_ids.normalize_inputs("V_f", scaler)
    mixture_V_f_transform = ScaleTransform.from_standard_scaler(
        scaler, pad=(SOLVENT_MP_OUTPUT_DIM + featurizer.mol_fdim - featurizer.extra_mol_fdim)
    )

    scaler = train_ids.normalize_inputs("E_f")
    val_ids.normalize_inputs("E_f", scaler)
    mixture_E_f_transform = ScaleTransform.from_standard_scaler(
        scaler, pad=featurizer.interaction_fdim - featurizer.extra_interaction_fdim
    )
    mixture_graph_transform = GraphTransform(mixture_V_f_transform, mixture_E_f_transform)

    scaler = train_ids.normalize_inputs("V_d")
    val_ids.normalize_inputs("V_d", scaler)
    mixture_V_d_transform = ScaleTransform.from_standard_scaler(scaler)

    featurizer = train_ds[1].featurizer
    scalers = train_ids.normalize_inputs_subgraph_datasets("V_f")
    val_ids.normalize_inputs_subgraph_datasets("V_f", scalers)
    V_f_transform = ScaleTransform.from_standard_scaler(
        scalers[1], pad=featurizer.atom_fdim - featurizer.extra_atom_fdim
    )

    scalers = train_ids.normalize_inputs_subgraph_datasets("E_f")
    val_ids.normalize_inputs_subgraph_datasets("E_f", scalers)
    E_f_transform = ScaleTransform.from_standard_scaler(
        scalers[1], pad=featurizer.bond_fdim - featurizer.extra_bond_fdim
    )
    graph_transform = GraphTransform(V_f_transform, E_f_transform)

    scalers = train_ids.normalize_inputs_subgraph_datasets("V_d")
    val_ids.normalize_inputs_subgraph_datasets("V_d", scalers)
    V_d_transform = ScaleTransform.from_standard_scaler(scalers[1])

    transforms = [
        output_transform,
        X_d_transform,
        graph_transform,
        V_d_transform,
        mixture_graph_transform,
        mixture_V_d_transform,
    ]
    return train_ids, val_ids, test_ids, transforms


def make_dataloader(dset, batch_size=3, shuffle=False):
    if isinstance(dset, MulticomponentDataset):
        collate_fn = collate_multicomponent_with_mixture
    elif isinstance(dset, InteractionDataset):
        collate_fn = collate_interaction_batch
    return DataLoader(dataset=dset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


def make_mixture_agg(which_m_agg, fp_dim):
    match which_m_agg:
        case "c_agg":
            return ConcatAggregation(fp_dim, 2)
        case "c_agg_rand":
            return ConcatAggregation(
                fp_dim,
                max_components=2,
                randomize_pad_position=True,
                randomize_component_order=True,
            )
        case "c_agg_rand_pad":
            return ConcatAggregation(
                fp_dim,
                max_components=2,
                randomize_pad_position=True,
                randomize_component_order=False,
            )
        case "c_agg_rand_order":
            return ConcatAggregation(
                fp_dim,
                max_components=2,
                randomize_pad_position=False,
                randomize_component_order=True,
            )
        case "ws_agg":
            return WeightedSumAggregation(fp_dim)
        case "ds_agg":
            return DeepsetsAggregation(fp_dim)
        case "ds_agg_custom":
            return DeepsetsAggregation(
                fp_dim,
                hidden_dim=100,
                n_hidden_layers=1,
                bias=True,
                activation=torch.nn.LeakyReLU(),
            )
        case "a_agg":
            return AttentiveAggregation(fp_dim)
        case "s2s_agg":
            return Set2SetAggregation(fp_dim)
        case "s2s_agg_custom":
            return Set2SetAggregation(fp_dim, processing_steps=2)


def make_MixtureMPNN(train_dset, transforms, which_m_agg):
    output_transform, X_d_transform, graph_transform, V_d_transform, *_ = transforms
    solute_mp = BondMessagePassing()
    solvent_mp = BondMessagePassing(
        d_v=train_dset.datasets[1].featurizer.atom_fdim,
        d_e=train_dset.datasets[1].featurizer.bond_fdim,
        graph_transform=graph_transform,
        V_d_transform=V_d_transform,
        d_vd=train_dset.d_vd,
    )
    mcmp = MulticomponentMessagePassing(blocks=[solute_mp, solvent_mp], n_components=2)
    agg = MeanAggregation()
    mixture_agg = make_mixture_agg(which_m_agg, solvent_mp.output_dim)
    ffn = RegressionFFN(
        input_dim=(solute_mp.output_dim + mixture_agg.output_dim + train_dset.d_xd),
        n_tasks=train_dset.datasets[0].Y.shape[1],
        output_transform=output_transform,
    )
    return MixtureMPNN(mcmp, agg, mixture_agg, ffn, X_d_transform=X_d_transform)


def make_interaction_mp(which_i_mp, dset, graph_transform, V_d_transform):
    mp_args = {
        "d_v": dset.featurizer.mol_fdim + SOLVENT_MP_OUTPUT_DIM,
        "d_e": dset.featurizer.interaction_fdim,
        "d_h": 100,
        "V_d_transform": V_d_transform,
        "graph_transform": graph_transform,
        "d_vd": dset.d_vd,
    }
    match which_i_mp:
        case "mixmp":
            return MixtureMessagePassing(**mp_args, activation=torch.nn.LeakyReLU())
        case "imp":
            return InteractionMessagePassing(**mp_args)
        case "molmp":
            return MolecularMessagePassing(**mp_args)
        case "nomp":
            return NoMessagePassing(**mp_args)


def make_InteractionMPNN(train_dset, transforms, which_m_agg, which_i_mp):
    (
        output_transform,
        X_d_transform,
        graph_transform,
        V_d_transform,
        mixture_graph_transform,
        mixture_V_d_transform,
    ) = transforms

    solvent_mp = BondMessagePassing(
        d_v=train_dset.subgraph_datasets[1].featurizer.atom_fdim,
        d_e=train_dset.subgraph_datasets[1].featurizer.bond_fdim,
        graph_transform=graph_transform,
        V_d_transform=V_d_transform,
        d_vd=train_dset.subgraph_datasets[1].d_vd,
    )
    solute_mp = BondMessagePassing(d_h=solvent_mp.output_dim)
    mcmp = MulticomponentMessagePassing(blocks=[solute_mp, solvent_mp], n_components=2)
    agg = MeanAggregation()
    interaction_mp = make_interaction_mp(
        which_i_mp, train_dset, mixture_graph_transform, mixture_V_d_transform
    )
    mixture_agg = make_mixture_agg(which_m_agg, interaction_mp.output_dim)
    ffn = RegressionFFN(
        input_dim=(interaction_mp.output_dim + mixture_agg.output_dim + train_dset.d_xd),
        n_tasks=train_dset.Y.shape[1],
        output_transform=output_transform,
    )
    return InteractionMPNN(mcmp, agg, interaction_mp, mixture_agg, ffn, X_d_transform=X_d_transform)


def make_trainer(max_epochs=2, **kwargs):
    return pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        accelerator="cpu",
        **kwargs,
    )


def build_model_and_loaders(which_m_agg, which_i_mp, batch_size=3):
    train_dp, val_dp, test_dp = make_datapoints()
    train_ds, val_ds, test_ds = make_datasets(train_dp, val_dp, test_dp)

    if which_i_mp is not None:
        train_dset, val_dset, test_dset, transforms = make_interaction_datasets(
            train_ds, val_ds, test_ds, train_dp, val_dp, test_dp
        )
        model = make_InteractionMPNN(train_dset, transforms, which_m_agg, which_i_mp)
    else:
        train_dset, val_dset, test_dset, transforms = make_mc_datasets(train_ds, val_ds, test_ds)
        model = make_MixtureMPNN(train_dset, transforms, which_m_agg)

    train_loader = make_dataloader(train_dset, batch_size=batch_size)
    val_loader = make_dataloader(val_dset, batch_size=batch_size)
    test_loader = make_dataloader(test_dset, batch_size=batch_size)
    return model, train_loader, val_loader, test_loader


@pytest.mark.parametrize("which_m_agg,which_i_mp", MODEL_CONFIGURATIONS)
def test_fit_test_passthrough(which_m_agg, which_i_mp):
    model, train_loader, val_loader, test_loader = build_model_and_loaders(which_m_agg, which_i_mp)

    trainer = make_trainer(max_epochs=2)
    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(model, test_loader)

    mse = results[0]["test/mse"]
    assert np.isfinite(mse), f"Non-finite test/mse for ({which_m_agg}, {which_i_mp})"


@pytest.mark.parametrize("which_m_agg,which_i_mp", MODEL_CONFIGURATIONS)
def test_overfit_mixture_agg_interaction(which_m_agg, which_i_mp):
    model, train_loader, _, _ = build_model_and_loaders(which_m_agg, which_i_mp, batch_size=6)

    trainer = make_trainer(max_epochs=50)
    trainer.fit(model, train_loader)

    train_loss = float(trainer.callback_metrics["train_loss_epoch"])
    assert (
        train_loss < 0.005
    ), f"({which_m_agg}, {which_i_mp}) failed to overfit: train_loss={train_loss:.4f}"

    test_dset = train_loader.dataset
    test_dset.reset()
    if isinstance(test_dset, InteractionDataset):
        test_dset._init_cache()
        [dset._init_cache() for dset in test_dset.subgraph_datasets]
    elif isinstance(test_dset, MulticomponentDataset):
        [dset._init_cache() for dset in test_dset.datasets]
    test_loader = make_dataloader(test_dset)
    results = trainer.test(model, test_loader)
    assert results[0]["test/mse"] < 0.01


def test_charge_featurizer_interaction_overfit():
    ys, solute_mols, solvent_mols, w_fpss = load_data()

    dp_solutes = [MoleculeDatapoint(m) for m in solute_mols]
    dp_solvents = [MixtureDatapoint(m, w_fps=w) for m, w in zip(solvent_mols, w_fpss)]

    mol_feats = [np.random.rand(len(m) + 1, N_EXTRA_MOLECULE_FEATURES) * 6 for m in solvent_mols]
    inter_feats = [
        np.random.rand((len(m) * (len(m) + 1)) // 2, N_EXTRA_INTERACTION_FEATURES)
        for m in solvent_mols
    ]
    dp_inter = [
        InteractionDatapoint(y=y, V_f=v, E_f=e) for y, v, e in zip(ys, mol_feats, inter_feats)
    ]

    train_dp = (dp_solutes[:6], dp_solvents[:6], dp_inter[:6])
    train_ds = [MoleculeDataset(train_dp[0]), MixtureDataset(train_dp[1])]

    featurizer = CompleteInteractionGraphFeaturizer(
        mol_featurizer=ChargeFeaturizer(),
        extra_mol_fdim=N_EXTRA_MOLECULE_FEATURES,
        extra_interaction_fdim=N_EXTRA_INTERACTION_FEATURES,
    )
    train_ids = InteractionDataset(
        subgraph_datasets=train_ds, data=train_dp[2], featurizer=featurizer
    )

    scaler = train_ids.normalize_targets()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    scaler = train_ids.normalize_inputs("V_f")
    mixture_V_f_transform = ScaleTransform.from_standard_scaler(
        scaler, pad=(300 + featurizer.mol_fdim - featurizer.extra_mol_fdim)
    )
    scaler = train_ids.normalize_inputs("E_f")
    mixture_E_f_transform = ScaleTransform.from_standard_scaler(
        scaler, pad=featurizer.interaction_fdim - featurizer.extra_interaction_fdim
    )
    mixture_graph_transform = GraphTransform(mixture_V_f_transform, mixture_E_f_transform)

    train_loader = make_dataloader(train_ids)

    mcmp = MulticomponentMessagePassing(
        blocks=[BondMessagePassing(), BondMessagePassing()], n_components=2
    )
    interaction_mp = InteractionMessagePassing(
        d_v=featurizer.mol_fdim + 300,
        d_e=featurizer.interaction_fdim,
        graph_transform=mixture_graph_transform,
    )
    mixture_agg = WeightedSumAggregation(interaction_mp.output_dim)
    ffn = RegressionFFN(
        input_dim=(interaction_mp.output_dim + mixture_agg.output_dim),
        n_tasks=train_ids.Y.shape[1],
        output_transform=output_transform,
    )
    model = InteractionMPNN(mcmp, MeanAggregation(), interaction_mp, mixture_agg, ffn)

    trainer = make_trainer(max_epochs=30)
    trainer.fit(model, train_loader)

    train_loss = float(trainer.callback_metrics["train_loss_epoch"])
    assert train_loss < 0.005

    test_dset = train_loader.dataset
    test_dset.reset()
    if isinstance(test_dset, InteractionDataset):
        test_dset._init_cache()
    elif isinstance(test_dset, MulticomponentDataset):
        [dset._init_cache() for dset in test_dset.datasets]
    test_loader = make_dataloader(test_dset)
    results = trainer.test(model, test_loader)
    assert results[0]["test/mse"] < 0.01


def test_interact_only_mixture_overfit():
    ys, solute_mols, solvent_mols, w_fpss = load_data()

    dp_solutes = [MoleculeDatapoint(m) for m in solute_mols]
    dp_solvents = [MixtureDatapoint(m, w_fps=w) for m, w in zip(solvent_mols, w_fpss)]
    extra_molecule_descriptors = [
        np.random.rand(len(mols), N_EXTRA_MOLECULE_DESCRIPTORS) for mols in solvent_mols
    ]
    dp_inter = [InteractionDatapoint(y=y, V_d=V_d) for y, V_d in zip(ys, extra_molecule_descriptors)]

    train_dp = (dp_solutes[:6], dp_solvents[:6], dp_inter[:6])
    train_ds = [MoleculeDataset(train_dp[0]), MixtureDataset(train_dp[1])]

    featurizer = CompleteInteractionGraphFeaturizer(interact_only_mixture=True)
    train_ids = InteractionDataset(
        subgraph_datasets=train_ds, data=train_dp[2], featurizer=featurizer
    )

    scaler = train_ids.normalize_targets()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    train_loader = make_dataloader(train_ids)

    mcmp = MulticomponentMessagePassing(
        blocks=[BondMessagePassing(), BondMessagePassing()], n_components=2
    )
    interaction_mp = InteractionMessagePassing(
        d_v=featurizer.mol_fdim + 300,
        d_e=featurizer.interaction_fdim,
        d_vd=train_ids.d_vd,
    )
    mixture_agg = WeightedSumAggregation(interaction_mp.output_dim)
    ffn = RegressionFFN(
        input_dim=(mcmp.blocks[0].output_dim + mixture_agg.output_dim),
        n_tasks=train_ids.Y.shape[1],
        output_transform=output_transform,
    )
    model = InteractionMPNN(
        mcmp, MeanAggregation(), interaction_mp, mixture_agg, ffn, interact_only_mixture=True
    )

    trainer = make_trainer(max_epochs=30)
    trainer.fit(model, train_loader)

    train_loss = float(trainer.callback_metrics["train_loss_epoch"])
    assert train_loss < 0.005

    test_dset = train_loader.dataset
    test_dset.reset()
    if isinstance(test_dset, InteractionDataset):
        test_dset._init_cache()
    elif isinstance(test_dset, MulticomponentDataset):
        [dset._init_cache() for dset in test_dset.datasets]
    test_loader = make_dataloader(test_dset)
    results = trainer.test(model, test_loader)
    assert results[0]["test/mse"] < 0.01


@pytest.mark.parametrize("which_m_agg,which_i_mp", MODEL_CONFIGURATIONS)
def test_batch_size_invariance(which_m_agg, which_i_mp):
    train_dp, val_dp, test_dp = make_datapoints()
    train_ds, val_ds, test_ds = make_datasets(train_dp, val_dp, test_dp)

    if which_i_mp is not None:
        train_dset, val_dset, test_dset, transforms = make_interaction_datasets(
            train_ds, val_ds, test_ds, train_dp, val_dp, test_dp
        )
        model = make_InteractionMPNN(train_dset, transforms, which_m_agg, which_i_mp)
    else:
        train_dset, val_dset, test_dset, transforms = make_mc_datasets(train_ds, val_ds, test_ds)
        model = make_MixtureMPNN(train_dset, transforms, which_m_agg)

    model.eval()
    preds = {}
    for bs in (1, 2, 3, 6):
        loader = make_dataloader(test_dset, batch_size=bs)
        with torch.inference_mode():
            preds[bs] = torch.cat([model.predict_step(b, 0) for b in loader])
    for bs in (1, 2, 3):
        assert torch.allclose(preds[bs], preds[6], atol=1e-5), \
            f"predictions depend on batch size at bs={bs}"


def copy_bmg(bmg):
    if isinstance(bmg, list):
        return [copy.copy(_bmg) for _bmg in bmg]
    if isinstance(bmg, BatchInteractionGraph):
        bmg_copy = copy.copy(bmg)
        bmg_copy.sub_bmgs = copy_bmg(bmg.sub_bmgs)
        return bmg_copy


@pytest.mark.parametrize("which_m_agg,which_i_mp", MODEL_CONFIGURATIONS)
def test_save_load_roundtrip(tmp_path, which_m_agg, which_i_mp):
    model, train_loader, _, test_loader = build_model_and_loaders(which_m_agg, which_i_mp)
    trainer = make_trainer(max_epochs=1)
    trainer.fit(model, train_loader)

    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)
    loaded = type(model).load_from_checkpoint(ckpt)

    batch = next(iter(test_loader))
    bmg, V_d, X_d, *_ = batch
    model.eval()
    loaded.eval()

    with torch.no_grad():
        bmg_copy = copy_bmg(bmg)
        out_a = model(bmg_copy, V_d, X_d)
        bmg_copy = copy_bmg(bmg)
        out_b = loaded(bmg_copy, V_d, X_d)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_interaction_dataset_requires_subgraph_datasets():
    train_dp, val_dp, test_dp = make_datapoints()
    train_ds, val_ds, test_ds = make_datasets(train_dp, val_dp, test_dp)

    with pytest.raises(
        ValueError,
        match="When using `InteractionDataset`, you must supply subgraph_datasets!",
    ):
        InteractionDataset(subgraph_datasets=None, data=train_dp[2])


def test_interaction_dataset_requires_same_length():
    train_dp, val_dp, test_dp = make_datapoints()
    train_ds, val_ds, test_ds = make_datasets(train_dp, val_dp, test_dp)

    mismatched_data = train_dp[2][:-1]
    with pytest.raises(ValueError, match="Datasets must have all same length!"):
        InteractionDataset(subgraph_datasets=train_ds, data=mismatched_data)


def test_interaction_dataset_smiles():
    train_dp, val_dp, test_dp = make_datapoints()
    train_ds, val_ds, test_ds = make_datasets(train_dp, val_dp, test_dp)

    ds = InteractionDataset(train_ds, data=train_dp[2])

    expected: list[tuple[str, ...]] = [
        ("C=CC(C)=CCC=C(C)C", "CC(=O)O.c1ccccc1"),
        ("CC1(C)CCCC2(C)C1CCCC21CCCO1", "CC(C)=O"),
        ("ClC(Cl)(Cl)Br", "CC(C)O"),
        ("CC(C(C)C(F)(F)F)C(F)(F)F", "CC(C)=O.CCC(C)=O"),
        ("FCC(OC(F)(F)F)C(F)(F)F", "CC(=O)O.CC(C)CC(C)(C)C"),
        ("CCO[Si](Cl)(OCC)OCC", "CC(=O)O.ClC(Cl)Cl"),
    ]
    assert ds.smiles == expected
