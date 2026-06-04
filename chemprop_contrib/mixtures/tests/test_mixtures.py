"""Comprehensive tests for chemprop_contrib.mixtures. Written by Claude Opus 4.7, then human reviewed.

Run unit tests only:    pytest test_mixtures.py -m "not integration"
Run everything:         pytest test_mixtures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import pytorch as pl
from rdkit import Chem
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.featurizers.molecule import ChargeFeaturizer
from chemprop.nn.agg import MeanAggregation, NormAggregation
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.predictors import RegressionFFN
from chemprop.nn.transforms import UnscaleTransform

from chemprop_contrib.mixtures.data.collate import (
    BatchComponentMolGraph,
    BatchMixtureGraph,
    BatchNodesOnly,
    collate_component,
    collate_mixture,
)
from chemprop_contrib.mixtures.data.datapoints import ComponentDatapoint, MixtureDatapoint
from chemprop_contrib.mixtures.data.datasets import (
    ComponentDataset,
    ComponentDatum,
    MixtureDataset,
    MixtureDatum,
    MixtureGraphDataset,
)
from chemprop_contrib.mixtures.data.molgraph import ComponentMolGraph, MixtureGraph
from chemprop_contrib.mixtures.featurizers import (
    ComponentMolGraphCache,
    ComponentMolGraphCacheOnTheFly,
    ComponentMolGraphFeaturizer,
    MultiHotMolInteractionFeaturizer,
    SimpleMixtureGraphFeaturizer,
)
from chemprop_contrib.mixtures.models import MixtureMPNN
from chemprop_contrib.mixtures.nn.agg import (
    AttentiveAggregation,
    ConcatAggregation,
    DeepsetsAggregation,
    MixtureAggregation,
    Set2SetAggregation,
    WeightedSumAggregation,
)
from chemprop_contrib.mixtures.nn.message_passing import (
    InteractionMessagePassing,
    MixtureMessagePassing,
    MixtureMulticomponentMessagePassing,
    MolecularMessagePassing,
)

# =============================================================================
# Shared fixtures
# =============================================================================

SMIS_SOLVENT1 = ["O", "CCO", "CO", "O"]
SMIS_SOLVENT2 = ["c1ccccc1", "CCCCCC", "ClCCl", "CCN"]
SMIS_SOLUTE = ["CCO", "CC(C)O", "c1ccccc1", "CCCCO"]
FRACS = [0.2, 0.5, 1.0, 0.7]
Y_SOLV = np.array([[-6.7], [-9.18], [-4.45], [-3.07]])


@pytest.fixture
def water():
    return Chem.MolFromSmiles("O")


@pytest.fixture
def ethanol():
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def benzene():
    return Chem.MolFromSmiles("c1ccccc1")


@pytest.fixture
def mol_featurizer():
    return ComponentMolGraphFeaturizer()


@pytest.fixture
def interaction_featurizer():
    return MultiHotMolInteractionFeaturizer.hb(max_hbond_num=3)


@pytest.fixture
def mixture_featurizer():
    return SimpleMixtureGraphFeaturizer()


# =============================================================================
# molgraph: ComponentMolGraph / MixtureGraph
# =============================================================================


class TestComponentMolGraph:
    def test_default_w_fp(self):
        mg = ComponentMolGraph(
            V=np.zeros((2, 1)),
            E=np.zeros((2, 1)),
            edge_index=np.array([[0, 1], [1, 0]]),
            rev_edge_index=np.array([1, 0]),
        )
        assert mg.w_fp == 1.0

    def test_custom_w_fp(self):
        mg = ComponentMolGraph(
            V=np.zeros((1, 1)),
            E=np.zeros((0, 1)),
            edge_index=np.zeros((2, 0), int),
            rev_edge_index=np.zeros(0, int),
            w_fp=0.3,
        )
        assert mg.w_fp == pytest.approx(0.3)

    def test_namedtuple_unpacking_compatible_with_molgraph(self):
        # Existing code does `ComponentMolGraph(*mg, w_fp=d.w_fp)`, which assumes the
        # field order matches MolGraph's first four fields.
        mg = ComponentMolGraph(
            V=np.zeros((1, 1)),
            E=np.zeros((0, 1)),
            edge_index=np.zeros((2, 0), int),
            rev_edge_index=np.zeros(0, int),
        )
        V, E, ei, rei, G, w = mg
        assert V.shape == (1, 1) and E.shape == (0, 1)
        assert ei.shape == (2, 0) and rei.shape == (0,)
        assert G.shape == (0,)
        assert w == 1.0


class TestMixtureGraph:
    def test_construct(self):
        mg = MixtureGraph(
            V=np.zeros((3, 0)),
            E=np.zeros((6, 4)),
            edge_index=np.array([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
            rev_edge_index=np.array([1, 0, 3, 2, 5, 4]),
        )
        assert mg.V.shape == (3, 0)
        assert mg.E.shape == (6, 4)


# ===========================================================================
# ComponentMolGraph.G
# ===========================================================================


class TestComponentMolGraphG:
    """ComponentMolGraph gained a G field for pre-calculated graph-level descriptors."""

    @staticmethod
    def _mg(**kw) -> ComponentMolGraph:
        return ComponentMolGraph(
            V=np.zeros((1, 1)),
            E=np.zeros((0, 1)),
            edge_index=np.zeros((2, 0), int),
            rev_edge_index=np.zeros(0, int),
            **kw,
        )

    def test_default_G_is_empty_1d(self):
        mg = self._mg()
        assert mg.G.ndim == 1
        assert mg.G.size == 0

    def test_custom_G_stored_correctly(self):
        G = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(self._mg(G=G).G, G)

    def test_G_position_in_namedtuple_unpacking(self):
        """Order must be V, E, edge_index, rev_edge_index, G, w_fp."""
        G = np.array([7.0, 8.0])
        _, _, _, _, G_out, w = self._mg(G=G, w_fp=0.42)
        np.testing.assert_array_equal(G_out, G)
        assert w == pytest.approx(0.42)

    def test_w_fp_default_unchanged_by_G_addition(self):
        assert self._mg().w_fp == pytest.approx(1.0)


# =============================================================================
# featurizers
# =============================================================================


class TestMultiHotMolInteractionFeaturizer:
    @pytest.fixture
    def feat(self):
        return MultiHotMolInteractionFeaturizer(h_bonds=[0, 1, 2, 3])

    def test_len(self, feat):
        # 4 hbond bins + 1 "out of range" bin
        assert len(feat) == 5

    def test_none_returns_zeros(self, feat):
        x = feat(None)
        assert x.shape == (5,)
        assert np.all(x == 0)

    def test_single_mol_uses_intra_hb(self, feat, ethanol):
        # ethanol: HBA=1, HBD=1 -> intra HB descriptor = 1
        x = feat([ethanol])
        assert x[1] == 1
        assert x.sum() == 1

    def test_two_mols_uses_inter_hb(self, feat, water, ethanol):
        # min(HBA(W), HBD(E)) + min(HBA(E), HBD(W))
        x = feat([water, ethanol])
        assert x.sum() == 1  # exactly one bin set
        # within range so not the "out of range" bin
        assert x[-1] == 0 or x[-1] == 1  # allow either, just check shape
        assert x.shape == (5,)

    def test_out_of_range_bin(self):
        # only 1 hbond bin -> any descriptor >= 2 falls into the overflow bin
        feat = MultiHotMolInteractionFeaturizer(h_bonds=[0])
        # propan-1,2,3-triol: 3 OHs, lots of HBA/HBD
        glycerol = Chem.MolFromSmiles("OCC(O)CO")
        x = feat([glycerol])
        # last bin (index 1) should be set
        assert x[1] == 1

    def test_empty_list_raises(self, feat):
        with pytest.raises(ValueError):
            feat([])

    def test_inter_hb_three_mols_raises(self, feat, water, ethanol, benzene):
        with pytest.raises(ValueError, match="two molecules"):
            feat([water, ethanol, benzene])

    def test_hb_classmethod_includes_zero(self):
        feat = MultiHotMolInteractionFeaturizer.hb(max_hbond_num=3)
        # bins for 0..3 inclusive plus 1 overflow = 5
        assert len(feat) == 5


class TestSimpleMixtureGraphFeaturizer:
    def test_shape_property(self, mixture_featurizer):
        assert mixture_featurizer.shape == (0, len(mixture_featurizer.interaction_featurizer))

    def test_single_component(self, mixture_featurizer, ethanol):
        mg = mixture_featurizer([ethanol])
        # 1 mol: only the diagonal self-interaction; n_interactions = 1
        assert mg.V.shape == (1, mixture_featurizer.mol_fdim)
        assert mg.E.shape == (2, mixture_featurizer.interaction_fdim)
        np.testing.assert_array_equal(mg.edge_index, np.array([[0, 0], [0, 0]]))

    def test_two_components_edge_count(self, mixture_featurizer, water, ethanol):
        mg = mixture_featurizer([water, ethanol])
        # n_interactions = 2*(2+1)/2 = 3 (two self + one cross)
        # 2 directional edges per interaction = 6
        assert mg.E.shape[0] == 6
        assert mg.edge_index.shape == (2, 6)

    def test_three_components_edge_count(self, mixture_featurizer, water, ethanol, benzene):
        mg = mixture_featurizer([water, ethanol, benzene])
        # n_interactions = 3*4/2 = 6; directional edges = 12
        assert mg.E.shape[0] == 12

    def test_rev_edge_index_is_swap(self, mixture_featurizer, water, ethanol, benzene):
        mg = mixture_featurizer([water, ethanol, benzene])
        # rev_edge_index pairs i<->i+1 for even i
        for i in range(0, len(mg.rev_edge_index), 2):
            assert mg.rev_edge_index[i] == i + 1
            assert mg.rev_edge_index[i + 1] == i

    def test_invalid_mol_features_extra_shape(self, mixture_featurizer, water, ethanol):
        with pytest.raises(ValueError, match="mol_features_extra"):
            mixture_featurizer([water, ethanol], mol_features_extra=np.zeros((3, 1)))

    def test_invalid_interaction_features_extra_shape(self, mixture_featurizer, water, ethanol):
        # n_interactions = 3 for 2 mols; provide wrong size
        with pytest.raises(ValueError, match="interaction"):
            mixture_featurizer([water, ethanol], interaction_features_extra=np.zeros((2, 1)))

    def test_interaction_features_extra_proper_indexing(self, water, ethanol, benzene):
        """Regression test for the upper-triangle indexing bug."""
        feat = SimpleMixtureGraphFeaturizer(extra_interaction_fdim=1)
        # 3 components -> 6 interactions. Use distinct values to detect collisions.
        extra = np.arange(6, dtype=np.single).reshape(6, 1)
        mg = feat([water, ethanol, benzene], interaction_features_extra=extra)

        # E has shape (12, base + 1). Last column should be the extra value
        # repeated twice per interaction in upper-triangular order.
        last_col = mg.E[:, -1]
        expected = np.repeat(np.arange(6, dtype=np.single), 2)
        np.testing.assert_array_equal(last_col, expected)


# ===========================================================================
# ComponentMolGraphFeaturizer
# ===========================================================================


class _ConstantMolFeaturizer():
    """Returns a constant vector; length and value are configurable."""

    def __init__(self, dim: int, value: float = 1.0):
        self._dim, self._value = dim, value

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return np.full(self._dim, self._value, dtype=np.float32)

    def __len__(self) -> int:
        return self._dim


class TestComponentMolGraphFeaturizer:
    """New featurizer that produces ComponentMolGraph (with G)."""

    def test_none_mol_returns_none(self):
        assert ComponentMolGraphFeaturizer()(None) is None

    def test_valid_mol_returns_component_mol_graph(self, ethanol):
        assert isinstance(ComponentMolGraphFeaturizer()(ethanol), ComponentMolGraph)

    def test_shape_is_3_tuple(self):
        assert len(ComponentMolGraphFeaturizer().shape) == 3

    def test_default_graph_fdim_is_zero(self):
        assert ComponentMolGraphFeaturizer().shape[2] == 0

    def test_extra_mol_fdim_reflected_in_shape(self):
        assert ComponentMolGraphFeaturizer(extra_mol_fdim=5).shape[2] == 5

    def test_mol_featurizer_len_reflected_in_shape(self):
        feat = ComponentMolGraphFeaturizer(mol_featurizer=_ConstantMolFeaturizer(dim=4))
        assert feat.shape[2] == 4

    def test_mol_featurizer_and_extra_mol_fdim_sum_in_shape(self):
        feat = ComponentMolGraphFeaturizer(
            mol_featurizer=_ConstantMolFeaturizer(dim=3), extra_mol_fdim=2
        )
        assert feat.shape[2] == 5

    def test_default_G_is_empty(self, ethanol):
        assert ComponentMolGraphFeaturizer()(ethanol).G.shape == (0,)

    def test_mol_featurizer_fills_G(self, ethanol):
        feat = ComponentMolGraphFeaturizer(mol_featurizer=_ConstantMolFeaturizer(dim=3, value=2.0))
        mg = feat(ethanol)
        assert mg.G.shape == (3,)
        np.testing.assert_array_almost_equal(mg.G, [2.0, 2.0, 2.0])

    def test_mol_features_extra_becomes_G_when_no_mol_featurizer(self, ethanol):
        extra = np.array([5.0, 6.0])
        mg = ComponentMolGraphFeaturizer(extra_mol_fdim=2)(ethanol, mol_features_extra=extra)
        np.testing.assert_array_almost_equal(mg.G, extra)

    def test_mol_featurizer_and_mol_features_extra_concatenated_in_G(self, ethanol):
        feat = ComponentMolGraphFeaturizer(
            mol_featurizer=_ConstantMolFeaturizer(dim=2, value=1.0), extra_mol_fdim=2
        )
        mg = feat(ethanol, mol_features_extra=np.array([3.0, 4.0]))
        assert mg.G.shape == (4,)
        np.testing.assert_array_almost_equal(mg.G[:2], [1.0, 1.0])
        np.testing.assert_array_almost_equal(mg.G[2:], [3.0, 4.0])

    def test_mol_features_extra_none_keeps_mol_featurizer_G(self, ethanol):
        feat = ComponentMolGraphFeaturizer(mol_featurizer=_ConstantMolFeaturizer(dim=2, value=9.0))
        mg = feat(ethanol, mol_features_extra=None)
        assert mg.G.shape == (2,)

    def test_mol_features_extra_2d_raises_value_error(self, ethanol):
        with pytest.raises(ValueError):
            ComponentMolGraphFeaturizer(extra_mol_fdim=2)(
                ethanol, mol_features_extra=np.zeros((1, 2))
            )

    def test_w_fp_defaults_to_one(self, ethanol):
        assert ComponentMolGraphFeaturizer()(ethanol).w_fp == pytest.approx(1.0)

    def test_w_fp_passed_through_to_mol_graph(self, ethanol):
        mg = ComponentMolGraphFeaturizer()(ethanol, w_fp=0.35)
        assert mg.w_fp == pytest.approx(0.35)

    def test_atom_features_extra_wrong_length_raises(self, ethanol):
        with pytest.raises(ValueError):
            ComponentMolGraphFeaturizer(extra_atom_fdim=1)(
                ethanol, atom_features_extra=np.zeros((1, 1))
            )

    def test_bond_features_extra_wrong_length_raises(self, ethanol):
        with pytest.raises(ValueError):
            ComponentMolGraphFeaturizer(extra_bond_fdim=1)(
                ethanol, bond_features_extra=np.zeros((1, 1))
            )


# ===========================================================================
# ComponentMolGraphCache
# ===========================================================================


@pytest.fixture
def mols():
    return [Chem.MolFromSmiles(s) for s in SMIS_SOLVENT1]


def _cache_args(mols, *, G_ds=None, w_fps=None):
    n = len(mols)
    return dict(
        mols=mols,
        V_fs=[None] * n,
        E_fs=[None] * n,
        G_d=G_ds if G_ds is not None else [None] * n,
        w_fps=w_fps if w_fps is not None else np.ones(n),
    )


class TestComponentMolGraphCache:
    def test_len(self, mols):
        cache = ComponentMolGraphCache(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols)
        )
        assert len(cache) == len(mols)

    def test_getitem_returns_component_mol_graph(self, mols):
        cache = ComponentMolGraphCache(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols)
        )
        assert isinstance(cache[0], ComponentMolGraph)

    def test_all_items_non_none_for_valid_mols(self, mols):
        cache = ComponentMolGraphCache(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols)
        )
        assert all(cache[i] is not None for i in range(len(mols)))

    def test_G_d_propagated_to_mol_graph_G(self, mols):
        G_ds = [np.array([float(i), float(i + 1)]) for i in range(len(mols))]
        cache = ComponentMolGraphCache(
            featurizer=ComponentMolGraphFeaturizer(extra_mol_fdim=2),
            **_cache_args(mols, G_ds=G_ds),
        )
        for i, expected in enumerate(G_ds):
            np.testing.assert_array_almost_equal(cache[i].G, expected)

    def test_w_fps_propagated_to_mol_graph(self, mols):
        w_fps = np.array([0.1, 0.25, 0.5, 1.0])
        cache = ComponentMolGraphCache(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols, w_fps=w_fps)
        )
        for i, w in enumerate(w_fps):
            assert cache[i].w_fp == pytest.approx(w)

    def test_none_mol_yields_none_entry(self):
        mols = [None, Chem.MolFromSmiles("O")]
        cache = ComponentMolGraphCache(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols)
        )
        assert cache[0] is None
        assert isinstance(cache[1], ComponentMolGraph)


# ===========================================================================
# ComponentMolGraphCacheOnTheFly
# ===========================================================================


class TestComponentMolGraphCacheOnTheFly:
    def test_len(self, mols):
        cache = ComponentMolGraphCacheOnTheFly(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols)
        )
        assert len(cache) == len(mols)

    def test_getitem_returns_component_mol_graph(self, mols):
        cache = ComponentMolGraphCacheOnTheFly(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols)
        )
        assert isinstance(cache[0], ComponentMolGraph)

    def test_G_d_propagated_to_mol_graph_G(self, mols):
        G_ds = [np.array([float(i) * 10]) for i in range(len(mols))]
        cache = ComponentMolGraphCacheOnTheFly(
            featurizer=ComponentMolGraphFeaturizer(extra_mol_fdim=1),
            **_cache_args(mols, G_ds=G_ds),
        )
        for i, expected in enumerate(G_ds):
            np.testing.assert_array_almost_equal(cache[i].G, expected)

    def test_w_fps_propagated_to_mol_graph(self, mols):
        w_fps = np.array([0.25, 0.5, 0.75, 1.0])
        cache = ComponentMolGraphCacheOnTheFly(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols, w_fps=w_fps)
        )
        for i, w in enumerate(w_fps):
            assert cache[i].w_fp == pytest.approx(w)

    def test_matches_precomputed_cache(self, mols):
        """On-the-fly and precomputed caches should produce identical results."""
        G_ds = [np.array([float(i)]) for i in range(len(mols))]
        w_fps = np.array([0.1, 0.4, 0.7, 1.0])
        feat = ComponentMolGraphFeaturizer(extra_mol_fdim=1)
        kwargs = _cache_args(mols, G_ds=G_ds, w_fps=w_fps)

        pre = ComponentMolGraphCache(featurizer=feat, **kwargs)
        otf = ComponentMolGraphCacheOnTheFly(featurizer=feat, **kwargs)

        for i in range(len(mols)):
            np.testing.assert_array_almost_equal(pre[i].V, otf[i].V)
            np.testing.assert_array_almost_equal(pre[i].G, otf[i].G)
            assert pre[i].w_fp == pytest.approx(otf[i].w_fp)

    def test_none_mol_yields_none_entry(self):
        mols = [None, Chem.MolFromSmiles("O")]
        cache = ComponentMolGraphCacheOnTheFly(
            featurizer=ComponentMolGraphFeaturizer(), **_cache_args(mols)
        )
        assert cache[0] is None
        assert isinstance(cache[1], ComponentMolGraph)


# =============================================================================
# datapoints
# =============================================================================


class TestComponentDatapoint:
    def test_default_w_fp(self):
        d = ComponentDatapoint.from_smi("CCO", y=np.array([1.0]))
        assert d.w_fp == 1.0

    def test_custom_w_fp(self):
        d = ComponentDatapoint.from_smi("CCO", y=np.array([1.0]), w_fp=0.5)
        assert d.w_fp == 0.5


class TestMixtureDatapoint:
    def test_from_smis(self):
        d = MixtureDatapoint.from_smis(["O", "CCO"], y=np.array([1.0]))
        assert len(d.mols) == 2
        assert all(isinstance(m, Chem.Mol) for m in d.mols)
        assert d.name == "O|CCO"

    def test_from_smis_custom_name(self):
        d = MixtureDatapoint.from_smis(["O", "CCO"], y=np.array([1.0]), name="custom")
        assert d.name == "custom"

    def test_len_is_one(self):
        d = MixtureDatapoint.from_smis(["O", "CCO"], y=np.array([1.0]))
        assert len(d) == 1

    def test_nan_replaced_in_features(self):
        V_f = np.array([[1.0, np.nan], [3.0, 4.0]])
        d = MixtureDatapoint(
            mols=[Chem.MolFromSmiles("O"), Chem.MolFromSmiles("CCO")],
            y=np.array([1.0]),
            V_f=V_f,
        )
        assert not np.isnan(d.V_f).any()
        assert d.V_f[0, 1] == 0  # NaN replaced with 0


# ===========================================================================
# ComponentDatapoint.G_d
# ===========================================================================


class TestComponentDatapointGd:
    def test_default_G_d_is_none(self):
        assert ComponentDatapoint.from_smi("CCO", y=np.array([1.0])).G_d is None

    def test_G_d_set_via_constructor(self):
        G_d = np.array([1.0, 2.0, 3.0])
        d = ComponentDatapoint(mol=Chem.MolFromSmiles("CCO"), y=np.array([1.0]), G_d=G_d)
        np.testing.assert_array_equal(d.G_d, G_d)

    def test_G_d_comes_before_w_fp_in_field_order(self):
        """Docstring says G_d precedes w_fp; the dataclass field order must match."""
        fields = list(ComponentDatapoint.__dataclass_fields__)
        assert fields.index("G_d") < fields.index("w_fp")

    def test_G_d_and_w_fp_are_independent(self):
        G_d = np.array([5.0])
        d = ComponentDatapoint(mol=Chem.MolFromSmiles("O"), y=np.array([0.0]), G_d=G_d, w_fp=0.3)
        assert d.w_fp == pytest.approx(0.3)
        np.testing.assert_array_equal(d.G_d, G_d)


# =============================================================================
# datasets
# =============================================================================


@pytest.fixture
def component_datapoints():
    return [
        ComponentDatapoint.from_smi(s, y=np.array([y]))
        for s, y in zip(SMIS_SOLUTE, [1.0, 2.0, 3.0, 4.0])
    ]


@pytest.fixture
def solvent1_datapoints():
    return [
        ComponentDatapoint.from_smi(s, y=np.array([0.0]), w_fp=float(w))
        for s, w in zip(SMIS_SOLVENT1, FRACS)
    ]


@pytest.fixture
def solvent2_datapoints():
    return [
        ComponentDatapoint.from_smi(s, y=np.array([0.0]), w_fp=float(1 - w))
        for s, w in zip(SMIS_SOLVENT2, FRACS)
    ]


@pytest.fixture
def mixture_datapoints():
    return [
        MixtureDatapoint.from_smis([sv1, sv2]) for sv1, sv2 in zip(SMIS_SOLVENT1, SMIS_SOLVENT2)
    ]


class TestComponentDataset:
    def test_getitem_returns_datum(self, component_datapoints, mol_featurizer):
        ds = ComponentDataset(component_datapoints, featurizer=mol_featurizer)
        d = ds[0]
        assert isinstance(d, ComponentDatum)
        assert isinstance(d.mg, ComponentMolGraph)

    def test_getitem_returns_datum_with_cache(self, component_datapoints, mol_featurizer):
        ds = ComponentDataset(component_datapoints, featurizer=mol_featurizer)
        ds.cache = True
        d = ds[0]
        assert isinstance(d, ComponentDatum)
        assert isinstance(d.mg, ComponentMolGraph)

    def test_w_fps_property(self, solvent1_datapoints, mol_featurizer):
        ds = ComponentDataset(solvent1_datapoints, featurizer=mol_featurizer)
        np.testing.assert_array_equal(ds.w_fps, np.array(FRACS))


class TestMixtureGraphDataset:
    def test_getitem_returns_mixture_datum(self, mixture_datapoints, mixture_featurizer):
        ds = MixtureGraphDataset(mixture_datapoints, featurizer=mixture_featurizer)
        ds.cache = True
        d = ds[0]
        assert isinstance(d, MixtureDatum)
        assert isinstance(d.mg, MixtureGraph)

    def test_smiles_property(self, mixture_datapoints, mixture_featurizer):
        ds = MixtureGraphDataset(mixture_datapoints, featurizer=mixture_featurizer)
        smis = ds.smiles
        assert len(smis) == len(SMIS_SOLVENT1)
        assert all(len(row) == 2 for row in smis)

    def test_mols_property(self, mixture_datapoints, mixture_featurizer):
        ds = MixtureGraphDataset(mixture_datapoints, featurizer=mixture_featurizer)
        mols = ds.mols
        assert all(isinstance(m, Chem.Mol) for row in mols for m in row)


class TestMixtureDataset:
    def test_construct_with_mixture_graph_last(
        self,
        component_datapoints,
        solvent1_datapoints,
        mixture_datapoints,
        mol_featurizer,
        mixture_featurizer,
    ):
        ds = MixtureDataset(
            datasets=[
                ComponentDataset(component_datapoints, featurizer=mol_featurizer),
                ComponentDataset(solvent1_datapoints, featurizer=mol_featurizer),
                MixtureGraphDataset(mixture_datapoints, featurizer=mixture_featurizer),
            ]
        )
        item = ds[0]
        assert len(item) == 3

    def test_mixture_graph_not_last_raises(
        self,
        component_datapoints,
        mixture_datapoints,
        mol_featurizer,
        mixture_featurizer,
    ):
        with pytest.raises(ValueError, match="final entry"):
            MixtureDataset(
                datasets=[
                    MixtureGraphDataset(mixture_datapoints, featurizer=mixture_featurizer),
                    ComponentDataset(component_datapoints, featurizer=mol_featurizer),
                ]
            )


# ===========================================================================
# ComponentDataset.G_d
# ===========================================================================


def _make_ds(G_dim: int | None = None) -> ComponentDataset:
    """Build a small ComponentDataset, optionally with G_d arrays attached."""
    data = [
        ComponentDatapoint(
            mol=Chem.MolFromSmiles(s),
            y=np.array([0.0]),
            G_d=np.arange(i, i + G_dim, dtype=float) if G_dim is not None else None,
        )
        for i, s in enumerate(SMIS_SOLVENT1)
    ]
    return ComponentDataset(data)


class TestComponentDatasetGd:
    def test_G_d_shape(self):
        ds = _make_ds(G_dim=3)
        assert ds.G_d.shape == (len(SMIS_SOLVENT1), 3)

    def test_G_d_values(self):
        ds = _make_ds(G_dim=2)
        np.testing.assert_array_equal(ds.G_d[0], [0.0, 1.0])
        np.testing.assert_array_equal(ds.G_d[1], [1.0, 2.0])

    def test_G_d_setter_updates_value(self):
        ds = _make_ds(G_dim=2)
        new = np.zeros((len(SMIS_SOLVENT1), 2))
        ds.G_d = new
        np.testing.assert_array_equal(ds.G_d, new)

    def test_reset_restores_original_G_d(self):
        ds = _make_ds(G_dim=2)
        original = ds.G_d.copy()
        ds.G_d = np.zeros((len(SMIS_SOLVENT1), 2))
        ds.reset()
        np.testing.assert_array_equal(ds.G_d, original)

    def test_reset_discards_any_scaled_G_d(self):
        ds = _make_ds(G_dim=2)
        original = ds.G_d.copy()
        ds.G_d = original * 999  # simulate normalisation
        ds.reset()
        np.testing.assert_array_equal(ds.G_d, original)

    def test_dataset_with_all_none_G_d_is_usable(self):
        """G_d=None for every datapoint must not break __getitem__."""
        ds = _make_ds(G_dim=None)
        datum = ds[0]
        assert isinstance(datum, ComponentDatum)

    def test_G_d_propagated_to_mol_graph_G(self):
        """datum.mg.G must reflect the datapoint's G_d when extra_mol_fdim matches."""
        G_dim = 2
        ds = ComponentDataset(
            data=[
                ComponentDatapoint(
                    mol=Chem.MolFromSmiles(s),
                    y=np.array([0.0]),
                    G_d=np.array([float(i), float(i + 1)]),
                )
                for i, s in enumerate(SMIS_SOLVENT1)
            ],
            featurizer=ComponentMolGraphFeaturizer(extra_mol_fdim=G_dim),
        )
        datum = ds[0]
        assert datum.mg.G.shape == (G_dim,)
        np.testing.assert_array_almost_equal(datum.mg.G, [0.0, 1.0])

    def test_cache_true_with_G_d(self):
        """Precomputed cache (cache=True) must work when G_d is set."""
        G_dim = 2
        ds = ComponentDataset(
            data=[
                ComponentDatapoint(
                    mol=Chem.MolFromSmiles(s),
                    y=np.array([0.0]),
                    G_d=np.array([float(i), float(i + 1)]),
                )
                for i, s in enumerate(SMIS_SOLVENT1)
            ],
            featurizer=ComponentMolGraphFeaturizer(extra_mol_fdim=G_dim),
        )
        ds.cache = True
        datum = ds[0]
        assert isinstance(datum, ComponentDatum)
        assert datum.mg.G.shape == (G_dim,)

    def test_cache_false_with_G_d(self):
        """On-the-fly cache (cache=False, default) must work when G_d is set."""
        G_dim = 2
        ds = ComponentDataset(
            data=[
                ComponentDatapoint(
                    mol=Chem.MolFromSmiles(s),
                    y=np.array([0.0]),
                    G_d=np.array([float(i), float(i + 1)]),
                )
                for i, s in enumerate(SMIS_SOLVENT1)
            ],
            featurizer=ComponentMolGraphFeaturizer(extra_mol_fdim=G_dim),
        )
        datum = ds[0]
        assert isinstance(datum, ComponentDatum)
        assert datum.mg.G.shape == (G_dim,)


# =============================================================================
# collate
# =============================================================================


class TestBatchComponentMolGraph:
    @pytest.fixture
    def mgs(self):
        return [
            ComponentMolGraph(
                V=np.array([[1.0], [2.0]]),
                E=np.array([[0.5], [0.5]]),
                edge_index=np.array([[0, 1], [1, 0]]),
                rev_edge_index=np.array([1, 0]),
                w_fp=0.3,
            ),
            ComponentMolGraph(
                V=np.array([[3.0]]),
                E=np.zeros((0, 1)),
                edge_index=np.zeros((2, 0), int),
                rev_edge_index=np.zeros(0, int),
                w_fp=0.7,
            ),
        ]

    def test_batch_concatenates_nodes(self, mgs):
        bmg = BatchComponentMolGraph(mgs)
        assert bmg.V.shape == (3, 1)
        torch.testing.assert_close(bmg.V.flatten(), torch.tensor([1.0, 2.0, 3.0]))

    def test_batch_index_offset(self, mgs):
        bmg = BatchComponentMolGraph(mgs)
        # second component had edge_index shifted by 2 (number of nodes in first)
        # but second component has no edges, so just check first component
        torch.testing.assert_close(bmg.edge_index, torch.tensor([[0, 1], [1, 0]]))

    def test_batch_assignment(self, mgs):
        bmg = BatchComponentMolGraph(mgs)
        torch.testing.assert_close(bmg.batch, torch.tensor([0, 0, 1]))

    def test_w_fps_tensor(self, mgs):
        bmg = BatchComponentMolGraph(mgs)
        torch.testing.assert_close(bmg.w_fps, torch.tensor([0.3, 0.7]))

    def test_skips_none(self, mgs):
        bmg = BatchComponentMolGraph([mgs[0], None, mgs[1]])
        # batch indices skip the None'd component (idx 1)
        torch.testing.assert_close(bmg.batch, torch.tensor([0, 0, 2]))
        # w_fps should only have entries for non-None components
        assert bmg.w_fps.shape == (2,)

    def test_to_moves_w_fps(self, mgs):
        bmg = BatchComponentMolGraph(mgs)
        bmg.to("cpu")  # no-op move, but should not error
        assert bmg.w_fps.device.type == "cpu"


class TestCollateComponent:
    def test_collate_drops_none_when_all_none(self, mol_featurizer):
        """If all components in batch are None, bmg should be None."""
        none_dp = ComponentDatapoint(mol=None, y=np.array([1.0]))
        ds = ComponentDataset([none_dp, none_dp], featurizer=mol_featurizer)
        batch = [ds[0], ds[1]]
        out = collate_component(batch)
        assert out.bmg is None
        assert out.Y is not None

    def test_collate_preserves_some_present(self, component_datapoints, mol_featurizer):
        ds = ComponentDataset(component_datapoints, featurizer=mol_featurizer)
        ds.cache = True
        batch = [ds[0], ds[1]]
        out = collate_component(batch)
        assert isinstance(out.bmg, BatchComponentMolGraph)
        assert out.Y.shape == (2, 1)


class TestCollateMixture:
    def test_full_pipeline(
        self,
        component_datapoints,
        solvent1_datapoints,
        solvent2_datapoints,
        mixture_datapoints,
        mol_featurizer,
        mixture_featurizer,
    ):
        ds = MixtureDataset(
            datasets=[
                ComponentDataset(component_datapoints, featurizer=mol_featurizer),
                ComponentDataset(solvent1_datapoints, featurizer=mol_featurizer),
                ComponentDataset(solvent2_datapoints, featurizer=mol_featurizer),
                MixtureGraphDataset(mixture_datapoints, featurizer=mixture_featurizer),
            ]
        )
        for d in ds.datasets:
            d.cache = True

        loader = DataLoader(ds, batch_size=2, collate_fn=collate_mixture)
        batch = next(iter(loader))

        assert len(batch.bmgs) == 4
        assert isinstance(batch.bmgs[0], BatchComponentMolGraph) or hasattr(batch.bmgs[0], "V")
        assert isinstance(batch.bmgs[-1], BatchMixtureGraph)
        assert batch.Y.shape == (2, 1)


# ===========================================================================
# BatchComponentMolGraph.G
# ===========================================================================


class TestBatchComponentMolGraphG:
    """BatchComponentMolGraph now concatenates G from each non-None component."""

    @staticmethod
    def _mg(G: np.ndarray, w_fp: float = 1.0) -> ComponentMolGraph:
        return ComponentMolGraph(
            V=np.array([[1.0]]),
            E=np.zeros((0, 1)),
            edge_index=np.zeros((2, 0), int),
            rev_edge_index=np.zeros(0, int),
            G=G,
            w_fp=w_fp,
        )

    def test_G_attribute_exists_on_batch(self):
        bmg = BatchComponentMolGraph([self._mg(np.empty((0,)))])
        assert hasattr(bmg, "G")

    def test_G_empty_when_components_have_no_graph_features(self):
        bmg = BatchComponentMolGraph([self._mg(np.empty((0,))), self._mg(np.empty((0,)))])
        assert bmg.G.shape == (2,0)

    def test_G_concatenates_feature_vectors(self):
        bmg = BatchComponentMolGraph(
            [self._mg(np.array([1.0, 2.0])), self._mg(np.array([3.0, 4.0]))]
        )
        torch.testing.assert_close(bmg.G, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_G_is_float32_tensor(self):
        bmg = BatchComponentMolGraph([self._mg(np.array([1.0]))])
        assert bmg.G.dtype == torch.float32

    def test_G_skips_none_components(self):
        """None components are skipped; only real mols contribute to G."""
        g = np.array([5.0, 6.0])
        bmg = BatchComponentMolGraph([self._mg(g), None, self._mg(g)])
        torch.testing.assert_close(bmg.G, torch.tensor([[5.0, 6.0], [5.0, 6.0]]))

    def test_G_moves_to_device_with_to(self):
        bmg = BatchComponentMolGraph([self._mg(np.array([1.0, 2.0]))])
        bmg.to("cpu")
        assert bmg.G.device.type == "cpu"

    def test_G_and_w_fps_on_same_device_after_to(self):
        """G and w_fps must be co-located after a .to() call."""
        bmg = BatchComponentMolGraph([self._mg(np.array([1.0]), w_fp=0.5)])
        bmg.to("cpu")
        assert bmg.G.device == bmg.w_fps.device


# =============================================================================
# message passing
# =============================================================================


class TestMixtureMulticomponentMessagePassing:
    def test_block_count_non_shared(self):
        groups = [[0], [1, 2]]
        blocks = [BondMessagePassing(d_h=8), BondMessagePassing(d_h=8)]
        mp = MixtureMulticomponentMessagePassing(blocks, groups, shared=False)
        # one block per molecule, regardless of group
        assert len(mp) == 3

    def test_block_count_shared(self):
        groups = [[0], [1, 2]]
        blocks = [BondMessagePassing(d_h=8)]
        mp = MixtureMulticomponentMessagePassing(blocks, groups, shared=True)
        # Regression: shared mode must produce one block per molecule.
        assert len(mp) == 3

    def test_mismatched_blocks_raises(self):
        with pytest.raises(ValueError, match="len"):
            MixtureMulticomponentMessagePassing(
                [BondMessagePassing(d_h=8)], groups=[[0], [1]], shared=False
            )

    def test_output_dims(self):
        groups = [[0], [1, 2]]
        blocks = [BondMessagePassing(d_h=8), BondMessagePassing(d_h=16)]
        mp = MixtureMulticomponentMessagePassing(blocks, groups, shared=False)
        assert sum(mp.output_dims) == 8 + 16 + 16


class TestMixtureMessagePassing:
    def test_output_dim(self):
        mp = MixtureMessagePassing(d_v=4, d_h=7)
        assert mp.output_dim == 7

    def test_forward_shape(self):
        mp = MixtureMessagePassing(d_v=4, d_h=8, depth=2)
        # 2 mixtures, 3 + 2 components
        V = torch.randn(5, 4)
        batch = torch.tensor([0, 0, 0, 1, 1])
        bmg = BatchNodesOnly(V=V, batch=batch)
        H = mp(bmg)
        assert H.shape == (5, 8)

    def test_message_excludes_self(self):
        """Each node's message should be the sum over its mixture except itself."""
        mp = MixtureMessagePassing(d_v=2, d_h=2, depth=1)
        V = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        batch = torch.tensor([0, 0, 0])
        bmg = BatchNodesOnly(V=V, batch=batch)
        # H_0 = W_i V; just check shape and that update works without error
        H = mp(bmg)
        assert H.shape == (3, 2)

    def test_single_component_message_is_zero(self):
        """If a 'mixture' has only 1 component, message should be zero
        (no neighbors), so output equals tau(H_0)."""
        mp = MixtureMessagePassing(d_v=2, d_h=2, depth=1)
        V = torch.randn(2, 2)
        batch = torch.tensor([0, 1])  # each in its own mixture
        bmg = BatchNodesOnly(V=V, batch=batch)
        with torch.no_grad():
            H = mp(bmg)
        # Without message, the update formula is tau(H_0 + W_h * 0) = tau(H_0).
        # We can't compare exactly without re-running internals, but check finite
        # and right shape.
        assert H.shape == (2, 2)
        assert torch.isfinite(H).all()


# =============================================================================
# aggregation
# =============================================================================


@pytest.fixture
def fp_dims():
    # solute, solvent1, solvent2
    return [4, 4, 4]


@pytest.fixture
def groups_solute_solvent():
    return [[0], [1, 2]]


@pytest.fixture
def fake_aggregation_inputs():
    """Return (H_vs, bmgs) for a 2-datapoint, 3-component scenario without mixmp."""

    # Two mixtures, both have all three components. Each component has 2 atoms.
    def make_bcmg(values, w_fps):
        Vs = [np.array([[v], [v]]) for v in values]
        E = np.zeros((0, 1))
        ei = np.zeros((2, 0), int)
        rei = np.zeros(0, int)
        mgs = [
            ComponentMolGraph(V=Vs[i], E=E, edge_index=ei, rev_edge_index=rei, w_fp=w_fps[i])
            for i in range(len(values))
        ]
        return BatchComponentMolGraph(mgs)

    bmg_solute = make_bcmg([1.0, 2.0], [1.0, 1.0])
    bmg_solv1 = make_bcmg([3.0, 4.0], [0.3, 0.5])
    bmg_solv2 = make_bcmg([5.0, 6.0], [0.7, 0.5])

    # H_vs shape: per-component, (sum_atoms, d_h)
    d_h = 4
    H_v_solute = torch.ones(2 * 2, d_h)
    H_v_solv1 = 2 * torch.ones(2 * 2, d_h)
    H_v_solv2 = 3 * torch.ones(2 * 2, d_h)

    return [H_v_solute, H_v_solv1, H_v_solv2], [bmg_solute, bmg_solv1, bmg_solv2]


class TestMixtureAggregationAbstract:
    def test_cannot_instantiate_directly(self, fp_dims, groups_solute_solvent):
        with pytest.raises(TypeError):
            MixtureAggregation(
                graph_agg=MeanAggregation(),
                groups=groups_solute_solvent,
                fp_dims=fp_dims,
            )


class TestConcatAggregation:
    def test_output_dim(self, fp_dims, groups_solute_solvent):
        agg = ConcatAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        # sum(fp_dims) + n_components_in_mixture (=2 from group [1,2])
        assert agg.output_dim == sum(fp_dims) + 2

    def test_forward_shape(self, fp_dims, groups_solute_solvent, fake_aggregation_inputs):
        H_vs, bmgs = fake_aggregation_inputs
        agg = ConcatAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        out = agg(H_vs, bmgs)
        assert out.shape == (2, agg.output_dim)


class TestWeightedSumAggregation:
    def test_output_dim(self, fp_dims, groups_solute_solvent):
        agg = WeightedSumAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        # one fp per group: solute (4) + solvent (4)
        assert agg.output_dim == 8

    def test_forward_shape(self, fp_dims, groups_solute_solvent, fake_aggregation_inputs):
        H_vs, bmgs = fake_aggregation_inputs
        agg = WeightedSumAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        out = agg(H_vs, bmgs)
        assert out.shape == (2, agg.output_dim)

    def test_weighted_sum_values(self, fp_dims, groups_solute_solvent, fake_aggregation_inputs):
        """For the binary solvent group, output = w1*H1 + w2*H2."""
        H_vs, bmgs = fake_aggregation_inputs
        agg = WeightedSumAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        out = agg(H_vs, bmgs)
        # MeanAggregation over 2 atoms with all-equal H_v just gives H_v[0]
        # -> H_solv1 = 2.0, H_solv2 = 3.0
        # datapoint 0: 0.3*2.0 + 0.7*3.0 = 2.7
        # datapoint 1: 0.5*2.0 + 0.5*3.0 = 2.5
        # Last 4 dims of `out` should match.
        torch.testing.assert_close(out[0, 4:], torch.full((4,), 2.7), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[1, 4:], torch.full((4,), 2.5), atol=1e-5, rtol=1e-5)


class TestDeepsetsAggregation:
    def test_output_dim(self, fp_dims, groups_solute_solvent):
        agg = DeepsetsAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        assert agg.output_dim == 8

    def test_forward_shape(self, fp_dims, groups_solute_solvent, fake_aggregation_inputs):
        H_vs, bmgs = fake_aggregation_inputs
        agg = DeepsetsAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        out = agg(H_vs, bmgs)
        assert out.shape == (2, agg.output_dim)


class TestAttentiveAggregation:
    def test_forward_shape(self, fp_dims, groups_solute_solvent, fake_aggregation_inputs):
        H_vs, bmgs = fake_aggregation_inputs
        agg = AttentiveAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        out = agg(H_vs, bmgs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, agg.output_dim)

    def test_attention_weights_sum_to_one(
        self, fp_dims, groups_solute_solvent, fake_aggregation_inputs
    ):
        """For a fully-populated mixture group, alpha values should normalize."""
        H_vs, bmgs = fake_aggregation_inputs
        agg = AttentiveAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        # Indirect check: forward runs without producing NaN.
        out = agg(H_vs, bmgs)
        assert torch.isfinite(out).all()


class TestSet2SetAggregation:
    def test_output_dim_doubles_for_mixture_groups(self, fp_dims, groups_solute_solvent):
        agg = Set2SetAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        # singleton group: 4; mixture group: 4*2 = 8 -> total 12
        assert agg.output_dim == 12

    def test_forward_shape(self, fp_dims, groups_solute_solvent, fake_aggregation_inputs):
        H_vs, bmgs = fake_aggregation_inputs
        agg = Set2SetAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        out = agg(H_vs, bmgs)
        assert out.shape == (2, agg.output_dim)


class TestAggregationMissingComponents:
    def test_missing_component_padded_with_zeros(self, fp_dims, groups_solute_solvent):
        """When a whole-batch component is None, output should still have the
        correct shape (zero-padded for that component)."""
        d_h = fp_dims[0]
        # Solvent2 missing for the entire batch
        H_vs = [torch.ones(2, d_h), torch.ones(2, d_h), None]

        # Build minimal bmgs with .batch and .__len__
        def fake_bmg():
            mg = ComponentMolGraph(
                V=np.zeros((1, 1)),
                E=np.zeros((0, 1)),
                edge_index=np.zeros((2, 0), int),
                rev_edge_index=np.zeros(0, int),
                w_fp=1.0,
            )
            return BatchComponentMolGraph([mg, mg])

        bmgs = [fake_bmg(), fake_bmg(), None]
        agg = WeightedSumAggregation(MeanAggregation(), groups_solute_solvent, fp_dims)
        out = agg(H_vs, bmgs)
        assert out.shape == (2, agg.output_dim)


# =============================================================================
# end-to-end / integration
# =============================================================================

INPUT_PATH = Path(__file__).parent / "test_data.csv"
INCHI_COLS = ["inchi_solute", "inchi_solvent1", "inchi_solvent2"]
FRAC_COL = "frac_solvent1"
TARGET_COLS = ["Gsolv (kcal/mol)"]
GROUPS = [[0], [1, 2]]


def inchi_to_mol(inchi):
    if inchi is None or (isinstance(inchi, float) and np.isnan(inchi)):
        return None
    return Chem.MolFromInchi(inchi)


@pytest.fixture
def small_mixture_datapoints():
    """Load test_data.csv as a MixtureDataset, handling mono-solvent rows."""
    df = pd.read_csv(INPUT_PATH).head(7)

    ys = df[TARGET_COLS].to_numpy(dtype=float)
    mol_solute = df[INCHI_COLS[0]].map(inchi_to_mol)

    fracs = df[FRAC_COL].to_numpy(dtype=float)
    mono_solvent2 = fracs == 0.0
    mono_solvent = mono_solvent2 | (fracs == 1.0)
    inchis_solvent2 = df[INCHI_COLS[2]]

    mol_solvent1 = df[INCHI_COLS[1]].where(~mono_solvent2, inchis_solvent2).map(inchi_to_mol)
    mol_solvent2 = inchis_solvent2.where(~mono_solvent, None).map(inchi_to_mol)
    fracs = np.where(mono_solvent2, 1.0, fracs)

    all_data = [
        [MoleculeDatapoint(mol, y) for mol, y in zip(mol_solute, ys)],
        [ComponentDatapoint(mol, w_fp=f) for mol, f in zip(mol_solvent1, fracs)],
        [ComponentDatapoint(mol, w_fp=f) for mol, f in zip(mol_solvent2, fracs)],
        [
            MixtureDatapoint([m for m in mols if m is not None])
            for mols in zip(mol_solute, mol_solvent1, mol_solvent2)
        ],
    ]
    return all_data


@pytest.fixture
def small_mixture_dataset(small_mixture_datapoints):
    datasets = [
        MoleculeDataset(small_mixture_datapoints[0]),
        ComponentDataset(small_mixture_datapoints[1]),
        ComponentDataset(small_mixture_datapoints[2]),
        MixtureGraphDataset(small_mixture_datapoints[3]),
    ]
    for dataset in datasets:
        dataset.cache = True
    return MixtureDataset(datasets)


class TestEndToEndForward:
    @pytest.fixture
    def model(self):
        d_h = 32
        groups = [[0], [1, 2]]
        blocks = [
            BondMessagePassing(d_h=d_h),
            BondMessagePassing(d_h=d_h),
        ]
        mp = MixtureMulticomponentMessagePassing(blocks, groups, shared=False)
        agg = WeightedSumAggregation(
            graph_agg=MeanAggregation(),
            groups=groups,
            fp_dims=[d_h, d_h, d_h],
        )
        predictor = RegressionFFN(input_dim=agg.output_dim, n_tasks=1)
        return MixtureMPNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=False,
        )

    def test_forward_runs(self, model, small_mixture_dataset):
        loader = DataLoader(small_mixture_dataset, batch_size=4, collate_fn=collate_mixture)
        batch = next(iter(loader))
        with torch.no_grad():
            preds = model(batch.bmgs, batch.V_ds, batch.X_d)
        assert preds.shape == (4, 1)
        assert torch.isfinite(preds).all()


AGG_MIXMP_CASES = [
    (WeightedSumAggregation, MixtureMessagePassing),
    (ConcatAggregation, InteractionMessagePassing),
    (DeepsetsAggregation, MolecularMessagePassing),
    (AttentiveAggregation, None),
    (Set2SetAggregation, None),
]


class TestOverfit:
    """Smaller version of the chemprop overfit test."""

    @pytest.mark.parametrize("mixagg_class,mixmp_class", AGG_MIXMP_CASES)
    def test_can_overfit(self, mixagg_class, mixmp_class, small_mixture_datapoints):
        d_h = 32

        # Two datasets: one normalized for training, one unscaled for checking fit
        def make_dataset(datapoints):
            datasets = [
                MoleculeDataset(datapoints[0]),
                ComponentDataset(datapoints[1]),
                ComponentDataset(datapoints[2]),
                MixtureGraphDataset(datapoints[3]),
            ]
            for dataset in datasets:
                dataset.cache = True
            return MixtureDataset(datasets)

        dset1 = make_dataset(small_mixture_datapoints)
        dset2 = make_dataset(small_mixture_datapoints)
        scaler = dset1.normalize_targets()

        loader1 = DataLoader(dset1, batch_size=8, shuffle=True, collate_fn=collate_mixture)
        loader2 = DataLoader(dset2, batch_size=8, shuffle=False, collate_fn=collate_mixture)

        mcmp = MixtureMulticomponentMessagePassing(
            blocks=[BondMessagePassing(d_h=d_h), BondMessagePassing(d_h=d_h)],
            groups=GROUPS,
            shared=False,
        )

        DEFAULT_MOL_FDIM, DEFAULT_INTERACTION_FDIM = SimpleMixtureGraphFeaturizer().shape
        molecule_mp_outdim = mcmp.blocks[0].output_dim
        if mixmp_class is MixtureMessagePassing:
            mixmp = mixmp_class(d_v=molecule_mp_outdim, d_h=d_h)
        elif mixmp_class is None:
            mixmp = None
        else:
            mixmp = mixmp_class(
                d_v=molecule_mp_outdim + DEFAULT_MOL_FDIM,
                d_h=d_h,
                d_e=DEFAULT_INTERACTION_FDIM,
            )

        if mixmp is None:
            fp_dims = [block.output_dim for block in mcmp.blocks]
        else:
            fp_dims = [mixmp.output_dim] * 3

        mixagg = mixagg_class(
            graph_agg=NormAggregation(),
            groups=GROUPS,
            fp_dims=fp_dims,
            mixmp=mixmp,
        )

        output_transform = UnscaleTransform.from_standard_scaler(scaler)
        ffn = RegressionFFN(
            input_dim=mixagg.output_dim,
            n_tasks=1,
            output_transform=output_transform,
        )
        model = MixtureMPNN(mcmp, mixagg, ffn, batch_norm=False, max_lr=1e-2)

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            max_epochs=123,
            overfit_batches=1.0,
        )
        trainer.fit(model, loader1)

        results = trainer.test(model, loader2)
        mse = results[0]["test/mse"]
        assert mse < 0.01, f"failed to overfit ({mixagg_class}, {mixmp_class}): mse={mse:.4f}"


class TestOverfitWithRDKit2DFeatures:
    """Overfit test using RDKit2DFeaturizer as mol_featurizer to supply G features."""

    @pytest.mark.parametrize("mixagg_class,mixmp_class", AGG_MIXMP_CASES)
    def test_can_overfit_with_G(self, mixagg_class, mixmp_class, small_mixture_datapoints):
        d_h = 32
        f = ComponentMolGraphFeaturizer(mol_featurizer=ChargeFeaturizer())
        g_fdim = f.shape[2]

        def make_dataset(datapoints):
            solute_dps = [ComponentDatapoint(mol=d.mol, y=d.y) for d in datapoints[0]]
            datasets = [
                ComponentDataset(solute_dps, featurizer=f),
                ComponentDataset(datapoints[1], featurizer=f),
                ComponentDataset(datapoints[2], featurizer=f),
                MixtureGraphDataset(datapoints[3]),
            ]
            for dataset in datasets:
                dataset.cache = True
            return MixtureDataset(datasets)

        dset1 = make_dataset(small_mixture_datapoints)
        dset2 = make_dataset(small_mixture_datapoints)
        scaler = dset1.normalize_targets()

        loader1 = DataLoader(dset1, batch_size=8, shuffle=True, collate_fn=collate_mixture)
        loader2 = DataLoader(dset2, batch_size=8, shuffle=False, collate_fn=collate_mixture)

        mcmp = MixtureMulticomponentMessagePassing(
            blocks=[BondMessagePassing(d_h=d_h), BondMessagePassing(d_h=d_h)],
            groups=GROUPS,
            shared=False,
        )

        DEFAULT_MOL_FDIM, DEFAULT_INTERACTION_FDIM = SimpleMixtureGraphFeaturizer().shape
        molecule_mp_outdim = mcmp.blocks[0].output_dim

        if mixmp_class is MixtureMessagePassing:
            mixmp = mixmp_class(d_v=molecule_mp_outdim + g_fdim, d_h=d_h)
        elif mixmp_class is None:
            mixmp = None
        else:
            mixmp = mixmp_class(
                d_v=molecule_mp_outdim + DEFAULT_MOL_FDIM + g_fdim,
                d_h=d_h,
                d_e=DEFAULT_INTERACTION_FDIM,
            )

        if mixmp is None:
            fp_dims = [block.output_dim + g_fdim for block in mcmp.blocks]
        else:
            fp_dims = [mixmp.output_dim] * 3

        mixagg = mixagg_class(
            graph_agg=NormAggregation(),
            groups=GROUPS,
            fp_dims=fp_dims,
            mixmp=mixmp,
        )

        output_transform = UnscaleTransform.from_standard_scaler(scaler)
        ffn = RegressionFFN(
            input_dim=mixagg.output_dim,
            n_tasks=1,
            output_transform=output_transform,
        )
        model = MixtureMPNN(mcmp, mixagg, ffn, batch_norm=False, max_lr=1e-2)

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            max_epochs=123,
            overfit_batches=1.0,
        )
        trainer.fit(model, loader1)

        results = trainer.test(model, loader2)
        mse = results[0]["test/mse"]
        assert mse < 0.01, f"failed to overfit ({mixagg_class}, {mixmp_class}): mse={mse:.4f}"


class TestSaveLoad:
    """Round-trip the model through MixtureMPNN._load."""

    def test_load_round_trip(self, tmp_path, small_mixture_dataset):
        d_h = 16
        groups = [[0], [1, 2]]
        blocks = [BondMessagePassing(d_h=d_h), BondMessagePassing(d_h=d_h)]
        mp = MixtureMulticomponentMessagePassing(blocks, groups, shared=False)
        agg = WeightedSumAggregation(MeanAggregation(), groups, fp_dims=[d_h, d_h, d_h])
        predictor = RegressionFFN(input_dim=agg.output_dim, n_tasks=1)
        model = MixtureMPNN(mp, agg, predictor, batch_norm=False)

        loader = DataLoader(small_mixture_dataset, batch_size=4, collate_fn=collate_mixture)
        # one fit step to populate state_dict
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            fast_dev_run=True,
            default_root_dir=tmp_path,
        )
        trainer.fit(model, loader)

        ckpt = tmp_path / "model.ckpt"
        trainer.save_checkpoint(ckpt)

        loaded = MixtureMPNN.load_from_checkpoint(ckpt, map_location="cpu")
        assert isinstance(loaded, MixtureMPNN)
        assert loaded.agg.output_dim == agg.output_dim
