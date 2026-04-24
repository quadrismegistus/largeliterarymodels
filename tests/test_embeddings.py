"""Tests for largeliterarymodels.analysis.embeddings (pure-numpy functions)."""

import numpy as np
import pandas as pd
import pytest

from largeliterarymodels.analysis.embeddings import center_by_group, mean_pool_to_text


class TestMeanPoolToText:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=['_id', 'seq', 'embedding'])

    def test_single_passage_per_text(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        df = self._make_df([('a', 0, v), ('b', 0, np.array([0, 1, 0], dtype=np.float32))])
        ids, X = mean_pool_to_text(df)
        assert set(ids) == {'a', 'b'}
        for i, _id in enumerate(ids):
            assert np.allclose(np.linalg.norm(X[i]), 1.0)

    def test_mean_pools_multiple_passages(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        df = self._make_df([('a', 0, v1), ('a', 1, v2)])
        ids, X = mean_pool_to_text(df)
        assert ids == ['a']
        expected = np.array([0.5, 0.5, 0.0])
        expected /= np.linalg.norm(expected)
        assert np.allclose(X[0], expected)

    def test_output_is_l2_normalized(self):
        rng = np.random.default_rng(42)
        rows = [('t', i, rng.standard_normal(8).astype(np.float32)) for i in range(10)]
        df = self._make_df(rows)
        ids, X = mean_pool_to_text(df)
        assert np.allclose(np.linalg.norm(X[0]), 1.0)

    def test_output_dtype(self):
        df = self._make_df([('a', 0, np.ones(4, dtype=np.float64))])
        _, X = mean_pool_to_text(df)
        assert X.dtype == np.float32

    def test_zero_vector_no_crash(self):
        df = self._make_df([('a', 0, np.zeros(3, dtype=np.float32))])
        ids, X = mean_pool_to_text(df)
        assert ids == ['a']
        assert np.allclose(X[0], 0.0)


class TestCenterByGroup:
    def test_single_group_suppresses_shared_component(self):
        # All vectors share a large dim-0 component; centering removes it
        X = np.array([[10, 1], [11, -1], [12, 0]], dtype=np.float32)
        Xc = center_by_group(X, ['a', 'a', 'a'])
        # After centering, the relative weight of dim 0 vs dim 1 should drop
        raw_ratio = np.abs(X[:, 0]).mean() / (np.abs(X[:, 1]).mean() + 1e-8)
        centered_ratio = np.abs(Xc[:, 0]).mean() / (np.abs(Xc[:, 1]).mean() + 1e-8)
        assert centered_ratio < raw_ratio

    def test_two_groups_centered_independently(self):
        X = np.array([
            [10, 0], [12, 0],   # group a: mean ~ [11, 0]
            [0, 10], [0, 12],   # group b: mean ~ [0, 11]
        ], dtype=np.float32)
        groups = ['a', 'a', 'b', 'b']
        Xc = center_by_group(X, groups)
        assert np.allclose(Xc[:2].mean(axis=0), 0, atol=1e-6)
        assert np.allclose(Xc[2:].mean(axis=0), 0, atol=1e-6)

    def test_output_is_l2_normalized(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((20, 8)).astype(np.float32)
        groups = ['en'] * 10 + ['fr'] * 10
        Xc = center_by_group(X, groups)
        norms = np.linalg.norm(Xc, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_does_not_mutate_input(self):
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        original = X.copy()
        center_by_group(X, ['a', 'a'])
        assert np.array_equal(X, original)

    def test_accepts_list_groups(self):
        X = np.ones((4, 3), dtype=np.float32)
        Xc = center_by_group(X, ['a', 'a', 'b', 'b'])
        assert Xc.shape == (4, 3)
