import numpy as np
import numpy.testing as npt
import pytest
from sklearn.preprocessing import OneHotEncoder

from timeserio.preprocessing import (
    FeatureIndexEncoder, StatelessOneHotEncoder,
    StatelessTemporalOneHotEncoder, StatelessPeriodicEncoder
)
from timeserio.preprocessing.encoding import PeriodicEncoder


class TestFeatureIndexEncoder:
    @pytest.mark.parametrize(
        'n_labels, expected_encoding', [
            (1, np.arange(1)),
            (2, np.arange(2)),
            (3, np.arange(3)),
        ]
    )
    def test_feature_encoder(self, n_labels, expected_encoding):
        encoder = FeatureIndexEncoder()
        labels = np.array(
            [f'label{idx}' for idx in range(n_labels)]
        ).reshape(-1, 1)
        new_ids = encoder.fit_transform(labels)
        assert isinstance(new_ids, np.ndarray)
        assert len(new_ids.shape) == 2
        assert new_ids.shape[1] == 1
        assert set(new_ids.ravel() == set(expected_encoding.ravel()))


class TestStatelessOneHotEncoder:
    n_rows = 10

    def test_invalid_n_values(self):
        with pytest.raises(ValueError):
            StatelessOneHotEncoder(n_features=1, n_values='auto')

    @pytest.mark.parametrize(
        'n_features, n_values, categories', [
            (1, 3, [[0, 1, 2]]),
            (2, 3, [[0, 1, 2], [0, 1, 2]])
        ]
    )
    def test_same_as_stateful(
        self, n_features, n_values, categories, random
    ):
        x = np.random.randint(
            0, np.min(n_values), size=(self.n_rows, n_features)
        )
        stateful_enc = OneHotEncoder(
            categories=categories,
            sparse=False
        )
        stateless_enc = StatelessOneHotEncoder(
            n_features=n_features,
            n_values=n_values,
            sparse=False
        )
        x0 = stateful_enc.fit_transform(x)
        x1 = stateless_enc.transform(x)
        npt.assert_allclose(x1, x0)

    @pytest.mark.parametrize(
        'n_features, n_values, categories', [
            (1, [3], [[0, 1, 2]]),
            (2, [3, 2], [[0, 1, 2], [0, 1]])
        ]
    )
    def test_same_as_stateful_for_multiple_n_values(
        self, n_features, n_values, categories, random
    ):
        x = np.hstack([
            np.random.randint(0, np.min(_n_values), size=(self.n_rows, 1))
            for _n_values in n_values
        ])
        stateful_enc = OneHotEncoder(
            categories=categories,
            sparse=False
        )
        stateless_enc = StatelessOneHotEncoder(
            n_features=n_features,
            n_values=n_values,
            sparse=False
        )
        x0 = stateful_enc.fit_transform(x)
        x1 = stateless_enc.transform(x)
        npt.assert_allclose(x1, x0)


class TestStatelessTemporalOneHotEncoder:
    n_rows = 3

    @pytest.mark.parametrize('n_values', ['all', [True], [0]])
    def test_invalid_n_values(self, n_values):
        with pytest.raises(ValueError):
            StatelessTemporalOneHotEncoder(n_features=1, n_values=n_values)

    def test_temporal_onehot(self):
        x = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ])
        y_expected = np.array(
            [
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
            ]
        )
        n_values = 2
        enc = StatelessTemporalOneHotEncoder(
            n_features=x.shape[1], n_values=n_values, sparse=False
        )
        y = enc.fit_transform(x)
        npt.assert_allclose(y, y_expected)


class TestPeriodicEncoder:
    n_rows = 10
    column = np.linspace(0, 1, num=n_rows)
    column_sin = np.sin(2 * np.pi * column)
    column_cos = np.cos(2 * np.pi * column)
    column_stacked = np.vstack([column_sin, column_cos]).T

    def array(self, n_features):
        x = np.arange(n_features)
        y = self.column
        _, X = np.meshgrid(x, y)
        return X

    @pytest.mark.parametrize('periodic_features', [[], [False]])
    def test_single_column_no_transform(self, periodic_features):
        enc = PeriodicEncoder(periodic_features=periodic_features, period=1)
        X = self.array(n_features=1)
        Xt = enc.fit_transform(X)
        npt.assert_array_equal(X, Xt)

    @pytest.mark.parametrize('periodic_features', ['all', [0], [True]])
    def test_single_column(self, periodic_features):
        enc = PeriodicEncoder(periodic_features=periodic_features, period=1)
        X = self.array(n_features=1)
        Xt = enc.fit_transform(X)
        npt.assert_allclose(Xt, self.column_stacked)

    @pytest.mark.parametrize('n_features', [2])
    @pytest.mark.parametrize(
        'periodic_features', ['all', [0, 1], [True, True]]
    )
    def test_multi_column(self, n_features, periodic_features):
        enc = PeriodicEncoder(periodic_features=periodic_features, period=1)
        X = self.array(n_features=2)
        Xt = enc.fit_transform(X)
        npt.assert_allclose(Xt[:, ::2], self.column_stacked)
        npt.assert_allclose(Xt[:, 1::2], self.column_stacked)


class TestStatelessPeriodicEncoder:
    n_rows = 10

    @pytest.mark.parametrize(
        'n_features, periodic_features, period', [
            (1, 'all', 1.), (2, 'all', 1.), (2, 'all', [1., 2.]),
            (2, [True, False], 3), (2, [1], 3)
        ]
    )
    def test_same_as_stateful(self, n_features, periodic_features, period):
        x = np.random.randint(0, 10, size=(self.n_rows, n_features))
        stateful_enc = PeriodicEncoder(
            periodic_features=periodic_features, period=period
        )
        stateless_enc = StatelessPeriodicEncoder(
            n_features=n_features,
            periodic_features=periodic_features,
            period=period
        )
        x0 = stateful_enc.fit_transform(x)
        x1 = stateless_enc.transform(x)
        npt.assert_array_equal(x1, x0)
