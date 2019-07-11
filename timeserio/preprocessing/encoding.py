import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing.base import _transform_selected
from sklearn.utils.validation import check_is_fitted


class FeatureIndexEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features.

    Like sklearn.preprocessing.LabelEncoder, but for Features - works for 2D.
    See also sklearn.preprocessing.OrdinalEncoder.
    Return values between 0 and n_classes-1.
    """

    @property
    def n_classes_(self):
        check_is_fitted(self, 'classes_')
        return len(self.classes_)

    def fit(self, X, y=None):
        """Fit feature encoder."""
        self.classes_ = np.unique(X)
        return self

    def transform(self, X, y=None):
        """Transform labels to normalized encoding."""
        check_is_fitted(self, 'classes_')
        classes = np.unique(X)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))
        return np.searchsorted(self.classes_, X)


class StatelessOneHotEncoder(OneHotEncoder):
    """Stateless version of sklearn.preprocessing.OneHotEncoder."""

    def __init__(
        self,
        *,
        n_features,
        n_values,
        dtype=np.float64,
        sparse=True,
        handle_unknown='error'
    ):
        if isinstance(n_values, str):
            raise ValueError('n_values must be specified explicitly!')
        super().__init__(
            categories=self.get_categories(n_features, n_values),
            dtype=dtype,
            sparse=sparse,
            handle_unknown=handle_unknown
        )
        self.n_features = n_features
        self.n_values = n_values
        self._init_stateless()

    @staticmethod
    def get_categories(n_features, n_values):
        if isinstance(n_values, int):
            n_values = [n_values] * n_features
        if len(n_values) != n_features:
            raise ValueError(
                "`n_values` be an int, or a list of length `n_features`."
            )
        categories = [
            range(_n_values)
            for _n_values in n_values
        ]
        return categories

    def _init_stateless(self):
        X = np.zeros((1, self.n_features))
        super().fit(X)

    @property
    def required_columns(self):
        return {None}

    def transformed_columns(self, input_columns):
        return {None}


class StatelessTemporalOneHotEncoder(StatelessOneHotEncoder):
    """Temporal version of StatelessOneHotEncoder.

    Groups columns by one-hot feature rather than temporal order.
    Requires constant `n_values` for all timesteps/features.
    """

    def __init__(
        self,
        *,
        n_features,
        n_values,
        dtype=np.float64,
        sparse=True,
        handle_unknown='error'
    ):
        if not isinstance(n_values, int):
            raise ValueError('n_values must be a single integer!')
        super().__init__(
            n_features=n_features,
            n_values=n_values,
            dtype=dtype,
            sparse=sparse,
            handle_unknown=handle_unknown
        )

    @staticmethod
    def _reshape_temporal(y, n_features):
        n_examples = y.shape[0]
        y = y.reshape((n_examples, n_features, -1))
        y = y.transpose(0, 2, 1)
        y = y.reshape((n_examples, -1))
        return y

    def transform(self, X):
        y = super().transform(X)
        return self._reshape_temporal(y, self.n_features)


class PeriodicEncoder(BaseEstimator, TransformerMixin):
    """Periodic feature encoder.

    Computes periodic features as
    `sin( 2 * np.pi * (X - phase) / period)`
    and
    `cos( 2 * np.pi * (X - phase) / period)`

    All `sin` features are stacked adjacently, followed by `cos` features
    and non-periodic features.

    Parameters:
        periodic_features : "all" or array of indices or mask
            Specify what features are treated as categorical.

            - 'all' (default): All features are treated as categorical.
            - array of indices: Array of categorical feature indices.
            - mask: Array of length n_features and with dtype=bool.

            Non-categorical features are always stacked
            to the right of the matrix.

        period: float or list of floats
            feature period, either scalar
            or list of values for each periodic feature

        phase: float or list of floats
            feature phase offset, as period

    """

    def __init__(
        self,
        *,
        periodic_features="all",
        period,
        phase=0,
    ):
        self.periodic_features = periodic_features
        self.period = period
        self.phase = phase

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def _check_value_per_feature(self, value, n_features):
        """Check one value is available per feature."""
        if isinstance(value, numbers.Number):
            return value
        if len(value) != n_features:
            raise ValueError(
                f'Must provide one constant or a list of {n_features} values,'
                f' got list of {len(value)}'
            )
        return np.array(value)

    def _transform(self, X, y=None):
        """Assume periodic features only."""
        n_features = X.shape[1]
        period = self._check_value_per_feature(self.period, n_features)
        phase = self._check_value_per_feature(self.phase, n_features)
        X = 2 * np.pi * (X - phase) / period
        X_sin, X_cos = np.sin(X), np.cos(X)
        return np.hstack([X_sin, X_cos])

    def transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        n_features = X.shape[1]
        if n_features != self.n_features_:
            raise ValueError(f'X has different shape than during fitting.'
                             f' Expected {self.n_features_} features,'
                             f' got {n_features}.')
        return _transform_selected(
            X,
            self._transform,
            dtype=X.dtype,
            selected=self.periodic_features,
            copy=True
        )

    @property
    def required_columns(self):
        return {None}

    def transformed_columns(self, input_columns):
        return {None}


class StatelessPeriodicEncoder(PeriodicEncoder):
    """Stateless version of PeriodicEncoder.

    Computes periodic features as
    `sin( 2 * np.pi * (X - phase) / period)`
    and
    `cos( 2 * np.pi * (X - phase) / period)`

    All `sin` features are stacked adjacently, followed by `cos` features
    and non-periodic features.

    Parameters:
        n_features: int
            number of columns in X

        periodic_features : "all" or array of indices or mask
            Specify what features are treated as categorical.

            - 'all' (default): All features are treated as categorical.
            - array of indices: Array of categorical feature indices.
            - mask: Array of length n_features and with dtype=bool.

            Non-categorical features are always stacked
            to the right of the matrix.

        period: float or list of floats
            feature period, either scalar
            or list of values for each periodic feature

        phase: float or list of floats
            feature phase offset, as period

    """

    def __init__(
        self,
        *,
        n_features,
        periodic_features="all",
        period,
        phase=0,
    ):
        super().__init__(
            periodic_features=periodic_features,
            period=period,
            phase=phase
        )
        self.n_features = n_features
        self._init_stateless()

    def _init_stateless(self):
        X = np.zeros((1, self.n_features))
        super().fit(X)

    @property
    def required_columns(self):
        return {None}

    def transformed_columns(self, input_columns):
        return {None}
