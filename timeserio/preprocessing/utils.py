import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.fixes import _astype_copy_false
from sklearn.utils.validation import FLOAT_DTYPES


def _as_list_of_str(columns):
    """Return none, one or more columns as a list."""
    columns = columns if columns else []
    if isinstance(columns, str):
        columns = [columns]
    return columns


class CallableMixin:
    """Makes transformer callable."""

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class IdentityRegressor(BaseEstimator, RegressorMixin, CallableMixin):
    """Use to turn a Pipeline into an estimator."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


def transform_selected(
    X, transform, dtype, selected="all", copy=True, retain_order=False
):
    """Apply a transform function to portion of selected features.

    Returns an array Xt, where the non-selected features appear on the right
    side (largest column indices) of Xt.

    Retained from https://github.com/scikit-learn/scikit-learn/pull/14866

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Dense array or sparse matrix.
    transform : callable
        A callable transform(X) -> X_transformed
    dtype : number type
        Desired dtype of output.
    copy : boolean, default=True
        Copy X even if it could be avoided.
    selected : "all" or array of indices or mask
        Specify which features to apply the transform to.
    retain_order : boolean, default=False
        If True, the non-selected features will not be displaced to the right
        side of the transformed array. The number of features in Xt must
        match the number of features in X. Furthermore, X and Xt cannot be
        sparse.
    Returns
    -------
    Xt : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    X = check_array(X, accept_sparse='csc', copy=copy, dtype=FLOAT_DTYPES)

    if sparse.issparse(X) and retain_order:
        raise ValueError(
            "The retain_order option can only be set to True "
            "for dense matrices."
        )

    if isinstance(selected, str) and selected == "all":
        return transform(X)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        # The columns of X which are not transformed need
        # to be casted to the desire dtype before concatenation.
        # Otherwise, the stacking will cast to the higher-precision dtype.
        X_not_sel = X[:, ind[not_sel]].astype(dtype, **_astype_copy_false(X))

    if retain_order:
        if X_sel.shape[1] + X_not_sel.shape[1] != n_features:
            raise ValueError(
                "The retain_order option can only be set to True "
                "if the dimensions of the input array match the "
                "dimensions of the transformed array."
            )

        # Fancy indexing not supported for sparse matrices
        X[:, ind[sel]] = X_sel
        return X

    if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
        return sparse.hstack((X_sel, X_not_sel))
    else:
        return np.hstack((X_sel, X_not_sel))
