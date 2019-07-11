from sklearn.base import BaseEstimator, RegressorMixin


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
