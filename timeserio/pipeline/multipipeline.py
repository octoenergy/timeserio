from typing import Dict

from sklearn.base import BaseEstimator


class MultiPipeline(BaseEstimator):
    """Umbrella object for holding multiple pipelines.

    Pipelines are scikit-learn transformers.
    """

    def __init__(self, pipelines: Dict[str, BaseEstimator]) -> None:
        if not pipelines:
            pipelines = {}
        self.pipelines = pipelines

    def _get_param_names(self):
        """Get parameter names for the estimator."""
        return sorted(self.pipelines.keys())

    def __getattr__(self, item):
        if item in self.pipelines:
            return self.pipelines[item]
        else:
            # Default behaviour
            raise AttributeError

    def __getitem__(self, item):
        if item in self.pipelines:
            return self.pipelines[item]
        else:
            # Default behaviour
            raise KeyError(f'Unknown pipe "{item}"')

    @property
    def required_columns(self):
        required_columns = set()
        for pipeline in self.pipelines:
            required_columns |= self.pipelines[pipeline].required_columns
        return required_columns

    def transformed_columns(self, input_columns):
        transformed_columns = set()
        for pipeline in self.pipelines:
            transformed_columns |= self.pipelines[
                pipeline].transformed_columns(input_columns)
        return transformed_columns
