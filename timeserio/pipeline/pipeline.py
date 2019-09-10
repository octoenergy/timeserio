import numpy as np
import pandas as pd

import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from .. import ini


def _parse_df_y(df, y):
    """Allow func(X, y) to be called as func(df, col_name)."""
    if isinstance(df, pd.DataFrame) and isinstance(y, str):
        df, y = df, df[y]
    return df, y


class FeatureUnion(sklearn.pipeline.FeatureUnion):
    """Adds column checking for pandas transformers."""

    def __init__(
        self,
        transformer_list,
        n_jobs=ini.n_jobs,
        transformer_weights=None
    ):
        super().__init__(
            transformer_list=transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights
        )

    @property
    def required_columns(self):
        required_columns = set()
        for _, transformer in self.transformer_list:
            required_columns |= transformer.required_columns
        return required_columns

    def transformed_columns(self, input_columns):
        transformed_columns = set()
        for _, transformer in self.transformer_list:
            transformed_columns_new = transformer.transformed_columns(
                input_columns
            )
            if transformed_columns_new is not None:
                transformed_columns |= transformed_columns_new
        return transformed_columns or None


class Pipeline(sklearn.pipeline.Pipeline):
    """Adds column checking for pandas transformers."""

    def __init__(self, steps, memory=None):
        super().__init__(steps=steps, memory=memory)

    @property
    def required_columns(self):
        first_transformer = self.steps[0][1]
        if len(self.steps) == 1:
            if not hasattr(first_transformer, 'required_columns'):
                return {None}
            return first_transformer.required_columns
        remaining_pipeline = Pipeline(
            steps=self.steps[1:]
        )
        required_columns = first_transformer.required_columns
        for _ in [0, 1]:
            intermediate_columns = first_transformer.transformed_columns(
                list(required_columns)
            )
            missing_columns = (
                remaining_pipeline.required_columns - intermediate_columns
            )
            required_columns |= missing_columns
        if missing_columns:
            raise ValueError(f'Pipeline not valid: {missing_columns}')

        return required_columns

    def transformed_columns(self, input_columns):
        transformed_columns = input_columns
        for _, transformer in self.steps:
            if not hasattr(transformer, 'transformed_columns'):
                transformed_columns = {None}
            else:
                transformed_columns = \
                    transformer.transformed_columns(transformed_columns)
        return transformed_columns


class GroupedPipeline(BaseEstimator, TransformerMixin):
    """Apply a pipeline seperately to each section of a DataFrame.groupby.

    Parameters:
        groupby : string or array of strings
            The column name(s) to group the input dataframe by.

        pipeline : Pipeline or Pipeline-like object
            The pipeline to apply to each section. Just needs to implement fit
            and predict or transform.

        errors : "raise", "return_df" or "return_empty"
            Specify behaviour if a groupby key unseen is encountered at
            transform or predict time.
            "raise" will always raise an error, "return_empty" will raise if
            every group is an unseen key, and "return_df" will not raise.

            - "raise" (default): raise a KeyError
            - "return_df": Return a copy of the the input dataframe. Useful if
                the pipeline returns the original dataframe with extra columns
                - the extra columns will be null for the missing key group,
                provided at least one group was a fitted key.
            - "return_none": Return an empty dataframe / numpy array. Useful if
                the pipeline returns new columns only - the missing key group
                will be all null rows.

    """

    def __init__(self, groupby, pipeline, errors='raise'):
        self.groupby = groupby
        self.pipeline = pipeline
        self.errors = errors

    def _iter_groups(self, df, y=None):
        """Iterate over groups of `df`, and, if provided, matching labels."""
        groups = df.groupby(self.groupby).indices
        for key, sub_idx in groups.items():
            sub_df = df.iloc[sub_idx]
            sub_y = y[sub_idx] if y is not None else None
            yield key, sub_df, sub_y

    def fit(self, df, y=None):
        df, y = _parse_df_y(df, y)
        self.pipelines_ = {}
        self.pipelines_ = {
            key: self._fit_subdf(sub_df, y=sub_y)
            for key, sub_df, sub_y in self._iter_groups(df, y=y)
        }
        return self

    def _fit_subdf(self, sub_df, y=None):
        return clone(self.pipeline).fit(sub_df, y=y)

    def transform(self, df):
        return self._call_pipeline(df, attr='transform')

    def predict(self, df):
        return self._call_pipeline(df, attr='predict').squeeze()

    def fit_predict(self, df, y=None):
        return self.fit(df, y).predict(df)

    def _call_pipeline(self, df, y=None, attr=None):
        check_is_fitted(self, 'pipelines_')
        self.one_transformed = False
        transformed = [
            self._call_pipeline_subdf(key, sub_df, attr=attr)
            for key, sub_df, sub_y in self._iter_groups(df, y=y)
        ]
        if not self.one_transformed and self.errors == 'return_empty':
            raise KeyError('All keys missing in fitted pipelines')
        out = pd.concat(transformed).reindex(df.index)
        # Convert back to np.array if the pipeline returns a np.array
        if self.one_transformed and self.cast_to_numpy:
            return out.values
        return out

    def _call_pipeline_subdf(self, key, sub_df, attr=None):
        try:
            out = getattr(self.pipelines_[key], attr)(sub_df)
            # type(out) is the same for each subdf
            self.cast_to_numpy = type(out) is np.ndarray
            # If out is pd.DataFrame, its unchanged
            # If out is np.array, its a newly indexed pd.DataFrame
            out = pd.DataFrame(out)
            out.index = sub_df.index
            self.one_transformed = True
            return out
        except KeyError:
            if self.errors == 'return_df':
                return sub_df
            if self.errors == 'return_empty':
                return pd.DataFrame(index=sub_df.index)
            raise KeyError(f"Missing key {key} in fitted pipelines")

    @property
    def required_columns(self):
        groupby = [self.groupby] if type(self.groupby) is str else self.groupby
        return self.pipeline.required_columns | set(groupby)

    def transformed_columns(self, input_columns):
        return self.pipeline.transformed_columns(input_columns)
