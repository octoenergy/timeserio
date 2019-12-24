from typing import Dict

from sklearn.base import BaseEstimator

from timeserio.externals import keras
from ..keras.multinetwork import MultiNetworkBase
from ..pipeline import MultiPipeline


def make_pipeline_generator(*args, **kwargs):
    """Delay the `keras` import."""
    from .pipegen import _PipelineGenerator
    return _PipelineGenerator(*args, **kwargs)


class MultiModel(BaseEstimator):
    """Multi-part model with pipelines."""

    def __init__(
        self, *, multinetwork: MultiNetworkBase, multipipeline: MultiPipeline,
        manifold: Dict
    ) -> None:
        self.multinetwork = multinetwork
        self.multipipeline = multipipeline
        self.manifold = manifold

    @property
    def model_names(self):
        return self.multinetwork.model_names

    @property
    def trainable_models(self):
        return self.multinetwork.trainable_models

    @trainable_models.setter
    def trainable_models(self, value):
        self.multinetwork.trainable_models = value

    @property
    def history(self):
        return self.multinetwork.history

    def get_model_pipes(self, model_name: str):
        x_pipe_names, y_pipe_names = self.manifold[model_name]
        if not x_pipe_names:
            x_pipe_names = []
        if not y_pipe_names:
            y_pipe_names = []
        if isinstance(x_pipe_names, str):
            x_pipe_names = [x_pipe_names]
        if isinstance(y_pipe_names, str):
            y_pipe_names = [y_pipe_names]
        x_pipes = [
            self.multipipeline[pipename] if pipename else None
            for pipename in x_pipe_names
        ]
        y_pipes = [
            self.multipipeline[pipename] if pipename else None
            for pipename in y_pipe_names
        ]
        return x_pipes, y_pipes

    @staticmethod
    def get_transformed(pipes, df):
        return [pipe.transform(df) for pipe in pipes]

    @staticmethod
    def get_fit_transformed(pipes, df):
        return [pipe.fit_transform(df) for pipe in pipes]

    def fit(self, df, model=None, **kwargs):
        """Fit one of the sub-networks from DataFrame."""
        x_pipes, y_pipes = self.get_model_pipes(model)
        x = self.get_fit_transformed(x_pipes, df)
        y = self.get_fit_transformed(y_pipes, df)
        if 'validation_data' in kwargs and \
                kwargs['validation_data'] is not None:
            val_df = kwargs['validation_data']
            val_x = self.get_transformed(x_pipes, val_df)
            val_y = self.get_transformed(y_pipes, val_df)
            kwargs['validation_data'] = (val_x, val_y)
        return self.multinetwork.fit(x, y, model=model, **kwargs)

    def predict(self, df, model=None, **kwargs):
        """Predict from one of the sub-networks."""
        x_pipes, _ = self.get_model_pipes(model)
        x = self.get_transformed(x_pipes, df)
        return self.multinetwork.predict(x, model=model, **kwargs)

    def predict_generator(self, df_generator, model=None, **kwargs):
        """Predict from one of the sub-networks on a DataFrame generator."""
        x_pipes, y_pipes = self.get_model_pipes(model)
        generator = make_pipeline_generator(
            x_pipes=x_pipes, y_pipes=y_pipes, df_generator=df_generator
        )
        return self.multinetwork.predict_generator(generator,
                                                   model=model,
                                                   **kwargs)

    def fit_generator(self, df_generator, model=None, **kwargs):
        """Fit one of the sub-networks from a DataFrame generator."""
        x_pipes, y_pipes = self.get_model_pipes(model)
        if 'validation_data' in kwargs:
            is_none = (kwargs['validation_data'] is None)
            is_sequence = isinstance(
                kwargs['validation_data'],
                keras.utils.Sequence
            )
            if is_none:
                pass
            elif is_sequence:  # validate from df generator
                val_df_gen = kwargs['validation_data']
                kwargs['validation_data'] = make_pipeline_generator(
                    x_pipes=x_pipes, y_pipes=y_pipes, df_generator=val_df_gen
                )
            else:  # validate from DataFrame
                val_df = kwargs['validation_data']
                val_x = self.get_fit_transformed(x_pipes, val_df)
                val_y = self.get_fit_transformed(y_pipes, val_df)
                kwargs['validation_data'] = (val_x, val_y)
        generator = make_pipeline_generator(
            x_pipes=x_pipes, y_pipes=y_pipes, df_generator=df_generator
        )
        return self.multinetwork.fit_generator(
            generator, model=model, **kwargs
        )

    def evaluate(self, df, model=None, **kwargs):
        """Evaluate a subnetwork from a DataFrame."""
        x_pipes, y_pipes = self.get_model_pipes(model)
        x = self.get_transformed(x_pipes, df)
        y = self.get_transformed(y_pipes, df)
        metrics = self.multinetwork.evaluate(x, y, model=model, **kwargs)
        return metrics

    def evaluate_generator(self, df_generator, model=None, **kwargs):
        """Evaluate a subnetwork from a DataFrame generator."""
        x_pipes, y_pipes = self.get_model_pipes(model)
        generator = make_pipeline_generator(
            x_pipes=x_pipes, y_pipes=y_pipes, df_generator=df_generator
        )
        metrics = self.multinetwork.evaluate_generator(
            generator,
            model=model,
            **kwargs
        )
        return metrics

    def freeze_models_except(self, model=None):
        """Freeze all subnetworks except the one(s) specified by 'model'.

        Raises DeprecationWarning
        """
        self.multinetwork.freeze_models_except(model)

    def freeze(self):
        """Freeze all models."""
        self.trainable_models = None

    def unfreeze(self):
        """Unfreeze all models."""
        self.trainable_models = self.model_names
