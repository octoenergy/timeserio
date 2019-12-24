import abc
import contextlib
import copy
from typing import Optional, Dict, Collection
import warnings

from timeserio.externals import keras, HABEMUS_KERAS
from ..utils.functools import get_default_args
from .utils import iterlayers


def make_history_logger(*args, **kwargs):
    from .callbacks import HistoryLogger
    return HistoryLogger(*args, **kwargs)


class MultiNetworkBase(abc.ABC):
    """Abstract base class for models that contain multiple keras sub-models.

    Examples:
        Define a simple pair of regression models using `MultiNetworkBase`:

        >>> from timeserio.keras.multinetwork import MultiNetworkBase
        >>> from keras.models import Model
        >>> from keras.layers import Input, Dense

        >>> class MyNetwork(MultiNetworkBase):
        ...     def _model(self, hidden_units=8):
        ...         input = Input((1,))
        ...         intermediate = Dense(hidden_units)(input)  # shared layer
        ...         output_1 = Dense(1)(intermediate)
        ...         output_2 = Dense(1)(intermediate)
        ...         model_1 = Model(input, output_1)
        ...         model_2 = Model(input, output_2)
        ...         model_1.compile(loss="MSE", optimizer="SGD")
        ...         model_2.compile(loss="MSE", optimizer="SGD")
        ...         return {"model_1": model_1, "model_2": model_2}

        Instantiate a multi-network, using keyword arguments to provide
        hyperparameters:

        >>> mnet = MyNetwork(hidden_units=32)

        Models are instantiated on demand:

        >>> mnet.model
        {'model_1': <keras.engine.training.Model ...>, 'model_2': <...>}


    """

    def __init__(self, **hyperparams):
        self._model_instance = None
        self.hyperparams = hyperparams
        self.weights = None
        self.optimizers_config = None
        self.history = []

    # Abstract methods that need sub-classing #
    @abc.abstractmethod
    def _model(self, **kwargs) -> Dict[str, "keras.models.Model"]:
        """Build and return a list of keras models."""
        raise NotImplementedError

    def _callbacks(self, **kwargs) -> Dict[str, "keras.callbacks.Callback"]:
        """Build and return a list of keras callbacks."""
        return {}

    # Hyperparameters #
    # Helper functions ##
    @property
    def _funcs_with_legal_params(self):
        Sequential = keras.models.Sequential
        return [
            Sequential.fit, Sequential.fit_generator, Sequential.predict,
            Sequential.predict_classes, Sequential.evaluate, self._model,
            self._callbacks
        ]

    @property
    def _funcs_with_default_params(self):
        return [self._model, self._callbacks]

    @property
    def _default_params(self):
        """Get default network parameters."""
        params = {'verbose': 0}
        for fn in self._funcs_with_default_params:
            params.update(get_default_args(fn))
        return params

    def _filter_hyperparams(self, fn, override=None):
        """Filter `hyperparams` and return those in `fn`'s arguments.

        Args:
            fn : arbitrary function
            override: dictionary, values to override `hyperparams`

        Returns:
            res : dictionary containing variables
                in both `hyperparams` and `fn`'s arguments.

        """
        override = override or {}
        res = {}
        for name, value in self.hyperparams.items():
            if keras.utils.generic_utils.has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res

    def check_params(self, params) -> None:
        """Check for user typos in `params`.

        Args:
            params: dictionary; the parameters to be checked

        Raises:
            ValueError: if any member of `params` is not a valid argument.

        """
        if not HABEMUS_KERAS:  # for unpickling without keras/tensorflow
            return

        for params_name in params:
            for fn in self._funcs_with_legal_params:
                if keras.utils.generic_utils.has_arg(fn, params_name):
                    break
            else:
                raise ValueError(f'{params_name} is not a legal parameter')

    # Public hyperparams setters/getters ##
    @property
    def hyperparams(self):
        params = self._default_params
        hyperparams = copy.deepcopy(self._hyperparams)
        params.update(hyperparams)
        return params

    @hyperparams.setter
    def hyperparams(self, params):
        self.check_params(params)
        if not hasattr(self, '_hyperparams'):
            self._hyperparams = {}
        self._hyperparams.update(params)

    def get_params(self, **params):
        """Get parameters for this estimator.

        Args:
            **params: ignored (exists for API compatibility).

        Returns:
            Dictionary of parameter names mapped to their values.

        """
        return self.hyperparams

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Args:
            **params: Dictionary of parameter names mapped to their values.

        Returns:
            self

        """
        self.hyperparams = params
        return self

    # Model #
    def _init_model(self, reset_weights=False, reset_optimizers=False):
        self._model_instance = self._model(
            **self._filter_hyperparams(self._model)
        )
        if not reset_weights and self.weights:
            self._set_weights(self.weights)
        self.weights = self._get_weights()

        if not reset_optimizers and self.optimizers_config:
            self._set_optimizers_config(self.optimizers_config)
        self.optimizers_config = self._get_optimizers_config()

        return self._model_instance

    @property
    def model(self):
        if not hasattr(self, '_model_instance') or not self._model_instance:
            self._init_model()
        return self._model_instance

    @property
    def model_names(self):
        return list(self.model.keys())

    def check_model_name(self, model):
        if model not in self.model_names:
            raise ValueError(
                f'Unknown sub-model name {model}: '
                f'must be one of {self.model_names}'
            )

    @property
    def callbacks(self):
        callbacks = {model: [] for model in self.model}
        callbacks.update(
            self._callbacks(**self._filter_hyperparams(self._callbacks))
        )
        return callbacks

    # Weights ##
    def _get_weights(self) -> Dict:
        """Get a dictionary of weights per keras model."""
        weights = {
            name: model.get_weights()
            for name, model in self.model.items()
        }
        return weights

    def _set_weights(self, weights: Dict):
        """Set model weights from dictionary of weights per model."""
        for name, model in self.model.items():
            model.set_weights(weights[name])

    # Optimizers ##
    def _get_optimizers_config(self) -> Dict:
        """
        Get optimizer parameters.
        """
        optimizers_config = {
            name: model.optimizer.get_config()
            if hasattr(model, 'optimizer') and model.optimizer else None
            for name, model in self.model.items()
        }
        return optimizers_config

    def _set_optimizers_config(self, optimizers_config: Dict):
        """
        Set optimizer parameters.
        """
        for name, config in optimizers_config.items():
            try:
                self.model[name].optimizer.__init__(**config)
            except (AttributeError, TypeError):
                pass

    # Model freezing and training #
    @property
    def trainable_models(self):
        try:
            self._trainable_models
        except AttributeError:
            self._trainable_models = self.model_names
        return self._trainable_models

    @trainable_models.setter
    def trainable_models(self, value):
        MESSAGE = (
            "`MultiNetworkBase.trainable_models` attribute is deprecated, "
            "use `trainable_models` keyword argument in "
            ".fit() or .fit_generator() instead."
        )
        warnings.warn(MESSAGE, DeprecationWarning, stacklevel=2)

        model_names = value or []
        if isinstance(model_names, str):
            model_names = [model_names]
        for model in model_names:
            self.check_model_name(model)
        self._trainable_models = value

    def _set_model_trainable(self, model=None, trainable=True):
        """Set trainable state for each layer in submodel."""
        self.check_model_name(model)
        submodel = self.model[model]
        for l in iterlayers(submodel):
            l.trainable = trainable

    def _compile_all_models(self):
        """Compile all submodels."""
        for model in self.model_names:
            submodel = self.model[model]
            try:
                submodel.compile(
                    loss=submodel.loss,
                    optimizer=submodel.optimizer,
                    metrics=submodel.metrics,
                )
            except AttributeError:
                pass

    def _freeze_models_except(self, model=None):
        """
        Freeze all layers of a given model except the layers given by 'model'.

        Args:
            model: string or list of strings

        """
        model = model or []
        if isinstance(model, str):
            model = [model]
        for m in self.model_names:
            self._set_model_trainable(model=m, trainable=False)
        for m in model:
            self._set_model_trainable(model=m, trainable=True)
        self._compile_all_models()

    def _freeze(self, trainable_models=None):
        """Freeze specified models.

        Use argument if a list is provided, else use self.trainable_models.
        """
        self._freeze_models_except(
            trainable_models if trainable_models is not None
            else self.trainable_models
        )

    def _unfreeze(self):
        """Set all keras models as trainable."""
        self._freeze_models_except(self.model_names)

    @contextlib.contextmanager
    def _model_context(
        self, *, reset_weights: bool, reset_optimizers: bool,
        reset_history: bool, freeze: bool, training: bool = False,
        persist_model: bool = False,
        trainable_models: Optional[Collection[str]] = None,
    ):
        try:
            if not persist_model or self._model_instance is None:
                self._init_model(
                    reset_weights=reset_weights,
                    reset_optimizers=reset_optimizers
                )
            if reset_history:
                self.history = []
            if freeze:
                self._freeze(trainable_models=trainable_models)
            yield
        finally:
            if freeze:
                self._unfreeze()
            if training:
                self.weights = self._get_weights()
                self.optimizers_config = self._get_optimizers_config()
            if not persist_model:
                self._model_instance = None

    def _training_context(
        self,
        *,
        reset_weights=False,
        reset_optimizers=True,
        reset_history=False,
        trainable_models=None,
    ):
        return self._model_context(
            reset_weights=reset_weights,
            reset_optimizers=reset_optimizers,
            reset_history=reset_history,
            freeze=True,
            training=True,
            persist_model=False,
            trainable_models=trainable_models,
        )

    def _prediction_context(self, persist_model=True):
        return self._model_context(
            reset_weights=False,
            reset_optimizers=False,
            reset_history=False,
            freeze=False,
            training=False,
            persist_model=persist_model
        )

    # History #
    def _add_history_record(self, *, model, history, trainable_models=None):
        """Add record of current training run to history."""
        record = {
            'model': model,
            'trainable_models': trainable_models or self.trainable_models,
            'history': history
        }
        self.history.append(record)

    # Fit/predict methods #
    def fit(
        self,
        x,
        y,
        model=None,
        reset_weights=False,
        reset_optimizers=True,
        reset_history=False,
        **kwargs
    ):
        """Run training on one model."""
        self.check_model_name(model)

        if 'trainable_models' in kwargs:
            trainable_models = kwargs.pop('trainable_models') or []
        else:
            trainable_models = None

        fit_args = self._filter_hyperparams(keras.models.Sequential.fit)
        fit_args.update(callbacks=self.callbacks.get(model))
        fit_args.update(kwargs)

        history_cbk = make_history_logger(batches=True)
        fit_args['callbacks'] += [history_cbk]

        training_context = self._training_context(
            reset_weights=reset_weights,
            reset_optimizers=reset_optimizers,
            reset_history=reset_history,
            trainable_models=trainable_models,
        )
        with training_context:
            self.model[model].fit(x, y, **fit_args)
        self._add_history_record(
            model=model, history=history_cbk.history,
            trainable_models=trainable_models
        )
        return history_cbk

    def fit_generator(
        self,
        generator,
        model=None,
        reset_weights=False,
        reset_optimizers=True,
        reset_history=False,
        **kwargs
    ):
        """Train model from `generator`.

        Args:
            generator : generator or keras.utils.Sequence
                Training batches as in Sequential.fit_generator
            init: bool
                initialize new model; use False to continue training
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit_generator`

        Returns:
            history : List[Dict]
                details about the training history at each epoch.

        """
        self.check_model_name(model)

        if 'trainable_models' in kwargs:
            trainable_models = kwargs.pop('trainable_models') or []
        else:
            trainable_models = None

        fit_args = self._filter_hyperparams(
            keras.models.Sequential.fit_generator
        )
        fit_args.update(callbacks=self.callbacks.get(model))
        fit_args.update(kwargs)

        history_cbk = make_history_logger(batches=True)
        fit_args['callbacks'] += [history_cbk]

        training_context = self._training_context(
            reset_weights=reset_weights,
            reset_optimizers=reset_optimizers,
            reset_history=reset_history,
            trainable_models=trainable_models,
        )
        with training_context:
            self.model[model].fit_generator(generator, **fit_args)
        self._add_history_record(
            model=model, history=history_cbk.history,
            trainable_models=trainable_models
        )
        return history_cbk

    def predict(self, x, model: str = None, **kwargs):
        """Return predictions for the given test data.

        Args:
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.predict`.

        Returns:
            preds: array-like, shape `(n_samples, ...)`
                Predictions.

        """
        self.check_model_name(model)
        pred_kwargs = self._filter_hyperparams(keras.models.Sequential.predict)
        pred_kwargs.update(kwargs)
        with self._prediction_context():
            predictions = self.model[model].predict(x, **pred_kwargs)
        return predictions

    def predict_generator(self, generator, model: str = None, **kwargs):
        """Return predictions from a batch generator.

        Args:
            generator : generator or keras.utils.Sequence
                predict batches as in Sequential.predict_generator
            **kwargs: dictionary arguments
                Legal arguments are the arguments of
                `Sequential.predict_generator`

        Returns:
            preds: array-like, shape `(n_samples, ...)`
                Predictions.

        """
        self.check_model_name(model)
        pred_kwargs = self._filter_hyperparams(
            keras.models.Sequential.predict_generator, kwargs
        )
        with self._prediction_context():
            predictions = (
                self.model[model].predict_generator(generator, **pred_kwargs)
            )
        return predictions

    def evaluate(self, x, y, model=None, **kwargs):
        """
        Evaluate estimator on given data.

        Args:
            x:  array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like
            model: string
            kwargs: keyword arguments.
                    Legal arguments are the arguments of `Sequential.evaluate`.

        Returns:
            metrics: List[float]
                A list of loss and evaluation metric values

        """
        self.check_model_name(model)
        ev_kwargs = self._filter_hyperparams(
            keras.models.Sequential.evaluate, kwargs
        )
        with self._prediction_context():
            metrics = self.model[model].evaluate(x, y, **ev_kwargs)
        return metrics

    def score(self, *args, **kwargs):
        """Alias for scikit compatibility."""
        metrics = self.evaluate(*args, **kwargs)
        try:
            score = metrics[0]  # in case we have loss + multiple metrics
        except (TypeError, IndexError):
            score = metrics
        return score

    def evaluate_generator(self, generator, model=None, **kwargs):
        """
        Evaluate estimator on given data.

        Args:
            generator : generator or keras.utils.Sequence
                predict batches as in Sequential.evaluate_generator
            model: string
            **kwargs: keyword arguments.
                Legal arguments are the arguments of
                `Sequential.evaluate_generator`.

        Returns:
            evaluation metric value(s)

        """
        self.check_model_name(model)
        ev_kwargs = self._filter_hyperparams(
            keras.models.Sequential.evaluate, kwargs
        )
        with self._prediction_context():
            evaluation = (
                self.model[model].evaluate_generator(generator, **ev_kwargs)
            )
        return evaluation

    # Pickling #
    def __getstate__(self) -> Dict:
        """Get a picklable state dictionary of the model."""
        state = {}
        state['hyperparams'] = self.hyperparams
        state['weights'] = self.weights
        state['optimizers'] = self.optimizers_config
        state['history'] = self.history
        return state

    def __setstate__(self, state: Dict):
        """Set model from unpickled state dictionary."""
        self.hyperparams = state.get('hyperparams')
        self.weights = state.get('weights')
        self.optimizers_config = state.get('optimizers')
        self.history = state.get('history')
