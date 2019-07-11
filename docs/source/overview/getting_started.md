# Getting Started

## Installation

Install `timeserio` from PyPi using `pip install timeserio`, or from source using `pip install -e .`.
You will need a working TensorFlow installation - the exact steps will depend on you OS and hardware.
We maintain base docker images for our own use at https://hub.docker.com/r/octoenergy/tensorflow

## Core concepts

`timeserio` builds on keras models and scikit-learn pipelines. It introduces two core abstraction levels:
*multinetwork* and *multimodel*.

### Multinetwork

A *multinetwork* allows defining multiple named, inter-connected, parametric keras models. These could
be e.g. an encoder, a decoder, a trainable encoder-decoder model, or a classification model combining
the encoder with several additional layers.

A multinetwork is defined by sub-classing `timeserio.keras.multinetwork.MultiNetworkBase`,
and creating an instance of the new class.

A multinetwork provides the usual methods such `fit()` and `predict()`, but each of theses methods
accepts a `model` keyword argument to use a specific named model, e.g.:

```python
multinetwork.fit(X, X, model='autoencoder')
encoding = multinetwork.predict(X, model='encoder')
```

### Multimodel

A *multimodel* combines a *multinetwork* with multiple pipelines that describe how to obtain feature arrays
for each input of each of the models defined by the multinetwork.

A complex model within a multinetwork may have multiple inputs, each requiring specific
feature pre-processing or encoding:
```python
churn_rate = multinetwork.predict([age_normalized, gender, postcode_one_hot], model='churn_model')
```

A multinetwork will encapsulate the pipelines and allow simple usage, e.g.

```python
churn_rate = multimodel.predict(user_df, model='churn_model')
```

A multimodel is defined as an instance of `timeserio.multimodel.MultiModel`.

### Pipelines

Pipelines are `scikit-learn` transformers. We mostly use pipelines that take a `pandas.DataFrame` as input,
and return a `numpy` array:

```python
X = pipeline.fit_transform(df)
```

A helper class, `timeserio.pipeline.MultiPipeline`, allows grouping multiple pipelines into a single object
for use in multimodels.

## Examples and Tutorials

The best way to see how the concepts above fit together is to go through some examples.

- [MNIST Autoencoder and Classifier example](../examples/mnist) shows how to use `MultiNetworkBase` to implement a neural network architecture with multiple models, training paths and callbacks, and how to
make use of pre-training and model freezing.

- [MovieLens Recommender Engine Example](../examples/movielens) demonstrates the use of pipelines, `MultiPipeline`, and `MultiModel` to feed data from `pandas` to a multi-inputs neural network.

- [PV Forecasting Example (Part 1)](../examples/SolarGenerationTimeSeries_Part1) demonstrates the use of a `MultiModel` for working with datetime features and multiple time series.

- [PV Forecasting Example (Part 2)](../examples/SolarGenerationTimeSeries_Part2) demonstrates the use of a `MultiModel` and batch generators for training auto-regressive time series models from large datasets.

We are working on adding more examples and tutorials, as well as improving documentation.
