[![CircleCI](https://circleci.com/gh/octoenergy/timeserio/tree/master.svg?style=svg)](https://circleci.com/gh/octoenergy/timeserio/tree/master) 
[![codecov](https://codecov.io/gh/octoenergy/timeserio/branch/master/graph/badge.svg)](https://codecov.io/gh/octoenergy/timeserio)
[![PyPI version](https://badge.fury.io/py/timeserio.svg)](https://badge.fury.io/py/timeserio)

# timeserio

`timeserio` is the missing link between `pandas`, `scikit-learn` and `keras`. It simplifies building end-to-end deep learning models - from a DataFrame through feature pipelines to multi-stage models with shared layers. While initially developed for tackling time series problems, it has since been used as a versatile tool for rapid ML model development and deployment.

Loosing track of big networks with multiple inputs and outputs? Forgetting to freeze the right layers?
Struggling to re-generate the input features? `timeserio` can help!

![complex_network](https://raw.githubusercontent.com/octoenergy/timeserio/master/docs/source/_static/multinetwork_complex.svg?sanitize=true)

## Documentation and Tutorials

Please see the [official documentation](http://tech.octopus.energy/timeserio/) on how to get started.

## Features

* Enable encapsulated, maintainable and reusable deep learning models
* Feed data from `pandas` through `scikit-learn` feature pipelines to multiple neural network inputs
* Manage complex architectures, layer sharing, partial freezing and re-training
* Provide collection of extensible building blocks with emphasis on time series problems

## Installation

`pip install timeserio`, or install from source - `pip install -e .`

See [Getting Started](http://tech.octopus.energy/timeserio/overview/getting_started.html#installation)

## Development

We welcome contributions and enhancements to any part of the code base, documentation, or tool chain.

See [CONTRIBUTING.md](https://github.com/octoenergy/timeserio/blob/master/CONTRIBUTING.md) for details on setting up the development environment, running tests,
etc.
