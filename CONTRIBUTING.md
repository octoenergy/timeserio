# Contributing

This document outlines the standards we (strive to) follow,
and the tools we use to make the process easier.

## Makefile

Useful commands to aid development are specifed in the `Makefile`. We try to re-use
`make` commands as much as possible for both local development and continuous integration (CI).

## Environment and dependencies

The development environment is created using [`pipenv`](https://docs.pipenv.org/en/latest/).

Run `make sync` to install specified dependency versions.

To update *development* or *test* dependencies, edit `Pipfile` and lock versions using `make lock`.
To add dependencies needed for installation and use of the `timeserio` as a stand-alone package, edit `install_requires` in the `setup.py` first, before running `make lock; make sync` to propagate these
to your development environment.

The key steps are:

- add package dependencies in `setup.py:install_requires` - use loose versioning as far as possible
- specify development and test requirements in `Pipfile` 
- run `make lock` to pin specific versions and update `Pipfile.lock` - commit this file to Git
- run `make sync` to install versions specified in the `Pipfile.lock`

### NB: pipenv, Tensorflow, and manylinux2010
At the time of writing (`2019-12-20`), newer Tensorflow builds (including `1.15.0` and `2.0`) for Linux are released as [`manylinux2010` wheels](https://discuss.python.org/t/the-next-manylinux-specification/1043). This causes [many issues](https://github.com/pypa/manylinux/issues/179), and requires `pipenv` to be installed from [source](https://github.com/pypa/pipenv) via `pip install git+https://github.com/pypa/pipenv.git` if you need to resolve dependencies or update the `Pipfile.lock`.

## Code Style

The code follows PEP8 and [`yapf`](https://github.com/google/yapf) code style.
We use `flake8`, `pydocstyle` and `mypy` to check code.

- run `make lint` to check style issues

## Tests

New code must pass unit tests. Tests are run using [`pytest`](https://docs.pytest.org/).

- run `make test` to test your code
- use advanced `pytest` features to run a sub-set of tests or run tests in parallel

## Documentation

### Docstrings

We use [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), and run `pydocstyle` to ensure basic compliance.

Examples in docstrings are written using [`doctest` format](https://docs.python.org/3/library/doctest.html),
and run during CI.

### Sphinx

We use [Sphinx](https://www.sphinx-doc.org) to auto-generate HTML documentation. The documentation is generated from three main sources:

1. Markdown documentation in `docs/source`
2. Jupyter Notebooks in `examples/`
3. Docstrings in sub-modules, classes and functions in `timeserio/`

The Sphinx set-up and array of plug-ins is somewhat convoluted, but `docs/source` should give
you a good idea of how to add new sections or examples. From the repository root, run

- `make docs-build` to build HTML documentation locally
- `make docs-serve` to start serving documentation on [0.0.0.0/8000](0.0.0.0/8000)

## Continuous Integration

We use [`CircleCI`](https://circleci.com) to run the following steps for each commit or pull request:

- run linters and static code checks (`make lint`)
- run unit tests (`make test`)
- run doctests (`make doctest`)
- submit code coverage reports to [Codecov](https://codecov.io)

In addition, after a successful merge into `master`, we

- build Sphinx HTML documentation
- deploy Sphinx documentation to a dedicate branch

## Base docker images

Official tensorflow docker images doesn't really fit our needs as they depend on python3.5 rather than pyton3.6. For this reason two docker files can be find in the `dockerfiles/` folder, one for a CPU image and another for CPU image.

These images live publicly in our (docker hub )[https://hub.docker.com/r/octoenergy/tensorflow/].

More info about (docker and tensorflow)[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles]
