GPU_IMAGE?="timeserio:latest-gpu"
CPU_IMAGE?="timeserio:latest"

.PHONY: yapf lint clean sync lock test test-parallel docs-build docs-clean circle build-cpu build-gpu release

yapf:
	pipenv run yapf -vv -ir .
	pipenv run isort -y

lint:
	pipenv run flake8 .
	pipenv run pydocstyle .
	pipenv run mypy .

clean:
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

sync:
	pipenv sync --dev

lock:
	pipenv lock --dev 

test:
	pipenv run pytest tests/

test-parallel:
	pipenv run pytest -n auto tests/

doctest:
	pipenv run pytest --doctest-modules timeserio/

docs-build:
	pipenv run $(MAKE) -C docs html

docs-clean:
	pipenv run $(MAKE) -C docs clean
	rm -rf docs/source/api

docs-serve:
	pipenv run $(SHELL) -c "cd docs/_build/html; python -m http.server 8000"

circle:
	circleci config validate
	circleci local execute --job build

build-cpu: 
	docker build -t ${CPU_IMAGE} .

build-gpu:
	docker build -t ${GPU_IMAGE} . --build-arg gpu_tag="-gpu"

version:
	@pipenv run python -c "import timeserio; print(timeserio.__version__)"

package:
	pipenv run python setup.py sdist
	pipenv run python setup.py bdist_wheel

test-release: package
	pipenv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: package
	pipenv run twine upload dist/*
