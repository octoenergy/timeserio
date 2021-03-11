GPU_IMAGE?="timeserio:latest-gpu"
CPU_IMAGE?="timeserio:latest"

.PHONY: yapf lint clean sync lock test test-parallel docs-build docs-clean circle build-cpu build-gpu release

yapf:
	poetry run yapf -vv -ir .
	poetry run isort -y

lint:
	poetry run flake8 .
	poetry run pydocstyle .
	poetry run mypy .

clean:
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
	rm -rf dist/
	rm -rf build/

sync:
	poetry install

lock:
	poetry lock

test:
	poetry run pytest tests/

test-parallel:
	poetry run pytest -n auto tests/

doctest:
	poetry run pytest --doctest-modules timeserio/

docs-build:
	poetry run $(MAKE) -C docs html

docs-clean:
	poetry run $(MAKE) -C docs clean
	rm -rf docs/source/api

docs-serve:
	poetry run $(SHELL) -c "cd docs/_build/html; python -m http.server 8000"

circle:
	circleci config validate
	circleci local execute --job build

build-cpu: 
	docker build -t ${CPU_IMAGE} .

build-gpu:
	docker build -t ${GPU_IMAGE} . --build-arg gpu_tag="-gpu"

version:
	@poetry run python -c "import timeserio; print(timeserio.__version__)"

package:
	poetry buil

test-release: package
	poetry run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: package
	poetry run twine upload dist/*
