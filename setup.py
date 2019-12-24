#!/usr/bin/env python
# -*- coding: utf-8 -*

import os

from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
CWD = os.getcwd()
# read __version__ attribute from version.py
ver_globals = {"HOME_DIR": CWD}
with open("timeserio/version.py") as f:
    exec(f.read(), ver_globals)
VERSION = ver_globals["__version__"]
# User README.md as long description
with open("README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="timeserio",
    version=VERSION,
    description="Machine Learning and Forecasting tools",
    long_description=README,
    long_description_content_type="text/markdown",
    # Credentials
    author="Octopus Energy",
    author_email="nerds@octopus.energy",
    url="https://github.com/octoenergy/timeserio",
    license="MIT",
    # Package
    packages=find_packages(exclude=("tests",)),
    entry_points={"console_scripts": []},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "joblib",
        "numpy",
        "pandas",
        "scikit-learn==0.20.3",
        "s3fs",
        "holidays",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
