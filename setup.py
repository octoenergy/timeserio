#!/usr/bin/env python
# -*- coding: utf-8 -*

import os

from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
# read __version__ attribute from version.py
exec(open('timeserio/version.py').read())
VERSION = __version__   # type: ignore  # noqa

setup(
    name='timeserio',
    version=VERSION,
    packages=find_packages(exclude=('tests', )),
    entry_points={'console_scripts': []},
    include_package_data=True,
    zip_safe=False,
    description='Machine Learning and Forecasting tools',
    author='Octopus Energy',
    author_email='igor@octopus.energy',
    license='MIT',
    install_requires=[
        "joblib",
        "numpy",
        "pandas",
        "scikit-learn==0.20.3",
        "keras==2.2.4",
        "s3fs",
        "holidays",
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
