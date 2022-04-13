#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "Algo-Lib"
DESCRIPTION = "Algorithm library"
AUTHOR = "Jahnavi"
REQUIRES_PYTHON = ">=3.8.0"

# Set package version
about = {}
about["__version__"] = "0.1.0"

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    #url=URL,
    packages=find_packages(exclude=("tests",)),
    #package_data={"calculations": ["VERSION"], "texts":["VERSION"]},
    #install_requires=["numpy==1.16.6"],
    extras_require={},
    include_package_data=True,
    #license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)