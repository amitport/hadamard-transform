# Hadamard Transform

[![PyPI](https://img.shields.io/pypi/v/hadamard-transform.svg)](https://pypi.org/project/hadamard-transform/)
[![Changelog](https://img.shields.io/github/v/release/amitport/hadamard-transform?include_prereleases&label=changelog)](https://github.com/amitport/hadamard-transform/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/amitport/hadamard-transform/blob/main/LICENSE)

An efficient Hadamard transform implementation in PyTorch.

## Installation

Install this library using `pip`:

    pip install hadamard-transform

## Usage

See `tests/test_hadamard_transform.py`.

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:

    cd hadamard-transform
    python -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
