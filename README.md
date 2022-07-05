# hadamard-transform

[![PyPI](https://img.shields.io/pypi/v/hadamard-transform.svg)](https://pypi.org/project/hadamard-transform/)
[![Changelog](https://img.shields.io/github/v/release/amitport/hadamard-transform?include_prereleases&label=changelog)](https://github.com/amitport/hadamard-transform/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/amitport/hadamard-transform/blob/main/LICENSE)

A Fast Walsh–Hadamard Transform (FWHT) implementation in PyTorch.

## Installation

Install this library using `pip`:

    pip install hadamard-transform

## Usage

For the Basic normalized fast Walsh–Hadamard transform, use:

```python
import torch
from hadamard_transform import hadamard_transform

x = torch.rand(2 ** 10, dtype=torch.float64)
y = hadamard_transform(x)
assert torch.allclose(
    hadamard_transform(y),
    x
)
```

Since the transform is not numerically-stable, it is recommended to use `float64` when possible.

The input is either a vector or a batch of vectors where the first dimension is the batch dimension. _Each vector's length
is expected to be a power of 2!_

This package also includes a `pad_to_power_of_2` util, which appends zeros up to the next power of 2 if needed.

In some common cases, we use the randomized Hadamard transform, which randomly flips the axes:

```python
import torch
from hadamard_transform import randomized_hadamard_transform, inverse_randomized_hadamard_transform

prng = torch.Generator(device='cpu')
x = torch.rand(2 ** 10, dtype=torch.float64)
seed = prng.seed()
y = randomized_hadamard_transform(x, prng),
assert torch.allclose(
    inverse_randomized_hadamard_transform(y, prng.manual_seed(seed)),
    x)
```

This package also includes `hadamard_transform_`, `randomized_hadamard_transform_`, and `inverse_randomized_hadamard_transform_`. These are in-place implementations of the previous methods. They can be useful when approaching memory limits.

#### See additional usage examples in `tests/test_hadamard_transform.py`.

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:

    cd hadamard-transform
    python -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e ".[test]"

To run the tests:

    pytest
