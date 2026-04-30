# hadamard-transform

[![PyPI](https://img.shields.io/pypi/v/hadamard-transform.svg)](https://pypi.org/project/hadamard-transform/)
[![Changelog](https://img.shields.io/github/v/release/amitport/hadamard-transform?include_prereleases&label=changelog)](https://github.com/amitport/hadamard-transform/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/amitport/hadamard-transform/blob/main/LICENSE)

A Fast Walsh–Hadamard Transform (FWHT) implementation in PyTorch.

## Installation

Install this library using `pip`:

    pip install hadamard-transform

To run on GPU, install a CUDA build of PyTorch from the
[official PyTorch index](https://pytorch.org/get-started/locally/) instead of
the default CPU wheel, e.g.:

    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install hadamard-transform

Once a CUDA-enabled `torch` is installed, this package works on CUDA tensors
without any additional configuration.

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
y = randomized_hadamard_transform(x, prng)
assert torch.allclose(
    inverse_randomized_hadamard_transform(y, prng.manual_seed(seed)),
    x)
```

> **Note on devices:** the transform itself is device-agnostic and works on any
> tensor (CPU, CUDA, MPS, etc.). For the randomized variants, the
> `torch.Generator` you pass must live on the same device as the input tensor,
> e.g. `torch.Generator(device='cuda')` for a CUDA input.

For a batch of vectors, you can pass `same_rotation_batch=True` to share the
same random sign-flip across all rows of the batch (instead of an independent
flip per row):

```python
y = randomized_hadamard_transform(x_batch, prng, same_rotation_batch=True)
```

This package also includes `hadamard_transform_`, `randomized_hadamard_transform_`, and `inverse_randomized_hadamard_transform_`. These are in-place implementations of the previous methods. They can be useful when approaching memory limits. The in-place version expects a 1D tensor.

#### See additional usage examples in `tests/test_hadamard_transform.py`.

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:

    cd hadamard-transform
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows

Now install the package in editable mode with test dependencies:

    pip install -e ".[test]"

To run the tests:

    pytest

GPU tests are marked with `@pytest.mark.gpu` and are skipped automatically when
CUDA is not available. On a machine with a CUDA-enabled `torch` install, they
run as part of `pytest`. In CI on a GPU runner, pass `--require-gpu` to fail
loudly if CUDA isn't actually wired up:

    pytest --require-gpu
