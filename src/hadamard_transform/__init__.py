from math import sqrt, floor, log2
from typing import Union, Literal

import torch
import torch.nn.functional as F


def rademacher_like(x: torch.Tensor, prng: torch.Generator):
    """ returns a random vector in {-1, 1}**d """
    return torch.empty_like(x).bernoulli_(generator=prng) * 2 - 1


def hadamard_transform(x: torch.Tensor):
    """Fast Walsh–Hadamard transform

    The hadamard transform is not numerically stable by nature (lots of subtractions),
    it is recommended to use with float64 when possible

    :param x: Either a vector or a batch of vectors where the first dimension is the batch dimension.
              Each vector's length is expected to be a power of 2! (or each row if it is batched)
    :return: The normalized Hadamard transform of each vector in x
    """
    original_shape = x.shape
    assert 1 <= len(original_shape) <= 2, 'input\'s dimension must be either 1 or 2'
    if len(original_shape) is 1:
        # add fake 1 batch dimension
        # for making the code a follow a single (batched) path
        x = x.unsqueeze(0)
    batch_dim, d = x.shape

    h = 2
    while h <= d:
        hf = h // 2
        x = x.view(batch_dim, d // h, h)

        half_1, half_2 = x[:, :, :hf], x[:, :, hf:]

        x = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)

        h *= 2

    return (x / sqrt(d)).view(*original_shape)


def randomized_hadamard_transform(x: torch.Tensor, prng: torch.Generator,
                                  same_rotation_batch: bool = False):
    if same_rotation_batch:
        d = rademacher_like(x[0], prng)
    else:
        d = rademacher_like(x, prng)

    return hadamard_transform(x * d)


def inverse_randomized_hadamard_transform(tx: torch.Tensor, prng: torch.Generator,
                                          same_rotation_batch: bool = False):
    if same_rotation_batch:
        d = rademacher_like(tx[0], prng)
    else:
        d = rademacher_like(tx, prng)

    return hadamard_transform(tx) * d


def hadamard_transform_(vec: torch.Tensor):
    """In-place fast Walsh–Hadamard transform

    Generally, the in-place version is not recommended
    since it is generally *not* faster.
    It is appropriate when we otherwise run out of memory.

    hadamard transform is not very numerically stable by nature (lots of subtractions)
    should try and use with float64 when possible

    :param vec: vec is expected to be a power of 2!
    :return: the Hadamard transform of vec
    """
    d = vec.numel()
    h = 2
    while h <= d:
        hf = h // 2
        vec = vec.view(d // h, h)

        # The following is in place of
        # half_1 = half_1 + half_2
        # half_2 = (half_1 + half_2) - 2 * half_2 =  half_1 - half_2
        vec[:, :hf] += vec[:, hf:]
        vec[:, hf:] *= -2
        vec[:, hf:] += vec[:, :hf]

        h *= 2

    vec /= sqrt(d)


def randomized_hadamard_transform_(vec: torch.Tensor, prng: torch.Generator):
    d = rademacher_like(vec, prng)
    vec *= d
    hadamard_transform_(vec)


def inverse_randomized_hadamard_transform_(tvec: torch.Tensor, prng: torch.Generator):
    d = rademacher_like(tvec, prng)
    hadamard_transform_(tvec)
    tvec *= d


def next_power_of_2(n):
    return 2 ** (floor(log2(n)) + 1)


def is_a_power_of_2(n):
    return n & (n - 1) == 0


def pad_to_power_of_2(x):
    """A util to pad to the next power of 2 (as required by the Hadamard transform)
    :param x: a 1d vector or a batch of 1d vectors (first dim is the batch dim)
    :return: x padded with zero until the next power-of-2
    """
    d = x.shape[-1]
    # pad to the nearest power of 2 if needed
    return x if is_a_power_of_2(d) else F.pad(x, (0, next_power_of_2(d) - d))