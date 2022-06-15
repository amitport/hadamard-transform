import math
from functools import cache

import torch


@cache
def sqrt(n):
    # cached version vectors' dimensions
    return math.sqrt(n)


def rademacher_like(x, prng):
    """ returns a random vector in {-1, 1}**d """
    return torch.empty_like(x).bernoulli_(generator=prng) * 2 - 1


def hadamard_transform(vec):
    """fast Walsh–Hadamard transform

    hadamard transform is not very numerically stable by nature (lots of subtractions)
    should try and use with float64 when possible

    :param vec: vec is expected to be a power of 2!
    :return: the Hadamard transform of vec
    """
    d = vec.numel()
    original_shape = vec.shape
    h = 2
    while h <= d:
        hf = h // 2
        vec = vec.view(d // h, h)

        # For an in=place version we could use
        #  vec[:, :hf] += vec[:, hf:]
        #  vec[:, hf:] *= -2
        #  vec[:, hf:] += vec[:, :hf]
        # Generally, the in-place version is not recommended
        # since it is generally *not* faster, and the in-place
        # side effect can easily introduce bugs.
        # *It is appropriate when we otherwise run out of memory*.

        # TODO add batch support with
        # half_1, half_2 = batch[:, :, :hf], batch[:, :, hf:]
        half_1, half_2 = vec[:, :hf], vec[:, hf:]

        vec = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)

        h *= 2

    return (vec / sqrt(d)).view(*original_shape)


def randomized_hadamard_transform(x, prng):
    d = rademacher_like(x, prng)

    return hadamard_transform(x * d)


def inverse_randomized_hadamard_transform(tx, prng):
    d = rademacher_like(tx, prng)

    return hadamard_transform(tx) * d


def example_function():
    return 1 + 1
