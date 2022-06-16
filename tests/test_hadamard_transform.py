import torch

from hadamard_transform import hadamard_transform, randomized_hadamard_transform, inverse_randomized_hadamard_transform, \
    hadamard_transform_, randomized_hadamard_transform_, inverse_randomized_hadamard_transform_


def test_hadamard_transform():
    x = torch.rand(2 ** 10, dtype=torch.float64)
    y = hadamard_transform(x)
    assert torch.allclose(hadamard_transform(y), x)


def test_batched_hadamard_transform():
    x = torch.rand([10, 2 ** 10], dtype=torch.float64)
    y = hadamard_transform(x, batched=True)
    assert torch.allclose(hadamard_transform(y, batched=True), x)


def test_randomized_hadamard_transform():
    prng = torch.Generator(device='cpu')
    x = torch.rand(2 ** 10, dtype=torch.float64)
    seed = prng.seed()
    assert torch.allclose(
        inverse_randomized_hadamard_transform(
            randomized_hadamard_transform(x, prng),
            prng.manual_seed(seed)
        ), x)


def test_batched_randomized_hadamard_transform():
    prng = torch.Generator(device='cpu')
    x = torch.rand([10, 2 ** 10], dtype=torch.float64)
    seed = prng.seed()
    tx = randomized_hadamard_transform(x, prng, batched=True)
    assert not torch.allclose(tx[1:], tx[0])
    assert torch.allclose(
        inverse_randomized_hadamard_transform(
            tx,
            prng.manual_seed(seed),
            batched=True,
        ), x)


def test_batched_same_randomized_hadamard_transform():
    prng = torch.Generator(device='cpu')
    x = torch.rand([1, 2 ** 10], dtype=torch.float64).repeat(10, 1)
    seed = prng.seed()
    tx = randomized_hadamard_transform(x, prng, batched='same_rotation')
    assert torch.allclose(tx[1:], tx[0])
    assert torch.allclose(
        inverse_randomized_hadamard_transform(
            tx,
            prng.manual_seed(seed),
            batched='same_rotation',
        ), x)


def test_in_place_hadamard_transform():
    x = torch.rand(2 ** 10, dtype=torch.float64)
    original_x = torch.clone(x)
    hadamard_transform_(x)
    assert not torch.allclose(x, original_x)
    hadamard_transform_(x)
    assert torch.allclose(x, original_x)


def test_in_place_randomized_hadamard_transform():
    prng = torch.Generator(device='cpu')
    x = torch.rand(2 ** 10, dtype=torch.float64)
    original_x = torch.clone(x)
    seed = prng.seed()
    randomized_hadamard_transform_(x, prng)
    assert not torch.allclose(x, original_x)
    inverse_randomized_hadamard_transform_(x, prng.manual_seed(seed))
    assert torch.allclose(x, original_x)
