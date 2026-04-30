import pytest
import torch

from hadamard_transform import hadamard_transform, randomized_hadamard_transform, inverse_randomized_hadamard_transform, \
    hadamard_transform_, randomized_hadamard_transform_, inverse_randomized_hadamard_transform_, \
    next_power_of_2, is_a_power_of_2, pad_to_power_of_2


def test_next_power_of_2_for_exact_power():
    # for inputs that are already a power of 2, next_power_of_2(n) should return n
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(8) == 8
    assert next_power_of_2(1024) == 1024


def test_next_power_of_2_for_non_power():
    assert next_power_of_2(3) == 4
    assert next_power_of_2(5) == 8
    assert next_power_of_2(9) == 16
    assert next_power_of_2(1000) == 1024


def test_is_a_power_of_2_zero_is_false():
    # 0 is not a power of 2
    assert is_a_power_of_2(0) is False


def test_is_a_power_of_2_basic():
    assert is_a_power_of_2(1) is True
    assert is_a_power_of_2(2) is True
    assert is_a_power_of_2(1024) is True
    assert is_a_power_of_2(3) is False
    assert is_a_power_of_2(1000) is False


def test_pad_to_power_of_2_already_power():
    x = torch.arange(8, dtype=torch.float64)
    out = pad_to_power_of_2(x)
    assert out.shape[-1] == 8
    assert torch.equal(out, x)


def test_pad_to_power_of_2_non_power():
    x = torch.arange(5, dtype=torch.float64)
    out = pad_to_power_of_2(x)
    assert out.shape[-1] == 8
    assert torch.equal(out[:5], x)
    assert torch.all(out[5:] == 0)


def test_pad_to_power_of_2_batched():
    x = torch.ones((3, 5), dtype=torch.float64)
    out = pad_to_power_of_2(x)
    assert out.shape == (3, 8)


def test_hadamard_transform_rejects_non_power_of_2():
    x = torch.rand(7, dtype=torch.float64)
    with pytest.raises(AssertionError):
        hadamard_transform(x)


def test_hadamard_transform_in_place_rejects_non_power_of_2():
    x = torch.rand(7, dtype=torch.float64)
    with pytest.raises(AssertionError):
        hadamard_transform_(x)


def test_hadamard_transform():
    x = torch.rand(2 ** 10, dtype=torch.float64)
    y = hadamard_transform(x)
    assert torch.allclose(hadamard_transform(y), x)


def test_batched_hadamard_transform():
    x = torch.rand([10, 2 ** 10], dtype=torch.float64)
    y = hadamard_transform(x)
    assert torch.allclose(hadamard_transform(y), x)


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
    x = torch.rand([1, 2 ** 10], dtype=torch.float64).repeat(10, 1)
    seed = prng.seed()
    tx = randomized_hadamard_transform(x, prng)
    # no row should be close to the other (unless some very rare random event)
    assert not torch.any(torch.all(torch.isclose(tx[1:], tx[0]), dim=1))
    # after the rotation all vector return to the original value
    assert torch.allclose(
        inverse_randomized_hadamard_transform(
            tx,
            prng.manual_seed(seed),
        ), x)


def test_batched_same_randomized_hadamard_transform():
    prng = torch.Generator(device='cpu')
    x = torch.rand([1, 2 ** 10], dtype=torch.float64).repeat(10, 1)
    seed = prng.seed()
    tx = randomized_hadamard_transform(x, prng, same_rotation_batch=True)
    assert torch.allclose(tx[1:], tx[0])
    assert torch.allclose(
        inverse_randomized_hadamard_transform(
            tx,
            prng.manual_seed(seed),
            same_rotation_batch=True,
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


# ---------------------------------------------------------------------------
# GPU tests. Marked with `@pytest.mark.gpu`; skipped when CUDA is unavailable
# (or xfailed-strict if --require-gpu is passed). See tests/conftest.py.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_hadamard_transform_cuda(cuda_device):
    x = torch.rand(2 ** 10, dtype=torch.float64, device=cuda_device)
    y = hadamard_transform(x)
    assert y.device.type == "cuda"
    assert torch.allclose(hadamard_transform(y), x)


@pytest.mark.gpu
def test_batched_hadamard_transform_cuda(cuda_device):
    x = torch.rand([10, 2 ** 10], dtype=torch.float64, device=cuda_device)
    y = hadamard_transform(x)
    assert y.device.type == "cuda"
    assert torch.allclose(hadamard_transform(y), x)


@pytest.mark.gpu
def test_randomized_hadamard_transform_cuda(cuda_device):
    prng = torch.Generator(device=cuda_device)
    x = torch.rand(2 ** 10, dtype=torch.float64, device=cuda_device)
    seed = prng.seed()
    y = randomized_hadamard_transform(x, prng)
    assert y.device.type == "cuda"
    assert torch.allclose(
        inverse_randomized_hadamard_transform(y, prng.manual_seed(seed)),
        x,
    )


@pytest.mark.gpu
def test_in_place_randomized_hadamard_transform_cuda(cuda_device):
    prng = torch.Generator(device=cuda_device)
    x = torch.rand(2 ** 10, dtype=torch.float64, device=cuda_device)
    original_x = torch.clone(x)
    seed = prng.seed()
    randomized_hadamard_transform_(x, prng)
    assert not torch.allclose(x, original_x)
    inverse_randomized_hadamard_transform_(x, prng.manual_seed(seed))
    assert torch.allclose(x, original_x)
