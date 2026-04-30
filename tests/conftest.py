import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu",
        action="store_true",
        default=False,
        help="Fail the test session if CUDA is unavailable. Without this flag, "
             "GPU-marked tests are skipped when CUDA is unavailable.",
    )


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return

    if config.getoption("--require-gpu"):
        pytest.exit(
            "--require-gpu was passed but torch.cuda.is_available() is False",
            returncode=1,
        )

    skip_marker = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def cuda_device():
    return torch.device("cuda")
