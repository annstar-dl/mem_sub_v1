import torch
import kornia
import pytest

from sampling_grid import gaussian_filter

def test_gaussian_filter_basic():
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1.0  # Impulse in the center
    kernel_size = (3, 3)
    sigma = (1.0, 1.0)

    filtered = gaussian_filter(mask, kernel_size, sigma)

    # Output should have the same shape
    assert filtered.shape == mask.shape
    # Output should be a torch.Tensor
    assert isinstance(filtered, torch.Tensor)
    # The center value should be less than 1 (smoothed)
    assert filtered[5, 5] < 1.0
    # The sum should be close to the original sum (energy preserved)
    assert torch.isclose(filtered.sum(), mask.sum(), atol=1e-4)

def test_gaussian_filter_invalid_dim():
    mask = torch.zeros(1)  # 1D tensor
    kernel_size = (3, 3)
    sigma = (1.0, 1.0)
    with pytest.raises(ValueError):
        gaussian_filter(mask, kernel_size, sigma)

if __name__ == "__main__":
    test_gaussian_filter_basic()
    test_gaussian_filter_invalid_dim()