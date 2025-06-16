import torch
import pytest
from sampling_grid import dilate_mask, gaussian_filter, get_sampling_grid

def dilate_mask_handles_2d_tensor():
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1.0
    d = 2

    dilated = dilate_mask(mask, d)

    assert dilated.shape == mask.shape
    assert dilated.dtype == mask.dtype
    assert dilated[5, 5] == 1.0
    assert dilated.sum() > mask.sum()

def dilate_mask_handles_empty_tensor():
    mask = torch.zeros(10, 10)
    d = 2

    dilated = dilate_mask(mask, d)

    assert dilated.sum() == 0.0

def gaussian_filter_handles_2d_tensor():
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1.0
    kernel_size = (3, 3)
    sigma = (1.0, 1.0)

    filtered = gaussian_filter(mask, kernel_size, sigma)

    assert filtered.shape == mask.shape
    assert filtered.dtype == mask.dtype
    assert filtered[5, 5] < 1.0
    assert torch.isclose(filtered.sum(), mask.sum(), atol=1e-4)

def gaussian_filter_raises_error_for_invalid_dim():
    mask = torch.zeros(1)
    kernel_size = (3, 3)
    sigma = (1.0, 1.0)

    with pytest.raises(ValueError):
        gaussian_filter(mask, kernel_size, sigma)

def get_sampling_grid_returns_correct_shapes():
    mask = torch.ones(10, 10)
    d = 2
    w = 2

    mask, x_indices, y_indices = get_sampling_grid(mask, d, w)

    assert mask.shape == (10, 10)
    assert x_indices.ndim == 1
    assert y_indices.ndim == 1
    assert len(x_indices) == len(y_indices)

def get_sampling_grid_handles_empty_mask():
    mask = torch.zeros(10, 10)
    d = 2
    w = 2

    mask, x_indices, y_indices = get_sampling_grid(mask, d, w)

    assert mask.sum() == 0.0
    assert len(x_indices) == 0
    assert len(y_indices) == 0

if __name__ == "__main__":
    pytest.main([__file__])
    # Run the tests
    dilate_mask_handles_2d_tensor()
    dilate_mask_handles_empty_tensor()
    gaussian_filter_handles_2d_tensor()
    gaussian_filter_raises_error_for_invalid_dim()
    get_sampling_grid_returns_correct_shapes()
    get_sampling_grid_handles_empty_mask()