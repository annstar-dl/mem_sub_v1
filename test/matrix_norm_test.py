import torch

def norm_of_basis_computes_correct_norms():
    basis = torch.tensor([
        [[3.0, 4.0], [0.0, 0.0]],    # norm = 5.0
        [[1.0, 2.0], [2.0, 0.0]],    # norm = sqrt(1+4+4+0) = 3.0
    ])
    norm_of_basis = torch.linalg.norm(basis, dim=(1, 2), keepdim=True)
    expected = torch.tensor([[[5.0]], [[3.0]]])
    assert torch.allclose(norm_of_basis, expected)

def norm_of_basis_preserves_shape_with_keepdim():
    basis = torch.randn(4, 5, 6)
    norm_of_basis = torch.linalg.norm(basis, dim=(1, 2), keepdim=True)
    assert norm_of_basis.shape == (4, 1, 1)

def norm_of_basis_handles_zero_basis():
    basis = torch.zeros(2, 3, 3)
    norm_of_basis = torch.linalg.norm(basis, dim=(1, 2), keepdim=True)
    assert torch.all(norm_of_basis == 0)

if __name__ == "__main__":
    norm_of_basis_computes_correct_norms()
    norm_of_basis_preserves_shape_with_keepdim()
    norm_of_basis_handles_zero_basis()
    print("All tests passed!")  # Indicate that all tests passed successfully