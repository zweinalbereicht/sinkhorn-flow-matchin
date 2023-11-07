import torch
import numpy as np


# A bunch of useful functions related to optimal transport

# taking fractional exponent values of a positive definite matrix
def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.linalg.eigh(matrix)
    vals_pow = vals.pow(p)
    print(vals)
    # vals_pow = torch.view_as_real(vals_pow)[:, 0]
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow

def wassertein_distance_gaussian(m1,cov1,m2,cov2):
    """
    returns the squared wasersetin distance between two gaussians with means and covs m1,cov1,m2,cov2
    """
    cov1_sqrt = _matrix_pow(cov1,0.5)
    bures_matrix = cov1 + cov2 - 2 * torch.sqrt(torch.matmul(cov1_sqrt, torch.matmul(cov2, cov1_sqrt)))
    return torch.sum((m1-m2) ** 2)+torch.trace(bures_matrix)

if __name__ == "__main__":
    m1 = torch.tensor([0.0])
    m2 = torch.ones(1) * 10.0
    cov1 = torch.eye(1)
    cov2 = torch.eye(1) * 1
    print(wassertein_distance_gaussian(m1,cov1,m2,cov2))
    print('nono')