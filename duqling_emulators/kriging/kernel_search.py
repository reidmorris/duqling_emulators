"""Brute force search to find the best GP kernel for emulating some dataset"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, DotProduct, ConstantKernel
)

def build_kernel(kernel_type: str):
    """
    Instantiate some vanilla kernel.
    Note that all kernels are ultimately scaled by some constant kernel.

    Args:
        kernel_type: the type of kernel to instantiate
    """

    if kernel_type == "RBF":
        base_kernel = RBF()

    elif kernel_type == "Matern_0.5":
        base_kernel = Matern(nu=0.5)

    elif kernel_type == "Matern_1.5":
        base_kernel = Matern(nu=1.5)

    elif kernel_type == "Matern_2.5":
        base_kernel = Matern(nu=2.5)

    elif kernel_type == "RQ":
        base_kernel = RationalQuadratic()

    elif kernel_type == "RBF+Linear":
        base_kernel = RBF() + DotProduct()

    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    base_kernel *= ConstantKernel()

    return base_kernel

def search_kernel(X_train, y_train):
    kernel_performances = []
    for kernel_type in ["RBF", "Matern_1.5", "Matern_2.5", "RQ", "RBF+Linear"]:
        gp = GaussianProcessRegressor(kernel=build_kernel(kernel_type), alpha=1e-3, normalize_y=True)
        gp.fit(X_train, y_train)
        kernel_performances.append((kernel_type, gp.log_marginal_likelihood()))
    return max(kernel_performances, key=lambda x: x[1])[0]
