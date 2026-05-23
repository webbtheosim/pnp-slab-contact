from __future__ import annotations
import numpy as np


def solve_poisson_neumann(eps: np.ndarray, rho: np.ndarray, dz: float) -> np.ndarray:
    """
    Solve
        -d/dz ( eps(z) dphi/dz ) = rho(z)
    with true Neumann BC at both ends:
        dphi/dz|_{z=0} = 0
        dphi/dz|_{z=L} = 0

    Because pure Neumann Poisson is singular, impose the gauge
        mean(phi) = 0
    using a Lagrange multiplier in an augmented linear system.

    Notes
    -----
    - A compatibility correction is applied by removing the mean of rho.
      On a uniform grid, this enforces the solvability condition.
    - Returns phi with zero mean.
    """
    rho = np.asarray(rho, dtype=float).copy()
    eps = np.asarray(eps, dtype=float)

    N = rho.size
    if eps.size != N:
        raise ValueError("eps and rho must have the same length")
    if N < 2:
        raise ValueError("Need at least 2 grid points")

    # Compatibility condition for Neumann Poisson:
    # integral rho dz = 0. On a uniform grid, subtracting the mean is appropriate.
    rho -= np.mean(rho)

    eps_half = 0.5 * (eps[:-1] + eps[1:])  # size N-1
    dz2 = dz * dz

    # Build the singular Neumann operator A
    A = np.zeros((N, N), dtype=float)
    rhs = -rho.copy()

    # Left boundary: dphi/dz|0 = 0
    # Ghost elimination gives:
    #   eps_half[0] * (phi[0] - phi[1]) / dz^2 = -rho[0]
    A[0, 0] =  eps_half[0] / dz2
    A[0, 1] = -eps_half[0] / dz2

    # Interior nodes
    for i in range(1, N - 1):
        epsL = eps_half[i - 1]
        epsR = eps_half[i]
        A[i, i - 1] =  epsL / dz2
        A[i, i]     = -(epsL + epsR) / dz2
        A[i, i + 1] =  epsR / dz2

    # Right boundary: dphi/dz|L = 0
    # Ghost elimination gives:
    #   eps_half[-1] * (phi[-1] - phi[-2]) / dz^2 = -rho[-1]
    A[-1, -2] = -eps_half[-1] / dz2
    A[-1, -1] =  eps_half[-1] / dz2

    # Augment with a mean-zero gauge:
    # [ A   1 ] [phi] = [rhs]
    # [1^T  0 ] [lam]   [ 0 ]
    M = np.zeros((N + 1, N + 1), dtype=float)
    M[:N, :N] = A
    M[:N, N] = 1.0
    M[N, :N] = 1.0

    b = np.zeros(N + 1, dtype=float)
    b[:N] = rhs
    b[N] = 0.0  # sum(phi) = 0

    try:
        sol = np.linalg.solve(M, b)
    except np.linalg.LinAlgError as e:
        raise FloatingPointError(f"Poisson solve failed: {e}") from e

    phi = sol[:N]

    if not np.all(np.isfinite(phi)):
        raise FloatingPointError("NaN/inf encountered in Poisson solution")

    # Enforce zero-mean gauge numerically
    phi -= np.mean(phi)

    return phi
