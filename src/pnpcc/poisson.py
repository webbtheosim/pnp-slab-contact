from __future__ import annotations
import numpy as np

def solve_poisson_dirichlet(eps: np.ndarray, rho: np.ndarray, dz: float) -> np.ndarray:
    """
    Solve -d/dz(eps dphi/dz) = rho with phi(0)=phi(L)=0 (Dirichlet).
    Finite-volume-like discretization with variable eps; tridiagonal Thomas solve.
    """
    N = rho.size
    phi = np.zeros(N, dtype=float)

    eps_half = 0.5 * (eps[:-1] + eps[1:])

    a = np.zeros(N)  # lower
    b = np.zeros(N)  # diag
    c = np.zeros(N)  # upper
    rhs = -rho.astype(float).copy()

    # Dirichlet BC
    b[0] = 1.0
    rhs[0] = 0.0
    b[-1] = 1.0
    rhs[-1] = 0.0

    for i in range(1, N - 1):
        epsL = eps_half[i - 1]
        epsR = eps_half[i]
        a[i] = epsL / dz**2
        c[i] = epsR / dz**2
        b[i] = -(epsL + epsR) / dz**2

    # Thomas algorithm
    cp = np.zeros(N)
    dp = np.zeros(N)

    cp[0] = c[0] / b[0]
    dp[0] = rhs[0] / b[0]

    for i in range(1, N):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < N - 1 else 0.0
        dp[i] = (rhs[i] - a[i] * dp[i - 1]) / denom

    phi[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        phi[i] = dp[i] - cp[i] * phi[i + 1]

    return phi
