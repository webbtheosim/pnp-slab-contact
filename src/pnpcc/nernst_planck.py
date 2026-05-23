from __future__ import annotations
import numpy as np
from .params import PNPParams

F = 96485.33212  # C/mol (Faraday constant)
R = 8.314462618  # J/(mol K)

def step_species(c: np.ndarray, zval: int, D: np.ndarray, phi: np.ndarray, dz: float, dt: float, p: PNPParams):
    """
    Explicit conservative update for one species:
      dc/dt = -dJ/dz
      J = -D (dc/dz + (z e/kBT) c dphi/dz)
    No-flux boundaries.
    Returns: c_new, J_half
    """
    dc = c[1:] - c[:-1]
    dphi = phi[1:] - phi[:-1]

    c_half = 0.5 * (c[1:] + c[:-1])
    D_half = 0.5 * (D[1:] + D[:-1])

    # NP flux in mol/(m^2 s): J = -D (dc/dz + (zF/RT) c dphi/dz)
    beta = (zval * F) / (R * p.T)

    J_half = -D_half * (dc / dz + beta * c_half * (dphi / dz))

    divJ = np.zeros_like(c)
    divJ[1:-1] = (J_half[1:] - J_half[:-1]) / dz
    divJ[0] = (J_half[0] - 0.0) / dz
    divJ[-1] = (0.0 - J_half[-1]) / dz

    c_new = c - dt * divJ
    nneg = np.count_nonzero(c_new < 0.0)
    if nneg > 0:
      mass_deficit = np.sum(np.minimum(c_new, 0.0)) * dz
      print(nneg,mass_deficit)
    c_new = np.maximum(c_new, 1e-12)

    return c_new, J_half

def step_species_reservoir(
    c: np.ndarray,
    z_ion: int,
    D: np.ndarray,
    phi: np.ndarray,
    dz: float,
    dt: float,
    p: PNPParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One explicit Nernst-Planck step for a single species with reservoir BCs.

    Boundary conditions:
      c(0) = c_bulk
      c(L) = c_bulk

    Electrostatics should be paired with Dirichlet Poisson BCs, e.g.
      phi(0) = 0, phi(L) = 0

    Returns
    -------
    c_new : ndarray
        Updated concentration at nodes.
    J_half : ndarray
        Flux at half nodes (size N-1), using the convention that positive
        J points in the +z direction.
    """
    c = np.asarray(c, dtype=float)
    D = np.asarray(D, dtype=float)
    phi = np.asarray(phi, dtype=float)

    N = c.size
    if D.size != N or phi.size != N:
        raise ValueError("c, D, and phi must all have the same length")

    beta = z_ion * F / (R * p.T)

    # Half-node quantities
    D_half = 0.5 * (D[:-1] + D[1:])
    c_half = 0.5 * (c[:-1] + c[1:])
    dc = c[1:] - c[:-1]
    dphi = phi[1:] - phi[:-1]

    # Standard centered NP flux
    J_half = -D_half * (dc / dz + beta * c_half * (dphi / dz))

    # Divergence of flux in the interior
    divJ = np.zeros_like(c)

    # Interior nodes
    divJ[1:-1] = (J_half[1:] - J_half[:-1]) / dz

    # Explicit update on interior only
    c_new = c.copy()
    c_new[1:-1] = c[1:-1] - dt * divJ[1:-1]

    # Enforce reservoir concentrations at the boundaries
    c_new[0] = p.c_bulk_molm3
    c_new[-1] = p.c_bulk_molm3

    # Positivity floor, if needed
    c_new = np.maximum(c_new, 1e-12)

    return c_new, J_half
