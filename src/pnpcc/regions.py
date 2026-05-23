from __future__ import annotations
import numpy as np
from .params import PNPParams

def build_regions(p: PNPParams, h_gap: float):
    """
    Construct 1D grid and piecewise material properties for A | gap | B.
    Unilateral motion: A fixed, B moves.

    Returns:
        z, dz, eps, D, rho_f, mask_A, mask_gap, mask_B, zA_end, zB_start
    """
    z = np.linspace(0.0, p.L, p.N)
    dz = z[1] - z[0]

    gap_mid = 0.5 * p.L

    zA_end = gap_mid - 0.5 * h_gap
    zA_start = zA_end - p.tA

    zB_start = gap_mid + 0.5 * h_gap
    zB_end = zB_start + p.tB

    if zA_start < 0 or zB_end > p.L:
        raise ValueError(
            f"Geometry does not fit in box: "
            f"zA_start={zA_start:.3e}, zB_end={zB_end:.3e}, L={p.L:.3e}"
        )

    mask_A = (z >= zA_start) & (z <= zA_end)
    mask_gap = (z > zA_end) & (z < zB_start)
    mask_B = (z >= zB_start) & (z <= zB_end)


    eps_r = np.full_like(z, p.eps_water_r, dtype=float)
    eps_r[mask_A | mask_B] = p.eps_dense_r
    eps = eps_r * p.eps0

    D = np.full_like(z, p.D_water, dtype=float)
    D[mask_A | mask_B] = p.D_dense_scale * p.D_water

    rho_f = np.zeros_like(z)
    rho_f[mask_A] = p.rho_f_A
    rho_f[mask_B] = p.rho_f_B

    return z, dz, eps, D, rho_f, mask_A, mask_gap, mask_B, zA_end, zB_start

def build_regions_bilateral(p: PNPParams, h_gap: float):
    z = np.linspace(0.0, p.L, p.N)
    dz = z[1] - z[0]

    gap_mid = 0.5 * p.L

    zA_end = gap_mid - 0.5 * h_gap
    zA_start = zA_end - p.tA

    zB_start = gap_mid + 0.5 * h_gap
    zB_end = zB_start + p.tB

    if zA_start < 0 or zB_end > p.L:
        raise ValueError(
            f"Geometry does not fit in box: "
            f"zA_start={zA_start:.3e}, zB_end={zB_end:.3e}, L={p.L:.3e}"
        )

    mask_A = (z >= zA_start) & (z <= zA_end)
    mask_gap = (z > zA_end) & (z < zB_start)
    mask_B = (z >= zB_start) & (z <= zB_end)

    eps_r = np.full_like(z, p.eps_water_r, dtype=float)
    eps_r[mask_A | mask_B] = p.eps_dense_r
    eps = eps_r * p.eps0

    D = np.full_like(z, p.D_water, dtype=float)
    D[mask_A | mask_B] = p.D_dense_scale * p.D_water

    rho_f = np.zeros_like(z)
    if p.charge_mode == "uniform":
        rho_f[mask_A] = p.rho_f_A
        rho_f[mask_B] = p.rho_f_B
    elif p.charge_mode == "surface_shell":
        mask_A_surf = (z >= max(zA_start, zA_end - p.surfA_thickness)) & (z <= zA_end)
        mask_B_surf = (z >= zB_start) & (z <= min(zB_end, zB_start + p.surfB_thickness))
        rho_f[mask_A_surf] = p.rho_f_A
        rho_f[mask_B_surf] = p.rho_f_B
    else:
        raise ValueError("Unknown charge_mode")

    return z, dz, eps, D, rho_f, mask_A, mask_gap, mask_B, zA_end, zB_start
