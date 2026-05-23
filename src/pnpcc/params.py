from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PNPParams:
    # Physical constants
    kBT: float = 4.114e-21          # J (kB*T at 298K)
    e: float = 1.602176634e-19      # C
    eps0: float = 8.8541878128e-12  # F/m

    # Conditions
    T: float = 298.15  # K
    z_plus: int = +1
    z_minus: int = -1
    c_bulk_molm3: float = 10.0      # mol/m^3 (10 mM)
    D_water: float = 1.5e-9         # m^2/s

    # Geometry
    L: float = 500e-9               # m
    N: int = 800                   # grid points
    tA: float = 150e-9              # m
    tB: float = 150e-9              # m
    h_gap0: float = 200e-9          # m
    surfA_thickness: float = 5e-9
    surfB_thickness: float = 5e-9

    # Dielectric constants (relative)
    eps_water_r: float = 80.0
    eps_dense_r: float = 40.0

    # Dense-phase diffusivity scaling
    D_dense_scale: float = 0.05

    # Fixed charge densities (C/m^3)
    rho_f_A: float = 0.0
    rho_f_B: float = 0.0

    # Time stepping
    dt: float = 1e-12                # s
    nsteps: int = 5000

    # Protocol
    n_pre:      int = 100
    n_approach: int = 600
    n_hold: int     = 400
    n_separate: int = 1000
    n_relax:    int = 0
    h_min: float = 5e-9             # m
    motion_mode: str = "unilateral"
    charge_mode: str = "uniform"   # or "surface_shell"

    # Sampling
    sample_every: int = 5

    def validate(self) -> None:
        assert self.N >= 50
        assert self.L > 0
        assert self.dt > 0
        assert self.nsteps > 0
        assert self.tA + self.tB + self.h_gap0 <= self.L, (
            "tA + tB + h_gap0 must fit inside domain length L. "
            "Increase L or decrease thicknesses."
        )
        assert self.n_pre + self.n_approach + self.n_hold + self.n_separate + self.n_relax == self.nsteps, (
            "n_pre + n_approach + n_hold + n_separate must sum to nsteps."
        )
        assert self.h_min > 0
        assert self.h_min <= self.h_gap0
