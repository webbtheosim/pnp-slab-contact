from __future__ import annotations
import numpy as np
from .params import PNPParams
from .regions import build_regions,build_regions_bilateral
from .protocol import gap_schedule
from .poisson import solve_poisson_dirichlet
from .nernst_planck import step_species,step_species_reservoir

NA = 6.02214076e23  # 1/mol
F = 96485.33212  # C/mol (Faraday constant)
R = 8.314462618  # J/(mol K)

def run(p: PNPParams):
    p.validate()

    # Initial conditions
    if p.motion_mode == "bilateral":
        z, dz, eps, D, rho_f, mask_A, mask_gap, mask_B, zA_end, zB_start = build_regions_bilateral(p, p.h_gap0)
    else:
        z, dz, eps, D, rho_f, mask_A, mask_gap, mask_B, zA_end, zB_start = build_regions(p, p.h_gap0)

    mask_outside = ~(mask_A | mask_gap | mask_B)

    overlap = mask_A.astype(int) + mask_gap.astype(int) + mask_B.astype(int)
    if np.any(overlap > 1):
        bad = np.where(overlap > 1)[0]
        raise RuntimeError(f"Initial region overlap: indices {bad[:10]}")

    cover_full = (
        mask_A.astype(int)
        + mask_gap.astype(int)
        + mask_B.astype(int)
        + mask_outside.astype(int)
    )
    if not np.all(cover_full == 1):
        bad = np.where(cover_full != 1)[0]
        raise RuntimeError(f"Initial full-domain partition failure: indices {bad[:10]}")

    c0 = p.c_bulk_molm3   # mol/m^3
    c_plus = np.full_like(z, c0, dtype=float)
    c_minus = np.full_like(z, c0, dtype=float)

    history: list[dict] = []

    # Phase boundaries in absolute step index
    pre_start = 0
    pre_end = max(0, p.n_pre - 1)

    approach_start = p.n_pre
    approach_end = p.n_pre + max(0, p.n_approach - 1)

    hold_start = p.n_pre + p.n_approach
    hold_end = p.n_pre + p.n_approach + max(0, p.n_hold - 1)

    separate_start = p.n_pre + p.n_approach + p.n_hold
    separate_end = p.n_pre + p.n_approach + p.n_hold + max(0, p.n_separate - 1)

    relax_start = p.n_pre + p.n_approach + p.n_hold + p.n_separate
    relax_end = p.nsteps - 1

    # Candidate snapshots: beginning, midpoint, and end of each phase
    snapshot_candidates = {
        "pre_start": pre_start,
        "pre_mid": p.n_pre // 2 if p.n_pre > 0 else pre_start,
        "pre_end": pre_end,

        "approach_start": approach_start,
        "approach_mid": approach_start + (p.n_approach // 2) if p.n_approach > 0 else approach_start,
        "approach_end": approach_end,

        "hold_start": hold_start,
        "hold_mid": hold_start + (p.n_hold // 2) if p.n_hold > 0 else hold_start,
        "hold_end": hold_end,

        "separate_start": separate_start,
        "separate_mid": separate_start + (p.n_separate // 2) if p.n_separate > 0 else separate_start,
        "separate_end": separate_end,

        "relax_start": relax_start,
        "relax_mid": relax_start + (p.n_relax // 2) if p.n_relax > 0 else relax_start,
        "relax_end": relax_end,

        # Convenience alias
        "final": relax_end,
    }

    # Deduplicate while preserving first occurrence
    snapshot_steps = {}
    used_steps = set()
    for label, step_idx in snapshot_candidates.items():
        if step_idx < 0 or step_idx >= p.nsteps:
            continue
        if step_idx in used_steps:
            continue
        snapshot_steps[label] = step_idx
        used_steps.add(step_idx)
    step_to_snapshot_label = {v: k for k, v in snapshot_steps.items()}
    print("Snapshot steps:", snapshot_steps)

    snapshots: dict[str, dict] = {}

    for n in range(p.nsteps):
        h = gap_schedule(p, n)

        if p.motion_mode == "bilateral":
            z, dz, eps, D, rho_f, mask_A, mask_gap, mask_B, zA_end, zB_start = build_regions_bilateral(p, h)
        else:
            z, dz, eps, D, rho_f, mask_A, mask_gap, mask_B, zA_end, zB_start = build_regions(p, h)

        # Regions outside the A|gap|B assembly, if any exist
        mask_outside = ~(mask_A | mask_gap | mask_B)

        # Structural checks:
        # 1) A, central gap, and B should not overlap
        overlap = (
            mask_A.astype(int) + mask_gap.astype(int) + mask_B.astype(int)
        )
        if np.any(overlap > 1):
            bad = np.where(overlap > 1)[0]
            raise RuntimeError(
                f"Region overlap at step {n}: indices {bad[:10]} "
                f"(showing up to 10 bad indices)"
            )

        # 2) Full-domain partition including outside electrolyte
        cover_full = (
            mask_A.astype(int)
            + mask_gap.astype(int)
            + mask_B.astype(int)
            + mask_outside.astype(int)
        )
        if not np.all(cover_full == 1):
            bad = np.where(cover_full != 1)[0]
            raise RuntimeError(
                f"Full-domain partition failure at step {n}: indices {bad[:10]} "
                f"(showing up to 10 bad indices)"
            )

        gap_idx = np.where(mask_gap)[0]
        if gap_idx.size > 2:
            mid_i = gap_idx[0] + gap_idx.size // 2
            mid_half = mid_i - 1  # J_half is defined on half nodes / interfaces
        else:
            mid_half = (len(z) - 1) // 2

        # Charge density at current time level
        rho = rho_f + F * (p.z_plus * c_plus + p.z_minus * c_minus)  # C/m^3
        phi = solve_poisson_dirichlet(eps=eps, rho=rho, dz=dz)

        # ---- FAIL FAST: check Poisson output ----
        if not np.isfinite(phi).all():
            raise FloatingPointError(
                f"NaN/inf in phi at step {n}. Reduce dt or check parameters."
            )

        c_plus, Jp_half = step_species_reservoir(c_plus, p.z_plus, D, phi, dz, p.dt, p)
        c_minus, Jm_half = step_species_reservoir(c_minus, p.z_minus, D, phi, dz, p.dt, p)

        # Charge density after species update
        rho_sample = rho_f + F * (p.z_plus * c_plus + p.z_minus * c_minus)  # C/m^3

        # Optionally store synchronized charge-density snapshots with phi from this step.
        if n in step_to_snapshot_label:
            label = step_to_snapshot_label[n]
            snapshots[label] = {
                "step": n,
                "t_s": n * p.dt,
                "z_m": z.copy(),
                "phi_V": phi.copy(),
                "rho_C_per_m3": rho_sample.copy(),
                "c_plus_molm3": c_plus.copy(),
                "c_minus_molm3": c_minus.copy(),
                "h_gap_m": h,
            }
            Q_mobile = F * np.trapezoid(c_plus.copy() - c_minus.copy(), z.copy())
            Q_fixed  = np.trapezoid(rho_f.copy(), z.copy())
            Q_total  = np.trapezoid(rho_sample.copy(), z.copy())
            print("mobile",Q_mobile)
            print("fixed",Q_fixed)
            print("total",Q_total)

        if (n % p.sample_every == 0) or (n == p.nsteps - 1):
            qdens = rho_sample  # C/m^3

            # Region-integrated charges per unit area
            QA = float(np.trapezoid(qdens * mask_A.astype(float), z))
            QB = float(np.trapezoid(qdens * mask_B.astype(float), z))
            Qgap = float(np.trapezoid(qdens * mask_gap.astype(float), z)) # central inter-slab gap
            Qoutside = float(np.trapezoid(qdens * mask_outside.astype(float), z))
            Qtot = float(np.trapezoid(qdens, z))

            # Consistency residual for the full-domain partition
            Qpartition_resid = Qtot - (QA + QB + Qgap + Qoutside)

            def avg_phi(mask, side: str, npts: int = 12) -> float:
                idx = np.where(mask)[0]
                if idx.size == 0:
                    return float("nan")
                if idx.size <= npts:
                    return float(np.mean(phi[idx]))
                return float(np.mean(phi[idx[-npts:]] if side == "right" else phi[idx[:npts]]))

            phi_A = avg_phi(mask_A, "right")
            phi_gap_L = avg_phi(mask_gap, "left")
            phi_gap_R = avg_phi(mask_gap, "right")
            phi_B = avg_phi(mask_B, "left")

            dphi_A = float(phi_gap_L - phi_A)
            dphi_B = float(phi_gap_R - phi_B)

            # Local mid-gap charge current density
            Jcharge = float(F * (p.z_plus * Jp_half[mid_half] + p.z_minus * Jm_half[mid_half]))

            # Interface-following gap-side windows
            w = 20e-9  # 20 nm

            mask_win_Agap = (z >= zA_end) & (z < min(zA_end + w, zB_start))
            mask_win_Bgap = (z > max(zB_start - w, zA_end)) & (z <= zB_start)

            Qwin_Agap = float(np.trapezoid(qdens * mask_win_Agap.astype(float), z))
            Qwin_Bgap = float(np.trapezoid(qdens * mask_win_Bgap.astype(float), z))

            history.append({
                "step": n,
                "t_s": n * p.dt,
                "h_gap_m": h,
                "QA_C_per_m2": QA,
                "QB_C_per_m2": QB,
                "Qgap_C_per_m2": Qgap,
                "Qoutside_C_per_m2": Qoutside,
                "Qtot_C_per_m2": Qtot,
                "Qpartition_resid_C_per_m2": Qpartition_resid,
                "Qwin_Agap_C_per_m2": Qwin_Agap,
                "Qwin_Bgap_C_per_m2": Qwin_Bgap,
                "dphi_A_V": dphi_A,
                "dphi_B_V": dphi_B,
                "Jcharge_mid_A_per_m2": Jcharge,
            })

        if n % 10000 == 0:
            print(
                f"step={n}, h={h:.3e}, "
                f"nout={np.count_nonzero(mask_outside)}, "
                f"nAwin={np.count_nonzero(mask_win_Agap)}, "
                f"nBwin={np.count_nonzero(mask_win_Bgap)}, "
                f"Qpart_resid={Qpartition_resid:.3e}"
            )

    return {
        "z_m": z,
        "phi_V": phi,
        "c_plus_molm3": c_plus,
        "c_minus_molm3": c_minus,
        "snapshots": snapshots,
        "history": history,
        "protocol": {
            "n_pre": p.n_pre,
            "n_approach": p.n_approach,
            "n_hold": p.n_hold,
            "n_separate": p.n_separate,
            "dt": p.dt,
        },
    }
