from __future__ import annotations

import argparse
from pathlib import Path

from pnpcc import PNPParams, run
from pnpcc.io import write_history_csv, write_snapshots_npz, write_metadata_json


# -----------------------------------------------------------------------------
# Core baseline
# -----------------------------------------------------------------------------
def base_physics() -> dict:
    """
    Fair baseline aligned with the latest quasi-static, high-salt reference.
    """
    return dict(
        N=3000,
        L=1500e-9,
        sample_every=1,
        c_bulk_molm3=150.0,
        h_gap0=200e-9,
        h_min=2e-9,
        tA=200e-9,
        tB=200e-9,
        motion_mode="bilateral",
        charge_mode="surface_shell",
        eps_dense_r=60.0,
        D_dense_scale=0.05,
    )


def base_protocol() -> dict:
    """
    Single protocol used for all sweep cases.
    Keep a long relaxation window and analyze different times afterward.
    """
    p = dict(
        dt=1e-12,
        n_pre=100000,
        n_approach=200000,
        n_hold=50000,
        n_separate=200000,
        n_relax=50000,
    )
    p["nsteps"] = (
        p["n_pre"]
        + p["n_approach"]
        + p["n_hold"]
        + p["n_separate"]
        + p["n_relax"]
    )
    return p


# -----------------------------------------------------------------------------
# Sweep families
# -----------------------------------------------------------------------------
def physics_params(case: str) -> dict:
    """
    Physics-only parameters.

    Naming convention:
      ctrl_XX    = controls
      asym_XX    = fixed-charge asymmetry sweep
      gap_XX     = h_min sweep
      diff_XX    = condensed-phase diffusivity sweep
      rate_XX    = protocol-rate sweep placeholder in physics (same baseline)
      salt_XX    = salt sweep placeholder in physics (same baseline)
    """
    common = base_physics()

    # --- controls ---
    controls = {
        "ctrl_00": dict(rho_f_A=-0e6,   rho_f_B=-0e6),      # neutral
        "ctrl_01": dict(rho_f_A=-2e6,  rho_f_B=-2e6),     # symmetric negative
        "ctrl_02": dict(rho_f_A=-2e6,  rho_f_B=-12e6),     # reference asymmetric
        "ctrl_03": dict(rho_f_A=-12e6,  rho_f_B=-2e6),     # swapped asymmetric
    }

    # --- asymmetry sweep: 5 points ---
    # Keep rho_f_A fixed and vary rho_f_B
    asymmetry = {
        "asym_01": dict(rho_f_A=-2e6, rho_f_B=-2e6),
        "asym_02": dict(rho_f_A=-2e6, rho_f_B=-4e6),
        "asym_03": dict(rho_f_A=-2e6, rho_f_B=-8e6),
        "asym_04": dict(rho_f_A=-2e6, rho_f_B=-12e6),
        "asym_05": dict(rho_f_A=-2e6, rho_f_B=-16e6),
    }

    # --- minimum-gap sweep: 5 points ---
    gap_sweep = {
        "gap_01": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=1e-9),
        "gap_02": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=2e-9),
        "gap_03": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=5e-9),
        "gap_04": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=10e-9),
        "gap_05": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=20e-9),
        "gap_06": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=40e-9),
        "gap_07": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=80e-9),
        "gap_08": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=120e-9),
        "gap_09": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=160e-9),
        "gap_10": dict(rho_f_A=-2e6, rho_f_B=-12e6, h_min=200e-9),
    }

    # --- condensed-phase diffusivity sweep: 5 points ---
    diff_sweep = {
        "diff_01": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=0.05),
        "diff_02": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=0.1),
        "diff_03": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=0.2),
        "diff_04": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=0.3),
        "diff_05": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=0.4),
        "diff_06": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=0.5),
        "diff_07": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=0.75),
        "diff_08": dict(rho_f_A=-2e6, rho_f_B=-12e6, D_dense_scale=1.0),
    }

    # --- rate sweep uses same physics as reference ---
    rate_sweep = {
        "rate_01": dict(rho_f_A=-2e6, rho_f_B=-12e6),
        "rate_02": dict(rho_f_A=-2e6, rho_f_B=-12e6),
        "rate_03": dict(rho_f_A=-2e6, rho_f_B=-12e6),
        "rate_04": dict(rho_f_A=-2e6, rho_f_B=-12e6),
        "rate_05": dict(rho_f_A=-2e6, rho_f_B=-12e6),
    }

    salt_sweep = {
        "salt_01": dict(rho_f_A=-2e6, rho_f_B=-12e6, c_bulk_molm3=0.),
        "salt_02": dict(rho_f_A=-2e6, rho_f_B=-12e6, c_bulk_molm3=10.0 ),
        "salt_03": dict(rho_f_A=-2e6, rho_f_B=-12e6, c_bulk_molm3=25.0),
        "salt_04": dict(rho_f_A=-2e6, rho_f_B=-12e6, c_bulk_molm3=50.0),
        "salt_05": dict(rho_f_A=-2e6, rho_f_B=-12e6, c_bulk_molm3=100.0),
    }

    all_cases = {}
    all_cases.update(controls)
    all_cases.update(asymmetry)
    all_cases.update(gap_sweep)
    all_cases.update(diff_sweep)
    all_cases.update(rate_sweep)
    all_cases.update(salt_sweep)

    if case not in all_cases:
        raise ValueError(f"Unknown physics case '{case}'.")

    params = dict(common)
    params.update(all_cases[case])
    return params

def protocol_params(case: str) -> dict:
    """
    Protocol / temporal-discretization parameters.

    All families use a long relax period.
    The only varied protocol family is rate_XX, where we change
    the approach/separation durations while keeping the same dt and long relax.
    """
    common = base_protocol()

    rate_cases = {
        # Faster to slower approach/separation; hold fixed
        "rate_01": dict(n_approach=50000,   n_separate=50000),
        "rate_02": dict(n_approach=100000,  n_separate=100000),
        "rate_03": dict(n_approach=500000,  n_separate=500000),
        "rate_04": dict(n_approach=1000000, n_separate=1000000),
        "rate_05": dict(n_approach=2000000, n_separate=2000000),
    }

    if case in rate_cases:
        p = dict(common)
        p.update(rate_cases[case])
    else:
        p = dict(common)

    p["nsteps"] = (
        p["n_pre"]
        + p["n_approach"]
        + p["n_hold"]
        + p["n_separate"]
        + p["n_relax"]
    )
    return p


# -----------------------------------------------------------------------------
# Assembly
# -----------------------------------------------------------------------------
def build_params(case: str) -> PNPParams:
    p_phys = physics_params(case)
    p_protocol = protocol_params(case)
    return PNPParams(**p_phys, **p_protocol)


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------
ALL_CASES = [
    "ctrl_00",
    "ctrl_01",
    "ctrl_02",
    "ctrl_03",
    "asym_01",
    "asym_02",
    "asym_03",
    "asym_04",
    "asym_05",
    "gap_01",
    "gap_02",
    "gap_03",
    "gap_04",
    "gap_05",
    "gap_06",
    "gap_07",
    "gap_08",
    "gap_09",
    "gap_10",
    "diff_01",
    "diff_02",
    "diff_03",
    "diff_04",
    "diff_05",
    "diff_06",
    "diff_07",
    "diff_08",
    "rate_01",
    "rate_02",
    "rate_03",
    "rate_04",
    "rate_05",
    "salt_01",
    "salt_02",
    "salt_03",
    "salt_04",
    "salt_05",
]


def run_case(case: str, outdir: Path, tag: str):
    print(f"\nRunning case: {case}")

    p = build_params(case)
    out = run(p)

    stem = f"run_{case}_{tag}"
    out_path = outdir / f"{stem}.csv"
    snap_path = outdir / f"{stem}_snapshots.npz"
    meta_path = outdir / f"{stem}_meta.json"

    print(
        "c_plus range:",
        float(out["c_plus_molm3"].min()),
        float(out["c_plus_molm3"].max()),
    )
    print("dt:", p.dt)
    print("n_pre:", p.n_pre)
    print("n_approach:", p.n_approach)
    print("n_hold:", p.n_hold)
    print("n_separate:", p.n_separate)
    print("n_relax:", p.n_relax)
    print("nsteps:", p.nsteps)
    print("motion:", p.motion_mode)

    write_history_csv(out_path, out["history"])
    print(f"Wrote {out_path} with {len(out['history'])} rows.")

    write_snapshots_npz(snap_path, out["snapshots"])
    print(f"Wrote {snap_path}")

    metadata = {
        "case": case,
        "dt": p.dt,
        "n_pre": p.n_pre,
        "n_approach": p.n_approach,
        "n_hold": p.n_hold,
        "n_separate": p.n_separate,
        "n_relax": p.n_relax,
        "nsteps": p.nsteps,
        "N": p.N,
        "L_m": p.L,
        "h_gap0_m": p.h_gap0,
        "h_min_m": p.h_min,
        "tA_m": p.tA,
        "tB_m": p.tB,
        "c_bulk_molm3": p.c_bulk_molm3,
        "eps_dense_r": p.eps_dense_r,
        "D_dense_scale": p.D_dense_scale,
        "rho_f_A_C_per_m3": p.rho_f_A,
        "rho_f_B_C_per_m3": p.rho_f_B,
        "motion_mode": p.motion_mode,
        "csv_file": str(out_path),
        "snapshot_file": str(snap_path),
    }

    write_metadata_json(meta_path, metadata)
    print(f"Wrote {meta_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run tidy sweep presets for 1D PNP contact simulations."
    )
    parser.add_argument(
        "--case",
        type=str,
        default="ctrl_02",
        choices=ALL_CASES + ["all", "controls", "asym", "gap", "diff", "rate","salt"],
        help="Case or family to run.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="sweep",
        help="Tag appended to output filenames.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.case == "all":
        cases = ALL_CASES
    elif args.case == "controls":
        cases = [c for c in ALL_CASES if c.startswith("ctrl_")]
    elif args.case == "asym":
        cases = [c for c in ALL_CASES if c.startswith("asym_")]
    elif args.case == "gap":
        cases = [c for c in ALL_CASES if c.startswith("gap_")]
    elif args.case == "diff":
        cases = [c for c in ALL_CASES if c.startswith("diff_")]
    elif args.case == "rate":
        cases = [c for c in ALL_CASES if c.startswith("rate_")]
    elif args.case == "salt":
        cases = [c for c in ALL_CASES if c.startswith("salt_")]
    else:
        cases = [args.case]

    for case in cases:
        run_case(case, outdir=outdir, tag=args.tag)


if __name__ == "__main__":
    main()
