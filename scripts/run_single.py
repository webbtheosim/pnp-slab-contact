from __future__ import annotations

import argparse
from pathlib import Path

from pnpcc import PNPParams, run
from pnpcc.io import write_history_csv, write_snapshots_npz, write_metadata_json


def physics_params(case: str) -> dict:
    """
    Physics-only parameters.
    """
    common_physics = dict(
        N=3000,
        L=1400e-9,
        sample_every=1,
        c_bulk_molm3=150.0,   # 10 mM
        h_gap0=200e-9,
        h_min=200e-9,
        tA=200e-9,
        tB=200e-9,
        motion_mode="bilateral",
        charge_mode="surface_shell",
    )

    if case in {"baseline","baseline_slow"}:
        return dict(
            **common_physics,
            eps_dense_r=80.0,
            D_dense_scale=1.0,
            rho_f_A=0.0,
            rho_f_B=0.0,
        )

    if case in {
        "sym_ref",
        "sym_ref_fast",
        "sym_ref_medium",
        "sym_ref_slow",
        "sym_ref_fast_dt2",
        "sym_ref_fast_dt4",
    }:
        return dict(
            **common_physics,
            eps_dense_r=60.0,
            D_dense_scale=0.01,
            rho_f_A=-2e4,
            rho_f_B=-2e4,
        )

    if case in {
        "neg_swap",
        "neg_swap_fast",
        "neg_swap_medium",
        "neg_swap_slow",
    }:
        return dict(
            **common_physics,
            eps_dense_r=60.0,
            D_dense_scale=0.01,
            rho_f_A=-4e4,
            rho_f_B=-1e4,
        )

    if case in {
        "sym_soft",
        "sym_soft_fast",
        "sym_soft_medium",
        "sym_soft_slow",
        "sym_soft_fast_dt2",
        "sym_soft_fast_dt4",
    }:
        return dict(
            **common_physics,
            eps_dense_r=60.0,
            D_dense_scale=0.01,
            rho_f_A=+2e4,
            rho_f_B=-2e4,
        )

    if case in {
        "neg_ref",
        "neg_ref_fast",
        "neg_ref_medium",
        "neg_ref_slow",
        "neg_ref_long_eq",
    }:
        return dict(
            **common_physics,
            eps_dense_r=60.0,
            D_dense_scale=0.01,
            rho_f_A=-2e6,
            rho_f_B=-8e6,
        )

    if case in {
        "neg_sym",
        "neg_sym_fast",
        "neg_sym_medium",
        "neg_sym_slow",
    }:
        return dict(
            **common_physics,
            eps_dense_r=60.0,
            D_dense_scale=0.01,
            rho_f_A=-2e4,
            rho_f_B=-2e4,
        )

    raise ValueError(f"Unknown physics case '{case}'.")

def protocol_params(case: str) -> dict:
    """
    Protocol / temporal-discretization parameters.

    Cases are mapped onto a small number of reusable protocol templates.
    nsteps is computed automatically from the phase counts.
    """

    protocol_templates = {
        "fast": dict(
            dt=1e-12,
            n_pre=100000,
            n_approach=1000,
            n_hold=1000,
            n_separate=1000,
            n_relax=2000,
        ),
        "medium": dict(
            dt=1e-12,
            n_pre=30000,
            n_approach=1200,
            n_hold=800,
            n_separate=2000,
            n_relax=2000,
        ),
        "slow": dict(
            dt=1e-12,
            n_pre=50000,
            n_approach=100000,
            n_hold=20000,
            n_separate=100000,
            n_relax=150000,
        ),
        "long_eq": dict(
            dt=2e-12,
            n_pre=100000,
            n_approach=50000,
            n_hold=20000,
            n_separate=50000,
            n_relax=100000,
        ),
        "fast_dt4": dict(
            dt=2.5e-12,
            n_pre=120000,
            n_approach=480,
            n_hold=320,
            n_separate=800,
            n_relax=8000,
        ),
    }

    case_to_protocol = {
        "baseline": "fast",
        "sym_ref": "fast",
        "sym_soft": "fast",
        "neg_ref_fast": "fast",
        "sym_ref_fast": "fast",
        "sym_soft_fast": "fast",

        "sym_ref_medium": "medium",
        "sym_soft_medium": "medium",
        "neg_ref_medium": "medium",

        "baseline_slow": "slow",
        "sym_ref_slow": "slow",
        "sym_soft_slow": "slow",
        "neg_ref_slow": "slow",
        "neg_swap_slow": "slow",
        "neg_sym_slow": "slow",

        "neg_ref_long_eq": "long_eq",
        "sym_ref_fast_dt2": "fast_dt2",
        "sym_soft_fast_dt2": "fast_dt2",

        "sym_ref_fast_dt4": "fast_dt4",
        "sym_soft_fast_dt4": "fast_dt4",
    }

    if case not in case_to_protocol:
        raise ValueError(f"Unknown protocol case '{case}'.")

    p = dict(protocol_templates[case_to_protocol[case]])
    p["nsteps"] = p["n_pre"] + p["n_approach"] + p["n_hold"] + p["n_separate"] + p["n_relax"]
    return p


    raise ValueError(f"Unknown protocol case '{case}'.")


def build_params(case: str) -> PNPParams:
    """
    Combine physics and protocol parameters into one PNPParams object.
    """
    p_phys = physics_params(case)
    p_protocol = protocol_params(case)
    return PNPParams(**p_phys, **p_protocol)


def main():
    parser = argparse.ArgumentParser(description="Run 1D PNP contact simulation.")
    parser.add_argument(
        "--case",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "baseline_slow",
            "sym_ref",
            "sym_soft",
            "sym_ref_fast",
            "sym_ref_medium",
            "sym_ref_slow",
            "sym_ref_fast_dt2",
            "sym_ref_fast_dt4",
            "sym_soft_fast",
            "sym_soft_medium",
            "sym_soft_slow",
            "sym_soft_fast_dt2",
            "sym_soft_fast_dt4",
            "neg_ref_fast",
            "neg_ref_medium",
            "neg_ref_slow",
            "neg_ref_long_eq",
            "neg_sym_slow",
            "neg_swap_slow"
        ],
        help="Which parameter preset to run.",
    )
    args = parser.parse_args()

    print(f"\nRunning case: {args.case}")

    p = build_params(args.case)
    out = run(p)

    out_path = Path(f"data/run_{args.case}_neumann.csv")
    snap_path = Path(f"data/run_{args.case}_neumann_snapshots.npz")
    meta_path = Path(f"data/run_{args.case}_neumann_meta.json")

    print("c_plus range:",
          float(out["c_plus_molm3"].min()),
          float(out["c_plus_molm3"].max()))
    print("dt:", p.dt)
    print("n_pre:", p.n_pre)
    print("n_approach:", p.n_approach)
    print("n_hold:", p.n_hold)
    print("n_separate:", p.n_separate)
    print("nsteps:", p.nsteps)
    print("motion:", p.motion_mode)

    write_history_csv(out_path, out["history"])
    print(f"Wrote {out_path} with {len(out['history'])} rows.")

    write_snapshots_npz(snap_path, out["snapshots"])
    print(f"Wrote {snap_path}")

    metadata = {
        "case": args.case,
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
        "csv_file": str(out_path),
        "snapshot_file": str(snap_path),
    }

    write_metadata_json(meta_path, metadata)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
