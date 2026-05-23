from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def infer_meta_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.endswith("_snapshots"):
        stem = stem[: -len("_snapshots")]
    return csv_path.with_name(f"{stem}_meta.json")


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
def phase_times_us(meta: dict) -> dict:
    dt_us = float(meta["dt"]) * 1e6

    n_pre = int(meta["n_pre"])
    n_approach = int(meta["n_approach"])
    n_hold = int(meta["n_hold"])
    n_separate = int(meta["n_separate"])
    n_relax = int(meta["n_relax"])

    # phase boundaries in step index convention used by history output
    i_pre_end = max(0, n_pre - 1)
    i_approach_start = n_pre
    i_approach_end = n_pre + n_approach - 1
    i_hold_start = n_pre + n_approach
    i_hold_end = n_pre + n_approach + n_hold - 1
    i_separate_start = n_pre + n_approach + n_hold
    i_separate_end = n_pre + n_approach + n_hold + n_separate - 1
    i_relax_start = n_pre + n_approach + n_hold + n_separate
    i_relax_end = n_pre + n_approach + n_hold + n_separate + n_relax - 1

    return {
        "pre_end_us": i_pre_end * dt_us,
        "approach_start_us": i_approach_start * dt_us,
        "approach_end_us": i_approach_end * dt_us,
        "hold_start_us": i_hold_start * dt_us,
        "hold_end_us": i_hold_end * dt_us,
        "separate_start_us": i_separate_start * dt_us,
        "separate_end_us": i_separate_end * dt_us,
        "relax_start_us": i_relax_start * dt_us,
        "relax_end_us": i_relax_end * dt_us,
    }


def nearest_idx(t_us: np.ndarray, target_us: float) -> int:
    return int(np.argmin(np.abs(t_us - target_us)))


def relax_probe_times_us(meta: dict, fractions: List[float]) -> Dict[str, float]:
    times = phase_times_us(meta)
    t0 = times["relax_start_us"]
    t1 = times["relax_end_us"]
    out = {}
    for frac in fractions:
        frac = float(frac)
        out[f"relax_{int(round(100*frac)):02d}_us"] = t0 + frac * (t1 - t0)
    return out


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------
def analyze_one(csv_path: Path, meta_path: Path, relax_fracs: List[float]) -> dict:
    df = pd.read_csv(csv_path)
    meta = load_meta(meta_path)

    required = [
        "t_s",
        "QA_C_per_m2",
        "QB_C_per_m2",
        "dphi_A_V",
        "dphi_B_V",
        "Jcharge_mid_A_per_m2",
        "h_gap_m",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {missing}")

    t_us = df["t_s"].to_numpy(dtype=float) * 1e6
    QA = df["QA_C_per_m2"].to_numpy(dtype=float)
    QB = df["QB_C_per_m2"].to_numpy(dtype=float)
    dphi_A_mV = 1e3 * df["dphi_A_V"].to_numpy(dtype=float)
    dphi_B_mV = 1e3 * df["dphi_B_V"].to_numpy(dtype=float)
    J = df["Jcharge_mid_A_per_m2"].to_numpy(dtype=float)
    h_nm = 1e9 * df["h_gap_m"].to_numpy(dtype=float)

    has_qwin = ("Qwin_Agap_C_per_m2" in df.columns) and ("Qwin_Bgap_C_per_m2" in df.columns)
    if has_qwin:
        QwinA = df["Qwin_Agap_C_per_m2"].to_numpy(dtype=float)
        QwinB = df["Qwin_Bgap_C_per_m2"].to_numpy(dtype=float)

    # phase landmark indices
    pt = phase_times_us(meta)
    idx_pre_end = nearest_idx(t_us, pt["pre_end_us"])
    idx_hold_end = nearest_idx(t_us, pt["hold_end_us"])
    idx_sep_end = nearest_idx(t_us, pt["separate_end_us"])
    idx_relax_start = nearest_idx(t_us, pt["relax_start_us"])
    idx_relax_end = nearest_idx(t_us, pt["relax_end_us"])

    # baseline referenced quantities
    QA_eq = QA[idx_pre_end]
    QB_eq = QB[idx_pre_end]
    dphi_A_eq = dphi_A_mV[idx_pre_end]
    dphi_B_eq = dphi_B_mV[idx_pre_end]

    dQA = QA - QA_eq
    dQB = QB - QB_eq
    dQdiff = dQB - dQA

    ddphi_A = dphi_A_mV - dphi_A_eq
    ddphi_B = dphi_B_mV - dphi_B_eq
    ddphi_diff = ddphi_B - ddphi_A

    # post-equilibration arrays
    post = slice(idx_pre_end, None)
    t_post_us = t_us[post]
    J_post = J[post]
    J_abs_post = np.abs(J_post)
    h_post_nm = h_nm[post]
    ddphi_A_post = ddphi_A[post]
    ddphi_B_post = ddphi_B[post]
    dQdiff_post = dQdiff[post]

    # scalar summaries
    row: Dict[str, Any] = {
        "case": meta.get("case", csv_path.stem),
        "csv_file": str(csv_path),
        "meta_file": str(meta_path),
        "dt_s": float(meta["dt"]),
        "n_pre": int(meta["n_pre"]),
        "n_approach": int(meta["n_approach"]),
        "n_hold": int(meta["n_hold"]),
        "n_separate": int(meta["n_separate"]),
        "n_relax": int(meta["n_relax"]),
        "N": int(meta["N"]),
        "L_m": float(meta["L_m"]),
        "h_gap0_m": float(meta["h_gap0_m"]),
        "h_min_m": float(meta["h_min_m"]),
        "tA_m": float(meta["tA_m"]),
        "tB_m": float(meta["tB_m"]),
        "c_bulk_molm3": float(meta["c_bulk_molm3"]),
        "eps_dense_r": float(meta["eps_dense_r"]),
        "D_dense_scale": float(meta["D_dense_scale"]),
        "rho_f_A_C_per_m3": float(meta["rho_f_A_C_per_m3"]),
        "rho_f_B_C_per_m3": float(meta["rho_f_B_C_per_m3"]),
        "motion_mode": meta.get("motion_mode", ""),
        "t_pre_end_us": pt["pre_end_us"],
        "t_hold_end_us": pt["hold_end_us"],
        "t_separate_end_us": pt["separate_end_us"],
        "t_relax_start_us": pt["relax_start_us"],
        "t_relax_end_us": pt["relax_end_us"],
        "QA_eq_C_per_m2": QA_eq,
        "QB_eq_C_per_m2": QB_eq,
        "dphi_A_eq_mV": dphi_A_eq,
        "dphi_B_eq_mV": dphi_B_eq,
        "dQA_hold_end_C_per_m2": dQA[idx_hold_end],
        "dQB_hold_end_C_per_m2": dQB[idx_hold_end],
        "dQdiff_hold_end_C_per_m2": dQdiff[idx_hold_end],
        "dQA_separate_end_C_per_m2": dQA[idx_sep_end],
        "dQB_separate_end_C_per_m2": dQB[idx_sep_end],
        "dQdiff_separate_end_C_per_m2": dQdiff[idx_sep_end],
        "dQA_relax_end_C_per_m2": dQA[idx_relax_end],
        "dQB_relax_end_C_per_m2": dQB[idx_relax_end],
        "dQdiff_relax_end_C_per_m2": dQdiff[idx_relax_end],
        "persistence_ratio": (
            abs(dQdiff[idx_relax_end]) / abs(dQdiff[idx_sep_end])
            if abs(dQdiff[idx_sep_end]) > 0 else np.nan
        ),
        "max_abs_dQA_post_C_per_m2": float(np.max(np.abs(dQA[post]))),
        "max_abs_dQB_post_C_per_m2": float(np.max(np.abs(dQB[post]))),
        "max_abs_dQdiff_post_C_per_m2": float(np.max(np.abs(dQdiff_post))),
        "max_abs_ddphi_A_post_mV": float(np.max(np.abs(ddphi_A_post))),
        "max_abs_ddphi_B_post_mV": float(np.max(np.abs(ddphi_B_post))),
        "max_abs_ddphi_diff_post_mV": float(np.max(np.abs(ddphi_diff[post]))),
        "J_abs_integral_post_C_per_m2": float(np.trapezoid(J_abs_post, t_post_us * 1e-6)),
        "J_rms_post_A_per_m2": float(np.sqrt(np.mean(J_post**2))),
        "J_p95_post_A_per_m2": float(np.percentile(J_abs_post, 95)),
        "J_max_abs_post_A_per_m2": float(np.max(J_abs_post)),
    }

    imax = int(np.argmax(J_abs_post))
    row["t_at_J_max_post_us"] = float(t_post_us[imax])
    row["h_at_J_max_post_nm"] = float(h_post_nm[imax])

    # optional total charge diagnostics
    if "Qtot_C_per_m2" in df.columns:
        Qtot = df["Qtot_C_per_m2"].to_numpy(dtype=float)
        row["Qtot_abs_max_C_per_m2"] = float(np.max(np.abs(Qtot)))
        row["Qtot_drift_C_per_m2"] = float(Qtot[-1] - Qtot[idx_pre_end])

    # optional window metrics
    if has_qwin:
        QwinA_eq = QwinA[idx_pre_end]
        QwinB_eq = QwinB[idx_pre_end]
        dQwinA = QwinA - QwinA_eq
        dQwinB = QwinB - QwinB_eq
        dQwin_diff = dQwinB - dQwinA

        row.update({
            "QwinA_eq_C_per_m2": QwinA_eq,
            "QwinB_eq_C_per_m2": QwinB_eq,
            "dQwinA_hold_end_C_per_m2": dQwinA[idx_hold_end],
            "dQwinB_hold_end_C_per_m2": dQwinB[idx_hold_end],
            "dQwin_diff_hold_end_C_per_m2": dQwin_diff[idx_hold_end],
            "dQwinA_separate_end_C_per_m2": dQwinA[idx_sep_end],
            "dQwinB_separate_end_C_per_m2": dQwinB[idx_sep_end],
            "dQwin_diff_separate_end_C_per_m2": dQwin_diff[idx_sep_end],
            "dQwinA_relax_end_C_per_m2": dQwinA[idx_relax_end],
            "dQwinB_relax_end_C_per_m2": dQwinB[idx_relax_end],
            "dQwin_diff_relax_end_C_per_m2": dQwin_diff[idx_relax_end],
            "max_abs_dQwin_diff_post_C_per_m2": float(np.max(np.abs(dQwin_diff[post]))),
        })

    # relaxation probe times
    probe_times = relax_probe_times_us(meta, relax_fracs)
    for key, t_probe_us in probe_times.items():
        idx = nearest_idx(t_us, t_probe_us)
        suffix = key.replace("_us", "")
        row[f"t_{suffix}_us"] = t_probe_us
        row[f"dQdiff_{suffix}_C_per_m2"] = dQdiff[idx]
        row[f"ddphi_diff_{suffix}_mV"] = ddphi_diff[idx]
        if has_qwin:
            row[f"dQwin_diff_{suffix}_C_per_m2"] = dQwin_diff[idx]

    return row


# -----------------------------------------------------------------------------
# Directory scan
# -----------------------------------------------------------------------------
def find_csvs(indir: Path, pattern: str) -> List[Path]:
    csvs = sorted(indir.glob(pattern))
    csvs = [p for p in csvs if p.is_file() and not p.name.endswith("_shells.csv")]
    return csvs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Summarize sweep run history CSVs into one analysis table."
    )
    parser.add_argument(
        "--indir",
        default="data",
        help="Directory containing run_*.csv and matching *_meta.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="run_*.csv",
        help="Glob pattern for input CSVs.",
    )
    parser.add_argument(
        "--out",
        default="data/sweep_summary.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--relax-fracs",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Fractions of the relax interval at which to record values.",
    )
    args = parser.parse_args()

    indir = Path(args.indir)
    out = Path(args.out)

    csvs = find_csvs(indir, args.pattern)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {indir} matching pattern '{args.pattern}'.")

    rows = []
    skipped = []

    for csv_path in csvs:
        meta_path = infer_meta_path(csv_path)
        if not meta_path.exists():
            skipped.append((csv_path.name, "missing meta json"))
            continue
        try:
            rows.append(analyze_one(csv_path, meta_path, args.relax_fracs))
        except Exception as e:
            skipped.append((csv_path.name, str(e)))

    if not rows:
        raise RuntimeError(f"No runs were successfully analyzed. Skipped: {skipped}")

    df = pd.DataFrame(rows)

    # nicer ordering of common columns
    front_cols = [
        "case",
        "csv_file",
        "dt_s",
        "n_pre",
        "n_approach",
        "n_hold",
        "n_separate",
        "n_relax",
        "c_bulk_molm3",
        "D_dense_scale",
        "rho_f_A_C_per_m3",
        "rho_f_B_C_per_m3",
        "h_min_m",
        "dQdiff_hold_end_C_per_m2",
        "dQdiff_separate_end_C_per_m2",
        "dQdiff_relax_end_C_per_m2",
        "persistence_ratio",
        "max_abs_ddphi_diff_post_mV",
        "J_max_abs_post_A_per_m2",
    ]
    ordered_cols = [c for c in front_cols if c in df.columns] + [c for c in df.columns if c not in front_cols]
    df = df[ordered_cols]

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote summary CSV: {out}")
    print(f"Analyzed {len(rows)} runs.")
    if skipped:
        print("\nSkipped:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
