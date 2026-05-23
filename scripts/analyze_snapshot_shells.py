from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_snapshots(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Supports flattened keys like:
      pre_end__t_s, pre_end__z_m, ...
    and dict-like payloads.
    """
    data = np.load(path, allow_pickle=True)

    # Case 1: flattened keys
    flattened_labels = sorted(set(k.split("__")[0] for k in data.files if "__" in k))
    if flattened_labels:
        snaps: Dict[str, Dict[str, Any]] = {}
        for label in flattened_labels:
            required = [
                f"{label}__t_s",
                f"{label}__z_m",
                f"{label}__phi_V",
                f"{label}__rho_C_per_m3",
                f"{label}__c_plus_molm3",
                f"{label}__c_minus_molm3",
                f"{label}__h_gap_m",
            ]
            if all(k in data.files for k in required):
                snaps[label] = {
                    "t_s": float(np.asarray(data[f"{label}__t_s"]).squeeze()),
                    "z_m": np.asarray(data[f"{label}__z_m"], dtype=float),
                    "phi_V": np.asarray(data[f"{label}__phi_V"], dtype=float),
                    "rho_C_per_m3": np.asarray(data[f"{label}__rho_C_per_m3"], dtype=float),
                    "c_plus_molm3": np.asarray(data[f"{label}__c_plus_molm3"], dtype=float),
                    "c_minus_molm3": np.asarray(data[f"{label}__c_minus_molm3"], dtype=float),
                    "h_gap_m": float(np.asarray(data[f"{label}__h_gap_m"]).squeeze()),
                }
        if snaps:
            return snaps

    # Case 2: dict-like payloads
    snaps = {}
    for label in data.files:
        try:
            item = data[label].item()
            if isinstance(item, dict):
                snaps[label] = item
        except Exception:
            pass
    if snaps:
        return snaps

    raise ValueError(f"Could not interpret snapshot file: {path}")


def load_metadata(snapshot_path: Path, meta_path: str | Path | None = None) -> dict:
    if meta_path is not None:
        mp = Path(meta_path)
    else:
        stem = snapshot_path.name.replace("_snapshots.npz", "")
        mp = snapshot_path.with_name(f"{stem}_meta.json")

    if not mp.exists():
        raise FileNotFoundError(
            f"Could not find metadata JSON: {mp}\n"
            "Pass --meta explicitly if the auto-detected path is wrong."
        )

    with mp.open("r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def interface_positions_m(
    h_gap_m: float,
    tA_m: float,
    tB_m: float,
    motion_mode: str,
    L_m: float,
) -> Tuple[float, float, float, float]:
    """
    Returns:
      zA_start, zA_end, zB_start, zB_end
    """
    if motion_mode == "bilateral":
        gap_mid = 0.5 * L_m
        zA_end = gap_mid - 0.5 * h_gap_m
        zA_start = zA_end - tA_m
        zB_start = gap_mid + 0.5 * h_gap_m
        zB_end = zB_start + tB_m
    else:
        zA_start = 0.0
        zA_end = tA_m
        zB_start = tA_m + h_gap_m
        zB_end = zB_start + tB_m

    return zA_start, zA_end, zB_start, zB_end


def integrate_masked(z: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return float("nan")
    return float(np.trapz(y[idx], z[idx]))


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------
def analyze_snapshot(
    label: str,
    snap: Dict[str, Any],
    *,
    tA_m: float,
    tB_m: float,
    L_m: float,
    motion_mode: str,
    shell_nm: float,
) -> dict:
    z = np.asarray(snap["z_m"], dtype=float)
    rho = np.asarray(snap["rho_C_per_m3"], dtype=float)
    phi = np.asarray(snap["phi_V"], dtype=float)
    c_plus = np.asarray(snap["c_plus_molm3"], dtype=float)
    c_minus = np.asarray(snap["c_minus_molm3"], dtype=float)
    h_gap_m = float(snap["h_gap_m"])
    t_s = float(snap["t_s"])

    shell_m = shell_nm * 1e-9
    zA_start, zA_end, zB_start, zB_end = interface_positions_m(
        h_gap_m=h_gap_m,
        tA_m=tA_m,
        tB_m=tB_m,
        motion_mode=motion_mode,
        L_m=L_m,
    )

    # Full regions
    mask_A = (z >= zA_start) & (z <= zA_end)
    mask_gap = (z >= zA_end) & (z <= zB_start)
    mask_B = (z >= zB_start) & (z <= zB_end)

    # Slab full charges
    QA_full = integrate_masked(z, rho, mask_A)
    QB_full = integrate_masked(z, rho, mask_B)
    Qgap = integrate_masked(z, rho, mask_gap)

    # Contact-facing shells inside slabs
    mask_A_contact_shell = (z >= max(zA_start, zA_end - shell_m)) & (z <= zA_end)
    mask_B_contact_shell = (z >= zB_start) & (z <= min(zB_end, zB_start + shell_m))

    QA_contact_shell = integrate_masked(z, rho, mask_A_contact_shell)
    QB_contact_shell = integrate_masked(z, rho, mask_B_contact_shell)

    # Outer-facing shells inside slabs
    mask_A_outer_shell = (z >= zA_start) & (z <= min(zA_end, zA_start + shell_m))
    mask_B_outer_shell = (z >= max(zB_start, zB_end - shell_m)) & (z <= zB_end)

    QA_outer_shell = integrate_masked(z, rho, mask_A_outer_shell)
    QB_outer_shell = integrate_masked(z, rho, mask_B_outer_shell)

    # Optional gap-side shells
    mask_gap_left_shell = (z >= zA_end) & (z <= min(zB_start, zA_end + shell_m))
    mask_gap_right_shell = (z >= max(zA_end, zB_start - shell_m)) & (z <= zB_start)

    Qgap_left_shell = integrate_masked(z, rho, mask_gap_left_shell)
    Qgap_right_shell = integrate_masked(z, rho, mask_gap_right_shell)

    # Some simple concentration summaries in contact shells
    cplus_A_contact = integrate_masked(z, c_plus, mask_A_contact_shell) / shell_m if np.count_nonzero(mask_A_contact_shell) >= 2 else np.nan
    cminus_A_contact = integrate_masked(z, c_minus, mask_A_contact_shell) / shell_m if np.count_nonzero(mask_A_contact_shell) >= 2 else np.nan
    cplus_B_contact = integrate_masked(z, c_plus, mask_B_contact_shell) / shell_m if np.count_nonzero(mask_B_contact_shell) >= 2 else np.nan
    cminus_B_contact = integrate_masked(z, c_minus, mask_B_contact_shell) / shell_m if np.count_nonzero(mask_B_contact_shell) >= 2 else np.nan

    # Potential jump across contact zone, sampled simply
    def avg_in_interval(zlo: float, zhi: float, arr: np.ndarray) -> float:
        m = (z >= zlo) & (z <= zhi)
        if np.count_nonzero(m) < 1:
            return float("nan")
        return float(np.mean(arr[m]))

    phi_A_contact = avg_in_interval(max(zA_start, zA_end - shell_m), zA_end, phi)
    phi_gap_left = avg_in_interval(zA_end, min(zB_start, zA_end + shell_m), phi)
    phi_gap_right = avg_in_interval(max(zA_end, zB_start - shell_m), zB_start, phi)
    phi_B_contact = avg_in_interval(zB_start, min(zB_end, zB_start + shell_m), phi)

    dphi_A_contact_mV = 1e3 * (phi_gap_left - phi_A_contact)
    dphi_B_contact_mV = 1e3 * (phi_gap_right - phi_B_contact)

    return {
        "label": label,
        "t_us": 1e6 * t_s,
        "h_nm": 1e9 * h_gap_m,
        "QA_full_C_per_m2": QA_full,
        "QB_full_C_per_m2": QB_full,
        "Qgap_C_per_m2": Qgap,
        "QA_contact_shell_C_per_m2": QA_contact_shell,
        "QB_contact_shell_C_per_m2": QB_contact_shell,
        "QA_outer_shell_C_per_m2": QA_outer_shell,
        "QB_outer_shell_C_per_m2": QB_outer_shell,
        "Qgap_left_shell_C_per_m2": Qgap_left_shell,
        "Qgap_right_shell_C_per_m2": Qgap_right_shell,
        "dphi_A_contact_mV": dphi_A_contact_mV,
        "dphi_B_contact_mV": dphi_B_contact_mV,
        "cplus_A_contact_molm3_avg": cplus_A_contact,
        "cminus_A_contact_molm3_avg": cminus_A_contact,
        "cplus_B_contact_molm3_avg": cplus_B_contact,
        "cminus_B_contact_molm3_avg": cminus_B_contact,
    }


def ordered_labels(snaps: Dict[str, Dict[str, Any]]) -> List[str]:
    def sort_key(label: str):
        phase_order = {"pre": 0, "approach": 1, "hold": 2, "separate": 3, "relax": 4}
        pos_order = {"start": 0, "mid": 1, "end": 2}

        parts = label.split("_")
        if len(parts) == 2 and parts[0] in phase_order and parts[1] in pos_order:
            return (phase_order[parts[0]], pos_order[parts[1]], label)
        return (999, 999, label)

    return sorted(snaps.keys(), key=sort_key)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze full-slab vs interfacial-shell charge from stored snapshots."
    )
    parser.add_argument("--snapshots", required=True, help="Path to *_snapshots.npz")
    parser.add_argument("--meta", default=None, help="Optional path to *_meta.json")
    parser.add_argument("--shell-nm", type=float, default=10.0, help="Shell thickness in nm")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional subset of snapshot labels")
    parser.add_argument("--out", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    snap_path = Path(args.snapshots)
    snaps = load_snapshots(snap_path)
    meta = load_metadata(snap_path, args.meta)

    labels = args.labels if args.labels is not None else ordered_labels(snaps)

    rows = []
    for label in labels:
        if label not in snaps:
            raise KeyError(f"Snapshot label '{label}' not found. Available: {sorted(snaps.keys())}")
        rows.append(
            analyze_snapshot(
                label,
                snaps[label],
                tA_m=float(meta["tA_m"]),
                tB_m=float(meta["tB_m"]),
                L_m=float(meta["L_m"]),
                motion_mode=str(meta.get("motion_mode", "bilateral")),
                shell_nm=float(args.shell_nm),
            )
        )

    df = pd.DataFrame(rows)

    if args.out is None:
        out_path = snap_path.with_name(snap_path.stem + f"_shells_{int(args.shell_nm)}nm.csv")
    else:
        out_path = Path(args.out)

    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\nWrote: {out_path}\n")
    print(df[[
        "label",
        "t_us",
        "h_nm",
        "QA_full_C_per_m2",
        "QB_full_C_per_m2",
        "QA_contact_shell_C_per_m2",
        "QB_contact_shell_C_per_m2",
        "QA_outer_shell_C_per_m2",
        "QB_outer_shell_C_per_m2",
        "dphi_A_contact_mV",
        "dphi_B_contact_mV",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
