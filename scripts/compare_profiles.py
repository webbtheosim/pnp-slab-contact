from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


SNAPSHOTS = ["pre_end", "hold_mid", "final"]


def load_snapshots(path: str | Path):
    data = np.load(path, allow_pickle=True)

    snaps = {}
    for snap in SNAPSHOTS:
        snaps[snap] = {
            "t_s": float(data[f"{snap}__t_s"]),
            "z_m": data[f"{snap}__z_m"],
            "phi_V": data[f"{snap}__phi_V"],
            "rho_C_per_m3": data[f"{snap}__rho_C_per_m3"],
            "c_plus_molm3": data[f"{snap}__c_plus_molm3"],
            "c_minus_molm3": data[f"{snap}__c_minus_molm3"],
            "h_gap_m": float(data[f"{snap}__h_gap_m"]),
        }
    return snaps


def add_interface_lines(ax, tA_nm: float = 150.0, tB_start_nm: float = 350.0):
    ax.axvline(tA_nm, linestyle=":", alpha=0.5)
    ax.axvline(tB_start_nm, linestyle=":", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="Compare snapshot profiles across runs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Snapshot NPZ files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels corresponding to the input files",
    )
    parser.add_argument(
        "--outdir",
        default="profile_comparison",
        help="Output directory",
    )
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_snaps = []
    for path in args.inputs:
        all_snaps.append(load_snapshots(path))

    # Potential profiles
    for snap in SNAPSHOTS:
        fig, ax = plt.subplots()
        for label, snaps in zip(args.labels, all_snaps):
            z_nm = 1e9 * snaps[snap]["z_m"]
            phi_mV = 1e3 * snaps[snap]["phi_V"]
            t_s = snaps[snap]["t_s"]
            ax.plot(z_nm, phi_mV, label=f"{label} (t={t_s:.2e} s)")
        add_interface_lines(ax)
        ax.set_xlabel("z (nm)")
        ax.set_ylabel("Potential φ (mV)")
        ax.set_title(f"{snap} potential profiles")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"{snap}_phi_compare.png", dpi=200)
        plt.close(fig)

    # Charge-density profiles
    for snap in SNAPSHOTS:
        fig, ax = plt.subplots()
        for label, snaps in zip(args.labels, all_snaps):
            z_nm = 1e9 * snaps[snap]["z_m"]
            rho = snaps[snap]["rho_C_per_m3"]
            ax.plot(z_nm, rho, label=label)
        add_interface_lines(ax)
        ax.set_xlabel("z (nm)")
        ax.set_ylabel("Charge density ρ (C/m³)")
        ax.set_title(f"{snap} charge-density profiles")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"{snap}_rho_compare.png", dpi=200)
        plt.close(fig)

    # Concentration profiles
    for snap in SNAPSHOTS:
        fig, ax = plt.subplots()
        for label, snaps in zip(args.labels, all_snaps):
            z_nm = 1e9 * snaps[snap]["z_m"]
            cp = snaps[snap]["c_plus_molm3"]
            cm = snaps[snap]["c_minus_molm3"]
            ax.plot(z_nm, cp, label=f"{label} c+")
            ax.plot(z_nm, cm, linestyle="--", label=f"{label} c-")
        add_interface_lines(ax)
        ax.set_xlabel("z (nm)")
        ax.set_ylabel("Concentration (mol/m³)")
        ax.set_title(f"{snap} concentration profiles")
        ax.legend(loc="best", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"{snap}_concentration_compare.png", dpi=200)
        plt.close(fig)

    print(f"Wrote profile comparison plots to: {outdir}")


if __name__ == "__main__":
    main()
