from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Styling helpers
# ---------------------------
def rgb(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


# Charge-based colors
COLOR_POS = rgb(160, 30, 30)      # deep red
COLOR_NEG = rgb(30, 60, 160)      # deep blue
COLOR_PHI = rgb(180, 90, 90)      # warm muted red for potential

ALPHA_MAP = {
    "pre_end": 0.35,
    "hold_mid": 0.70,
    "relax_end": 1.00,
}

LINEWIDTH = 1.0
INTERFACE_COLOR = "0.55"
INTERFACE_LW = 0.8

# Publication-oriented defaults
plt.rcParams.update({
    "text.usetex": False,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "lines.linewidth": LINEWIDTH,
    "savefig.dpi": 600,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


def panel_size(wide: bool = False):
    return (4.5, 1.8) if wide else (2.25, 1.8)


def load_snapshots(path: str | Path):
    data = np.load(path, allow_pickle=True)
    labels = ["pre_end", "hold_mid", "relax_end"]
    snaps = {}
    for label in labels:
        snaps[label] = {
            "t_s": float(data[f"{label}__t_s"]),
            "z_m": data[f"{label}__z_m"],
            "phi_V": data[f"{label}__phi_V"],
            "rho_C_per_m3": data[f"{label}__rho_C_per_m3"],
            "c_plus_molm3": data[f"{label}__c_plus_molm3"],
            "c_minus_molm3": data[f"{label}__c_minus_molm3"],
            "h_gap_m": float(data[f"{label}__h_gap_m"]),
        }
    return snaps


def add_interfaces(ax, h_gap_m: float, tA_m: float, tB_m: float):
    x1 = 1e9 * tA_m
    x2 = 1e9 * (tA_m + h_gap_m)
    x3 = 1e9 * (tA_m + h_gap_m + tB_m)
    for x in (x1, x2, x3):
        ax.axvline(
            x,
            linestyle=":",
            color=INTERFACE_COLOR,
            linewidth=INTERFACE_LW,
            alpha=0.9,
            zorder=0,
        )


def style_axes(ax):
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_close(fig, path: Path):
    fig.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot snapshot profiles from PNP run.")
    parser.add_argument("--input", required=True, help="NPZ snapshot file")
    parser.add_argument("--outdir", default="data/figures", help="Output directory")
    parser.add_argument("--tA-nm", type=float, default=150.0, help="Slab A thickness in nm")
    parser.add_argument("--tB-nm", type=float, default=150.0, help="Slab B thickness in nm")
    parser.add_argument(
        "--window-nm",
        type=float,
        default=40.0,
        help="Half-width of local moving-frame window around slab-B interface",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps = load_snapshots(args.input)
    stem = Path(args.input).stem

    tA_m = args.tA_nm * 1e-9
    tB_m = args.tB_nm * 1e-9
    window_nm = args.window_nm

    # ---------------------------
    # 1) Global potential profiles
    # ---------------------------
    fig, ax = plt.subplots(figsize=panel_size(wide=True))
    for label, s in snaps.items():
        z_nm = 1e9 * s["z_m"]
        phi_mV = 1e3 * s["phi_V"]
        alpha = ALPHA_MAP[label]
        ax.plot(
            z_nm,
            phi_mV,
            color=COLOR_PHI,
            alpha=alpha,
            linewidth=LINEWIDTH,
            label=f"{label}, t={s['t_s']:.2e} s",
        )
        add_interfaces(ax, s["h_gap_m"], tA_m, tB_m)
    ax.set_xlabel("z (nm)")
    ax.set_ylabel(r"Potential $\phi$ (mV)")
    ax.set_title("Global potential profiles")
    ax.legend(loc="best", frameon=False)
    style_axes(ax)
    save_close(fig, out_dir / f"{stem}_phi_profiles_global.png")

    # ---------------------------
    # 2) Global concentration profiles
    # ---------------------------
    fig, ax = plt.subplots(figsize=panel_size(wide=True))
    for label, s in snaps.items():
        z_nm = 1e9 * s["z_m"]
        alpha = ALPHA_MAP[label]

        ax.plot(
            z_nm,
            s["c_plus_molm3"],
            color=COLOR_POS,
            alpha=alpha,
            linewidth=LINEWIDTH,
            label=fr"$c_+$ {label}",
        )
        ax.plot(
            z_nm,
            s["c_minus_molm3"],
            color=COLOR_NEG,
            alpha=alpha,
            linewidth=LINEWIDTH,
            linestyle="--",
            label=fr"$c_-$ {label}",
        )
        add_interfaces(ax, s["h_gap_m"], tA_m, tB_m)

    ax.set_xlabel("z (nm)")
    ax.set_ylabel(r"Concentration (mol m$^{-3}$)")
    ax.set_title("Global concentration profiles")
    ax.legend(loc="best", ncol=2, frameon=False)
    style_axes(ax)
    save_close(fig, out_dir / f"{stem}_concentration_profiles_global.png")

    # ---------------------------
    # 3) Local slab-B-frame potential profiles
    # ---------------------------
    fig, ax = plt.subplots(figsize=panel_size(wide=False))
    for label, s in snaps.items():
        alpha = ALPHA_MAP[label]
        z_m = s["z_m"]
        xi_nm = 1e9 * (z_m - (tA_m + s["h_gap_m"]))
        mask = np.abs(xi_nm) <= window_nm
        phi_mV = 1e3 * s["phi_V"][mask]

        ax.plot(
            xi_nm[mask],
            phi_mV,
            color=COLOR_PHI,
            alpha=alpha,
            linewidth=LINEWIDTH,
            label=f"{label}",
        )

    ax.axvline(0.0, linestyle=":", color=INTERFACE_COLOR, linewidth=INTERFACE_LW, alpha=0.9)
    ax.set_xlim(-window_nm, window_nm)
    ax.set_xlabel(r"$\xi_B = z - (t_A + h)$ (nm)")
    ax.set_ylabel(r"Potential $\phi$ (mV)")
    ax.set_title("Potential near moving slab-B interface")
    ax.legend(loc="best", frameon=False)
    style_axes(ax)
    save_close(fig, out_dir / f"{stem}_phi_profiles_slabB_frame.png")

    # ---------------------------
    # 4) Local slab-B-frame concentration profiles
    # ---------------------------
    fig, ax = plt.subplots(figsize=panel_size(wide=False))
    for label, s in snaps.items():
        alpha = ALPHA_MAP[label]
        z_m = s["z_m"]
        xi_nm = 1e9 * (z_m - (tA_m + s["h_gap_m"]))
        mask = np.abs(xi_nm) <= window_nm

        ax.plot(
            xi_nm[mask],
            s["c_plus_molm3"][mask],
            color=COLOR_POS,
            alpha=alpha,
            linewidth=LINEWIDTH,
            label=fr"$c_+$ {label}",
        )
        ax.plot(
            xi_nm[mask],
            s["c_minus_molm3"][mask],
            color=COLOR_NEG,
            alpha=alpha,
            linewidth=LINEWIDTH,
            linestyle="--",
            label=fr"$c_-$ {label}",
        )

    ax.axvline(0.0, linestyle=":", color=INTERFACE_COLOR, linewidth=INTERFACE_LW, alpha=0.9)
    ax.set_xlim(-window_nm, window_nm)
    ax.set_xlabel(r"$\xi_B = z - (t_A + h)$ (nm)")
    ax.set_ylabel(r"Concentration (mol m$^{-3}$)")
    ax.set_title("Concentrations near moving slab-B interface")
    ax.legend(loc="best", ncol=1, frameon=False)
    style_axes(ax)
    save_close(fig, out_dir / f"{stem}_concentration_profiles_slabB_frame.png")

    # ---------------------------
    # 5) Difference profiles relative to pre_end in slab-B frame
    # ---------------------------
    pre = snaps["pre_end"]

    # Potential differences
    fig, ax = plt.subplots(figsize=panel_size(wide=False))
    for label in ["hold_mid", "relax_end"]:
        s = snaps[label]
        alpha = ALPHA_MAP[label]
        ls = "-" if label == "hold_mid" else ":"

        z_m = s["z_m"]
        xi_nm = 1e9 * (z_m - (tA_m + s["h_gap_m"]))
        mask = np.abs(xi_nm) <= window_nm

        dphi_mV = 1e3 * (s["phi_V"] - pre["phi_V"])
        ax.plot(
            xi_nm[mask],
            dphi_mV[mask],
            color=COLOR_PHI,
            alpha=alpha,
            linewidth=LINEWIDTH,
            linestyle=ls,
            label=f"{label} - pre_end",
        )

    ax.axvline(0.0, linestyle=":", color=INTERFACE_COLOR, linewidth=INTERFACE_LW, alpha=0.9)
    ax.set_xlim(-window_nm, window_nm)
    ax.set_xlabel(r"$\xi_B = z - (t_A + h)$ (nm)")
    ax.set_ylabel(r"$\Delta \phi$ (mV)")
    ax.set_title("Potential change near moving slab-B interface")
    ax.legend(loc="best", frameon=False)
    style_axes(ax)
    save_close(fig, out_dir / f"{stem}_dphi_profiles_slabB_frame.png")

    # Concentration differences
    fig, ax = plt.subplots(figsize=panel_size(wide=False))
    for label in ["hold_mid", "relax_end"]:
        s = snaps[label]
        alpha = ALPHA_MAP[label]
        ls = "-" if label == "hold_mid" else ":"

        z_m = s["z_m"]
        xi_nm = 1e9 * (z_m - (tA_m + s["h_gap_m"]))
        mask = np.abs(xi_nm) <= window_nm

        dcp = s["c_plus_molm3"] - pre["c_plus_molm3"]
        dcm = s["c_minus_molm3"] - pre["c_minus_molm3"]

        ax.plot(
            xi_nm[mask],
            dcp[mask],
            color=COLOR_POS,
            alpha=alpha,
            linewidth=LINEWIDTH,
            linestyle=ls,
            label=fr"$\Delta c_+$ ({label})",
        )
        ax.plot(
            xi_nm[mask],
            dcm[mask],
            color=COLOR_NEG,
            alpha=alpha,
            linewidth=LINEWIDTH,
            linestyle=ls,
            label=fr"$\Delta c_-$ ({label})",
        )

    ax.axvline(0.0, linestyle=":", color=INTERFACE_COLOR, linewidth=INTERFACE_LW, alpha=0.9)
    ax.set_xlim(-window_nm, window_nm)
    ax.set_xlabel(r"$\xi_B = z - (t_A + h)$ (nm)")
    ax.set_ylabel(r"Concentration change (mol m$^{-3}$)")
    ax.set_title("Concentration change near moving slab-B interface")
    ax.legend(loc="best", ncol=1, frameon=False)
    style_axes(ax)
    save_close(fig, out_dir / f"{stem}_dc_profiles_slabB_frame.png")

    print(f"Wrote profile figures to: {out_dir}")


if __name__ == "__main__":
    main()
