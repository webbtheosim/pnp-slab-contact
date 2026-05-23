from __future__ import annotations

import argparse
import csv, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False


def read_csv(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Did you run run_single.py?")

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        cols: dict[str, list[float]] = {k: [] for k in (r.fieldnames or [])}
        for row in r:
            for k, v in row.items():
                cols[k].append(float(v))

    return {k: np.asarray(v) for k, v in cols.items()}

def read_metadata_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {path}")
    with path.open("r") as f:
        return json.load(f)

def protocol_times_from_counts(
    dt: float,
    n_pre: int,
    n_approach: int,
    n_hold: int,
    n_separate: int,
    n_relax: int,
):
    t_pre_end = n_pre * dt
    t_approach_end = (n_pre + n_approach) * dt
    t_hold_end = (n_pre + n_approach + n_hold) * dt
    t_sep_end = (n_pre + n_approach + n_hold + n_separate) * dt
    t_relax_end = (n_pre + n_approach + n_hold + n_separate + n_relax) * dt
    return t_pre_end, t_approach_end, t_hold_end, t_sep_end, t_relax_end


def shade_protocol(
    ax,
    dt: float,
    n_pre: int,
    n_approach: int,
    n_hold: int,
    n_separate: int,
    n_relax: int,
    time_scale: float = 1.0,
):
    """
    Shade protocol regions with solid light-gray fills.
    """
    t_pre_end, t_approach_end, t_hold_end, t_sep_end, t_relax_end = protocol_times_from_counts(
        dt, n_pre, n_approach, n_hold, n_separate, n_relax
    )

    spans = [
        (0.0, t_pre_end, "Pre-equil.", 0.94),
        (t_pre_end, t_approach_end, "Approach", 0.88),
        (t_approach_end, t_hold_end, "Contact/Hold", 0.82),
        (t_hold_end, t_sep_end, "Separate", 0.88),
        (t_sep_end, t_relax_end, "Relax", 0.94),
    ]

    for a, b, label, gray in spans:
        if b <= a:
            continue
        ax.axvspan(
            a * time_scale,
            b * time_scale,
            facecolor=str(gray),
            edgecolor="none",
            alpha=1.0,
            label=label,
            zorder=0,
        )


def deduplicate_legend(ax1, ax2=None, loc="best"):
    handles, labels = ax1.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)

    if uniq_labels:
        ax1.legend(uniq_handles, uniq_labels, loc=loc)


def phase_masks(
    t: np.ndarray,
    dt: float,
    n_pre: int,
    n_approach: int,
    n_hold: int,
    n_separate: int,
    n_relax: int,
):
    """
    Return boolean masks for each protocol phase.
    """
    t_pre_end, t_approach_end, t_hold_end, t_sep_end, t_relax_end = protocol_times_from_counts(
        dt, n_pre, n_approach, n_hold, n_separate, n_relax
    )

    masks = {
        "pre": (t >= 0.0) & (t < t_pre_end),
        "approach": (t >= t_pre_end) & (t < t_approach_end),
        "hold": (t >= t_approach_end) & (t < t_hold_end),
        "separate": (t >= t_hold_end) & (t < t_sep_end),
        "relax": (t >= t_sep_end) & (t <= t_relax_end + 0.5 * dt),
        "post_pre": (t >= t_pre_end - 0.5 * dt),
    }
    return masks


def print_phase_diagnostics(
    name: str,
    t: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    dphi_A_mV: np.ndarray,
    dphi_B_mV: np.ndarray,
):
    if t.size == 0:
        print(f"{name}: no data")
        return

    Qtrans = float(np.trapezoid(J, t))
    J_abs = np.abs(J)
    imax = int(np.argmax(J_abs))

    print(f"\n{name}")
    print("  Integrated charge transfer (C/m^2):", Qtrans)
    print("  J 95th percentile (A/m^2):", float(np.percentile(J_abs, 95)))
    print("  J RMS (A/m^2):", float(np.sqrt(np.mean(J**2))))
    print("  Time of max |J| (s):", float(t[imax]))
    print("  h at max |J| (m):", float(h[imax]))
    print("  Max |Δφ_A| (mV):", float(np.max(np.abs(dphi_A_mV))))
    print("  Max |Δφ_B| (mV):", float(np.max(np.abs(dphi_B_mV))))
    print("  Max |J| (A/m^2):", float(np.max(J_abs)))


def main():
    parser = argparse.ArgumentParser(description="Generate figures from PNP CSV output.")
    parser.add_argument("--input", type=str, required=True, help="CSV path from run_single.py")
    parser.add_argument("--outdir", type=str, default="data/figures", help="Output directory")
    parser.add_argument("--dt", type=float, default=None, help="Time step used in the simulation")
    parser.add_argument("--n-pre", type=int, default=None, help="Number of pre-equilibration steps")
    parser.add_argument("--n-approach", type=int, default=None, help="Number of approach steps")
    parser.add_argument("--n-hold", type=int, default=None, help="Number of hold steps")
    parser.add_argument("--n-separate", type=int, default=None, help="Number of separation steps")
    parser.add_argument("--n-relax", type=int, default=None, help="Number of relaxation steps")
    args = parser.parse_args()

    in_path = Path(args.input)
    meta_path = in_path.with_name(in_path.stem + "_meta.json")
    meta = {}

    if meta_path.exists():
        meta = read_metadata_json(meta_path)
        print(f"Loaded metadata from: {meta_path}")

    dt = args.dt if args.dt is not None else meta.get("dt")
    n_pre = args.n_pre if args.n_pre is not None else meta.get("n_pre")
    n_approach = args.n_approach if args.n_approach is not None else meta.get("n_approach")
    n_hold = args.n_hold if args.n_hold is not None else meta.get("n_hold")
    n_separate = args.n_separate if args.n_separate is not None else meta.get("n_separate")
    n_relax = args.n_relax if args.n_relax is not None else meta.get("n_relax")

    if None in (dt, n_pre, n_approach, n_hold, n_separate, n_relax):
        raise ValueError(
            "Could not determine full protocol metadata. "
            "Either provide CLI arguments or ensure *_meta.json exists."
        )

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_name = in_path.stem
    print(f"\nReading: {in_path}")

    d = read_csv(in_path)

    # Data
    t = d["t_s"]
    t_us = 1e6 * t
    h = d["h_gap_m"]
    dphi_A = d["dphi_A_V"]
    dphi_B = d["dphi_B_V"]
    J = d["Jcharge_mid_A_per_m2"]
    QA = d["QA_C_per_m2"]
    QB = d["QB_C_per_m2"]

    # Derived
    dphi_A_mV = 1e3 * dphi_A
    dphi_B_mV = 1e3 * dphi_B

    masks = phase_masks(
        t,
        dt,
        n_pre,
        n_approach,
        n_hold,
        n_separate,
        n_relax,
    )

    # Diagnostics by phase
    print_phase_diagnostics(
        "Post-pre (entire driven protocol + relaxation)",
        t[masks["post_pre"]],
        h[masks["post_pre"]],
        J[masks["post_pre"]],
        dphi_A_mV[masks["post_pre"]],
        dphi_B_mV[masks["post_pre"]],
    )

    print_phase_diagnostics(
        "Approach",
        t[masks["approach"]],
        h[masks["approach"]],
        J[masks["approach"]],
        dphi_A_mV[masks["approach"]],
        dphi_B_mV[masks["approach"]],
    )

    print_phase_diagnostics(
        "Hold",
        t[masks["hold"]],
        h[masks["hold"]],
        J[masks["hold"]],
        dphi_A_mV[masks["hold"]],
        dphi_B_mV[masks["hold"]],
    )

    print_phase_diagnostics(
        "Separate",
        t[masks["separate"]],
        h[masks["separate"]],
        J[masks["separate"]],
        dphi_A_mV[masks["separate"]],
        dphi_B_mV[masks["separate"]],
    )

    print_phase_diagnostics(
        "Relax",
        t[masks["relax"]],
        h[masks["relax"]],
        J[masks["relax"]],
        dphi_A_mV[masks["relax"]],
        dphi_B_mV[masks["relax"]],
    )

    # Plot window: show end of pre-equilibration onward
    t_pre_end = n_pre * dt
    t_plot_start = 0.90 * t_pre_end
    x_max = t_us[-1]

    # ---- Figure 0: Combined Δφ + J ----
    fig, ax1 = plt.subplots()
    shade_protocol(
        ax1,
        dt,
        n_pre,
        n_approach,
        n_hold,
        n_separate,
        n_relax,
        time_scale=1e6,
    )

    ax1.plot(t_us, dphi_A_mV, label=r"$\Delta \phi_A$ (mV)")
    ax1.plot(t_us, dphi_B_mV, label=r"$\Delta \phi_B$ (mV)")
    ax1.set_xlabel(r"Time ($\mu$s)")
    ax1.set_ylabel(r"Interfacial potential jump $\Delta \phi$ (mV)")
    ax1.set_title(case_name)
    ax1.set_xlim(t_plot_start * 1e6, x_max)

    ax2 = ax1.twinx()
    ax2.plot(t_us, J, linestyle="--", label=r"$J$ (A/m$^2$)")
    ax2.set_ylabel(r"Charge current density $J$ (A/m$^2$)")

    deduplicate_legend(ax1, ax2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_name}_combined_dphiJ.png", dpi=200)
    plt.close(fig)

    # ---- Figure 1: Gap vs time ----
    fig, ax = plt.subplots()
    shade_protocol(
        ax,
        dt,
        n_pre,
        n_approach,
        n_hold,
        n_separate,
        n_relax,
        time_scale=1e6,
    )
    ax.plot(t_us, h / 1e-9, color="black")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"Gap thickness $h$ (nm)")
    ax.set_title(case_name)
    ax.set_xlim(t_plot_start * 1e6, x_max)
    deduplicate_legend(ax, None, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_name}_gap_vs_time.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: Δφ only ----
    fig, ax = plt.subplots()
    shade_protocol(
        ax,
        dt,
        n_pre,
        n_approach,
        n_hold,
        n_separate,
        n_relax,
        time_scale=1e6,
    )
    ax.plot(t_us, dphi_A_mV, label=r"$\Delta \phi_A$ (mV)")
    ax.plot(t_us, dphi_B_mV, label=r"$\Delta \phi_B$ (mV)")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"Interfacial potential jump $\Delta \phi$ (mV)")
    ax.set_title(case_name)
    ax.set_xlim(t_plot_start * 1e6, x_max)
    deduplicate_legend(ax, None, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_name}_dphi_vs_time.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: Current only ----
    fig, ax = plt.subplots()
    shade_protocol(
        ax,
        dt,
        n_pre,
        n_approach,
        n_hold,
        n_separate,
        n_relax,
        time_scale=1e6,
    )
    ax.plot(t_us, J, color="black")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"Charge current density $J$ (A/m$^2$)")
    ax.set_title(case_name)
    ax.set_xlim(t_plot_start * 1e6, x_max)
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_name}_current_vs_time.png", dpi=200)
    plt.close(fig)

    # ---- Figure 4: Region charge ----
    fig, ax = plt.subplots()
    shade_protocol(
        ax,
        dt,
        n_pre,
        n_approach,
        n_hold,
        n_separate,
        n_relax,
        time_scale=1e6,
    )
    ax.plot(t_us, QA, label=r"$Q_A$ (C/m$^2$)")
    ax.plot(t_us, QB, label=r"$Q_B$ (C/m$^2$)")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"Region-integrated charge (C/m$^2$)")
    ax.set_title(case_name)
    ax.set_xlim(t_plot_start * 1e6, x_max)
    deduplicate_legend(ax, None, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_name}_region_charge_vs_time.png", dpi=200)
    plt.close(fig)

    print(f"\nWrote figures to: {out_dir}\n")


if __name__ == "__main__":
    main()
