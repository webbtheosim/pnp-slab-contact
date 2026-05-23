from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Styling helpers
# ---------------------------
def rgb(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


COLOR_POS = rgb(160, 30, 30)      # deep red
COLOR_NEG = rgb(30, 60, 160)      # deep blue
COLOR_PHI = rgb(180, 75, 180)     # purple
COLOR_QA = rgb(160, 30, 30)
COLOR_QB = rgb(30, 60, 160)
COLOR_WIN_A = rgb(180, 50, 50)
COLOR_WIN_B = rgb(50, 50, 180)
COLOR_PROTOCOL = "0.20"

INTERFACE_COLOR = "0.55"
INTERFACE_LW = 0.8
GRID_ALPHA = 0.22
LINEWIDTH = 1.2
VLINE_ALPHA = 0.75

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
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})

DEFAULT_PROFILE_LABELS = ["pre_end", "hold_mid", "relax_end"]


def panel_size(kind: str = "standard"):
    if kind == "profile":
        return (3.5, 2.2)
    if kind == "timeseries":
        return (3.5, 2.2)
    if kind == "protocol":
        return (3.5, 2.0)
    raise ValueError(f"Unknown panel kind: {kind}")


def style_axes(ax):
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_close(fig, path: Path):
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def write_plot_data_csv(outpath: Path, suffix: str, data: dict):
    """
    Write the raw arrays used for a plot to a CSV file.

    Parameters
    ----------
    outpath : Path
        Path of the figure being written.
    suffix : str
        Descriptive suffix for the CSV file.
    data : dict
        Dictionary of column_name: array-like.
    """
    outpath = Path(outpath)
    csv_path = outpath.with_name(f"{outpath.stem}_{suffix}.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Wrote raw plot data: {csv_path}")

# ---------------------------
# Metadata
# ---------------------------
def load_metadata(csv_path: str | Path) -> dict:
    csv_path = Path(csv_path)
    meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
    if not meta_path.exists():
        return {}
    with meta_path.open("r") as f:
        return json.load(f)


def pick_value(cli_value, meta: dict, meta_key: str, default=None):
    if cli_value is not None:
        return cli_value
    if meta_key in meta:
        return meta[meta_key]
    return default


# ---------------------------
# Snapshot loading
# ---------------------------
def load_snapshots(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Supports:
    1) flattened keys like pre_end__z_m
    2) object entries keyed by label, each containing a dict
    3) keys like label/subkey
    """
    data = np.load(path, allow_pickle=True)

    # Case 1: flattened keys with double underscore
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
                    "z_m": np.asarray(data[f"{label}__z_m"]),
                    "phi_V": np.asarray(data[f"{label}__phi_V"]),
                    "rho_C_per_m3": np.asarray(data[f"{label}__rho_C_per_m3"]),
                    "c_plus_molm3": np.asarray(data[f"{label}__c_plus_molm3"]),
                    "c_minus_molm3": np.asarray(data[f"{label}__c_minus_molm3"]),
                    "h_gap_m": float(np.asarray(data[f"{label}__h_gap_m"]).squeeze()),
                }
        if snaps:
            return snaps

    # Case 2: dict-like object payloads
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

    # Case 3: keys like label/subkey
    snaps = {}
    for key in data.files:
        if "/" in key:
            label, subkey = key.split("/", 1)
            snaps.setdefault(label, {})[subkey] = data[key]
    if snaps:
        return snaps

    raise ValueError(
        f"Could not interpret snapshot structure in {path}. "
        f"Found keys: {list(data.files)}"
    )


# ---------------------------
# Snapshot ordering / naming
# ---------------------------
_PHASE_ORDER = {
    "pre": 0,
    "approach": 1,
    "hold": 2,
    "separate": 3,
    "relax": 4,
}

_POS_ORDER = {
    "start": 0,
    "mid": 1,
    "end": 2,
}


def snapshot_sort_key(label: str) -> Tuple[int, int, str]:
    m = re.match(r"^(pre|approach|hold|separate|relax)_(start|mid|end)$", label)
    if m:
        phase, pos = m.groups()
        return (_PHASE_ORDER[phase], _POS_ORDER[pos], label)
    # fallback for legacy or unexpected labels
    return (999, 999, label)


def ordered_profile_labels(snaps: Dict[str, Dict[str, Any]], requested: List[str] | None = None) -> List[str]:
    if requested:
        missing = [lab for lab in requested if lab not in snaps]
        if missing:
            raise KeyError(f"Requested snapshot labels not found: {missing}. Available: {sorted(snaps.keys())}")
        return sorted(requested, key=snapshot_sort_key)

    # Use all labels, chronologically when possible
    return sorted(snaps.keys(), key=snapshot_sort_key)


def pretty_profile_name(label: str) -> str:
    m = re.match(r"^(pre|approach|hold|separate|relax)_(start|mid|end)$", label)
    if m:
        phase, pos = m.groups()
        phase_name = {
            "pre": "Pre-equilibration",
            "approach": "Approach",
            "hold": "Contact/Hold",
            "separate": "Separate",
            "relax": "Relax",
        }[phase]
        pos_name = {
            "start": "start",
            "mid": "midpoint",
            "end": "end",
        }[pos]
        return f"{phase_name} {pos_name}"
    return label.replace("_", " ")


def filename_panel_index(i: int) -> str:
    return f"{i+1:02d}"


# ---------------------------
# Nice limits / ticks
# ---------------------------
def _nice_step(raw_step: float) -> float:
    if raw_step <= 0 or not np.isfinite(raw_step):
        return 1.0
    exp = math.floor(math.log10(raw_step))
    frac = raw_step / (10 ** exp)
    if frac <= 1.0:
        nice = 1.0
    elif frac <= 2.0:
        nice = 2.0
    elif frac <= 2.5:
        nice = 2.5
    elif frac <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10 ** exp)


def nice_limits(xmin: float, xmax: float, nticks: int = 5) -> Tuple[float, float, float]:
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return 0.0, 1.0, 0.2
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if np.isclose(xmin, xmax):
        if np.isclose(xmin, 0.0):
            xmin, xmax = -1.0, 1.0
        else:
            pad = 0.1 * abs(xmin)
            xmin -= pad
            xmax += pad
    raw_step = (xmax - xmin) / max(nticks - 1, 1)
    step = _nice_step(raw_step)
    lo = math.floor(xmin / step) * step
    hi = math.ceil(xmax / step) * step
    return lo, hi, step


def symmetric_nice_limits(values, nticks: int = 5, pad_frac: float = 0.03) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0, 0.5
    vmax = np.max(np.abs(arr))
    vmax *= (1.0 + pad_frac)
    lo, hi, step = nice_limits(-vmax, vmax, nticks=nticks)
    m = max(abs(lo), abs(hi))
    return -m, m, step


# ---------------------------
# Geometry helpers
# ---------------------------
def interface_positions_nm(
    h_gap_m: float,
    tA_m: float,
    tB_m: float,
    motion_mode: str,
    L_m: float,
) -> Tuple[float, float, float, float]:
    """
    Returns x positions in nm:
    (zA_start, zA_end, zB_start, zB_end)
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

    return 1e9 * zA_start, 1e9 * zA_end, 1e9 * zB_start, 1e9 * zB_end


def add_region_shading(
    ax,
    snapshot: Dict[str, Any],
    tA_m: float,
    tB_m: float,
    motion_mode: str,
    L_m: float,
):
    zA_start_nm, zA_end_nm, zB_start_nm, zB_end_nm = interface_positions_nm(
        float(snapshot["h_gap_m"]), tA_m, tB_m, motion_mode, L_m
    )

    # shaded slab regions
    ax.axvspan(zA_start_nm, zA_end_nm, color="0.94", alpha=1.0, zorder=0)
    ax.axvspan(zB_start_nm, zB_end_nm, color="0.94", alpha=1.0, zorder=0)

    # outer slab edges
    for x in (zA_start_nm, zB_end_nm):
        ax.axvline(
            x,
            linestyle="-",
            color="0.72",
            linewidth=0.7,
            alpha=0.8,
            zorder=1,
        )

    # interfaces defining the gap
    for x in (zA_end_nm, zB_start_nm):
        ax.axvline(
            x,
            linestyle="-",
            color=INTERFACE_COLOR,
            linewidth=INTERFACE_LW,
            alpha=0.95,
            zorder=1,
        )


def pretty_time_us(t_s: float) -> str:
    return f"{t_s * 1e6:.3f} µs"


# ---------------------------
# Limits
# ---------------------------
def compute_global_limits(
    df: pd.DataFrame,
    snaps: Dict[str, Dict[str, Any]],
    profile_labels: List[str],
) -> Dict[str, Tuple[float, float, float]]:

    profile_labels_subset = [label for label in profile_labels if label[:3] != "pre"]
    z_all_nm = np.concatenate([1e9 * np.asarray(snaps[k]["z_m"]) for k in profile_labels])
    xprof = nice_limits(np.min(z_all_nm), np.max(z_all_nm), nticks=6)

    c_all = np.concatenate(
        [np.asarray(snaps[k]["c_plus_molm3"]) for k in profile_labels_subset]
        + [np.asarray(snaps[k]["c_minus_molm3"]) for k in profile_labels_subset]
    )

    yconc = nice_limits(float(np.min(c_all)), float(np.max(c_all)), nticks=5)

    #yconc = symmetric_nice_limits(c_all, nticks=5,pad_frac=0.0)
    phi_all_mV = np.concatenate([1e3 * np.asarray(snaps[k]["phi_V"]) for k in profile_labels_subset])
    yphi = symmetric_nice_limits(phi_all_mV, nticks=5, pad_frac=0.04)
    #yphi = nice_limits(float(np.min(phi_all_mV)),float(np.max(phi_all_mV)), nticks=5)

    t_us = 1e6 * df["t_s"].to_numpy()
    xtime = nice_limits(float(np.min(t_us)), float(np.max(t_us)), nticks=6)

    q_arrays = [df["QA_C_per_m2"].to_numpy(), df["QB_C_per_m2"].to_numpy()]
    if "Qwin_Agap_C_per_m2" in df.columns:
        q_arrays.append(df["Qwin_Agap_C_per_m2"].to_numpy())
    if "Qwin_Bgap_C_per_m2" in df.columns:
        q_arrays.append(df["Qwin_Bgap_C_per_m2"].to_numpy())
    q_all = np.concatenate(q_arrays)
    yq = symmetric_nice_limits(q_all, nticks=5, pad_frac=0.04)
    #yq = nice_limits(float(np.min(q_all)),float(np.max(q_all)), nticks=5)

    h_nm = 1e9 * df["h_gap_m"].to_numpy()
    yh = nice_limits(float(np.min(h_nm)), float(np.max(h_nm)), nticks=5)

    return {
        "xprof": xprof,
        "yconc": yconc,
        "yphi": yphi,
        "xtime": xtime,
        "yq": yq,
        "yh": yh,
    }


# ---------------------------
# Panel plotting
# ---------------------------
def plot_profile_panel(
    snapshot: Dict[str, Any],
    outpath: Path,
    panel_label: str,
    panel_title: str,
    tA_m: float,
    tB_m: float,
    motion_mode: str,
    L_m: float,
    limits: Dict[str, Tuple[float, float, float]],
    verbose: bool = False,
):
    z_nm = 1e9 * np.asarray(snapshot["z_m"])
    c_plus = np.asarray(snapshot["c_plus_molm3"])
    c_minus = np.asarray(snapshot["c_minus_molm3"])
    c_bulk =  (c_plus[0] + c_minus[0] + c_plus[-1] + c_minus[-1])*0.25
    phi_mV = 1e3 * np.asarray(snapshot["phi_V"])
    h_gap_nm = 1e9 * float(snapshot["h_gap_m"])
    t_s = float(snapshot["t_s"])

    if verbose:
        print(panel_label)
        print("c_plus integral =", float(np.trapezoid(c_plus, z_nm)))
        print("c_minus integral =", float(np.trapezoid(c_minus, z_nm)))

    fig, ax = plt.subplots(figsize=panel_size("profile"))

    add_region_shading(ax, snapshot, tA_m, tB_m, motion_mode, L_m)

    ax.plot(z_nm, c_plus-c_bulk, color=COLOR_POS, linestyle="--", label=r"$c_+(z)$")
    ax.plot(z_nm, c_minus-c_bulk, color=COLOR_NEG, linestyle="--", label=r"$c_-(z)$")
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.spines["top"].set_visible(False)


    #ax.set_xlabel("z (nm)")
    #ax.set_ylabel(r"Concentration (mol m$^{-3}$)")
    #ax.grid(alpha=GRID_ALPHA)

    ax2 = ax.twinx()
    ax2.plot(z_nm, phi_mV, color=COLOR_PHI, linestyle="-", label=r"$\phi(z)$")
    ax2.tick_params(direction="out", length=3, width=0.8)
    ax2.spines["top"].set_visible(False)

    write_plot_data_csv(
    outpath=outpath,
    suffix="profile_data",
    data={
        "z_nm": z_nm,
        "c_plus_minus_bulk_molm3": c_plus - c_bulk,
        "c_minus_minus_bulk_molm3": c_minus - c_bulk,
        "phi_mV": phi_mV,
    },)
    #ax2.set_ylabel(r"Potential $\phi$ (mV)")

    #title = f"{panel_label}. {panel_title}\n$t$ = {pretty_time_us(t_s)},  h = {h_gap_nm:.1f} nm"
    #ax.set_title(title)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    #ax.legend(lines, labels, loc="best", frameon=False)

    #ax.set_xlim(limits["xprof"][0], limits["xprof"][1])
    #ax.set_xlim(550, 950)
    #print(limits["yconc"][0]-c_bulk, limits["yconc"][1]-c_bulk)
    ax.set_ylim(limits["yconc"][0]-c_bulk, limits["yconc"][1]-c_bulk)
    ax2.set_ylim(limits["yphi"][0], limits["yphi"][1])
    #ax2.set_ylim(-250, 250)

    style_axes(ax)
    style_axes(ax2)
    save_close(fig, outpath)


def get_snapshot_times_us(
    snaps: Dict[str, Dict[str, Any]],
    profile_labels: List[str],
    strict: bool = False,
) -> List[float]:
    times_us = []
    missing = []

    for lab in profile_labels:
        if lab not in snaps or "t_s" not in snaps[lab]:
            missing.append(lab)
            continue
        times_us.append(float(np.asarray(snaps[lab]["t_s"]).squeeze()) * 1e6)

    if strict and missing:
        raise KeyError(
            f"Missing snapshot labels or t_s entries for: {missing}. "
            f"Available labels: {list(snaps.keys())}"
        )

    return times_us


def plot_integrated_charge_panel(
    df: pd.DataFrame,
    snaps: Dict[str, Dict[str, Any]],
    profile_labels: List[str],
    outpath: Path,
    limits: Dict[str, Tuple[float, float, float]],
    stride: int = 1,
):
    t_us = df["t_s"].to_numpy() * 1e6
    #QA = df["QA_C_per_m2"].to_numpy()
    #QB = df["QB_C_per_m2"].to_numpy()
    QA = df["QA_C_per_m2"].to_numpy()
    QB = df["QB_C_per_m2"].to_numpy()

    fig, ax = plt.subplots(figsize=panel_size("timeseries"))
    ax.plot(t_us[::stride], QA[::stride], color=COLOR_QA, label=r"$Q_A$")
    ax.plot(t_us[::stride], QB[::stride], color=COLOR_QB, label=r"$Q_B$")

    for tmark in get_snapshot_times_us(snaps, profile_labels):
        ax.axvline(tmark, linestyle=":", color="0.65", alpha=VLINE_ALPHA, linewidth=0.9)

    #ax.set_title("D. Integrated charge in slab-associated regions")
    #ax.set_xlabel("Time (µs)")
    #ax.set_ylabel(r"Integrated charge (C m$^{-2}$)")
    #ax.grid(alpha=GRID_ALPHA)
    #ax.legend(frameon=False)

    ax.set_xlim(limits["xtime"][0], limits["xtime"][1])
    #ax.set_xlim(0.1, 0.135)
    ax.set_ylim(limits["yq"][0], limits["yq"][1])
    #ax.set_ylim(-0.01, 0.)

    style_axes(ax)
    ax.spines["right"].set_visible(False)
    save_close(fig, outpath)


def plot_integrated_charge_diff_panel(
    df: pd.DataFrame,
    snaps: Dict[str, Dict[str, Any]],
    profile_labels: List[str],
    outpath: Path,
    limits: Dict[str, Tuple[float, float, float]],
    stride: int = 1,
    baseline_label: str = "pre_end",
):
    t_us = df["t_s"].to_numpy() * 1e6
    QA = df["QA_C_per_m2"].to_numpy()
    QB = df["QB_C_per_m2"].to_numpy()

    if baseline_label not in snaps:
        raise ValueError(f"Baseline snapshot '{baseline_label}' not found in snapshots.")

    t0_us = float(np.asarray(snaps[baseline_label]["t_s"]).squeeze()) * 1e6
    idx0 = int(np.argmin(np.abs(t_us - t0_us)))

    QA0 = QA[idx0]
    QB0 = QB[idx0]

    fig, ax = plt.subplots(figsize=panel_size("timeseries"))
    ax.plot(t_us[::stride], QA[::stride] - QA0, color=COLOR_QA, label=r"$\Delta Q_A$")
    ax.plot(t_us[::stride], QB[::stride] - QB0, color=COLOR_QB, label=r"$\Delta Q_B$")

    marker_labels = sorted(set(profile_labels + [baseline_label]), key=snapshot_sort_key)
    for tmark in get_snapshot_times_us(snaps, marker_labels):
        ax.axvline(tmark, linestyle=":", color="0.65", alpha=VLINE_ALPHA, linewidth=0.9)

    write_plot_data_csv(
    outpath=outpath,
    suffix="integrated_charge_data",
    data={
        "time_us": t_us[::stride],
        "QA_C_per_m2": QA[::stride],
        "QB_C_per_m2": QB[::stride],
        "dQA_C_per_m2": QA[::stride]-QA0,
        "dQB_C_per_m2": QB[::stride]-QB0,
        "dQdiff_C_per_m2": QB[::stride]-QB0-(QA[::stride]-QA0),
    },
    )
    #ax.set_title("D-alt. Integrated charge difference in slab-associated regions")
    #ax.set_xlabel("Time (µs)")
    #ax.set_ylabel(r"Integrated change in charge (C m$^{-2}$)")
    #ax.grid(alpha=GRID_ALPHA)
    #ax.legend(frameon=False)

    ax.set_xlim(limits["xtime"][0], limits["xtime"][1])
    ax.set_ylim(limits["yq"][0], limits["yq"][1])

    style_axes(ax)
    ax.spines["right"].set_visible(False)
    save_close(fig, outpath)


def plot_window_charge_panel(
    df: pd.DataFrame,
    snaps: Dict[str, Dict[str, Any]],
    profile_labels: List[str],
    outpath: Path,
    limits: Dict[str, Tuple[float, float, float]],
    stride: int = 1,
):
    if "Qwin_Agap_C_per_m2" not in df.columns or "Qwin_Bgap_C_per_m2" not in df.columns:
        return

    t_us = df["t_s"].to_numpy() * 1e6
    QwinA = df["Qwin_Agap_C_per_m2"].to_numpy()
    QwinB = df["Qwin_Bgap_C_per_m2"].to_numpy()

    fig, ax = plt.subplots(figsize=panel_size("timeseries"))
    ax.plot(t_us[::stride], QwinA[::stride], color=COLOR_WIN_A, label=r"$Q_{\mathrm{win,A}}$")
    ax.plot(t_us[::stride], QwinB[::stride], color=COLOR_WIN_B, label=r"$Q_{\mathrm{win,B}}$")

    for tmark in get_snapshot_times_us(snaps, profile_labels):
        ax.axvline(tmark, linestyle=":", color="0.65", alpha=VLINE_ALPHA, linewidth=0.9)

    #ax.set_title("E. Interfacial gap-side charge in fixed-width windows")
    #ax.set_xlabel("Time (µs)")
    #ax.set_ylabel(r"Window-integrated charge (C m$^{-2}$)")
    #ax.grid(alpha=GRID_ALPHA)
    #ax.legend(frameon=False)

    ax.set_xlim(limits["xtime"][0], limits["xtime"][1])
    ax.set_ylim(limits["yq"][0], limits["yq"][1])

    style_axes(ax)
    ax.spines["right"].set_visible(False)
    save_close(fig, outpath)


def plot_protocol_panel(
    df: pd.DataFrame,
    snaps: Dict[str, Dict[str, Any]],
    profile_labels: List[str],
    outpath: Path,
    limits: Dict[str, Tuple[float, float, float]],
):
    t_us = df["t_s"].to_numpy() * 1e6
    h_nm = df["h_gap_m"].to_numpy() * 1e9

    fig, ax = plt.subplots(figsize=panel_size("protocol"))
    ax.plot(t_us, h_nm, color=COLOR_PROTOCOL, label=r"$h(t)$")

    for tmark in get_snapshot_times_us(snaps, profile_labels):
        ax.axvline(tmark, linestyle=":", color="0.65", alpha=VLINE_ALPHA, linewidth=0.9)

    #ax.set_title("Protocol context")
    #ax.set_xlabel("Time (µs)")
    #ax.set_ylabel("Gap thickness (nm)")
    #ax.grid(alpha=GRID_ALPHA)
    #ax.legend(frameon=False)

    ax.set_xlim(limits["xtime"][0], limits["xtime"][1])
    ax.set_ylim(limits["yh"][0], limits["yh"][1])

    style_axes(ax)
    ax.spines["right"].set_visible(False)
    save_close(fig, outpath)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Create standalone panels for contact-charging figure set.")
    parser.add_argument("--csv", required=True, help="History CSV")
    parser.add_argument("--snapshots", required=True, help="NPZ snapshot file")
    parser.add_argument("--outdir", required=True, help="Output directory for standalone panels")
    parser.add_argument("--prefix", default=None, help="Prefix for output filenames; defaults to CSV stem")
    parser.add_argument("--labels", nargs="+", default=None, help="Snapshot labels to use for profile panels")
    parser.add_argument("--verbose", action="store_true", help="Print profile integral diagnostics")

    # Optional overrides; otherwise load from metadata (which is already in SI units)
    parser.add_argument("--tA-nm", type=float, default=None, help="Slab A thickness in nm")
    parser.add_argument("--tB-nm", type=float, default=None, help="Slab B thickness in nm")
    parser.add_argument("--h-gap0-nm", type=float, default=None, help="Initial gap thickness in nm")
    parser.add_argument("--motion-mode", choices=["unilateral", "bilateral"], default=None, help="Motion convention")
    parser.add_argument("--L-nm", type=float, default=None, help="Simulation cell length in nm")
    parser.add_argument("--baseline-label", default="pre_end", help="Snapshot label to use as reference in ΔQ panel")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    snap_path = Path(args.snapshots)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(csv_path)

    # CLI is in nm; metadata sidecar is already in meters
    tA_m = pick_value(args.tA_nm * 1e-9 if args.tA_nm is not None else None, meta, "tA_m")
    tB_m = pick_value(args.tB_nm * 1e-9 if args.tB_nm is not None else None, meta, "tB_m")
    h_gap0_m = pick_value(args.h_gap0_nm * 1e-9 if args.h_gap0_nm is not None else None, meta, "h_gap0_m")
    L_m = pick_value(args.L_nm * 1e-9 if args.L_nm is not None else None, meta, "L_m")
    motion_mode = pick_value(args.motion_mode, meta, "motion_mode", default="bilateral")

    if None in (tA_m, tB_m, h_gap0_m, L_m):
        raise ValueError(
            "Could not determine geometry from metadata. "
            "Provide --tA-nm, --tB-nm, --h-gap0-nm, and --L-nm, "
            "or ensure the *_meta.json sidecar contains tA_m, tB_m, h_gap0_m, and L_m."
        )

    # calculate stride to avoid gridpoint crossing artifact
    L_approach = (meta['h_gap0_m']-meta['h_min_m'])*0.5 # distance traveled by one slab
    t_approach = meta['n_approach']*meta['dt']  # time elapsed during approach
    v_approach = L_approach / t_approach # velocity of slab
    if v_approach > 0:
        dz         = meta['L_m']/meta['N']   # grid point spacing
        dt_cross   = dz/v_approach           # time to cross a grid point
        stride     = int(np.ceil(dt_cross/meta['dt']))     # time steps associated with crossing
    else:
        stride     = 100

    prefix = args.prefix or csv_path.stem
    df = pd.read_csv(csv_path)
    snaps = load_snapshots(snap_path)

    profile_labels = ordered_profile_labels(snaps, args.labels)

    limits = compute_global_limits(df, snaps, profile_labels)

    # Plot all selected profiles in progression order
    for i, label in enumerate(profile_labels):
        plot_profile_panel(
            snaps[label],
            outdir / f"{filename_panel_index(i)}_{prefix}_panel_{label}.png",
            filename_panel_index(i),
            pretty_profile_name(label),
            tA_m,
            tB_m,
            motion_mode,
            L_m,
            limits,
            verbose=args.verbose,
        )

    # Time-trace panels
    plot_integrated_charge_panel(
        df,
        snaps,
        profile_labels,
        outdir / f"{prefix}_panel_D_integrated_charge.png",
        limits,
        stride,
    )
    plot_window_charge_panel(
        df,
        snaps,
        profile_labels,
        outdir / f"{prefix}_panel_E_window_charge.png",
        limits,
        stride,
    )
    plot_integrated_charge_diff_panel(
        df,
        snaps,
        profile_labels,
        outdir / f"{prefix}_panel_D_integrated_charge_alt.png",
        limits,
        stride,
        baseline_label=args.baseline_label,
    )
    plot_protocol_panel(
        df,
        snaps,
        profile_labels,
        outdir / f"{prefix}_panel_protocol.png",
        limits,
    )

    print(f"Wrote standalone panels to: {outdir}")


if __name__ == "__main__":
    main()
