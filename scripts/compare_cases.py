from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path: Path):
    with path.open("r") as f:
        reader = csv.DictReader(f)
        cols = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                cols[k].append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


def compute_metrics(data, dt, n_pre):
    t = data["t_s"]
    J = data["Jcharge_mid_A_per_m2"]
    dphi_A = data["dphi_A_V"]
    dphi_B = data["dphi_B_V"]

    t_pre_end = n_pre * dt
    mask = t >= t_pre_end

    t_post = t[mask]
    J_post = J[mask]
    dphi_A_post = dphi_A[mask]
    dphi_B_post = dphi_B[mask]

    Qtrans = float(np.trapezoid(J_post, t_post))
    J95 = float(np.percentile(np.abs(J_post), 95))
    Jrms = float(np.sqrt(np.mean(J_post**2)))

    return dict(
        Qtrans=Qtrans,
        max_dphi_A=1e3 * float(np.max(np.abs(dphi_A_post))),
        max_dphi_B=1e3 * float(np.max(np.abs(dphi_B_post))),
        J95=J95,
        Jrms=Jrms,
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="CSV files to compare",
    )

    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--n-pre", type=int, required=True)

    parser.add_argument("--outdir", default="comparison")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []

    plt.figure()

    for casefile in args.cases:

        path = Path(casefile)
        name = path.stem.replace("run_", "")

        data = read_csv(path)

        metrics = compute_metrics(data, args.dt, args.n_pre)
        metrics["case"] = name
        results.append(metrics)

        t = data["t_s"]
        dphi = 1e3 * data["dphi_A_V"]

        plt.plot(t, dphi, label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("Δφ_A (mV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "dphi_overlay.png", dpi=200)
    plt.close()

    # Current overlay
    plt.figure()

    for casefile in args.cases:
        path = Path(casefile)
        name = path.stem.replace("run_", "")

        data = read_csv(path)

        plt.plot(data["t_s"], data["Jcharge_mid_A_per_m2"], label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("J (A/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "current_overlay.png", dpi=200)
    plt.close()

    # Summary table
    print("\nSummary metrics\n")

    header = [
        "case",
        "Qtrans (C/m²)",
        "max|Δφ_A| (mV)",
        "max|Δφ_B| (mV)",
        "J95 (A/m²)",
        "Jrms (A/m²)",
    ]

    print("{:20s} {:12s} {:12s} {:12s} {:12s} {:12s}".format(*header))

    for r in results:
        print(
            "{:20s} {:12.3e} {:12.2f} {:12.2f} {:12.3e} {:12.3e}".format(
                r["case"],
                r["Qtrans"],
                r["max_dphi_A"],
                r["max_dphi_B"],
                r["J95"],
                r["Jrms"],
            )
        )

    # Bar chart comparison
    labels = [r["case"] for r in results]
    qvals = [r["Qtrans"] for r in results]

    plt.figure()
    plt.bar(labels, qvals)
    plt.ylabel("Integrated charge transfer (C/m²)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / "Qtrans_bar.png", dpi=200)
    plt.close()

    print(f"\nWrote comparison figures to: {outdir}")


if __name__ == "__main__":
    main()
