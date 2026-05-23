from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Style setup (matches your example)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.8,
})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def write_plot_data_csv(df: pd.DataFrame, outpath: Path, xcol: str, ycol: str):
    """
    Write the exact rows and columns used for the sweep plot.
    Includes case names when available.
    """
    outpath = Path(outpath)
    csv_path = outpath.with_suffix(".csv")

    cols = []
    if "case" in df.columns:
        cols.append("case")
    cols.extend([xcol, ycol])


    df[cols].to_csv(csv_path, index=False)
    print(f"Wrote plot data: {csv_path}")

def parse_value(text: str) -> Any:
    try:
        if any(c in text for c in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def apply_filters(df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
    out = df.copy()

    for filt in filters:
        if "=" not in filt:
            raise ValueError(f"Bad filter '{filt}'")

        col, val = filt.split("=", 1)
        col = col.strip()
        val = val.strip()

        if col not in out.columns:
            raise KeyError(f"{col} not in dataframe")

        series = out[col]

        # wildcard matching for strings
        if isinstance(val, str) and "*" in val:
            out = out[series.astype(str).apply(lambda x: fnmatch.fnmatch(x, val))]
        else:
            parsed = parse_value(val)
            if pd.api.types.is_numeric_dtype(series):
                out = out[series.astype(float) == float(parsed)]
            else:
                out = out[series.astype(str) == str(parsed)]

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--where", action="append", default=[])
    parser.add_argument("--sort-x", action="store_true")
    parser.add_argument("--fit", action="store_true", help="Add linear fit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--title", default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = apply_filters(df, args.where)

    if df.empty:
        raise ValueError("No data after filtering")

    if args.sort_x:
        df = df.sort_values(args.x)

    x = df[args.x].values
    y = -1*df[args.y].values

    # -----------------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(2.0, 0.65))


    # Optional linear fit
    if args.fit and len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        xfit = np.linspace(min(x), max(x), 100)
        yfit = coeffs[0] * xfit + coeffs[1]

        ax.plot(xfit, yfit, linestyle="--",linewidth=0.5,color=(0.2,0.2,0.2,1))

        # annotation (like example figure)
        #ax.text(
        #    0.05,
        #    0.05,
        #    f"y = {coeffs[0]:.2g}x + {coeffs[1]:.2g}",
        #    transform=ax.transAxes,
        #)

    ax.scatter(x, y, s=12,clip_on=False,color=(0.3,0.3,0.9))  # circular markers

    # Labels
    #ax.set_xlabel(args.xlabel or args.x)
    #ax.set_ylabel(args.ylabel or args.y)
    ax.set_ylim(0,0.025)

    #if args.title:
    #    ax.set_title(args.title)

    # Styling to match your figure
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    plt.close(fig)

    write_plot_data_csv(df, out, args.x, args.y)

    print(f"Wrote {out}")
    print(f"N = {len(df)} points")


if __name__ == "__main__":
    main()
