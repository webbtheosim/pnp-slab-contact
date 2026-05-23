#!/bin/bash

## asym sweep
#for i in {1..5}; do
#    tag=$(printf "asym_%02d" "$i")
#    echo "Processing ${tag}"
#    #python scripts/plot_panels_v2.py \
#    python scripts/plot_contact_panels.py \
#  --csv data/run_${tag}_sweep.csv \
#  --snapshots data/run_${tag}_sweep_snapshots.npz \
#  --outdir analysis/${tag} \
#  --motion-mode bilateral
#done
#
## gap sweep
#for i in {1..10}; do
#    tag=$(printf "gap_%02d" "$i")
#    echo "Processing ${tag}"
#    python scripts/plot_contact_panels.py \
#  --csv data/run_${tag}_sweep.csv \
#  --snapshots data/run_${tag}_sweep_snapshots.npz \
#  --outdir analysis/${tag} \
#  --motion-mode bilateral
#done
#
## rate sweep
#for i in {1..6}; do
#    tag=$(printf "rate_%02d" "$i")
#    echo "Processing ${tag}"
#    python scripts/plot_contact_panels.py \
#  --csv data/run_${tag}_sweep.csv \
#  --snapshots data/run_${tag}_sweep_snapshots.npz \
#  --outdir analysis/${tag} \
#  --motion-mode bilateral
#done
#
## diff sweep
#for i in {1..8}; do
#    tag=$(printf "diff_%02d" "$i")
#    echo "Processing ${tag}"
#    python scripts/plot_contact_panels.py \
#  --csv data/run_${tag}_sweep.csv \
#  --snapshots data/run_${tag}_sweep_snapshots.npz \
#  --outdir analysis/${tag} \
#  --motion-mode bilateral
#done
#
## salt sweep
#for i in {1..5}; do
#    tag=$(printf "salt_%02d" "$i")
#    echo "Processing ${tag}"
#    python scripts/plot_contact_panels.py \
#  --csv data/run_${tag}_sweep.csv \
#  --snapshots data/run_${tag}_sweep_snapshots.npz \
#  --outdir analysis/${tag} \
#  --motion-mode bilateral
#done
#
#python scripts/plot_sweep_summary.py \
#  --input analysis/sweep_summary_added.csv \
#  --x asymmetry \
#  --y dQdiff_separate_end_C_per_m2 \
#  --where case=asym_* \
#  --sort-x \
#  --fit \
#  --out analysis/asymmetry.png
#

python scripts/plot_sweep_summary.py \
  --input analysis/sweep_summary_added.csv \
  --x h_min_nm \
  --y dQdiff_separate_end_C_per_m2 \
  --where case=gap_* \
  --sort-x \
  --fit \
  --out analysis/gap.png

#python scripts/plot_sweep_summary.py \
#  --input analysis/sweep_summary_added.csv \
#  --x D_dense_scale \
#  --y dQdiff_separate_end_C_per_m2 \
#  --where case=diff_* \
#  --sort-x \
#  --out analysis/diff.png
#
#
#python scripts/plot_sweep_summary.py \
#  --input analysis/sweep_summary_added.csv \
#  --x rate_m_s \
#  --y dQdiff_separate_end_C_per_m2 \
#  --where case=rate_* \
#  --sort-x \
#  --out analysis/rate.png
#
#python scripts/plot_sweep_summary.py \
#  --input analysis/sweep_summary_added.csv \
#  --x Pe \
#  --y dQdiff_separate_end_C_per_m2 \
#  --where case=rate_* \
#  --sort-x \
#  --out analysis/peclet.png
#python scripts/plot_sweep_summary.py \
#  --input analysis/sweep_summary_added.csv \
#  --x c_bulk_molm3 \
#  --y dQdiff_separate_end_C_per_m2 \
#  --where case=salt_* \
#  --sort-x \
#  --out analysis/salt.png
