#!/usr/bin/env bash

#cases=(
#  baseline_slow
#  neg_ref_slow
#  neg_swap_slow
#  neg_sym_slow
#)
cases=(
    neg_ref_fast
)

for case in "${cases[@]}"; do
  echo "Running case: $case"

  # Run simulation
  python scripts/run_single.py --case "$case"

  #python scripts/run_single.py --case "$case"
  #python scripts/plot_contact_panels.py \
  #--csv data/run_neg_ref_slow_neumann.csv \
  #--snapshots data/run_neg_ref_slow_neumann_snapshots.npz \
  #--outdir neg_ref_slow_panels \
  #--tA-nm 200 \
  #--tB-nm 200 \
  #--window-nm 20

  echo "Completed case: $case"
  echo "----------------------------------------"
done

#python scripts/run_single.py --case neg_ref_fast
#python scripts/run_single.py --case neg_ref_medium
#
#python scripts/make_figures.py \
#  --input data/run_baseline_neumann.csv \
#  --outdir baseline \
#  --dt 1e-11 \
#  --n-pre 2000 \
#  --n-approach 12000 \
#  --n-hold 8000 \
#  --n-separate 20000
#
#python scripts/make_figures.py \
#  --input data/run_sym_ref_slow_neumann.csv \
#  --outdir sym_ref_slow \
#  --dt 1e-11 \
#  --n-pre 2000 \
#  --n-approach 12000 \
#  --n-hold 8000 \
#  --n-separate 20000
#
#python scripts/make_figures.py \
#  --input data/run_sym_ref_fast_neumann.csv \
#  --outdir sym_ref_fast \
#  --dt 1e-11 \
#  --n-pre 2000 \
#  --n-approach 120 \
#  --n-hold 80 \
#  --n-separate 200

#python scripts/make_figures.py \
#  --input data/run_sym_ref_medium_neumann.csv \
#  --outdir sym_ref_medium \
#  --dt 1e-11 \
#  --n-pre 2000 \
#  --n-approach 1200 \
#  --n-hold 800 \
#  --n-separate 2000

#python scripts/make_figures.py \
#  --input data/run_neg_ref_slow_neumann.csv \
#  --outdir neg_ref_slow \
#  --dt 1e-11 \
#  --n-pre 30000 \
#  --n-approach 12000 \
#  --n-hold 8000 \
#  --n-separate 20000

#python scripts/make_figures.py \
#  --input data/run_sym_ref_medium_neumann.csv \
#  --outdir sym_ref_medium \
#  --dt 1e-11 \
#  --n-pre 30000 \
#  --n-approach 1200 \
#  --n-hold 800 \
#  --n-separate 2000

#python scripts/make_figures.py \
#  --input data/run_neg_ref_medium_neumann.csv \
#  --outdir neg_ref_medium \
#  --dt 1e-11 \
#  --n-pre 30000 \
#  --n-approach 1200 \
#  --n-hold 800 \
#  --n-separate 2000
#
#python scripts/make_figures.py \
#  --input data/run_neg_ref_fast_neumann.csv \
#  --outdir neg_ref_fast \
#  --dt 1e-11 \
#  --n-pre 30000 \
#  --n-approach 120 \
#  --n-hold 80 \
#  --n-separate 200

##python scripts/make_figures.py \
#  --input data/run_sym_soft_fast_dt2_neumann.csv \
#  --outdir sym_soft_fast_dt2 \
#  --dt 5e-12 \
#  --n-pre 4000 \
#  --n-approach 240 \
#  --n-hold 160 \
#  --n-separate 400


#python scripts/compare_cases.py \
#  --cases \
#  data/run_neg_ref_slow_neumann.csv \
#  data/run_neg_ref_medium_neumann.csv \
#  data/run_neg_ref_fast_neumann.csv \
#  --dt 1e-11 \
#  --n-pre 30000 \
#  --outdir comparison_soft
#
#
#python scripts/compare_profiles.py \
#  --inputs \
#  data/run_neg_ref_fast_neumann_snapshots.npz \
#  data/run_neg_ref_medium_neumann_snapshots.npz \
#  data/run_neg_ref_slow_neumann_snapshots.npz \
#  --labels fast medium slow \
#  --outdir profile_compare_neg_ref
#
#
#python scripts/make_profiles.py \
#  --input data/run_sym_ref_medium_neumann_snapshots.npz \
#  --outdir sym_ref_medium_profiles\
#  --tA-nm 150 \
#  --tB-nm 150 \
#  --window-nm 40
#
#python scripts/make_profiles.py \
#  --input data/run_neg_ref_fast_neumann_snapshots.npz \
#  --outdir neg_ref_fast_profiles\
#  --tA-nm 150 \
#  --tB-nm 150 \
#  --window-nm 40
#
#python scripts/make_profiles.py \
#  --input data/run_neg_ref_medium_neumann_snapshots.npz \
#  --outdir neg_ref_medium_profiles\
#    --tA-nm 150 \
#  --tB-nm 150 \
#  --window-nm 40
#
#python scripts/make_profiles.py \
#  --input data/run_neg_ref_slow_neumann_snapshots.npz \
#  --outdir neg_ref_slow_profiles\
#    --tA-nm 150 \
#  --tB-nm 150 \
#  --window-nm 40
