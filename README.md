# PNP slab contact simulations

This repository contains the one-dimensional Poisson-Nernst-Planck (PNP) simulations used to support the analysis of contact-dependent ion redistribution between charged soft interfaces in aqueous solution. 

The simulations are intended as a minimal electrokinetic model. They describe two (charged) soft slabs separated by an electrolyte-filled gap and surrounded by bulk electrolyte reservoirs. Mobile ions evolve by diffusion and electromigration, while the electrostatic potential is obtained from the Poisson equation. The slabs undergo a prescribed approach-hold-separation-relaxation protocol.

## Repository contents

```text
src/pnpcc/
    params.py              # Model parameters and defaults
    poisson.py             # Poisson solver
    nernst_planck.py       # Nernst-Planck ion updates
    regions.py             # Domain/region construction
    protocol.py            # Gap/contact protocol
    simulate.py            # Main time integration routine
    io.py                  # Output utilities

scripts/
    run_sweep_case.py      # Runs individual cases or parameter-sweep families
    analyze_sweep.py       # Summarizes output CSV files

analysis/
    sweep_summary_added.csv      # Processed sweep metrics used for figures
    *.csv                        # csv summaries of processed sweeps
    *.png                        # example figures generated from csv

## System requirements

The code was developed and tested on macOS using Python 3.11. It should also run on Linux or macOS with a standard scientific Python environment. The simulations are CPU-only and do not require specialized hardware.

Required Python packages:

```text
numpy
pandas
matplotlib
```

For reproducibility, the recommended setup is a fresh conda or virtual environment with Python 3.11.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd pnp-contact-charging
```

Create and activate a conda environment:

```bash
conda create -n pnpcc python=3.11 numpy pandas matplotlib
conda activate pnpcc
```

Install the local package in editable mode:

```bash
pip install -e .
```

Alternatively, using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib
pip install -e .
```

Typical installation time on a current laptop is less than a few minutes.

## Running simulations

The main script for running simulations is:

```bash
python scripts/run_sweep_case.py --case <case_name> --outdir data --tag sweep
```

For example, to run the representative asymmetric surface-charge case used for the main simulation figure:

```bash
python scripts/run_sweep_case.py --case asym_04 --outdir data --tag sweep
```

This writes:

```text
data/run_asym_04_sweep.csv
data/run_asym_04_sweep_snapshots.npz
data/run_asym_04_sweep_meta.json
```

The `.csv` file contains time-resolved integrated quantities, the `.npz` file contains saved concentration and potential profiles, and the `.json` file records simulation metadata.

## Running sweep families

The same script can run predefined parameter-sweep families:

```bash
python scripts/run_sweep_case.py --case controls --outdir data --tag sweep
python scripts/run_sweep_case.py --case asym     --outdir data --tag sweep
python scripts/run_sweep_case.py --case gap      --outdir data --tag sweep
python scripts/run_sweep_case.py --case diff     --outdir data --tag sweep
python scripts/run_sweep_case.py --case rate     --outdir data --tag sweep
python scripts/run_sweep_case.py --case salt     --outdir data --tag sweep
```

Defined sweep families include:

```text
controls   neutral, symmetric, asymmetric, and swapped-asymmetry controls
asym       fixed-charge asymmetry sweep
gap        minimum-gap sweep
diff       dense-phase ion diffusivity sweep
rate       approach/separation rate sweep
salt       bulk electrolyte concentration sweep
```

Depending on the sweep family and hardware, full sweeps may take from minutes to hours.
Parameters for the sweeps can be edited ```scripts/run_sweep_case.py```


## Representative case

The representative case is:

```bash
python scripts/run_sweep_case.py --case asym_04 --outdir data --tag sweep
```

This case uses asymmetric fixed interfacial charge densities:

```text
rho_f_A = -2.0e6 C m^-3
rho_f_B = -1.2e7 C m^-3
```

with a 150 mM monovalent electrolyte, 200 nm initial gap, 2 nm minimum gap, and bilateral approach/separation protocol.
