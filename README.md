# CuspDM: Flux-Ratio Anomalies in Cusp-Configured Strong Lenses

[![arXiv](https://img.shields.io/badge/arXiv-2601.16818-green.svg)](https://arxiv.org/abs/2601.16818)

This repository provides a forward-modeling and Bayesian inference framework for studying
flux-ratio anomalies in cusp-configured strongly lensed quasars.

The framework compares predictions from different dark matter scenarios:
- Cold dark matter (CDM)
- Self-interacting dark matter (SIDM)
- Fuzzy dark matter (FDM)

It generates large ensembles of mock lens realizations and compares them with observations.
The code implements macromodel-independent predictions of the normalized cusp relation,
and enables robust statistical comparisons with microlensing-free flux-ratio measurements.

- Authors: Siyuan Hou, Shucheng Xiang, Yue-Lin Sming Tsai, Daneng Yang, Yiping Shu, Nan Li,
  Jiang Dong, Zizhao He, Guoliang Li, Yizhong Fan

![pic](rcusp_overview.png)

---

# Data availability

The complete simulated data products are publicly available on Zenodo:

DOI:
https://zenodo.org/records/18979291

This dataset contains simulated data used for the statistical analysis of
flux-ratio anomalies in cusp-configured strongly lensed quasars. The data
characterize the dependence of the Rcusp statistic on the opening angle phi
for different dark matter scenarios, and provide the theoretical data basis
for the Bayesian inference analysis in arXiv:2601.16818.

File description (place the downloaded files into the new Data/ folder):

- merged_by_axis_type.pkl
  Simulated dataset based on the SIE plus external shear macromodel, including
  Smooth, CDM, SIDM, and FDM realizations of mock cusp lenses, grouped by cusp
  axis type (major-axis vs. minor-axis).
- merged_by_axis_type_mul.pkl
  Extended dataset constructed from merged_by_axis_type.pkl by incorporating
  higher-order multipole perturbations with m=3,4.
- lensed_qso_mock.fits
  Base mock lens catalog (SIE + external shear) used as the starting point for
  generating multipole-perturbed realizations and selecting cusp systems.
- lensed_qso_mock_multipole.fits
  Mock lens catalog with higher-order multipole perturbations added to the
  macromodel; contains merged lensing predictions for each simulated system.
- cusp_all_observable.fits
  Observable cusp-configured subset selected from lensed_qso_mock.fits after
  applying the observability cuts used throughout the analysis.
- cusp_all_observable_multipole.fits
  Observable cusp-configured subset selected from lensed_qso_mock_multipole.fits.
- OneSystem.tar.gz
  Example per-system output archive (lightcone, Rcusp-phi data, and MCMC chains)
  for a single mock lens system, useful for quick inspection or debugging.

The observational data used in this work (compiled cusp-quasar flux ratios) are available at:

Notion database:
https://broken-yam-b18.notion.site/Lensed-Cusp-Quasar-2fffc9067a748018a1b0e1f13e404ae2

---

# System requirements

Operating systems: any

Tested on:
- macOS (Apple M2 CPU)
- Linux + NVIDIA A100 GPU (CUDA 11)

Python:
Python 3.10+

Hardware (optional):
NVIDIA GPU (CUDA) for large-scale runs

---

## Software dependencies (tested versions)

Linux (A100, CUDA 11):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-linux.txt
```

macOS (Apple M2 CPU):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
```

---

# Code overview

![pic](Flowchart.png)

See the Method section of https://arxiv.org/abs/2601.16818 for details.

# Demo (small dataset)

A small-scale example is included.

## 1. Download SIE + external shear mock data

Zenodo:
https://zenodo.org/records/12739548

China-VO:
https://nadc.china-vo.org/res/r101465/

Reference paper:
Dong+2024
http://arxiv.org/abs/2407.10470

Place the file at:

Data/lensed_qso_mock.fits

## 2. Generate SIE + external shear + multipole mock data

Note: only 1000 lenses are randomly selected in this demo.

```sh
python demo/generate_multipule_mock_catalog.py

python lib/compute_lensing_for_mock_catalog.py \
  --sim-idx 0 \
  --num-sim 1000 \
  --fix 1 \
  --nnn 1000 \
  --fits demo/Data/lensed_qso_mock_multipole_temp.fits

python lib/merge_calc_results_to_fits.py \
  --fits demo/Data/lensed_qso_mock_multipole_temp.fits \
  --json-dir demo/Data/Data_json \
  --out-fits demo/Data/lensed_qso_mock_multipole.fits
```

Outputs:

After completion, you will get:
- demo/Data/Data_json/ (intermediate JSON files)
- demo/Data/lensed_qso_mock_multipole.fits

## 3. Select cusp systems

```sh
python lib/select_cusp_lens_systems.py \
  --fits demo/Data/lensed_qso_mock_multipole.fits \
  --out-fits demo/Data/cusp_all_observable_multipole.fits
```

Note: the demo sample is too small to yield cusp systems, so the result is empty.
For non-empty outputs, use the full simulation. For subsequent calculations, download:
Data/cusp_all_observable_multipole.fits and Data/cusp_all_observable.fits, and inspect
via demo/inspect_catalog_structure.ipynb.

Output:
Merged lens catalog:
- demo/Data/cusp_all_observable_multipole.fits (selected cusp systems)

## 4. Dark-matter lightcone generation and MCMC sampling

Note: due to the heavy compute cost, the demo randomly selects 2 lens systems.

```sh
python demo/run_two_mock_systems.py \
  --fits demo/Data/cusp_all_observable_multipole.fits \
  --num-systems 2 \
  --mode both
```

Apple users are advised to use CPU:

```sh
python demo/run_two_mock_systems.py --mode both --compute cpu
```

### Lightcone-only runtime (Apple M2 CPU)

- FDM: about 10 minutes per system
- CDM/SIDM: about 15 minutes per system
- MCMC: one phi bin

Runtime depends on subhalo counts and hardware.

### MCMC-only test

```sh
python demo/run_two_mock_systems.py --mode mcmc_each_phi --indices xx --compute cpu
```

Note: runtime is long. One phi bin takes 10+ minutes on Apple M2 CPU.

## 5. Bayesian analysis

Merge the final results:

```sh
python demo/Organizing_Rcusp_phi.py
```

This produces the final Rcusp-phi data file for each dark matter model:
merged_by_axis_type.pkl

For Bayesian analysis, use the completed simulation sample. Download the full sample data
from https://zenodo.org/records/18979291, place merged_by_axis_type.pkl and
merged_by_axis_type_mul.pkl into Data, then go to Paper_image/Bey.ipynb for analysis.
This notebook reproduces the Bayesian results in https://arxiv.org/abs/2601.16818.

---

# Full sample simulation workflow

Full simulations are computationally heavy and are recommended on Linux with NVIDIA GPU (CUDA).

Expected runtime:
- Lightcone generation (4 x A100 GPU): about 2 days
- MCMC Rcusp-phi statistics: about 3 weeks

## 1. Download SIE + external shear mock data

Same as the demo.

---

## 2. Generate multipole mock catalog

```sh
python Run_Full_Simulation/generate_multipule_mock_catalog.py
```

---

## 3. Lightcone simulation

```sh
bash Run_Full_Simulation/run_lightcone.sh
```

---

## 4. MCMC for each phi bin

```sh
bash Run_Full_Simulation/run_MCMC_each_phi.sh
```

---

## Using your own data

Prepare a FITS catalog with the same structure as Data/cusp_all_observable.fits,
then follow the simulation workflow above.

# Reproducibility

Example notebooks to reproduce all figures from https://arxiv.org/abs/2601.16818
are provided in the Paper_image folder.

To use them, download all files from https://zenodo.org/records/18979291
and place them in Data.

For full reproducibility of the dataset, run the complete simulation workflow.

---

# Citation

If you use this code or the simulated data, please cite:

Flux-ratio anomalies in cusp quasars reveal dark matter beyond CDM
https://doi.org/10.48550/arXiv.2601.16818

---

# License and code release

MIT License (see LICENSE).

---

# Acknowledgments

pyHalo: substructure and line-of-sight halo modeling
https://github.com/dangilman/pyHalo

Multipole priors reference work:
M. S. H. Oh, A. Nierenberg, D. Gilman, S. Birrer
Joint Semi-Analytic Multipole Priors from Galaxy Isophotes and Their Constraints from Lensed Arcs
