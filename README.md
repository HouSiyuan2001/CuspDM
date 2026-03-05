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
https://doi.org/10.5281/zenodo.18368466

The observational data used in this work (compiled cusp-quasar flux ratios) are available at:
https://broken-yam-b18.notion.site/Lensed-Cusp-Quasar-2fffc9067a748018a1b0e1f13e404ae2

---

# System requirements

- Operating systems: any
- Tested on:
  - macOS (Apple M2 CPU)
  - Linux + NVIDIA A100 GPU (CUDA 11)
- Python: 3.10+
- Hardware (optional): NVIDIA GPU (CUDA) for large-scale runs

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

---

# Demo (small dataset)

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
- demo/Data/Data_json/ (intermediate JSON files)
- demo/Data/lensed_qso_mock_multipole.fits

## 3. Select cusp systems

```sh
python lib/select_cusp_lens_systems.py \
  --fits demo/Data/lensed_qso_mock_multipole.fits \
  --out-fits demo/Data/cusp_all_observable_multipole.fits
```

Note: the demo sample is too small to yield cusp systems, so the result is empty.
For non-empty outputs, use the full simulation. The files
Data/cusp_all_observable_multipole.fits and Data/cusp_all_observable.fits can be
downloaded, and inspected via demo/inspect_catalog_structure.ipynb.

Output:
- demo/Data/cusp_all_observable_multipole.fits

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
python demo/run_two_mock_systems.py --mode mcmc_each_phi --indices 69 --compute cpu
```

## 5. Bayesian analysis

See the paper for the statistical analysis workflow.

---

# Full sample simulation workflow

## 1. Download SIE + external shear mock data

Same as the demo.

## 2. Generate multipole mock catalog

```sh
python Run_Full_Simulation/generate_multipule_mock_catalog.py
```

## 3. Lightcone simulation

```sh
bash Run_Full_Simulation/run_lightcone.sh
```

## 4. MCMC for each phi bin

```sh
bash Run_Full_Simulation/run_MCMC_each_phi.sh
```

## Expected runtime (full simulation)

- Lightcone generation (4 x A100 GPU): about 2 days
- MCMC Rcusp-phi statistics: about 3 weeks

---

# Using your own data

1. Prepare a FITS catalog with the same structure as:
   Data/cusp_all_observable.fits

2. Run the simulation pipeline:

```sh
python Run_Simulation/Mock_sim.py \
  --fits /path/to/your_catalog.fits \
  --start-idx 0 \
  --count 100 \
  --mode lightcone
```

3. For MCMC across phi bins, use:
   --mode mcmc_each_phi

---

# Reproducibility

Example notebooks to reproduce all figures from https://arxiv.org/abs/2601.16818
are provided in the Paper_image folder.

To use them, download all files from https://doi.org/10.5281/zenodo.18368466
and place them in Data.

For full reproducibility of the dataset, run the complete simulation workflow.

---

# Citation

If you use this code or the simulated data, please cite:

Flux-ratio anomalies in cusp quasars reveal dark matter beyond CDM
https://doi.org/10.48550/arXiv.2601.16818

---

# License

MIT License (see LICENSE).

---

# Acknowledgments

pyHalo: substructure and line-of-sight halo modeling
https://github.com/dangilman/pyHalo

Multipole priors reference work:
M. S. H. Oh, A. Nierenberg, D. Gilman, S. Birrer
Joint Semi-Analytic Multipole Priors from Galaxy Isophotes and Their Constraints from Lensed Arcs
