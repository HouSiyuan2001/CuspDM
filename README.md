# CuspDM: Flux-Ratio Anomalies in Cusp-Configured Strong Lenses

[![arXiv](https://img.shields.io/badge/arXiv-2601.16818-green.svg)](https://arxiv.org/abs/2601.16818)

This repository provides a **forward-modeling and Bayesian inference framework** for studying **flux-ratio anomalies in cusp-configured strongly lensed quasars**.  
It is designed to compare predictions from different dark matter scenarios, including **CDM, SIDM, and fuzzy dark matter (FDM)**, using large ensembles of mock lens realizations.

The code implements macromodel-independent predictions of the normalized cusp relation,  
and enables robust statistical comparisons with observed microlensing-free flux-ratio measurements.

- **Authors**: Siyuan Hou, Shucheng Xiang, Yue-Lin Sming Tsai, Daneng Yang, Yiping Shu, Nan Li, Jiang Dong, Zizhao He, Guoliang Li, Yizhong Fan

![pic](rcusp_overview.png)

---

## Data availability

The complete set of simulated data products is publicly available on Zenodo: **DOI:** https://doi.org/10.5281/zenodo.18368466

The observational data used in this work, including compiled flux-ratio measurements for lensed cusp quasars, are publicly accessible at: [Notion](https://broken-yam-b18.notion.site/Lensed-Cusp-Quasar-2fffc9067a748018a1b0e1f13e404ae2)

## System requirements

- **Operating systems**: Linux or macOS.
  - **Tested on**: macOS 14 (Apple M2, CPU; Metal backend is experimental), Linux + NVIDIA A100 (distribution/version: <fill>).
- **Python**: 3.9+ recommended.
- **Non-standard hardware (optional)**: NVIDIA GPU (CUDA) for large-scale runs; Apple Metal is experimental and may fail for float64.

### Software dependencies (with versions)

Please record the exact versions used for your submission by running:

```shell
python - <<'PY'
import pkg_resources as pr
pkgs = [
    "numpy","scipy","astropy","matplotlib","tqdm",
    "jax","jaxlib","lenstronomy","pyhalo",
]
for name in pkgs:
    try:
        print(f"{name}=={pr.get_distribution(name).version}")
    except Exception:
        print(f"{name}==<not installed>")
PY
```

Then replace the list below with your actual versions:

- numpy==<fill>
- scipy==<fill>
- astropy==<fill>
- matplotlib==<fill>
- tqdm==<fill>
- jax==<fill>
- jaxlib==<fill>
- lenstronomy==<fill>
- pyhalo==<fill>

## Installation guide

```shell
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib jupyter
pip install astropy tqdm
pip install lenstronomy pyhalo
pip install jax jaxlib
```

Notes:
- For GPU/Metal acceleration, follow the official JAX install instructions for your platform.
- Typical install time on a normal desktop is ~5-15 minutes (network dependent).
## Demo (small dataset)

A small demo dataset is included in `demo/Data/` and `demo/Theory_Mock/`.

```shell
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

python lib/select_cusp_lens_systems.py \
  --fits demo/Data/lensed_qso_mock_multipole.fits \
  --target-image-number 5 \
  --max-images 5 \
  --Rcusp 1 \
  --phi 140 \
  --sym 5 \
  --out-fits demo/Data/cusp_all_observable_multipole.fits
```

### Demo: expected output

- `demo/Data/Data_json/` contains intermediate JSON outputs.
- `demo/Data/lensed_qso_mock_multipole.fits` is the merged lensing catalog.
- `demo/Data/cusp_all_observable_multipole.fits` contains selected cusp systems.

### Demo: expected runtime (normal desktop)

- Apple M2 CPU: ~10 min per system for FDM; ~15 min per system for CDM/SIDM.
- GPU/CPU performance depends strongly on the subhalo population and hardware.

### Quick demo (2 random systems)

```shell
python demo/run_two_mock_systems.py \
  --fits demo/Data/cusp_all_observable_multipole.fits \
  --num-systems 2 \
  --mode both
```

Note: Apple Metal is experimental and may fail for float64. Use CPU by default:

```shell
python demo/run_two_mock_systems.py --mode both --compute cpu
```

## Full simulation workflow

运行顺序:

1. Download SIE lens + shear mockdata from [zenodo](https://zenodo.org/records/12739548) or [nadc.china](https://nadc.china-vo.org/res/r101465/) (Paper: [Dong+2024](http://arxiv.org/abs/2407.10470)), 然后放到Data文件夹里
2. 生成多极钜 mockdata: `python Run_Full_Simulation/generate_multipule_mock_catalog.py`
3. Light-cone simulation:
   - `bash Run_Full_Simulation/run_lightcone.sh`
4. MCMC for each phi bin:
   - `bash Run_Full_Simulation/run_MCMC_each_phi.sh`

### Expected runtime (full simulation)

- Light-cone generation with 4x A100: ~2 days.
- MCMC for sufficient Rcusp-phi statistics: ~2-3 weeks.

## Instructions for use on your data

1. Prepare a FITS catalog of lens systems similar to `Data/cusp_all_observable.fits`.
2. Run the pipeline entry point (lightcone or MCMC):

```shell
python Run_Simulation/Mock_sim.py \
  --fits /path/to/your_catalog.fits \
  --start-idx 0 \
  --count 100 \
  --mode lightcone
```

3. For MCMC across phi bins, use `--mode mcmc_each_phi`.

## Reproducibility (optional)

- The demo notebook `demo/inspect_catalog_structure.ipynb` provides a guided walkthrough.
- For full reproduction of manuscript figures, follow the simulation workflow above and replace the demo data with the full dataset.

## Citation

If you use this code or the accompanying simulations, please cite:

Flux-ratio anomalies in cusp quasars reveal dark matter beyond CDM, https://doi.org/10.48550/arXiv.2601.16818

## License and code availability

- License: MIT (see `LICENSE`).
- Code repository: https://github.com/HouSiyuan2001/CuspDM

## Acknowledgments

- Interfaces leverage [`pyHalo`](https://github.com/dangilman/pyHalo) for substructure and line-of-sight halo modeling.
- M. S. H. Oh, A. Nierenberg, D. Gilman, and S. Birrer, Joint Semi-Analytic Multipole Priors from Galaxy Isophotes and Their Constraints from Lensed Arcs.
