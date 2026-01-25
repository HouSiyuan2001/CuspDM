# CuspDM: Flux-Ratio Anomalies in Cusp-Configured Strong Lenses

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-green.svg)](https://arxiv.org/abs/XXXX.XXXXX)

This repository provides a **forward-modeling and Bayesian inference framework** for studying **flux-ratio anomalies in cusp-configured strongly lensed quasars**.  
It is designed to compare predictions from different dark matter scenarios, including **CDM, SIDM, and fuzzy dark matter (FDM)**, using large ensembles of mock lens realizations.

The code implements macromodel-independent predictions of the normalized cusp relation,  
and enables robust statistical comparisons with observed microlensing-free flux-ratio measurements.

- **Authors**: Siyuan Hou, Shucheng Xiang, Yue-Lin Sming Tsai, Daneng Yang, Yiping Shu, Nan Li, Jiang Dong, Zizhao He, Guoliang Li, Yizhong Fan

![pic](rcusp_overview.png)

---

## File structure

- **pipeline/**
  - Core scripts for generating mock cusp lenses, sampling source positions, and computing flux ratios.
  - Includes forward modeling with subhalos and line-of-sight structures.

- **models/**
  - Implementations of different dark matter scenarios:
    - CDM
    - SIDM
    - FDM
  - Interfaces with [`pyHalo`](https://github.com/dangilman/pyHalo) and parametric macromodels.

- **analysis/**
  - KDE construction, per-$\phi$ normalization, and Bayesian model comparison.
  - Scripts for producing $R_{\rm cusp}$–$\phi$ distributions and Bayes-factor tables.

- **example/**
  - Minimal notebooks demonstrating the full workflow:
    - mock generation
    - lensing calculation
    - comparison with observed systems

---

## Requirements

The code is written in **Python** and relies on standard scientific packages.
Some components use [JAX](https://github.com/google/jax) for accelerated computations.

```shell
pip install numpy scipy matplotlib jupyter
pip install astropy tqdm
pip install lenstronomy pyhalo
pip install jax jaxlib
```


## Citation

If you use this code or the accompanying simulations, please cite:

Flux-ratio anomalies in cusp quasars reveal dark matter beyond CDM,
arXiv:XXXX.XXXXX

## Acknowledgments

- Interfaces leverage [`pyHalo`](https://github.com/dangilman/pyHalo) for substructure and line-of-sight halo modeling.
- M. S. H. Oh, A. Nierenberg, D. Gilman, and S. Birrer, Joint Semi-Analytic Multipole Priors from Galaxy Isophotes and Their Constraints from Lensed Arcs.
