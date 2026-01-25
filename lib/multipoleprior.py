"""
This is a supplemantary code for "Joint Semi-Analytic Multipole Priors from Galaxy Isophotes and Their Constarints from Lensed Arcs" by Maverick S. H. Oh, Anna Nierenberg, Daniel Gilman, and Simon Birrer, submitted to Journal of Cosmology and Astroparticle Physics (JCAP). Please cite this work if you use this package.
"""
import json
from pathlib import Path
from typing import Union, Dict, Any, Optional


def load_default_params(json_source: Optional[Union[str, Path, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Load prior parameters and limits, returning a dict with top-level keys "params" and "limits".
    Usage:
      - load_default_params()                             # Read params_for_prior.json in the same directory
      - load_default_params("path/to/params.json")        # Read from a specified path
      - load_default_params({...})                        # Use an already parsed dict

    The parameter file should follow the example JSON structure:
      {
        "params": { ... },
        "limits": { ... }
      }
    """
    def _ensure_top_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(d, dict) or "params" not in d or "limits" not in d:
            raise ValueError('Parameter JSON must contain top-level keys "params" and "limits".')
        return d

    if json_source is None:
        # Default: read params_for_prior.json in the same directory as this script
        default_path = Path(__file__).with_name("params_for_prior.json")
        with open(default_path, "r", encoding="utf-8") as f:
            return _ensure_top_keys(json.load(f))

    if isinstance(json_source, dict):
        return _ensure_top_keys(json_source)

    if isinstance(json_source, (str, Path)):
        # Also support a JSON string instead of a file path
        src = str(json_source)
        try:
            if "{" in src and "}" in src:
                return _ensure_top_keys(json.loads(src))
        except json.JSONDecodeError:
            pass  # Not JSON; fall back to treating it as a file path

        with open(json_source, "r", encoding="utf-8") as f:
            return _ensure_top_keys(json.load(f))

    raise TypeError("json_source must be None, dict, str, or pathlib.Path.")

#!/usr/bin/env python3
"""
DistributionModel_nontorch
--------------------------
A NumPy / SciPy replacement for the original torch-based `DistributionModel`.

* Reads **all parameters** and **domain limits** from a single JSON file that
  must contain two top-level keys: `"params"` and `"limits"`.
* Hard-codes `n_interp_m3 = 2` and `n_interp_m4 = 3`.
* Exposes a convenience method
      prob_single_point(x3, y3, x4, y4, z)
  that returns the fully-normalised joint probability density
  P(x3, y3, x4, y4, z).
  The symbols here correspond to physical variable as below.
  x3: a3/a          (strength of m=3 multipole, denoted as 'a3_a' in kwargs_lens)
  y3: phi3-\phi0    (angle of m=3 multipole, denoted as 'delta_phi_m3' in kwargs_lens)
  x4: a4/a          (strength of m=4 multipole, denoted as 'a4_a' in kwargs_lens)
  y4: phi4-\phi0    (angle of m=4 multipole, denoted as 'delta_phi_m4' in kwargs_lens)
  z : q             (axis ratio b/a, converted from 'e1' and 'e2' of kwargs_lens using q = ellipticity2phi_q(e1, e2)[1])
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
#                               Helper functions                               #
# -----------------------------------------------------------------------------#

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def linear_spline_eval(x, x_ctrl, y_ctrl):
    """Simple piece-wise linear interpolation (1-D)."""
    return np.interp(x, x_ctrl, y_ctrl, left=y_ctrl[0], right=y_ctrl[-1])

def skew_normal_pdf(x, alpha, xi, omega):
    """
    Skew-normal PDF:
        f(x) = 2/ω · φ( (x-ξ)/ω ) · Φ( α·(x-ξ)/ω )
    where φ, Φ are the standard normal pdf/cdf.
    """
    t = (x - xi) / omega
    return 2.0 / omega * norm.pdf(t) * norm.cdf(alpha * t)

def generalized_gaussian_unnorm(x, alpha, beta):
    """
    Un-normalised Generalised-Gaussian:
        f(x) ∝ exp( - |x/α|^β )
    """
    return np.exp(-np.abs(x / alpha) ** beta)

def flat_unnorm(x):
    """Uniform (unnormalised) → 1 everywhere."""
    return np.ones_like(x, dtype=float)

# -----------------------------------------------------------------------------#
#                           The main (non-torch) class                          #
# -----------------------------------------------------------------------------#

class DistributionModel_nontorch:
    """NumPy / SciPy port of the original DistributionModel."""

    # --------------------------------------------------------------------- #
    #                                constructor                            #
    # --------------------------------------------------------------------- #
    def __init__(self, param_dict):

        p = param_dict["params"]
        lim = param_dict["limits"]

        # ------------------- basic limits ------------------- #
        self.x3_min, self.x3_max = lim["x3_min"], lim["x3_max"]
        self.y3_min, self.y3_max = lim["y3_min"], lim["y3_max"]
        self.x4_min, self.x4_max = lim["x4_min"], lim["x4_max"]
        self.y4_min, self.y4_max = lim["y4_min"], lim["y4_max"]
        self.z_min,  self.z_max  = lim["z_min"],  lim["z_max"]

        # ---------------------- P(Z) ------------------------- #
        self.alpha_Z = p["alpha_Z"]
        self.xi_Z    = p["xi_Z"]
        self.omega_Z = p["omega_Z"]

        # ------------------ controls (m = 3) ---------------- #
        self.sigma_z_x_ctrl_3 = np.asarray(p["sigma_z_x_ctrl_3"], dtype=float)
        self.sigma_z_y_ctrl_3 = np.asarray(p["sigma_z_y_ctrl_3"], dtype=float)

        # ------------------ controls (m = 4) ---------------- #
        self.alpha_z_x_ctrl_4  = np.asarray(p["alpha_z_x_ctrl_4"], dtype=float)
        self.alpha_z_y_ctrl_4  = np.asarray(p["alpha_z_y_ctrl_4"], dtype=float)
        self.xi_z_x_ctrl_4     = np.asarray(p["xi_z_x_ctrl_4"], dtype=float)
        self.xi_z_y_ctrl_4     = np.asarray(p["xi_z_y_ctrl_4"], dtype=float)
        self.omega_z_x_ctrl_4  = np.asarray(p["omega_z_x_ctrl_4"], dtype=float)
        self.omega_z_y_ctrl_4  = np.asarray(p["omega_z_y_ctrl_4"], dtype=float)

        self.alpha_x_ctrl_4    = np.asarray(p["alpha_x_ctrl_4"], dtype=float)
        self.alpha_y_ctrl_4    = np.asarray(p["alpha_y_ctrl_4"], dtype=float)
        self.beta_x_ctrl_4     = np.asarray(p["beta_x_ctrl_4"], dtype=float)
        self.beta_y_ctrl_4     = np.asarray(p["beta_y_ctrl_4"], dtype=float)

        # fixed – required by spec
        self.n_interp_m3 = 2
        self.n_interp_m4 = 3

    # --------------------------------------------------------------------- #
    #                          one-dimensional PDFs                         #
    # --------------------------------------------------------------------- #

    # ---------- P(Z) ----------
    def pz(self, z, single_point=True, n_grid=100):
        val = skew_normal_pdf(z, self.alpha_Z, self.xi_Z, self.omega_Z)


        z_grid = np.linspace(self.z_min, self.z_max, n_grid)
        normalize   = np.trapz(skew_normal_pdf(z_grid, self.alpha_Z,
                                          self.xi_Z,  self.omega_Z),
                          x=z_grid)
        return val / (normalize + 1e-12)

    # ---------- P(X3 | Z) ----------
    def px3z(self, x3, z, single_point=True, n_grid=100):
        sigma = linear_spline_eval(z, self.sigma_z_x_ctrl_3,
                                      self.sigma_z_y_ctrl_3)
        pdf   = norm.pdf(x3, loc=0.0, scale=sigma)


        x_grid = np.linspace(self.x3_min, self.x3_max, n_grid)
        normalize   = np.trapz(norm.pdf(x_grid, loc=0.0, scale=sigma), x=x_grid)
        return pdf / (normalize + 1e-12)

    # ---------- P(Y3 | X3) ----------
    def py3x3(self, y3, single_point=True):
        pdf = flat_unnorm(y3)
        return pdf / (self.y3_max - self.y3_min + 1e-12)

    # ---------- P(X4 | Z) ----------
    def px4z(self, x4, z, single_point=True, n_grid=100):
        alpha = linear_spline_eval(z, self.alpha_z_x_ctrl_4,
                                      self.alpha_z_y_ctrl_4)
        xi    = linear_spline_eval(z, self.xi_z_x_ctrl_4,
                                      self.xi_z_y_ctrl_4)
        omega = linear_spline_eval(z, self.omega_z_x_ctrl_4,
                                      self.omega_z_y_ctrl_4)

        pdf = skew_normal_pdf(x4, alpha, xi, omega)

        x_grid = np.linspace(self.x4_min, self.x4_max, n_grid)
        normalize   = np.trapz(skew_normal_pdf(x_grid, alpha, xi, omega),
                          x=x_grid)
        return pdf / (normalize + 1e-12)

    # ---------- P(Y4 | X4) ----------
    def py4x4(self, y4, x4, single_point=True, n_grid=100):
        alpha = linear_spline_eval(x4, self.alpha_x_ctrl_4,
                                      self.alpha_y_ctrl_4)
        beta  = linear_spline_eval(x4, self.beta_x_ctrl_4,
                                      self.beta_y_ctrl_4)

        pdf_unn = generalized_gaussian_unnorm(y4, alpha, beta)

        y_grid = np.linspace(self.y4_min, self.y4_max, n_grid)
        normalize   = np.trapz(generalized_gaussian_unnorm(y_grid, alpha, beta),
                          x=y_grid)
        return pdf_unn / (normalize + 1e-12)

    # --------------------------------------------------------------------- #
    #                     joint densities & public API                      #
    # --------------------------------------------------------------------- #

    def forward_m3(self, x3, y3, z, single_point=True):
        """P(Z) · P(X3|Z) · P(Y3|X3)."""
        return ( self.pz(z, single_point) *
                 self.px3z(x3, z, single_point) *
                 self.py3x3(y3, single_point) )

    def forward_m4(self, x4, y4, z, single_point=True):
        """P(Z) · P(X4|Z) · P(Y4|X4)."""
        return ( self.pz(z, single_point) *
                 self.px4z(x4, z, single_point) *
                 self.py4x4(y4, x4, single_point) )

    def forward(self, x3, y3, x4, y4, z, single_point=True):
        """
        Full joint assuming (X3,Y3) ⟂ (X4,Y4) | Z:
            P(X3,Y3,X4,Y4,Z) =
            P(Z)·P(X3|Z)·P(Y3|X3)·P(X4|Z)·P(Y4|X4) / P(Z)
        (The / P(Z) keeps the conditional-independence structure identical
        to the original torch implementation.)
        """
        return ( self.forward_m3(x3, y3, z, single_point) *
                 self.forward_m4(x4, y4, z, single_point) /
                 self.pz(z, single_point) )

    # --- convenience wrapper (matches old forward_from_nontorch_single_point) ---
    def prob_single_point(self, x3, y3, x4, y4, z):
        """Return the *fully* normalised joint PDF value at a single point."""
        return float(self.forward(x3, y3, x4, y4, z, single_point=True))



import numpy as np
from astropy.io import fits
from numba import njit
from lenstronomy.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity



params = load_default_params()
model = DistributionModel_nontorch(params)


@njit
def log_prior_q_phi(e1, e2):
    e = np.sqrt(e1 * e1 + e2 * e2)
    if e >= 1.0:
        return -np.inf
    eps = 1e-12
    return np.log(1.0 / np.pi / ((e + eps) * (1 + e)**2))

def e_to_q(e):  # From Appendix E: c = e = (1-q)/(1+q) → q = (1-e)/(1+e)
    # From paper Appendix E: c = e = (1-q)/(1+q) → q = (1-e)/(1+e)
    return (1.0 - e) / (1.0 + e)


def _inverse_sample_from_pdf(x_grid, pdf, size=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pdf = np.asarray(pdf, dtype=float)
    pdf = np.clip(pdf, 0.0, np.inf)
    area = np.trapz(pdf, x_grid)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("PDF normalization failed.")
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * np.diff(x_grid) / 2.0)
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]
    u = rng.uniform(0.0, 1.0, size=size)
    return np.interp(u, cdf, x_grid)


def _sample_z(n, rng=None, n_grid=2048):
    if rng is None:
        rng = np.random.default_rng()
    z_grid = np.linspace(model.z_min, model.z_max, n_grid)
    pdf = model.pz(z_grid, single_point=False, n_grid=n_grid)
    return _inverse_sample_from_pdf(z_grid, pdf, size=n, rng=rng)



def _sample_x3_given_z(z, rng=None, n_grid=2048):
    if rng is None:
        rng = np.random.default_rng()
    x_grid = np.linspace(model.x3_min, model.x3_max, n_grid)
    out = np.empty_like(z, dtype=float)
    for i, zi in enumerate(np.atleast_1d(z)):
        pdf = model.px3z(x_grid, zi, single_point=False, n_grid=n_grid)
        out[i] = _inverse_sample_from_pdf(x_grid, pdf, size=1, rng=rng)[0]
    return out


def _sample_y3_uniform(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(model.y3_min, model.y3_max, size=n)


def _sample_x4_given_z(z, rng=None, n_grid=2048):
    if rng is None:
        rng = np.random.default_rng()
    x_grid = np.linspace(model.x4_min, model.x4_max, n_grid)
    out = np.empty_like(z, dtype=float)
    for i, zi in enumerate(np.atleast_1d(z)):
        pdf = model.px4z(x_grid, zi, single_point=False, n_grid=n_grid)
        out[i] = _inverse_sample_from_pdf(x_grid, pdf, size=1, rng=rng)[0]
    return out



def _sample_y4_given_x4(x4, rng=None, n_grid=2048):
    if rng is None:
        rng = np.random.default_rng()
    y_grid = np.linspace(model.y4_min, model.y4_max, n_grid)
    out = np.empty_like(x4, dtype=float)
    for i, x in enumerate(np.atleast_1d(x4)):
        pdf = model.py4x4(y_grid, x, single_point=True, n_grid=n_grid)
        out[i] = _inverse_sample_from_pdf(y_grid, pdf, size=1, rng=rng)[0]
    return out


def _shear_amp_pa_to_gamma(amp, pa_deg):
    """
    Convert external shear (amplitude, position angle in degrees) → (gamma1, gamma2).
    Convention: position angle measured counterclockwise from +x to +y.
    """
    phi = np.deg2rad(pa_deg)
    gamma1 = amp * np.cos(2.0 * phi)
    gamma2 = amp * np.sin(2.0 * phi)
    return float(gamma1), float(gamma2)


def _q_phi0_to_e1e2(q, phi0_deg):
    """
    Convert (q, phi0 in degrees) → (e1, e2) using lenstronomy ellipticity convention.
    """
    phi0 = np.deg2rad(phi0_deg)
    e1, e2 = phi_q2_ellipticity(phi0, q)
    return float(e1), float(e2)

def _get_names_safe(data_or_row):
    # Compatibility helper for FITS recarray and Table.Row
    if hasattr(data_or_row, "names"):
        return list(data_or_row.names)
    if hasattr(data_or_row, "array") and hasattr(data_or_row.array, "names"):
        return list(data_or_row.array.names)
    if hasattr(data_or_row, "dtype") and data_or_row.dtype.names is not None:
        return list(data_or_row.dtype.names)
    return []



from tqdm import tqdm

def generate_mock_kwargs_from_fits(
    data,
    n_samples_per_row=1,
    seed=None,
    row_indices=None
):
    
    rng = np.random.default_rng(seed)
    out = []

    if row_indices is None:
        row_indices = np.arange(len(data))

    names = _get_names_safe(data)
    have_amp = 'amp_shear' in names
    have_pa  = 'pa_shear' in names
    have_qsie = 'q_SIE' in names

    for i in tqdm(row_indices, desc="Generating mock kwargs"):
        row = data[i]

        # External shear
        if have_amp and have_pa:
            gamma1, gamma2 = _shear_amp_pa_to_gamma(float(row['amp_shear']), float(row['pa_shear']))
        else:
            gamma1, gamma2 = 0.0, 0.0

        # Use q from the row; otherwise sample from P(q) and share with e and multipole
        if have_qsie:
            q_use = float(row['q_SIE'])
        else:
            q_use = float(_sample_z(1, rng=rng)[0])

        # Match e1, e2 to q_use
        phi_rad = np.deg2rad(0)
        e1_sie = (1 - q_use) / (1 + q_use) * np.cos(2 * phi_rad)
        e2_sie = (1 - q_use) / (1 + q_use) * np.sin(2 * phi_rad)

        # Multipole parameters sampled conditional on q_use
        mp = sample_multipole_params_given_q(np.full(n_samples_per_row, q_use), rng=rng)

        for j in range(n_samples_per_row):
            kwargs_lens0 = {
                "e1": float(e1_sie),
                "e2": float(e2_sie),
                "a3_a": float(mp["a3_a"][j]),
                "delta_phi_m3": float(mp["delta_phi_m3"][j]),
                "a4_a": float(mp["a4_a"][j]),
                "delta_phi_m4": float(mp["delta_phi_m4"][j]),
                "gamma1": float(gamma1),
                "gamma2": float(gamma2),
            }

            # recarray compatibility; row.array.names may be missing
            row_names = _get_names_safe(row)
            meta = {
                "lens_redshift": float(row['lens_redshift']) if 'lens_redshift' in row_names else np.nan,
                "source_redshift": float(row['source_redshift']) if 'source_redshift' in row_names else np.nan,
                "image_sep": float(row['image_sep']) if 'image_sep' in row_names else np.nan,
                "q_from_fits": bool(have_qsie),
                "q_used": float(q_use),
                "log_joint": float(mp["log_joint"][j]),
                # Keep original row to allow writing full columns later
                "orig_row": row
            }
            out.append(([kwargs_lens0], meta))

    return out


# --------- Helper: retrieve column names from FITS Row / Table Row ---------
def _get_colnames(row):
    if hasattr(row, "colnames"):                 # astropy.table.Row
        return list(row.colnames)
    if hasattr(row, "array") and hasattr(row.array, "names"):  # FITS recarray Row
        return list(row.array.names)
    if hasattr(row, "dtype") and row.dtype.names is not None:   # numpy.void with named dtype
        return list(row.dtype.names)
    raise AttributeError("Cannot determine column names from row object.")


def multipole_logL(**kwargs):
    """
    Evaluate log-prior + log-likelihood (from the joint prior model) for one sample.
    Expects kwargs['kwargs_lens'][0] with keys:
        e1, e2, a3_a, delta_phi_m3, a4_a, delta_phi_m4
    """
    lp = log_prior_q_phi(kwargs['kwargs_lens'][0]['e1'],
                         kwargs['kwargs_lens'][0]['e2'])
    L2 = model.prob_single_point(
        x3=kwargs['kwargs_lens'][0]['a3_a'],
        y3=kwargs['kwargs_lens'][0]['delta_phi_m3'],
        x4=kwargs['kwargs_lens'][0]['a4_a'],
        y4=kwargs['kwargs_lens'][0]['delta_phi_m4'],
        z=ellipticity2phi_q(kwargs['kwargs_lens'][0]['e1'],
                            kwargs['kwargs_lens'][0]['e2'])[1]
    )
    if L2 <= 0.0:
        return lp + -np.inf
    return lp + np.log(L2)



import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========= 1) Conditional sampling given q =========
def sample_multipole_params_given_q(q_array, rng=None, n_grid=2048):
    """
    Sample multipole parameters conditioned on z=q in line with the model;
    returns dict: a3_a, delta_phi_m3, a4_a, delta_phi_m4, log_joint
    """
    if rng is None:
        rng = np.random.default_rng()
    q_array = np.atleast_1d(q_array).astype(float)
    n = q_array.size

    # Use the existing 1D sampler at fixed z
    x3 = _sample_x3_given_z(q_array, rng=rng, n_grid=n_grid)
    y3 = _sample_y3_uniform(n, rng=rng)
    x4 = _sample_x4_given_z(q_array, rng=rng, n_grid=n_grid)
    y4 = _sample_y4_given_x4(x4, rng=rng, n_grid=n_grid)

    # Compute joint log-probability for debugging/filtering
    L2 = np.array([model.prob_single_point(x3[i], y3[i], x4[i], y4[i], q_array[i]) for i in range(n)], dtype=float)
    log_joint = np.where(L2 > 0.0, np.log(L2), -np.inf)

    return {
        "a3_a": x3,
        "delta_phi_m3": y3,
        "a4_a": x4,
        "delta_phi_m4": y4,
        "log_joint": log_joint,
    }

# ========= 2) Conditional multipole sampling on q for selected cusp_data entries =========
def generate_mock_kwargs_from_cusp_condq(cusp_data, n_samples_per_row=1, seed=None):
    """
    Similar to generate_mock_kwargs_from_cusp, but multipole parameters are conditioned on each row's q.
    Stores 'orig_row' in meta to enable writing complete FITS columns later.
    """
    rng = np.random.default_rng(seed)
    out = []
    if len(cusp_data) == 0:
        return out

    # Column name adaptation
    def _get_colnames(row):
        if hasattr(row, "colnames"):
            return list(row.colnames)
        if hasattr(row, "array") and hasattr(row.array, "names"):
            return list(row.array.names)
        if hasattr(row, "dtype") and row.dtype.names is not None:
            return list(row.dtype.names)
        raise AttributeError("Cannot determine column names from row object.")
    names = _get_colnames(cusp_data[0]["data"])

    have_amp  = "amp_shear" in names
    have_pa   = "pa_shear" in names
    have_qsie = "q_SIE"    in names

    for item in tqdm(cusp_data, desc="Generating kwargs (conditional on row q)"):
        row = item["data"]

        # External shear
        if have_amp and have_pa:
            gamma1, gamma2 = _shear_amp_pa_to_gamma(float(row["amp_shear"]), float(row["pa_shear"]))
        else:
            gamma1, gamma2 = 0.0, 0.0

        # Use the row q; otherwise sample from P(q) and share with e and multipole
        if have_qsie:
            q_use = float(row["q_SIE"])
        else:
            q_use = float(_sample_z(1, rng=rng)[0])

        # e1, e2 share the same q; use phi0=0.0 deg to match later conversion
        e1_sie, e2_sie = _q_phi0_to_e1e2(q_use, phi0_deg=0.0)

        # Multipole parameters are conditioned on the same q_use
        mp = sample_multipole_params_given_q(np.full(n_samples_per_row, q_use), rng=rng)

        for j in range(n_samples_per_row):
            kwargs_lens0 = {
                "e1": float(e1_sie),
                "e2": float(e2_sie),
                "a3_a": float(mp["a3_a"][j]),
                "delta_phi_m3": float(mp["delta_phi_m3"][j]),
                "a4_a": float(mp["a4_a"][j]),
                "delta_phi_m4": float(mp["delta_phi_m4"][j]),
                "gamma1": float(gamma1),
                "gamma2": float(gamma2),
            }
            meta = {
                "lens_redshift": float(row["lens_redshift"]) if "lens_redshift" in names else np.nan,
                "source_redshift": float(row["source_redshift"]) if "source_redshift" in names else np.nan,
                "image_sep": float(row["image_sep"]) if "image_sep" in names else np.nan,
                "q_from_fits": bool(have_qsie),
                "q_used": q_use,
                "log_joint": float(mp["log_joint"][j]),
                "R_cusp": float(item.get("R_cusp", np.nan)),
                "phi": float(item.get("phi", np.nan)),
                "phi1": float(item.get("phi1", np.nan)),
                "phi2": float(item.get("phi2", np.nan)),
                # Keep the original row for writing full FITS columns later
                "orig_row": row
            }
            out.append(([kwargs_lens0], meta))
    return out


def _is_scalar_like(val):
    """
    Keep only scalar/0D (or safely scalarizable) columns to avoid length mismatches.
    """
    # astropy Masked / numpy scalar
    if np.isscalar(val):
        return True
    # numpy 0-d array
    if isinstance(val, np.ndarray) and val.shape == ():
        return True
    # Common scalar types
    if isinstance(val, (int, float, bool, np.number, np.bool_)):
        return True
    # bytes/str count as single values
    if isinstance(val, (str, bytes, np.bytes_)):
        return True
    return False

def _safe_scalar(val):
    """
    Convert scalar-like content to plain Python scalars to avoid astropy splitting object columns.
    """
    if np.isscalar(val):
        return val.item() if hasattr(val, "item") else val
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    return val

from astropy.table import Table

def save_samples_to_fits(samples, file_path):
    """
    Note: this stores EPL_MULTIPOLE_M3M4 parameters from Lenstronomy; results use units: 2404.17124
    """
    if len(samples) == 0:
        Table().write(file_path, overwrite=True)
        return

    first_meta = samples[0][1]
    orig_row = first_meta.get("orig_row", None)
    if orig_row is None:
        raise ValueError("samples is missing 'orig_row'; generate samples with generate_mock_kwargs_from_cusp_condq.")

    # Candidate original columns; drop non-scalar columns
    if hasattr(orig_row, "colnames"):
        candidate_cols = list(orig_row.colnames)
    elif hasattr(orig_row, "array") and hasattr(orig_row.array, "names"):
        candidate_cols = list(orig_row.array.names)
    elif hasattr(orig_row, "dtype") and orig_row.dtype.names is not None:
        candidate_cols = list(orig_row.dtype.names)
    else:
        raise AttributeError("Cannot determine column names from orig_row.")

    base_cols, dropped_cols = [], []
    for c in candidate_cols:
        if _is_scalar_like(orig_row[c]):
            base_cols.append(c)
        else:
            dropped_cols.append(c)

    # Remove specific original columns
    drop_orig_cols = {
        'apparent_mag_i_band', 'image_sep', 'apparent_mag_first_arrival_i_band',
        'image_number', 'Rcusp', 'phi', 'phi1', 'phi2',
        'imageA_x', 'imageA_y', 'imageB_x', 'imageB_y',
        'imageC_x', 'imageC_y', 'imageD_x', 'imageD_y',
        'cusp_id'
    }
    explicit_dropped = [c for c in base_cols if c in drop_orig_cols]
    if explicit_dropped:
        base_cols = [c for c in base_cols if c not in drop_orig_cols]
        dropped_cols.extend(explicit_dropped)

    add_cols = [
        "a3_over_a_signed", "delta_phi_m3",
        "a4_over_a_signed", "delta_phi_m4",
        "log_joint"
    ]

    all_cols = base_cols + add_cols

    col_data = {c: [] for c in all_cols}

    sid = 0
    for ([kw], meta) in samples:
        sid += 1
        row = meta["orig_row"]

        # Original scalar columns
        for c in base_cols:
            col_data[c].append(_safe_scalar(row[c]))

        # Additional fields
        e1 = float(kw["e1"]); e2 = float(kw["e2"])
        gamma1 = float(kw["gamma1"]); gamma2 = float(kw["gamma2"])

        a3_signed = float(kw["a3_a"])
        dphi3 = float(kw["delta_phi_m3"])
        a4_signed = float(kw["a4_a"])
        dphi4 = float(kw["delta_phi_m4"])


        col_data["a3_over_a_signed"].append(a3_signed)
        col_data["delta_phi_m3"].append(dphi3)
        col_data["a4_over_a_signed"].append(a4_signed)
        col_data["delta_phi_m4"].append(dphi4)
        col_data["log_joint"].append(float(meta.get("log_joint", np.nan)))

    # Consistency check
    lengths = {len(v) for v in col_data.values()}
    if len(lengths) != 1:
        raise ValueError(f"Column lengths still inconsistent: { {k: len(v) for k,v in col_data.items()} }")

    Table(col_data).write(file_path, overwrite=True)
    if len(dropped_cols) > 0:
        print(f"[save_samples_to_fits] Dropped columns: {dropped_cols}")
    print(f"Wrote {len(next(iter(col_data.values())))} samples to {file_path} (readable via hdul[1].data)")
