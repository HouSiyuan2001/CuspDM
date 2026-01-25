import sys
sys.path.append("../lib")
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from fit_position import *
from Lensing_tool import *
from lenstronomy.LensModel.lens_model import LensModel

def fits_row_to_obj(row, nnn=256):
    """
    Convert one FITS row (astropy.io.fits.fitsrec.FITS_record) to the obj dict used in __init__.
    """
    obj = {
        "zlens": row["lens_redshift"],
        "zsource": row["source_redshift"],
        "v_disp": row["v_disp"],
        "q": row["q_SIE"],
        "lambda_q": row["lambda_q"],


        # multipole m=3
        "a3_over_a_signed": row["a3_over_a_signed"],
        "delta_phi_m3": row["delta_phi_m3"],

        # multipole m=4
        "a4_over_a_signed": row["a4_over_a_signed"],
        "delta_phi_m4": row["delta_phi_m4"],
        
        # Extra parameters carried directly from FITS
        "gamma_external": row["amp_shear"],
        "phi_external": row["pa_shear"],
        "source_xlocation": row["source_xlocation"],
        "source_ylocation": row["source_ylocation"],
        "absolute_mag_i_band_ab": row["absolute_mag_i_band_ab"],
        "log_joint": row["log_joint"],

        # Additional parameters
        "nnn": nnn,
        "gamma_slope": 2,
        "center_x": 0,
        "center_y": 0
        
    }
    return obj
def to_elliptical_params(m, a_signed, dphi, phi0_abs,rescale_am):
    a_m = abs(a_signed)*rescale_am
    phi_m = phi0_abs + dphi
    if a_signed < 0:
        phi_m += np.pi / m
    return a_m, phi_m
def _shear_amp_pa_to_gamma(amp, pa_deg):
    """
    Convert external shear (amplitude, position angle in deg) to (gamma1, gamma2).
    Convention: pa is CCW from +x to +y in degrees.
    """
    phi = np.deg2rad(pa_deg)
    gamma1 = amp * np.cos(2.0 * phi)
    gamma2 = amp * np.sin(2.0 * phi)
    return float(gamma1), float(gamma2)

class Simulated_Lensing_with_multipole:
    """ 
    Recompute lensing with multipole terms and output results for a single system.
    """
    def __init__(self,obj):

        self.obj = obj
        self.cosmo_astropy = FlatwCDM(H0=70, Om0=0.3, Ob0=0.05)
        # self.cosmo_astropy = FlatLambdaCDM(H0=72,Om0=0.26,Tcmb0=2.725)
        self.cosmo = cosmology.setCosmology('planck18')
        self.zlens = self.obj["zlens"]
        self.zsource = self.obj["zsource"]
        self.Da_lens = Da0(self.zlens)
        self.Da_src = Da0(self.zsource)
        self.Da_ls = Da20(self.zlens, self.zsource)
        self.arcsec_1 = 1/apr * self.Da_lens #mpc/h
        self.thetaE_rad =  calculate_theta_E(self.obj["v_disp"], self.Da_src, self.Da_ls)
        self.obj["thetaE"] = Rad_to_arcsec(self.thetaE_rad)
        self.thetaE =  self.obj["thetaE"]* self.obj.get("lambda_q", 1)
        self.thetaE_mpc_h = self.thetaE* self.arcsec_1  # mpc/h
        self.sigma_g = calculate_velocity_dispersion(self.thetaE_rad, self.Da_src, self.Da_ls)


        self.halomass = Sigma_g_to_Mvir(self.thetaE_mpc_h,self.sigma_g,self.zlens)/self.cosmo.h #M_sun
        self.obj["halomass"] = self.halomass *self.cosmo.h  #M_sun/h
        self.sigma_crit = SigmaCrit(self.zlens,self.zsource)
        self.rescale_am = self.thetaE / np.sqrt(self.obj["q"])
        self.bsz_arc = 3.0*self.rescale_am
        self.nnn = self.obj["nnn"]
        self.dsx_arc = self.bsz_arc / self.nnn

        self.Get_mainhalo_Mock()


    def Get_mainhalo_Mock_mulell(self):
        obj = self.obj

        # --- Grid ---
        nnn = obj["nnn"]
        bsz_arc = self.bsz_arc
        self.xi2, self.xi1 = make_c_coor(bsz_arc, nnn)  # (y,x) convention
        phi_rad = np.deg2rad(0)
        a3_m, phi3_m = to_elliptical_params(3, obj["a3_over_a_signed"], obj["delta_phi_m3"], phi_rad,self.rescale_am)
        a4_m, phi4_m = to_elliptical_params(4, obj["a4_over_a_signed"], obj["delta_phi_m4"], phi_rad,self.rescale_am)
        
        q_use = self.obj["q"]
        e1_sie = (1 - q_use) / (1 + q_use) * np.cos(2 * phi_rad)
        e2_sie = (1 - q_use) / (1 + q_use) * np.sin(2 * phi_rad)
        gamma1, gamma2 = _shear_amp_pa_to_gamma(obj['gamma_external'], obj['phi_external'])




        # --- Lens model: EPL + SHEAR + MULTIPOLE_ELL (m=3,m=4) ---
        kwargs_epl = {
            "theta_E": self.thetaE,
            "e1": e1_sie,
            "e2": e2_sie,
            "gamma": float(obj["gamma_slope"]),   # 2.0 → SIE
            "center_x": float(obj["center_x"]),
            "center_y": float(obj["center_y"]),
        }
        kwargs_shear = {
            "gamma1": gamma1,
            "gamma2": gamma2,
        }
        kwargs_m3 = {
            "m": 3,
            "a_m": a3_m,
            "phi_m": phi3_m,
            "q": float(obj["q"]),
            "center_x": float(obj["center_x"]),
            "center_y": float(obj["center_y"]),
            "r_E": self.thetaE,
        }
        kwargs_m4 = {
            "m": 4,
            "a_m": a4_m,
            "phi_m": phi4_m,
            "q": float(obj["q"]),
            "center_x": float(obj["center_x"]),
            "center_y": float(obj["center_y"]),
            "r_E": self.thetaE,
        }

        lens_model_list = ["EPL", "SHEAR", "MULTIPOLE_ELL", "MULTIPOLE_ELL"]
        lensModel = LensModel(lens_model_list=lens_model_list)

        self.psi_map = np.asarray(
            lensModel.potential(
                self.xi2.ravel(), self.xi1.ravel(),
                kwargs=[kwargs_epl, kwargs_shear, kwargs_m3, kwargs_m4]
            )
        ).reshape(nnn, nnn)

        # --- Compute deflection ---
        self.alpha1_global, self.alpha2_global = potential_to_alphas(self.psi_map, self.dsx_arc)

        self.yi1, self.yi2, mu_map, kappa_map, gamma1_map, gamma2_map = alphas_to_mu(
            self.alpha1_global, self.alpha2_global, self.dsx_arc, self.xi1, self.xi2
        )
        timedelay_map = timedelay(self.psi_map, self.alpha1_global, self.alpha2_global,self.zlens, self.zsource)

        self.maps_dict = {
            "kappa": kappa_map,
            "gamma1": gamma1_map,
            "gamma2": gamma2_map,
            "magnification": mu_map,
            "timedelay": timedelay_map
        }
    def Get_mainhalo_Mock(self):
        from cal_mul_fits import to_elliptical_params,_shear_amp_pa_to_gamma
        obj = self.obj
        # print(obj)


        # --- Grid ---
        nnn = obj["nnn"]
        bsz_arc = self.bsz_arc
        self.xi2, self.xi1 = make_c_coor(bsz_arc, nnn)  # (y,x) convention
        phi_rad = np.deg2rad(0)

        
        q_use = self.obj["q"]
        e1_sie = (1 - q_use) / (1 + q_use) * np.cos(2 * phi_rad)
        e2_sie = (1 - q_use) / (1 + q_use) * np.sin(2 * phi_rad)
        gamma1, gamma2 = _shear_amp_pa_to_gamma(obj['gamma_external'], obj['phi_external'])


        # --- Lens model: SHEAR + EPL_MULTIPOLE_M3M4 (m=3,m=4) ---

        kwargs_shear = {
            "gamma1": gamma1,
            "gamma2": gamma2,
        }
        kwargs_epl34 = {
            'theta_E': self.thetaE,
            'e1': e1_sie, 'e2': e2_sie,
            'gamma': 2.0,
            'center_x': 0.0, 'center_y': 0.0,
            'a3_a': obj["a3_over_a_signed"] , 'delta_phi_m3': obj["delta_phi_m3"],
            'a4_a': obj["a4_over_a_signed"], 'delta_phi_m4': obj["delta_phi_m4"],
        }

        lens_model_list = ["EPL_MULTIPOLE_M3M4","SHEAR"]
        lensModel = LensModel(lens_model_list=lens_model_list)

        self.psi_map = np.asarray(
            lensModel.potential(
                self.xi2.ravel(), self.xi1.ravel(),
                kwargs=[kwargs_epl34 , kwargs_shear]
            )
        ).reshape(nnn, nnn)

        # --- Compute deflection ---
        self.alpha1_global, self.alpha2_global = potential_to_alphas(self.psi_map, self.dsx_arc)

        self.yi1, self.yi2, mu_map, kappa_map, gamma1_map, gamma2_map = alphas_to_mu(
            self.alpha1_global, self.alpha2_global, self.dsx_arc, self.xi1, self.xi2
        )
        timedelay_map = timedelay(self.psi_map, self.alpha1_global, self.alpha2_global,self.zlens, self.zsource)

        self.maps_dict = {
            "kappa": kappa_map,
            "gamma1": gamma1_map,
            "gamma2": gamma2_map,
            "magnification": mu_map,
            "timedelay": timedelay_map
        }
    def Get_mainhalo_Mock_SIE(self,ql = None, pa = None, g_external = None,phi_external = None):
        if ql == None:
            ql = self.obj["q"]
            pa = 0
        xi2, xi1 = make_c_coor(self.bsz_arc, self.nnn)
        pa = np.deg2rad(pa)

        pa_use = 0
        

        sie = SIE_Model(self.thetaE, ql = ql,pa = np.deg2rad(pa_use))

        # Potentian0 = sie.potential(xi1, xi2)


        alpha1_SIE,alpha2_SIE = sie.deflection_angle(xi1, xi2)


        if g_external == None: 
            g_external = self.obj["gamma_external"]
            phi_external = -np.deg2rad(self.obj["phi_external"]+pa_use-90)

        eg = ext_shear(g=g_external, phi_g=phi_external)
        phi_use = np.arctan2(xi2, xi1)  # angular coordinate
        x = np.sqrt(xi1**2 + xi2**2)    # radial coordinate

        # psi_eg = eg.psi(x, phi_use)
        # Potentian_main = Potentian0+psi_eg
        # alpha1_global, alpha2_global = potential_to_alphas(Potentian_main, self.dsx_arc)
        al1_eg,al2_eg = eg.alpha(x, phi_use)

        alpha1_global = alpha1_SIE+al1_eg
        alpha2_global = alpha2_SIE+al2_eg

        return alpha1_global, alpha2_global
    def cal_each_image(self):
        """
        Returns a dict:
        {
            'images': [ {position, kappa, gamma1, gamma2, magnification, timedelay, apparent_mag_i_band}, ... ],
            'image_sep': float,                           # max pairwise separation
            'apparent_mag_first_arrival_i_band': float,   # apparent mag of earliest arrival
            'image_number': int                           # number of images (len xroots)
        }
        """
        import numpy as np
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u

        ys = np.array([self.obj["source_xlocation"], self.obj["source_ylocation"]])

        xroots_all, mask = mapping_triangles_vec_jax(ys, self.xi1, self.xi2, self.yi1, self.yi2)
        xroots = xroots_all[mask]
        result = interpolate_maps_at_points(xroots, self.maps_dict, self.xi1, self.xi2)

        positions = np.array([r["position"] for r in result], dtype=float) if len(result) else np.empty((0, 2))
        magnifications = np.array([r["magnification"] for r in result], dtype=float) if len(result) else np.empty((0,))
        timedelays = np.array([r["timedelay"] for r in result], dtype=float) if len(result) else np.empty((0,))

        # Unlensed apparent magnitude
        M_abs = float(self.obj["absolute_mag_i_band_ab"])
        zsrc = float(self.obj["zsource"])
        DL_Mpc = cosmo.luminosity_distance(zsrc).to(u.Mpc).value
        DM = 5.0 * np.log10(DL_Mpc) + 25.0
        m_unlensed = M_abs + DM

        # Apparent magnitude for each image after magnification
        eps = 1e-12
        apparent_mags = m_unlensed - 2.5 * np.log10(np.maximum(np.abs(magnifications), eps))

        # Maximum image separation in the system
        if len(positions) >= 2:
            diffs = positions[:, None, :] - positions[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1))
            image_sep = float(np.max(dists))
        else:
            image_sep = 0.0

        # Apparent magnitude of the earliest arrival
        if len(timedelays) > 0:
            first_idx = int(np.argmin(timedelays))
            apparent_mag_first_arrival = float(apparent_mags[first_idx])
        else:
            apparent_mag_first_arrival = float("nan")

        # Write apparent_mag_i_band for each image
        for i, r in enumerate(result):
            r["apparent_mag_i_band"] = float(apparent_mags[i])

        out = {
            "images": result,
            "image_sep": image_sep,
            "apparent_mag_first_arrival_i_band": apparent_mag_first_arrival,
            "image_number": int(len(xroots))
        }


        return out

import time
import json
import numpy as np
from astropy.io import fits

# Assumed helpers available:
# fits_row_to_obj(row, nnn)
# Simulated_Lensing_with_multipole(obj)  # with method sim.cal_each_image()

# ---- Unified JSON serialization hook (handles numpy types)----
def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def _to_serializable_dict(res, prefix="calc"):
    """
    Flatten cal_each_image() output into a dict for saving:
      - scalar numbers -> float
      - 1D numeric arrays/lists -> np.ndarray(float)
      - other structures -> JSON string
    """
    out = {}
    def _coerce(name, val):
        key = f"{prefix}_{name}"
        # Scalars
        if np.isscalar(val) and val is not None:
            try:
                out[key] = float(val)
                return
            except Exception:
                pass
        # 1D numeric arrays
        try:
            arr = np.asarray(val)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                out[key] = arr.astype(float)
                return
        except Exception:
            pass
        # Fallback: JSON string
        out[key] = json.dumps(val, ensure_ascii=False, default=_json_default)

    if isinstance(res, dict):
        for k, v in res.items():
            _coerce(k, v)
    else:
        _coerce("result", res)
    return out

from tqdm import tqdm

def run_cal_each_image_serial(
    fits_path: str,
    nnn: int = 1000,
    limit_rows: int | None = 100,   # e.g. process first 100
    start_idx: int = 0,             # start from middle if needed
    prefix: str = "calc",
):
    """
    Run serially over FITS table (HDU 1) rows starting at start_idx for limit_rows.
    Builds sim per row and calls cal_each_image().
    Returns (indices, results_ser, elapsed_s):
      - indices: processed row indices
      - results_ser: serialized results (list of dict)
    """
    import time
    t0 = time.time()

    with fits.open(fits_path, memmap=True) as hdul:
        table = hdul[1].data
        n_total = len(table)
        if start_idx < 0 or start_idx >= n_total:
            raise IndexError(f"start_idx out of range: {start_idx} (total {n_total} rows)")
        nrows = n_total - start_idx if limit_rows is None else min(limit_rows, n_total - start_idx)

        indices = list(range(start_idx, start_idx + nrows))
        results_ser = []

        # tqdm progress bar
        for idx in tqdm(indices, desc="Processing rows", unit="row"):
            row = table[idx]
            obj = fits_row_to_obj(row, nnn=nnn)
            sim = Simulated_Lensing_with_multipole(obj)
            res = sim.cal_each_image()
            results_ser.append(_to_serializable_dict(res, prefix=prefix))

    elapsed = time.time() - t0
    print(f"[done] processed {nrows} rows | total elapsed {elapsed:.1f}s")
    return indices, results_ser, elapsed

def save_results_json(
    indices: list[int],
    results_ser: list[dict],
    out_json_path: str,
    extra_meta: dict | None = None,
    ensure_ascii: bool = False,
    indent: int | None = 2,
):
    """
    Save results as a single JSON file without modifying the FITS.
    File structure:
    {
      "meta": {...},
      "data": [
        {"idx": 0, "result": {...}},
        {"idx": 1, "result": {...}},
        ...
      ]
    }
    """
    import os, json

    meta = extra_meta or {}
    data = [{"idx": int(idx), "result": ser}
            for idx, ser in zip(indices, results_ser)]
    payload = {"meta": meta, "data": data}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=ensure_ascii, indent=indent, default=_json_default)

    print(f"[done] JSON written to {out_json_path} | rows={len(indices)}")
# ==== choose a free GPU BEFORE importing jax/torch/etc. ====
import os, subprocess

def find_free_gpu():
    try:
        # Query GPU memory usage and totals
        mem_output = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits",
            shell=True
        )
        mem_info = [line.split(",") for line in mem_output.decode().splitlines()]
        gpu_ids = [int(x[0].strip()) for x in mem_info]
        used = [int(x[1].strip()) for x in mem_info]
        total = [int(x[2].strip()) for x in mem_info]
        mem_ratio = [u / t for u, t in zip(used, total)]

        print("=== GPU memory usage ===")
        for gid, u, t, r in zip(gpu_ids, used, total, mem_ratio):
            print(f"GPU {gid}: {u}/{t} MiB ({r:.1%})")

        # If all GPUs are >30% memory usage, pick the one using the least
        if all(r > 0.3 for r in mem_ratio):
            min_mem_gpu = gpu_ids[used.index(min(used))]
            print(f"All GPUs >30% memory usage, choosing GPU with least memory used: {min_mem_gpu}")
            return str(min_mem_gpu)

        # Otherwise pick GPU with lowest utilization
        util_output = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            shell=True
        )
        utils = [int(x.strip()) for x in util_output.decode().splitlines()]
        min_util_gpu = gpu_ids[utils.index(min(utils))]

        print("=== GPU utilization ===")
        for gid, u in zip(gpu_ids, utils):
            print(f"GPU {gid}: {u}%")

        print(f"Selected GPU with lowest utilization: {min_util_gpu}")
        return str(min_util_gpu)

    except Exception as e:
        print("GPU query failed:", e)
        return "0"
# =========================
# Usage example (serial, save as JSONL only, no FITS writeback)
# =========================
import os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-idx", type=int, required=True)
    parser.add_argument("--num-sim", type=int, default=10000)
    parser.add_argument("--fix", type=float, default=2.5)
    parser.add_argument("--fits", type=str, default="Theory_Mock/lensed_qso_mock_multipole_temp.fits")
    parser.add_argument("--nnn", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default=None)  # Optional: explicitly choose GPU
    args = parser.parse_args()

    # Prefer explicit --gpu, otherwise use find_free_gpu()
    gpu = args.gpu if args.gpu is not None else find_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    print(f"⚙️ Using GPU for JAX: {gpu}")

    fits_path = args.fits
    num_sim   = args.num_sim
    fix       = args.fix
    sim_idx   = args.sim_idx  # Note: hyphen becomes underscore

    limit_rows = int(fix * num_sim)
    start_idx = int(sim_idx * num_sim * fix)
    end_idx   = start_idx + limit_rows - 1

    print(f"start:{start_idx}, end:{end_idx}")

    indices, results_ser, elapsed = run_cal_each_image_serial(
        fits_path=fits_path,
        nnn=1000,
        limit_rows=limit_rows,
        start_idx=start_idx,
        prefix="calc",
    )

    out_json_path = f"Theory_Mock/Data_json_cir/calc_results_{start_idx}_{end_idx}.json"

    save_results_json(
        indices=indices,
        results_ser=results_ser,
        out_json_path=out_json_path,
        extra_meta={
            "fits": fits_path,
            "nnn": 1000,
            "prefix": "calc",
            "start_idx": start_idx,
            "end_idx": end_idx,
        },
        ensure_ascii=False,
        indent=2,
    )
    print(f"[total] elapsed {elapsed:.1f}s")
