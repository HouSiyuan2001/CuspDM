import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import subprocess
import numpy as np
import pickle
import jax
import jax.numpy as jnp

from jax import device_put

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from notion_client import Client
from tqdm import tqdm
from colossus.cosmology import cosmology
from lenstronomy.Cosmo.nfw_param import NFWParam
from astropy.cosmology import FlatwCDM,FlatLambdaCDM

sys.path.append("../lib")
from Sidm_tool import get_taub
from Lensing_tool import *
from notion import *
from SIDM_Parametric_Model_jax import SIDM_parametric_Multiplane, multiLensPlaneLensingSimsWithPICS_JAX
from pyHalo.PresetModels.cdm import CDM
from pyHalo.PresetModels.wdm import WDM

from pyHalo.PresetModels.uldm import ULDM

from copy import deepcopy
import matplotlib.pyplot as plt

import numpy as np
from FDM import CNFW_parametric_Multiplane, ULDM_parametric_Multiplane, GAUSSIAN_parametric_Multiplane,compute_delta_kappa_rms, compute_eps2_dtheta2,ULDM_r
import gc

from lenstronomy.LensModel.Profiles.cnfw import CNFW
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.uldm import Uldm
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

import os
import shutil
from skimage import measure
from MCMC import *
import math

apr = 1.0 / np.pi * 180.0 * 3600.0
cosmo = FlatwCDM(H0=70.0, Om0=0.3, Ob0=0.05, w0=-1.000000)

def round_nonzero(x, sig=1):
    """Keep sig significant digits while avoiding rounding tiny values to zero."""
    if x == 0:
        return 0.0
    # Keep significant digits
    rounded = round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
    # Ensure very small values do not round to zero
    if rounded == 0:
        rounded = math.copysign(10**(math.floor(math.log10(abs(x)))), x)
    return rounded
# cosmo=FlatLambdaCDM(H0=72,Om0=0.26,Tcmb0=2.725)
# Ensure directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Get_multi_alpha(obj,bsz, nnn, filename_sub, redshift_list_file, z_main, zsource, filename_main=None, output_file=None):
    data = jnp.load(filename_sub)
    alpha1_sub = data['alpha1_sub']
    alpha2_sub = data['alpha2_sub']

    alpha1_sub = jnp.array(alpha1_sub)
    alpha2_sub = jnp.array(alpha2_sub)
    redshift_list = np.load(redshift_list_file)

    if filename_main is not None:
        main = np.load(filename_main)
        alpha1_main = main['alpha1_global']
        alpha2_main = main['alpha2_global']

        xi2, xi1 = make_c_coor(bsz, nnn)
        for j in range(len(alpha1_sub)):
            kappa_this = get_eff_kappa(alpha1_sub[j], alpha2_sub[j], bsz / nnn)
            mean_k = jnp.mean(kappa_this)
            a1, a2 = adding_external_to_alpha(alpha1_sub[j], alpha2_sub[j], xi1, xi2, -mean_k)
            alpha1_sub = alpha1_sub.at[j].set(a1)
            alpha2_sub = alpha2_sub.at[j].set(a2)

        idx_main = int(np.argmin(np.abs(redshift_list - z_main)))

        alpha1_sub = alpha1_sub.at[idx_main].set(alpha1_sub[idx_main] + alpha1_main)
        alpha2_sub = alpha2_sub.at[idx_main].set(alpha2_sub[idx_main] + alpha2_main)

    xi2, xi1 = make_c_coor(bsz, nnn)
    af1, af2 = multiLensPlaneLensingSimsWithPICS_JAX(
        xi1, xi2,
        alpha1_sub, alpha2_sub,
        redshift_list,
        zs_srcs_ref=zsource,
        zs_srcs=zsource
    )

    if output_file is not None:
        np.savez(output_file, alpha1_global=af1, alpha2_global=af2)
        print(f"Data saved to '{output_file}'.")
    else:
        return af1, af2 



def merge_FDM_alpha_maps_jax(analyzer_name, filename, suffix: str = None):
    base_path = filename
    models = ['gaussian', 'uldm', 'cnfw']

    # Handle filename suffix
    name_suffix = f"{analyzer_name}" if suffix is None else f"{analyzer_name}_{suffix}"

    # Collect all data
    all_alpha1, all_alpha2, all_redshifts = [], [], []

    for model in models:
        alpha_file = os.path.join(base_path, f"{name_suffix}_{model}_alpha_mul.npz")
        redshift_file = os.path.join(base_path, f"{name_suffix}_{model}_redshift_list.npy")

        data = np.load(alpha_file)
        redshifts = np.load(redshift_file)

        all_alpha1.append(data['alpha1_sub'])
        all_alpha2.append(data['alpha2_sub'])
        all_redshifts.append(redshifts)

    alpha1_all = device_put(jnp.concatenate(all_alpha1, axis=0))
    alpha2_all = device_put(jnp.concatenate(all_alpha2, axis=0))
    redshift_all = np.concatenate(all_redshifts, axis=0)  # Grouping handled on CPU

    # Get unique redshifts and build mapping
    unique_redshifts = np.unique(redshift_all)
    alpha1_merged = []
    alpha2_merged = []

    for z in unique_redshifts:
        idx = np.where(redshift_all == z)[0]
        idx = device_put(jnp.array(idx))  # Convert indices to JAX array
        summed_alpha1 = jnp.sum(alpha1_all[idx], axis=0)
        summed_alpha2 = jnp.sum(alpha2_all[idx], axis=0)
        alpha1_merged.append(summed_alpha1)
        alpha2_merged.append(summed_alpha2)

    alpha1_merged = jnp.stack(alpha1_merged)
    alpha2_merged = jnp.stack(alpha2_merged)

    # Save to numpy files (suffix appended after FDM)
    if suffix is None:
        alpha_outfile = os.path.join(base_path, f"{analyzer_name}_FDM_alpha_mul.npz")
        redshift_outfile = os.path.join(base_path, f"{analyzer_name}_FDM_redshift_list.npy")
    else:
        alpha_outfile = os.path.join(base_path, f"{analyzer_name}_FDM_{suffix}_alpha_mul.npz")
        redshift_outfile = os.path.join(base_path, f"{analyzer_name}_FDM_{suffix}_redshift_list.npy")

    np.savez(alpha_outfile,
             alpha1_sub=np.array(alpha1_merged), alpha2_sub=np.array(alpha2_merged))
    np.save(redshift_outfile, unique_redshifts)

    print(f"[JAX] Merged alpha maps saved to {alpha_outfile}")
    print(f"[JAX] Merged redshift list saved to {redshift_outfile}") 


# Replace with your Notion token and database ID
NOTION_AUTH = "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"
DATABASE_ID = "1f8fc9067a748057a9ebf052fd25a8bd"

class HaloAnalyzer:
    def __init__(self, name = None,obj = None,  sigma0=8000, w0=6, savefile = "Test_Data", globalfile= None, IsMock = True):
        self.name = name
        self.sigma0 = sigma0
        self.w0 = w0
        

        if IsMock: 
            self.obj =obj
        else: 
            self.notion = Client(auth=NOTION_AUTH)
            response = self.notion.databases.query(database_id=DATABASE_ID, page_size=100)
            self.obj = get_lens_data_by_name(response, name)



        self.cosmo_astropy = FlatwCDM(H0=70, Om0=0.3, Ob0=0.05)
        # self.cosmo_astropy = FlatLambdaCDM(H0=72,Om0=0.26,Tcmb0=2.725)
        self.cosmo = cosmology.setCosmology('planck18')
        self.zlens =  round_nonzero(self.obj["zlens"],2)

        self.zsource = round_nonzero(self.obj["zsource"],2)
        self.Da_lens = Da0(self.zlens)
        self.Da_src = Da0(self.zsource)
        self.Da_ls = Da20(self.zlens, self.zsource)


        self.arcsec_1 = 1/apr * self.Da_lens #mpc/h
        self.arcsec_s = 1/apr * self.Da_src #mpc/h
        if self.obj["thetaE"] is not None: 
            self.thetaE = self.obj["thetaE"]
            self.thetaE_rad = arcsec_to_Rad(self.thetaE)
        else: 
            self.thetaE_rad =  calculate_theta_E(self.obj["v_disp"], self.Da_src, self.Da_ls)
            self.obj["thetaE"] = Rad_to_arcsec(self.thetaE_rad)
            self.thetaE =  self.obj["thetaE"]* self.obj.get("lambda_q")

        
        self.thetaE_mpc_h = self.thetaE* self.arcsec_1  # mpc/h
        self.sigma_g = calculate_velocity_dispersion(self.thetaE_rad, self.Da_src, self.Da_ls)

        self.halomass = Sigma_g_to_Mvir(self.thetaE_mpc_h,self.sigma_g,self.zlens)/self.cosmo.h   #M_sun
        self.obj["halomass"] = self.halomass *self.cosmo.h  #M_sun/h
        print(f"Main halo mass: {self.halomass:.2e} M_sun")
        self.sigma_crit = SigmaCrit(self.zlens,self.zsource)
        nfwpara = NFWParam(cosmo)
        self.r200, self.rhos, self.Concentration, self.rs = nfwpara.nfw_Mz(self.halomass*self.cosmo.h, self.zlens)

        self.bsz_arc = 3.0*self.thetaE/np.sqrt(self.obj["q"])
        self.nnn = self.obj["nnn"]
        self.dsx_arc = self.bsz_arc / self.nnn
        self.rescale_am = self.thetaE / np.sqrt(self.obj["q"])

        self.savefile = savefile
        ensure_dir(self.savefile)
        if globalfile is None:
            self.globalfile = savefile
        else:
            self.globalfile = globalfile
        ensure_dir(self.globalfile)
    def fix_parameter_mock(self):
        """
        Populate opening angle and FDM perturbation info.
        """
        self.get_delta_theta()
        angle_mean, angle_std = self.compute_opening_angle_with_uncertainty()
        print(f"Opening angle: {angle_mean:.2f} ± {angle_std:.2f} deg")
        (angle_AOB, angle_AOB_err), (angle_AOC, angle_AOC_err) = self.compute_AOB_AOC_with_uncertainty()
        print(f"AOB angle: {angle_AOB:.2f} ± {angle_AOB_err:.2f} deg")
        print(f"AOC angle: {angle_AOC:.2f} ± {angle_AOC_err:.2f} deg")
        print(f"Effective radius: {self.get_Sample_reff()}")
        axis_type = self.compute_axis_type()
        print(f"axistype: {axis_type}")


    def _find_free_gpu(self):
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



    def update_thetaE(self, new_thetaE):
        self.thetaE = new_thetaE
        self.thetaE_rad = arcsec_to_Rad(self.thetaE)
        self.thetaE_mpc_h = self.thetaE * self.arcsec_1
        self.sigma_g = calculate_velocity_dispersion(self.thetaE_rad, self.Da_src, self.Da_ls)
        self.halomass = Sigma_g_to_Mvir(self.thetaE_mpc_h, self.sigma_g, self.zlens) / self.cosmo.h
        print(f"🔁 Updated thetaE: {self.thetaE} arcsec")
        print(f"🔁 Updated main halo mass: {self.halomass:.2e} M_sun")
        nfwpara = NFWParam(cosmo)
        self.r200, self.rhos, self.Concentration, self.rs = nfwpara.nfw_Mz(self.halomass*self.cosmo.h, self.zlens)
        self.bsz_arc = 3.0*self.thetaE/np.sqrt(self.obj["q"])
        self.dsx_arc = self.bsz_arc / self.nnn
    def get_mainhalo_geometry(self):
        q = self.obj["q"]
        thetaE = self.thetaE
        a = thetaE / np.sqrt(q)
        b = thetaE * np.sqrt(q)
        phi = -self.obj["phi_lens"]
        return a, b, phi

    def Get_mainhalo(self,ql = None, pa = None, g_external = None,phi_external = None):
        if ql == None:
            ql = self.obj["q"]
            pa = self.obj["phi_lens"]
        xi2, xi1 = make_c_coor(self.bsz_arc, self.nnn)
        pa = np.deg2rad(pa)
        Potentian0 = SIELensingPot(xi1, xi2, 0, 0, self.sigma_g, ql, pa, self.zlens, self.zsource)
        if g_external == None: 
            g_external = self.obj["gamma_external"]
            phi_external = np.deg2rad(self.obj["phi_external"])

        eg = ext_shear(g=g_external, phi_g=phi_external)
        phi_use = np.arctan2(xi2, xi1)  # angular coordinate
        x = np.sqrt(xi1**2 + xi2**2)    # radial coordinate
        psi_eg = eg.psi(x, phi_use)

        Potentian_main = Potentian0 + psi_eg
        alpha1_global, alpha2_global = potential_to_alphas(Potentian_main, self.dsx_arc)

        data_file = f'{self.globalfile}/{self.obj["name"]}_Global_alpha.npz'
        np.savez(data_file, alpha1_global=alpha1_global, alpha2_global=alpha2_global)
    def Get_mainhalo_SIE(self,ql = None, pa = None, g_external = None,phi_external = None):
        """
        Lenstronomy-compatible SIE main halo builder.
        """
        if ql == None:
            ql = self.obj["q"]
            pa = self.obj["phi_lens"]
        xi2, xi1 = make_c_coor(self.bsz_arc, self.nnn)
        # pa = np.deg2rad(pa)

        pa_use = pa
        

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


        data_file = f'{self.globalfile}/{self.obj["name"]}_Global_alpha.npz'
        np.savez(data_file, alpha1_global=alpha1_global, alpha2_global=alpha2_global)
        return alpha1_global, alpha2_global
    def Get_mainhalo_SIE_Lenstronomy(self, thetaE=None, q_l=None, pa=None, g_external=None, phi_external=None):
        """
        Compute SIE + external shear deflection using Lenstronomy.
        Args:
            thetaE: Einstein radius (arcsec)
            q_l: axis ratio
            pa: position angle (deg)
            g_external: shear strength
            phi_external: shear position angle (deg)
        """
        from lenstronomy.LensModel.lens_model import LensModel

        # Default parameters from stored object
        if q_l is None or pa is None:
            q_l = self.obj["q"]
            pa = self.obj["phi_lens"]
        if g_external is None or phi_external is None:
            g_external = self.obj["gamma_external"]
            phi_external = self.obj["phi_external"]
        if thetaE is None:
            thetaE = self.obj["theta_E"]

        # Build coordinate grid
        xi2, xi1 = make_c_coor(self.bsz_arc, self.nnn)

        # Define Lenstronomy lens model
        lens_model_list = ['SIE', 'SHEAR']
        lensModel = LensModel(lens_model_list=lens_model_list)

        # Convert (q_l, pa) to Lenstronomy e1, e2
        e = (1 - q_l) / (1 + q_l)
        e1 = e * np.cos(2 * np.deg2rad(pa))
        e2 = e * np.sin(2 * np.deg2rad(pa))

        # Convert (g_external, phi_external) to Lenstronomy gamma1, gamma2
        gamma1 = g_external * np.cos(2 * np.deg2rad(phi_external))
        gamma2 = g_external * np.sin(2 * np.deg2rad(phi_external))

        # Set lens parameters
        lens_kwargs = [
            {   # SIE main lens
                'theta_E': thetaE,
                'e1': e1,
                'e2': e2,
                'center_x': 0.0,
                'center_y': 0.0
            },
            {   # External shear
                'gamma1': gamma1,
                'gamma2': gamma2
            }
        ]

        # Compute deflection angles
        alpha_x, alpha_y = lensModel.alpha(xi1.ravel(), xi2.ravel(), kwargs=lens_kwargs)

        # Reshape back to 2D fields
        alpha1_global = alpha_x.reshape(self.nnn, self.nnn)
        alpha2_global = alpha_y.reshape(self.nnn, self.nnn)

        # Save result
        data_file = f'{self.globalfile}/{self.obj["name"]}_Global_alpha.npz'
        np.savez(data_file, alpha1_global=alpha1_global, alpha2_global=alpha2_global)

        return alpha1_global, alpha2_global
    def Get_mainhalo_Mock_SIE_dong(self,ql = None, pa = None, g_external = None,phi_external = None):
        """
        SIE main halo compatible with Dong Jiang workflow.
        """
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
            phi_external = np.deg2rad(self.obj["phi_external"]+pa_use-90)

        eg = ext_shear(g=g_external, phi_g=phi_external)
        phi_use = np.arctan2(xi2, xi1)  # angular coordinate
        x = np.sqrt(xi1**2 + xi2**2)    # radial coordinate

        # psi_eg = eg.psi(x, phi_use)
        # Potentian_main = Potentian0+psi_eg
        # alpha1_global, alpha2_global = potential_to_alphas(Potentian_main, self.dsx_arc)
        al1_eg,al2_eg = eg.alpha(x, phi_use)

        alpha1_global = alpha1_SIE+al1_eg
        alpha2_global = alpha2_SIE+al2_eg


        data_file = f'{self.globalfile}/{self.obj["name"]}_Global_alpha.npz'
        np.savez(data_file, alpha1_global=alpha1_global, alpha2_global=alpha2_global)
        return alpha1_global, alpha2_global
    def Get_mainhalo_Mock_SIE(self,ql = None, pa = None, g_external = None,phi_external = None):
        """
        Lenstronomy-friendly mock SIE main halo.
        """
        if ql == None:
            ql = self.obj["q"]
            pa = self.obj["phi_lens"]
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


        data_file = f'{self.globalfile}/{self.obj["name"]}_Global_alpha.npz'
        np.savez(data_file, alpha1_global=alpha1_global, alpha2_global=alpha2_global)
        return alpha1_global, alpha2_global
    def Get_mainhalo_Mock(self):
        from cal_mul_fits import to_elliptical_params,_shear_amp_pa_to_gamma
        obj = self.obj
        # print(obj)


        # --- Grid ---
        nnn = obj["nnn"]
        bsz_arc = self.bsz_arc
        xi2, xi1 = make_c_coor(bsz_arc, nnn)  # (y,x) convention
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

        # --- Compute deflection ---
        alpha_x, alpha_y = lensModel.alpha(
            xi2.ravel(), xi1.ravel(),
            kwargs=[kwargs_epl, kwargs_shear, kwargs_m3, kwargs_m4]
        )
        alpha2_global = np.asarray(alpha_x).reshape(nnn, nnn)
        alpha1_global = np.asarray(alpha_y).reshape(nnn, nnn)

        # --- Save ---
        os.makedirs(self.globalfile, exist_ok=True)
        data_file = f'{self.globalfile}/{obj["name"]}_Global_alpha.npz'
        np.savez(data_file, alpha1_global=alpha1_global, alpha2_global=alpha2_global)
        return alpha1_global, alpha2_global
    def Get_mainhalo_Mock_mul_cir(self):
        from cal_mul_fits import to_elliptical_params,_shear_amp_pa_to_gamma
        obj = self.obj
        # print(obj)


        # --- Grid ---
        nnn = obj["nnn"]
        bsz_arc = self.bsz_arc
        xi2, xi1 = make_c_coor(bsz_arc, nnn)  # (y,x) convention
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

        lens_model_list = ["SHEAR", "EPL_MULTIPOLE_M3M4"]
        lensModel = LensModel(lens_model_list=lens_model_list)

        # --- Compute deflection ---
        alpha_x, alpha_y = lensModel.alpha(
            xi2.ravel(), xi1.ravel(),
            kwargs=[kwargs_shear, kwargs_epl34]
        )
        alpha2_global = np.asarray(alpha_x).reshape(nnn, nnn)
        alpha1_global = np.asarray(alpha_y).reshape(nnn, nnn)

        # --- Save ---
        os.makedirs(self.globalfile, exist_ok=True)
        data_file = f'{self.globalfile}/{obj["name"]}_Global_alpha.npz'
        np.savez(data_file, alpha1_global=alpha1_global, alpha2_global=alpha2_global)
        return alpha1_global, alpha2_global
    def get_Sample_reff(self):
        xi2, xi1 = make_c_coor(self.bsz_arc, self.nnn)
        alpha1_global, alpha2_global = self.Get_mainhalo_Mock()
        yi1, yi2, mu_global, kappa_, gamma1_gobal, gamma2_gobal = alphas_to_mu(
            alpha1_global, alpha2_global, self.dsx_arc, xi1, xi2
        )
        gamma_global = np.sqrt(gamma1_gobal**2 + gamma2_gobal**2)
        lambdat_global = 1 - kappa_ - gamma_global
        lambdat_global_np = np.array(lambdat_global)

        fig, ax = plt.subplots()
        lambdat_contour = ax.contour(yi2, yi1, lambdat_global_np, levels=[0])
        # plt.close(fig)

        lambdat_contour_paths = lambdat_contour.allsegs[0]
        areas = []
        for seg in lambdat_contour_paths:
            x, y = seg[:, 0], seg[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas.append(area)

        if areas:
            max_index = np.argmax(areas)
            area_max = areas[max_index]
            reff = np.sqrt(area_max / np.pi)
            self.obj['Sample_reff'] = reff
            return reff
        else:
            return None


    
    def get_delta_theta(self, r_ein=1.0):
        """
        Estimate rms angular perturbation (Δθ_rms) from ULDM fluctuations to gauge image shifts.
        r_ein: characteristic angular scale (arcsec), e.g., Einstein radius used to evaluate Σ_host and sensitivity.
        """
        sigma_host_value = compute_sigma_host(r_ein, self.rs, self.rhos,self.arcsec_1)
        kappa_host_value = sigma_host_value/self.sigma_crit
        delta_kappa_rms = compute_delta_kappa_rms(kappa_host_value)
        self.delta_theta = compute_eps2_dtheta2(delta_kappa_rms,self.thetaE)
        self.obj["delta_theta_FDM"] = self.delta_theta
        return self.delta_theta
    def compute_axis_type(self):
        negative_point = np.array([self.obj["x_imageA"], self.obj["y_imageA"]])
        non_cusp_point = np.array([self.obj["x_imageD"], self.obj["y_imageD"]])
        center_point = np.array([0,0])
        dist_negative = np.linalg.norm(negative_point - center_point)
        dist_non_cusp = np.linalg.norm(non_cusp_point - center_point)

        if dist_negative > dist_non_cusp:
            axis_type = "long_axis"
        else:
            axis_type = "short_axis"
        
        self.obj["axis_type"] = axis_type
        return axis_type

    def compute_opening_angle_with_uncertainty(self, n_samples=1000, seed=42):
        rng = np.random.default_rng(seed)
        theo_sigma = self.delta_theta / 2

        xB, yB = self.obj["x_imageB"], self.obj["y_imageB"]
        xC, yC = self.obj["x_imageC"], self.obj["y_imageC"]

        angles = []
        for _ in range(n_samples):
            p1 = rng.normal(loc=[xB, yB], scale=[theo_sigma, theo_sigma])
            p2 = rng.normal(loc=[xC, yC], scale=[theo_sigma, theo_sigma])

            v1 = p1  # O at origin
            v2 = p2
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.degrees(np.arccos(cos_theta))
            angles.append(theta)

        self.obj["phi_sigma"] = np.std(angles)
        return np.mean(angles), np.std(angles)
    def compute_AOB_AOC_with_uncertainty(self, n_samples=1000, seed=42):
        rng = np.random.default_rng(seed)
        theo_sigma = self.delta_theta / 2

        xA, yA = self.obj["x_imageA"], self.obj["y_imageA"]
        xB, yB = self.obj["x_imageB"], self.obj["y_imageB"]
        xC, yC = self.obj["x_imageC"], self.obj["y_imageC"]

        angles_AOB = []
        angles_AOC = []

        for _ in range(n_samples):
            ptA = rng.normal(loc=[xA, yA], scale=[theo_sigma, theo_sigma])
            ptB = rng.normal(loc=[xB, yB], scale=[theo_sigma, theo_sigma])
            ptC = rng.normal(loc=[xC, yC], scale=[theo_sigma, theo_sigma])

            # AOB: angle between OA and OB
            vOA = ptA
            vOB = ptB
            cos_AOB = np.dot(vOA, vOB) / (np.linalg.norm(vOA) * np.linalg.norm(vOB))
            angle_AOB = np.degrees(np.arccos(np.clip(cos_AOB, -1.0, 1.0)))
            angles_AOB.append(angle_AOB)

            # AOC: angle between OA and OC
            vOC = ptC
            cos_AOC = np.dot(vOA, vOC) / (np.linalg.norm(vOA) * np.linalg.norm(vOC))
            angle_AOC = np.degrees(np.arccos(np.clip(cos_AOC, -1.0, 1.0)))
            angles_AOC.append(angle_AOC)

        mean_AOB, std_AOB = np.mean(angles_AOB), np.std(angles_AOB)
        mean_AOC, std_AOC = np.mean(angles_AOC), np.std(angles_AOC)

        self.obj["phi1_sigma"] = std_AOB
        self.obj["phi2_sigma"] = std_AOC

        return (mean_AOB, std_AOB), (mean_AOC, std_AOC)
    


    def generate_halo_realization(self):
        halofile = f'{self.savefile}/{self.name}_halolist.pkl'
        try:
            realization = CDM(
                self.zlens, self.zsource,
                cone_opening_angle_arcsec=math.ceil(self.bsz_arc*1.5),
                subhalo_spatial_distribution='PROJECTED_NFW',
                log_m_host=np.log10(self.halomass), # Msun (no h factor)
                LOS_normalization=1.0,
                log_mlow=np.log10(self.halomass) - 7.5,
                log_mhigh=np.log10(self.halomass) - 1
            )
            print(f"Realization created with {len(realization.halos)} halos.")

            if len(realization.halos) == 0:
                log_file = os.path.join(os.getcwd(), "halo_realization_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"No halos generated for {halofile}\n")
                return None

            params = []
            for halo in tqdm(realization.halos, desc="Processing CDM halos"):
                try:
                    params.append({
                        'mass': halo.mass,
                        'concentration': halo.c,
                        'tnfw_params': halo.params_physical,
                        'isSubhalo': halo.is_subhalo,
                        'halo_age': halo.halo_age,
                        'redshift': halo.z,
                        'position_x': halo.x,
                        'position_y': halo.y
                    })
                except:
                    continue

            with open(halofile, 'wb') as f:
                pickle.dump(params, f) 
            print(f"Saved halo list to {halofile}")
            return "Finish!"
        except Exception as e:
            log_file = os.path.join(os.getcwd(), "halo_realization_log.txt")
            with open(log_file, "a") as f:
                f.write(f"CDM realization failed for {halofile}, "
                        f"name={self.name}, error={str(e)}\n")
            return None
    def generate_halo_realization_WDM(self):
        halofile = f'{self.savefile}/{self.name}_halolist_WDM.pkl'

        kwargs_concentration_model_subhalos = {'scatter': True}
        kwargs_concentration_model_fieldhalos = {'scatter': True}
        log10_half_mode_mass = 7.0
        realization = WDM(
            self.zlens, self.zsource,
            cone_opening_angle_arcsec=math.ceil(self.bsz_arc*1.5),
            log_mc=log10_half_mode_mass, 
            subhalo_spatial_distribution='PROJECTED_NFW',
            log_m_host=np.log10(self.halomass),
            LOS_normalization=1.0,
            log_mlow=np.log10(self.halomass) - 7.5,
            log_mhigh=np.log10(self.halomass) - 1,
            kwargs_concentration_model_subhalos=kwargs_concentration_model_subhalos,
            kwargs_concentration_model_fieldhalos=kwargs_concentration_model_fieldhalos
        )
        print(f"Realization created with {len(realization.halos)} halos.")
        params = []
        for halo in tqdm(realization.halos, desc="Processing WDM halos"):
            try:
                params.append({
                    'mass': halo.mass,
                    'concentration': halo.c,
                    'tnfw_params': halo.params_physical,
                    'isSubhalo': halo.is_subhalo,
                    'halo_age': halo.halo_age,
                    'redshift': halo.z,
                    'position_x': halo.x,
                    'position_y': halo.y
                })
            except:
                continue

        with open(halofile, 'wb') as f:
            pickle.dump(params, f)
        print(f"Saved halo list to {halofile}")

    
    def generate_halo_realization_FDM(self, mass=0.8*10**(-22), Amplitude=-1.3):
        log10_m_uldms = np.log10(mass)
        try:
            realizationsULDM = ULDM(
                self.zlens, self.zsource,
                cone_opening_angle_arcsec=math.ceil(self.bsz_arc*1.5),
                log10_m_uldm=log10_m_uldms,
                flucs_shape='aperture',
                flucs_args={'aperture': 3 * self.thetaE, 'x_images': [0.0], 'y_images': [0.0]},
                log10_fluc_amplitude=Amplitude,
                n_cut=1000000,
                log_m_host=np.log10(self.halomass), 
                LOS_normalization=1.0,
                log_mlow=np.log10(self.halomass) - 7.5,
                log_mhigh=np.log10(self.halomass) - 1
            )
            return realizationsULDM
        except Exception as e:
            log_file = os.path.join(os.getcwd(), "halo_realization_log.txt")
            with open(log_file, "a") as f:
                f.write(f"FDM realization failed for mass={mass}, Amplitude={Amplitude}, "
                        f"name={self.name}, error={str(e)}\n")
                
            return None
        
    def generate_halo_realization_FDM_r(self, mass=0.8e-22, Amplitude=-1.3):
        log10_m_uldms = np.log10(mass)
        try:
            realizationsULDM = ULDM_r(
                self.zlens,
                self.zsource,
                cone_opening_angle_arcsec=math.ceil(self.bsz_arc*1.5),
                log10_m_uldm=log10_m_uldms,
                flucs_shape='aperture',
                flucs_args={
                    'aperture': 3 * self.thetaE,
                    'x_images': [0.0],
                    'y_images': [0.0]
                },
                log10_fluc_amplitude=Amplitude,
                n_cut=1000000,
                log_m_host=np.log10(self.halomass),
                LOS_normalization=1.0,
                log_mlow=np.log10(self.halomass) - 7.5,
                log_mhigh=np.log10(self.halomass) - 1,
                r_ein=self.thetaE  # Pass Einstein radius to ULDM
            )
            return realizationsULDM
        except Exception as e:
                log_file = os.path.join(os.getcwd(), "halo_realization_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"FDM realization failed for mass={mass}, Amplitude={Amplitude}, "
                            f"name={self.name}, error={str(e)}\n")
                    
                return None



    def convert_CDM_to_SIDM_halo_list(self):
        sidmfile = f'{self.savefile}/{self.name}_halolist_sigma0_{self.sigma0}_w0_{self.w0}.pkl'


        with open(f'{self.savefile}/{self.name}_halolist.pkl', 'rb') as f:
            data = pickle.load(f)

        masses = jnp.array([h['mass'] for h in data])
        rhoss = jnp.array([h['tnfw_params']['rhos'] for h in data]) 
        rss   = jnp.array([h['tnfw_params']['rs']   for h in data]) #kpc
        zeros = jnp.zeros_like(rss)
        taus = jax.vmap(get_taub, in_axes=(0,0,0,0,0,None,None))(
            masses, rhoss, rss, zeros, zeros, self.sigma0, self.w0
        )
        for h, t in zip(data, taus):
            h['tau'] = float(t)
        with open(sidmfile, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved SIDM halo list to {sidmfile}")

    def process_halo_alpha_from_massfunc(self, mass_function, seed=None):
        """
        Generate subhalos in the field using a mass function and compute multi-plane alpha maps.

        Parameters
        ----------
        mass_function : dict or list
            - dict: {mass_in_Msun: count}
            - list/tuple: [(mass_in_Msun, count), ...]
        seed : int, optional
            Random seed
        """
        import numpy as np

        rng = np.random.default_rng(seed)

        # Normalize mass_function into (mass, count) list
        if isinstance(mass_function, dict):
            mc_list = [(float(m), int(c)) for m, c in mass_function.items() if int(c) > 0]
        else:
            mc_list = [(float(m), int(c)) for m, c in list(mass_function) if int(c) > 0]

        total_halos = sum(c for _, c in mc_list)
        if total_halos == 0:
            raise ValueError("mass_function contains no positive counts.")

        # Uniformly sample positions within the field (arcsec)
        half = self.bsz_arc / 2.0
        xs = rng.uniform(-half, half, size=total_halos)
        ys = rng.uniform(-half, half, size=total_halos)

        # Build tnfw parameter list
        tnfw = []
        idx = 0
        for mass, count in mc_list:
            # lenstronomy NFWParam returns rs, rhos in Mpc/h and Msun*h^2/Mpc^3; no extra conversions needed
            # Assume all subhalos share redshift self.zlens
            for _ in range(count):
                x, y = float(xs[idx]), float(ys[idx])
                idx += 1
                from lenstronomy.Cosmo.nfw_param import NFWParam
                nfwpara = NFWParam(cosmo)
                r200, rhos, conc, rs = nfwpara.nfw_Mz(mass, self.zlens)  # From lenstronomy.Cosmo.nfw_param.NFWParam

                tnfw.append({
                    'mass': mass * cosmo.h,  # Msun -> Msun/h (keep consistent with framework)
                    'tnfw_params': {
                        'rs': rs,       # Already Mpc/h
                        'rhos': rhos    # Already Msun*h^2/Mpc^3
                    },
                    'redshift': float(self.zlens),
                    'tau': 1.08,       # type_halo=None
                    'position_x': x,
                    'position_y': y,
                    'ql': 1, 'pa': 0
                })

        # Compute multi-plane alpha and save
        alphafile = f"{self.savefile}/{self.name}_None_alpha_mul.npz"
        model = SIDM_parametric_Multiplane(
            tnfw_params=tnfw,
            bsz_arc=self.bsz_arc,
            dsx_arc=self.dsx_arc,
            nnn=self.nnn,
            zsource=self.zsource
        )
        a1, a2 = model._compute_grouped_alpha_maps(batch_size=30)
        np.savez(alphafile, alpha1_sub=a1, alpha2_sub=a2)
        np.save(f"{self.savefile}/{self.name}_None_redshift_list.npy", model.grouped_redshift_array)
        print(f"Saved alpha maps to {alphafile}")

    def process_halo_alpha(self, type_halo="CDM"):
        alphafile = f'{self.savefile}/{self.name}_{type_halo}_alpha_mul.npz'

        if type_halo == "WDM":
            with open(f'{self.savefile}/{self.name}_halolist_WDM.pkl', 'rb') as f:
                halodata = pickle.load(f)
        else:
            with open(f'{self.savefile}/{self.name}_halolist_sigma0_{self.sigma0}_w0_{self.w0}.pkl', 'rb') as f:
                halodata = pickle.load(f)

        tnfw = []
        half = self.bsz_arc / 2
        for h in halodata:
            x, y = h['position_x'], h['position_y']
            if -half <= x <= half and -half <= y <= half:
                if type_halo=="CDM" or type_halo=="WDM":
                    tau = 0.0
                elif type_halo=="SIDM_col":
                    tau = 1.08
                elif type_halo=="SIDM":
                    tau = h['tau']
                else:
                    raise ValueError("Unsupported type_halo: {}".format(type_halo))


                p = h['tnfw_params']
                tnfw.append({
                    'mass': h['mass']* cosmo.h, #Msun/h
                    'tnfw_params': {'rs': p['rs']/1000 * cosmo.h, 'rhos': p['rhos']*1000**3/ cosmo.h / cosmo.h}, #rs: Mpc/h, rhos: Msun *h^2 /Mpc^3 
                    'redshift': h['redshift'],
                    'tau': tau,
                    'position_x': x,
                    'position_y': y,
                    'ql': 1, 'pa': 0
                })

        model = SIDM_parametric_Multiplane(
            tnfw_params=tnfw, bsz_arc=self.bsz_arc,
            dsx_arc=self.dsx_arc,nnn = self.nnn, zsource=self.zsource
        )
        a1, a2 = model._compute_grouped_alpha_maps(batch_size=30)
        np.savez(alphafile, alpha1_sub=a1, alpha2_sub=a2)
        np.save(f"{self.savefile}/{self.name}_{type_halo}_redshift_list.npy", model.grouped_redshift_array)
        print(f"Saved alpha maps to {alphafile}")
    
    def process_halo_alpha_FDM(self, realizationsULDM, suffix: str = None):
        lens_model_list, lens_redshift_array, kwargs_halos, numerical_deflection_class = realizationsULDM.lensing_quantities()
        print('Number of FDM halos:', len(kwargs_halos))
        lens_model_list     = np.array(lens_model_list)
        lens_redshift_array = np.array(lens_redshift_array)
        half = self.bsz_arc / 2

        # 2. Index by model type
        idx_gaussian = np.where(lens_model_list == 'GAUSSIAN')[0]
        idx_uldm     = np.where(lens_model_list == 'ULDM')[0]
        idx_cnfw     = np.where(lens_model_list == 'CNFW')[0]

        # 3. Build parameter lists including redshift and position_*
        Gaussian_params = []
        for i in idx_gaussian:
            p = kwargs_halos[i].copy()
            if 'center_x' in p and 'center_y' in p:
                if -half <= p.get('center_x') <= half and -half <= p.get('center_y') <= half:
                    p['position_x'] = p.pop('center_x')
                    p['position_y'] = p.pop('center_y')
                else: 
                    continue
            p['redshift'] = float(lens_redshift_array[i])
            Gaussian_params.append(p)

        ULDM_params = []
        for i in idx_uldm:
            p = kwargs_halos[i].copy()
            if 'center_x' in p and 'center_y' in p:
                if -half <= p.get('center_x') <= half and -half <= p.get('center_y') <= half:
                    p['position_x'] = p.pop('center_x')
                    p['position_y'] = p.pop('center_y')
                else: 
                    continue
            p['redshift'] = float(lens_redshift_array[i])
            ULDM_params.append(p)

        CNFW_params = []
        for i in idx_cnfw:
            p = kwargs_halos[i].copy()
            if 'center_x' in p and 'center_y' in p:
                if -half <= p.get('center_x') <= half and -half <= p.get('center_y') <= half:
                    p['position_x'] = p.pop('center_x')
                    p['position_y'] = p.pop('center_y')
                else: 
                    continue
            p['redshift'] = float(lens_redshift_array[i])
            CNFW_params.append(p)

        del lens_model_list, lens_redshift_array, kwargs_halos, idx_gaussian, idx_uldm, idx_cnfw  
        gc.collect()

        # Handle filename suffix
        name_suffix = f"{self.name}" if suffix is None else f"{self.name}_{suffix}"

        model_g = GAUSSIAN_parametric_Multiplane(
            Gaussian_params=Gaussian_params,
            bsz_arc=self.bsz_arc, dsx_arc=self.dsx_arc, nnn=self.nnn,
            zsource=self.zsource
        )
        alpha1_gauss, alpha2_gauss = model_g._compute_grouped_alpha_maps(batch_size=30)

        alphafile = f"{self.savefile}/{name_suffix}_gaussian_alpha_mul.npz"
        np.savez(alphafile, alpha1_sub=alpha1_gauss, alpha2_sub=alpha2_gauss)
        np.save(f"{self.savefile}/{name_suffix}_gaussian_redshift_list.npy",
                model_g.grouped_redshift_array)
        print(f"Saved Gaussian alpha maps to {alphafile}")
        del model_g, alpha1_gauss, alpha2_gauss
        gc.collect()

        model_u = ULDM_parametric_Multiplane(
            ULDM_params=ULDM_params,
            bsz_arc=self.bsz_arc, dsx_arc=self.dsx_arc, nnn=self.nnn,
            zsource=self.zsource
        )
        alpha1_uldm, alpha2_uldm = model_u._compute_grouped_alpha_maps(batch_size=30)

        alphafile = f"{self.savefile}/{name_suffix}_uldm_alpha_mul.npz"
        np.savez(alphafile, alpha1_sub=alpha1_uldm, alpha2_sub=alpha2_uldm)
        np.save(f"{self.savefile}/{name_suffix}_uldm_redshift_list.npy",
                model_u.grouped_redshift_array)
        print(f"Saved ULDM alpha maps to {alphafile}")
        del model_u, alpha1_uldm, alpha2_uldm
        gc.collect()

        model_c = CNFW_parametric_Multiplane(
            CNFW_params=CNFW_params,
            bsz_arc=self.bsz_arc, dsx_arc=self.dsx_arc, nnn=self.nnn,
            zsource=self.zsource
        )
        alpha1_cnfw, alpha2_cnfw = model_c._compute_grouped_alpha_maps(batch_size=30)

        alphafile = f"{self.savefile}/{name_suffix}_cnfw_alpha_mul.npz"
        np.savez(alphafile, alpha1_sub=alpha1_cnfw, alpha2_sub=alpha2_cnfw)
        np.save(f"{self.savefile}/{name_suffix}_cnfw_redshift_list.npy",
                model_c.grouped_redshift_array)
        print(f"Saved CNFW alpha maps to {alphafile}")
        del model_c, alpha1_cnfw, alpha2_cnfw
        gc.collect()
    

    
    def Simulation_FDM_lenstronomy(self, realizationsULDM):
        lens_model_list, lens_redshift_array, kwargs_halos, numerical_deflection_class = realizationsULDM.lensing_quantities()
        astropy_instance = realizationsULDM.astropy_instance

        # Filter only GAUSSIAN halos
        gaussian_halo_indices = [
            i for i, model in enumerate(lens_model_list) if model == 'GAUSSIAN'
        ]
        filtered_lens_model_list = [lens_model_list[i] for i in gaussian_halo_indices]
        filtered_lens_redshift_array = [lens_redshift_array[i] for i in gaussian_halo_indices]
        filtered_kwargs_halos = [kwargs_halos[i] for i in gaussian_halo_indices]

        print(f"Number of Gaussian halos: {len(filtered_lens_model_list)}") 
        npix = 2000
        grid_size = 4 * self.thetaE
        _x = _y = np.linspace(-grid_size / 2, grid_size / 2, npix)
        xx, yy = np.meshgrid(_x, _y)
        shape0 = xx.shape

        # Build lens model containing only Gaussian halos
        lens_model = LensModel(
            filtered_lens_model_list,
            self.zlens, self.zsource,
            lens_redshift_list=filtered_lens_redshift_array,
            cosmo=astropy_instance,
            multi_plane=False
        )

        alpha1, alpha2 = lens_model.alpha(xx.ravel(), yy.ravel(), filtered_kwargs_halos)
        alpha1_2d = alpha1.reshape(xx.shape)
        alpha2_2d = alpha2.reshape(yy.shape)

        data_file = f"{self.name}_FDM_alpha_gaussian_test.npz"
        np.savez(data_file, alpha1_global=alpha2_2d, alpha2_global=alpha1_2d)
        print(f"Data saved to '{data_file}'.")


    def get_ray_trace_alpha(self, type_halo="CDM"):
        infile = f'{self.savefile}/{self.name}_{type_halo}_alpha_mul.npz'
        outfile = f'{self.savefile}/{self.name}_{type_halo}_alpha_mul_ray.npz'


        redshift_list_file = f"{self.savefile}/{self.name}_{type_halo}_redshift_list.npy"
        global_file = f"{self.globalfile}/{self.name}_Global_alpha.npz"
        Get_multi_alpha(self.obj, self.bsz_arc, self.nnn,infile, redshift_list_file, self.zlens, self.zsource,
                        filename_main=global_file, output_file=outfile)
        
    def Simulation_FDM(self):
        realizationsULDM = self.generate_halo_realization_FDM()
        if realizationsULDM is None:
            return None
        self.process_halo_alpha_FDM(realizationsULDM)
        merge_FDM_alpha_maps_jax(self.name, self.savefile)
        self.get_ray_trace_alpha(type_halo="FDM")

    def Simulation_FDM_r(self, ifsuffix=None):
        realizationsULDM = self.generate_halo_realization_FDM_r()
        if realizationsULDM is None:
            return None
        suffix = "r" if ifsuffix is not None else None
        self.process_halo_alpha_FDM(realizationsULDM, suffix=suffix)
        merge_FDM_alpha_maps_jax(self.name, self.savefile, suffix=suffix)
        type_halo = "FDM_r" if ifsuffix is not None else "FDM"
        self.get_ray_trace_alpha(type_halo=type_halo)
    def Constract_High_Resolution_FDM(self,resolution_num_list):
        realizationsULDM = self.generate_halo_realization_FDM()
        n_init = self.nnn
        savefile_init = self.savefile
        # High-resolution processing
        for resolution_num in resolution_num_list:
            self.nnn = int(resolution_num* n_init)
            self.savefile = os.path.join(savefile_init,f"{resolution_num}_resolution")
            ensure_dir(self.savefile)
            self.globalfile = self.savefile
            self.Get_mainhalo() 
            self.process_halo_alpha_FDM(realizationsULDM)

            merge_FDM_alpha_maps_jax(self.name, self.savefile)
            self.get_ray_trace_alpha(type_halo="FDM")


    
    def Simulation_WDM(self, ifgenerate=True):
        if ifgenerate:
            self.generate_halo_realization_WDM()
        self.process_halo_alpha(type_halo="WDM")
        self.get_ray_trace_alpha(type_halo="WDM")

    def Simulation_CDM_SIDM(self, ifgenerate=True):
        if ifgenerate:
            realization = self.generate_halo_realization()
            if realization is None:
                return None
            self.convert_CDM_to_SIDM_halo_list()

        self.process_halo_alpha(type_halo="CDM")
        self.get_ray_trace_alpha(type_halo="CDM")

        self.process_halo_alpha(type_halo="SIDM_col")
        self.get_ray_trace_alpha(type_halo="SIDM_col")
    def Simulation_SIDM(self, ifgenerate=True):
        if ifgenerate:
            realization = self.generate_halo_realization()
            if realization is None:
                return None
            self.convert_CDM_to_SIDM_halo_list()

        self.process_halo_alpha(type_halo="SIDM")
        self.get_ray_trace_alpha(type_halo="SIDM")

    def Simulation_MCMC_Mock(self, sim_type, plot_png=True):
        xi2, xi1 = make_c_coor(self.bsz_arc, self.nnn)
        y_opt = np.array([self.obj["source_xlocation"], self.obj["source_ylocation"]])

        out_file = f'{self.savefile}/{self.name}_{sim_type}_alpha_mul_ray.npz'
        print(f"Loading: {out_file}")
        if not os.path.exists(out_file):
            return None
        kappa_multi, yi1, yi2, mu_global, lambdat_global,_ = get_kappa(
            out_file, self.bsz_arc, self.nnn
        )
        r_eff = self.obj["Sample_reff"] / 8

        save_path_combined = f"{self.savefile}/{self.name}_{sim_type}_chain_combined.npz"

        # Skip when MCMC results already exist
        if not os.path.exists(save_path_combined):
            # Coarse MCMC
            sampler_coarse = run_mcmc(
                mu_global, xi1, xi2, yi1, yi2,
                obs_phi=self.obj["phi"], obs_phi_sigma=2 * self.obj["phi_sigma"],
                obs_phi1=self.obj["phi1"], obs_phi1_sigma=2 * self.obj["phi1_sigma"],
                obs_phi2=self.obj["phi2"], obs_phi2_sigma=2 * self.obj["phi2_sigma"],
                x_center=y_opt[1], y_center=y_opt[0],
                n_walkers=5, n_steps=600, initial_radius=r_eff
            )

            log_probs = sampler_coarse.get_log_prob(discard=50, flat=True)
            chi2 = -2 * log_probs
            samples_coarse = sampler_coarse.get_chain(discard=50, flat=True)

            best_index = np.argmin(chi2)
            best_point = samples_coarse[best_index]
            best_chi2 = chi2[best_index]

            x_best, y_best = best_point
            print(f"Best (x, y): ({x_best:.6f}, {y_best:.6f}) with chi² = {best_chi2}")

            if math.isinf(best_chi2):
                log_file = os.path.join(os.getcwd(), "mcmc_rebin_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"{save_path_combined}\n")
                return

            # Fine MCMC
            sampler = run_mcmc(
                mu_global, xi1, xi2, yi1, yi2,
                obs_phi=self.obj["phi"], obs_phi_sigma=self.obj["phi_sigma"],
                obs_phi1=self.obj["phi1"], obs_phi1_sigma=self.obj["phi1_sigma"],
                obs_phi2=self.obj["phi2"], obs_phi2_sigma=self.obj["phi2_sigma"],
                x_center=x_best, y_center=y_best,
                n_walkers=10, n_steps=1000, initial_radius=r_eff / 2
            )

            os.makedirs(self.savefile, exist_ok=True)
            save_sampler_result(sampler, save_path_combined)
        else:
            print(f"Found existing chain file: {save_path_combined}, skipping MCMC...")

        selected_points = rebin_mcmc_points_full_bs_jax(save_path_combined, self.bsz_arc, sigma_level=5)
        if selected_points is None:
            return
        print(f"Selected points within 1σ: {len(selected_points)}")

        filtered_samples = selected_points[:, :2]
        chi2 = selected_points[:, 2]
        xroots_list, mu_info_list, Rcusp_arr, weights, angles_arr, axis_type_list = get_xroots_and_mu_info(
            filtered_samples, chi2, xi1, xi2, yi1, yi2, mu_global
        )

        save_path_rcusp = f"{self.savefile}/{self.name}_{sim_type}_Rcusp_phi.npz"
        np.savez(save_path_rcusp, Rcusp_arr=Rcusp_arr, weights=weights, angles=angles_arr,
                xroots=xroots_list, mu_info=mu_info_list, axis_types=axis_type_list)

        if plot_png:
            x_all = []
            y_all = []
            weights_all = []
            for xroots, weight in zip(xroots_list, weights):
                for pt in xroots:
                    x_all.append(pt[1])       # x
                    y_all.append(pt[0])       # y
                    weights_all.append(weight)

            from show_kappa import plot_kappa, get_main_plan_kappa

            k_main = get_main_plan_kappa(self.bsz_arc, filename_main=f"{self.globalfile}/{self.name}_Global_alpha.npz", nnn=self.nnn)
            self._plot_kappa_with_xroots_and_observations(
                x_all=x_all,
                y_all=y_all,
                weights=weights_all,
                sim_type=sim_type,
                plot_kappa_func=plot_kappa,
                kappa=kappa_multi - k_main,
                mu=mu_global,
                bsz=self.bsz_arc,
                nnn=self.nnn,
                save_path=f"{self.savefile}/{self.name}_{sim_type}_lens_plane.png"
            )

        return "Finish!"
    def Simulation_MCMC_Mock_each_phibin(self, sim_type, phi_center, delta_phi = 5):
        xi2, xi1 = make_c_coor(self.bsz_arc, self.nnn)
        y_opt = np.array([0, 0])
        if sim_type == 'None':
            out_file = f'{self.savefile}/{self.name}_Global_alpha.npz'
        else:
            out_file = f'{self.savefile}/{self.name}_{sim_type}_alpha_mul_ray.npz'
        print(f"Loading: {out_file}")
        if not os.path.exists(out_file):
            return None

        kappa_multi, yi1, yi2, mu_global, lambdat_global,_ = get_kappa(
            out_file, self.bsz_arc, self.nnn
        )
        r_eff = self.obj["Sample_reff"]

        

        print(f"--- φ = {phi_center}° ---")

        # Dynamically set n_steps: smaller phi → more steps
        n_steps_dynamic = int(-16.6667 * phi_center + 2333.33)
        n_steps_dynamic = max(n_steps_dynamic, 1000)

        # save_path_mcmc = f"{self.savefile}/MCMC_phibin"
        save_path_mcmc = f"{self.savefile}"
        os.makedirs(save_path_mcmc, exist_ok=True)

        save_path_combined = f"{save_path_mcmc}/{self.name}_{sim_type}_{phi_center}_chain_combined.npz"
        save_path_rcusp = f"{save_path_mcmc}/{self.name}_{sim_type}_{phi_center}_Rcusp_phi.npz"
        if os.path.exists(save_path_rcusp):
            print(f"✔️ File already exists, skipping: {save_path_rcusp}")
            return

        if not os.path.exists(save_path_combined):
            sampler = run_mcmc(
                mu_global, xi1, xi2, yi1, yi2,
                obs_phi=phi_center, obs_phi_sigma=delta_phi,
                obs_phi1=phi_center / 2, obs_phi1_sigma=delta_phi / 2,
                obs_phi2=phi_center / 2, obs_phi2_sigma=delta_phi / 2,
                x_center=y_opt[1], y_center=y_opt[0],
                n_walkers=10, n_steps=n_steps_dynamic, initial_radius=r_eff
            )

            save_sampler_result(sampler, save_path_combined, skip_num=200)
        else:
            print(f"Found existing chain file: {save_path_combined}, skipping MCMC...")

        selected_points = rebin_mcmc_points_full_bs_jax(save_path_combined, self.bsz_arc, sigma_level=5)
        if selected_points is None:
            print(f"No selected points for φ = {phi_center}°")
            return

        print(f"Selected points within 1σ: {len(selected_points)}")
        # If too many points, sample 3000 randomly
        if len(selected_points) > 3000:
            idx_choice = np.random.choice(len(selected_points), size=3000, replace=False)
            selected_points = selected_points[idx_choice]
            print("⚠️ Too many points, randomly sampled 3000 for calculation")


        filtered_samples = selected_points[:, :2]
        chi2 = selected_points[:, 2]
        xroots_list, mu_info_list, Rcusp_arr, weights, angles_arr, axis_type_list = get_xroots_and_mu_info(
            filtered_samples, chi2, xi1, xi2, yi1, yi2, mu_global
        )

        np.savez(save_path_rcusp, Rcusp_arr=Rcusp_arr, weights=weights, angles=angles_arr,
                xroots=xroots_list, mu_info=mu_info_list, axis_types=axis_type_list)
        print(f"Saved Rcusp data for φ = {phi_center}° to {save_path_rcusp}")
    def _plot_kappa_multi_minus_main(self, all_types, iter_idx=0, vmin=-0.09, vmax=0.09,
                                scale_arcsec=0.3, figsize=(18, 6), output_name=None,
                                show=False):
        """
        Plot kappa maps for each simulation type minus the main plane as a grid and save to PNG.
        Returns the saved PNG path.
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from show_kappa import plot_kappa, get_kappa, get_main_plan_kappa
        # Main-plane κ
        k_main = get_main_plan_kappa(
            self.bsz_arc,
            filename_main=f"{self.globalfile}/{self.name}_Global_alpha.npz",
            nnn=self.nnn
        )

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, len(all_types),
                            width_ratios=[1] * len(all_types),
                            wspace=0.01)

        im_main = None
        for i, sim_type in enumerate(all_types):
            out_file = f"{self.savefile}/{self.name}_{sim_type}_alpha_mul_ray.npz"
            kappa_multi, yi1, yi2, mu_global, lambdat_global, _ = get_kappa(
                out_file, self.bsz_arc, self.nnn
            )

            ax = fig.add_subplot(gs[0, i])
            im_main = plot_kappa(
                ax,
                kappa_multi - k_main,
                mu_global,
                self.bsz_arc,
                self.nnn,
                vmin=vmin,
                vmax=vmax
            )

            # Add scale bar to the first subplot
            if i == 0:
                x_start = -self.bsz_arc / 2 * 0.7
                y_start = self.bsz_arc / 2 * 0.7
                ax.plot([x_start, x_start + scale_arcsec], [y_start, y_start],
                        color="black", lw=4)
                ax.text(
                    x_start + scale_arcsec / 2,
                    y_start + 0.1,
                    f"{scale_arcsec * self.arcsec_1 * 1000:.2f} kpc/h",
                    color="black",
                    ha="center",
                    va="bottom"
                )

            if i != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])

        # Colorbar
        cbar_ax = fig.add_axes([0.91, 0.12, 0.005, 0.75])
        fig.colorbar(im_main, cax=cbar_ax)

        # Layout and save
        os.makedirs(self.savefile, exist_ok=True)
        if output_name is None:
            output_name = f"{self.name}_kappa_all_types.png"
        out_path = os.path.join(self.savefile, output_name)

        # Leave room on the right for the colorbar
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        fig.savefig(out_path, dpi=100, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return out_path
    def _plot_kappa_with_xroots_and_observations(
        self,
        x_all,
        y_all,
        weights,
        sim_type,
        plot_kappa_func,
        kappa,
        mu,
        bsz, nnn,
        x_flip=True,
        zoom_size=0.05,
        inset_labels=["A", "B", "C"],
        inset_size="25%",
        save_path=None,
        vmin=-0.04,
        vmax=0.04
    ):
        from matplotlib import gridspec
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        object_lensing = self.obj
        image_positions_relative_to_L = {
            "A": (-object_lensing["y_imageA"], object_lensing["x_imageA"],
                object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
            "B": (-object_lensing["y_imageB"], object_lensing["x_imageB"],
                object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
            "C": (-object_lensing["y_imageC"], object_lensing["x_imageC"],
                object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
            "D": (-object_lensing["y_imageD"], object_lensing["x_imageD"],
                object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        }

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        weights = np.array(weights)

        obs_points = {
            label: ((-x if x_flip else x), y, x_err, y_err)
            for label, (x, y, x_err, y_err) in image_positions_relative_to_L.items()
        }

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03], wspace=0.1)
        ax = fig.add_subplot(gs[0, 0])

        im_fdm = plot_kappa_func(ax, kappa, mu, bsz, nnn,
                                vmin=vmin, vmax=vmax, text=sim_type)

        ax.scatter(x_all, y_all, s=5, c=weights, cmap='viridis', alpha=0.8, label="xroots (weighted)")

        for label, (x, y, xerr, yerr) in obs_points.items():
            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                        fmt='o', capsize=2, elinewidth=1, markersize=4,
                        color='red', label=f"Image {label}" if label == "A" else None)
            ax.text(x + 0.03, y + 0.03, label, fontsize=10, color='red')

        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)

        inset_loc_map = {"C": 'upper left', "A": 'upper right', "B": 'lower right'}

        for label in inset_labels:
            if label not in obs_points:
                continue
            x0, y0, xerr, yerr = obs_points[label]

            inset_ax = inset_axes(ax, width=inset_size, height=inset_size,
                                loc=inset_loc_map[label], borderpad=0)

            plot_kappa_func(inset_ax, kappa, mu, bsz, nnn, vmin=vmin, vmax=vmax, text=None)

            mask = (np.abs(x_all - x0) < zoom_size) & (np.abs(y_all - y0) < zoom_size)
            inset_ax.scatter(x_all[mask], y_all[mask], s=6, c=weights[mask], cmap='viridis', alpha=0.8)
            inset_ax.errorbar(x0, y0, xerr=xerr, yerr=yerr, fmt='o', color='red', capsize=2, elinewidth=1)
            inset_ax.text(x0 + 0.005, y0 + 0.005, label, fontsize=9, color='red')

            inset_ax.set_xlim(x0 - zoom_size, x0 + zoom_size)
            inset_ax.set_ylim(y0 - zoom_size, y0 + zoom_size)
            inset_ax.tick_params(axis='both', labelsize=10)

            if label == "C":
                inset_ax.yaxis.tick_right()
                inset_ax.yaxis.set_label_position("right")
            elif label == "B":
                inset_ax.xaxis.tick_top()
                inset_ax.xaxis.set_label_position("top")

            inset_ax.set_ylabel("")
            inset_ax.set_xlabel("")
            inset_ax.set_aspect('equal')

        cbar_ax = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(im_fdm, cax=cbar_ax)
        cbar.set_label(r"$\kappa_{\mathrm{eff,sub}}$")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.show()
    


class HaloAnalyzer_mass_bin(HaloAnalyzer):
    def __init__(self, name, sigma0=8000, w0=6, savefile="Data", globalfile=None):
        super().__init__(name, sigma0, w0, savefile, globalfile)
        self.mass_bins = []  # Initialized empty; set via set_mass_bins

    def set_mass_bins(self, n_bins, m_min, m_max, logscale=True):
        """
        Configure mass bin upper edges.

        Args:
            n_bins (int): Number of bins
            m_min (float): Minimum mass (inclusive)
            m_max (float): Maximum mass (inclusive)
            logscale (bool): Whether to use logarithmic spacing (default True)
        """
        if logscale:
            edges = np.logspace(np.log10(m_min), np.log10(m_max), n_bins+1)
        else:
            edges = np.linspace(m_min, m_max, n_bins+1)

        # Use upper edges in descending order to match prior behavior
        self.mass_bins = edges[1:][::-1]
        # self.mass_bins = [1e11, 1e10, 1e9]
        print(f"[INFO] Mass bins set: {self.mass_bins}")

    def process_halo_alpha_by_massbin(self, type_halo="CDM"):  # &&&!!!(2025-06-24 19:00)
        """
        For each mass bin (> upper edge), filter halos and save deflection maps.

        Args:
            type_halo (str): 'CDM', 'WDM', 'SIDM' or 'SIDM_col'
        """
        # Load halo list for the requested type
        if type_halo == "WDM":
            halofile = os.path.join(self.savefile, f"{self.name}_halolist_WDM.pkl")
        else:
            halofile = os.path.join(self.savefile, f"{self.name}_halolist_sigma0_{self.sigma0}_w0_{self.w0}.pkl")

        with open(halofile, 'rb') as f:
            halodata = pickle.load(f)

        half = self.bsz_arc / 2

        # Iterate over cumulative upper bounds
        for i, mass_max in enumerate(self.mass_bins[1:], start=1):

            # Filter: mass above bound and inside field of view
            subhalos = [h for h in halodata
                        if h['mass'] > mass_max
                        and -half <= h['position_x'] <= half
                        and -half <= h['position_y'] <= half]

            # Build multi-plane tnfw parameter list (shared for CDM/WDM/SIDM)
            tnfw_params = []
            for h in subhalos:
                if type_halo=="CDM" or type_halo=="WDM":
                    tau = 0.0
                elif type_halo=="SIDM_col":
                    tau = 1.08
                elif type_halo=="SIDM":
                    tau = h['tau']
                else:
                    raise ValueError("Unsupported type_halo: {}".format(type_halo))

                p = h['tnfw_params']

                tnfw_params.append({
                    'mass': h['mass']* cosmo.h, #Msun/h
                    'tnfw_params': {'rs': p['rs']/1000 * cosmo.h, 'rhos': p['rhos']*1000**3/ cosmo.h / cosmo.h}, #rs: Mpc/h, rhos: Msun *h^2 /Mpc^3 
                    'redshift': h['redshift'],
                    'tau': tau,
                    'position_x': h['position_x'],
                    'position_y': h['position_y'],
                    'ql': 1, 'pa': 0
                })
            if tnfw_params == []:
                continue  # Skip bins with no qualifying halos

            
            # Build model and compute deflection
            model = SIDM_parametric_Multiplane(
                tnfw_params=tnfw_params,
                bsz_arc=self.bsz_arc,
                dsx_arc=self.dsx_arc,
                zsource=self.zsource
            )
            alpha1, alpha2 = model._compute_grouped_alpha_maps(batch_size=30)

            # Save results
            label = f"lt{int(mass_max):.2e}"
            out_file = os.path.join(
                self.savefile,
                f"{self.name}_{type_halo}_{label}_alpha_mul_massbin.npz"
            )  
            np.savez(out_file, alpha1_sub=alpha1, alpha2_sub=alpha2)
             # Save corresponding redshift list
            redshift_file = os.path.join(
                self.savefile,
                f"{self.name}_{type_halo}_{label}_redshift_list_massbin.npy"
            )
            np.save(redshift_file, model.grouped_redshift_array)

            print(f"[MassBin>{mass_max:.1e}] Saved: {out_file}")

    def get_ray_trace_alpha_by_massbin(self, type_halo="CDM"):
        """
        Ray-trace alpha maps for each mass bin and save them.
        """
        for i, mass_max in enumerate(self.mass_bins[1:], start=1):
            label = f"lt{int(mass_max):.2e}"
            infile = os.path.join(
                self.savefile,
                f"{self.name}_{type_halo}_{label}_alpha_mul_massbin.npz"
            )
            outfile = os.path.join(
                self.savefile,
                f"{self.name}_{type_halo}_{label}_alpha_mul_massbin_ray.npz"
            )
            redshift_file = os.path.join(
                self.savefile,
                f"{self.name}_{type_halo}_{label}_redshift_list_massbin.npy"
            )
            global_file = os.path.join(
                self.globalfile,
                f"{self.name}_Global_alpha.npz"
            )
            if os.path.exists(infile):
                Get_multi_alpha(
                    self.obj, self.bsz_arc, self.nnn,
                    infile, redshift_file,
                    self.zlens, self.zsource,
                    filename_main=global_file,
                    output_file=outfile
                )
                print(f"[MassBin>{mass_max:.1e}] Ray saved: {outfile}")
            else:
                print(f"[MassBin>{mass_max:.1e}] No halos: copying Global_alpha...")
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                shutil.copy(global_file, outfile)
                print(f"Copied {os.path.basename(global_file)} -> {os.path.basename(outfile)}")

                

    def Simulation_WDM_massbin(self, ifgenerate=True):
        if ifgenerate:
            self.generate_halo_realization_WDM()
        self.process_halo_alpha_by_massbin(type_halo="WDM")
        self.get_ray_trace_alpha_by_massbin(type_halo="WDM")

    def Simulation_CDM_SIDM_massbin(self, ifgenerate=True):
        if ifgenerate:
            self.generate_halo_realization()
            self.convert_CDM_to_SIDM_halo_list()

        self.process_halo_alpha_by_massbin(type_halo="CDM")
        self.get_ray_trace_alpha_by_massbin(type_halo="CDM")

        self.process_halo_alpha_by_massbin(type_halo="SIDM_col")
        self.get_ray_trace_alpha_by_massbin(type_halo="SIDM_col")
    def Simulation_SIDM_massbin(self, ifgenerate=True):
        if ifgenerate:
            self.generate_halo_realization()
            self.convert_CDM_to_SIDM_halo_list()

        self.process_halo_alpha_by_massbin(type_halo="SIDM")
        self.get_ray_trace_alpha_by_massbin(type_halo="SIDM")
        

# Generate mock data

# Version sampling from distributions
class HaloAnalyzer_Mock_data(HaloAnalyzer):
    def __init__(self, name, dist_sampler, sigma0=8000, w0=6, savefile="Data", globalfile=None):
        self.name = name
        self.sigma0 = sigma0
        self.w0 = w0

        # === Configure device ===
        gpu = self._find_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        print(f"⚙️ Using GPU for JAX: {gpu}")

        # === Sample system parameters from distribution ===
        self.obj = dist_sampler.sample_lens_system()
        
        # === Configure cosmology ===
        self.cosmo_astropy = FlatwCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.cosmo = cosmology.setCosmology('planck18')
        
        self.zlens = round(self.obj["zlens"], 2)
        self.zsource = round(self.obj["zsource"], 2)
        self.Da_lens = Da0(self.zlens)
        self.Da_src = Da0(self.zsource)
        self.Da_ls = Da20(self.zlens, self.zsource)

        self.arcsec_1 = 1 / apr * self.Da_lens
        self.thetaE = self.obj["thetaE"]
        self.thetaE_rad = arcsec_to_Rad(self.thetaE)
        self.thetaE_mpc_h = self.thetaE * self.arcsec_1
        self.sigma_g = calculate_velocity_dispersion(self.thetaE_rad, self.Da_src, self.Da_ls)

        self.halomass = Sigma_g_to_Mvir(self.thetaE_mpc_h, self.sigma_g, self.zlens) / self.cosmo.h
        print(f"Main halo mass: {self.halomass:.2e} M_sun")

        self.sigma_crit = SigmaCrit(self.zlens, self.zsource)
        nfwpara = NFWParam(cosmo)
        self.r200, self.rhos, self.Concentration, self.rs = nfwpara.nfw_Mz(
            self.halomass * self.cosmo.h, self.zlens)

        self.bsz_arc = 4 * self.thetaE
        self.nnn = self.obj.get("nnn", 1500)  # Default value if missing
        self.dsx_arc = self.bsz_arc / self.nnn

        self.savefile = savefile
        self.globalfile = globalfile if globalfile else savefile

class DistributionSampler:
    def __init__(self):
        # Primary distributions for θ_E, q, φ
        self.thetaE_dist = lambda: np.random.normal(1.2, 0.3)  # arcsec
        self.q_dist = lambda: np.clip(np.random.normal(0.7, 0.1), 0.3, 1.0)
        self.q_sigma_dist = lambda: 0.05  # Fixed or adjustable
        self.phi_dist = lambda: np.random.uniform(0, 180)  # deg
        self.phi_sigma_dist = lambda: 5.0  # deg

        # External shear and uncertainty
        self.gamma_ext_dist = lambda: np.random.exponential(0.03)
        self.gamma_ext_sigma_dist = lambda: 0.01
        self.phi_ext_dist = lambda: np.random.uniform(0, 180)  # deg
        self.phi_ext_sigma_dist = lambda: 10.0  # deg

        # Mass profile slope
        self.gamma_slope_dist = lambda: np.random.normal(2.0, 0.1)
        self.gamma_slope_sigma_dist = lambda: 0.05

        # Redshift
        self.zlens_dist = lambda: np.random.uniform(0.2, 0.6)
        self.zsource_dist = lambda zl: np.random.uniform(zl + 0.3, 3.0)

    def sample_lens_system(self, name=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        zlens = self.zlens_dist()
        zsource = self.zsource_dist(zlens)

        thetaE = self.thetaE_dist()
        q = self.q_dist()
        phi_lens = self.phi_dist()
        gamma_ext = self.gamma_ext_dist()
        phi_ext = self.phi_ext_dist()

        return {
            "name": name or f"mock_{seed}",
            "thetaE": thetaE,
            "q": q,
            "q_sigma": self.q_sigma_dist(),
            "phi_lens": phi_lens,
            "phi_lens_sigma": self.phi_sigma_dist(),
            "Gamma": self.gamma_slope_dist(),
            "Gamma_sigma": self.gamma_slope_sigma_dist(),
            "gamma_external": gamma_ext,
            "gamma_external_sigma": self.gamma_ext_sigma_dist(),
            "phi_external": phi_ext,
            "phi_external_sigma": self.phi_ext_sigma_dist(),
            "zlens": zlens,
            "zsource": zsource,
            "nnn": 1500
        }
