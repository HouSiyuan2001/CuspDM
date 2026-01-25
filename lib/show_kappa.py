from Lensing_tool import make_c_coor,alphas_to_mu
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax
import jax.numpy as jnp

from matplotlib.colors import LinearSegmentedColormap


jax.config.update("jax_enable_x64", True)  # Magnification needs 64-bit precision


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from Lensing_tool import adding_external_to_alpha

# Define the RGB values based on the color chart in the image
colors = [
    (30/255, 70/255, 110/255),
    (55/255, 103/255, 149/255),
    (82/255, 143/255, 173/255),
    (114/255, 188/255, 213/255),
    (170/255, 220/255, 224/255),
    (255/255, 255/255, 255/255),
    (255/255, 230/255, 183/255),
    (255/255, 208/255, 111/255),
    (247/255, 170/255, 88/255),
    (239/255, 138/255, 71/255),
    (231/255, 98/255, 84/255)
]



cmap256 = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
params = {
    'axes.labelsize': 24,
    'font.size': 22,
    'lines.linewidth': 1.5,
    'legend.fontsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'text.usetex': False,
    'figure.figsize': [8, 6],
    'axes.linewidth': 1.5
}
plt.rcParams.update(params)

def get_kappa(datafile,bsz_arc,nnn, external_kappa =None):
    # Assumes the saved file is named result_af.npz
    bsz_deg = bsz_arc / 3600.0
    dsx_arc = bsz_arc/ nnn

    xi2, xi1 = make_c_coor(bsz_arc, nnn)
    data = np.load(datafile)

    af1 = data["alpha1_global"]
    af2 = data["alpha2_global"]
    if external_kappa is not None:
        af1,af2 = adding_external_to_alpha(af1,af2,xi1,xi2, external_kappa)

    yi1, yi2, mu_global, kappa_, gamma1_gobal, gamma2_gobal = alphas_to_mu(af1, af2 , dsx_arc, xi1, xi2)

    gamma_global = np.sqrt(gamma1_gobal**2 + gamma2_gobal**2)
    lambdat_global = 1 - kappa_ - gamma_global
    lambdar_global = 1 - kappa_ + gamma_global
    return kappa_, yi1, yi2 ,mu_global,lambdat_global,gamma_global

def get_kappa_rescale(datafile,bsz_arc,kappa_main):
    # Assumes the saved file is named result_af.npz
    nnn = 2000
    bsz_deg = bsz_arc / 3600.0
    dsx_arc = bsz_arc/ nnn

    xi2, xi1 = make_c_coor(bsz_arc, nnn)
    data = np.load(datafile)

    af1 = data["alpha1_global"]
    af2 = data["alpha2_global"]
    yi1, yi2, mu_global, kappa_multi, gamma1_gobal, gamma2_gobal = alphas_to_mu(af1, af2 , dsx_arc, xi1, xi2)
    rescale_factor = kappa_main.sum()/kappa_multi.sum()
    print(f"rescale_factor: {rescale_factor}")
    alpha1_global =  af1 * rescale_factor
    alpha2_global = af2 * rescale_factor
    yi1, yi2, mu_global, kappa_global, gamma1_gobal, gamma2_gobal = alphas_to_mu(alpha1_global, alpha2_global, dsx_arc, xi1, xi2)
    gamma_global = np.sqrt(gamma1_gobal**2 + gamma2_gobal**2)
    lambdat_global = 1 - kappa_global - gamma_global
    lambdar_global = 1 - kappa_global + gamma_global



    return kappa_global, yi1, yi2 ,mu_global

def get_main_plan_kappa(bsz_arc,filename_main=None, nnn=2000):
    bsz_deg = bsz_arc / 3600.0
    dsx_arc = bsz_arc/ nnn

    xi2, xi1 = make_c_coor(bsz_arc, nnn)
    if filename_main is not None:
        data_file_main = jnp.load(filename_main)
        alpha1_global = data_file_main['alpha1_global']
        alpha2_global = data_file_main['alpha2_global']
    else:
        raise ValueError("filename_main is None, please provide a valid filename_main.")


    yi1, yi2, mu_global, kappa_, gamma1_gobal, gamma2_gobal = alphas_to_mu(alpha1_global, alpha2_global , dsx_arc, xi1, xi2)


    return kappa_  

def get_kappa_sub_mean(filename_sub,redshift_list_file,z_main,bsz_arc,if_no_mainplane = False):
    data = jnp.load(filename_sub)
    alpha1_sub = data['alpha1_sub']
    alpha2_sub = data['alpha2_sub']

    alpha1_sub = jnp.array(alpha1_sub)
    alpha2_sub = jnp.array(alpha2_sub)
    redshift_list = np.load(redshift_list_file)
    nnn = 2000
    bsz_deg = bsz_arc / 3600.0
    dsx_arc = bsz_arc/ nnn

    xi2, xi1 = make_c_coor(bsz_arc, nnn)
    if if_no_mainplane:
        # Find index closest to z_main
        i = int(np.argmin(np.abs(redshift_list - z_main)))
        # Drop the ith lens plane
        mask = jnp.arange(alpha1_sub.shape[0]) != i
        alpha1_selected = alpha1_sub[mask]
        alpha2_selected = alpha2_sub[mask]
        # Sum remaining planes
        alpha1_global = jnp.sum(alpha1_selected, axis=0)
        alpha2_global = jnp.sum(alpha2_selected, axis=0)

        del alpha1_sub,alpha2_sub,alpha1_selected,alpha2_selected
    else:
        # Sum all planes
        alpha1_global = jnp.sum(alpha1_sub, axis=0)
        alpha2_global = jnp.sum(alpha2_sub, axis=0)
        del alpha1_sub,alpha2_sub

    yi1, yi2, mu_global, kappa_, gamma1_gobal, gamma2_gobal = alphas_to_mu(alpha1_global, alpha2_global , dsx_arc, xi1, xi2)


    return jnp.mean(kappa_ )

def get_kappa_sub_mean_multi(datafile,bsz_arc):
    # Assumes the saved file is named result_af.npz
    nnn = 2000
    bsz_deg = bsz_arc / 3600.0
    dsx_arc = bsz_arc/ nnn

    xi2, xi1 = make_c_coor(bsz_arc, nnn)
    data = np.load(datafile)

    af1 = data["alpha1_global"]
    af2 = data["alpha2_global"]
    yi1, yi2, mu_global, kappa_, gamma1_gobal, gamma2_gobal = alphas_to_mu(af1, af2 , dsx_arc, xi1, xi2)

    return jnp.mean(kappa_ )

def plot_kappa(ax,kappa_eff, mu,bsz_arc,nnn, vmin = -0.05, vmax = 0.05,text=None, color_use='black',plot_reverse_coodinates=False,cmap_use = cmap256):
    
    bsz_deg = bsz_arc / 3600.0
    dsx_arc = bsz_arc/ nnn
    if plot_reverse_coodinates:
         xi1, xi2 = make_c_coor(bsz_arc, nnn)
    else:
        xi2, xi1 = make_c_coor(bsz_arc, nnn)
    c = ax.pcolormesh(xi2, xi1, kappa_eff, cmap=cmap_use, vmin=vmin, vmax=vmax)

    ax.contour(xi2, xi1, mu,  colors='black', linewidths=1.5, alpha=0.9, linestyles='-')

    ax.set_xlabel(r"X (arcsec)")
    ax.set_ylabel(r"Y (arcsec)")

    ax.set_aspect('equal')
    if text is not None:
        ax.text(
            0.04, 0.04, text,
            transform=ax.transAxes,
            # fontsize=22,
            color=color_use,
            ha='left',
            va='bottom',
            # bbox=dict(facecolor='white', edgecolor=color_use, boxstyle='round,pad=0.3', alpha=0.5)
        )
    return c
