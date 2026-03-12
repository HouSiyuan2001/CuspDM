import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import pickle

import jax
import jax.numpy as jnp
# Magnification computation needs 64-bit precision.
jax.config.update("jax_enable_x64", True)


from matplotlib.colors import LinearSegmentedColormap, Normalize
from notion_client import Client
from astropy.cosmology import FlatwCDM

# Local modules
sys.path.append("lib")
from Lensing_tool import (
    make_c_coor
)
from show_kappa import (
    get_main_plan_kappa, get_kappa, plot_kappa
)
from notion import get_lens_data_by_name

# Notion config
NOTION_TOKEN = "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"
DATABASE_ID = "2fffc9067a7481d1a61ccfc1e40ccfd0"
rgb_values = [
    [231, 98, 84],
    [239, 138, 71],
    [247, 170, 88],
    [255, 208, 111],
    [255, 230, 183],
    [170, 220, 224],
    [114, 188, 213],
    [82, 143, 173],
    [55, 103, 149],
    [30, 70, 110]
]

# Normalize RGB values to the [0, 1] range.
colors = [[x / 255 for x in rgb] for rgb in rgb_values]
# Color palette
COLOR_LIST = [
    (30/255, 70/255, 110/255), (55/255, 103/255, 149/255),
    (82/255, 143/255, 173/255), (114/255, 188/255, 213/255),
    (170/255, 220/255, 224/255), (255/255, 230/255, 183/255),
    (255/255, 208/255, 111/255), (247/255, 170/255, 88/255),
    (239/255, 138/255, 71/255), (231/255, 98/255, 84/255)
]
colors_line = [
    (55/255, 103/255, 149/255),    # #c7522a
    (114/255, 188/255, 213/255),  # #e5c185
    (247/255, 170/255, 88/255),  # #fbf2c4
    (231/255, 98/255, 84/255)     # #008585
]
# Global plotting style
def setup_plotting():
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
    return LinearSegmentedColormap.from_list("custom_cmap", COLOR_LIST, N=256)

# Ensure directory exists.
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Load lens data from Notion.
def load_lens_params(name):
    notion = Client(auth=NOTION_TOKEN)
    resp = notion.databases.query(database_id=DATABASE_ID, page_size=100)
    return get_lens_data_by_name(resp, name)

# Load multi-plane alpha and compute kappa_multi, yi, mu, lambdat.
def load_kappa_multi(name, halo_type, thetaE, filename,nnn=2000,bsz = None):
    if bsz == None:
        bsz = 4 * thetaE
    xi2, xi1 = make_c_coor(bsz, nnn)
    data_file = f"{filename}/{name}_{halo_type}_alpha_mul_ray.npz"
    kappa_multi, yi1, yi2, mu_global, lambdat_global,_ = get_kappa(data_file, bsz, nnn)
    return {
        'bsz': bsz, 'nnn': nnn,
        'xi1': xi1, 'xi2': xi2,
        'yi1': yi1, 'yi2': yi2,
        'mu': mu_global,
        'lambdat': lambdat_global,
        'kappa_multi': kappa_multi
    }
# Plot kappa difference.
def plot_kappa_diff(cfg, halo_type, out_dir="images",global_fime = "Data",simulation_file = "Simulation"):
    ensure_dir(out_dir)
    name = cfg['name']
    params = load_kappa_multi(name, halo_type, cfg['thetaE'], filename=simulation_file,nnn= cfg['nnn'] )
    k_main = get_main_plan_kappa(params['bsz'], filename_main=f"{global_fime}/{name}_Global_alpha.npz", nnn = cfg['nnn'])
    diff = params['kappa_multi']  # Using mu as a kappa placeholder? Should be kappa_multi - k_main.

    fig, ax = plt.subplots()
    c = plot_kappa(ax, diff - k_main, params['mu'], params['bsz'], cfg['nnn'], text=halo_type)
    fig.colorbar(c, ax=ax)
    fig.savefig(os.path.join(out_dir, f"kappa_diff_{halo_type}.png"), dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

def plot_image_plane_with_points(global_kappa, points, halo_type, out_dir):
    xi1 = global_kappa['xi1']
    xi2 = global_kappa['xi2']
    lambdat = global_kappa['lambdat']

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.contour(xi2, xi1, lambdat, levels=[0.0], colors='red', linewidths=2, alpha=0.9, linestyles='-', zorder=4)
    ax.set_aspect('equal')
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")

    # Distance from origin for color mapping.
    all_points = np.array([pinfo["point"] for pinfo in points.values()])
    radii = np.linalg.norm(all_points, axis=1)
    norm = mcolors.Normalize(vmin=0, vmax=radii.max())
    cmap = cm.get_cmap('coolwarm')

    # Scatter image points, colored by distance from origin.
    for pinfo in points.values():
        xroots = pinfo["xroots"]
        dist = np.linalg.norm(pinfo["point"])
        color = cmap(norm(dist))
        for xroot in xroots:
            circ = plt.Circle((xroot[1], xroot[0]), 0.05, color=color, fill=True, zorder=3)
            ax.add_patch(circ)

    ax.text(0.95, 0.95, f"{halo_type}", transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    fig.savefig(os.path.join(out_dir, f"rcusp_image_plane_{halo_type}.png"), dpi=300)
    plt.close(fig)


def plot_lens_plane_with_points(global_kappa, points, halo_type, out_dir, xlim=None, ylim=None):
    yi1 = global_kappa['yi1']
    yi2 = global_kappa['yi2']
    lambdat = global_kappa['lambdat']

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.contour(yi2, yi1, lambdat, levels=[0.0], colors='green', linewidths=2, alpha=0.9, linestyles='-', zorder=4)
    ax.set_aspect('equal')
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    # Distance from origin for color mapping.
    all_points = np.array([pinfo["point"] for pinfo in points.values()])
    radii = np.linalg.norm(all_points, axis=1)
    norm = mcolors.Normalize(vmin=0, vmax=radii.max())
    cmap = cm.get_cmap('coolwarm')

    # Scatter source points.
    for pinfo in points.values():
        point = pinfo["point"]
        dist = np.linalg.norm(point)
        color = cmap(norm(dist))
        ax.scatter(point[0], point[1], color=color, s=50, zorder=3)

    ax.text(0.95, 0.95, f"{halo_type}", transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    fig.savefig(os.path.join(out_dir, f"rcusp_lens_plane_{halo_type}.png"), dpi=300)
    plt.close(fig)

def extract_caustic_crop_region(global_kappa, center=(0.0, 0.0), threshold=1.0, margin=0.2):
    """
    Extract the caustic contour near the origin and return suggested xlim/ylim.
    """
    yi1 = global_kappa['yi1']
    yi2 = global_kappa['yi2']
    lambdat = global_kappa['lambdat']
    
    # Compute contour.
    fig, ax = plt.subplots()
    contour = ax.contour(yi2, yi1, lambdat, levels=[0.0])
    plt.close(fig)  # We only need the contour path data.

    caustic_paths = []
    center = np.array(center)

    for segment in contour.allsegs[0]:
        dists = np.linalg.norm(segment - center, axis=1)
        if np.any(dists < threshold):
            caustic_paths.append(segment)

    if not caustic_paths:
        raise RuntimeError("No central caustic region found; increase threshold.")

    # Compute bounds from all segment points.
    all_points = np.vstack(caustic_paths)
    xmin, xmax = all_points[:, 0].min(), all_points[:, 0].max()
    ymin, ymax = all_points[:, 1].min(), all_points[:, 1].max()

    # Add margin.
    xlim = (xmin - margin, xmax + margin)
    ylim = (ymin - margin, ymax + margin)

    return xlim, ylim

def plot_rcusp_vs_angle(points, halo_type, out_dir, obs_angle_deg=None, obs_rcusp=None, cal_inverse=False):
    angles, rcusps, dists = [], [], []
    for info in points.values():
        ang = info['Rcusp_info'].get('cusp_angle_deg')
        if ang is None:
            continue
        angles.append(ang)
        rcusps.append(info['Rcusp_info']['R_cusp'])
        dists.append(np.linalg.norm(info['point']))
    
    angles = np.array(angles)
    rcusps = np.array(rcusps)
    dists = np.array(dists)
    norm = Normalize(vmin=dists.min(), vmax=dists.max())

    fig = plt.figure()
    sc = plt.scatter(angles, rcusps, c=dists, cmap='coolwarm', norm=norm)
    plt.colorbar(sc, label='Distance')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Rcusp')
    plt.text(0.95, 0.95, f"{halo_type}", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=20,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    if obs_angle_deg is not None and obs_rcusp is not None:
        plt.scatter(obs_angle_deg, obs_rcusp, color='k', marker='*', s=150)

    # Bin and compute the 30th/70th Rcusp percentiles in each bin.
    bins = np.linspace(20, 130, 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    perc_30, perc_70 = [], []

    for i in range(len(bins) - 1):
        mask = (angles >= bins[i]) & (angles < bins[i+1])
        if np.any(mask):
            perc_30.append(np.percentile(rcusps[mask], 30))
            perc_70.append(np.percentile(rcusps[mask], 70))
        else:
            perc_30.append(np.nan)
            perc_70.append(np.nan)

    # Plot percentile lines (skip NaNs).
    bin_centers = np.array(bin_centers)
    perc_30 = np.array(perc_30)
    perc_70 = np.array(perc_70)
    valid_30 = ~np.isnan(perc_30)
    valid_70 = ~np.isnan(perc_70)
    plt.plot(bin_centers[valid_30], perc_30[valid_30], linestyle='--', color='k')
    plt.plot(bin_centers[valid_70], perc_70[valid_70], linestyle='--', color='k')

    plt.ylim(0, 0.7)
    plt.xlim(20, 130)
    if cal_inverse:
        fig.savefig(os.path.join(out_dir, f"rcusp_angle_inverse_{halo_type}.png"), dpi=300, bbox_inches='tight', pad_inches=0.2)
    else:
        fig.savefig(os.path.join(out_dir, f"rcusp_angle_{halo_type}.png"), dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

def plot_phi_vs_Rcusp(combined_mpc_dict):
    phi_list = []
    Rcusp_list = []
    Rcusp_err_list = []
    size_list = []
    color_list = []
    alpha_list = []
    names = []

    for name, data in combined_mpc_dict.items():
        phi = data.get("phi")
        Rcusp = data.get("Rcusp")
        Rcusp_sigma = data.get("sigma", 0.0)
        theta_kpc = data.get("theta_kpc")
        axis_type = data.get("axis_type")
        is_paper = data.get("is_paper")  # Defaults to True.

        if phi is None or Rcusp is None or theta_kpc is None:
            continue

        phi_list.append(phi)
        Rcusp_list.append(Rcusp)
        Rcusp_err_list.append(Rcusp_sigma)
        size_list.append(theta_kpc)
        names.append(name)

        # Set color.
        if axis_type == "shout":
            base_color = "blue"
        elif axis_type == "long":
            base_color = "red"
        else:
            base_color = "gray"

        color_list.append(base_color)

        # Set alpha.
        alpha_list.append(1 if is_paper else 0.5)

    # Convert colors to RGBA with alpha.
    from matplotlib.colors import to_rgba
    rgba_colors = [to_rgba(c, alpha=a) for c, a in zip(color_list, alpha_list)]

    # Plot.
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        phi_list, Rcusp_list,
        yerr=Rcusp_err_list,
        fmt='o', markersize=0, elinewidth=1.5, capsize=4, color='k'
    )
    plt.scatter(
        phi_list,
        Rcusp_list,
        s=[s * 30 for s in size_list],
        c=rgba_colors,
        edgecolors='w'
    )

    for i, name in enumerate(names):
        if Rcusp_list[i] < 0.6:
            plt.text(phi_list[i], Rcusp_list[i] + 0.01, name, fontsize=9, ha='center', va='bottom')

        plt.xlabel("phi")
    plt.ylabel("Rcusp")
    plt.grid(True)
    plt.ylim(0, 0.6)
    plt.tight_layout()

    # Add legend.
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                      markersize=8, label='Shout axis'),
        mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                      markersize=8, label='Long axis'),
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                      markersize=8, label='Other')
    ]
    plt.legend(handles=legend_elements, title='Axis Type', loc='upper right')

    plt.show()


def main_plot():
    # Initialize plotting style and output directory.
    cmap = setup_plotting()
    out_dir = "images"
    ensure_dir(out_dir)

    # Load lens parameters.
    name = "PSJ0147+4630"
    params = load_lens_params(name)
    cfg = {
        'name': name,
        'zlens': params['zlens'],
        'zsource': params['zsource'],
        'thetaE': params['thetaE'],
        'x_source_guess': params.get('x_source_guess', 0.0),
        'y_source_guess': params.get('y_source_guess', 0.0)
    }

    # Compute and plot each halo type.
    for ht in ["CDM", "SIDM"]:
        # Load multi-plane deflection and compute kappa and coordinates.
        global_kappa = load_kappa_multi(name=cfg['name'], halo_type=ht, thetaE=cfg['thetaE'])

        # Compute Rcusp distribution and save to file.
        points = compute_Rcusp_distribution(
            global_kappa['mu'],
            cfg['x_source_guess'], cfg['y_source_guess'],
            global_kappa['xi1'], global_kappa['xi2'],
            global_kappa['yi1'], global_kappa['yi2'],
            desired_points=1000
        )
        rcusp_pkl = f"Data/{ht}_Rcusp_distribution.pkl"
        with open(rcusp_pkl, 'wb') as f:
            pickle.dump(points, f)

        # Extract a crop window around the central caustic.
        xlim, ylim = extract_caustic_crop_region(
            global_kappa,
            center=(cfg['x_source_guess'], cfg['y_source_guess']),
            threshold=1.0,
            margin=0.2
        )

        # Plot kappa difference.
        plot_kappa_diff(cfg, ht, out_dir)

        # Plot Rcusp points on the image plane.
        plot_image_plane_with_points(global_kappa, points, ht, out_dir)

        # Plot Rcusp points on the lens plane.
        plot_lens_plane_with_points(global_kappa, points, ht, out_dir, xlim=xlim, ylim=ylim)

        # Plot Rcusp vs opening angle.
        plot_rcusp_vs_angle(points, ht, out_dir)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_combined_Rcusp_all_models(base_dir, obj_name, sim_type_list, total_runs, ob_Rcusp):
    
    plt.figure(figsize=(12, 8))
    
    for idx, sim_type in enumerate(sim_type_list):
        all_Rcusp = []
        all_weights = []

        for i in range(1, total_runs + 1):
            rcusp_file = os.path.join(base_dir, f"{obj_name}/{i}/{obj_name}_{sim_type}_{i}_Rcusp.npz")
            if not os.path.exists(rcusp_file):
                continue
            data = np.load(rcusp_file)
            Rcusp_arr = data["Rcusp_arr"]
            weights = data["weights"]
            all_Rcusp.append(Rcusp_arr)
            all_weights.append(weights * len(weights))

        if not all_Rcusp:
            print(f"Model {sim_type} has no valid data, skipping.")
            continue

        Rcusp_all = np.concatenate(all_Rcusp)
        weights_all = np.concatenate(all_weights)

        kde = gaussian_kde(Rcusp_all, weights=weights_all)
        x_eval = np.linspace(min(Rcusp_all), max(Rcusp_all), 500)
        pdf = kde(x_eval)
        pdf /= np.trapz(pdf, x_eval)

        plt.plot(x_eval, pdf, label=sim_type, color=colors_line[idx])
    
    plt.axvline(ob_Rcusp, color='gray', linestyle='--', label=fr'Observed $R_{{\rm cusp}} = {ob_Rcusp:.3f}$')
    plt.xlabel(r'$R_{\rm cusp}$')
    plt.ylabel('Weighted PDF')
    plt.title(f'{obj_name}')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_combined_Rcusp_multi_objects(base_dir, simu_obj_list, response, sim_type_list, total_runs):
    n_obj = len(simu_obj_list)
    n_cols = 4
    n_rows = (n_obj + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows), sharex=False, sharey=False)
    axes = axes.flatten()

    for obj_idx, simu_obj in enumerate(simu_obj_list):
        ax = axes[obj_idx]
        object_lensing = get_lens_data_by_name(response, simu_obj)
        ob_Rcusp = object_lensing["Rcusp_use"]

        for idx, sim_type in enumerate(sim_type_list):
            all_Rcusp = []
            all_weights = []

            for i in range(1, total_runs + 1):
                rcusp_file = os.path.join(base_dir, f"{simu_obj}/{i}/{simu_obj}_{sim_type}_{i}_Rcusp.npz")
                if not os.path.exists(rcusp_file):
                    continue
                data = np.load(rcusp_file)
                Rcusp_arr = data["Rcusp_arr"]
                weights = data["weights"]
                all_Rcusp.append(Rcusp_arr)
                all_weights.append(weights * len(weights))  # Correct for sample size.

            if not all_Rcusp:
                print(f"Model {sim_type} has no valid data, skipping {simu_obj}.")
                continue

            Rcusp_all = np.concatenate(all_Rcusp)
            weights_all = np.concatenate(all_weights)

            kde = gaussian_kde(Rcusp_all, weights=weights_all)
            x_eval = np.linspace(min(Rcusp_all), max(Rcusp_all), 500)
            pdf = kde(x_eval)
            pdf /= np.trapz(pdf, x_eval)

            ax.plot(x_eval, pdf, label=sim_type, color=colors_line[idx % len(colors_line)],linewidth = 2)

        # Mark observed value.
        ax.axvline(ob_Rcusp, color='gray', linestyle='--')
        ax.text(ob_Rcusp*1.1, ax.get_ylim()[1]*0.5, f'{ob_Rcusp:.3f}', color='gray', fontsize=15, ha='left', va='top')

        ax.set_title(simu_obj)
        ax.set_xlabel(r'$R_{\rm cusp}$')
        ax.set_ylabel('PDF')
        # ax.grid(True)

    # Remove extra subplots.
    for i in range(n_obj, len(axes)):
        fig.delaxes(axes[i])

    # Put the legend only on the last subplot.
    handles, labels = ax.get_legend_handles_labels()
    axes[min(n_obj - 1, len(axes) - 1)].legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_fdm_with_xroots_and_observations(
    analyzer,
    i,
    x_all,
    y_all,
    weights,
    sim_type,
    plot_kappa_func,
    x_flip=True,
    zoom_size=0.05,
    inset_labels=["A", "B", "C"],
    inset_size="42%",  # Control inset size.
    save_path=None,
    savefile     = "Simulation",
    vmin = -0.02,
    vmax = 0.02
):
    object_lensing = analyzer.obj
    image_positions_relative_to_L = {
        "A": (object_lensing["x_imageA"], object_lensing["y_imageA"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "B": (object_lensing["x_imageB"], object_lensing["y_imageB"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "C": (object_lensing["x_imageC"], object_lensing["y_imageC"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "D": (object_lensing["x_imageD"], object_lensing["y_imageD"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
    }
    thetaE  = analyzer.obj["thetaE"]
    nnn     = analyzer.obj["nnn"]
    simu_obj = analyzer.obj["name"]
    source_dir   = os.path.join(savefile, simu_obj, str(i))
    global_file  = os.path.join(savefile, simu_obj)

    params = load_kappa_multi(simu_obj, sim_type, thetaE, filename=source_dir, nnn=nnn)

    k_main         = get_main_plan_kappa(params['bsz'],
                                         filename_main=f"{global_file}/{simu_obj}_Global_alpha.npz", nnn=nnn)
    kappa_fdm = params["kappa_multi"] - k_main


    x_all = np.array(x_all)
    y_all = np.array(y_all)
    weights = np.array(weights)

    # Transform observed point coordinates.
    obs_points = {
        label: ((-x if x_flip else x), y, x_err, y_err)
        for label, (x, y, x_err, y_err) in image_positions_relative_to_L.items()
    }

    # Main plot.
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])

    bsz = params['bsz']
    nnn = params['mu'].shape[0]

    # Use plot_kappa for the background.
    im_fdm = plot_kappa_func(ax, kappa_fdm,
                             params['mu'], bsz, nnn,
                             vmin=vmin, vmax=vmax, text=sim_type)

    # Color points by weights.
    sc = ax.scatter(x_all, y_all, s=5, c=weights, cmap='viridis', alpha=0.8, label="xroots (weighted)")

    for label, (x, y, xerr, yerr) in obs_points.items():
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt='o', capsize=2, elinewidth=1, markersize=4,
                    color='red', label=f"Image {label}" if label == "A" else None)
        ax.text(x + 0.03, y + 0.03, label, fontsize=10, color='red')

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Inset positions can be adjusted manually.
    # Upper-left, upper-right, lower-right positions (relative to main axes).
    # Corner mapping: A upper-left, B upper-right, C lower-right.
    inset_loc_map = {
        "C": 'upper left',
        "A": 'upper right',
        "B": 'lower right'
    }

    for label in inset_labels:
        if label not in obs_points:
            continue
        x0, y0, xerr, yerr = obs_points[label]

        inset_ax = inset_axes(
            ax,
            width=inset_size, height=inset_size,
            loc=inset_loc_map[label],
            borderpad=0  # Add padding by changing to 0.02-0.1 inches.
        )
    
        # Use the same plot_kappa for the background.
        plot_kappa_func(inset_ax, kappa_fdm,
                        params['mu'], bsz, nnn,
                        vmin=vmin, vmax=vmax, text=None)

        mask = (np.abs(x_all - x0) < zoom_size) & (np.abs(y_all - y0) < zoom_size)
        inset_ax.scatter(x_all[mask], y_all[mask],
                         s=6, c=weights[mask], cmap='viridis', alpha=0.8)
        inset_ax.errorbar(x0, y0, xerr=xerr, yerr=yerr, fmt='o', color='red', capsize=2, elinewidth=1)
        inset_ax.text(x0 + 0.005, y0 + 0.005, label, fontsize=9, color='red')

        inset_ax.set_xlim(x0 - zoom_size, x0 + zoom_size)
        inset_ax.set_ylim(y0 - zoom_size, y0 + zoom_size)
        inset_ax.tick_params(axis='both', labelsize=10)

        # Set tick/label positions.
        if label == "C":  # Upper-left -> y-axis on the right.
            inset_ax.yaxis.tick_right()
            inset_ax.yaxis.set_label_position("right")
        elif label == "B":  # Lower-right -> x-axis on the top.
            inset_ax.xaxis.tick_top()
            inset_ax.xaxis.set_label_position("top")
        # Default A: no change.

        inset_ax.set_ylabel("")
        inset_ax.set_xlabel("")
        inset_ax.set_aspect('equal')

    # Colorbar.
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im_fdm, cax=cbar_ax)
    cbar.set_label(r"$\kappa_{\mathrm{eff,sub}}$")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
def plot_fdm_with_xroots_and_observations_main(
    analyzer,
    x_all,
    y_all,
    weights,
    sim_type,
    plot_kappa_func,
    x_flip=True,
    zoom_size=0.05,
    inset_labels=["A", "B", "C"],
    inset_size="42%",  # Control inset size.
    save_path=None,
    savefile     = "Simulation",
    vmin = -0.02,
    vmax = 0.02
):
    object_lensing = analyzer.obj
    image_positions_relative_to_L = {
        "A": (object_lensing["x_imageA"], object_lensing["y_imageA"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "B": (object_lensing["x_imageB"], object_lensing["y_imageB"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "C": (object_lensing["x_imageC"], object_lensing["y_imageC"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "D": (object_lensing["x_imageD"], object_lensing["y_imageD"],
            object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
    }
    nnn     = analyzer.obj["nnn"]

    data_file = f'{analyzer.globalfile}/{analyzer.obj["name"]}_Global_alpha.npz'
    kappa_multi, yi1, yi2, mu_global, lambdat_global,_ = get_kappa(data_file, analyzer.bsz_arc, analyzer.nnn)
    kappa_fdm = kappa_multi


    x_all = np.array(x_all)
    y_all = np.array(y_all)
    weights = np.array(weights)

    # Transform observed point coordinates.
    obs_points = {
        label: ((-x if x_flip else x), y, x_err, y_err)
        for label, (x, y, x_err, y_err) in image_positions_relative_to_L.items()
    }

    # Main plot.
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])

    bsz = analyzer.bsz_arc

    # Use plot_kappa for the background.
    im_fdm = plot_kappa_func(ax, kappa_fdm,
                             mu_global, bsz, nnn,
                             vmin=vmin, vmax=vmax, text=sim_type)

    # Color points by weights.
    sc = ax.scatter(x_all, y_all, s=5, c=weights, cmap='viridis', alpha=0.8, label="xroots (weighted)")

    for label, (x, y, xerr, yerr) in obs_points.items():
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt='o', capsize=2, elinewidth=1, markersize=4,
                    color='red', label=f"Image {label}" if label == "A" else None)
        ax.text(x + 0.03, y + 0.03, label, fontsize=10, color='red')

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Inset positions can be adjusted manually.
    # Upper-left, upper-right, lower-right positions (relative to main axes).
    # Corner mapping: A upper-left, B upper-right, C lower-right.
    inset_loc_map = {
        "C": 'upper left',
        "A": 'upper right',
        "B": 'lower right'
    }

    for label in inset_labels:
        if label not in obs_points:
            continue
        x0, y0, xerr, yerr = obs_points[label]

        inset_ax = inset_axes(
            ax,
            width=inset_size, height=inset_size,
            loc=inset_loc_map[label],
            borderpad=0  # Add padding by changing to 0.02-0.1 inches.
        )
    
        # Use the same plot_kappa for the background.
        plot_kappa_func(inset_ax, kappa_fdm,
                        mu_global, bsz, nnn,
                        vmin=vmin, vmax=vmax, text=None)

        mask = (np.abs(x_all - x0) < zoom_size) & (np.abs(y_all - y0) < zoom_size)
        inset_ax.scatter(x_all[mask], y_all[mask],
                         s=6, c=weights[mask], cmap='viridis', alpha=0.8)
        inset_ax.errorbar(x0, y0, xerr=xerr, yerr=yerr, fmt='o', color='red', capsize=2, elinewidth=1)
        inset_ax.text(x0 + 0.005, y0 + 0.005, label, fontsize=9, color='red')

        inset_ax.set_xlim(x0 - zoom_size, x0 + zoom_size)
        inset_ax.set_ylim(y0 - zoom_size, y0 + zoom_size)
        inset_ax.tick_params(axis='both', labelsize=10)

        # Set tick/label positions.
        if label == "C":  # Upper-left -> y-axis on the right.
            inset_ax.yaxis.tick_right()
            inset_ax.yaxis.set_label_position("right")
        elif label == "B":  # Lower-right -> x-axis on the top.
            inset_ax.xaxis.tick_top()
            inset_ax.xaxis.set_label_position("top")
        # Default A: no change.

        inset_ax.set_ylabel("")
        inset_ax.set_xlabel("")
        inset_ax.set_aspect('equal')

    # Colorbar.
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im_fdm, cax=cbar_ax)
    cbar.set_label(r"$\kappa_{\mathrm{eff,sub}}$")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
def plot_fdm_with_xroots_and_observations_ax(
    analyzer,
    i,
    x_all,
    y_all,
    weights,
    sim_type,
    plot_kappa_func,
    ax=None,  # New parameter.
    x_flip=True,
    zoom_size=0.05,
    inset_labels=["A", "B", "C"],
    inset_size="42%",
    save_path=None,
    savefile="Simulation",
    vmin=-0.02,
    vmax=0.02
):
    object_lensing = analyzer.obj
    image_positions_relative_to_L = {
        "A": (object_lensing["x_imageA"], object_lensing["y_imageA"],
              object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "B": (object_lensing["x_imageB"], object_lensing["y_imageB"],
              object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "C": (object_lensing["x_imageC"], object_lensing["y_imageC"],
              object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
        "D": (object_lensing["x_imageD"], object_lensing["y_imageD"],
              object_lensing["delta_theta_FDM"], object_lensing["delta_theta_FDM"]),
    }
    thetaE = analyzer.obj["thetaE"]
    nnn = analyzer.obj["nnn"]
    simu_obj = analyzer.obj["name"]
    source_dir = os.path.join(savefile, simu_obj, str(i))
    global_file = os.path.join(savefile, simu_obj)

    params = load_kappa_multi(simu_obj, sim_type, thetaE, filename=source_dir, nnn=nnn)

    k_main = get_main_plan_kappa(params['bsz'],
                                 filename_main=f"{global_file}/{simu_obj}_Global_alpha.npz", nnn=nnn)
    kappa_fdm = params["kappa_multi"] - k_main

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    weights = np.array(weights)

    obs_points = {
        label: ((-x if x_flip else x), y, x_err, y_err)
        for label, (x, y, x_err, y_err) in image_positions_relative_to_L.items()
    }

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03], wspace=0.1)
        ax = fig.add_subplot(gs[0, 0])
        created_fig = True  # Track whether a new figure was created.
    else:
        fig = ax.figure

    bsz = params['bsz']
    nnn = params['mu'].shape[0]

    im_fdm = plot_kappa_func(ax, kappa_fdm, params['mu'], bsz, nnn,
                             vmin=vmin, vmax=vmax, text=sim_type)

    sc = ax.scatter(x_all, y_all, s=5, c=weights, cmap='viridis', alpha=0.8, label="xroots (weighted)")

    for label, (x, y, xerr, yerr) in obs_points.items():
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt='o', capsize=2, elinewidth=1, markersize=4,
                    color='red', label=f"Image {label}" if label == "A" else None)
        ax.text(x + 0.03, y + 0.03, label, fontsize=10, color='red')

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    inset_loc_map = {
        "C": 'upper left',
        "A": 'upper right',
        "B": 'lower right'
    }

    for label in inset_labels:
        if label not in obs_points:
            continue
        x0, y0, xerr, yerr = obs_points[label]

        inset_ax = inset_axes(
            ax,
            width=inset_size, height=inset_size,
            loc=inset_loc_map[label],
            borderpad=0
        )

        plot_kappa_func(inset_ax, kappa_fdm, params['mu'], bsz, nnn,
                        vmin=vmin, vmax=vmax, text=None)

        mask = (np.abs(x_all - x0) < zoom_size) & (np.abs(y_all - y0) < zoom_size)
        inset_ax.scatter(x_all[mask], y_all[mask],
                         s=6, c=weights[mask], cmap='viridis', alpha=0.8)
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

    if created_fig:
        cbar_ax = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(im_fdm, cax=cbar_ax)
        cbar.set_label(r"$\kappa_{\mathrm{eff,sub}}$")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    return im_fdm

def plot_Caustics_no_inset(ax_, yi2, yi1, mu_global, ys1, ys2, arcsec_1,
                  yoff=False, xoff=False, ifscale=False, text=None,
                  numcaustic=2,ifsourcepoint = False):
    vmin = 0.5
    vmax = 4.5
    levels = np.logspace(np.log10(vmin), np.log10(vmax), numcaustic)
    cutd = 8
    cut_c_p = 0.04

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lambdat_contour2 = ax_.contour(yi2, yi1, np.log10(np.abs(mu_global)),
                                   levels=levels, cmap='cubehelix', linewidths=2, norm=norm)

    ax_.set_facecolor('black')
    ax_.set_xlabel("X (arcsec)")
    ax_.set_ylabel("Y (arcsec)")
    ax_.set_xlim(-cutd * cut_c_p, cutd * cut_c_p)
    ax_.set_ylim(-cutd * cut_c_p, cutd * cut_c_p)

    color_use = 'white'

    # Main-plot cusp point.
    if ifsourcepoint:
        ax_.scatter(ys2, ys1, color='red', s=50, edgecolor=None, zorder=3)

    ax_.set_aspect('equal')

    if xoff:
        ax_.get_xaxis().set_visible(False)
    if yoff:
        ax_.get_yaxis().set_visible(False)

    # Lower-left annotation.
    base_x = -cutd * cut_c_p * 0.95
    base_y = -cutd * cut_c_p * 0.95
    if text is not None:
        ax_.text(base_x, base_y, text, fontsize=22, color=color_use, va='bottom', ha='left')

    # Scale bar (moved to the upper-left).
    if ifscale:
        scalebar_length = 0.1
        scale_kpc = scalebar_length * arcsec_1 * 1000
        scalebar_x = -cutd * cut_c_p * 0.8
        scalebar_y = cutd * cut_c_p * 0.7  # Near the upper-left.
        ax_.plot([scalebar_x, scalebar_x + scalebar_length],
                 [scalebar_y, scalebar_y], color=color_use, lw=2)
        ax_.text(scalebar_x + scalebar_length * 0.5,
                 scalebar_y + scalebar_length * 0.3,
                 f'{scale_kpc:.2f} kpc/h', color=color_use, ha='center', va='bottom')


    return lambdat_contour2

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_Caustics(ax_, yi2, yi1, mu_global, ys1, ys2, arcsec_1,
                  yoff=False, xoff=False, ifscale=False, text=None,
                  numcaustic=2, inset_size="35%", zoom_size=0.01, ifsourcepoint = False):
    vmin = 0.5
    vmax = 4.5
    levels = np.logspace(np.log10(vmin), np.log10(vmax), numcaustic)
    cutd = 8
    cut_c_p = 0.06

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lambdat_contour2 = ax_.contour(yi2, yi1, np.log10(np.abs(mu_global)),
                                   levels=levels, cmap='cubehelix', linewidths=2, norm=norm)

    ax_.set_facecolor('black')
    ax_.set_xlabel("X (arcsec)")
    ax_.set_ylabel("Y (arcsec)")
    ax_.set_xlim(-cutd * cut_c_p, cutd * cut_c_p)
    ax_.set_ylim(-cutd * cut_c_p, cutd * cut_c_p)

    color_use = 'white'

    # Main-plot cusp point.
    if ifsourcepoint:
        ax_.scatter(ys2, ys1, color='red', s=100, edgecolor='white', zorder=3)

    ax_.set_aspect('equal')

    x_ticks = ax_.get_xticks()
    y_ticks = ax_.get_yticks()
    # ax_.set_xticklabels([f'{tick * 1000:.0f}' for tick in x_ticks])
    # ax_.set_yticklabels([f'{tick * 1000:.0f}' for tick in y_ticks])

    if xoff:
        ax_.get_xaxis().set_visible(False)
    if yoff:
        ax_.get_yaxis().set_visible(False)

    # Lower-left annotation.
    base_x = -cutd * cut_c_p * 0.95
    base_y = -cutd * cut_c_p * 0.95
    if text is not None:
        ax_.text(base_x, base_y, text, fontsize=22, color=color_use, va='bottom', ha='left')

    # Scale bar.
    if ifscale:
        scalebar_length = 0.1
        scale_pc = scalebar_length * arcsec_1 * 1000 * 1000
        scalebar_x = base_x
        scalebar_y = base_y + 0.1
        ax_.plot([scalebar_x, scalebar_x + scalebar_length], [scalebar_y, scalebar_y], color=color_use, lw=2)
        ax_.text(scalebar_x + scalebar_length, scalebar_y + scalebar_length * 0.3,
                 f'{scale_pc:.2f} pc/h', fontsize=12, color=color_use, ha='center')

    # Inset in the upper-right.
    inset_ax = inset_axes(ax_, width=inset_size, height=inset_size,
                          loc='upper left', borderpad=0)
    inset_ax.yaxis.tick_right()
    inset_ax.yaxis.set_label_position("right")
    # Background contours.
    inset_ax.contour(yi2, yi1, np.log10(np.abs(mu_global)),
                     levels=levels, cmap='cubehelix', linewidths=1.5, norm=norm)

    # Mark the ys point.
    inset_ax.scatter(ys2, ys1, color='red', s=50, edgecolor='white', linewidth=0.6, zorder=3)

    # Zoom range.
    inset_ax.set_xlim(ys2 - zoom_size, ys2 + zoom_size)
    inset_ax.set_ylim(ys1 - zoom_size, ys1 + zoom_size)

    # Axis ticks and styling.
    inset_ax.tick_params(axis='both', labelsize=10, colors='white')
    inset_ax.set_facecolor('black')
    inset_ax.set_aspect('equal')

    for spine in inset_ax.spines.values():
        spine.set_color('white')

    return lambdat_contour2




from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, ConnectionPatch

def plot_Caustics_list(ax_, yi2, yi1, mu_global, ys1, ys2, arcsec_1,
                       yoff=False, xoff=False, text=None,
                       numcaustic=2, inset_size="35%", zoom_size=0.01,
                       chi2_values=None, scatter_cmap='magma',
                       vmin=0.5, vmax=4.5):

    levels = np.logspace(np.log10(vmin), np.log10(vmax), numcaustic)
    cutd = 8
    cut_c_p = 0.06

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lambdat_contour2 = ax_.contour(
        yi2, yi1, np.log10(np.abs(mu_global)),
        levels=levels, cmap='cubehelix', linewidths=2, norm=norm
    )

    ax_.set_facecolor('black')
    ax_.set_xlabel("X (arcsec)")
    ax_.set_ylabel("Y (arcsec)")
    ax_.set_xlim(-cutd * cut_c_p, cutd * cut_c_p)
    ax_.set_ylim(-cutd * cut_c_p, cutd * cut_c_p)

    color_use = 'white'

    ys1_arr = np.atleast_1d(ys1)
    ys2_arr = np.atleast_1d(ys2)

    if chi2_values is not None and len(chi2_values) == len(ys1_arr):
        ax_.scatter(ys2_arr, ys1_arr, c=-chi2_values,
                    cmap=scatter_cmap, s=5, zorder=5)
    else:
        color = 'black' if text in ['No Subhalo', 'CDM'] else 'red'
        ax_.scatter(ys2_arr, ys1_arr, color=color, s=10, edgecolor='white')

    ax_.set_aspect('equal')

    if xoff:
        ax_.get_xaxis().set_visible(False)
    if yoff:
        ax_.get_yaxis().set_visible(False)

    # Lower-left annotation.
    base_x = -cutd * cut_c_p * 0.95
    base_y = -cutd * cut_c_p * 0.95
    if text is not None:
        ax_.text(base_x, base_y, text, fontsize=24,
                 color=color_use, va='bottom', ha='left')

    # Scale bar.
    if "CDM" in text:
        scalebar_length = 0.1
        scale_pc = scalebar_length * arcsec_1 * 1e3
        scalebar_x = base_x +0.05
        scalebar_y = base_y + 0.12
        ax_.plot([scalebar_x, scalebar_x + scalebar_length],
                 [scalebar_y, scalebar_y],
                 color=color_use, lw=3)
        ax_.text(scalebar_x + scalebar_length*1.5,
                 scalebar_y + scalebar_length * 0.5,
                 f'{scale_pc:.2f} kpc/h',
                 fontsize=24, color=color_use, ha='center')
        
    ax_.set_xlabel("")

    # Inset subplot.
    inset_ax = inset_axes(ax_, width=inset_size, height=inset_size,
                          loc='upper left', borderpad=0)
    inset_ax.yaxis.tick_right()
    inset_ax.yaxis.set_label_position("right")

    inset_ax.contour(yi2, yi1, np.log10(np.abs(mu_global)),
                     levels=levels, cmap='cubehelix',
                     linewidths=1.5, norm=norm)

    if chi2_values is not None and len(chi2_values) == len(ys1_arr):
        inset_ax.scatter(ys2_arr, ys1_arr, c=-chi2_values,
                         cmap=scatter_cmap, s=5, zorder=5)
    else:
        inset_ax.scatter(ys2_arr, ys1_arr, color='red',
                         s=50, edgecolor='white', linewidth=0.6)

    # Zoom to the ys center.
    ys1_mean = np.mean(ys1_arr)
    ys2_mean = np.mean(ys2_arr)
    inset_ax.set_xlim(ys2_mean - zoom_size, ys2_mean + zoom_size)
    inset_ax.set_ylim(ys1_mean - zoom_size, ys1_mean + zoom_size)

    # inset_ax.tick_params(axis='both', labelsize=12, colors='white')
    # Remove all tick marks and labels.
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.tick_params(
        axis='both',
        length=0,           # Do not draw tick marks.
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False
    )

    inset_ax.set_ylabel("")
    inset_ax.set_xlabel("")
    inset_ax.set_facecolor('black')
    inset_ax.set_aspect('equal')
    for spine in inset_ax.spines.values():
        spine.set_color('white')

    # Add a white rectangle to the main plot showing the inset zoom region.
    rect = Rectangle(
        (ys2_mean - zoom_size, ys1_mean - zoom_size),
        2 * zoom_size, 2 * zoom_size,
        linewidth=1, edgecolor='white',
        facecolor='none', zorder=10
    )
    ax_.add_patch(rect)

    # Add a white connector from the rectangle to the inset.
    # Use the rectangle upper-right corner to the inset lower-right.
    start_x = ys2_mean + zoom_size
    start_y = ys1_mean + zoom_size
    con = ConnectionPatch(
        xyA=(start_x, start_y), coordsA=ax_.transData,
        xyB=(1.0, 0.0),        coordsB=inset_ax.transAxes,  # Inset lower-right.
        color='white', lw=1.0, zorder=11
    )
    ax_.figure.add_artist(con)

    return lambdat_contour2
