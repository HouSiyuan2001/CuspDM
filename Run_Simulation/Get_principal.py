from update_single import *

import os
import numpy as np
import matplotlib.pyplot as plt


import argparse
import re
from update_single import *

def compute_reff_from_alpha(alpha_file, dsx_arc, bsz_arc, nnn):
    """
    Compute equivalent radius reff from an alpha file.
    """
    # Load data
    data = np.load(alpha_file)
    alpha1_global = data['alpha1_global']
    alpha2_global = data['alpha2_global']

    # Build coordinate grid
    xi2, xi1 = make_c_coor(bsz_arc, nnn)

    # Compute magnification and related quantities
    yi1, yi2, mu_global, kappa_, gamma1_global, gamma2_global = alphas_to_mu(
        alpha1_global, alpha2_global, dsx_arc, xi1, xi2
    )

    # Calculate lambdat_global = 1 - κ - γ
    gamma_global = np.sqrt(gamma1_global ** 2 + gamma2_global ** 2)
    lambdat_global = 1 - kappa_ - gamma_global
    lambdat_global_np = np.array(lambdat_global)

    # Find lambdat=0 critical curve and compute enclosed area
    fig, ax = plt.subplots()
    try:
        lambdat_contour = ax.contour(yi2, yi1, lambdat_global_np, levels=[0])
    except Exception as e:
        plt.close(fig)
        print(f"⚠️ Unable to draw lambdat=0 contour ({alpha_file}): {e}")
        return None
    plt.close(fig)

    lambdat_contour_paths = lambdat_contour.allsegs[0]
    areas = []
    for seg in lambdat_contour_paths:
        x, y = seg[:, 0], seg[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)

    if not areas:
        print(f"⚠️ No lambdat=0 region detected: {alpha_file}")
        return None

    # Use the largest closed region to compute equivalent radius
    area_max = max(areas)
    reff = np.sqrt(area_max / np.pi)
    print(f"✅ Computed reff = {reff:.4f} (from {os.path.basename(alpha_file)})")
    return reff


def Simulation_MCMC_Mock_each_phibin_singlefile(out_file, savefile, name, sim_type, phi_center, delta_phi=5,
                                                bsz_arc=10.0, nnn=1500, r_eff=1.0):
    """
    Single-file MCMC helper to process alpha files in Data_single_para.
    """
    xi2, xi1 = make_c_coor(bsz_arc, nnn)
    y_opt = np.array([0, 0])

    print(f"Loading: {out_file}")
    if not os.path.exists(out_file):
        print(f"❌ File does not exist: {out_file}")
        return

    kappa_multi, yi1, yi2, mu_global, lambdat_global, _ = get_kappa(out_file, bsz_arc, nnn)

    print(f"--- φ = {phi_center}° ---")

    n_steps_dynamic = int(-16.6667 * phi_center + 2333.33)
    n_steps_dynamic = max(n_steps_dynamic, 1000)

    os.makedirs(savefile, exist_ok=True)
    save_path_combined = f"{savefile}/{name}_{sim_type}_{phi_center}_chain_combined.npz"
    save_path_rcusp = f"{savefile}/{name}_{sim_type}_{phi_center}_Rcusp_phi.npz"

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

    selected_points = rebin_mcmc_points_full_bs_jax(save_path_combined, bsz_arc, sigma_level=5)
    if selected_points is None:
        print(f"No selected points for φ = {phi_center}°")
        return

    print(f"Selected points within 1σ: {len(selected_points)}")
    if len(selected_points) > 3000:
        idx_choice = np.random.choice(len(selected_points), size=3000, replace=False)
        selected_points = selected_points[idx_choice]
        print("⚠️ Too many points, sampled 3000 at random")

    filtered_samples = selected_points[:, :2]
    chi2 = selected_points[:, 2]

    xroots_list, mu_info_list, Rcusp_arr, weights, angles_arr, axis_type_list = get_xroots_and_mu_info(
        filtered_samples, chi2, xi1, xi2, yi1, yi2, mu_global
    )

    np.savez(save_path_rcusp,
             Rcusp_arr=Rcusp_arr,
             weights=weights,
             angles=angles_arr,
             xroots=xroots_list,
             mu_info=mu_info_list,
             axis_types=axis_type_list,
             filtered_samples=filtered_samples)

    print(f"Saved Rcusp data for φ = {phi_center}° to {save_path_rcusp}")



def parse_params_from_filename(filename):
    """
    Extract parameters from filename, e.g.
    q0.6_thetaE1.0_pa0_g0.1_phi0_Global_alpha.npz
    Returns dict: { 'q_l':0.6, 'thetaE':1.0, 'pa':0, 'g_external':0.1, 'phi_external':0 }
    """
    pattern = r"q(?P<q>[\d\.]+)_thetaE(?P<thetaE>[\d\.]+)_pa(?P<pa>[\d\.-]+)_g(?P<g>[\d\.-]+)_phi(?P<phi>[\d\.-]+)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"❌ Cannot parse parameters from filename: {filename}")
    params = {
        "q_l": float(match.group("q")),
        "thetaE": float(match.group("thetaE")),
        "pa": float(match.group("pa")),
        "g_external": float(match.group("g")),
        "phi_external": float(match.group("phi")),
    }
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", type=str, default=None, help="File list to process (txt)")
    parser.add_argument("--save-folder", type=str, default="Data_single_para", help="Output directory")

    args = parser.parse_args()

    data_folder = "Data_single_para"
    save_folder = args.save_folder 
    sim_type = "mock"
    delta_phi = 5
    phi_bins = np.arange(20, 141, 2 * delta_phi, dtype=np.int32)
    nnn_default = 1500

    # Load file paths from file-list when provided
    if args.file_list:
        with open(args.file_list, 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]
    else:
        all_files = [os.path.join(data_folder, f)
                     for f in os.listdir(data_folder)
                     if f.endswith("_Global_alpha.npz")]

    if not all_files:
        print("⚠️ No files to process")
        return

    print(f"Found {len(all_files)} files, running simulations...\n")

    for out_file in all_files:
        # Ensure `file` is defined before use
        file = os.path.basename(out_file)
        name = file.replace("_Global_alpha.npz", "")

        # Parse parameters from filename
        try:
            params = parse_params_from_filename(file)
        except ValueError as e:
            print(e)
            continue

        q_l = params["q_l"]
        thetaE = params["thetaE"]
        pa = params["pa"]
        g_external = params["g_external"]
        phi_external = params["phi_external"]

        # Automatically compute bsz_arc and dsx_arc from q_l and thetaE
        bsz_arc = 3.0 * thetaE / np.sqrt(q_l)
        dsx_arc = bsz_arc / nnn_default

        # Compute equivalent radius for this file
        r_eff = compute_reff_from_alpha(out_file, dsx_arc=dsx_arc, bsz_arc=bsz_arc, nnn=nnn_default)
        if r_eff is None:
            print(f"⚠️ Skipping {file}, failed to compute reff")
            continue

        print(f"\nProcessing {file}: q={q_l:.2f}, thetaE={thetaE:.2f}, g={g_external:.2f}, phi={phi_external:.1f}")
        print(f"bsz_arc={bsz_arc:.3f}, dsx_arc={dsx_arc:.6f}, r_eff={r_eff:.3f}")

        # Run MCMC simulation
        for phi_center in phi_bins:
            Simulation_MCMC_Mock_each_phibin_singlefile(
                out_file=out_file,
                savefile=save_folder,
                name=name,
                sim_type=sim_type,
                phi_center=phi_center,
                delta_phi=delta_phi,
                bsz_arc=bsz_arc,
                nnn=nnn_default,
                r_eff=r_eff
            )


if __name__ == "__main__":
    main()
