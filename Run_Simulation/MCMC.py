from fit_phi import *
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax.numpy as jnp
from jax import vmap
import numpy as np
import jax

def save_single_chain_result(sampler, save_path, walker_index=0):
    # Get chain for a specific walker, discarding burn-in steps
    samples = sampler.get_chain(discard=100, flat=False)[:, walker_index, :]
    log_prob = sampler.get_log_prob(discard=100, flat=False)[:, walker_index]
    np.savez(save_path, samples=samples, log_prob=log_prob)

def save_sampler_result(sampler, save_path, skip_num = 0):
    samples = sampler.get_chain(discard=skip_num, flat=True)
    log_prob = sampler.get_log_prob(discard=skip_num, flat=True)
    np.savez(save_path, samples=samples, log_prob=log_prob)

def save_combined_sampler_result(sampler1, sampler2, save_path):
    # Get samples from sampler1 and sampler2 (including burn-in)
    samples1 = sampler1.get_chain(discard=0, flat=True)
    log_prob1 = sampler1.get_log_prob(discard=0, flat=True)

    samples2 = sampler2.get_chain(discard=0, flat=True)
    log_prob2 = sampler2.get_log_prob(discard=0, flat=True)

    # Concatenate samples and log_prob
    samples_combined = np.concatenate([samples1, samples2], axis=0)
    log_prob_combined = np.concatenate([log_prob1, log_prob2], axis=0)

    # Save combined result
    np.savez(save_path, samples=samples_combined, log_prob=log_prob_combined)

def Simulation_MCMC(object_lensing, sim_type,index=None,plot_png = False, datapath = 'Data',npz_dir=None):
    obj = object_lensing["name"]
    if npz_dir==None:
        if sim_type == 'Global': 
            npz_path = f"{datapath}/{obj}/{obj}_{sim_type}_opt_result.npz"
        else:
            npz_path = f"{datapath}/{obj}/{index}/{obj}_{sim_type}_{index}_opt_result.npz"
    else:
        npz_path = f"{npz_dir}_opt_result.npz"
    print(npz_path)
    if not os.path.exists(npz_path):

        return 

    save_dir = f"{datapath}/{obj}/{index}"

    data = np.load(npz_path, allow_pickle=True)
    mu_global = data["mu_global"]
    y_opt = data["ys_final"]
    xi1 = data["xi1"]
    xi2 = data["xi2"]
    yi1 = data["yi1"]
    yi2 = data["yi2"]
    r_eff = object_lensing["Sample_reff"]/8
    print("Running")
    # Coarse MCMC
    sampler_coarse = run_mcmc(
                    mu_global, xi1, xi2, yi1, yi2,
                    obs_phi=object_lensing["phi"], obs_phi_sigma=2*object_lensing["phi_sigma"],
                    obs_phi1=object_lensing["phi1"], obs_phi1_sigma=2*object_lensing["phi1_sigma"],
                    obs_phi2=object_lensing["phi2"], obs_phi2_sigma=2*object_lensing["phi2_sigma"],
                    x_center=y_opt[1], y_center=y_opt[0],
                    n_walkers=5, n_steps=600, initial_radius=r_eff
                )

    # Get log posterior and compute chi²
    log_probs = sampler_coarse.get_log_prob(discard=50, flat=True)
    chi2 = -2 * log_probs
    samples_coarse = sampler_coarse.get_chain(discard=50, flat=True)
    # Find index of minimum chi²
    best_index = np.argmin(chi2)
    best_point = samples_coarse[best_index]
    best_chi2 = chi2[best_index]

    # Unpack to x, y
    x_best, y_best = best_point

    print(f"Best (x, y): ({x_best:.6f}, {y_best:.6f}) with chi² = {best_chi2}")
    # Fine MCMC
    sampler = run_mcmc(
                    mu_global, xi1, xi2, yi1, yi2,
                    obs_phi=object_lensing["phi"], obs_phi_sigma=object_lensing["phi_sigma"],
                    obs_phi1=object_lensing["phi1"], obs_phi1_sigma=object_lensing["phi1_sigma"],
                    obs_phi2=object_lensing["phi2"], obs_phi2_sigma=object_lensing["phi2_sigma"],
                    x_center=x_best, y_center=y_best,
                    n_walkers=10, n_steps=1000,initial_radius=r_eff/2
                )

    # Save chain for current sigma
    os.makedirs(save_dir, exist_ok=True)
    # Save combined coarse + fine MCMC chains
    if npz_dir==None:
        save_path_combined = f"{save_dir}/{obj}_{sim_type}_{index}_chain_combined.npz"
    else:
        save_path_combined = f"{npz_dir}_chain_combined.npz"
    save_sampler_result(sampler, save_path_combined)
    selected_points = rebin_mcmc_points_within_nsigma_jax(save_path_combined, sigma_level=5)
    print(f"Selected points within 1σ: {len(selected_points)}")
    
    filtered_samples = selected_points[:, :2]
    chi2 = selected_points[:, 2]
    Rcusp_arr, weights = plot_weighted_rcusp_pdf(filtered_samples, chi2, xi1, xi2, yi1, yi2, mu_global)
    if npz_dir==None:
        save_path_rcusp= f"{save_dir}/{obj}_{sim_type}_{index}_Rcusp.npz"
    else:
        save_path_rcusp= f"{npz_dir}_Rcusp.npz"
    np.savez(save_path_rcusp, Rcusp_arr=Rcusp_arr, weights=weights)
    if plot_png: 
        save_name_chi2_distribution = f"{obj}_{sim_type}_{index}_chi2_distribution.png"
        save_name_PDF = f"{obj}_{sim_type}_{index}_PDF.png"
        plot_chi2_distribution(selected_points, save_dir,save_name_chi2_distribution)
        plot_PDF(Rcusp_arr, weights, object_lensing["Rcusp_use"],save_dir,save_name_PDF)




from scipy.stats import norm, chi2 as chi2_dist

def rebin_mcmc_points_within_nsigma_jax(npz_path, bins=100, sigma_level=5, ndim=2, filter_by_sigma=True):
    data = np.load(npz_path, allow_pickle=True)
    samples = jnp.array(data["samples"])  # shape (N, 2)
    log_probs = jnp.array(data["log_prob"])
    chi2 = -2.0 * log_probs

    if filter_by_sigma:
        # Compute confidence corresponding to sigma_level (two-sided)
        conf = norm.cdf(sigma_level) * 2 - 1  # e.g. 5σ ≈ 0.9999994
        delta_chi2 = chi2_dist.ppf(conf, df=ndim)

        chi2_min = jnp.min(chi2)
        threshold = chi2_min + delta_chi2

        mask = chi2 <= threshold
        samples = samples[mask]
        chi2 = chi2[mask]


    if samples.shape[0] == 0:
        raise RuntimeError("No points found within 1σ; MCMC results may be invalid.")

    # Grid info
    x_min, x_max = jnp.min(samples[:, 0]), jnp.max(samples[:, 0])
    y_min, y_max = jnp.min(samples[:, 1]), jnp.max(samples[:, 1])

    dx = (x_max - x_min) / bins
    dy = (y_max - y_min) / bins

    # Grid index for each point
    ix = jnp.floor((samples[:, 0] - x_min) / dx).astype(int)
    iy = jnp.floor((samples[:, 1] - y_min) / dy).astype(int)

    # Avoid out-of-bounds
    ix = jnp.clip(ix, 0, bins - 1)
    iy = jnp.clip(iy, 0, bins - 1)

    # Flatten 2D grid indices to 1D
    grid_id = ix * bins + iy

    # Aggregate minimum chi² by grid_id
    unique_ids, inv_idx = jnp.unique(grid_id, return_inverse=True)
    min_chi2_per_cell = jax.ops.segment_min(chi2, inv_idx, num_segments=unique_ids.shape[0])

    # Find best point index in each cell
    def get_best_idx(cell_id, min_chi):
        idx = jnp.where((grid_id == cell_id) & (chi2 == min_chi), size=1, fill_value=-1)[0]
        return idx

    best_indices = vmap(get_best_idx)(unique_ids, min_chi2_per_cell)
    valid_mask = best_indices != -1
    best_indices = best_indices[valid_mask]

    # Extract results
    best_samples = samples[best_indices]
    best_chi2 = chi2[best_indices]
    selected_points = jnp.concatenate([best_samples, best_chi2[:, None]], axis=1)

    return np.array(selected_points)

def rebin_mcmc_points_full_bs_jax(npz_path, bs, nc=20000, sigma_level=5, ndim=2, filter_by_sigma=True):
    data = np.load(npz_path, allow_pickle=True)
    samples = jnp.array(data["samples"])  # shape (N, 2)
    log_probs = jnp.array(data["log_prob"])
    chi2 = -2.0 * log_probs

    if filter_by_sigma:
        conf = norm.cdf(sigma_level) * 2 - 1
        delta_chi2 = chi2_dist.ppf(conf, df=ndim)
        chi2_min = jnp.min(chi2)
        threshold = chi2_min + delta_chi2
        mask = chi2 <= threshold
        samples = samples[mask]
        chi2 = chi2[mask]

    if samples.shape[0] == 0:
        log_file = os.path.join(os.getcwd(), "mcmc_rebin_log.txt")
        with open(log_file, "a") as f:
            f.write(f"{npz_path}\n")
        return None

    # Define full space using bs
    x_min, x_max = -bs / 2.0, bs / 2.0
    y_min, y_max = -bs / 2.0, bs / 2.0

    dx = (x_max - x_min) / nc
    dy = (y_max - y_min) / nc

    ix = jnp.floor((samples[:, 0] - x_min) / dx).astype(int)
    iy = jnp.floor((samples[:, 1] - y_min) / dy).astype(int)

    ix = jnp.clip(ix, 0, nc - 1)
    iy = jnp.clip(iy, 0, nc - 1)

    grid_id = ix * nc + iy
    unique_ids, inv_idx = jnp.unique(grid_id, return_inverse=True)
    min_chi2_per_cell = jax.ops.segment_min(chi2, inv_idx, num_segments=unique_ids.shape[0])

    def get_best_idx(cell_id, min_chi):
        idx = jnp.where((grid_id == cell_id) & (chi2 == min_chi), size=1, fill_value=-1)[0]
        return idx

    best_indices = vmap(get_best_idx)(unique_ids, min_chi2_per_cell)
    valid_mask = best_indices != -1
    best_indices = best_indices[valid_mask]

    best_samples = samples[best_indices]
    best_chi2 = chi2[best_indices]
    selected_points = jnp.concatenate([best_samples, best_chi2[:, None]], axis=1)

    return np.array(selected_points)

from scipy.stats import gaussian_kde


def plot_weighted_rcusp_pdf(filtered_samples, chi2_values, xi1, xi2, yi1, yi2, mu_global):
    Rcusp_list = []
    weight_list = []

    for point, chi2 in tqdm(zip(filtered_samples, chi2_values), total=len(filtered_samples), desc="Computing Rcusp"):
        ys = np.array([point[1], point[0]])  # Note y, x order
        xroots_all, mask = mapping_triangles_vec_jax(ys, xi1, xi2, yi1, yi2)
        xroots = xroots_all[mask]
        if len(xroots) != 5:
            continue

        mu_info = get_mu_of_three_points(xroots, mu_global, xi1, xi2)
        Rcusp_info = Get_Rcusp_and_phi(mu_info)

        Rcusp = Rcusp_info["R_cusp"]
        Rcusp_list.append(Rcusp)

        # Compute weights w ∝ exp(-0.5 * chi²)
        weight = np.exp(-0.5 * chi2)
        weight_list.append(weight)

    Rcusp_arr = np.array(Rcusp_list)
    weights = np.array(weight_list)

    if len(Rcusp_arr) == 0:
        raise RuntimeError("Rcusp results are empty; filtering may be too strict.")

    # Normalize weights
    weights /= np.sum(weights)
    return Rcusp_arr, weights

def get_xroots_and_mu_info(filtered_samples, chi2_values, xi1, xi2, yi1, yi2, mu_global):
    xroots_list = []
    mu_info_list = []
    Rcusp_list = []
    weight_list = []
    angles_list = []
    axis_type_list = []

    for point, chi2 in tqdm(zip(filtered_samples, chi2_values), total=len(filtered_samples), desc="Extracting xroots and mu_info"):
        ys = np.array([point[1], point[0]])  # Note y, x order
        xroots_all, mask = mapping_triangles_vec_jax(ys, xi1, xi2, yi1, yi2)
        xroots = xroots_all[mask]
        if len(xroots) != 5:
            continue

        mu_info = get_mu_of_three_points(xroots, mu_global, xi1, xi2)
        xroots_list.append(xroots)
        mu_info_list.append(mu_info)
        Rcusp_info = Get_Rcusp_and_phi(mu_info)
        edge_points = np.array([p["position"] for p in Rcusp_info["edge_points"]])
        negative_point = Rcusp_info["negative_point"]
        center_point = np.array(Rcusp_info["center_point"]["position"])
        non_cusp_point = np.array(Rcusp_info["non_cusp_point"]["position"])
        angles = compute_theoretical_angles(edge_points, negative_point, center_point) # angle relative to lens center

        dist_negative = np.linalg.norm(negative_point - center_point)
        dist_non_cusp = np.linalg.norm(non_cusp_point - center_point)

        if dist_negative > dist_non_cusp:
            axis_type = "long_axis"
        else:
            axis_type = "short_axis"

        Rcusp = Rcusp_info["R_cusp"]
        Rcusp_list.append(Rcusp)
        axis_type_list.append(axis_type)



        # Compute weights w ∝ exp(-0.5 * chi²)
        weight = np.exp(-0.5 * chi2)
        weight_list.append(weight)

        angles_list.append(angles["phi"])

    Rcusp_arr = np.array(Rcusp_list)
    weights = np.array(weight_list)
    angles_arr = np.array(angles_list)

    # if len(Rcusp_arr) == 0:
    #     return None

    # Normalize weights
    weights /= np.sum(weights)

    return xroots_list, mu_info_list,Rcusp_arr, weights, angles_arr,axis_type_list

def plot_PDF(Rcusp_arr, weights, ob_Rcusp, save_dir,save_name):
    # KDE estimation
    kde = gaussian_kde(Rcusp_arr, weights=weights)
    x_eval = np.linspace(min(Rcusp_arr), max(Rcusp_arr), 500)
    pdf = kde(x_eval)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(x_eval, pdf, color='blue')
    plt.axvline(ob_Rcusp, color='red', linestyle='--', label=fr'Observed $R_{{\rm cusp}} = {ob_Rcusp:.3f}$')
    plt.xlabel(r'$R_{\rm cusp}$')
    plt.ylabel('Weighted PDF')
    plt.title(r'Weighted $R_{\rm cusp}$ Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    # Save figure
    plt.savefig(save_path, dpi=300)

def plot_chi2_distribution(selected_points, save_dir,save_name):
    x = selected_points[:, 0]
    y = selected_points[:, 1]
    filtered_samples = selected_points[:, :2]
    chi2 = selected_points[:, 2]
    
    # min chi²
    chi2_min = np.min(chi2)
    print(f"Minimum chi²: {chi2_min}")
    
    # Scatter plot colored by chi²
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=chi2, cmap='viridis', s=50, edgecolor='none')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(sc, label=r'$\chi^2$')
    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    # Save figure
    plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    from notion_client import Client
    import sys 
    from notion import *
    from fit_phi import _find_free_gpu
    gpu = _find_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"Using GPU: {gpu}")
    sys.path.append("../lib")
    # Initialize Notion client
    notion = Client(auth="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")

    # Database ID
    database_id = "1f8fc9067a748057a9ebf052fd25a8bd"

    # Query database
    response = notion.databases.query(
        database_id=database_id,
        page_size=100
    )
    simu_obj = "PSJ0147+4630"
    object_lensing = get_lens_data_by_name(response, simu_obj)
    sim_type_list = ["FDM","CDM","SIDM","SIDM_col"]
    num_iterations = 100
    start_num = 1
    import time
    import os
    for i in range(start_num, num_iterations + start_num):
        print("........." * 10)
        print(f"🔄 Simulation {i} start")
        start_time = time.time()  # Record start time for iteration i
        for sim_type in sim_type_list:
            obj = object_lensing["name"]
            save_dir = f"Data/{obj}/{i}"
            os.makedirs(save_dir, exist_ok=True)
            save_path_rcusp = f"{save_dir}/{obj}_{sim_type}_{i}_Rcusp.npz"
            save_name_chi2_distribution = f"{obj}_{sim_type}_{i}_chi2_distribution.png"
            save_name_PDF = f"{obj}_{sim_type}_{i}_PDF.png"
            png_path_chi2 = os.path.join(save_dir, save_name_chi2_distribution)
            png_path_pdf = os.path.join(save_dir, save_name_PDF)

            need_run_simulation = not os.path.exists(save_path_rcusp)
            need_plot_only = os.path.exists(save_path_rcusp) and (not os.path.exists(png_path_chi2) or not os.path.exists(png_path_pdf))

            if need_run_simulation:
                print(f"🧪 Starting full simulation: {obj}_{sim_type}_{i}")
                Simulation_MCMC(object_lensing, sim_type, index=i)

            elif need_plot_only:
                print(f"📈 Plotting missing images only: {obj}_{sim_type}_{i}")
                data = np.load(save_path_rcusp)
                Rcusp_arr = data["Rcusp_arr"]
                weights = data["weights"]

                if not os.path.exists(png_path_chi2):
                    # Reload selected_points (only from chain_combined)
                    chain_path = f"{save_dir}/{obj}_{sim_type}_{i}_chain_combined.npz"
                    selected_points = rebin_mcmc_points_within_nsigma_jax(chain_path, sigma_level=5)
                    plot_chi2_distribution(selected_points, save_dir, save_name_chi2_distribution)

                if not os.path.exists(png_path_pdf):
                    plot_PDF(Rcusp_arr, weights, object_lensing["Rcusp_use"], save_dir, save_name_PDF)

            else:
                print(f"✅ Simulation and images already complete: {obj}_{sim_type}_{i}, skipping")
        # End of current iteration; estimate remaining time
        end_time = time.time()
        elapsed_time = end_time - start_time  # seconds
        remaining_rounds = (num_iterations + start_num - 1) - i
        est_total_remaining_sec = remaining_rounds * elapsed_time
        
        hrs = int(est_total_remaining_sec // 3600)
        mins = int((est_total_remaining_sec % 3600) // 60)
        secs = int(est_total_remaining_sec % 60)
        
        print(f"⏱️ Simulation {i} duration: {elapsed_time/60:.2f} minutes")
        print(f"📊 Estimated remaining time: {hrs} hours {mins} minutes {secs} seconds")
