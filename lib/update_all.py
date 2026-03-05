import time  # Timing helper
import os
import glob
from update_single import *

def cleanup_simulation_directory(sim_dir):
    """
    Remove intermediate files under sim_dir, keeping only:
    - *_alpha_mul_ray.npz
    - *_halolist_WDM.pkl
    - *_halolist_sigma0_8000_w0_6.pkl
    """

    # File suffixes to keep
    keep_keywords = [
        "_alpha_mul_ray.npz",
        "_Global_alpha.npz",
        "_params.json",
        "_chain_combined.npz",
        "_Rcusp_phi.npz",
        "_chi2_distribution.png",
        "_PDF.png",
        "_lens_plane.png",
        "_kappa_all_types.png"
    ]

    # Walk all files
    for root, dirs, files in os.walk(sim_dir):
        for file in files:
            full_path = os.path.join(root, file)

            # Delete files that do not match keep list
            if not any(file.endswith(key) for key in keep_keywords):
                try:
                    os.remove(full_path)
                except Exception as e:
                    print(f"⚠️ Failed to delete: {full_path} | error: {e}")

    print("✅ Cleanup complete; only required files remain.")

def simulate_main(simu_obj):
    savefile = "Simulation"  # Output directory

    main_folder = f"{savefile}/{simu_obj}"
    ensure_dir(main_folder)

    # Toggle to force rerun even when results exist
    force_run =  False

    global_alpha_file = os.path.join(main_folder, f"{simu_obj}_Global_alpha.npz")
    if not os.path.exists(global_alpha_file):
        print("🌌 Global_alpha not found, building main halo...")
        analyzer = HaloAnalyzer(simu_obj, savefile=main_folder)
        analyzer.Get_mainhalo()
        del analyzer
    else:
        print(f"✔️ Global_alpha exists: {global_alpha_file}, skipping main halo build")

    num_iterations = 100
    start_num = 1

    all_types = ["CDM", "WDM", "FDM", "SIDM", "SIDM_col"]
    # all_types = ["SIDM"]
    total_time = 0
    completed = 0

    for i in range(start_num, num_iterations + start_num):
        print("........." * 10)
        print(f"🔄 Simulation {i} start")

        iteration_folder = f"{savefile}/{simu_obj}/{i}"
        ensure_dir(iteration_folder)

        expected_files = [
            os.path.join(iteration_folder, f"{simu_obj}_{t}_alpha_mul_ray.npz")
            for t in all_types
        ]
        missing_types = [
            t for t, f in zip(all_types, expected_files) if not os.path.exists(f)
        ]

        if not missing_types and not force_run:
            print(f"✔️ Simulation {i} all types complete, skipping")
            completed += 1
            continue

        start_time = time.time()

        if force_run:
            print(f"🚨 Simulation {i}: forcing all types to rerun")
            missing_types = all_types
        else:
            print(f"🚀 Simulation {i}: missing {missing_types}, rerunning these types")

        analyzer = HaloAnalyzer(
            simu_obj,
            savefile=iteration_folder,
            globalfile=f"{savefile}/{simu_obj}"
        )

        if "FDM" in missing_types:
            analyzer.Simulation_FDM()

        if "WDM" in missing_types:
            wdm_pkl = os.path.join(iteration_folder, f"{simu_obj}_halolist_WDM.pkl")
            ifgenerate_wdm = not os.path.exists(wdm_pkl) or force_run
            analyzer.Simulation_WDM(ifgenerate=ifgenerate_wdm)

        if any(t in missing_types for t in ["CDM", "SIDM_col"]):
            cdm_pkl = os.path.join(iteration_folder, f"{simu_obj}_halolist_sigma0_8000_w0_6.pkl")
            ifgenerate_cdm = not os.path.exists(cdm_pkl) or force_run
            analyzer.Simulation_CDM_SIDM(ifgenerate=ifgenerate_cdm)
        
        if "SIDM" in missing_types:
            sidm_pkl = os.path.join(iteration_folder, f"{simu_obj}_halolist_sigma0_8000_w0_6.pkl")
            ifgenerate_sid = not os.path.exists(sidm_pkl) or force_run
            analyzer.Simulation_SIDM(ifgenerate=ifgenerate_sid)

        del analyzer
        cleanup_simulation_directory(f"{savefile}/{simu_obj}")

        end_time = time.time()
        elapsed = (end_time - start_time) / 3600
        total_time += elapsed
        completed += 1
        remaining = (num_iterations - completed) * (total_time / completed)

        print(f"⏱️ Simulation {i} runtime: {elapsed:.2f} hours")
        print(f"📊 Total elapsed runtime: {total_time:.2f} hours")
        print(f"📅 Estimated remaining runtime: {remaining:.2f} hours")

import json

# Draw directly from distribution
def simulate_main_Mock(simu_obj, dist_sampler):
    savefile = "Simulation"
    main_folder = f"{savefile}/{simu_obj}"
    ensure_dir(main_folder)

    force_run = False  # Set True to ignore skip logic
    num_iterations = 2
    start_num = 1
    all_types = ["CDM", "FDM", "SIDM", "SIDM_col"]

    total_time = 0
    completed = 0

    for i in range(start_num, num_iterations + start_num):
        print("........." * 10)
        print(f"🔄 Simulation {i} start")

        iteration_folder = f"{savefile}/{simu_obj}/{i}"
        ensure_dir(iteration_folder)

        # Set seed and sample lens system parameters
        lens_params = dist_sampler.sample_lens_system(name=f"{simu_obj}_{i}", seed=i)

        # Persist parameters for traceability
        param_file = os.path.join(iteration_folder, f"{simu_obj}_{i}_params.json")
        with open(param_file, "w") as f:
            json.dump(lens_params, f, indent=2)
        # Initialize main halo for each iteration
        analyzer = HaloAnalyzer_Mock_data(
            name=lens_params["name"],
            dist_sampler=dist_sampler,
            savefile=iteration_folder,
            globalfile=iteration_folder
        )
        analyzer.obj = lens_params

        sim_name_path = analyzer.obj["name"]
        global_alpha_file = os.path.join(iteration_folder, f"{sim_name_path}_Global_alpha.npz")
        if not os.path.exists(global_alpha_file):
            print("🌌 Global_alpha not found, building main halo...")
            analyzer.Get_mainhalo()
        else:
            print(f"✔️ Global_alpha exists: {global_alpha_file}, skipping main halo build")

        expected_files = [
            os.path.join(iteration_folder, f"{sim_name_path}_{t}_alpha_mul_ray.npz")
            for t in all_types
        ]
        missing_types = [
            t for t, f in zip(all_types, expected_files) if not os.path.exists(f)
        ]

        if not missing_types and not force_run:
            print(f"✔️ Simulation {i} all types complete, skipping")
            completed += 1
            continue

        start_time = time.time()
        print(f"🚀 Simulation {i}: missing {missing_types}, starting simulations")
        if any(t in missing_types for t in ["CDM", "SIDM_col"]):
            cdm_pkl = os.path.join(iteration_folder, f"{simu_obj}_halolist_sigma0_8000_w0_6.pkl")
            ifgenerate_cdm = not os.path.exists(cdm_pkl) or force_run
            analyzer.Simulation_CDM_SIDM(ifgenerate=ifgenerate_cdm)

        if "SIDM" in missing_types:
            sidm_pkl = os.path.join(iteration_folder, f"{simu_obj}_halolist_sigma0_8000_w0_6.pkl")
            ifgenerate_sid = not os.path.exists(sidm_pkl) or force_run
            analyzer.Simulation_SIDM(ifgenerate=ifgenerate_sid)
        if "FDM" in missing_types:
            analyzer.Simulation_FDM()

        if "WDM" in missing_types:
            wdm_pkl = os.path.join(iteration_folder, f"{simu_obj}_halolist_WDM.pkl")
            ifgenerate_wdm = not os.path.exists(wdm_pkl) or force_run
            analyzer.Simulation_WDM(ifgenerate=ifgenerate_wdm)

        

        del analyzer
        cleanup_simulation_directory(f"{savefile}/{simu_obj}")

        end_time = time.time()
        elapsed = (end_time - start_time) / 3600
        total_time += elapsed
        completed += 1
        remaining = (num_iterations - completed) * (total_time / completed)

        print(f"⏱️ Simulation {i} runtime: {elapsed:.2f} hours")
        print(f"📊 Total elapsed runtime: {total_time:.2f} hours")
        print(f"📅 Estimated remaining runtime: {remaining:.2f} hours")
if __name__ == "__main__":
    sampler = DistributionSampler()
    simulate_main_Mock("mockset_seeded", dist_sampler=sampler)  
