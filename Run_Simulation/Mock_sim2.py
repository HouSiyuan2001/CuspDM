import sys
sys.path.append("../lib")
from notion import get_Mock_lens_system_params_mul
from astropy.io import fits
from tqdm import tqdm
from update_single import *
from update_all import *
import time
import os
import gc
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import os


def to_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, "tolist"):  # numpy or jax arrays
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj

def Simulate_Mock_data_light_cone(
    file_path,
    savefile="Theory_Mock",
    start_idx=0,
    count=None,
    force_run=False,
    all_types=["CDM", "FDM", "SIDM"],
    repeat=1,   # Default fallback when axis_type cannot be determined
    repeat_long = 1,
    repeat_short = 3,
):
    if "high_observable" in file_path:
        simu_obj = "long_axis"
    elif "low_observable" in file_path:
        simu_obj = "short_axis"
    elif "cusp_all_observable" in file_path:
        simu_obj = "all_sim"
    else:
        simu_obj = "unknown"

    with fits.open(file_path) as hdul:
        data = hdul[1].data

    main_folder = f"{savefile}/{simu_obj}"
    ensure_dir(main_folder)

    total_len = len(data)
    end_idx = total_len if count is None else min(start_idx + count, total_len)

    total_time = 0.0
    processed_runs = 0
    planned_runs = 0

    for idx in range(start_idx, end_idx):
        # Build a temporary analyzer to detect axis_type without entering the run directory
        base_folder = f"{savefile}/{simu_obj}/{idx}"
        ensure_dir(base_folder)

        lens_params = get_Mock_lens_system_params_mul(data, idx, simu_obj)


        analyzer_probe = HaloAnalyzer(
            name=lens_params["name"],
            obj=lens_params,
            savefile=base_folder,    # Only for probing parameters
            globalfile=base_folder,
            IsMock=True
        )
        analyzer_probe.fix_parameter_mock()
        axis_type = analyzer_probe.obj.get("axis_type", None)
        del analyzer_probe

        # Determine repeat based on axis_type: long→1, short→3, otherwise use provided repeat
        if axis_type == "long_axis":
            dynamic_repeat = repeat_long
        elif axis_type == "short_axis":
            dynamic_repeat = repeat_short
        else:
            dynamic_repeat = repeat

        planned_runs += dynamic_repeat

        for run_id in range(dynamic_repeat):
            print("........." * 10)
            print(f"🔄 idx {idx} | run {run_id} generation start (axis_type={axis_type}, repeat={dynamic_repeat})")

            iteration_folder = f"{savefile}/{simu_obj}/{idx}/{run_id}"
            ensure_dir(iteration_folder)

            # Retrieve lens_params again to stay consistent with original logic
            lens_params = get_Mock_lens_system_params_mul(data, idx, simu_obj)
            use_name = lens_params["name"]

            # JSON parameter file check (same filename, one level deeper)
            param_file = os.path.join(iteration_folder, f"{simu_obj}_{idx}_params.json")
            json_missing = not os.path.exists(param_file)

            expected_files = [
                os.path.join(iteration_folder, f"{use_name}_{t}_alpha_mul_ray.npz")
                for t in all_types
            ]
            missing_types = [
                t for t, f in zip(all_types, expected_files) if not os.path.exists(f)
            ]

            # Skip when all alpha files and JSON exist unless force_run
            if not missing_types and not json_missing and not force_run:
                print(f"✔️ idx {idx} | run {run_id} all types and params exist, skipping")
                processed_runs += 1
                continue

            start_time = time.time()

            analyzer = HaloAnalyzer(
                name=lens_params["name"],
                obj=lens_params,
                savefile=iteration_folder,
                globalfile=iteration_folder,
                IsMock=True
            )
            analyzer.fix_parameter_mock()

            # Create parameter file if missing
            if json_missing or force_run:
                with open(param_file, "w") as f:
                    json.dump(to_serializable(analyzer.obj), f, indent=2)
                print(f"📝 Wrote parameter file: {param_file}")

            if force_run:
                print(f"🚨 idx {idx} | run {run_id}: forcing all types to regenerate")
                missing_types = all_types
            else:
                print(f"🚀 idx {idx} | run {run_id}: missing {missing_types}, regenerating")

            if "FDM" in missing_types:
                analyzer.Simulation_FDM_r()

            if "WDM" in missing_types:
                wdm_pkl = os.path.join(iteration_folder, f"{analyzer.name}_halolist_WDM.pkl")
                ifgenerate = not os.path.exists(wdm_pkl)
                analyzer.Simulation_WDM(ifgenerate=ifgenerate)

            if any(t in missing_types for t in ["CDM", "SIDM_col"]):
                cdm_pkl = os.path.join(iteration_folder, f"{analyzer.name}_halolist_sigma0_8000_w0_6.pkl")
                ifgenerate = not os.path.exists(cdm_pkl)
                analyzer.Simulation_CDM_SIDM(ifgenerate=ifgenerate)

            if "SIDM" in missing_types:
                sidm_pkl = os.path.join(iteration_folder, f"{analyzer.name}_halolist_sigma0_8000_w0_6.pkl")
                ifgenerate = not os.path.exists(sidm_pkl)
                analyzer.Simulation_SIDM(ifgenerate=ifgenerate)
            analyzer._plot_kappa_multi_minus_main(all_types)

            del analyzer

            cleanup_simulation_directory(iteration_folder)

            end_time = time.time()
            elapsed = (end_time - start_time) / 3600
            total_time += elapsed
            processed_runs += 1


            print(f"📊 Total elapsed runtime: {total_time:.2f} hours")



def _detect_repeat_from_existing_runs(base_folder: str) -> int:
    """
    Determine repeat count from existing numeric subdirectories (0,1,2,...).
    Default to 1 when no numeric subdirectories exist.
    """
    if not os.path.exists(base_folder):
        return 1
    run_ids = [
        int(d) for d in os.listdir(base_folder)
        if d.isdigit() and os.path.isdir(os.path.join(base_folder, d))
    ]
    return len(run_ids) if len(run_ids) > 0 else 1


def Simulate_Mock_data_mcmc(
    file_path,
    savefile="Theory_Mock",
    start_idx=0,
    count=None,
    force_run=False,
    repeat=1,   # Overwritten by auto-detection based on existing run directories
):
    if "high_observable" in file_path:
        simu_obj = "long_axis"
    elif "low_observable" in file_path:
        simu_obj = "short_axis"
    elif "cusp_all_observable" in file_path:
        simu_obj = "all_sim"
    else:
        simu_obj = "unknown"

    with fits.open(file_path) as hdul:
        data = hdul[1].data

    main_folder = f"{savefile}/{simu_obj}"
    ensure_dir(main_folder)

    all_types = ["CDM", "FDM", "SIDM"]

    total_len = len(data)
    end_idx = total_len if count is None else min(start_idx + count, total_len)

    total_time = 0.0
    processed_runs = 0
    planned_runs = 0

    for idx in range(start_idx, end_idx):
        base_folder = f"{savefile}/{simu_obj}/{idx}"
        ensure_dir(base_folder)

        # Auto-detect repeat count from existing run subdirectories
        dynamic_repeat = _detect_repeat_from_existing_runs(base_folder)
        planned_runs += dynamic_repeat

        for run_id in range(dynamic_repeat):
            print("........." * 10)
            print(f"🔄 idx {idx} | run {run_id} MCMC start (repeat={dynamic_repeat})")
            iteration_folder = os.path.join(base_folder, f"{run_id}")
            ensure_dir(iteration_folder)

            lens_params = get_Mock_lens_system_params(data, idx, simu_obj)
            use_name = lens_params["name"]

            expected_files = [
                os.path.join(iteration_folder, f"{use_name}_{t}_Rcusp_phi.npz")
                for t in all_types
            ]
            missing_types = [
                t for t, f in zip(all_types, expected_files) if not os.path.exists(f)
            ]
            if not missing_types and not force_run:
                print(f"✔️ idx {idx} | run {run_id} all types complete, skipping")
                processed_runs += 1
                continue

            start_time = time.time()

            analyzer = HaloAnalyzer(
                name=lens_params["name"],
                obj=lens_params,
                savefile=iteration_folder,
                globalfile=iteration_folder,
                IsMock=True
            )
            analyzer.fix_parameter_mock()
            print(analyzer.obj)

            if force_run:
                print(f"🚨 idx {idx} | run {run_id} MCMC: forcing all types to rerun")
                missing_types = all_types
            else:
                print(f"🚀 idx {idx} | run {run_id} MCMC: missing {missing_types}, rerunning")

            for sim_type in missing_types:
                situation = analyzer.Simulation_MCMC_Mock(sim_type)
                if situation is None:
                    print(f"Skipping {sim_type} because Simulation_MCMC_Mock returned None.")
                    continue

            del analyzer
            cleanup_simulation_directory(iteration_folder)

            end_time = time.time()
            elapsed = (end_time - start_time) / 3600
            total_time += elapsed
            processed_runs += 1

            remaining_runs = max(planned_runs - processed_runs, 0)
            est_per_run = total_time / processed_runs if processed_runs > 0 else 0.0
            eta_hours = remaining_runs * est_per_run

            print(f"⏱️ idx {idx} | run {run_id} MCMC duration: {elapsed:.2f} hours")
            print(f"📊 Total elapsed runtime: {total_time:.2f} hours")
            print(f"📅 Estimated remaining runtime: {eta_hours:.2f} hours")


def Simulate_Mock_data_mcmc_each_phibin(
    file_path,
    savefile="Theory_Mock",
    start_idx=0,
    count=None,
    force_run=False,
    clear_jax_cache=True,     # Clear JAX cache after each iteration
    block_until_ready=True,   # Wait for device computation before releasing objects
    repeat=1,                 # Overwritten by auto-detection based on existing run directories
):
    if "high_observable" in file_path:
        simu_obj = "long_axis"
    elif "low_observable" in file_path:
        simu_obj = "short_axis"
    elif "cusp_all_observable" in file_path:
        simu_obj = "all_sim"
    else:
        simu_obj = "unknown"

    with fits.open(file_path) as hdul:
        data = hdul[1].data

    main_folder = f"{savefile}/{simu_obj}"
    ensure_dir(main_folder)

    all_types = ["None","CDM", "FDM", "SIDM"]

    total_len = len(data)
    end_idx = total_len if count is None else min(start_idx + count, total_len)

    total_time = 0.0
    processed_runs = 0
    planned_runs = 0

    # Fix phi_bins ahead of time to keep shapes/types stable and reduce recompilation
    delta_phi = 5
    phi_bins = np.arange(20, 141, 2 * delta_phi, dtype=np.int32)

    for idx in range(start_idx, end_idx):
        base_folder = f"{savefile}/{simu_obj}/{idx}"
        ensure_dir(base_folder)

        # Auto-detect repeat count from existing run subdirectories
        dynamic_repeat = _detect_repeat_from_existing_runs(base_folder)
        planned_runs += dynamic_repeat

        for run_id in range(dynamic_repeat):
            print("........." * 10)
            print(f"🔄 idx {idx} | run {run_id} MCMC start (repeat={dynamic_repeat})")
            iteration_folder = os.path.join(base_folder, f"{run_id}")
            ensure_dir(iteration_folder)

            lens_params = get_Mock_lens_system_params_mul(data, idx, simu_obj)
            use_name = lens_params["name"]

            # Check which types are missing for this run_id
            missing_types = []
            for t in all_types:
                all_bins_exist = True
                for phi_center in phi_bins:
                    rcusp_file = os.path.join(
                        iteration_folder, f"{use_name}_{t}_{int(phi_center)}_Rcusp_phi.npz"
                    )
                    if not os.path.exists(rcusp_file):
                        all_bins_exist = False
                        break
                if not all_bins_exist:
                    missing_types.append(t)

            if not missing_types and not force_run:
                print(f"✔️ idx {idx} | run {run_id} all types already finished, skipping")
                processed_runs += 1
                continue

            start_time = time.time()

            analyzer = HaloAnalyzer(
                name=lens_params["name"],
                obj=lens_params,
                savefile=iteration_folder,
                globalfile=iteration_folder,
                IsMock=True,
            )

            try:
                analyzer.fix_parameter_mock()
                print(analyzer.obj)

                if force_run:
                    print(f"🚨 idx {idx} | run {run_id} MCMC: forcing all types to rerun")
                    missing_types = all_types
                else:
                    print(f"🚀 idx {idx} | run {run_id} MCMC: missing {missing_types}, rerunning")

                # Keep dtype/shape consistent to avoid repeated JIT compilation
                for sim_type in missing_types:
                    for phi_center in phi_bins:
                        phi_center_i32 = np.int32(phi_center)
                        situation = analyzer.Simulation_MCMC_Mock_each_phibin(
                            sim_type, int(phi_center_i32), int(delta_phi)
                        )
                        # Optionally wait for device to finish to avoid buffer accumulation
                        if block_until_ready and hasattr(situation, "block_until_ready"):
                            try:
                                situation.block_until_ready()
                            except Exception:
                                pass

            finally:
                try:
                    if hasattr(analyzer, "__dict__"):
                        for k, v in list(analyzer.__dict__.items()):
                            analyzer.__dict__[k] = None
                except Exception:
                    pass
                del analyzer

                cleanup_simulation_directory(iteration_folder)

                try:
                    for d in jax.devices():
                        _ = d
                    jax.random.uniform(jax.random.PRNGKey(0), ()).block_until_ready()
                except Exception:
                    pass

                if clear_jax_cache:
                    try:
                        jax.clear_caches()
                    except Exception:
                        pass

                gc.collect()

            end_time = time.time()
            elapsed = (end_time - start_time) / 3600
            total_time += elapsed
            processed_runs += 1

            remaining_runs = max(planned_runs - processed_runs, 0)
            est_per_run = total_time / processed_runs if processed_runs > 0 else 0.0
            eta_hours = remaining_runs * est_per_run

            print(f"⏱️ idx {idx} | run {run_id} MCMC duration: {elapsed:.2f} hours")
            print(f"📊 Total elapsed runtime: {total_time:.2f} hours")
            print(f"📅 Estimated remaining runtime: {eta_hours:.2f} hours")

if __name__ == "__main__":
    import argparse
    import os
    from cal_mul_fits import find_free_gpu

    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=str, default="Theory_Mock/cusp_all_observable_multipole.fits")
    parser.add_argument("--start-idx", type=int, required=True)
    parser.add_argument("--count", type=int, default=62)
    parser.add_argument("--gpu", type=str, default=None)  # Optional: explicitly select GPU
    args = parser.parse_args()

    # Prefer explicit --gpu, otherwise choose via find_free_gpu()
    gpu = args.gpu if args.gpu is not None else find_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    print(f"⚙️ Using GPU for JAX: {gpu}")

    # Simulate_Mock_data_light_cone(
    #     args.fits,
    #     start_idx=args.start_idx,
    #     count=args.count
    # )

    # Simulate_Mock_data_light_cone("Theory_Mock/cusp_all_observable_multipole.fits", start_idx=246, count=5)

    
    # Iterate over mock phi bins
    Simulate_Mock_data_mcmc_each_phibin(
       args.fits,
       start_idx=args.start_idx,
       count=args.count
    )

    # Simulate_Mock_data_mcmc_each_phibin(
    #     "Theory_Mock/cusp_high_observable.fits",
    #     start_idx=240,
    #     count=30
    # )
