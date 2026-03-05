if __name__ == "__main__":
      import argparse
      import os
      from utilize import (
         apply_compute_env,
         find_free_gpu,
         has_apple_metal,
         has_nvidia_smi,
      )

      parser = argparse.ArgumentParser()
      parser.add_argument("--fits", type=str, default="Theory_Mock/cusp_all_observable_multipole.fits")
      parser.add_argument("--start-idx", type=int, required=True)
      parser.add_argument("--count", type=int, default=62)
      parser.add_argument("--gpu", type=str, default=None)  # Optional: explicitly select GPU
      parser.add_argument(
         "--compute",
         type=str,
         default="auto",
         choices=["auto", "cpu", "gpu", "metal"],
         help="Compute backend selection. Defaults to auto-detect.",
      )
      parser.add_argument(
         "--allow-metal",
         action="store_true",
         help="Allow Apple Metal backend (experimental, may fail for float64).",
      )
      parser.add_argument(
         "--mode",
         type=str,
         default="mcmc_each_phi",
         choices=["lightcone", "mcmc_each_phi"],
         help="Which simulation pipeline to run.",
      )
      args = parser.parse_args()

      allow_metal = args.allow_metal or os.environ.get("RCUSP_ALLOW_METAL") == "1"

      if args.compute == "auto":
         if os.environ.get("JAX_PLATFORM_NAME") or os.environ.get("CUDA_VISIBLE_DEVICES"):
               print("⚙️ Using existing compute environment from env vars")
         elif has_nvidia_smi():
               gpu = args.gpu if args.gpu is not None else find_free_gpu()
               apply_compute_env("gpu", gpu=gpu, verbose=True)
         elif has_apple_metal() and allow_metal:
               apply_compute_env("metal", verbose=True)
         else:
               if has_apple_metal() and not allow_metal:
                     print("⚠️ Metal detected but disabled; falling back to CPU")
               apply_compute_env("cpu", verbose=True)
      elif args.compute == "gpu":
         gpu = args.gpu if args.gpu is not None else find_free_gpu()
         apply_compute_env("gpu", gpu=gpu, verbose=True)
      else:
         if args.compute == "metal" and not allow_metal:
               print("⚠️ Metal disabled; using CPU instead")
               apply_compute_env("cpu", verbose=True)
         else:
               apply_compute_env(args.compute, verbose=True)

      from Simulation_Rcusp import (
         Simulate_Mock_data_light_cone,
         Simulate_Mock_data_mcmc_each_phibin,
      )

      if args.mode == "lightcone":
         Simulate_Mock_data_light_cone(
               args.fits,
               start_idx=args.start_idx,
               count=args.count,
         )
      else:
         # Iterate over mock phi bins
         Simulate_Mock_data_mcmc_each_phibin(
               args.fits,
               start_idx=args.start_idx,
               count=args.count,
         )
