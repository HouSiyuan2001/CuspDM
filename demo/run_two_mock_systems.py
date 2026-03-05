import argparse
import os
import random
import sys

from astropy.io import fits


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB = os.path.join(ROOT, "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

from utilize import (
    apply_compute_env,
    find_free_gpu,
    has_apple_metal,
    has_nvidia_smi,
)


def apply_compute_mode(mode: str, gpu: str | None, allow_metal: bool) -> None:
    if mode == "auto":
        if os.environ.get("JAX_PLATFORM_NAME") or os.environ.get("CUDA_VISIBLE_DEVICES"):
            print("⚙️ Using existing compute environment from env vars")
            return
        if has_nvidia_smi():
            gpu_id = gpu if gpu is not None else find_free_gpu()
            apply_compute_env("gpu", gpu=gpu_id, verbose=True)
        elif has_apple_metal() and allow_metal:
            apply_compute_env("metal", verbose=True)
        else:
            if has_apple_metal() and not allow_metal:
                print("⚠️ Metal detected but disabled; falling back to CPU")
            apply_compute_env("cpu", verbose=True)
        return

    if mode == "gpu":
        gpu_id = gpu if gpu is not None else find_free_gpu()
        apply_compute_env("gpu", gpu=gpu_id, verbose=True)
    else:
        if mode == "metal" and not allow_metal:
            print("⚠️ Metal disabled; using CPU instead")
            apply_compute_env("cpu", verbose=True)
        else:
            apply_compute_env(mode, verbose=True)


def parse_indices(raw: str) -> list[int]:
    indices: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            indices.append(int(chunk))
        except ValueError as exc:
            raise ValueError(f"Invalid index value: {chunk}") from exc
    if not indices:
        raise ValueError("No valid indices provided.")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fits",
        type=str,
        default="Data/cusp_all_observable_multipole.fits",
        help="Input FITS catalog.",
    )
    parser.add_argument("--num-systems", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["lightcone", "mcmc_each_phi", "both"],
        help="Which calculations to run.",
    )
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
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--savefile", type=str, default="demo/Theory_Mock")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of catalog indices to run (overrides random selection).",
    )
    args = parser.parse_args()

    allow_metal = args.allow_metal or os.environ.get("RCUSP_ALLOW_METAL") == "1"
    apply_compute_mode(args.compute, args.gpu, allow_metal)

    from Simulation_Rcusp import (
        Simulate_Mock_data_light_cone,
        Simulate_Mock_data_mcmc_each_phibin,
    )

    with fits.open(args.fits) as hdul:
        data = hdul[1].data
        total = len(data)

    if total == 0:
        print("No systems found in FITS file.")
        return

    if args.indices:
        indices = parse_indices(args.indices)
        invalid = [idx for idx in indices if idx < 0 or idx >= total]
        if invalid:
            raise ValueError(f"Index out of range (0..{total - 1}): {invalid}")
        print(f"Using provided indices: {indices}")
    else:
        num_pick = min(args.num_systems, total)
        rng = random.Random(args.seed)
        indices = rng.sample(range(total), k=num_pick)
        print(f"Selected indices: {indices}")

    for idx in indices:
        if args.mode in ("lightcone", "both"):
            Simulate_Mock_data_light_cone(
                args.fits,
                savefile=args.savefile,
                start_idx=idx,
                count=1,
            )
        if args.mode in ("mcmc_each_phi", "both"):
            Simulate_Mock_data_mcmc_each_phibin(
                args.fits,
                savefile=args.savefile,
                start_idx=idx,
                count=1,
            )


if __name__ == "__main__":
    main()
