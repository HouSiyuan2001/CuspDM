# utilize.py
import os
import shutil
import subprocess

def has_nvidia_smi() -> bool:
    """Return True if nvidia-smi exists and can be executed."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_output("nvidia-smi -L", shell=True)
        return True
    except Exception:
        return False
def has_apple_metal() -> bool:
    import platform
    if platform.system() != "Darwin":
        return False
    try:
        import jax
        return any("metal" in str(d).lower() for d in jax.devices())
    except Exception:
        return False

def find_free_gpu(verbose: bool = True) -> str:
    """
    Pick a GPU id using nvidia-smi.
    Only valid when has_nvidia_smi() is True.
    """
    mem_output = subprocess.check_output(
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits",
        shell=True
    )
    mem_info = [line.split(",") for line in mem_output.decode().splitlines()]
    gpu_ids = [int(x[0].strip()) for x in mem_info]
    used = [int(x[1].strip()) for x in mem_info]
    total = [int(x[2].strip()) for x in mem_info]
    mem_ratio = [u / t for u, t in zip(used, total)]

    if verbose:
        print("=== GPU memory usage ===")
        for gid, u, t, r in zip(gpu_ids, used, total, mem_ratio):
            print(f"GPU {gid}: {u}/{t} MiB ({r:.1%})")

    # If all GPUs are >30% memory usage, pick the one with least memory used
    if all(r > 0.3 for r in mem_ratio):
        min_mem_gpu = gpu_ids[used.index(min(used))]
        if verbose:
            print(f"All GPUs >30% memory usage, choosing GPU with least memory used: {min_mem_gpu}")
        return str(min_mem_gpu)

    util_output = subprocess.check_output(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
        shell=True
    )
    utils = [int(x.strip()) for x in util_output.decode().splitlines()]
    min_util_gpu = gpu_ids[utils.index(min(utils))]

    if verbose:
        print("=== GPU utilization ===")
        for gid, u in zip(gpu_ids, utils):
            print(f"GPU {gid}: {u}%")
        print(f"Selected GPU with lowest utilization: {min_util_gpu}")

    return str(min_util_gpu)


def apply_compute_env(mode: str, gpu: str | None = None, verbose: bool = True) -> None:
    mode = mode.lower().strip()

    if mode == "cpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        if verbose:
            print("⚙️ Compute mode: CPU")
        return

    if mode == "gpu":
        if gpu is None:
            raise ValueError("gpu id must be provided when mode='gpu'")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        if verbose:
            print(f"⚙️ Compute mode: NVIDIA GPU {gpu}")
        return

    if mode == "metal":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["JAX_PLATFORM_NAME"] = "metal"
        os.environ.setdefault("JAX_ENABLE_X64", "0")
        if verbose:
            print("⚙️ Compute mode: Apple Metal GPU")
        return

    raise ValueError(f"Unknown mode {mode}")


def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
