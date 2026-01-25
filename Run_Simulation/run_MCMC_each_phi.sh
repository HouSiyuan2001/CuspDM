#!/usr/bin/env bash
set -euo pipefail

# -------- User configurable --------
SCRIPT="Mock_sim2.py"
FITS="Theory_Mock/cusp_all_observable.fits"
COUNT=23
NWIN=4                    # Number of screens (ideally ≤ available GPUs)
STARTIDX=240
LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOGDIR"
# ---------------------------

# Use Python to pick the best GPU not already assigned by this script
choose_gpu_excluding() {
  local excluded_csv="${1:-}"   # e.g. "0,2"
  python - <<'PY'
import os, sys, subprocess

def parse_list(s):
    return set(int(x) for x in s.split(",") if x.strip().isdigit()) if s else set()

excluded = parse_list(os.environ.get("EXCLUDED", ""))

def pick(excluded):
    try:
        mem = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits",
            shell=True, stderr=subprocess.STDOUT
        ).decode().strip().splitlines()
        if not mem:
            return None
        mem = [list(map(str.strip, l.split(","))) for l in mem if l.strip()]
        gpu_ids = [int(x[0]) for x in mem]
        used = [int(x[1]) for x in mem]
        total = [int(x[2]) for x in mem]
        ratio = [ (u / t) if t>0 else 1.0 for u, t in zip(used, total) ]

        util = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            shell=True, stderr=subprocess.STDOUT
        ).decode().strip().splitlines()
        utils = [int(x.strip() or "100") for x in util]

        candidates = [gid for gid in gpu_ids if gid not in excluded]
        if not candidates:
            candidates = gpu_ids

        def score(gid):
            i = gpu_ids.index(gid)
            return (ratio[i], utils[i], used[i], gid)

        best = min(candidates, key=score)
        print("=== GPU summary ===", file=sys.stderr)
        for gid, r, u, m, t in zip(gpu_ids, ratio, utils, used, total):
            flag = "(excluded)" if gid in excluded else ""
            print(f"GPU {gid}: mem {m}/{t} MiB ({r:.1%}), util {u}% {flag}", file=sys.stderr)
        print(f"Selected GPU: {best}", file=sys.stderr)
        return best
    except Exception as e:
        print("GPU query failed:", e, file=sys.stderr)
        return None

gid = pick(excluded)
print("" if gid is None else gid)
PY
}

ASSIGNED=()

for ((i=0; i<NWIN; i++)); do
  start_idx=$(( i * COUNT + STARTIDX))
  sname="lc_${i}"

  if (( ${#ASSIGNED[@]} > 0 )); then
    EXCLUDED_STR=$(IFS=, ; echo "${ASSIGNED[*]}")
  else
    EXCLUDED_STR=""
  fi

  chosen_gpu="$(EXCLUDED="${EXCLUDED_STR}" choose_gpu_excluding 2> >(sed "s/^/[GPU-PICK ${sname}] /" >&2))"
  if [[ -z "${chosen_gpu}" ]]; then
    chosen_gpu=$(( i % 4 ))
    echo "[GPU-PICK ${sname}] Query failed, falling back to round-robin GPU: ${chosen_gpu}"
  else
    echo "[GPU-PICK ${sname}] Selected GPU: ${chosen_gpu}"
  fi
  ASSIGNED+=("${chosen_gpu}")

  # Launch screen and periodically truncate logs (hourly)
  screen -dmLS "${sname}" bash -lc "
    export CUDA_VISIBLE_DEVICES=${chosen_gpu}
    export JAX_PLATFORM_NAME=gpu
    (
      while true; do
        sleep 3600
        : > ${LOGDIR}/${sname}.log
      done
    ) &
    python ${SCRIPT} \
      --fits ${FITS} \
      --start-idx ${start_idx} \
      --count ${COUNT} \
      --gpu ${chosen_gpu} \
      |& tee -a ${LOGDIR}/${sname}.log
  "
  echo "Launched screen ${sname} (start_idx=${start_idx}) on GPU ${chosen_gpu} → log: ${LOGDIR}/${sname}.log"
done

echo "All ${NWIN} screens launched. Use: screen -ls / screen -r lc_0"
