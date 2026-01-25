#!/usr/bin/env bash
set -euo pipefail

# -------- User configurable --------
SCRIPT="Mock_sim2.py"
FITS="Theory_Mock/cusp_all_observable.fits"
COUNT=1
NWIN=2                                # Prefer ≤ GPU count
LOGROOT="logs"                        # Shared log root
STARTIDX=157
LOGDIR="${LOGROOT}/$(date +%Y%m%d_%H%M%S)"
TRUNCATE_SECS=3600                    # How often to truncate the current log
RETENTION_HOURS=24                    # Retain logs newer than this (hours)
NGPU_FALLBACK=4                       # GPU count to round-robin if query fails
mkdir -p "$LOGDIR"
# Clean older logs and empty dirs (does not affect current batch)
find "$LOGROOT" -type f -name "*.log" -mmin +$((RETENTION_HOURS*60)) -delete 2>/dev/null || true
find "$LOGROOT" -type d -empty -mindepth 1 -delete 2>/dev/null || true
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
        # Query memory
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

        # Query utilization
        util = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            shell=True, stderr=subprocess.STDOUT
        ).decode().strip().splitlines()
        utils = [int(x.strip() or "100") for x in util]

        # Candidates = GPUs not excluded
        candidates = [gid for gid in gpu_ids if gid not in excluded] or gpu_ids

        # Score: memory ratio -> utilization -> used memory -> id
        def score(gid):
            i = gpu_ids.index(gid)
            return (ratio[i], utils[i], used[i], gid)

        best = min(candidates, key=score)
        # Diagnostics to stderr
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

# Track GPUs assigned by this script to avoid duplicates in a single loop
ASSIGNED=()  # bash array storing assigned GPU ids

for ((i=0; i<NWIN; i++)); do
  start_idx=$(( i * COUNT + STARTIDX ))
  sname="lc_${i}"
  logfile="${LOGDIR}/${sname}.log"

  # Pass assigned list to Python to pick different GPUs when possible
  if (( ${#ASSIGNED[@]} > 0 )); then
    EXCLUDED_STR=$(IFS=, ; echo "${ASSIGNED[*]}")
  else
    EXCLUDED_STR=""
  fi

  # Pick GPU
  chosen_gpu="$(EXCLUDED="${EXCLUDED_STR}" choose_gpu_excluding 2> >(sed "s/^/[GPU-PICK ${sname}] /" >&2))"
  if [[ -z "${chosen_gpu}" ]]; then
    # Fall back to round-robin when query fails
    chosen_gpu=$(( i % NGPU_FALLBACK ))
    echo "[GPU-PICK ${sname}] Query failed, using round-robin GPU: ${chosen_gpu}"
  else
    echo "[GPU-PICK ${sname}] Selected GPU: ${chosen_gpu}"
  fi
  ASSIGNED+=("${chosen_gpu}")

  # Launch screen with a lightweight log truncation loop
  screen -dmLS "${sname}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOGDIR}'
    touch '${logfile}'

    # Background: periodically truncate current log (tee handle unaffected)
    (
      while true; do
        sleep '${TRUNCATE_SECS}'
        : > '${logfile}' || true
      done
    ) &

    export CUDA_VISIBLE_DEVICES='${chosen_gpu}'
    export JAX_PLATFORM_NAME='gpu'
    # Optional: source ~/.bashrc && conda activate your_env || true

    python '${SCRIPT}' \
      --fits '${FITS}' \
      --start-idx '${start_idx}' \
      --count '${COUNT}' \
      --gpu '${chosen_gpu}' \
      |& tee -a '${logfile}'
  "

  echo "Launched screen ${sname} (start_idx=${start_idx}) on GPU ${chosen_gpu} → log: ${logfile}"
done

echo "All ${NWIN} screens launched. Use: screen -ls  /  screen -r lc_0"
