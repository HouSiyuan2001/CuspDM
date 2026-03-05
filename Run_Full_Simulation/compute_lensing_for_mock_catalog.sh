#!/usr/bin/env bash
set -euo pipefail   # safer bash: stop on error / undefined variables

# ---- Main program and parameters ----
SCRIPT="lib/compute_lensing_for_mock_catalog.py"   # Python script to run
NPROC=4                # number of parallel jobs (screens)
NUM_SIM=10000           # simulations per job (used in python)
FIX=2.5  #rows per job = FIX * NUM_SIM
NNN=1000 #lensing grid
FITS="demo/Data/lensed_qso_mock_multipole_temp.fits"

# ---- Create log directory with timestamp ----
LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# ---- Detect NVIDIA GPUs automatically ----
NGPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')   # count GPUs
  fi
fi

# flag for GPU availability
HAS_NVIDIA=0
if [[ "$NGPU" -gt 0 ]]; then
  HAS_NVIDIA=1
fi

echo "HAS_NVIDIA=${HAS_NVIDIA}  NGPU=${NGPU}"

# Print running mode
if [[ "$HAS_NVIDIA" -eq 1 ]]; then
  echo "NVIDIA GPUs detected → using GPU round-robin scheduling"
else
  echo "No NVIDIA GPU detected → running on CPU only"
fi

# ---- Compute number of rows per job (same logic as python script) ----
limit_rows=$(python - <<PY
NUM_SIM=${NUM_SIM}
FIX=${FIX}
print(int(NUM_SIM*FIX))
PY
)

# ---- Launch jobs ----
for ((i=0; i<NPROC; i++)); do
  sname="calc_${i}"   # screen session name

  # Compute start/end indices for this job
  start_idx=$(python - <<PY
NUM_SIM=${NUM_SIM}
FIX=${FIX}
i=${i}
print(int(i*NUM_SIM*FIX))
PY
)
  end_idx=$(( start_idx + limit_rows - 1 ))

  # Output JSON file for this job
  OUTJSON="Run_Full_Simulation/Data/Data_json/calc_results_${start_idx}_${end_idx}.json"

  # ---- GPU mode ----
  if [[ "$HAS_NVIDIA" -eq 1 ]]; then
    gpu=$(( i % NGPU ))   # assign GPU in round-robin

    screen -dmLS "$sname" bash -lc "
      export CUDA_VISIBLE_DEVICES=${gpu}
      export JAX_PLATFORM_NAME=gpu
      python ${SCRIPT} \
        --sim-idx ${i} \
        --num-sim ${NUM_SIM} \
        --fix ${FIX} \
        --nnn ${NNN} \
        --fits ${FITS} \
        --gpu ${gpu} \
        --out-json ${OUTJSON} \
        |& tee ${LOGDIR}/${sname}.log
    "

    echo "Launched ${sname} on GPU ${gpu} → log: ${LOGDIR}/${sname}.log"

  # ---- CPU mode ----
  else
    screen -dmLS "$sname" bash -lc "
      python ${SCRIPT} \
        --sim-idx ${i} \
        --num-sim ${NUM_SIM} \
        --fix ${FIX} \
        --nnn ${NNN} \
        --fits ${FITS} \
        --out-json ${OUTJSON} \
        |& tee ${LOGDIR}/${sname}.log
    "

    echo "Launched ${sname} on CPU → log: ${LOGDIR}/${sname}.log"
  fi

done

echo "All ${NPROC} jobs launched."
echo "Use: screen -ls  to list sessions"
echo "Use: screen -r calc_0  to attach one job"