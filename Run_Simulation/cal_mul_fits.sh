#!/usr/bin/env bash
set -euo pipefail

SCRIPT="cal_mul_fits.py"      # ← Replace with your main program filename
NPROC=24 # Number of processes to start
NUM_SIM=10000
FIX=2.5
NNN=1000
FITS="Theory_Mock/lensed_qso_mock_multipole_temp.fits"

NGPU=4                       # GPU count on the machine (A100 ×4)
LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

for ((i=0; i<NPROC; i++)); do
  gpu=$(( i % NGPU ))       # Round-robin assignment to GPU IDs
  sname="calc_${i}"
  screen -dmLS "$sname" bash -lc "
    export CUDA_VISIBLE_DEVICES=${gpu}
    export JAX_PLATFORM_NAME=gpu
    # Optional: add env activation command, e.g., conda activate xxx || true
    python ${SCRIPT} \
      --sim-idx ${i} \
      --num-sim ${NUM_SIM} \
      --fix ${FIX} \
      --nnn ${NNN} \
      --fits ${FITS} \
      --gpu ${gpu} \
      |& tee ${LOGDIR}/${sname}.log
  "
  echo "Launched screen ${sname} on GPU ${gpu}  → log: ${LOGDIR}/${sname}.log"
done

echo "All ${NPROC} screens launched. Use:  screen -ls  /  screen -r calc_0"
