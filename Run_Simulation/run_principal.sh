#!/usr/bin/env bash
set -euo pipefail

# -------- User configurable --------
SCRIPT="Get_principal.py"                    # Python entrypoint
DATAFOLDER="Theory_para/Data_single_para_m_use"  # Data & output folder
FILTERRULE="_Global_alpha.npz"               # Filename match rule
NPROC=4                                      # GPU/concurrent process count
LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
# ---------------------------

# Collect matching files
mapfile -t ALL_FILES < <(find "$DATAFOLDER" -type f -name "*${FILTERRULE}" | sort)
TOTAL=${#ALL_FILES[@]}
if (( TOTAL == 0 )); then
  echo "❌ No matching files (*.npz) in $DATAFOLDER"
  exit 1
fi

echo "✅ Found $TOTAL lens models, distributing to $NPROC processes"
FILES_PER_PROC=$(( (TOTAL + NPROC - 1) / NPROC ))  # Ceiling

# Evenly split tasks
for (( i=0; i<NPROC; i++ )); do
  splitfile="input_list_${i}.txt"
  > "$splitfile"
  for (( j=i; j<TOTAL; j+=NPROC )); do
    echo "${ALL_FILES[j]}" >> "$splitfile"
  done

  sname="mcmc_${i}"
  echo "→ Task $i: $(wc -l < "$splitfile") files ➜ log: ${LOGDIR}/${sname}.log"

  screen -dmLS "$sname" bash -lc "
    export CUDA_VISIBLE_DEVICES=${i}
    export JAX_PLATFORM_NAME=gpu
    python ${SCRIPT} --file-list ${splitfile} --save-folder ${DATAFOLDER} |& tee ${LOGDIR}/${sname}.log
  "
done

echo "🎬 All $NPROC tasks started"
echo "🧪 Check status: screen -ls"
echo "🔍 Inspect a task: screen -r mcmc_0"
