#!/usr/bin/env bash
# Run one tuning job from ${TUNE_OUT:-exp/tuning}/jobs.tsv.  Usage: bash scripts/tuning/run_job.sh <job_id>
# SLURM array: JOB=${SLURM_ARRAY_TASK_ID}.  Skips jobs whose outputs all exist.
set -euo pipefail
cd "$(dirname "$0")/../.."
JOB=${1:-${SLURM_ARRAY_TASK_ID:?give a job id}}
LINE=$(awk -F'\t' -v j="$JOB" 'NR>1 && $1==j' ${TUNE_OUT:-exp/tuning}/jobs.tsv)
[ -n "$LINE" ] || { echo "no job $JOB"; exit 1; }
MODEL=$(echo "$LINE" | cut -f2); CFG=$(echo "$LINE" | cut -f3)
OUTPUTS=$(echo "$LINE" | cut -f5); CMD=$(echo "$LINE" | cut -f6-)
done_all=1
IFS=';' read -ra OUTS <<< "$OUTPUTS"
for o in "${OUTS[@]}"; do [ -f "${o#*=}" ] || done_all=0; done
if [ $done_all = 1 ]; then echo "job $JOB ($MODEL/$CFG) already done"; exit 0; fi
mkdir -p ${TUNE_OUT:-exp/tuning}/logs
echo "=== job $JOB $MODEL/$CFG start $(date)"
eval "$CMD" > "${TUNE_OUT:-exp/tuning}/logs/${MODEL}_${CFG}.log" 2>&1
echo "=== job $JOB $MODEL/$CFG end $(date)"
