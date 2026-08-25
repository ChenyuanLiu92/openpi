#!/usr/bin/env bash
# Slurm resource flags are injected by submit_pretrain_slurm.py.
set -uo pipefail

OPENPI_CONFIG=$1
OPENPI_GPUS_PER_NODE=$2
OPENPI_MAX_RESTARTS=$3
OPENPI_OVERRIDES_JSON=$4
OPENPI_COORDINATOR_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
OPENPI_COORDINATOR_PORT=$((20000 + SLURM_JOB_ID % 20000))
OPENPI_COORDINATOR_ADDRESS="${OPENPI_COORDINATOR_HOST}:${OPENPI_COORDINATOR_PORT}"
export OPENPI_CONFIG OPENPI_GPUS_PER_NODE OPENPI_OVERRIDES_JSON OPENPI_COORDINATOR_PORT OPENPI_COORDINATOR_ADDRESS

child_pid=""
preempted=0
forward_termination() {
  preempted=1
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap forward_termination USR1 TERM INT

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  uv run --group rlds python scripts/slurm_pretrain_rank.py &
child_pid=$!
wait "$child_pid"
status=$?

restart_count=${SLURM_RESTART_COUNT:-0}
if (( status != 0 || preempted != 0 )) && (( restart_count < OPENPI_MAX_RESTARTS )); then
  echo "Requeueing OpenPI job ${SLURM_JOB_ID}: status=${status}, restart=${restart_count}/${OPENPI_MAX_RESTARTS}" >&2
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
