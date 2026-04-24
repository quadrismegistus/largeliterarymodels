#!/bin/bash
# ============================================================
# Run SocialNetworkTask batch on CSD3 against a local vLLM server.
#
# Two modes of operation:
#
# A) Same-node (recommended): submit as a SLURM job that runs alongside vLLM
#    sbatch scripts/hpc/launch_vllm.sbatch
#    # Wait for vLLM to start (check logs/vllm_*.out for "Uvicorn running")
#    # Then from the same node or via srun:
#    srun --jobid=<VLLM_JOB_ID> --overlap bash scripts/hpc/run_batch.sh Early_English_Prose_Fiction
#
# B) SSH tunnel from laptop:
#    ssh -L 8000:<COMPUTE_NODE>:8000 login.hpc.cam.ac.uk
#    # Then locally:
#    VLLM_BASE_URL=http://localhost:8000/v1 python scripts/batch_social_network.py \
#        --subcollection Early_English_Prose_Fiction --model vllm/qwen3.6-27b
#
# ============================================================

set -euo pipefail

SUBCOLLECTION="${1:?Usage: $0 <subcollection> [model]}"
MODEL="${2:-vllm/Qwen/Qwen3.6-27B}"

# Wait for vLLM to be ready
echo "Waiting for vLLM at localhost:8000..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM is ready."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: vLLM did not start within 10 minutes."
        exit 1
    fi
    sleep 5
done

# Activate project venv
source ~/llm_env/bin/activate  # adjust path as needed

export VLLM_BASE_URL="http://localhost:8000/v1"

echo "=========================================="
echo "Subcollection: $SUBCOLLECTION"
echo "Model: $MODEL"
echo "VLLM_BASE_URL: $VLLM_BASE_URL"
echo "=========================================="

python scripts/batch_social_network.py \
    --subcollection "$SUBCOLLECTION" \
    --model "$MODEL"
