#!/bin/bash
#SBATCH --job-name=vllm-social-network
#SBATCH --partition=ampere
#SBATCH --account=HEUSER-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00  # SL3 max; change to 36:00:00 for SL1/2
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# ============================================================
# Launch vLLM server on a CSD3 A100 node.
#
# Usage:
#   sbatch scripts/hpc/launch_vllm.sbatch [model_name]
#
# Default model: Qwen/Qwen3.6-27B (fits A100 80GB comfortably)
# After job starts, note the compute node name from the log,
# then SSH tunnel from your laptop:
#
#   ssh -L 8000:<COMPUTE_NODE>:8000 login.hpc.cam.ac.uk
#
# Or run the batch script on the same node (see run_batch.sh).
# ============================================================

MODEL="${1:-Qwen/Qwen3.6-27B}"
PORT=8000

module purge
module load rhel8/default-amp
module load python/3.11.9/gcc/nptrdpll
module load cuda/12.1

# Activate venv (see README.md for setup)
source ~/rds/hpc-work/venvs/llm/bin/activate

mkdir -p logs

echo "=========================================="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Launch vLLM with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests
