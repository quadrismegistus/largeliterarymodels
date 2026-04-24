# Running SocialNetworkTask on Cambridge CSD3

You have 3,000 A100 GPU-hours on CSD3. This guide gets you from zero to
bulk social network extraction across all of Chadwyck.

## What's happening

1. You submit a SLURM job that starts a **vLLM server** on an A100 node.
   vLLM loads a model (default: Qwen 3.6 27B) and exposes an OpenAI-compatible
   HTTP API on port 8000.

2. You run the **batch script** which loops through texts in a Chadwyck
   subcollection, calling that API for each one. Each text produces a JSON
   file with characters, relations, events, dialogue, and narrative summaries.

3. Everything is **resumable**. If a job dies, re-run the same command.
   Completed texts are skipped (output file exists), and partially-completed
   texts resume from the last finished chunk (cached by hashstash).

## One-time setup

SSH into CSD3 and set up the Python environments:

```bash
# vLLM environment (GPU node will use this)
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate
pip install vllm

# Project environment (batch script uses this)
python3 -m venv ~/llm_env
source ~/llm_env/bin/activate
pip install git+https://github.com/quadrismegistus/largeliterarymodels.git
pip install lltk-dh  # for passage loading

# Clone the repo for the scripts
git clone https://github.com/quadrismegistus/largeliterarymodels.git ~/largeliterarymodels
cd ~/largeliterarymodels
```

Edit `launch_vllm.sbatch` and replace `PLACEHOLDER_ACCOUNT` with your
CSD3 account code (e.g. `--account=CASTLE-SL3-GPU`). Also update the
venv path in `run_batch.sh` if you put it somewhere other than `~/llm_env`.

## Running

### Step 1: Start the vLLM server

```bash
mkdir -p logs
sbatch scripts/hpc/launch_vllm.sbatch
```

This queues a 36-hour GPU job. Monitor it:

```bash
squeue -u $USER                    # check job status
tail -f logs/vllm_<JOBID>.out      # watch for "Uvicorn running on http://0.0.0.0:8000"
```

Wait until you see the "Uvicorn running" line — model loading takes 2-5 minutes.

### Step 2: Run the batch

**Option A — same node (simplest):**

```bash
# Find the job ID from squeue, then:
srun --jobid=<JOBID> --overlap bash scripts/hpc/run_batch.sh Early_English_Prose_Fiction
```

The script waits for vLLM to be healthy, then starts processing texts.

**Option B — from your laptop via SSH tunnel:**

```bash
# Terminal 1: tunnel to the compute node (find node name in the vLLM log)
ssh -L 8000:<NODE>:8000 <username>@login.hpc.cam.ac.uk

# Terminal 2: run locally
VLLM_BASE_URL=http://localhost:8000/v1 \
python scripts/batch_social_network.py \
    --subcollection Early_English_Prose_Fiction \
    --model vllm/Qwen/Qwen3.6-27B
```

### Step 3: Collect results

Output JSONs land in `data/social_network_*.json`. Copy them back to
your laptop:

```bash
scp 'csd3:~/largeliterarymodels/data/social_network_*.json' data/
```

## Subcollections and time estimates

At ~120s/chunk on A100 (likely faster with vLLM batching):

| Subcollection | Texts | Passages | Est. time |
|---|---|---|---|
| Early_English_Prose_Fiction | 110 | 11,964 | ~5-10 hours |
| Eighteenth-Century_Fiction | 95 | 21,911 | ~10-18 hours |
| Nineteenth-Century_Fiction | 250 | 75,617 | ~1-2 days |
| Early_American_Fiction | 882 | 123,030 | ~2-4 days |

A single 36-hour job should cover Early English + 18C Fiction. For 19C
and Early American, submit multiple jobs or chain them.

## Using a different model

```bash
# Llama 3.1 70B (fits A100 80GB at 4-bit)
sbatch scripts/hpc/launch_vllm.sbatch meta-llama/Meta-Llama-3.1-70B-Instruct
bash scripts/hpc/run_batch.sh Early_English_Prose_Fiction vllm/meta-llama/Meta-Llama-3.1-70B-Instruct
```

## Troubleshooting

**vLLM OOM**: Reduce `--gpu-memory-utilization` in the sbatch file (default 0.90).

**Job killed at 36h wall time**: Just re-run. The batch script skips
completed texts and hashstash resumes mid-text.

**"Connection refused" from batch script**: vLLM isn't ready yet. The
`run_batch.sh` script auto-waits up to 10 minutes.

**lltk can't connect to ClickHouse**: CSD3 compute nodes may not have
outbound access. If so, pre-export the text list and passage data on
the login node, then pass `--text-id` for individual texts or modify
the batch script to read from a local manifest CSV.
