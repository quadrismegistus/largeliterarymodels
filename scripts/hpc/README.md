# Running SocialNetworkTask on Cambridge CSD3

You have 3,000 A100 GPU-hours on CSD3. This guide gets you from zero to
bulk social network extraction across pre-1800 English prose fiction.

## Overview

1. Export passage data locally (where ClickHouse lives)
2. Rsync passages to CSD3
3. Launch a Jupyter notebook on an A100 node via `salloc`
4. Start vLLM + run the batch from notebook cells
5. Rsync results back

Everything is **resumable**. If the 12h wall time kills your job, just
relaunch — completed texts are skipped (output file exists).

## Storage layout

Home directory is **50GB** — don't put venvs or model weights there.
Use `/rds/user/$USER/hpc-work/` (1TB, not backed up) for everything heavy.

```
~/rds/hpc-work/venvs/llm/          # Python venv (vllm + largeliterarymodels)
~/rds/hpc-work/largeliterarymodels/ # git clone of this repo
~/rds/hpc-work/texts/              # exported JSONL passages
~/rds/hpc-work/output/             # social network results
~/rds/hpc-work/.cache/huggingface/ # model weights (~50GB)
```

## One-time setup

### 1. Check available modules

SSH into CSD3 and check what's available:

```bash
module avail python    # need 3.9+ for vLLM
module avail cuda      # need 12.x for vLLM
module avail miniconda # fallback if no python 3.9+
```

### 2. Create the environment

Build on an ampere node (Rocky Linux 8, different arch from login nodes):

```bash
sintr -t 1:0:0 --gres=gpu:1 -A HEUSER-SL3-GPU -p ampere --nodes=1

module purge
module load rhel8/default-amp
module load python/3.11.9/gcc/nptrdpll
module load cuda/12.1

python3 -m venv ~/rds/hpc-work/venvs/llm
source ~/rds/hpc-work/venvs/llm/bin/activate
pip install --upgrade pip
pip install vllm jupyter ipykernel
pip install git+https://github.com/quadrismegistus/largeliterarymodels.git
```

### 3. Register Jupyter kernel

```bash
ipython kernel install --user --name=llm
```

### 4. Redirect HuggingFace cache to rds

```bash
mkdir -p ~/rds/hpc-work/.cache/huggingface
ln -sf ~/rds/hpc-work/.cache/huggingface ~/.cache/huggingface
```

### 5. Pre-download the model

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B')"
```

This takes a while (~50GB) but only needs to happen once.

### 6. Clone the repo

```bash
cd ~/rds/hpc-work
git clone https://github.com/quadrismegistus/largeliterarymodels.git
```

## Export passages (on your laptop)

Run locally where ClickHouse is available:

```bash
# Single subcollection
python scripts/hpc/export_passages.py \
    --subcollection Early_English_Prose_Fiction \
    --out texts/

# All pre-1800 English fiction
python scripts/hpc/export_passages.py \
    --query "genre='Fiction' AND year<1800 AND lang='en'" \
    --out texts/

# Upload to CSD3
rsync -avz texts/ csd3:~/rds/hpc-work/texts/
```

Each JSONL file has one passage per line: `{"seq": N, "text": "...", "n_words": N}`.

## Running (Jupyter on GPU node)

### Launch the notebook

```bash
salloc -t 12:0:0 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 \
    --cpus-per-task=32 -p ampere -A HEUSER-SL3-GPU \
    jupyter notebook --no-browser --ip=* --port=8081
```

Note the `gpu-q-XX` node name from the output. Then from your laptop:

```bash
ssh -L 8081:gpu-q-XX:8081 -l USERNAME login-q-1.hpc.cam.ac.uk
```

Open the URL Jupyter printed (replace `gpu-q-XX` with `127.0.0.1`).
Select the `llm` kernel when creating a new notebook.

### In the notebook

See `scripts/hpc/run_social_network.ipynb` for a ready-to-run notebook,
or run these cells:

```python
# Cell 1: Start vLLM server
import subprocess, os
proc = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen3.6-27B",
    "--port", "8000",
    "--host", "127.0.0.1",
    "--enable-prefix-caching",
    "--gpu-memory-utilization", "0.90",
    "--max-model-len", "32768",
    "--disable-log-requests",
], stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT)
print(f"vLLM starting (pid {proc.pid}), check vllm.log")
```

```python
# Cell 2: Wait for vLLM to be ready
import time, urllib.request
for i in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health")
        print("vLLM ready!")
        break
    except:
        if i % 6 == 0: print(f"Waiting... ({i*5}s)")
        time.sleep(5)
else:
    print("vLLM failed to start — check vllm.log")
```

```python
# Cell 3: Run batch (4 workers to saturate GPU)
rds = os.path.expanduser("~/rds/hpc-work")
!python {rds}/largeliterarymodels/scripts/batch_social_network.py \
    --text-dir {rds}/texts/ \
    --output-dir {rds}/output/ \
    --model vllm-qwen36 \
    --workers 4
```

### After the run

```bash
# On your laptop — pull results back
rsync -avz csd3:~/rds/hpc-work/output/ output/
```

## Time estimates

At ~18-35s/chunk on A100 with vLLM (5-10x faster than local):

| Scope | Texts | Est. time (1 GPU) |
|---|---|---|
| Early English Prose Fiction | 110 | 6-12 hours |
| All pre-1800 fiction (with passages) | 251 | 1-3 days |
| All pre-1800 fiction (full 1,966) | 1,966 | 4-5 days |

SL3 wall time limit is 12 hours per job. Just relaunch — skip-existing
handles resumption automatically.

## Alternative: sbatch (no Jupyter)

If you prefer batch submission over interactive Jupyter:

```bash
# Step 1: submit vLLM server
sbatch scripts/hpc/launch_vllm.sbatch

# Step 2: run batch on same node
srun --jobid=<JOBID> --overlap bash scripts/hpc/run_batch.sh
```

## Troubleshooting

**vLLM OOM**: Reduce `--gpu-memory-utilization` (default 0.90) or
`--max-model-len` (default 32768).

**Job killed at 12h**: Just relaunch. Completed texts are skipped,
partially-completed texts resume from the last finished chunk (hashstash cache).

**"Connection refused"**: vLLM isn't ready yet. Model loading takes 2-5 min.

**Python too old**: Use `module load python/3.11.9/gcc/nptrdpll` (confirmed
available on CSD3 as of April 2026).

**Home quota exceeded**: Move venvs and HF cache to `~/rds/hpc-work/`.
Symlink `~/.cache/huggingface` to rds.
