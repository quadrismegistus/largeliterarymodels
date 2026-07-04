#!/usr/bin/env python3
"""Vast.ai GPU instance manager for running social network extraction at scale.

Manages the full lifecycle: launch → setup → upload → run → status → download → stop.

State is stored in .vastai.json in the project root. Each command reads/writes it
so you can run them independently and resume after disconnects. If state is ever
lost or stale (e.g. a crashed launch), `sync` rebuilds it from `vastai show
instances`.

Prerequisites:
    pip install vastai
    vastai set api-key YOUR_KEY
    Upload SSH public key at https://cloud.vast.ai/manage-keys/

Usage:
    litmod cloud launch              # find + rent cheapest A100 80GB
    litmod cloud setup               # install vLLM + largeliterarymodels
    litmod cloud upload passages_c19  # rsync a passages dir
    litmod cloud run passages_c19     # start batch in tmux
    litmod cloud status              # check progress + cost
    litmod cloud download             # rsync results back
    litmod cloud stop                # destroy instance
    litmod cloud sync                # rebuild state from vast.ai
"""

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = PROJECT_ROOT / '.vastai.json'
REMOTE_WORK = '/workspace'
REMOTE_PASSAGES = f'{REMOTE_WORK}/passages'
REMOTE_RESULTS = f'{REMOTE_WORK}/results'
REMOTE_REPO = f'{REMOTE_WORK}/largeliterarymodels'
LOCAL_RESULTS = PROJECT_ROOT / 'data' / 'cloud_results'

VLLM_MODEL = 'cyankiwi/Qwen3.6-27B-AWQ-INT4'
VLLM_SERVED_NAME = 'qwen3.6-27b'
VLLM_PORT = 8000
BATCH_WORKERS = 4

DOCKER_IMAGE = 'vllm/vllm-openai:latest'
DISK_GB = 80
MIN_GPU_RAM = 79

# Single source of truth for batchable tasks. `litmod batch` (cli/main.py)
# imports SUMMARY_TASK_MAP; keep additions here so local + cloud stay in sync.
PASSAGE_TASKS = {'passage_setting', 'passage_narrativity'}
SUMMARY_TASK_MAP = {
    'plot_genre': 'PlotGenreTask',
    'subgenre': 'SubgenreTask',
    'character_type': 'CharacterTypeTask',
    'subgenre_modern': 'ModernSubgenreTask',
}

q = shlex.quote


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            print(f"Warning: corrupt state file {STATE_FILE} — ignoring it.",
                  file=sys.stderr)
            print("Run 'litmod cloud sync' to rebuild state from vast.ai.",
                  file=sys.stderr)
    return {}


def save_state(state):
    tmp = STATE_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(tmp, STATE_FILE)


def vastai(*args, capture=True, allow_fail=False):
    cmd = ['vastai'] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            if allow_fail:
                return None
            print(f"vastai error: {r.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        return r.stdout.strip()
    else:
        subprocess.run(cmd)


def _find_instance(instance_id):
    """Return the instance dict from `vastai show instances`, or None."""
    raw = vastai('show', 'instances', '--raw', allow_fail=True)
    if raw is None:
        return None
    try:
        instances = json.loads(raw)
    except json.JSONDecodeError:
        return None
    for inst in instances:
        if str(inst.get('id')) == str(instance_id):
            return inst
    return None


def _backfill_ssh(state):
    """Fill missing ssh_host/ssh_port from vast.ai. Returns True on success."""
    inst = _find_instance(state['instance_id'])
    if inst and inst.get('ssh_host') and inst.get('ssh_port'):
        state['ssh_host'] = inst['ssh_host']
        state['ssh_port'] = int(inst['ssh_port'])
        save_state(state)
        return True
    return False


def require_ssh(state):
    """Exit with guidance unless state has usable SSH coordinates."""
    if not state.get('instance_id'):
        print("No instance. Run 'launch' first.", file=sys.stderr)
        sys.exit(1)
    if state.get('ssh_host') and state.get('ssh_port'):
        return state
    print("SSH details missing from state — refreshing from vast.ai...",
          file=sys.stderr)
    if _backfill_ssh(state):
        return state
    print(f"Could not get SSH details for instance {state['instance_id']} — "
          f"it may still be starting or was destroyed.", file=sys.stderr)
    print("Check 'vastai show instances', then 'litmod cloud sync'.",
          file=sys.stderr)
    sys.exit(1)


def _session_name(running):
    """tmux session name for a batch label (sanitized)."""
    return 'batch_' + re.sub(r'[^\w-]', '_', running)


def _coerce_price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _est_cost_line(state):
    price = state.get('price_per_hour')
    if isinstance(price, (int, float)) and state.get('launched_at'):
        from datetime import datetime
        launched = datetime.fromisoformat(state['launched_at'])
        hours = (datetime.now() - launched).total_seconds() / 3600
        return f"Running: {hours:.1f}h, est. cost: ${hours * price:.2f}"
    return None


def ssh_cmd(state):
    """Build base SSH command from state."""
    host = state['ssh_host']
    port = state['ssh_port']
    return [
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        '-p', str(port), f'root@{host}',
    ]


def ssh_run(state, command, check=True, capture=False):
    """Run a command on the remote instance."""
    cmd = ssh_cmd(state) + [command]
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            print(f"SSH error: {r.stderr.strip()}", file=sys.stderr)
        return r
    else:
        r = subprocess.run(cmd)
        if check and r.returncode != 0:
            sys.exit(1)
        return r


def rsync_to(state, local_path, remote_path):
    """rsync local → remote."""
    host = state['ssh_host']
    port = state['ssh_port']
    cmd = [
        'rsync', '-avz', '--progress',
        '-e', f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}',
        str(local_path) + '/',
        f'root@{host}:{q(remote_path)}/',
    ]
    subprocess.run(cmd, check=True)


def rsync_from(state, remote_path, local_path):
    """rsync remote → local."""
    host = state['ssh_host']
    port = state['ssh_port']
    os.makedirs(local_path, exist_ok=True)
    cmd = [
        'rsync', '-avz', '--progress',
        '-e', f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}',
        f'root@{host}:{q(remote_path)}/',
        str(local_path) + '/',
    ]
    subprocess.run(cmd, check=True)


# ── Commands ─────────────────────────────────────────────────────────────


def cmd_launch(args):
    """Find cheapest A100 80GB and create an instance."""
    state = load_state()
    if state.get('instance_id'):
        print(f"Instance already exists: {state['instance_id']}")
        if state.get('ssh_host'):
            print(f"SSH: ssh -p {state['ssh_port']} root@{state['ssh_host']}")
        print("Run 'stop' first to destroy it.")
        return

    print("Searching for A100 80GB offers...", file=sys.stderr)
    raw = vastai(
        'search', 'offers',
        f'gpu_name=A100_SXM4 num_gpus=1 gpu_ram>={MIN_GPU_RAM} reliability>0.95 disk_space>={DISK_GB}',
        '-o', 'dph+',
        '--raw',
    )
    offers = json.loads(raw)
    if not offers:
        raw = vastai(
            'search', 'offers',
            f'gpu_name=A100 num_gpus=1 gpu_ram>={MIN_GPU_RAM} reliability>0.95 disk_space>={DISK_GB}',
            '-o', 'dph+',
            '--raw',
        )
        offers = json.loads(raw)

    if not offers:
        print("No suitable offers found.", file=sys.stderr)
        sys.exit(1)

    offer = offers[0]
    offer_id = offer['id']
    price = _coerce_price(offer.get('dph_total', offer.get('dph')))
    gpu = offer.get('gpu_name', '?')
    ram = offer.get('gpu_ram', '?')
    loc = offer.get('geolocation', '?')

    price_str = f"${price}/hr" if price is not None else "price unknown"
    print(f"Best offer: #{offer_id} — {gpu} {ram}GB, {price_str}, {loc}")

    if not args.yes:
        confirm = input("Launch this instance? [y/N] ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return

    print("Creating instance...", file=sys.stderr)
    result = vastai(
        'create', 'instance', str(offer_id),
        '--image', DOCKER_IMAGE,
        '--disk', str(DISK_GB),
        '--ssh',
        '--direct',
        '--env', f'VLLM_MODEL={VLLM_MODEL}',
    )
    print(result)

    # The CLI echoes either JSON or a Python dict literal; try both, then a
    # targeted regex. Never eval() — output derives from the API response.
    instance_id = None
    for line in result.split('\n'):
        line = line.strip()
        if not line:
            continue
        for substring in [line, line.split('. ', 1)[-1] if '. ' in line else line]:
            for parse in (json.loads, ast.literal_eval):
                try:
                    parsed = parse(substring)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, dict) and parsed.get('new_contract'):
                    instance_id = str(parsed['new_contract'])
                    break
            if instance_id:
                break
        if instance_id:
            break
    if not instance_id:
        match = re.search(r"['\"]?new_contract['\"]?\s*:\s*(\d+)", result)
        if match:
            instance_id = match.group(1)
    if not instance_id:
        print("Could not parse instance ID from output.", file=sys.stderr)
        print("An instance may have been created — check 'vastai show instances' "
              "and run 'litmod cloud sync' (or destroy it on the dashboard).",
              file=sys.stderr)
        sys.exit(1)

    # Persist the ID immediately: from here on, money is flowing, and any
    # crash below must not orphan the instance.
    state = {
        'instance_id': instance_id,
        'offer_id': str(offer_id),
        'gpu': f"{gpu} {ram}GB",
        'price_per_hour': price,
        'launched_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'uploaded': [],
    }
    save_state(state)

    print(f"Instance {instance_id} created. Waiting for SSH...", file=sys.stderr)

    ssh_host, ssh_port = None, None
    status = '?'
    for attempt in range(60):
        inst = _find_instance(instance_id)
        if inst:
            status = inst.get('actual_status', inst.get('status', '')) or '?'
            ssh_host = inst.get('ssh_host')
            ssh_port = inst.get('ssh_port')
            if status == 'running' and ssh_host and ssh_port:
                break
        if attempt % 6 == 0:
            print(f"  Waiting... ({attempt * 5}s, status={status})",
                  file=sys.stderr)
        time.sleep(5)
    else:
        print("Timed out waiting for instance to start.", file=sys.stderr)
        print(f"Instance ID {instance_id} is saved — it is still billing.",
              file=sys.stderr)
        print("Once it's up, 'litmod cloud status' or 'sync' will pick up SSH "
              "details; 'litmod cloud stop' destroys it.", file=sys.stderr)
        sys.exit(1)

    state['ssh_host'] = ssh_host
    state['ssh_port'] = int(ssh_port)
    save_state(state)

    print(f"\nInstance {instance_id} running!")
    print(f"SSH: ssh -p {ssh_port} root@{ssh_host}")
    if price is not None:
        print(f"Cost: ${price}/hr")
    print("\nNext: litmod cloud setup")


def cmd_setup(args):
    """Install vLLM and largeliterarymodels on the instance."""
    state = require_ssh(load_state())

    print("Installing dependencies...", file=sys.stderr)

    setup_script = f"""
set -ex
mkdir -p {REMOTE_PASSAGES} {REMOTE_RESULTS}

# Ensure python points to python3
which python || ln -sf $(which python3) /usr/local/bin/python

# Install largeliterarymodels
if [ ! -d {REMOTE_REPO} ]; then
    git clone https://github.com/quadrismegistus/largeliterarymodels.git {REMOTE_REPO}
else
    cd {REMOTE_REPO} && git pull
fi
pip install -e {REMOTE_REPO}

# Check vLLM (comes with the docker image)
python -c "import vllm; print(f'vLLM {{vllm.__version__}}')"

echo "SETUP COMPLETE"
"""
    ssh_run(state, setup_script)

    print("\nSetup complete.")
    print("Next: litmod cloud upload <passages_dir>")


def cmd_upload(args):
    """Upload a passages directory to the instance."""
    state = require_ssh(load_state())

    passages_dir = args.passages_dir
    local_path = PROJECT_ROOT / 'data' / passages_dir
    if not local_path.exists():
        local_path = Path(passages_dir)
    if not local_path.exists():
        print(f"Not found: {local_path}", file=sys.stderr)
        sys.exit(1)

    n_files = len(list(local_path.rglob('*.jsonl')))
    remote_name = local_path.name
    remote_path = f'{REMOTE_PASSAGES}/{remote_name}'

    print(f"Uploading {n_files} files from {local_path} → {remote_path}",
          file=sys.stderr)
    ssh_run(state, f'mkdir -p {q(remote_path)}')
    rsync_to(state, local_path, remote_path)

    if remote_name not in state.get('uploaded', []):
        state.setdefault('uploaded', []).append(remote_name)
        save_state(state)

    print(f"\nUploaded {remote_name}.")
    print(f"Next: litmod cloud run {remote_name}")


def cmd_run(args):
    """Start vLLM server and batch processing in a tmux session."""
    state = require_ssh(load_state())

    passages_name = args.passages_dir
    task = args.task
    remote_passages = f'{REMOTE_PASSAGES}/{passages_name}'
    if task != 'social_network':
        running_label = f'{passages_name}_{task}'
    else:
        running_label = passages_name
    remote_output = f'{REMOTE_RESULTS}/{running_label}'
    session_name = _session_name(running_label)
    workers = args.workers or BATCH_WORKERS
    log_path = f'/workspace/batch_{passages_name}.log'

    run_script = f"""
set -ex
mkdir -p {q(remote_output)}

# Start vLLM if not already running
if ! curl -s http://127.0.0.1:{VLLM_PORT}/health >/dev/null 2>&1; then
    echo "Starting vLLM..."
    python -m vllm.entrypoints.openai.api_server \\
        --model {VLLM_MODEL} \\
        --served-model-name {VLLM_SERVED_NAME} \\
        --port {VLLM_PORT} \\
        --host 127.0.0.1 \\
        --enable-prefix-caching \\
        --gpu-memory-utilization 0.95 \\
        --max-model-len 32768 \\
        --no-enable-log-requests \\
        > /workspace/vllm.log 2>&1 &

    echo "Waiting for vLLM..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:{VLLM_PORT}/health >/dev/null 2>&1; then
            echo "vLLM ready after $((i*5))s"
            break
        fi
        sleep 5
    done
fi

# Verify vLLM is up
curl -sf http://127.0.0.1:{VLLM_PORT}/health || {{ echo "vLLM not healthy"; exit 1; }}

# Count what we have (nested dirs included)
n_texts=$(find {q(remote_passages)} -name "*.jsonl" 2>/dev/null | wc -l)
n_done=$(find {q(remote_output)} -name "*.json" 2>/dev/null | wc -l)
echo "Texts: $n_texts, Already done: $n_done"
"""
    print("Starting vLLM server...", file=sys.stderr)
    ssh_run(state, run_script)

    if task == 'social_network':
        # batch_social_network.py resolves keys via its MODEL_TABLE and passes
        # full model strings straight through.
        model = args.model or 'vllm-qwen36'
        batch_cmd = (
            f'cd {REMOTE_REPO} && '
            f'python scripts/batch_social_network.py '
            f'--text-dir {q(remote_passages)} '
            f'--output-dir {q(remote_output)} '
            f'--model {q(model)} '
            f'--workers {workers} '
            f'2>&1 | tee {q(log_path)}'
        )
    elif task in PASSAGE_TASKS:
        model = args.model or f'vllm/{VLLM_SERVED_NAME}'
        batch_cmd = (
            f'cd {REMOTE_REPO} && '
            f'python scripts/batch_passage_task.py '
            f'--task {q(task)} '
            f'--input {q(remote_passages)} '
            f'--output {q(remote_output)} '
            f'--model {q(model)} '
            f'--workers {workers} '
            f'2>&1 | tee {q(log_path)}'
        )
    elif task in SUMMARY_TASK_MAP:
        model = args.model or f'vllm/{VLLM_SERVED_NAME}'
        batch_cmd = (
            f'cd {REMOTE_REPO} && '
            f'python scripts/batch_summary_task.py '
            f'--task {q(task)} '
            f'--input {q(remote_passages)} '
            f'--output {q(remote_output)} '
            f'--model {q(model)} '
            f'--workers {workers} '
            f'2>&1 | tee {q(log_path)}'
        )
    else:
        options = sorted({'social_network'} | PASSAGE_TASKS | set(SUMMARY_TASK_MAP))
        print(f"Unknown task: {task}. Options: {', '.join(options)}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Starting batch in tmux session '{session_name}'...", file=sys.stderr)
    ssh_run(state, f"tmux kill-session -t {q('=' + session_name)} 2>/dev/null || true")
    ssh_run(state, f"tmux new-session -d -s {q(session_name)} {q(batch_cmd)}")

    state['running'] = running_label
    state['run_started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    save_state(state)

    print(f"\nBatch started in tmux session '{session_name}'.")
    print(f"Model: {model}")
    print(f"Workers: {workers}")
    print(f"Output: {remote_output}")
    print("\nMonitor: litmod cloud status")
    print(f"SSH in:  ssh -p {state['ssh_port']} root@{state['ssh_host']}")
    print(f"Attach:  tmux attach -t {session_name}")


def cmd_status(args):
    """Check instance status, batch progress, and running cost."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance.", file=sys.stderr)
        return

    print(f"Instance: {state['instance_id']}")
    print(f"GPU: {state.get('gpu', '?')}")
    if not (state.get('ssh_host') and state.get('ssh_port')):
        _backfill_ssh(state)
    if state.get('ssh_host') and state.get('ssh_port'):
        print(f"SSH: ssh -p {state['ssh_port']} root@{state['ssh_host']}")
    print(f"Launched: {state.get('launched_at', '?')}")

    cost = _est_cost_line(state)
    if cost:
        print(cost)

    if not (state.get('ssh_host') and state.get('ssh_port')):
        print("\nSSH details not available yet — instance may still be "
              "starting. Try again shortly, or 'litmod cloud sync'.")
        return

    print()

    for name in state.get('uploaded', []):
        remote_passages = f'{REMOTE_PASSAGES}/{name}'
        r = ssh_run(state, (
            f'n_texts=$(find {q(remote_passages)} -name "*.jsonl" 2>/dev/null | wc -l); '
            f'for d in {q(f"{REMOTE_RESULTS}/{name}")}*/; do '
            f'  [ -d "$d" ] || continue; '
            f'  dn=$(basename "$d"); '
            f'  n=$(find "$d" -name "*.json" 2>/dev/null | wc -l); '
            f'  echo "$n_texts $n $dn"; '
            f'done'
        ), capture=True, check=False)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    n_texts, n_done, dn = int(parts[0]), int(parts[1]), parts[2]
                    pct = (n_done / n_texts * 100) if n_texts else 0
                    done_mark = '  ← complete' if n_texts and n_done >= n_texts else ''
                    print(f"  {dn}: {n_done}/{n_texts} done ({pct:.0f}%){done_mark}")

    running = state.get('running')
    if running:
        session_name = _session_name(running)
        r = ssh_run(state,
                    f"tmux has-session -t {q('=' + session_name)} 2>/dev/null "
                    f"&& echo RUNNING || echo STOPPED",
                    capture=True, check=False)
        status = r.stdout.strip()
        print(f"\n  Batch: {status}")
        if status == 'RUNNING':
            log = f'/workspace/batch_{running}.log'
            r = ssh_run(state, f'tail -5 {q(log)} 2>/dev/null',
                        capture=True, check=False)
            if r.stdout.strip():
                print("\n  Last log lines:")
                for line in r.stdout.strip().split('\n'):
                    print(f"    {line}")
        elif status == 'STOPPED':
            print("  Batch is no longer running. The instance still bills "
                  "until destroyed:")
            print("  'litmod cloud download' to fetch results, then "
                  "'litmod cloud stop'.")


def cmd_download(args):
    """Download results from the instance."""
    state = require_ssh(load_state())

    os.makedirs(LOCAL_RESULTS, exist_ok=True)

    for name in state.get('uploaded', []):
        r = ssh_run(state, f'ls -d {q(f"{REMOTE_RESULTS}/{name}")}*/ 2>/dev/null',
                    capture=True, check=False)
        if r.returncode != 0 or not r.stdout.strip():
            print(f"  {name}: no results yet")
            continue
        for remote_dir in r.stdout.strip().splitlines():
            remote_dir = remote_dir.rstrip('/')
            dir_name = os.path.basename(remote_dir)
            local_dir = LOCAL_RESULTS / dir_name
            r2 = ssh_run(state,
                         f'find {q(remote_dir)} -name "*.json" 2>/dev/null | wc -l',
                         capture=True, check=False)
            n = int(r2.stdout.strip()) if r2.returncode == 0 else 0
            if n == 0:
                print(f"  {dir_name}: no results yet")
                continue
            print(f"Downloading {n} results for {dir_name}...", file=sys.stderr)
            rsync_from(state, remote_dir, local_dir)
            print(f"  {dir_name}: {n} files → {local_dir}")

    print(f"\nResults in {LOCAL_RESULTS}/")
    print(f"Ingest locally: lltk ingest-tasks social_network {LOCAL_RESULTS}/<dir>")


def cmd_stop(args):
    """Destroy the instance (stops all billing)."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance to stop.", file=sys.stderr)
        return

    instance_id = state['instance_id']
    cost = _est_cost_line(state)
    if cost:
        print(f"Instance {instance_id} — {cost}")

    if not args.yes:
        confirm = input("Destroy this instance? (data will be lost) [y/N] ").strip().lower()
        if confirm != 'y':
            print("Aborted. Run 'download' first if you haven't.")
            return

    print(f"Destroying instance {instance_id}...", file=sys.stderr)
    r = subprocess.run(['vastai', 'destroy', 'instance', str(instance_id)],
                       input='y\n', text=True, capture_output=True)
    if r.returncode != 0:
        print(f"Destroy FAILED: {(r.stderr or r.stdout).strip()}", file=sys.stderr)
        print("State kept — the instance is still billing. Retry, or destroy "
              "it at https://cloud.vast.ai/instances/", file=sys.stderr)
        sys.exit(1)

    # Trust nothing until vast.ai stops listing it.
    for _ in range(6):
        raw = vastai('show', 'instances', '--raw', allow_fail=True)
        if raw is not None:
            try:
                still_listed = any(str(i.get('id')) == str(instance_id)
                                   for i in json.loads(raw))
            except json.JSONDecodeError:
                still_listed = True
            if not still_listed:
                STATE_FILE.unlink(missing_ok=True)
                print("Instance destroyed. All billing stopped.")
                return
        time.sleep(5)

    print(f"Destroy submitted, but instance {instance_id} is still listed.",
          file=sys.stderr)
    print("State kept — verify at https://cloud.vast.ai/instances/ before "
          "assuming billing stopped.", file=sys.stderr)
    sys.exit(1)


def cmd_sync(args):
    """Rebuild local state from `vastai show instances` (orphan recovery)."""
    state = load_state()
    raw = vastai('show', 'instances', '--raw')
    try:
        instances = json.loads(raw)
    except json.JSONDecodeError:
        print("Could not parse 'vastai show instances' output.", file=sys.stderr)
        sys.exit(1)

    if not instances:
        if state.get('instance_id'):
            print(f"No instances on vast.ai; clearing stale local state "
                  f"(was {state['instance_id']}).")
            STATE_FILE.unlink(missing_ok=True)
        else:
            print("No instances on vast.ai and no local state.")
        return

    target = None
    wanted = args.id or state.get('instance_id')
    if wanted:
        for inst in instances:
            if str(inst.get('id')) == str(wanted):
                target = inst
                break
        if target is None and args.id:
            print(f"Instance {args.id} not found. Running: "
                  f"{[i.get('id') for i in instances]}", file=sys.stderr)
            sys.exit(1)
    if target is None:
        if len(instances) == 1:
            target = instances[0]
        else:
            print("Multiple instances found — pick one with 'sync --id <ID>':")
            for i in instances:
                print(f"  {i.get('id')}: {i.get('gpu_name', '?')} "
                      f"{i.get('actual_status', '?')} "
                      f"${i.get('dph_total', '?')}/hr")
            sys.exit(1)

    state.update({
        'instance_id': str(target.get('id')),
        'ssh_host': target.get('ssh_host'),
        'ssh_port': int(target['ssh_port']) if target.get('ssh_port') else None,
        'gpu': f"{target.get('gpu_name', '?')} {target.get('gpu_ram', '?')}GB",
        'price_per_hour': _coerce_price(target.get('dph_total')),
    })
    state.setdefault('launched_at', time.strftime('%Y-%m-%dT%H:%M:%S'))
    state.setdefault('uploaded', [])
    save_state(state)
    print(f"State synced: instance {state['instance_id']} ({state['gpu']}), "
          f"status={target.get('actual_status', '?')}")
    if state.get('ssh_host'):
        print(f"SSH: ssh -p {state['ssh_port']} root@{state['ssh_host']}")


def cmd_ssh(args):
    """Open an interactive SSH session."""
    state = require_ssh(load_state())
    cmd = ssh_cmd(state)
    if args.ssh_command:
        cmd += [shlex.join(args.ssh_command)]
    os.execvp(cmd[0], cmd)


def cmd_attach(args):
    """Attach to the running tmux batch session."""
    state = require_ssh(load_state())
    running = state.get('running')
    if running:
        attach = (f"tmux attach -t {q('=' + _session_name(running))} "
                  f"2>/dev/null || tmux attach")
    else:
        attach = 'tmux attach'
    cmd = ssh_cmd(state) + ['-t', attach]
    os.execvp(cmd[0], cmd)


def cmd_cancel(args):
    """Cancel the running batch (kill its tmux session)."""
    state = require_ssh(load_state())
    running = state.get('running')
    if not running:
        print("No batch recorded as running. tmux sessions on instance:",
              file=sys.stderr)
        ssh_run(state, 'tmux ls 2>/dev/null || echo "(none)"', check=False)
        return
    session_name = _session_name(running)
    ssh_run(state,
            f"tmux kill-session -t {q('=' + session_name)} 2>/dev/null "
            f"&& echo cancelled || echo 'session not found (already finished?)'",
            capture=False, check=False)
    state['running'] = None
    save_state(state)
    print(f"Batch '{running}' cancelled. vLLM/instance still running — "
          f"use 'stop' to destroy instance.", file=sys.stderr)


def cmd_log(args):
    """Tail the batch log."""
    state = require_ssh(load_state())
    n = args.lines or 30
    cmd = ssh_cmd(state) + [f'tail -{n} /workspace/batch*.log 2>/dev/null || echo "No log found"']
    os.execvp(cmd[0], cmd)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='litmod cloud',
        description='Vast.ai GPU instance manager for social network extraction')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts')
    sub = parser.add_subparsers(dest='command', required=True)

    p_launch = sub.add_parser('launch', help='Find and rent cheapest A100 80GB')
    # default=SUPPRESS so a subcommand-level flag can only turn --yes on,
    # never clobber a `litmod cloud -y <cmd>` back to False.
    p_launch.add_argument('--yes', '-y', action='store_true',
                          default=argparse.SUPPRESS,
                          help='Skip confirmation prompt')
    sub.add_parser('setup', help='Install vLLM + largeliterarymodels on instance')

    p_upload = sub.add_parser('upload', help='Upload a passages directory')
    p_upload.add_argument('passages_dir', help='Directory name under data/ or full path')

    task_options = sorted({'social_network'} | PASSAGE_TASKS | set(SUMMARY_TASK_MAP))
    p_run = sub.add_parser('run', help='Start batch processing in tmux')
    p_run.add_argument('passages_dir', help='Name of uploaded passages dir')
    p_run.add_argument('--task', default='social_network',
                       choices=task_options,
                       help='Task to run (default: social_network)')
    p_run.add_argument('--model', default=None,
                       help=f'Model to use (default: vllm/{VLLM_SERVED_NAME}; '
                            f'social_network default: vllm-qwen36)')
    p_run.add_argument('--workers', type=int, default=None,
                       help=f'Number of parallel workers (default: {BATCH_WORKERS})')

    sub.add_parser('status', help='Check progress and running cost')
    sub.add_parser('download', help='Download results to local machine')

    p_stop = sub.add_parser('stop', help='Destroy instance (stops all billing)')
    p_stop.add_argument('--yes', '-y', action='store_true',
                        default=argparse.SUPPRESS,
                        help='Skip confirmation prompt')

    p_sync = sub.add_parser('sync',
                            help="Rebuild state from 'vastai show instances' "
                                 "(recovers orphaned instances)")
    p_sync.add_argument('--id', default=None,
                        help='Instance ID to adopt (needed if several are running)')

    sub.add_parser('attach', help='Attach to running batch tmux session')
    sub.add_parser('cancel', help='Cancel running batch (keeps instance)')

    p_log = sub.add_parser('log', help='Tail the batch log')
    p_log.add_argument('--lines', '-n', type=int, default=30)

    p_ssh = sub.add_parser('ssh', help='Open interactive SSH session')
    p_ssh.add_argument('ssh_command', nargs=argparse.REMAINDER,
                       help='Optional command to run (flags pass through)')

    args = parser.parse_args(argv)

    commands = {
        'launch': cmd_launch,
        'setup': cmd_setup,
        'upload': cmd_upload,
        'run': cmd_run,
        'status': cmd_status,
        'download': cmd_download,
        'stop': cmd_stop,
        'sync': cmd_sync,
        'attach': cmd_attach,
        'cancel': cmd_cancel,
        'log': cmd_log,
        'ssh': cmd_ssh,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
