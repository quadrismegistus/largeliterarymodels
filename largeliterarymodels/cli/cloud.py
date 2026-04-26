#!/usr/bin/env python3
"""Vast.ai GPU instance manager for running social network extraction at scale.

Manages the full lifecycle: launch → setup → upload → run → status → download → stop.

State is stored in .vastai.json in the project root. Each command reads/writes it
so you can run them independently and resume after disconnects.

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
"""

import argparse
import json
import os
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


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2) + '\n')


def vastai(*args, capture=True):
    cmd = ['vastai'] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"vastai error: {r.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        return r.stdout.strip()
    else:
        subprocess.run(cmd)


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
        f'root@{host}:{remote_path}/',
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
        f'root@{host}:{remote_path}/',
        str(local_path) + '/',
    ]
    subprocess.run(cmd, check=True)


# ── Commands ─────────────────────────────────────────────────────────────


def cmd_launch(args):
    """Find cheapest A100 80GB and create an instance."""
    state = load_state()
    if state.get('instance_id'):
        print(f"Instance already exists: {state['instance_id']}")
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
    price = offer.get('dph_total', offer.get('dph', '?'))
    gpu = offer.get('gpu_name', '?')
    ram = offer.get('gpu_ram', '?')
    loc = offer.get('geolocation', '?')

    print(f"Best offer: #{offer_id} — {gpu} {ram}GB, ${price}/hr, {loc}")

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

    instance_id = None
    try:
        parsed = json.loads(result.split('\n')[-1])
        instance_id = str(parsed.get('new_contract', ''))
    except (json.JSONDecodeError, IndexError):
        pass
    if not instance_id:
        for word in result.split():
            if word.isdigit():
                instance_id = word
                break
    if not instance_id:
        print("Could not parse instance ID from output.", file=sys.stderr)
        print("Check 'vastai show instances' manually.", file=sys.stderr)
        sys.exit(1)

    print(f"Instance {instance_id} created. Waiting for SSH...", file=sys.stderr)

    ssh_host, ssh_port = None, None
    for attempt in range(60):
        raw = vastai('show', 'instances', '--raw')
        instances = json.loads(raw)
        for inst in instances:
            if str(inst.get('id')) == instance_id:
                status = inst.get('actual_status', inst.get('status', ''))
                ssh_host = inst.get('ssh_host')
                ssh_port = inst.get('ssh_port')
                if status == 'running' and ssh_host and ssh_port:
                    break
        else:
            if attempt % 6 == 0:
                print(f"  Waiting... ({attempt * 5}s, status={status})",
                      file=sys.stderr)
            time.sleep(5)
            continue
        break
    else:
        print("Timed out waiting for instance to start.", file=sys.stderr)
        print(f"Instance ID: {instance_id} — check 'vastai show instances'",
              file=sys.stderr)
        state['instance_id'] = instance_id
        save_state(state)
        sys.exit(1)

    state = {
        'instance_id': instance_id,
        'offer_id': str(offer_id),
        'ssh_host': ssh_host,
        'ssh_port': int(ssh_port),
        'gpu': f"{gpu} {ram}GB",
        'price_per_hour': float(price) if isinstance(price, (int, float)) else price,
        'launched_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'uploaded': [],
    }
    save_state(state)

    print(f"\nInstance {instance_id} running!")
    print(f"SSH: ssh -p {ssh_port} root@{ssh_host}")
    print(f"Cost: ${price}/hr")
    print(f"\nNext: litmod cloud setup")


def cmd_setup(args):
    """Install vLLM and largeliterarymodels on the instance."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance. Run 'launch' first.", file=sys.stderr)
        sys.exit(1)

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

    state['setup_done'] = True
    save_state(state)
    print("\nSetup complete.")
    print(f"Next: litmod cloud upload <passages_dir>")


def cmd_upload(args):
    """Upload a passages directory to the instance."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance. Run 'launch' first.", file=sys.stderr)
        sys.exit(1)

    passages_dir = args.passages_dir
    local_path = PROJECT_ROOT / 'data' / passages_dir
    if not local_path.exists():
        local_path = Path(passages_dir)
    if not local_path.exists():
        print(f"Not found: {local_path}", file=sys.stderr)
        sys.exit(1)

    n_files = len(list(local_path.glob('*.jsonl')))
    remote_name = local_path.name
    remote_path = f'{REMOTE_PASSAGES}/{remote_name}'

    print(f"Uploading {n_files} files from {local_path} → {remote_path}",
          file=sys.stderr)
    ssh_run(state, f'mkdir -p {remote_path}')
    rsync_to(state, local_path, remote_path)

    if remote_name not in state.get('uploaded', []):
        state.setdefault('uploaded', []).append(remote_name)
        save_state(state)

    print(f"\nUploaded {remote_name}.")
    print(f"Next: litmod cloud run {remote_name}")


def cmd_run(args):
    """Start vLLM server and batch processing in a tmux session."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance. Run 'launch' first.", file=sys.stderr)
        sys.exit(1)

    passages_name = args.passages_dir
    remote_passages = f'{REMOTE_PASSAGES}/{passages_name}'
    remote_output = f'{REMOTE_RESULTS}/{passages_name}'
    session_name = f'batch_{passages_name}'
    workers = args.workers or BATCH_WORKERS

    run_script = f"""
set -ex
mkdir -p {remote_output}

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

# Count what we have
n_texts=$(ls {remote_passages}/*.jsonl 2>/dev/null | wc -l)
n_done=$(ls {remote_output}/*.json 2>/dev/null | wc -l)
echo "Texts: $n_texts, Already done: $n_done"
"""
    print("Starting vLLM server...", file=sys.stderr)
    ssh_run(state, run_script)

    # TODO: make task configurable via --task flag when we have more
    # cloud-scale sequential tasks beyond SocialNetworkTask
    batch_cmd = (
        f'cd {REMOTE_REPO} && '
        f'python scripts/batch_social_network.py '
        f'--text-dir {remote_passages} '
        f'--output-dir {remote_output} '
        f'--model vllm-qwen36 '
        f'--workers {workers} '
        f'2>&1 | tee /workspace/batch_{passages_name}.log'
    )

    print(f"Starting batch in tmux session '{session_name}'...", file=sys.stderr)
    ssh_run(state, f"tmux kill-session -t {session_name} 2>/dev/null || true")
    ssh_run(state, f"tmux new-session -d -s {session_name} '{batch_cmd}'")

    state['running'] = passages_name
    state['run_started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    save_state(state)

    print(f"\nBatch started in tmux session '{session_name}'.")
    print(f"Workers: {workers}")
    print(f"Output: {remote_output}")
    print(f"\nMonitor: litmod cloud status")
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
    print(f"SSH: ssh -p {state['ssh_port']} root@{state['ssh_host']}")
    print(f"Launched: {state.get('launched_at', '?')}")

    price = state.get('price_per_hour', 0)
    if price and state.get('launched_at'):
        from datetime import datetime
        launched = datetime.fromisoformat(state['launched_at'])
        hours = (datetime.now() - launched).total_seconds() / 3600
        print(f"Running: {hours:.1f}h, est. cost: ${hours * price:.2f}")

    print()

    for name in state.get('uploaded', []):
        remote_passages = f'{REMOTE_PASSAGES}/{name}'
        remote_output = f'{REMOTE_RESULTS}/{name}'
        r = ssh_run(state, (
            f'n_texts=$(ls {remote_passages}/*.jsonl 2>/dev/null | wc -l); '
            f'n_done=$(ls {remote_output}/*.json 2>/dev/null | wc -l); '
            f'echo "$n_texts $n_done"'
        ), capture=True, check=False)
        if r.returncode == 0:
            parts = r.stdout.strip().split()
            if len(parts) == 2:
                n_texts, n_done = int(parts[0]), int(parts[1])
                pct = (n_done / n_texts * 100) if n_texts else 0
                print(f"  {name}: {n_done}/{n_texts} done ({pct:.0f}%)")

    running = state.get('running')
    if running:
        session_name = f'batch_{running}'
        r = ssh_run(state, f"tmux has-session -t {session_name} 2>/dev/null && echo RUNNING || echo STOPPED",
                    capture=True, check=False)
        status = r.stdout.strip()
        print(f"\n  Batch: {status}")
        if status == 'RUNNING':
            log = f'/workspace/batch_{running}.log'
            r = ssh_run(state, f'tail -5 {log} 2>/dev/null',
                        capture=True, check=False)
            if r.stdout.strip():
                print(f"\n  Last log lines:")
                for line in r.stdout.strip().split('\n'):
                    print(f"    {line}")


def cmd_download(args):
    """Download results from the instance."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(LOCAL_RESULTS, exist_ok=True)

    for name in state.get('uploaded', []):
        remote_output = f'{REMOTE_RESULTS}/{name}'
        local_dir = LOCAL_RESULTS / name
        r = ssh_run(state, f'ls {remote_output}/*.json 2>/dev/null | wc -l',
                    capture=True, check=False)
        n = int(r.stdout.strip()) if r.returncode == 0 else 0
        if n == 0:
            print(f"  {name}: no results yet")
            continue
        print(f"Downloading {n} results for {name}...", file=sys.stderr)
        rsync_from(state, remote_output, local_dir)
        print(f"  {name}: {n} files → {local_dir}")

    print(f"\nResults in {LOCAL_RESULTS}/")
    print(f"Ingest locally: lltk ingest-tasks social_network {LOCAL_RESULTS}/<dir>")


def cmd_stop(args):
    """Destroy the instance (stops all billing)."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance to stop.", file=sys.stderr)
        return

    instance_id = state['instance_id']
    price = state.get('price_per_hour', 0)
    if price and state.get('launched_at'):
        from datetime import datetime
        launched = datetime.fromisoformat(state['launched_at'])
        hours = (datetime.now() - launched).total_seconds() / 3600
        print(f"Instance {instance_id} running {hours:.1f}h, est. cost: ${hours * price:.2f}")

    if not args.yes:
        confirm = input("Destroy this instance? (data will be lost) [y/N] ").strip().lower()
        if confirm != 'y':
            print("Aborted. Run 'download' first if you haven't.")
            return

    print(f"Destroying instance {instance_id}...", file=sys.stderr)
    vastai('destroy', 'instance', instance_id)
    STATE_FILE.unlink(missing_ok=True)
    print("Instance destroyed. All billing stopped.")


def cmd_ssh(args):
    """Open an interactive SSH session."""
    state = load_state()
    if not state.get('instance_id'):
        print("No instance.", file=sys.stderr)
        sys.exit(1)
    cmd = ssh_cmd(state)
    if args.ssh_command:
        cmd += [' '.join(args.ssh_command)]
    os.execvp(cmd[0], cmd)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='litmod cloud',
        description='Vast.ai GPU instance manager for social network extraction')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('launch', help='Find and rent cheapest A100 80GB')
    sub.add_parser('setup', help='Install vLLM + largeliterarymodels on instance')

    p_upload = sub.add_parser('upload', help='Upload a passages directory')
    p_upload.add_argument('passages_dir', help='Directory name under data/ or full path')

    p_run = sub.add_parser('run', help='Start batch processing in tmux')
    p_run.add_argument('passages_dir', help='Name of uploaded passages dir')
    p_run.add_argument('--workers', type=int, default=None,
                       help=f'Number of parallel workers (default: {BATCH_WORKERS})')

    sub.add_parser('status', help='Check progress and running cost')
    sub.add_parser('download', help='Download results to local machine')
    sub.add_parser('stop', help='Destroy instance (stops all billing)')

    p_ssh = sub.add_parser('ssh', help='Open interactive SSH session')
    p_ssh.add_argument('ssh_command', nargs='*', help='Optional command to run')

    args = parser.parse_args(argv)

    commands = {
        'launch': cmd_launch,
        'setup': cmd_setup,
        'upload': cmd_upload,
        'run': cmd_run,
        'status': cmd_status,
        'download': cmd_download,
        'stop': cmd_stop,
        'ssh': cmd_ssh,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
