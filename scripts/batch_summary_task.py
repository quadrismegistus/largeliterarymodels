"""Batch runner for summary-based tasks (PlotGenreTask, SubgenreTask, etc.)

Reads social network JSONs (from lltk export-task-results social_network),
runs a specified task on each, writes output JSONs for lltk ingest-tasks.

Usage:
    # PlotGenreTask with Sonnet
    python scripts/batch_summary_task.py -t plot_genre -i data/social_network_export/ -m claude-sonnet-4-6

    # SubgenreTask with Sonnet, 4 workers
    python scripts/batch_summary_task.py -t subgenre -i data/social_network_export/ -m claude-sonnet-4-6 -w 4

    # Resume (skips existing outputs)
    python scripts/batch_summary_task.py -t subgenre -i data/social_network_export/ -m claude-sonnet-4-6

    # Dry run with cost estimate
    python scripts/batch_summary_task.py -t subgenre -i data/social_network_export/ -m claude-sonnet-4-6 --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

TASK_MAP = {
    'plot_genre': ('largeliterarymodels.tasks', 'PlotGenreTask'),
    'subgenre': ('largeliterarymodels.tasks', 'SubgenreTask'),
}


def load_task_class(task_name):
    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_MAP.keys())}")
    module_path, class_name = TASK_MAP[task_name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def model_slug(model: str) -> str:
    return model.replace('/', '-').replace('.', '')


def run_one(task, task_class, input_path, output_dir, task_name, model, force=False):
    with open(input_path) as f:
        sn = json.load(f)

    canonical_id = sn.get('metadata', {}).get('_canonical_id', '')
    if not canonical_id:
        canonical_id = Path(input_path).stem

    slug = model_slug(model)
    out_name = f"{canonical_id.replace('/', '_')}_{slug}.json"
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path) and not force:
        return 'skip', canonical_id, 0

    prompt = task_class.format_input(sn)
    if not prompt.strip() or len(prompt) < 50:
        return 'empty', canonical_id, 0

    t0 = time.time()
    try:
        result = task.run(prompt, model=model, force=force)
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR {canonical_id}: {e} ({elapsed:.1f}s)", flush=True)
        return 'error', canonical_id, elapsed

    output = {
        'metadata': {
            'source': canonical_id,
            '_canonical_id': canonical_id,
            'model': model,
            'task': task_name,
        },
        **result.model_dump(),
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    return 'ok', canonical_id, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', required=True,
                        choices=list(TASK_MAP.keys()),
                        help='Task to run')
    parser.add_argument('--input', '-i', required=True,
                        help='Dir of social network JSONs')
    parser.add_argument('--output', '-o', default=None,
                        help='Output dir (default: data/<task>_output/)')
    parser.add_argument('--model', '-m', default='claude-sonnet-4-6')
    parser.add_argument('--workers', '-w', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    task_class = load_task_class(args.task)

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), '..', 'data',
                                   f'{args.task}_output')

    os.makedirs(args.output, exist_ok=True)

    files = sorted(Path(args.input).glob('*.json'))
    if args.limit:
        files = files[:args.limit]

    print(f"Task: {args.task} ({task_class.__name__})", file=sys.stderr)
    print(f"Files: {len(files)}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    if args.dry_run:
        slug = model_slug(args.model)
        existing = set(os.listdir(args.output)) if os.path.exists(args.output) else set()
        n_skip = 0
        for f in files:
            sn = json.load(open(f))
            cid = sn.get('metadata', {}).get('_canonical_id', f.stem)
            out_name = f"{cid.replace('/', '_')}_{slug}.json"
            if out_name in existing:
                n_skip += 1
        print(f"Would process {len(files) - n_skip}, skip {n_skip}", file=sys.stderr)
        print(file=sys.stderr)
        from largeliterarymodels.costs import dry_run as cost_dry_run
        cost_dry_run(task_class, args.input, output_tokens=200)
        return

    task = task_class()
    counts = {'ok': 0, 'skip': 0, 'error': 0, 'empty': 0}
    total_time = 0
    done = 0

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fpath in files:
                fut = pool.submit(run_one, task, task_class, fpath, args.output,
                                  args.task, args.model, args.force)
                futures[fut] = fpath
            for fut in as_completed(futures):
                done += 1
                status, cid, elapsed = fut.result()
                counts[status] += 1
                if status == 'ok':
                    total_time += elapsed
                    avg = total_time / counts['ok']
                    remaining = (len(files) - done) * avg / args.workers
                    print(f"  [{done}/{len(files)}] {cid} ({elapsed:.1f}s, "
                          f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
                elif status == 'skip':
                    if done % 100 == 0:
                        print(f"  [{done}/{len(files)}] skipping...", flush=True)
    else:
        for i, fpath in enumerate(files):
            status, cid, elapsed = run_one(task, task_class, fpath, args.output,
                                           args.task, args.model, args.force)
            counts[status] += 1
            if status == 'ok':
                total_time += elapsed
                avg = total_time / counts['ok']
                remaining = (len(files) - i - 1) * avg
                print(f"  [{i+1}/{len(files)}] {cid} ({elapsed:.1f}s, "
                      f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
            elif status == 'skip':
                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(files)}] skipping...", flush=True)

    print(f"\nDone: {counts['ok']} ok, {counts['skip']} skipped, "
          f"{counts['error']} errors, {counts['empty']} empty",
          file=sys.stderr)
    print(f"Total API time: {total_time:.0f}s ({total_time/60:.1f}min)",
          file=sys.stderr)


if __name__ == '__main__':
    main()
