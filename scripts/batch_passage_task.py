"""Batch runner for passage-level tasks (PassageSettingTask, etc.)

Reads passage JSONL files (from lltk export-passages), runs a specified task
on each passage, writes output JSONs for lltk ingest.

Usage:
    # PassageSettingTask with vLLM on CSD3
    python scripts/batch_passage_task.py -t passage_setting -i passages_dir/ -o results/ -m vllm/qwen3.6-27b -w 4

    # Resume (skips existing)
    python scripts/batch_passage_task.py -t passage_setting -i passages_dir/ -o results/ -m vllm/qwen3.6-27b -w 4

    # Sonnet pilot
    python scripts/batch_passage_task.py -t passage_setting -i passages_dir/ -o results/ -m claude-sonnet-4-6 --limit 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

TASK_MAP = {
    'passage_setting': ('largeliterarymodels.tasks', 'PassageSettingTask'),
}


def load_task_class(task_name):
    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_MAP.keys())}")
    import importlib
    module_path, class_name = TASK_MAP[task_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def model_slug(model: str) -> str:
    return model.replace('/', '-').replace('.', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', required=True, choices=list(TASK_MAP.keys()))
    parser.add_argument('--input', '-i', required=True, help='Dir of passage JSONL files')
    parser.add_argument('--output', '-o', required=True, help='Output dir')
    parser.add_argument('--model', '-m', default='lmstudio/qwen/qwen3.6-27b')
    parser.add_argument('--workers', '-w', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Max passages to process (0=all)')
    args = parser.parse_args()

    task_class = load_task_class(args.task)
    task = task_class()
    slug = model_slug(args.model)
    os.makedirs(args.output, exist_ok=True)

    from largeliterarymodels.tasks.classify_passage import format_passage

    # Collect all passages from all JSONL files
    passages = []
    for f in sorted(Path(args.input).glob('*.jsonl')):
        with open(f) as fh:
            lines = [json.loads(l) for l in fh]
        if not lines:
            continue
        header = lines[0]
        text_id = header.get('_id', f.stem)
        title = header.get('title', '?')
        author = header.get('author', '?')
        year = header.get('year', None)

        for line in lines[1:]:
            passages.append({
                '_id': text_id,
                'seq': line.get('seq', 0),
                'position': line.get('position', None),
                'text': line.get('text', ''),
                'n_words': line.get('n_words', 0),
                'title': title,
                'author': author,
                'year': year,
            })

    if args.limit:
        passages = passages[:args.limit]

    print(f"Task: {args.task}", file=sys.stderr)
    print(f"Passages: {len(passages)}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    def run_one(p):
        text_id = p['_id']
        seq = p['seq']
        out_name = f"{text_id.replace('/', '_')}_p{seq}_{slug}.json"
        out_path = os.path.join(args.output, out_name)

        if os.path.exists(out_path) and not args.force:
            return 'skip', text_id, seq, 0

        if not p['text'].strip() or len(p['text']) < 20:
            return 'empty', text_id, seq, 0

        prompt, _ = format_passage(
            p['text'], title=p['title'], author=p['author'], year=p['year'],
            _id=p['_id'], section_id=f"p500:{p['seq']}",
        )

        t0 = time.time()
        try:
            result = task.run(prompt, model=args.model, force=args.force)
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR {text_id} p{seq}: {e} ({elapsed:.1f}s)", flush=True)
            return 'error', text_id, seq, elapsed

        output = {
            'metadata': {
                '_id': text_id,
                'seq': seq,
                'position': p.get('position'),
                'model': args.model,
                'task': args.task,
                'n_words': p['n_words'],
            },
            **result.model_dump(),
        }
        with open(out_path, 'w') as fh:
            json.dump(output, fh, indent=2)
        return 'ok', text_id, seq, elapsed

    counts = {'ok': 0, 'skip': 0, 'error': 0, 'empty': 0}
    total_time = 0
    done = 0

    if args.workers > 1:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for p in passages:
                futures[pool.submit(run_one, p)] = p
            for fut in as_completed(futures):
                done += 1
                status, tid, seq, elapsed = fut.result()
                counts[status] += 1
                if status == 'ok':
                    total_time += elapsed
                    avg = total_time / counts['ok']
                    remaining = (len(passages) - done) * avg / args.workers
                    if done % 50 == 0 or done <= 10:
                        print(f"  [{done}/{len(passages)}] {tid} p{seq} ({elapsed:.1f}s, "
                              f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
                elif status == 'skip' and done % 500 == 0:
                    print(f"  [{done}/{len(passages)}] skipping...", flush=True)
    else:
        for i, p in enumerate(passages):
            status, tid, seq, elapsed = run_one(p)
            counts[status] += 1
            if status == 'ok':
                total_time += elapsed
                avg = total_time / counts['ok']
                remaining = (len(passages) - i - 1) * avg
                if (i + 1) % 50 == 0 or i < 10:
                    print(f"  [{i+1}/{len(passages)}] {tid} p{seq} ({elapsed:.1f}s, "
                          f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
            elif status == 'skip' and (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(passages)}] skipping...", flush=True)

    print(f"\nDone: {counts['ok']} ok, {counts['skip']} skipped, "
          f"{counts['error']} errors, {counts['empty']} empty", file=sys.stderr)
    print(f"Total API time: {total_time:.0f}s ({total_time/60:.1f}min)", file=sys.stderr)


if __name__ == '__main__':
    main()
