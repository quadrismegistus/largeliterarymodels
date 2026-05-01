"""Batch runner for MajorGenreTask.

Reads a JSONL of {_id, title, author_surname}, formats each as
"Title by Surname", runs MajorGenreTask, writes output JSONs.

Usage:
    python scripts/batch_major_genre.py -i data/major_genre_input.jsonl -m claude-opus-4-6 -w 4
    python scripts/batch_major_genre.py -i data/major_genre_input.jsonl -m claude-opus-4-6 -w 4  # resume
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)


def model_slug(model: str) -> str:
    return model.replace('/', '-').replace('.', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='JSONL with _id, title, author_surname')
    parser.add_argument('--output', '-o', default=None, help='Output dir (default: data/major_genre_output/)')
    parser.add_argument('--model', '-m', default='claude-opus-4-6')
    parser.add_argument('--workers', '-w', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    from largeliterarymodels.tasks import MajorGenreTask
    task = MajorGenreTask()
    slug = model_slug(args.model)

    if args.output is None:
        args.output = str(Path(__file__).parent.parent / 'data' / 'major_genre_output')
    os.makedirs(args.output, exist_ok=True)

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        rows = rows[:args.limit]

    print(f"Task: MajorGenreTask", file=sys.stderr)
    print(f"Texts: {len(rows)}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    def run_one(row):
        text_id = row['_id']
        title = row.get('title', '')
        surname = row.get('author_surname', '')

        out_name = f"{text_id.replace('/', '_')}_{slug}.json"
        out_path = os.path.join(args.output, out_name)

        if os.path.exists(out_path) and not args.force:
            return 'skip', text_id, 0

        prompt = f"{title} by {surname}" if surname else title
        if len(prompt) < 5:
            return 'empty', text_id, 0

        t0 = time.time()
        try:
            result = task.run(prompt, model=args.model, force=args.force)
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR {text_id}: {e} ({elapsed:.1f}s)", flush=True)
            return 'error', text_id, elapsed

        output = {
            'metadata': {
                '_id': text_id,
                'model': args.model,
                'task': 'major_genre',
                'title': title,
                'author_surname': surname,
            },
            **result.model_dump(),
        }
        with open(out_path, 'w') as fh:
            json.dump(output, fh, indent=2)
        return 'ok', text_id, elapsed

    counts = {'ok': 0, 'skip': 0, 'error': 0, 'empty': 0}
    total_time = 0
    done = 0

    if args.workers > 1:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in rows:
                futures[pool.submit(run_one, row)] = row
            for fut in as_completed(futures):
                done += 1
                status, tid, elapsed = fut.result()
                counts[status] += 1
                if status == 'ok':
                    total_time += elapsed
                    avg = total_time / counts['ok']
                    remaining = (len(rows) - done) * avg / max(args.workers, 1)
                    if done <= 10 or done % 50 == 0:
                        print(f"  [{done}/{len(rows)}] {tid} ({elapsed:.1f}s, "
                              f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
                elif status == 'skip' and done % 500 == 0:
                    print(f"  [{done}/{len(rows)}] skipping...", flush=True)
    else:
        for i, row in enumerate(rows):
            status, tid, elapsed = run_one(row)
            counts[status] += 1
            if status == 'ok':
                total_time += elapsed
                avg = total_time / counts['ok']
                remaining = (len(rows) - i - 1) * avg
                if i < 10 or (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(rows)}] {tid} ({elapsed:.1f}s, "
                          f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
            elif status == 'skip' and (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(rows)}] skipping...", flush=True)

    print(f"\nDone: {counts['ok']} ok, {counts['skip']} skipped, "
          f"{counts['error']} errors, {counts['empty']} empty", file=sys.stderr)
    print(f"Total API time: {total_time:.0f}s ({total_time/60:.1f}min)", file=sys.stderr)


if __name__ == '__main__':
    main()
