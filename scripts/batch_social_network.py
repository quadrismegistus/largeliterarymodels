"""Batch runner: SocialNetworkTask across a Chadwyck subcollection.

Queries lltk for all texts in a subcollection, runs SocialNetworkTask
on each, skipping any that already have saved output. Supports parallel
processing of multiple texts via --workers (each text's chunks are still
sequential, but multiple texts run concurrently to saturate the GPU).

Usage:
    # Early English Prose Fiction (110 texts, ~40 hours local)
    python scripts/batch_social_network.py --subcollection Early_English_Prose_Fiction

    # 4 texts in parallel on vLLM (saturates a single A100)
    python scripts/batch_social_network.py --subcollection Early_English_Prose_Fiction --workers 4

    # Shard across multiple processes (e.g. 4 terminals)
    python scripts/batch_social_network.py --subcollection Nineteenth-Century_Fiction --shard 1/4
    python scripts/batch_social_network.py --subcollection Nineteenth-Century_Fiction --shard 2/4

    # Single text (for testing)
    python scripts/batch_social_network.py --text-id _chadwyck/Early_English_Prose_Fiction/ee80010.02

    # Resume after interruption — just re-run the same command (cache handles it)
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import lltk

from largeliterarymodels.tasks import SocialNetworkTask

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

MODEL_TABLE = {
    'qwen36-27b': 'lmstudio/qwen/qwen3.6-27b',
    'sonnet': 'claude-sonnet-4-6',
    'vllm-qwen36': 'vllm/qwen3.6-27b',
    'vllm-llama70b': 'vllm/meta-llama-3.1-70b-instruct',
}


def output_path(text_id, model_str):
    """Match the filename convention from SequentialTask._save_result."""
    source_slug = text_id.replace('/', '_').replace(' ', '_').strip('_')
    m_slug = model_str.split('/')[-1].replace('.', '').replace(' ', '_')
    return os.path.join(DATA_DIR, f'social_network_{source_slug}_{m_slug}.json')


def get_text_ids(subcollection):
    df = lltk.db.query(f"""
        SELECT t._id, t.title, t.author, t.year, count(p._id) as n_passages
        FROM (SELECT * FROM texts FINAL) AS t
        JOIN passages AS p ON t._id = p._id
        WHERE t._id LIKE '_chadwyck/{subcollection}/%'
        GROUP BY t._id, t.title, t.author, t.year
        ORDER BY t.year, t._id
    """)
    return df


def run_one_text(text_id, model, verbose=True):
    """Process a single text. Runs in a worker process."""
    try:
        task = SocialNetworkTask(model=model)
        result = task.run(text_id, save=True, verbose=verbose)
        n_chars = len(result.get('characters', []))
        n_rels = len(result.get('relations', []))
        return text_id, 'done', f'{n_chars} chars, {n_rels} rels'
    except Exception as e:
        return text_id, 'fail', str(e)


def main():
    parser = argparse.ArgumentParser(description='Batch SocialNetworkTask runner')
    parser.add_argument('--subcollection', type=str,
                        help='Chadwyck subcollection name (e.g. Early_English_Prose_Fiction)')
    parser.add_argument('--text-id', type=str,
                        help='Single text ID to run (overrides --subcollection)')
    parser.add_argument('--model', default='qwen36-27b',
                        help=f'Model key or full model string. Keys: {list(MODEL_TABLE.keys())}')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of texts to process in parallel (default: 1). '
                             'Use 4-8 on vLLM to saturate a single GPU.')
    parser.add_argument('--shard', type=str, default=None,
                        help='Process only shard N of M texts (e.g. "1/4", "2/4"). '
                             'For running multiple independent processes.')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip texts that already have output files (default: True)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Force re-run even if output exists')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit to first N texts (0 = all)')
    parser.add_argument('--min-passages', type=int, default=5,
                        help='Skip texts with fewer passages than this')
    args = parser.parse_args()

    model = MODEL_TABLE.get(args.model, args.model)
    skip_existing = not args.no_skip

    if args.text_id:
        texts = [{'_id': args.text_id}]
        print(f"Single text: {args.text_id}", file=sys.stderr)
    elif args.subcollection:
        df = get_text_ids(args.subcollection)
        if args.min_passages:
            df = df[df['n_passages'] >= args.min_passages]
        texts = df.to_dict('records')
        total_passages = df['n_passages'].sum()
        print(f"{args.subcollection}: {len(texts)} texts, {total_passages} passages, "
              f"~{total_passages // 10} chunks", file=sys.stderr)
    else:
        parser.error('Provide --subcollection or --text-id')

    if args.shard:
        n, m = map(int, args.shard.split('/'))
        texts = [t for i, t in enumerate(texts) if i % m == (n - 1)]
        print(f"Shard {n}/{m}: {len(texts)} texts", file=sys.stderr)

    if args.limit:
        texts = texts[:args.limit]

    os.makedirs(DATA_DIR, exist_ok=True)

    # Filter out already-completed texts
    todo = []
    skipped = 0
    for t in texts:
        out = output_path(t['_id'], model)
        if skip_existing and os.path.exists(out):
            skipped += 1
        else:
            todo.append(t)

    if skipped:
        print(f"Skipping {skipped} texts with existing output", file=sys.stderr)
    print(f"Processing {len(todo)} texts with {args.workers} worker(s)",
          file=sys.stderr)

    t0 = time.time()
    done, failed = 0, 0

    if args.workers <= 1:
        for i, t in enumerate(todo):
            text_id = t['_id']
            label = f"[{i+1}/{len(todo)}]"
            title = t.get('title', '')[:50]
            year = t.get('year', '?')
            n_psg = t.get('n_passages', '?')
            print(f"\n{label} {text_id} ({year}) {title}... [{n_psg} passages]",
                  file=sys.stderr)

            text_id, status, msg = run_one_text(text_id, model)
            elapsed_text = time.time() - t0
            if status == 'done':
                done += 1
                print(f"{label} DONE {text_id}: {msg} [{elapsed_text/60:.0f}m elapsed]",
                      file=sys.stderr)
            else:
                failed += 1
                print(f"{label} FAIL {text_id}: {msg}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for t in todo:
                f = pool.submit(run_one_text, t['_id'], model, verbose=False)
                futures[f] = t

            for f in as_completed(futures):
                t = futures[f]
                text_id, status, msg = f.result()
                elapsed_text = time.time() - t0
                if status == 'done':
                    done += 1
                    print(f"DONE {text_id}: {msg} [{elapsed_text/60:.0f}m elapsed]",
                          file=sys.stderr)
                else:
                    failed += 1
                    print(f"FAIL {text_id}: {msg}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Batch complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {elapsed/3600:.1f}h", file=sys.stderr)


if __name__ == '__main__':
    main()
