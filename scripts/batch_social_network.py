"""Batch runner: SocialNetworkTask across a Chadwyck subcollection.

Queries lltk for all texts in a subcollection, runs SocialNetworkTask
on each sequentially, skipping any that already have saved output.

Usage:
    # Early English Prose Fiction (110 texts, ~40 hours local)
    python scripts/batch_social_network.py --subcollection Early_English_Prose_Fiction

    # 18C Fiction
    python scripts/batch_social_network.py --subcollection Eighteenth-Century_Fiction

    # Single text (for testing)
    python scripts/batch_social_network.py --text-id _chadwyck/Early_English_Prose_Fiction/ee80010.02

    # With a different model
    python scripts/batch_social_network.py --subcollection Early_English_Prose_Fiction --model vllm/qwen3.6-27b

    # Resume after interruption — just re-run the same command (cache handles it)
"""

import argparse
import json
import os
import re
import sys
import time

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


def model_slug(model_str):
    return re.sub(r'[^a-zA-Z0-9]', '-', model_str).strip('-')


def output_path(text_id, model_str):
    slug = text_id.replace('_chadwyck/', '').replace('/', '_')
    return os.path.join(DATA_DIR, f'social_network_{slug}_{model_slug(model_str)}.json')


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


def main():
    parser = argparse.ArgumentParser(description='Batch SocialNetworkTask runner')
    parser.add_argument('--subcollection', type=str,
                        help='Chadwyck subcollection name (e.g. Early_English_Prose_Fiction)')
    parser.add_argument('--text-id', type=str,
                        help='Single text ID to run (overrides --subcollection)')
    parser.add_argument('--model', default='qwen36-27b',
                        help=f'Model key or full model string. Keys: {list(MODEL_TABLE.keys())}')
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

    if args.limit:
        texts = texts[:args.limit]

    os.makedirs(DATA_DIR, exist_ok=True)

    task = SocialNetworkTask(model=model)
    t0 = time.time()
    done, skipped, failed = 0, 0, 0

    for i, t in enumerate(texts):
        text_id = t['_id']
        out = output_path(text_id, model)
        label = f"[{i+1}/{len(texts)}]"

        if skip_existing and os.path.exists(out):
            print(f"{label} SKIP {text_id} (output exists)", file=sys.stderr)
            skipped += 1
            continue

        title = t.get('title', '')[:50]
        year = t.get('year', '?')
        n_psg = t.get('n_passages', '?')
        print(f"\n{label} {text_id} ({year}) {title}... [{n_psg} passages]",
              file=sys.stderr)

        try:
            result = task.run(text_id, save=True, verbose=True)
            done += 1
            n_chars = len(result.get('characters', []))
            n_rels = len(result.get('relations', []))
            elapsed_text = time.time() - t0
            print(f"{label} DONE {text_id}: {n_chars} chars, {n_rels} rels "
                  f"[{elapsed_text/60:.0f}m elapsed]", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"{label} FAIL {text_id}: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Batch complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {elapsed/3600:.1f}h", file=sys.stderr)


if __name__ == '__main__':
    main()
