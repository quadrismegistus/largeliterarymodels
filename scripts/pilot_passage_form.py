"""Pilot: run PassageFormTask on an existing passage manifest.

Replays the sample manifest produced by pilot_passage_content.py (V1: 1,110
passages across ~130 romance/novel/contrast texts) through PassageFormTask.
No sampling logic here — use the content-pilot script to (re)produce the
manifest if needed.

Usage:
    python scripts/pilot_passage_form.py --model sonnet --num-workers 4
    python scripts/pilot_passage_form.py --model opus --num-workers 2  \\
        --manifest data/pilot_2026-04_passage_content_sample.csv

Audit CSV columns: manifest cols + all PassageFormAnnotation fields + _failed.
"""

import argparse
import os
import random
import sys
import time

import lltk
import pandas as pd
from largeliterarymodels.tasks import PassageFormTask, format_passage


MODEL_TAGS = {
    'sonnet':       'claude-sonnet-4-6',
    'opus':         'claude-opus-4-7',
    'haiku':        'claude-haiku-4-5-20251001',
    'gemini-flash': 'gemini-2.5-flash',
    'gemini-pro':   'gemini-2.5-pro',
    'qwen-35b':     'lmstudio/qwen3.5-35b-a3b',
    'gemma-31b':    'lmstudio/gemma-4-31b-it',
}


def resolve_model(tag):
    if tag in MODEL_TAGS:
        return MODEL_TAGS[tag]
    if '/' in tag or tag.startswith(('claude-', 'gemini-', 'gpt-')):
        return tag
    raise SystemExit(f"Unknown model tag: {tag!r}. Known: {sorted(MODEL_TAGS)}")


def load_manifest(path):
    manifest = pd.read_csv(path)
    for col in ('title', 'author', 'tag_label'):
        if col in manifest.columns:
            manifest[col] = manifest[col].fillna('')
    return manifest


def build_prompts(manifest):
    ids = list(dict.fromkeys(manifest['_id'].tolist()))
    tag_labels = dict(zip(manifest['_id'], manifest['tag_label']))
    text_meta = {
        r['_id']: {
            'title': r.get('title') or '',
            'author': r.get('author') or '',
            'year': r.get('year') if pd.notna(r.get('year')) else None,
        }
        for _, r in manifest.drop_duplicates('_id').iterrows()
    }

    wanted = set(zip(manifest['_id'], manifest['seq'].astype(int)))
    pdf = lltk.db.get_passages(ids)
    pdf = pdf[pdf.apply(lambda r: (r['_id'], int(r['seq'])) in wanted, axis=1)]

    prompts, metas, audit = [], [], []
    for _, r in pdf.sort_values(['_id', 'seq']).iterrows():
        tm = text_meta.get(r['_id'], {})
        prompt, meta = format_passage(
            r['text'],
            title=tm.get('title') or '',
            author=tm.get('author') or '',
            year=tm.get('year'),
            _id=r['_id'],
            section_id=f"p500:{int(r['seq'])}",
        )
        prompts.append(prompt)
        metas.append(meta)
        audit.append({
            '_id': r['_id'],
            'seq': int(r['seq']),
            'tag_label': tag_labels.get(r['_id'], ''),
            'title': tm.get('title') or '',
            'author': tm.get('author') or '',
            'year': tm.get('year'),
            'n_words': int(r['n_words']),
        })
    return prompts, metas, audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='sonnet')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--manifest', default=(
        '/Users/rj416/github/largeliterarymodels/data/'
        'pilot_2026-04_passage_content_sample.csv'
    ))
    p.add_argument('--audit-csv', default=None,
                   help='Default: auto-named under data/ by model')
    p.add_argument('--shuffle-seed', type=int, default=42)
    args = p.parse_args()

    model = resolve_model(args.model)
    slug = args.model.replace('.', '').replace(':', '-').replace('/', '-')
    audit_csv = args.audit_csv or (
        f'/Users/rj416/github/largeliterarymodels/data/'
        f'pilot_2026-04_passage_form_{slug}.csv'
    )
    print(f"model={model} num_workers={args.num_workers}", file=sys.stderr)
    print(f"manifest={args.manifest}", file=sys.stderr)
    print(f"audit_csv={audit_csv}", file=sys.stderr)

    manifest = load_manifest(args.manifest)
    prompts, metas, audit = build_prompts(manifest)
    print(f"Total passages: {len(prompts)}", file=sys.stderr)

    rng = random.Random(args.shuffle_seed)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    metas = [metas[i] for i in order]
    audit = [audit[i] for i in order]

    task = PassageFormTask()
    t0 = time.time()
    results = task.map(prompts, model=model, metadata_list=metas,
                       num_workers=args.num_workers, verbose=True)
    elapsed = time.time() - t0
    print(f"\ntask.map done: {len(results)} in {elapsed/60:.1f} min "
          f"({elapsed/max(1,len(results)):.2f}s/passage)", file=sys.stderr)

    rows, n_failed = [], 0
    for a, result in zip(audit, results):
        if result is None:
            n_failed += 1
            rows.append({**a, '_failed': True})
            continue
        rows.append({**a, **result.model_dump(), '_failed': False})
    if n_failed:
        print(f"Skipped {n_failed} failed extractions", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(audit_csv), exist_ok=True)
    out_df.to_csv(audit_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {audit_csv}", file=sys.stderr)


if __name__ == '__main__':
    main()
