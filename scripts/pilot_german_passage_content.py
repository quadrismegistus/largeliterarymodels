"""Pilot: PassageContentTask on ~10K balanced German passages (arc_fiction_de).

Sample: ~2,500 passages per half-century (1600-49, 1650-99, 1700-49, 1750-99),
pre-shuffled so early cancellation still yields a useful cross-section.

Usage:
    python scripts/pilot_german_passage_content.py --model qwen3.5-35b-a3b --num-workers 3
    python scripts/pilot_german_passage_content.py --model sonnet --num-workers 4
"""

import argparse
import os
import sys
import time

import lltk
import pandas as pd

from largeliterarymodels.tasks import PassageContentTask, format_passage

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_CSV = '/Users/rj416/github/largeliterarymodels/data/german_passages_sample_balanced.csv'

MODEL_TABLE = {
    'qwen3.5-35b-a3b': 'lmstudio/qwen3.5-35b-a3b',
    'gemma-31b':        'lmstudio/gemma-4-31b-it',
    'llama-70b':        'lmstudio/meta-llama-3.1-70b-instruct',
    'sonnet':           'claude-sonnet-4-6',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3.5-35b-a3b',
                        choices=list(MODEL_TABLE.keys()))
    parser.add_argument('--num-workers', type=int, default=3)
    parser.add_argument('--sample-csv', default=SAMPLE_CSV)
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit to first N passages (0 = all)')
    args = parser.parse_args()

    model = MODEL_TABLE[args.model]
    slug = args.model.replace('.', '').replace(':', '-').replace('/', '-')
    audit_csv = f'/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_german_passage_content_{slug}.csv'

    print(f"model={model} workers={args.num_workers}", file=sys.stderr, flush=True)

    sample = pd.read_csv(args.sample_csv)
    if args.limit:
        sample = sample.head(args.limit)
    print(f"Sample: {len(sample)} passages from {sample['_id'].nunique()} texts",
          file=sys.stderr, flush=True)

    ids = sample['_id'].unique().tolist()
    print(f"Fetching passage text for {len(ids)} texts…", file=sys.stderr, flush=True)
    pdf = lltk.db.get_passages(ids)
    print(f"  got {len(pdf)} total passages", file=sys.stderr, flush=True)

    escaped = ', '.join("'" + _id.replace("'", "''") + "'" for _id in ids)
    meta_df = lltk.db.query(f"""
        SELECT _id, title, author, year
        FROM lltk.texts FINAL
        WHERE _id IN ({escaped})
    """)
    text_meta = {}
    for _, r in meta_df.iterrows():
        text_meta[r['_id']] = {
            'title': str(r.get('title') or ''),
            'author': str(r.get('author') or ''),
            'year': int(r['year']) if r.get('year') else None,
        }

    wanted = set(zip(sample['_id'], sample['seq'].astype(int)))
    pdf = pdf[pdf.apply(lambda r: (r['_id'], int(r['seq'])) in wanted, axis=1)]
    print(f"  matched {len(pdf)} passages to sample", file=sys.stderr, flush=True)

    pdf_lookup = {}
    for _, r in pdf.iterrows():
        pdf_lookup[(r['_id'], int(r['seq']))] = r

    prompts, metas, audit = [], [], []
    skipped = 0
    for _, sr in sample.iterrows():
        key = (sr['_id'], int(sr['seq']))
        if key not in pdf_lookup:
            skipped += 1
            continue
        r = pdf_lookup[key]
        tm = text_meta.get(sr['_id'], {})
        prompt, meta = format_passage(
            r['text'],
            title=tm.get('title') or '',
            author=tm.get('author') or '',
            year=tm.get('year'),
            _id=sr['_id'],
            section_id=f"p500:{int(sr['seq'])}",
        )
        prompts.append(prompt)
        metas.append(meta)
        audit.append({
            '_id': sr['_id'],
            'seq': int(sr['seq']),
            'year': sr.get('year'),
            'title': tm.get('title', ''),
            'author': tm.get('author', ''),
            'n_words': int(r['n_words']),
        })

    if skipped:
        print(f"  skipped {skipped} passages (not found in lltk.passages)",
              file=sys.stderr, flush=True)
    print(f"Total passages to annotate: {len(prompts)}", file=sys.stderr, flush=True)

    task = PassageContentTask()
    t0 = time.time()
    results = task.map(
        prompts, model=model, metadata_list=metas,
        num_workers=args.num_workers, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\ntask.map done: {len(results)} results in {elapsed/60:.1f} min "
          f"({elapsed/max(1,len(results)):.1f}s/passage)",
          file=sys.stderr, flush=True)

    rows = []
    n_failed = 0
    for a, result in zip(audit, results):
        if result is None:
            n_failed += 1
            rows.append({**a, '_failed': True})
            continue
        d = result.model_dump()
        rows.append({**a, **d, '_failed': False})

    if n_failed:
        print(f"  {n_failed} passages failed after retries",
              file=sys.stderr, flush=True)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(audit_csv), exist_ok=True)
    out_df.to_csv(audit_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {audit_csv}", file=sys.stderr, flush=True)
    print(f"Done in {elapsed/60:.1f} min total", file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
