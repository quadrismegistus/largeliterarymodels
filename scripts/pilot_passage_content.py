"""Pilot: run PassageContentTask on a curated sample of romance/novel-tagged texts.

Pulls _ids via lltk.text_genre_tags + lltk.passages_meta (per lltk-claude's SQL
template), fetches passage text via lltk.db.get_passages, formats with
format_passage, runs task.map with configurable parallelism, writes an audit CSV.

Default sample (from 2026-04-20 coordination with lltk-claude):
    50 romance (form=romance, has passages, year<=1800)
    50 novel-only (form=novel AND NOT tagged romance, has passages, year<=1800)
    30 contrast (gothic / amatory / picaresque at mode+register, has passages, year<=1800)
Total ≈ 130 texts → ~1500-2000 passages at ~10-15 per text.

Usage:
    python scripts/pilot_passage_content.py --model qwen3.5-35b-a3b --num-workers 3
    python scripts/pilot_passage_content.py --model sonnet --num-workers 4
    python scripts/pilot_passage_content.py --n-romance 10 --n-novel 10 --n-contrast 5 \\
        --model qwen3.5-35b-a3b --num-workers 3  # smaller smoke run
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import date

import lltk
from largeliterarymodels.tasks import PassageContentTask, format_passage


# Tags (per lltk-claude) — novel/romance are at facet='form'; gothic/amatory/picaresque
# live at facet IN ('mode', 'register'). We don't filter by facet on contrast tags
# because they're cleanly unambiguous in the text_genre_tags table.
CONTRAST_TAGS = ('gothic', 'amatory', 'picaresque')


def resolve_model(tag):
    table = {
        'e2b':             'ollama/gemma4:e2b',
        'e4b':             'ollama/gemma4:e4b',
        '31b':             'ollama/gemma4:31b',
        'gguf-31b':        'lmstudio/gemma-4-31b-it',
        'qwen3.5-35b-a3b': 'lmstudio/qwen3.5-35b-a3b',
        'llama-70b':       'lmstudio/meta-llama-3.1-70b-instruct',
        'sonnet':          'claude-sonnet-4-6',
        'opus':            'claude-opus-4-7',
    }
    if tag not in table:
        raise SystemExit(f"Unknown model tag: {tag}. Choices: {list(table)}")
    return table[tag]


def sample_ids_haywood_era(n_total, year_min=1680, year_max=1740, corpus=None):
    """Sample ~n_total texts tagged as novel OR amatory in the 1680-1740 window.
    Single bucket tagged 'haywood_era'. Used to characterize the Haywood/Behn/Manley
    short amatory novel and its transition into Richardsonian formal realism."""
    year_clause = f"AND tgt._id IN (SELECT _id FROM lltk.texts FINAL WHERE year BETWEEN {year_min} AND {year_max})"
    corpus_clause = f"AND tgt._id LIKE '_{corpus}/%'" if corpus else ""
    sql = f"""
        SELECT DISTINCT tgt._id FROM lltk.text_genre_tags tgt
        JOIN (SELECT DISTINCT _id FROM lltk.passages_meta) pm ON tgt._id = pm._id
        WHERE tgt.tag IN ('novel', 'amatory')
          AND tgt.facet IN ('form', 'mode', 'register')
        {year_clause} {corpus_clause}
        ORDER BY tgt._id
        LIMIT {n_total}
    """
    ids = lltk.db.query(sql)["_id"].tolist()
    print(f"Haywood-era sample: {len(ids)} texts ({year_min}-{year_max}, novel|amatory)",
          file=sys.stderr, flush=True)
    tag_labels = {tid: 'haywood_era' for tid in ids}
    return ids, tag_labels


def sample_ids(n_romance, n_novel, n_contrast, year_max, corpus=None):
    """Return the combined deterministic _id list."""
    year_clause = f"AND tgt._id IN (SELECT _id FROM lltk.texts FINAL WHERE year <= {year_max})" if year_max else ""
    corpus_clause = f"AND tgt._id LIKE '_{corpus}/%'" if corpus else ""

    romance_sql = f"""
        SELECT DISTINCT tgt._id FROM lltk.text_genre_tags tgt
        JOIN (SELECT DISTINCT _id FROM lltk.passages_meta) pm ON tgt._id = pm._id
        WHERE tgt.tag = 'romance' AND tgt.facet = 'form'
        {year_clause} {corpus_clause}
        ORDER BY tgt._id
        LIMIT {n_romance}
    """
    novel_only_sql = f"""
        SELECT DISTINCT tgt._id FROM lltk.text_genre_tags tgt
        JOIN (SELECT DISTINCT _id FROM lltk.passages_meta) pm ON tgt._id = pm._id
        WHERE tgt.tag = 'novel' AND tgt.facet = 'form'
          AND tgt._id NOT IN (SELECT _id FROM lltk.text_genre_tags WHERE tag = 'romance')
        {year_clause} {corpus_clause}
        ORDER BY tgt._id
        LIMIT {n_novel}
    """
    contrast_tag_list = ", ".join(f"'{t}'" for t in CONTRAST_TAGS)
    contrast_sql = f"""
        SELECT DISTINCT tgt._id FROM lltk.text_genre_tags tgt
        JOIN (SELECT DISTINCT _id FROM lltk.passages_meta) pm ON tgt._id = pm._id
        WHERE tgt.tag IN ({contrast_tag_list})
        {year_clause} {corpus_clause}
        ORDER BY tgt._id
        LIMIT {n_contrast}
    """

    romance_ids = lltk.db.query(romance_sql)["_id"].tolist()
    novel_ids = lltk.db.query(novel_only_sql)["_id"].tolist()
    contrast_ids = lltk.db.query(contrast_sql)["_id"].tolist()

    print(f"Sampled: {len(romance_ids)} romance + {len(novel_ids)} novel-only + "
          f"{len(contrast_ids)} contrast = {len(romance_ids)+len(novel_ids)+len(contrast_ids)} texts",
          file=sys.stderr, flush=True)

    # Text-level tag labels for downstream audit
    tag_labels = {}
    for tid in romance_ids:
        tag_labels[tid] = 'romance'
    for tid in novel_ids:
        tag_labels[tid] = 'novel_only'
    for tid in contrast_ids:
        tag_labels[tid] = 'contrast'
    return list(tag_labels.keys()), tag_labels


def fetch_text_metadata(ids):
    """Return {_id: {title, author, year}} for the sampled ids via lltk.texts.meta."""
    if not ids:
        return {}
    id_list = ", ".join(f"'{i}'" for i in ids)
    df = lltk.db.query(f"SELECT _id, meta FROM lltk.texts FINAL WHERE _id IN ({id_list})")
    out = {}
    for _, r in df.iterrows():
        m = json.loads(r['meta']) if isinstance(r['meta'], str) else (r['meta'] or {})
        out[r['_id']] = {
            'title': m.get('title') or m.get('title_main') or '',
            'author': m.get('author') or '',
            'year': m.get('year') or m.get('year_orig') or None,
        }
    # Fallback via Text object for any missing or empty metadata
    missing = [i for i in ids if not out.get(i, {}).get('title')]
    if missing:
        for tid in missing:
            try:
                corpus_id, local_id = tid.lstrip('_').split('/', 1)
                C = lltk.load(corpus_id)
                t = C.text(local_id)
                out[tid] = {
                    'title': getattr(t, 'title', '') or '',
                    'author': getattr(t, 'author', '') or '',
                    'year': getattr(t, 'year', None),
                }
            except Exception as e:
                print(f"  meta lookup failed for {tid}: {e}", file=sys.stderr)
                out.setdefault(tid, {'title': '', 'author': '', 'year': None})
    return out


def build_prompts(ids, tag_labels, text_meta, passages_per_text=None):
    """Pull passages via lltk.db.get_passages, format each with format_passage.
    If passages_per_text is set, sample that many per text evenly across seqs
    (stratified — first, last, and evenly spaced middle). Returns (prompts,
    metadata_list, audit_rows)."""
    print(f"Fetching passages for {len(ids)} texts from lltk.passages…",
          file=sys.stderr, flush=True)
    df = lltk.db.get_passages(ids)
    print(f"  got {len(df)} total passages across {df['_id'].nunique()} texts",
          file=sys.stderr, flush=True)

    if passages_per_text:
        # Stratified: pick passages_per_text evenly spaced seqs per text.
        picked = []
        for tid, group in df.sort_values(['_id', 'seq']).groupby('_id'):
            n = len(group)
            k = min(passages_per_text, n)
            if k == n:
                picked.append(group)
            else:
                # Evenly spaced indices including first and last.
                indices = [int(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
                picked.append(group.iloc[indices])
        df = __import__('pandas').concat(picked, ignore_index=True)
        print(f"  sampled down to {len(df)} passages "
              f"({passages_per_text}/text, stratified)", file=sys.stderr, flush=True)

    prompts, metas, audit = [], [], []
    for _, r in df.sort_values(['_id', 'seq']).iterrows():
        tid = r['_id']
        tm = text_meta.get(tid, {})
        prompt, meta = format_passage(
            r['text'],
            title=tm.get('title') or '',
            author=tm.get('author') or '',
            year=tm.get('year'),
            _id=tid,
            section_id=f"p500:{int(r['seq'])}",
        )
        prompts.append(prompt)
        metas.append(meta)
        audit.append({
            '_id': tid,
            'seq': int(r['seq']),
            'tag_label': tag_labels.get(tid, ''),
            'title': tm.get('title') or '',
            'author': tm.get('author') or '',
            'year': tm.get('year'),
            'n_words': int(r['n_words']),
        })
    return prompts, metas, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3.5-35b-a3b',
                        choices=['e2b', 'e4b', '31b', 'gguf-31b',
                                 'qwen3.5-35b-a3b', 'llama-70b',
                                 'sonnet', 'opus'])
    parser.add_argument('--num-workers', type=int, default=3)
    parser.add_argument('--n-romance', type=int, default=50)
    parser.add_argument('--n-novel', type=int, default=50)
    parser.add_argument('--n-contrast', type=int, default=30)
    parser.add_argument('--passages-per-text', type=int, default=10,
                        help='Stratified sample of N passages per text (first+last+evenly '
                             'spaced middle). 0 = all passages. Default 10.')
    parser.add_argument('--year-max', type=int, default=1800,
                        help='Filter texts to year <= this (0 = no filter)')
    parser.add_argument('--corpus', default=None,
                        help='Restrict to _<corpus>/... prefix (e.g. chadwyck)')
    parser.add_argument('--audit-csv', default=None,
                        help='Output audit CSV path (default auto-named under data/)')
    parser.add_argument('--ids-csv', default='/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_passage_content_sample.csv',
                        help='Canonical sample manifest (_id, seq, tag_label, title, author, year, n_words). '
                             'Written if absent + sampling runs fresh; read if present + sampling skipped. '
                             'Use this to replay the exact sample across models.')
    parser.add_argument('--shuffle-seed', type=int, default=42,
                        help='Shuffle prompts before task.map so early results span all texts/subgenres '
                             'for mid-run inspection. Deterministic (same seed = same order).')
    parser.add_argument('--preset', default='main', choices=['main', 'haywood-era'],
                        help='Sampling preset. "main" = 50 romance + 50 novel + 30 contrast. '
                             '"haywood-era" = ~40 texts tagged novel|amatory in 1680-1740.')
    parser.add_argument('--n-haywood', type=int, default=40,
                        help='Total texts for --preset=haywood-era (default 40)')
    args = parser.parse_args()

    model = resolve_model(args.model)
    slug = args.model.replace('.', '').replace(':', '-').replace('/', '-')
    preset_slug = '' if args.preset == 'main' else f'_{args.preset.replace("-","_")}'
    audit_csv = args.audit_csv or \
        f'/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_passage_content{preset_slug}_{slug}.csv'
    if args.ids_csv == '/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_passage_content_sample.csv' and args.preset != 'main':
        args.ids_csv = f'/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_passage_content{preset_slug}_sample.csv'

    print(f"model={model} num_workers={args.num_workers}", file=sys.stderr, flush=True)
    print(f"audit_csv={audit_csv}", file=sys.stderr, flush=True)

    import pandas as pd
    if args.ids_csv and os.path.exists(args.ids_csv):
        print(f"Loading sample manifest from {args.ids_csv}", file=sys.stderr, flush=True)
        manifest = pd.read_csv(args.ids_csv)
        # Coerce NaN to empty string / None so downstream format_passage sees strings not floats.
        for col in ('title', 'author', 'tag_label'):
            if col in manifest.columns:
                manifest[col] = manifest[col].fillna('')
        ids = list(dict.fromkeys(manifest['_id'].tolist()))
        tag_labels = dict(zip(manifest['_id'], manifest['tag_label']))
        text_meta = {r['_id']: {'title': r.get('title') or '',
                                'author': r.get('author') or '',
                                'year': r.get('year') if pd.notna(r.get('year')) else None}
                     for _, r in manifest.drop_duplicates('_id').iterrows()}
        # Rebuild prompts from the explicit (_id, seq) pairs in the manifest
        wanted = set(zip(manifest['_id'], manifest['seq']))
        pdf = lltk.db.get_passages(ids)
        pdf = pdf[pdf.apply(lambda r: (r['_id'], int(r['seq'])) in wanted, axis=1)]
        prompts, metas, audit = [], [], []
        for _, r in pdf.sort_values(['_id','seq']).iterrows():
            tm = text_meta.get(r['_id'], {})
            prompt, meta = format_passage(
                r['text'], title=tm.get('title') or '', author=tm.get('author') or '',
                year=tm.get('year'), _id=r['_id'], section_id=f"p500:{int(r['seq'])}",
            )
            prompts.append(prompt)
            metas.append(meta)
            audit.append({'_id': r['_id'], 'seq': int(r['seq']),
                          'tag_label': tag_labels.get(r['_id'], ''),
                          'title': tm.get('title') or '', 'author': tm.get('author') or '',
                          'year': tm.get('year'), 'n_words': int(r['n_words'])})
    else:
        if args.preset == 'haywood-era':
            ids, tag_labels = sample_ids_haywood_era(
                args.n_haywood, year_min=1680, year_max=1740, corpus=args.corpus,
            )
        else:
            ids, tag_labels = sample_ids(
                args.n_romance, args.n_novel, args.n_contrast,
                args.year_max if args.year_max > 0 else None,
                args.corpus,
            )
        if not ids:
            raise SystemExit("No texts sampled — check filters.")

        text_meta = fetch_text_metadata(ids)
        prompts, metas, audit = build_prompts(
            ids, tag_labels, text_meta,
            passages_per_text=args.passages_per_text or None,
        )
        # Persist the manifest so future runs (and different models) use the exact same sample.
        if args.ids_csv:
            os.makedirs(os.path.dirname(args.ids_csv), exist_ok=True)
            pd.DataFrame(audit).to_csv(args.ids_csv, index=False)
            print(f"Wrote sample manifest to {args.ids_csv}", file=sys.stderr, flush=True)
    print(f"Total passages to annotate: {len(prompts)}", file=sys.stderr, flush=True)

    # Shuffle deterministically so early results span all texts/subgenres.
    rng = random.Random(args.shuffle_seed)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    metas = [metas[i] for i in order]
    audit = [audit[i] for i in order]

    task = PassageContentTask()
    t0 = time.time()
    results = task.map(
        prompts, model=model, metadata_list=metas,
        num_workers=args.num_workers, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\ntask.map done: {len(results)} results in {elapsed/60:.1f} min "
          f"({elapsed/max(1,len(results)):.1f}s/passage effective)",
          file=sys.stderr, flush=True)

    # Write audit CSV with bool flags + summary
    import pandas as pd
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
        print(f"Skipped {n_failed} passages where extraction failed after retries",
              file=sys.stderr, flush=True)
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(audit_csv), exist_ok=True)
    out_df.to_csv(audit_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {audit_csv}", file=sys.stderr, flush=True)

    # Quick subgenre-discrimination sanity summary (V2: list-field presence rates)
    print("\nList-field presence rates by tag_label (top 12 most variable):",
          file=sys.stderr, flush=True)
    ok_df = out_df[~out_df.get('_failed', False)].copy()
    list_fields = ['scene_content', 'setting', 'character_classes', 'character_genders',
                   'fantastical_elements', 'threats']
    all_rates = []
    for field in list_fields:
        if field not in ok_df.columns:
            continue
        sub = ok_df[['tag_label', field]].copy()
        sub = sub.explode(field).dropna(subset=[field])
        if sub.empty:
            continue
        cross = pd.crosstab(sub[field], sub['tag_label'])
        passage_counts = ok_df.groupby('tag_label').size()
        rates = cross.div(passage_counts, axis=1).fillna(0)
        rates.index = [f"{field}={v}" for v in rates.index]
        all_rates.append(rates)
    if all_rates:
        combined = pd.concat(all_rates)
        combined['spread'] = combined.max(axis=1) - combined.min(axis=1)
        top = combined.sort_values('spread', ascending=False).head(12)
        print(top.round(3).to_string(), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
