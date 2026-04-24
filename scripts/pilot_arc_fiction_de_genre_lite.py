"""Pilot: GenreTaskLite on arc_fiction_de pre-1800 texts (~277 texts).

Uses the same 60-tag vocabulary as the English pilot. The system prompt
references "early modern English" but the tag vocabulary (novel, romance,
Bildungsroman, picaresque, etc.) transfers to German fiction.

Usage:
    python scripts/pilot_arc_fiction_de_genre_lite.py --model gemini-pro --num-workers 4
    python scripts/pilot_arc_fiction_de_genre_lite.py --model qwen-35b --num-workers 3
"""

import argparse
import hashlib
import logging
import sys
import time

import lltk
import pandas as pd

from largeliterarymodels.tasks import GenreTaskLite
from largeliterarymodels.tasks.classify_genre import format_text_for_classification
from lltk.tools import annotations as A

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('pilot-genre-lite-de')

MODEL_CONFIGS = {
    'gemini-pro': {
        'model': 'gemini-3.1-pro-preview',
        'source': 'llm:gemini-3.1-pro-genre-lite',
    },
    'sonnet': {
        'model': 'claude-sonnet-4-6',
        'source': 'llm:claude-sonnet-4-6-genre-lite',
    },
    'qwen-35b': {
        'model': 'lmstudio/qwen3.5-35b-a3b',
        'source': 'llm:qwen3.5-35b-a3b-genre-lite',
    },
    'gemma-31b': {
        'model': 'lmstudio/gemma-4-31b-it',
        'source': 'llm:gemma4-31b-genre-lite',
    },
}

YEAR_MAX = 1800


def load_targets(source):
    log.info("loading arc_fiction_de texts pre-%d …", YEAR_MAX)
    c = lltk.Corpus('arc_fiction_de')
    mdf = c.metadf
    mdf = mdf[mdf['year'].notna() & (mdf['year'] > 0) & (mdf['year'] < YEAR_MAX)]
    log.info("  %d texts in arc_fiction_de pre-%d", len(mdf), YEAR_MAX)

    done_df = A.resolve_by_source(source, fields=['genre_raw'])
    done = set(done_df['_id']) if '_id' in done_df.columns and len(done_df) else set()
    if done:
        log.info("  %d already annotated by source='%s', skipping", len(done), source)
        mdf = mdf[~mdf['_id'].isin(done)]

    log.info("  %d texts to annotate", len(mdf))
    return mdf


def build_prompts(mdf):
    prompts = []
    metadata_list = []
    for _, r in mdf.iterrows():
        title = str(r.get('title') or '').strip()
        author = str(r.get('author') or '').strip() or None
        year = int(r['year']) if r.get('year') and 1400 <= int(r['year']) <= 1800 else None
        prompt = format_text_for_classification(
            title=title,
            author=author,
            year=year,
        )
        prompts.append(prompt)
        metadata_list.append({'_id': r['_id']})
    return prompts, metadata_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemini-pro',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    MODEL = cfg['model']
    SOURCE = cfg['source']
    slug = args.model.replace('-', '_')
    RUN_ID = f'2026-04-arc-fiction-de-genre-lite-{slug}'
    AUDIT_CSV = f'/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_arc_fiction_de_genre_lite_{slug}.csv'

    log.info("model=%s source=%s workers=%d", MODEL, SOURCE, args.num_workers)

    t0 = time.time()
    mdf = load_targets(SOURCE)
    if mdf.empty:
        log.info("nothing to do — all targets already annotated")
        return

    prompts, metadata_list = build_prompts(mdf)
    batch_ids = {m['_id'] for m in metadata_list}

    log.info("running GenreTaskLite on %d texts …", len(prompts))
    task = GenreTaskLite(model=MODEL)

    t_run = time.time()
    task.map(
        prompts,
        metadata_list=metadata_list,
        num_workers=args.num_workers,
        verbose=True,
    )
    log.info("task.map complete in %.1f min", (time.time() - t_run) / 60)

    task_df = task.df
    task_df = task_df[task_df['model'] == MODEL]
    task_df = task_df[task_df['meta__id'].isin(batch_ids)]
    task_df = task_df.drop_duplicates(subset=['meta__id'], keep='last')
    log.info("task.df returned %d unique rows", len(task_df))

    task_df['genre_tags_str'] = task_df['genre_tags'].apply(
        lambda x: '; '.join(x) if isinstance(x, list) else str(x)
    )

    log.info("top tag combinations:")
    for combo, n in task_df['genre_tags_str'].value_counts().head(20).items():
        log.info("  %s: %d", combo, n)

    # Write to lltk.annotations
    A.ensure_schema()
    PROMPT_META = {
        'prompt_variant': 'genre_lite_constrained',
        'sp_sha256_12': hashlib.sha256(task.system_prompt.encode()).hexdigest()[:12],
        'sp_len': len(task.system_prompt),
        'n_examples': len(task.examples),
        'n_tags_vocab': 60,
        'corpus': 'arc_fiction_de',
        'lang': 'de',
    }
    rows = []
    for _, row in task_df.iterrows():
        _id = row['meta__id']
        if not _id or str(_id) in ('nan', 'None'):
            continue
        base = {'_id': str(_id), 'confidence': 1.0, 'meta': PROMPT_META}

        tags_str = row.get('genre_tags_str', '')
        if tags_str:
            rows.append({**base, 'field': 'genre_raw', 'value': tags_str})

        y = row.get('year_estimated')
        if y and str(y).strip() and str(y).strip() != '0':
            try:
                yi = int(float(y))
                if -500 <= yi <= 2100:
                    rows.append({**base, 'field': 'year_estimated', 'value': yi})
            except (ValueError, TypeError):
                pass

        fn = row.get('author_first_name')
        if fn and str(fn).strip() and str(fn).strip().lower() not in ('nan', 'none'):
            rows.append({**base, 'field': 'author_first_name', 'value': str(fn).strip()})

    n_written = A.write(source=SOURCE, rows=rows, run_id=RUN_ID)
    log.info("wrote %d annotation rows to lltk.annotations", n_written)

    # Audit CSV
    audit = mdf.merge(task_df, left_on='_id', right_on='meta__id', how='left')
    audit_cols = ['_id', 'corpus', 'year', 'author', 'title',
                  'genre_tags_str', 'author_first_name', 'year_estimated']
    audit = audit[[c for c in audit_cols if c in audit.columns]]
    audit.to_csv(AUDIT_CSV, index=False)
    log.info("wrote audit CSV to %s", AUDIT_CSV)

    log.info("pilot complete in %.1f min total", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
