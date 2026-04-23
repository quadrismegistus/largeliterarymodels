"""Pilot: GenreTaskLite (constrained 60-tag vocabulary) on the 666 arc_fiction
pre-1800 texts. Comparable to the free-text GenreTask runs for reliability
measurement.

Model: lmstudio/gemma-4-31b-it (GGUF, chosen for best local GenreTask Jaccard).
Source: llm:gemma4-31b-genre-lite
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
from largeliterarymodels.integrations.lltk import write_task_to_lltk
from lltk.tools import annotations as A

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('pilot-genre-lite')

MODEL_CONFIGS = {
    'gemma-31b': {
        'model': 'lmstudio/gemma-4-31b-it',
        'source': 'llm:gemma4-31b-genre-lite',
    },
    'qwen-35b': {
        'model': 'lmstudio/qwen3.5-35b-a3b',
        'source': 'llm:qwen3.5-35b-a3b-genre-lite',
    },
    'llama-70b': {
        'model': 'lmstudio/meta-llama-3.1-70b-instruct',
        'source': 'llm:llama-3.1-70b-genre-lite',
    },
    'sonnet': {
        'model': 'claude-sonnet-4-6',
        'source': 'llm:claude-sonnet-4-6-genre-lite',
    },
    'gemini-pro': {
        'model': 'gemini-3.1-pro-preview',
        'source': 'llm:gemini-3.1-pro-genre-lite',
    },
}

REFERENCE_CSV = '/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_arc_fiction_pre1800_genre_raw_gemma_4_31b_it_gguf.csv'


def load_target_ids_and_metadata(source, ids_file=None):
    if ids_file:
        with open(ids_file) as f:
            target_ids = [line.strip() for line in f if line.strip()]
        log.info("loaded %d target _ids from %s", len(target_ids), ids_file)
    else:
        ref = pd.read_csv(REFERENCE_CSV)
        target_ids = ref['_id'].tolist()
        log.info("loaded %d target _ids from reference CSV", len(target_ids))

    done_df = A.resolve_by_source(source, fields=['genre_raw'])
    done = set(done_df['_id']) if '_id' in done_df.columns and len(done_df) else set()
    if done:
        log.info("  %d _ids already annotated by source='%s', skipping", len(done), source)
        target_ids = [i for i in target_ids if i not in done]

    log.info("  %d _ids to annotate", len(target_ids))
    return target_ids, ref if not ids_file else None


def load_metadata_batch(ids):
    log.info("fetching metadata …")
    escaped = ', '.join(f"'{_id.replace(chr(39), chr(39)+chr(39))}'" for _id in ids)
    df = lltk.db.query(f"""
        SELECT
          _id, title, author, year, corpus,
          JSONExtractString(meta, 'estc_title') AS estc_title,
          JSONExtractString(meta, 'estc_title_sub') AS estc_title_sub,
          JSONExtractString(meta, 'estc_subject_topic') AS estc_subject_topic,
          JSONExtractString(meta, 'estc_form') AS estc_form
        FROM lltk.texts FINAL
        WHERE _id IN ({escaped})
    """)
    log.info("  returned %d rows", len(df))
    return df


def build_title(row):
    title = (row.get('estc_title') or '').strip()
    if not title:
        title = (row.get('title') or '').strip()
    sub = (row.get('estc_title_sub') or '').strip()
    if sub:
        if title and not title.endswith((':', ';', '.', '?', '!', ',')):
            title = f"{title}: {sub}"
        elif title:
            title = f"{title} {sub}"
        else:
            title = sub
    return title.strip()


def build_prompts(df):
    prompts = []
    metadata_list = []
    for _, r in df.iterrows():
        title = build_title(r)
        year = int(r['year']) if r['year'] and 1400 <= int(r['year']) <= 1800 else None
        author = (r.get('author') or '').strip() or None
        subject_topic = (r.get('estc_subject_topic') or '').strip() or None
        form = (r.get('estc_form') or '').strip() or None
        prompt = format_text_for_classification(
            title=title or '',
            author=author,
            year=year,
            subject_topic=subject_topic,
            form=form,
        )
        prompts.append(prompt)
        metadata_list.append({'_id': r['_id']})
    return prompts, metadata_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma-31b',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model tag (default: gemma-31b)')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--ids-file', default=None,
                        help='Path to newline-separated _id list (overrides reference CSV)')
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    MODEL = cfg['model']
    SOURCE = cfg['source']
    slug = args.model.replace('-', '_')
    RUN_ID = f'2026-04-arc-fiction-genre-lite-{slug}'
    AUDIT_CSV = f'/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_arc_fiction_genre_lite_{slug}.csv'

    log.info("model=%s source=%s workers=%d", MODEL, SOURCE, args.num_workers)

    t0 = time.time()
    target_ids, ref = load_target_ids_and_metadata(SOURCE, ids_file=args.ids_file)
    if not target_ids:
        log.info("nothing to do — all targets already annotated")
        return

    df = load_metadata_batch(target_ids)
    prompts, metadata_list = build_prompts(df)
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

    # genre_tags is a list — join with '; ' for genre_raw-compatible annotation
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
    audit = df.merge(task_df, left_on='_id', right_on='meta__id', how='left')
    audit_cols = ['_id', 'corpus', 'year', 'author', 'title',
                  'estc_title', 'estc_title_sub', 'estc_subject_topic', 'estc_form',
                  'genre_tags_str', 'author_first_name', 'year_estimated']
    audit = audit[[c for c in audit_cols if c in audit.columns]]
    audit.to_csv(AUDIT_CSV, index=False)
    log.info("wrote audit CSV to %s", AUDIT_CSV)

    log.info("pilot complete in %.1f min total", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
