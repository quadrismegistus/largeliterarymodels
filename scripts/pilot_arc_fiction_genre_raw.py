"""Pilot: local gemma4:31b GenreTask subgenre tagging (`genre_raw`) on the
~781 arc_fiction pre-1800 texts that currently lack a `genre_raw` annotation.

Scope set by lltk-claude: arc_fiction already enforces biblio-only pre-1801,
so these texts are all confirmed Fiction — the task here is subgenre
precision, not Fiction/Not-Fiction. genre_raw only is written; top-level
`genre` is left alone.

Prompt uses ESTC metadata when available:
  title = estc_title + ": " + estc_title_sub (fall back to native title)
  + estc_subject_topic (MARC 650$a) passed as Subject
  + estc_form (MARC 655$a) passed as Form

Model: ollama/gemma4:31b (chosen for subgenre precision; ~10s/call local).
Source: llm:gemma4-31b
Run ID: 2026-04-arc-fiction-pre1800-genre-raw-gemma4-31b

Canon_fiction texts (~55: Longus, Apuleius, Chaucer, etc.) have no ESTC
data; they fall back to native title + author and the model will know
them by classical/medieval recognition.
"""

import argparse
import logging
import sys
import time

import lltk
import pandas as pd

from largeliterarymodels.tasks.classify_genre import GenreTask, format_text_for_classification
from largeliterarymodels.integrations.lltk import write_task_to_lltk
from lltk.tools import annotations as A

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('pilot')

# Resolved at runtime from --model flag (see argparse below).
SOURCE = None
RUN_ID = None
MODEL = None
MODEL_CACHE_KEY = None
AUDIT_CSV = None

# Override GenreTask's system prompt: these 666 texts are biblio-confirmed
# Fiction, so we want the model to stop re-adjudicating Fiction/not-Fiction
# and focus on subgenre precision (genre_raw). Without this override the
# e2b run was leaving genre_raw empty on ~40% of inputs because the model
# was calling genre=Nonfiction/Treatise/Biography/etc.
SUBGENRE_SYSTEM_PROMPT = """You are an expert in early modern English literature (1475-1800) with deep knowledge of print culture, genre conventions, and literary history.

**The texts you are classifying have ALREADY been confirmed as fiction** by scholarly library bibliographies (Mish, Odell, fiction_biblio, ravengarside, chadwyck, ESTC form codes). Do NOT try to re-adjudicate whether they are fiction. Trust the bibliographic attribution.

Your job is to assign a specific **subgenre label** (`genre_raw`) to each text, drawing on the title, subtitle (estc_title_sub), ESTC Subject and Form codes when present, and author.

## Required output conventions

1. **Always set `genre` to exactly `"Fiction"`.** Do not use Nonfiction, Biography, Treatise, Essay, Poetry, Drama, or any other top-level category. These texts are fiction.
2. **Focus your effort on `genre_raw`** — the specific subgenre label. Use `reasoning` to briefly explain your subgenre choice (1-2 sentences).
3. Fill in `author_first_name`, `is_translated`, `translated_from`, `year_estimated` as usual.

## Preferred subgenre vocabulary

The existing `genre_raw` column in this corpus (1,165 already-labeled texts) uses these labels most often — prefer them when they fit, but feel free to invent precise labels (like "Romance, mock-chivalric" or "Criminal biography" or "Hagiography, legendary") when they better describe a work.

Top existing labels, with count:
- Novel (775)
- Novel, epistolary (221)
- Satire (18)
- Imaginary voyage (15)
- Fable (8)
- Novel, sentimental (6)
- Rogue fiction (2)
- Picaresque (2)
- Romance (2)
- Allegory (2)

## Other useful subgenre terms

Novel (amatory, Gothic, anti-Jacobin, Jacobin, sentimental, scandal, historical, epistolary, It-Narrative, picaresque, religious, psychological, roman à clef, heroic romance);
Romance (chivalric, heroic, pastoral, prose, allegorical, historical, didactic, gallant, biblical, oriental, classical, comic, sentimental, frame narrative, legendary, religious, mock-chivalric, satirical);
Tale (prose, fairy, satirical, comic, amorous, allegorical, jestbook, exemplary, misogynist, novella);
Fable (beast epic, allegorical, moral);
Satire (prose); Allegory (moral, religious); Picaresque; Jestbook; Chapbook;
Criminal biography; Rogue fiction; Imaginary voyage; Utopia; Tragical history; Novella;
Letters, fictional; Character writing; Dialogue (comic, satirical); Pamphlet fiction, satirical;
Hagiography (legendary).

For compound or mixed-genre works, use semicolons to list multiple labels (e.g. "Novel, historical; Romance, pastoral").

## Principles

- Base your classification on what is ACTUALLY in the title/metadata, not on author's broader reputation. A specific work's subgenre may differ from the author's typical style.
- If the title is sparse and subgenre is genuinely hard to infer, use a broader label like "Novel" or "Romance, prose" with lower confidence rather than inventing a spurious specific label.
- ESTC Subject and Form codes (when provided) are high-signal — e.g. Form = "Epistolary fiction" confirms "Novel, epistolary".
"""


def load_target_ids(ids_file=None):
    """Return the list of arc_fiction pre-1800 _ids for this run.

    If ids_file is given, read one _id per line from that file and return
    the intersection with this model's source-specific done set (for
    idempotent resume). Otherwise derive candidates from arc.meta as usual.

    Idempotency is scoped PER SOURCE: each model run only skips _ids it has
    itself previously annotated. This lets e2b / e4b / 31b each annotate the
    same underlying 666-ID target set independently for cross-model comparison.
    """
    if ids_file:
        log.info("loading target _ids from file: %s", ids_file)
        with open(ids_file) as f:
            candidate_ids = [line.strip() for line in f if line.strip()]
        log.info("  %d _ids loaded from file", len(candidate_ids))
    else:
        log.info("loading arc_fiction metadata …")
        arc = lltk.load("arc_fiction")
        candidate_ids = [
            f"_{row.corpus}/{idx}"
            for idx, row in arc.meta.iterrows()
            if row.year and row.year < 1801
            and not str(row.genre_raw).strip() and str(row.genre_raw).strip().lower() != 'nan'
        ]
        log.info("  %d arc_fiction pre-1800 candidates (no genre_raw in source metadata)",
                 len(candidate_ids))

    log.info("subtracting _ids already annotated by source='%s' …", SOURCE)
    done_df = lltk.db.query(
        f"SELECT DISTINCT _id FROM lltk.annotations_latest "
        f"WHERE field='genre_raw' AND value != '' AND source='{SOURCE}'"
    )
    done = set(done_df['_id']) if '_id' in done_df.columns and len(done_df) else set()
    log.info("  %d _ids already have a genre_raw annotation from THIS source", len(done))

    target = [i for i in candidate_ids if i not in done]
    log.info("  %d target _ids for this run", len(target))
    return target


def load_metadata_batch(ids):
    """One-round-trip CH query: pull core + ESTC fields for all target ids."""
    log.info("fetching metadata + ESTC fields via JSONExtractString …")
    id_tuple = tuple(ids)
    df = lltk.db.query(f"""
        SELECT
          _id, title, author, year, corpus,
          JSONExtractString(meta, 'estc_title') AS estc_title,
          JSONExtractString(meta, 'estc_title_sub') AS estc_title_sub,
          JSONExtractString(meta, 'estc_subject_topic') AS estc_subject_topic,
          JSONExtractString(meta, 'estc_form') AS estc_form
        FROM lltk.texts FINAL
        WHERE _id IN {id_tuple}
    """)
    log.info("  returned %d rows", len(df))

    # Coverage report
    for col in ['estc_title', 'estc_title_sub', 'estc_subject_topic', 'estc_form']:
        n = (df[col].astype(str).str.strip() != '').sum()
        log.info("    %s populated: %d/%d (%.1f%%)", col, n, len(df), 100*n/len(df))
    log.info("  corpus breakdown:")
    for corpus, n in df['corpus'].value_counts().items():
        log.info("    %s: %d", corpus, n)
    return df


def build_title(row):
    """ESTC-preferred title construction per lltk-claude's spec."""
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


def resolve_model_config(model_tag):
    """Given a short tag like 'e2b' or '31b', set the global run constants."""
    global SOURCE, RUN_ID, MODEL, MODEL_CACHE_KEY, AUDIT_CSV
    if model_tag == 'e2b':
        MODEL = 'ollama/gemma4:e2b'
        MODEL_CACHE_KEY = 'gemma4:e2b'
        SOURCE = 'llm:gemma4-e2b'
    elif model_tag == 'e4b':
        MODEL = 'ollama/gemma4:e4b'
        MODEL_CACHE_KEY = 'gemma4:e4b'
        SOURCE = 'llm:gemma4-e4b'
    elif model_tag == '31b':
        MODEL = 'ollama/gemma4:31b'
        MODEL_CACHE_KEY = 'gemma4:31b'
        SOURCE = 'llm:gemma4-31b'
    elif model_tag == 'qwen3:14b':
        MODEL = 'ollama/qwen3:14b'
        MODEL_CACHE_KEY = 'qwen3:14b'
        SOURCE = 'llm:qwen3-14b'
    elif model_tag == 'mlx-31b':
        MODEL = 'lmstudio/gemma-4-31b-it-mlx'
        MODEL_CACHE_KEY = 'gemma-4-31b-it-mlx'
        SOURCE = 'llm:gemma4-31b-mlx-4bit'
    elif model_tag == 'gguf-31b':
        MODEL = 'lmstudio/gemma-4-31b-it'
        MODEL_CACHE_KEY = 'gemma-4-31b-it-gguf'
        SOURCE = 'llm:gemma4-31b-gguf-q4-specdec'
    elif model_tag == 'qwen3.5-35b-a3b':
        # Routed through LM Studio (port 1234) — Ollama's Parallel:1 regression + thinking-mode
        # mess made it unusable. Template edit {%- set enable_thinking = false %} also required.
        MODEL = 'lmstudio/qwen3.5-35b-a3b'
        MODEL_CACHE_KEY = 'qwen3.5-35b-a3b-lmstudio'
        SOURCE = 'llm:qwen3.5-35b-a3b-q4-lmstudio'
    elif model_tag == 'qwen3.5-27b':
        # Routed through LM Studio (port 1234) — Ollama's Parallel:1 regression tanked it
        MODEL = 'lmstudio/qwen3.5-27b'
        MODEL_CACHE_KEY = 'qwen3.5-27b-lmstudio'
        SOURCE = 'llm:qwen3.5-27b-q4-lmstudio'
    elif model_tag == 'llama-70b':
        # Meta family for cross-family recognition diversity beyond Gemma + Qwen
        MODEL = 'lmstudio/meta-llama-3.1-70b-instruct'
        MODEL_CACHE_KEY = 'llama-3.1-70b-lmstudio'
        SOURCE = 'llm:llama-3.1-70b-q4-lmstudio'
    elif model_tag == 'sonnet':
        # Frontier adjudication + high-trust validation against local consensus
        MODEL = 'claude-sonnet-4-6'
        MODEL_CACHE_KEY = 'claude-sonnet-4-6'
        SOURCE = 'llm:claude-sonnet-4-6-subgenre'
    elif model_tag == 'opus':
        # Frontier ceiling check — validates whether Sonnet is high-trust enough
        MODEL = 'claude-opus-4-7'
        MODEL_CACHE_KEY = 'claude-opus-4-7'
        SOURCE = 'llm:claude-opus-4-7-subgenre'
    else:
        raise ValueError(f"Unknown model tag {model_tag!r}")
    slug = MODEL_CACHE_KEY.replace(':', '-')
    RUN_ID = f'2026-04-arc-fiction-pre1800-genre-raw-{slug}'
    AUDIT_CSV = f'/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_arc_fiction_pre1800_genre_raw_{slug.replace("-", "_")}.csv'
    log.info("model config: MODEL=%s SOURCE=%s RUN_ID=%s", MODEL, SOURCE, RUN_ID)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='e2b',
                        choices=['e2b', 'e4b', '31b', 'qwen3:14b', 'mlx-31b', 'gguf-31b', 'qwen3.5-35b-a3b', 'qwen3.5-27b', 'llama-70b', 'sonnet', 'opus'],
                        help='Short model tag (default: e2b)')
    parser.add_argument('--ids-file', default=None,
                        help='Optional path to a newline-separated _id list. When given, '
                             'overrides arc.meta-based candidate discovery — use this to '
                             'replay the exact same target set across models.')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='task.map worker count. Set >1 to pipeline Ollama requests. '
                             'Requires ollama serve started with OLLAMA_NUM_PARALLEL>=num_workers.')
    args = parser.parse_args()

    resolve_model_config(args.model)

    t0 = time.time()

    target_ids = load_target_ids(ids_file=args.ids_file)
    if not target_ids:
        log.info("nothing to do — all targets already annotated")
        return

    df = load_metadata_batch(target_ids)
    prompts, metadata_list = build_prompts(df)
    batch_ids = {m['_id'] for m in metadata_list}

    log.info("running GenreTask via %s (num_workers=1, verbose streaming, "
             "subgenre-only system prompt)…", MODEL)
    task = GenreTask()
    # Filter few-shot examples to Fiction-only so they don't contradict the
    # "always set genre=Fiction" override in SUBGENRE_SYSTEM_PROMPT.
    fiction_examples = [(p, r) for p, r in task.examples if r.genre == 'Fiction']
    log.info("using %d Fiction-only few-shot examples (of %d total)",
             len(fiction_examples), len(task.examples))

    t_run = time.time()
    task.map(
        prompts,
        metadata_list=metadata_list,
        model=MODEL,
        num_workers=args.num_workers,
        verbose=True,
        system_prompt=SUBGENRE_SYSTEM_PROMPT,
        examples=fiction_examples,
    )
    log.info("task.map complete in %.1f min", (time.time() - t_run) / 60)

    # task.df stores the FULL model string (with provider prefix), not the
    # cache key. e.g. 'ollama/gemma4:e2b', not 'gemma4:e2b'. So filter on MODEL.
    task_df = task.df
    task_df = task_df[task_df['model'] == MODEL]
    task_df = task_df[task_df['meta__id'].isin(batch_ids)]
    # Dedupe: if a prior run on the same _ids with a different prompt landed
    # in cache (e.g. earlier non-subgenre-prompt run), keep only the most
    # recent entry per _id — that's the one from THIS run.
    task_df = task_df.drop_duplicates(subset=['meta__id'], keep='last')
    log.info("task.df returned %d unique rows matching model=%s + batch ids",
             len(task_df), MODEL)

    log.info("LLM genre_raw distribution:")
    for g, n in task_df['genre_raw'].value_counts().head(30).items():
        if g:
            log.info("  %s: %d", g, n)

    # Write genre_raw + year_estimated + author_first_name to lltk.annotations.
    # year + author power the recognition trust filter (see
    # project_recognition_metric_for_trust_filter.md in Claude memory).
    # Each row tagged with meta identifying prompt variant so future analysts
    # can distinguish prompt versions via content hash (not just a label).
    import hashlib
    A.ensure_schema()
    PROMPT_META = {
        'prompt_variant': 'subgenre_override',
        'sp_sha256_12': hashlib.sha256(SUBGENRE_SYSTEM_PROMPT.encode()).hexdigest()[:12],
        'sp_len': len(SUBGENRE_SYSTEM_PROMPT),
        'n_examples': len(fiction_examples),
    }
    rows = []
    for _, row in task_df.iterrows():
        _id = row['meta__id']
        if not _id or str(_id) in ('nan', 'None'):
            continue
        conf = float(row.get('confidence') or 1.0)
        base = {'_id': str(_id), 'confidence': conf, 'meta': PROMPT_META}

        genre_raw = row.get('genre_raw')
        if genre_raw and str(genre_raw).strip():
            rows.append({**base, 'field': 'genre_raw', 'value': genre_raw})

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
    log.info("wrote %d annotation rows (genre_raw + year + author) to lltk.annotations",
             n_written)

    # Audit CSV
    audit = df.merge(task_df, left_on='_id', right_on='meta__id', how='left')[[
        '_id', 'corpus', 'year', 'author', 'title', 'estc_title', 'estc_title_sub',
        'estc_subject_topic', 'estc_form', 'genre', 'genre_raw', 'confidence', 'reasoning',
    ]].rename(columns={
        'genre': 'llm_genre',
        'genre_raw': 'llm_genre_raw',
        'confidence': 'llm_confidence',
        'reasoning': 'llm_reasoning',
    })
    audit.to_csv(AUDIT_CSV, index=False)
    log.info("wrote audit CSV to %s", AUDIT_CSV)

    log.info("pilot complete in %.1f min total", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
