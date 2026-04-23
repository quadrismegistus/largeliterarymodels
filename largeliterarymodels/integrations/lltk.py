"""Write Task results to lltk's ClickHouse annotations table.

Optional integration — requires lltk[clickhouse] and a configured lltk
ClickHouse connection. Not imported at the package top level.

Usage:

    from largeliterarymodels.integrations.lltk import write_task_to_lltk

    task = GenreTask()
    task.map(prompts, metadata_list=[{'_id': tid} for tid in text_ids])
    write_task_to_lltk(
        task,
        source='llm:claude-opus-4-7',
        field_map={
            'genre': 'genre',
            'genre_raw': 'genre_raw',
            'is_translated': 'is_translated',
            'translated_from': 'original_lang',
            'year_estimated': 'year_estimated',
            'author_first_name': 'author_first_name',
        },
        run_id='2026-04-estc-dogfood',
    )
"""

import logging
import pandas as pd
from typing import Optional, Mapping

log = logging.getLogger(__name__)

ID_COL = 'meta__id'


def write_task_to_lltk(
    task,
    source: str,
    field_map: Mapping[str, str],
    run_id: Optional[str] = None,
    confidence_col: str = 'confidence',
    skip_zero_ints: bool = True,
    only_ids=None,
    only_model: Optional[str] = None,
) -> int:
    """Write a Task's cached results to lltk.annotations.

    Pulls rows from `task.df`, which requires that `task.run(...)` or
    `task.map(...)` was called with `metadata={'_id': text_id}` so the
    `meta__id` column is populated. The task's top-level `confidence`
    column is duplicated across every emitted field row (acceptable when
    the task emits a single global confidence; revisit if schemas grow
    per-field confidence).

    Args:
        task: A Task instance whose results are in its HashStash.
        source: Annotation source label, e.g. 'llm:claude-opus-4-7'. If
            not registered, auto-registers at lltk's DEFAULT_LLM_PRIORITY.
        field_map: {task_df_column: lltk_field_name}. Only listed fields
            are written. Use this to rename (e.g. task's 'translated_from'
            → lltk's 'original_lang').
        run_id: Optional run tag. If None, lltk generates one.
        confidence_col: Column in task.df holding the global confidence
            score (default 'confidence'). If missing, defaults to 1.0.
        skip_zero_ints: If True, rows with int value 0 are skipped for
            int-typed lltk fields. Matches GenreTask's `year_estimated=0
            means unknown` convention. Set False if 0 is a meaningful
            value in your schema.
        only_ids: Optional iterable of _ids to restrict the write to.
            task.df may contain historical cached rows from prior runs;
            pass the _ids of the current batch to avoid writing those.
        only_model: Optional model name (e.g. 'claude-sonnet-4-6') to
            restrict the write to rows cached with that model. Useful when
            the task cache holds results from multiple models.

    Returns:
        Number of rows written to lltk.annotations.
    """
    from lltk.tools import annotations as A

    df = task.df
    if df.empty:
        log.warning("task.df is empty — nothing to write")
        return 0

    if ID_COL not in df.columns:
        raise ValueError(
            f"task.df missing {ID_COL!r} column — call task.run/map with "
            "metadata={'_id': text_id} so the id propagates into the cache key"
        )

    if only_model is not None:
        if 'model' not in df.columns:
            log.warning("only_model=%r requested but task.df has no 'model' column", only_model)
        else:
            before = len(df)
            df = df[df['model'] == only_model]
            log.info("filtered task.df from %d → %d rows matching model=%r",
                     before, len(df), only_model)

    if only_ids is not None:
        only_ids = set(only_ids)
        before = len(df)
        df = df[df[ID_COL].isin(only_ids)]
        log.info("filtered task.df from %d → %d rows matching only_ids",
                 before, len(df))
        if df.empty:
            log.warning("no rows in task.df match only_ids — nothing to write")
            return 0

    rows = []
    normalize_losses = 0

    for _, row in df.iterrows():
        _id = row[ID_COL]
        if pd.isna(_id) or not _id:
            continue

        confidence = row.get(confidence_col, 1.0)
        if pd.isna(confidence):
            confidence = 1.0

        for task_col, lltk_field in field_map.items():
            if task_col not in df.columns:
                log.warning("field_map references missing column %r", task_col)
                continue

            value = row[task_col]
            if _is_skip(value, lltk_field, skip_zero_ints=skip_zero_ints):
                continue

            if _will_silently_lose(value, lltk_field):
                normalize_losses += 1
                log.warning(
                    "normalize→None on %s: _id=%r field=%r raw=%r "
                    "(will be stored as empty string)",
                    source, _id, lltk_field, value,
                )

            rows.append({
                '_id': str(_id),
                'field': lltk_field,
                'value': value,
                'confidence': float(confidence),
            })

    if not rows:
        log.warning("no writable rows after filtering")
        return 0

    written = A.write(source=source, rows=rows, run_id=run_id)

    log.info(
        "wrote %d rows to lltk.annotations (source=%s, run_id=%s, "
        "normalize_losses=%d)",
        written, source, run_id, normalize_losses,
    )
    return written


def _is_skip(value, lltk_field: str, skip_zero_ints: bool) -> bool:
    """True if this value should be omitted from the write batch.

    Skip rules (applied before lltk's own validation):
    - None / NaN
    - Empty string (treat '' as 'unknown', not 'empty string')
    - Int 0 for int-typed fields when skip_zero_ints is True
    """
    try:
        if value is None or pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(value, str) and value.strip() == '':
        return True

    if skip_zero_ints and isinstance(value, (int,)) and not isinstance(value, bool):
        from lltk.tools import annotations as A
        spec = A.field_spec(lltk_field)
        if spec is not None and spec.get('type') == 'int' and value == 0:
            return True

    return False


def _will_silently_lose(value, lltk_field: str) -> bool:
    """True if lltk's normalize hook would return None and silently drop info.

    Used only for warning; the write still proceeds and stores ''.
    """
    from lltk.tools import annotations as A
    spec = A.field_spec(lltk_field)
    if spec is None:
        return False
    norm = spec.get('normalize')
    if norm is None:
        return False
    try:
        return norm(value) is None
    except Exception:
        return False


def compute_recognition_flags(
    sources,
    target_ids=None,
    year_tol: int = 10,
):
    """Compute "did the model recognize this specific work" flags.

    For each (_id, source) in the annotations table, join against
    lltk.texts biblio ground truth and flag whether the model's emitted
    year_estimated is within year_tol years of biblio year AND
    author_first_name matches the biblio author's first name (case-
    insensitive substring match either direction).

    This is the trust filter from the 2026-04 arc_fiction pilot:
    self-reported LLM confidence is miscalibrated; year+author cross-
    check against biblio is a much stronger "the model actually knows
    this text" signal.

    Args:
        sources: Iterable of source labels (e.g. ['llm:gemma4-31b-gguf-q4-specdec']).
            Only rows matching these sources are evaluated.
        target_ids: Optional iterable of _ids to restrict the evaluation to.
            If None, all _ids present in the sources are evaluated.
        year_tol: Year tolerance for recognition (±years from biblio year).
            Default 10 — generous enough for early-modern dating uncertainty.

    Returns:
        pd.DataFrame with columns:
          _id, source, genre_raw, year_est, year_biblio, year_match,
          author_est, author_biblio, author_match, recognized
        One row per (_id, source) that has at least one of the required
        fields written.
    """
    import lltk
    import pandas as pd

    sources = list(sources)
    src_sql = ",".join(f"'{s}'" for s in sources)
    id_clause = ""
    if target_ids is not None:
        target_ids = list(target_ids)
        ids_sql = ",".join(f"'{i}'" for i in target_ids)
        id_clause = f"AND _id IN ({ids_sql})"

    # Pull latest value per (_id, source, field) for the relevant fields
    ann = lltk.db.query(f"""
        SELECT _id, source, field, argMax(value, annotated_at) AS value
        FROM lltk.annotations
        WHERE source IN ({src_sql})
          AND field IN ('genre_raw', 'year_estimated', 'author_first_name')
          {id_clause}
        GROUP BY _id, source, field
    """)
    if not len(ann):
        return pd.DataFrame()

    # Pivot to one row per (_id, source) with the 3 fields as columns
    wide = ann.pivot_table(
        index=['_id', 'source'], columns='field', values='value', aggfunc='first'
    ).reset_index()

    # Pull biblio ground truth for the _ids we have
    ids_in_ann = tuple(wide['_id'].unique().tolist())
    biblio = lltk.db.query(f"""
        SELECT _id, author AS biblio_author, year AS biblio_year_raw
        FROM lltk.texts FINAL
        WHERE _id IN {ids_in_ann}
    """)
    biblio['year_biblio'] = pd.to_numeric(biblio['biblio_year_raw'], errors='coerce')

    def _first_name(s: str) -> str:
        if not s:
            return ''
        s = str(s).strip()
        if ',' in s:
            rest = s.split(',', 1)[1].strip()
            first = rest.split()[0] if rest else ''
            return first.lower().rstrip('.').rstrip(',')
        return ''
    biblio['author_biblio'] = biblio['biblio_author'].apply(_first_name)

    merged = wide.merge(biblio[['_id', 'year_biblio', 'author_biblio']], on='_id', how='left')

    # year matching
    merged['year_est'] = pd.to_numeric(merged.get('year_estimated'), errors='coerce')
    merged['year_match'] = (
        merged['year_est'].notna()
        & merged['year_biblio'].notna()
        & ((merged['year_est'] - merged['year_biblio']).abs() <= year_tol)
    )

    # author matching (case-insensitive substring, either direction)
    merged['author_est'] = merged.get('author_first_name', '').fillna('').astype(str).str.lower().str.strip().str.rstrip('.').str.rstrip(',')

    def _author_match(row):
        ae, ab = row['author_est'], row['author_biblio']
        if not ae or not ab:
            return False
        return ae == ab or ae in ab or ab in ae
    merged['author_match'] = merged.apply(_author_match, axis=1)

    merged['recognized'] = merged['year_match'] & merged['author_match']

    cols = ['_id', 'source', 'genre_raw', 'year_est', 'year_biblio',
            'year_match', 'author_est', 'author_biblio', 'author_match', 'recognized']
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].sort_values(['_id', 'source']).reset_index(drop=True)
