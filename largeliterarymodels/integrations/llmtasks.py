"""ClickHouse-backed long-form annotation store for LLM-generated passage
annotations.

Lives in the `llmtasks` ClickHouse database (separate from `lltk.*`).
Uses lltk.db.client for CH access (inherits credentials + HTTP config).

Design (agreed 2026-04-20 with lltk-claude):
- Plain MergeTree + argMax view (not ReplacingMergeTree) for history preservation
- 4-column source split (family / agent / task / task_version) for queryability
- Side-table `task_configs` holds invariant run config keyed by config_sha256
- Per-annotation `meta` blob references config_sha256 + per-call state

See memory: project-level design notes attached to this session.

Usage:
    from largeliterarymodels.integrations import llmtasks

    llmtasks.ensure_schema()  # idempotent

    llmtasks.write_passage_annotations(
        task=my_passage_task,
        source_agent='qwen3.5-35b-a3b',
        task_name='passage-content',
        task_version=1,
        only_model='lmstudio/qwen3.5-35b-a3b',
        run_id='2026-04-20-main-pilot',
    )

    df = llmtasks.read_passage_annotations(
        task_name='passage-content', task_version=1,
        fields=['has_courtship', 'setting_domestic_interior'],
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

log = logging.getLogger(__name__)

# CH database + table names. Centralized so we don't typo across calls.
DB = 'llmtasks'
PASSAGE_TABLE = f'{DB}.passage_annotations'
PASSAGE_LATEST_VIEW = f'{DB}.passage_annotations_latest'
CONFIGS_TABLE = f'{DB}.task_configs'


# ── Schema DDL ───────────────────────────────────────────────────────────

DDL_STATEMENTS = [
    f"CREATE DATABASE IF NOT EXISTS {DB}",

    f"""
    CREATE TABLE IF NOT EXISTS {PASSAGE_TABLE} (
        _id              LowCardinality(String),
        scheme           LowCardinality(String),
        seq              UInt32,
        field            LowCardinality(String),
        value            String,
        source_family    LowCardinality(String),
        source_agent     LowCardinality(String),
        task             LowCardinality(String),
        task_version     UInt16,
        run_id           String,
        annotated_at     DateTime,
        meta             String
    ) ENGINE = MergeTree
    ORDER BY (_id, scheme, seq, field, source_family, source_agent,
              task, task_version, annotated_at)
    """,

    # Latest-per-source view. Preserves V1 and V2 as distinct rows (groups
    # include source_family+agent+task+task_version), so version-drift audits
    # are native. Also emits a canonical colon-joined `source` alias for
    # string-based downstream tooling.
    f"""
    CREATE VIEW IF NOT EXISTS {PASSAGE_LATEST_VIEW} AS
    SELECT
        _id, scheme, seq, field,
        source_family, source_agent, task, task_version,
        concat(source_family, ':', source_agent, ':', task, ':v',
               toString(task_version)) AS source,
        argMax(value, annotated_at) AS value,
        argMax(meta, annotated_at)  AS meta,
        max(annotated_at) AS latest_at
    FROM {PASSAGE_TABLE}
    GROUP BY _id, scheme, seq, field,
             source_family, source_agent, task, task_version
    """,

    # Config registry. ReplacingMergeTree dedups on config_sha256 at merge
    # time (idempotent first-write-wins via first_seen version column).
    f"""
    CREATE TABLE IF NOT EXISTS {CONFIGS_TABLE} (
        config_sha256    String,
        task_class       String,
        source_family    LowCardinality(String),
        source_agent     LowCardinality(String),
        task             LowCardinality(String),
        task_version     UInt16,
        model            String,
        temperature      Float32,
        max_tokens       UInt32,
        system_prompt    String,
        schema_json      String,
        examples_json    String,
        first_seen       DateTime
    ) ENGINE = ReplacingMergeTree(first_seen)
    ORDER BY config_sha256
    """,
]


def ensure_schema(client=None) -> None:
    """Create llmtasks database + tables + view if absent. Idempotent."""
    if client is None:
        import lltk
        client = lltk.db.client
    for stmt in DDL_STATEMENTS:
        client.command(stmt)
    log.info("llmtasks schema ensured (%d DDL statements)", len(DDL_STATEMENTS))


# ── Config hashing ───────────────────────────────────────────────────────

def _canonicalize_examples(examples) -> list:
    """Normalize a few-shot examples list for stable hashing.

    Task.examples is typically list[tuple[str, PydanticModel]]. Coerce the
    Pydantic side to a dict (via model_dump) so changing the class name
    without changing semantics doesn't alter the hash."""
    if not examples:
        return []
    out = []
    for item in examples:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            prompt, output = item
            if hasattr(output, 'model_dump'):
                output = output.model_dump()
            elif hasattr(output, 'dict'):
                output = output.dict()
            out.append([str(prompt), output])
        else:
            out.append(str(item))
    return out


def compute_config_sha256(*, task_class: str, model: str,
                          temperature: float, max_tokens: int,
                          system_prompt: str, schema, examples) -> str:
    """Hash over the invariants of a bulk task.map() run. Any change to any
    field produces a new config_sha256 (and thus a new registry row).

    `schema` can be a Pydantic model class OR a pre-computed JSON-schema dict.
    When given a class, we call model_json_schema() to capture field
    descriptions, defaults, literals — so edits to Field(description=...)
    produce a new hash."""
    if hasattr(schema, 'model_json_schema'):
        schema_dict = schema.model_json_schema()
    elif isinstance(schema, dict):
        schema_dict = schema
    else:
        schema_dict = {'repr': repr(schema)}

    payload = {
        'task_class': task_class,
        'model': model,
        'temperature': float(temperature) if temperature is not None else None,
        'max_tokens': int(max_tokens) if max_tokens is not None else None,
        'system_prompt': system_prompt or '',
        'schema': schema_dict,
        'examples': _canonicalize_examples(examples),
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def register_config(*, task_class: str, source_family: str, source_agent: str,
                    task: str, task_version: int,
                    model: str, temperature: float, max_tokens: int,
                    system_prompt: str, schema, examples,
                    client=None) -> str:
    """Compute config_sha256 and INSERT the row into task_configs (dedup
    via ReplacingMergeTree). Returns the sha256 for use in annotations."""
    if client is None:
        import lltk
        client = lltk.db.client

    config_sha256 = compute_config_sha256(
        task_class=task_class, model=model,
        temperature=temperature, max_tokens=max_tokens,
        system_prompt=system_prompt, schema=schema, examples=examples,
    )
    schema_json = json.dumps(
        schema.model_json_schema() if hasattr(schema, 'model_json_schema')
        else schema,
        sort_keys=True, default=str,
    )
    examples_json = json.dumps(_canonicalize_examples(examples),
                               sort_keys=True, default=str)

    row = [(
        config_sha256, task_class, source_family, source_agent,
        task, int(task_version), model,
        float(temperature) if temperature is not None else 0.0,
        int(max_tokens) if max_tokens is not None else 0,
        system_prompt or '', schema_json, examples_json,
        datetime.now(timezone.utc).replace(tzinfo=None),
    )]
    client.insert(
        CONFIGS_TABLE, row,
        column_names=['config_sha256', 'task_class', 'source_family',
                      'source_agent', 'task', 'task_version', 'model',
                      'temperature', 'max_tokens', 'system_prompt',
                      'schema_json', 'examples_json', 'first_seen'],
    )
    return config_sha256


# ── Writing annotations ──────────────────────────────────────────────────

# Fields on task.df that are NEVER written as annotations (internal / meta).
_RESERVED_COLS = {
    'model', 'prompt', 'system_prompt', 'examples', 'temperature',
    'max_tokens', 'cache_key', 'response', 'error',
    'annotated_at', 'wall_time_ms', 'tokens_in', 'tokens_out',
}


def _serialize_value(v) -> str:
    """Convert a Pydantic-output field value to the String column format."""
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=str)
    if v is None:
        return ''
    return str(v)


def _parse_section_id(section_id: str) -> tuple[str, int]:
    """Parse 'p500:25' → ('p500', 25). Accepts 'scheme:seq' or raw int."""
    if section_id is None or section_id == '':
        return 'unknown', 0
    m = re.match(r'^([a-zA-Z0-9_]+):(\d+)$', str(section_id))
    if m:
        return m.group(1), int(m.group(2))
    try:
        return 'unknown', int(section_id)
    except (TypeError, ValueError):
        return 'unknown', 0


def write_passage_annotations(
    task,
    *,
    source_agent: str,
    task_name: str = 'passage-content',
    task_version: int = 1,
    source_family: str = 'llm',
    run_id: Optional[str] = None,
    only_model: Optional[str] = None,
    only_ids: Optional[Iterable[str]] = None,
    fields: Optional[Sequence[str]] = None,
    client=None,
    dry_run: bool = False,
) -> int:
    """Write a passage-annotation task's cached results to llmtasks.passage_annotations.

    Args:
        task: A Task instance whose PassageContentTask-style results are cached.
            task.df must have meta__id and meta_section_id columns populated
            (the pilot script sets section_id='p500:{seq}').
        source_agent: e.g. 'qwen3.5-35b-a3b' or 'claude-opus-4-7' or 'ryan'
        task_name: e.g. 'passage-content'
        task_version: integer starting at 1; bump when you want v1 and v2
            kept as distinct audit-able rows in the latest view.
        source_family: 'llm' | 'human' | 'derived'
        run_id: free-form run identifier (e.g. '2026-04-20-main-pilot')
        only_model: restrict to rows cached under this model (important when
            task.df has results from multiple models)
        only_ids: restrict to these text _ids
        fields: restrict the written field set (None = all bool + list + str
            Pydantic output fields)
        dry_run: if True, compute everything but skip INSERT

    Returns the number of rows inserted.
    """
    import pandas as pd
    if client is None:
        import lltk
        client = lltk.db.client

    df = task.df
    if df.empty:
        log.warning("task.df is empty — nothing to write")
        return 0

    if only_model is not None and 'model' in df.columns:
        df = df[df['model'] == only_model]
    if only_ids is not None:
        only_ids_set = set(only_ids)
        df = df[df['meta__id'].isin(only_ids_set)]
    if df.empty:
        log.warning("task.df empty after filters")
        return 0

    # task.df may contain multiple cached rows per passage if prompts
    # changed across runs. Keep the most-recent row per (_id, section_id):
    # HashStash appends in insertion order so 'last' is the latest.
    before = len(df)
    df = df.drop_duplicates(
        subset=['meta__id', 'meta_section_id'], keep='last'
    ).reset_index(drop=True)
    if len(df) != before:
        log.info("deduped task.df %d → %d rows by (_id, section_id), keep=last",
                 before, len(df))

    # Determine fields to write: Pydantic schema fields minus reserved meta.
    schema = task.schema
    schema_fields = list(schema.model_fields.keys()) if hasattr(schema, 'model_fields') else []
    write_fields = list(fields) if fields else schema_fields
    write_fields = [f for f in write_fields
                    if f in df.columns and f not in _RESERVED_COLS]

    if not write_fields:
        log.warning("no writable fields found in task.df")
        return 0

    # Register the config (all rows share the same config since only_model
    # filter pins to one model, and system/schema/examples are on the Task
    # class — not per-row).
    config_sha256 = register_config(
        task_class=task.__class__.__name__,
        source_family=source_family, source_agent=source_agent,
        task=task_name, task_version=task_version,
        model=only_model or df['model'].iloc[0] if 'model' in df.columns else 'unknown',
        temperature=getattr(task, 'temperature', 0.2),
        max_tokens=getattr(task, 'max_tokens', 4096),
        system_prompt=getattr(task, 'system_prompt', '') or '',
        schema=schema,
        examples=getattr(task, 'examples', []),
        client=client,
    )
    log.info("registered config_sha256=%s", config_sha256[:12])

    # Build annotation rows.
    rows = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for _, r in df.iterrows():
        _id = r.get('meta__id', '')
        if pd.isna(_id) or not _id:
            continue
        scheme, seq = _parse_section_id(r.get('meta_section_id', ''))
        meta_blob = json.dumps({
            'config_sha256': config_sha256,
            'confidence': r.get('confidence'),
        }, ensure_ascii=False, default=str)

        for field_name in write_fields:
            value = r.get(field_name)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            rows.append((
                str(_id), scheme, int(seq), field_name,
                _serialize_value(value),
                source_family, source_agent, task_name, int(task_version),
                run_id or '', now, meta_blob,
            ))

    if not rows:
        log.warning("no annotation rows produced")
        return 0

    log.info("prepared %d rows from %d passages × %d fields",
             len(rows), df['meta__id'].nunique(), len(write_fields))

    if dry_run:
        log.info("dry_run=True — skipping INSERT")
        return len(rows)

    client.insert(
        PASSAGE_TABLE, rows,
        column_names=['_id', 'scheme', 'seq', 'field', 'value',
                      'source_family', 'source_agent', 'task', 'task_version',
                      'run_id', 'annotated_at', 'meta'],
    )
    log.info("inserted %d rows into %s", len(rows), PASSAGE_TABLE)
    return len(rows)


# ── Reading annotations ──────────────────────────────────────────────────

def read_passage_annotations(
    *,
    ids: Optional[Sequence[str]] = None,
    fields: Optional[Sequence[str]] = None,
    source_agent: Optional[str] = None,
    task_name: Optional[str] = None,
    task_version: Optional[int] = None,
    use_latest_view: bool = True,
    client=None,
):
    """Read passage annotations from CH and pivot long→wide.

    Returns a DataFrame indexed by (_id, scheme, seq) with one column per
    field. Filters (source_agent, task_name, task_version) narrow the slice.

    Args:
        use_latest_view: if True, reads from passage_annotations_latest
            (argMax-resolved values); if False, reads raw passage_annotations.
    """
    import pandas as pd
    if client is None:
        import lltk
        client = lltk.db.client

    table = PASSAGE_LATEST_VIEW if use_latest_view else PASSAGE_TABLE
    where = []
    if ids:
        id_list = ", ".join(f"'{i}'" for i in ids)
        where.append(f"_id IN ({id_list})")
    if fields:
        field_list = ", ".join(f"'{f}'" for f in fields)
        where.append(f"field IN ({field_list})")
    if source_agent:
        where.append(f"source_agent = '{source_agent}'")
    if task_name:
        where.append(f"task = '{task_name}'")
    if task_version is not None:
        where.append(f"task_version = {int(task_version)}")
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""

    sql = (f"SELECT _id, scheme, seq, field, value "
           f"FROM {table} {where_clause}")
    long = client.query_df(sql)
    if long.empty:
        return long

    wide = long.pivot_table(
        index=['_id', 'scheme', 'seq'],
        columns='field', values='value', aggfunc='first',
    ).reset_index()
    # Coerce bool-looking columns back to bool
    for c in wide.columns:
        if c in ('_id', 'scheme', 'seq'):
            continue
        sample = wide[c].dropna().unique()
        if len(sample) > 0 and set(map(str, sample)) <= {'true', 'false'}:
            wide[c] = wide[c].map({'true': True, 'false': False})
    return wide


# ── Prompt reconstruction (for audit) ────────────────────────────────────

def reconstruct_prompt(
    _id: Optional[str] = None,
    scheme: str = 'p500',
    seq: Optional[int] = None,
    config_sha256: Optional[str] = None,
    *,
    annotation_row: Optional[dict] = None,
    client=None,
) -> dict:
    """Reconstruct the exact prompt that produced a given annotation.

    Either pass explicit (_id, scheme, seq, config_sha256), or an
    annotation_row (a dict-like from passage_annotations — e.g. the result
    of read_raw_annotation_row or a row iterated from a query) from which
    those fields are extracted automatically.

    Returns a dict with keys:
        system_prompt    — system-side instructions
        schema_json      — JSON-schema spec for the Pydantic schema
        examples_json    — few-shot examples (may be '[]' for V1)
        user_prompt      — user-side prompt (post-format_passage)
        passage_text     — raw passage text from lltk.passages
        text_meta        — {title, author, year} from lltk.texts.meta
        config           — the full task_configs row as a dict

    Byte-for-byte recoverable provenance: (system_prompt, examples_json,
    schema_json, user_prompt) are exactly what the LLM was called with,
    modulo the _build_extract_prompt wrapping layer (which re-concatenates
    system+schema+examples at runtime).
    """
    import json as _json
    if client is None:
        import lltk
        client = lltk.db.client

    # Unpack from annotation_row if provided
    if annotation_row is not None:
        _id = _id or annotation_row.get('_id')
        scheme = annotation_row.get('scheme', scheme) or scheme
        seq = seq if seq is not None else annotation_row.get('seq')
        if config_sha256 is None:
            meta_raw = annotation_row.get('meta', '{}')
            meta_obj = _json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            config_sha256 = meta_obj.get('config_sha256')

    if not (_id and scheme and seq is not None and config_sha256):
        raise ValueError("Need _id, scheme, seq, and config_sha256 "
                         "(either explicit or via annotation_row)")

    # 1. Passage text from lltk.passages
    p_df = client.query_df(
        f"SELECT text FROM lltk.passages "
        f"WHERE _id = '{_id}' AND scheme = '{scheme}' AND seq = {int(seq)} LIMIT 1"
    )
    if p_df.empty:
        raise LookupError(f"No passage found for ({_id!r}, {scheme!r}, {seq})")
    passage_text = p_df.iloc[0]['text']

    # 2. Text-level metadata from lltk.texts
    t_df = client.query_df(
        f"SELECT meta FROM lltk.texts FINAL WHERE _id = '{_id}' LIMIT 1"
    )
    if t_df.empty:
        text_meta = {'title': None, 'author': None, 'year': None}
    else:
        meta_raw = t_df.iloc[0]['meta']
        m = _json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
        text_meta = {
            'title': m.get('title') or m.get('title_main') or '',
            'author': m.get('author') or '',
            'year': m.get('year') or m.get('year_orig'),
        }

    # 3. Rebuild user prompt via format_passage (same helper the pilot used)
    from largeliterarymodels.tasks.classify_passage import format_passage
    user_prompt, _user_meta = format_passage(
        passage_text,
        title=text_meta['title'], author=text_meta['author'],
        year=text_meta['year'], _id=_id,
        section_id=f"{scheme}:{int(seq)}",
    )

    # 4. System-side config
    c_df = client.query_df(
        f"SELECT * FROM {CONFIGS_TABLE} FINAL "
        f"WHERE config_sha256 = '{config_sha256}' LIMIT 1"
    )
    if c_df.empty:
        raise LookupError(f"No task_configs row for config_sha256={config_sha256!r}")
    cfg = c_df.iloc[0].to_dict()

    return {
        'system_prompt': cfg['system_prompt'],
        'schema_json': cfg['schema_json'],
        'examples_json': cfg['examples_json'],
        'user_prompt': user_prompt,
        'passage_text': passage_text,
        'text_meta': text_meta,
        'config': cfg,
    }
