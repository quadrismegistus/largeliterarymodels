"""Pretty-printing for Pydantic-model task results.

Field-type dispatched so it works for any task without per-task hand-coding:
- list fields  → printed inline
- bool fields  → aggregated (TRUE: a, b, c / (False: d, e))
- scalar enums, strs, numbers → key: value per line
"""

import typing


def _classify_fields(model_cls):
    lists, bools, scalars = [], [], []
    for name, fld in model_cls.model_fields.items():
        t = fld.annotation
        origin = typing.get_origin(t)
        if origin in (list, typing.List):
            lists.append(name)
        elif t is bool:
            bools.append(name)
        else:
            scalars.append(name)
    return lists, bools, scalars


def pretty_print(result, header: str) -> None:
    print(f"\n{'='*70}\n{header}\n{'='*70}", flush=True)
    d = result.model_dump()
    lists, bools, scalars = _classify_fields(type(result))

    for k in lists:
        v = d.get(k)
        if v:
            print(f"  {k}: {v}", flush=True)

    true_b = [k for k in bools if d.get(k) is True]
    false_b = [k for k in bools if d.get(k) is False]
    if true_b:
        print(f"  TRUE: {', '.join(true_b)}", flush=True)
    if false_b:
        print(f"  (False: {', '.join(false_b)})", flush=True)

    for k in scalars:
        v = d.get(k)
        if v in (None, ''):
            continue
        print(f"  {k}: {v}", flush=True)


# Fields treated as long-form text in compare view — shown as blocks
# below the grid rather than in the side-by-side table.
LONG_TEXT_FIELDS = {'notes', 'passage_summary', 'summary', 'reasoning'}


def compare_print(results_by_model: dict, header: str) -> None:
    """Side-by-side comparison for one record across N models.

    `results_by_model` maps model tag → Pydantic result (or None for failure).
    Rows with disagreement across non-None results are marked with `*`.
    Long-text fields (notes, passage_summary, ...) are printed as blocks
    below the grid.
    """
    print(f"\n{'='*78}\n{header}\n{'='*78}", flush=True)

    model_cls = None
    for res in results_by_model.values():
        if res is not None:
            model_cls = type(res)
            break
    if model_cls is None:
        print("  (all models failed)", flush=True)
        return

    lists, bools, scalars = _classify_fields(model_cls)
    dumps = {tag: (r.model_dump() if r is not None else None)
             for tag, r in results_by_model.items()}
    tags = list(results_by_model.keys())

    col_w = max(14, min(30, max(len(t) for t in tags) + 2))
    label_w = 32

    def _fmt(v):
        if v is None:
            return '—'
        if isinstance(v, list):
            return ','.join(str(x) for x in v) if v else '[]'
        if isinstance(v, bool):
            return 'T' if v else 'F'
        s = str(v)
        return s if len(s) <= col_w else s[: col_w - 1] + '…'

    def _row(label, values, mark=' '):
        cells = [_fmt(v).ljust(col_w) for v in values]
        print(f"{mark} {label.ljust(label_w)}  {'  '.join(cells)}", flush=True)

    def _disagree(values):
        vals = [v for v in values if v is not None]
        if len(vals) < 2:
            return False
        if isinstance(vals[0], list):
            return len({frozenset(v) for v in vals}) > 1
        return len(set(vals)) > 1

    header_cells = [t.ljust(col_w) for t in tags]
    print(f"  {'field'.ljust(label_w)}  {'  '.join(header_cells)}", flush=True)
    print(f"  {'-' * label_w}  {'  '.join(['-' * col_w for _ in tags])}",
          flush=True)

    grid_fields = [f for f in lists + bools + scalars
                   if f not in LONG_TEXT_FIELDS]
    for field in grid_fields:
        vals = [dumps[t][field] if dumps[t] else None for t in tags]
        mark = '*' if _disagree(vals) else ' '
        _row(field, vals, mark)

    long_fields = [f for f in model_cls.model_fields if f in LONG_TEXT_FIELDS]
    for field in long_fields:
        any_value = any(dumps[t] and dumps[t].get(field) for t in tags)
        if not any_value:
            continue
        print(f"\n  --- {field} ---", flush=True)
        for t in tags:
            v = dumps[t].get(field) if dumps[t] else None
            print(f"    [{t}] {v if v else '(none)'}", flush=True)


def header_for(record: dict) -> str:
    """Best-effort one-line identifier for a record."""
    parts = []
    if '_id' in record:
        parts.append(str(record['_id']))
    if 'seq' in record:
        parts.append(f"seq={record['seq']}")
    title = record.get('title')
    year = record.get('year')
    if title or year:
        parts.append(f"({title or '?'}, {year or '?'})")
    return '  '.join(parts) if parts else repr(record)[:80]
