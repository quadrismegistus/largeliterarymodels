"""Schema introspection → boolean feature matrix.

Given a Pydantic schema class, classify each field as list / bool / enum /
scalar, then unpack a wide DataFrame (rows=passages, cols=field values) into
a boolean feature matrix (rows=passages, cols=feature indicators like
'content.scene_content:courtship').

No per-task adapter code. The schema IS the adapter — the Pydantic Literal
types already declare list vs enum vs bool, and we just read them.
"""

import ast
import json
import typing

import pandas as pd


def classify_schema_fields(model_cls) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (list_fields, bool_fields, enum_fields, other_scalar_fields).

    A field with annotation `list[Literal[...]]` → list_fields.
    A field with annotation `bool` → bool_fields.
    A field with annotation `Literal[...]` → enum_fields (scalar enum).
    Everything else (str, int, float, ...) → other_scalar_fields. These
    are not included in the feature matrix (confidence, notes, etc.).
    """
    lists, bools, enums, others = [], [], [], []
    for name, fld in model_cls.model_fields.items():
        t = fld.annotation
        origin = typing.get_origin(t)
        if origin in (list, typing.List):
            lists.append(name)
        elif t is bool:
            bools.append(name)
        elif origin is typing.Literal:
            enums.append(name)
        else:
            others.append(name)
    return lists, bools, enums, others


def _coerce_to_list(cell):
    """Parse a cell that may be a Python-repr list, a JSON array, or already
    a Python list."""
    if isinstance(cell, list):
        return cell
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else []
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def wide_to_features(wide_df: pd.DataFrame, model_cls,
                     prefix: str = '') -> pd.DataFrame:
    """Convert a wide annotation DataFrame to a boolean feature matrix.

    Args:
        wide_df: rows = passages, cols = schema fields. Must be indexed by
            whatever you want the feature matrix indexed by (typically
            ('_id', 'scheme', 'seq'))
        model_cls: the Pydantic schema class (not instance). Used to classify
            fields by type.
        prefix: prepend to every output column ('content.' or 'form.' etc.)

    Returns:
        Boolean DataFrame with the same index as `wide_df`. Columns:
        - list fields → '{prefix}{field}:{value}' per distinct value seen
        - enum fields → '{prefix}{field}:{value}' per distinct value seen
        - bool fields → '{prefix}{field}'
        - scalar fields (str, int, float) → NOT included
    """
    lists, bools, enums, _others = classify_schema_fields(model_cls)
    feats: dict[str, pd.Series] = {}

    for f in lists:
        if f not in wide_df.columns:
            continue
        parsed = wide_df[f].apply(_coerce_to_list)
        distinct = sorted({v for lst in parsed for v in lst})
        for v in distinct:
            feats[f'{prefix}{f}:{v}'] = parsed.apply(
                lambda lst, v=v: v in lst)

    for f in enums:
        if f not in wide_df.columns:
            continue
        distinct = sorted(v for v in wide_df[f].dropna().unique() if v != '')
        for v in distinct:
            feats[f'{prefix}{f}:{v}'] = (wide_df[f] == v)

    for f in bools:
        if f not in wide_df.columns:
            continue
        col = wide_df[f]
        # CH read path already coerces 'true'/'false' → bool. Accept either.
        feats[f'{prefix}{f}'] = col.apply(
            lambda x: x is True or (isinstance(x, str) and x.lower() == 'true'))

    out = pd.DataFrame(feats, index=wide_df.index).fillna(False).astype(bool)
    return out
