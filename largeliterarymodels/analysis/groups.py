"""Passage-shape group-matrix builder.

Given a passage-indexed DataFrame (rows = (_id, scheme, seq)), build a
boolean group matrix where columns are tag / tag-pair / half-century /
halfcent×tag membership. Filters groups below min_group_n.

The fully-generic `lltk.analysis.stats.group_matrix` will handle the
underlying math once it ships; this helper wraps passage-specific group
definitions (text genre tags from lltk, text years from lltk, half-century
bucketing) on top.
"""

from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _halfcent(year) -> Optional[str]:
    if year is None or pd.isna(year):
        return None
    lo = (int(year) // 50) * 50
    return f'{lo}-{lo + 49}'


def _load_text_tags(text_ids: list[str]) -> dict[str, set[str]]:
    """Return {_id: set-of-tags} via lltk.text_genre_tags."""
    if not text_ids:
        return {}
    import lltk
    id_list = ",".join(f"'{i}'" for i in text_ids)
    df = lltk.db.query(
        f"SELECT _id, tag FROM lltk.text_genre_tags WHERE _id IN ({id_list})"
    )
    return df.groupby('_id')['tag'].apply(set).to_dict()


def _load_text_years(text_ids: list[str]) -> dict[str, Optional[int]]:
    """Return {_id: year} via lltk.texts."""
    if not text_ids:
        return {}
    import lltk
    id_list = ",".join(f"'{i}'" for i in text_ids)
    df = lltk.db.query(
        f"SELECT _id, year FROM lltk.texts FINAL WHERE _id IN ({id_list})"
    )
    return dict(zip(df['_id'], df['year']))


def passage_groups(
    index: pd.MultiIndex,
    *,
    include_tags: bool = True,
    include_pairs: bool = True,
    include_halfcent: bool = True,
    include_halfcent_tag: bool = True,
    min_group_n: int = 30,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a boolean group matrix for passage-shape data.

    Args:
        index: MultiIndex with levels including `_id` (and typically `scheme`, `seq`).
            The `_id` level is used to look up text-level tags + year.
        include_tags: include single-tag groups (romance, gothic, novel, ...)
        include_pairs: include tag-pair groups (epistolary+novel, ...)
        include_halfcent: include half-century groups (1700-1749, ...)
        include_halfcent_tag: include half-century × tag (1700-1749|novel)
        min_group_n: skip groups with fewer passages than this

    Returns:
        (group_matrix, group_kind) where
        - group_matrix: bool DataFrame indexed same as `index`, cols = group labels
        - group_kind: {group_label: 'single'|'pair'|'halfcent'|'halfcent_tag'}
    """
    if '_id' not in index.names:
        raise ValueError("index must have a `_id` level")

    text_ids = sorted({tid for tid in index.get_level_values('_id').unique()})
    tags = _load_text_tags(text_ids) if (include_tags or include_pairs
                                          or include_halfcent_tag) else {}
    years = _load_text_years(text_ids) if (include_halfcent
                                             or include_halfcent_tag) else {}

    # Per-passage attributes
    id_col = index.get_level_values('_id').values
    passage_tags = np.array([tags.get(tid, set()) for tid in id_col], dtype=object)
    passage_hc = np.array([_halfcent(years.get(tid)) for tid in id_col], dtype=object)

    cols: dict[str, np.ndarray] = {}
    kind: dict[str, str] = {}

    if include_tags:
        all_tags = sorted({t for s in tags.values() for t in s})
        for tag in all_tags:
            mask = np.array([tag in s for s in passage_tags])
            if mask.sum() >= min_group_n:
                cols[tag] = mask
                kind[tag] = 'single'

    if include_pairs:
        pair_counts: dict[tuple[str, str], int] = {}
        for ts in passage_tags:
            for a, b in combinations(sorted(ts), 2):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
        for (a, b), n in pair_counts.items():
            if n >= min_group_n:
                label = f'{a}+{b}'
                cols[label] = np.array([(a in s and b in s) for s in passage_tags])
                kind[label] = 'pair'

    if include_halfcent:
        for hc in sorted({h for h in passage_hc if h is not None}):
            mask = np.array([h == hc for h in passage_hc])
            if mask.sum() >= min_group_n:
                cols[hc] = mask
                kind[hc] = 'halfcent'

    if include_halfcent_tag:
        hct: dict[tuple[str, str], int] = {}
        for hc, ts in zip(passage_hc, passage_tags):
            if hc is None:
                continue
            for t in ts:
                hct[(hc, t)] = hct.get((hc, t), 0) + 1
        for (hc, t), n in hct.items():
            if n >= min_group_n:
                label = f'{hc}|{t}'
                cols[label] = np.array(
                    [(ph == hc and t in ts)
                     for ph, ts in zip(passage_hc, passage_tags)])
                kind[label] = 'halfcent_tag'

    return (pd.DataFrame(cols, index=index).astype(bool), kind)
