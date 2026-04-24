"""Cross-language feature-rate comparison from CH passage annotations.

Joins passage annotations with lltk.texts to get language, then computes
per-field rates by (lang, period) for trend comparison.

Caveat: language coverage is corpus-dependent. As of 2026-04, 99.7% of French
content annotations come from gallica_literary_fictions. Trends reflect what
each language's corpus curates as 'literary fiction', not all fiction in that
language. Downstream should note corpus composition when interpreting divergences.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd


def compare_cross_language(
    fields: list[Union[str, tuple[str, str]]],
    *,
    langs: list[str] = ['en', 'fr'],
    period_bins: list[int] = [1600, 1650, 1700, 1750, 1800],
    task_name: str = 'passage-content',
    task_version: Optional[int] = None,
    source_agent: Optional[str] = None,
    source_family: Optional[str] = None,
    prose_only: bool = True,
    client=None,
) -> pd.DataFrame:
    """Compare feature rates across languages and periods.

    Works with two kinds of CH fields:
    - Boolean fields (value is 'true'/'false') — e.g. expanded propagated
      fields like 'threats__social_or_reputational'.
    - List fields (value is a JSON array) — e.g. 'fantastical_elements'
      with values like '["supernatural"]'. For these, pass a tuple
      ('fantastical_elements::supernatural', 'supernatural') where the
      '::' separates field name from the list element to check.

    Args:
        fields: field names to compare. Each entry is one of:
            - 'field_name' — boolean field, used as both CH name and label
            - ('field_name', 'label') — boolean field with display rename
            - ('field_name::element', 'label') — list-contains check
        langs: language codes to include (matched against lltk.texts.lang).
        period_bins: bin edges for half-century grouping.
        task_name: CH task name.
        task_version: restrict to this version (None = latest).
        source_agent: restrict to this agent (None = any).
        source_family: restrict to this family (None = any; use 'derived'
            to include classifier-propagated labels).
        prose_only: if True, exclude passages where is_prose_fiction != 'true'.
        client: CH client override.

    Returns:
        Long-form DataFrame with columns:
            lang, field, period, n, n_true, pct
        where period is a string like '1600-1649'.
    """
    if client is None:
        import lltk
        client = lltk.db.client

    specs: list[dict] = []
    for f in fields:
        if isinstance(f, tuple):
            key, label = f[0], f[1]
        else:
            key, label = f, f
        if '::' in key:
            ch_field, element = key.split('::', 1)
            specs.append({'ch_field': ch_field, 'element': element, 'label': label})
        else:
            specs.append({'ch_field': key, 'element': None, 'label': label})

    ch_fields = sorted(set(s['ch_field'] for s in specs))
    escaped_fields = ', '.join(f"'{f}'" for f in ch_fields)
    escaped_langs = ', '.join(f"'{l}'" for l in langs)

    version_filter = f"AND pa.task_version = {int(task_version)}" if task_version else ""
    agent_filter = f"AND pa.source_agent = '{source_agent}'" if source_agent else ""
    family_filter = f"AND pa.source_family = '{source_family}'" if source_family else ""

    prose_join = ""
    prose_where = ""
    if prose_only:
        prose_join = f"""
        LEFT JOIN (
            SELECT _id, scheme, seq, value as ipf
            FROM llmtasks.passage_annotations_latest
            WHERE task = '{task_name}' AND field = 'is_prose_fiction'
            {version_filter} {agent_filter} {family_filter}
        ) pf ON pa._id = pf._id AND pa.scheme = pf.scheme AND pa.seq = pf.seq
        """
        prose_where = "AND (pf.ipf IS NULL OR pf.ipf = 'true')"

    sql = f"""
    SELECT
        t.lang,
        pa.field,
        t.year,
        pa.value
    FROM llmtasks.passage_annotations_latest pa
    JOIN lltk.texts t ON pa._id = t._id
    {prose_join}
    WHERE pa.task = '{task_name}'
      AND pa.field IN ({escaped_fields})
      AND t.lang IN ({escaped_langs})
      AND t.year > 0
      {version_filter} {agent_filter} {family_filter}
      {prose_where}
    """

    df = client.query_df(sql)
    if df.empty:
        return pd.DataFrame(columns=['lang', 'field', 'period', 'n', 'n_true', 'pct'])

    bins = sorted(period_bins)
    labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins) - 1)]
    df['period'] = pd.cut(
        df['year'], bins=bins, right=False, labels=labels,
    )
    df = df.dropna(subset=['period'])

    import json as _json

    parts = []
    for spec in specs:
        subset = df[df['field'] == spec['ch_field']].copy()
        if subset.empty:
            continue
        if spec['element'] is not None:
            element = spec['element']
            def _contains(val, el=element):
                try:
                    lst = _json.loads(val) if isinstance(val, str) else val
                    return el in lst if isinstance(lst, list) else False
                except (ValueError, TypeError):
                    return False
            subset['is_true'] = subset['value'].apply(_contains)
        else:
            subset['is_true'] = subset['value'].str.lower().isin(['true', '1', 'yes'])
        subset['label'] = spec['label']
        parts.append(subset)

    if not parts:
        return pd.DataFrame(columns=['lang', 'field', 'period', 'n', 'n_true', 'pct'])

    tagged = pd.concat(parts, ignore_index=True)

    grouped = tagged.groupby(['lang', 'label', 'period'], observed=True).agg(
        n=('is_true', 'count'),
        n_true=('is_true', 'sum'),
    ).reset_index()
    grouped['pct'] = (grouped['n_true'] / grouped['n'] * 100).round(1)
    grouped = grouped.rename(columns={'label': 'field'})
    grouped = grouped.sort_values(['field', 'lang', 'period']).reset_index(drop=True)
    return grouped
