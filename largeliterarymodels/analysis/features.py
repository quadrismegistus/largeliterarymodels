"""Mixed-type feature matrix builder for variance partitioning.

Extends the boolean-only `wide_to_features` adapter to produce DataFrames
with bool + ordinal + continuous columns, plus support for external features
(genre tags, period bins, abstractness scores) passed as a side DataFrame.

Returns (features_df, groups_dict) so downstream variance partitioning knows
which columns belong to which semantic group.

Usage:
    from largeliterarymodels.analysis import build_feature_matrix, fit_partition_model

    X, groups = build_feature_matrix(
        tasks=['passage-content', 'passage-form'],
        task_versions={'passage-content': 3, 'passage-form': 2},
        source_agents={'passage-form': 'ensemble-maj4-trust60',
                       'passage-content': 'qwen3.5-35b-a3b'},
        extras=external_df,  # must be indexed by (_id, scheme, seq)
        extras_groups={'genre': ['genre_novel', ...], 'period': ['hc_1600', ...]},
    )

    partition = fit_partition_model(Y=embeddings, X=X, groups=groups)
"""

from __future__ import annotations

import typing
from typing import Iterable, Optional

import pandas as pd

from .adapters import _coerce_to_list, classify_schema_fields
from .reader import load_task_annotations
from .registry import prefix_for, resolve_task_class


# Known ordinal encodings for PassageFormTask fields.
# Bumped to here from the task definition since Pydantic Literal doesn't
# carry order metadata.
DEFAULT_ORDINAL_ENCODINGS: dict[str, dict[str, int]] = {
    'physicality_level': {'absent': 0, 'incidental': 1, 'sustained': 2},
    'narrate_vs_describe': {'narrate': 0, 'balanced': 1, 'describe': 2},
    'story_time_span': {
        '5m': 0, '1h': 1, '6h': 2, '1d': 3, '3d': 4, '1w': 5,
        '1mo': 6, '3mo': 7, '1y': 8, 'years': 9,
    },
    'distance_traveled': {
        '0': 0, '10m': 1, '100m': 2, '1km': 3, '10km': 4,
        '100km': 5, '1000km+': 6,
    },
}


def _expand_task_features(
    wide: pd.DataFrame,
    schema,
    prefix: str,
    ordinal_encodings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Expand a task's wide annotations into a mixed-type feature frame.

    - bool fields → 0/1 int columns
    - list[Literal] fields → binary indicator per enum value
    - Literal (scalar enum): ordinal encoding if registered, else dummies
    - other scalars (str/float like notes/confidence): skipped
    """
    lists, bools, enums, _others = classify_schema_fields(schema)
    cols: dict[str, pd.Series] = {}

    for f in bools:
        if f not in wide.columns:
            continue
        col = wide[f]
        cols[f'{prefix}{f}'] = col.apply(
            lambda x: 1 if (x is True or (isinstance(x, str) and x.lower() == 'true'))
            else 0
        ).astype('int8')

    for f in lists:
        if f not in wide.columns:
            continue
        parsed = wide[f].apply(_coerce_to_list)
        distinct = sorted({v for lst in parsed for v in lst})
        for v in distinct:
            cols[f'{prefix}{f}__{v}'] = parsed.apply(
                lambda lst, v=v: 1 if v in lst else 0
            ).astype('int8')

    for f in enums:
        if f not in wide.columns:
            continue
        enc = ordinal_encodings.get(f)
        if enc is not None:
            # Ordinal: numeric column, -1 for unknown/empty
            cols[f'{prefix}{f}'] = wide[f].apply(
                lambda v: enc.get(str(v), -1) if v is not None and str(v) != '' else -1
            ).astype('int8')
        else:
            # Nominal: dummy-encode
            distinct = sorted(v for v in wide[f].dropna().unique() if v != '')
            for v in distinct:
                cols[f'{prefix}{f}__{v}'] = (wide[f] == v).astype('int8')

    return pd.DataFrame(cols, index=wide.index)


def build_feature_matrix(
    tasks: list[str],
    *,
    task_versions: Optional[dict[str, int]] = None,
    source_agents: Optional[dict[str, str]] = None,
    is_prose_fiction: bool = True,
    ids: Optional[Iterable[str]] = None,
    extras: Optional[pd.DataFrame] = None,
    extras_groups: Optional[dict[str, list[str]]] = None,
    ordinal_encodings: Optional[dict[str, dict[str, int]]] = None,
    client=None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build a mixed-type feature matrix across N tasks + optional extras.

    For each task:
      1. Read wide annotations from CH (via load_task_annotations).
      2. Optionally filter to is_prose_fiction=True.
      3. Expand bools to 0/1, lists to binary indicators, ordinals to numeric,
         nominal enums to dummies. Free-text / confidence fields dropped.
      4. Prefix columns by task slug (e.g. 'content.', 'form.').

    Extras: pass a DataFrame indexed by (_id, scheme, seq) with additional
    features (genre tags, period dummies, abstractness scores). Declare
    group membership via extras_groups={'genre': [cols], 'period': [cols]}.

    Args:
        tasks: CH task names to include
        task_versions: {task_name: int} — None = latest
        source_agents: {task_name: str} — single agent per task
        is_prose_fiction: filter to is_prose_fiction=True per task
        ids: restrict to these text _ids
        extras: external features DataFrame, indexed by (_id, scheme, seq)
        extras_groups: {group_name: [cols]} — declares which extras cols
            belong to which group for variance partitioning
        ordinal_encodings: override DEFAULT_ORDINAL_ENCODINGS
        client: CH client

    Returns:
        (features_df, groups_dict) where features_df is indexed by
        (_id, scheme, seq) and groups_dict maps group name → list of cols.
        Task groups use the task slug ('content', 'form'); extras groups
        use the names passed in extras_groups.
    """
    task_versions = task_versions or {}
    source_agents = source_agents or {}
    ordinal_encodings = {**DEFAULT_ORDINAL_ENCODINGS, **(ordinal_encodings or {})}

    frames: list[pd.DataFrame] = []
    groups: dict[str, list[str]] = {}

    for task_name in tasks:
        task_class = resolve_task_class(task_name)
        wide = load_task_annotations(
            task_name,
            task_version=task_versions.get(task_name),
            source_agent=source_agents.get(task_name),
            ids=list(ids) if ids else None,
            client=client,
        )
        if wide.empty:
            raise ValueError(
                f"No CH rows for task={task_name!r}, "
                f"version={task_versions.get(task_name)}, "
                f"agent={source_agents.get(task_name)}"
            )

        if is_prose_fiction and 'is_prose_fiction' in wide.columns:
            n_before = len(wide)
            wide = wide[wide['is_prose_fiction'] == True]
            n_after = len(wide)
            if n_after < n_before:
                print(
                    f"[{task_name}] filtered {n_before - n_after} non-prose "
                    f"passages ({n_after} kept)", flush=True,
                )

        wide = wide.set_index(['_id', 'scheme', 'seq'])
        slug = prefix_for(task_name).rstrip('.')
        feats = _expand_task_features(
            wide, task_class.schema, prefix=f'{slug}.', ordinal_encodings=ordinal_encodings,
        )
        frames.append(feats)
        groups[slug] = list(feats.columns)

    # Inner-join across tasks
    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, how='inner')

    # Extras
    if extras is not None:
        result = result.join(extras, how='inner')
        if extras_groups:
            for gname, cols in extras_groups.items():
                # Keep only cols that actually landed in the joined frame
                keep = [c for c in cols if c in result.columns]
                if keep:
                    groups[gname] = keep

    return result, groups


def fit_partition_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    groups: dict[str, list[str]],
    *,
    standardize: bool = True,
    pca_components: Optional[int] = None,
) -> pd.DataFrame:
    """Variance partitioning for multivariate Y via group-wise R².

    For each group G in `groups`:
      - marginal_r2: R²(Y ~ X[G])
      - unique_r2:   full_r2 - R²(Y ~ X[all but G])

    Returns a DataFrame indexed by group name with columns:
      n_features, marginal_r2, unique_r2

    Plus .attrs['full_r2'] on the returned frame (full joint R²).

    Args:
        Y: response matrix (e.g. embeddings), indexed to match X
        X: full feature matrix with all group columns present
        groups: {group_name: [cols]} — partition of X's columns
        standardize: z-score features before fitting (recommended for mixed scales)
        pca_components: if set, reduce Y to this many PCs first (recommended
            for high-dim Y like 1024-d embeddings)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Align Y and X by index
    common = X.index.intersection(Y.index)
    if len(common) < len(X) or len(common) < len(Y):
        print(f"[partition] aligning on {len(common)} common rows "
              f"(X={len(X)}, Y={len(Y)})", flush=True)
    X = X.loc[common]
    Y = Y.loc[common]

    if pca_components is not None and Y.shape[1] > pca_components:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        Y_arr = pca.fit_transform(Y.values)
        print(f"[partition] reduced Y from {Y.shape[1]} to {pca_components} PCs "
              f"(explaining {pca.explained_variance_ratio_.sum():.1%} of variance)",
              flush=True)
    else:
        Y_arr = Y.values

    Y_centered = Y_arr - Y_arr.mean(axis=0)
    ss_total = (Y_centered ** 2).sum()

    def r2(X_sub: np.ndarray) -> float:
        if X_sub.shape[1] == 0:
            return 0.0
        if standardize:
            X_sub = StandardScaler().fit_transform(X_sub)
        Y_hat = LinearRegression().fit(X_sub, Y_arr).predict(X_sub)
        Y_hat_centered = Y_hat - Y_arr.mean(axis=0)
        ss_explained = (Y_hat_centered ** 2).sum()
        return float(ss_explained / ss_total)

    all_cols = [c for cols in groups.values() for c in cols]
    full_r2 = r2(X[all_cols].values.astype(float))

    rows = []
    for gname, cols in groups.items():
        other_cols = [c for c in all_cols if c not in cols]
        marginal = r2(X[cols].values.astype(float)) if cols else 0.0
        without = r2(X[other_cols].values.astype(float)) if other_cols else 0.0
        unique = full_r2 - without
        rows.append({
            'group': gname,
            'n_features': len(cols),
            'marginal_r2': marginal,
            'unique_r2': unique,
        })

    out = pd.DataFrame(rows).set_index('group')
    out.attrs['full_r2'] = full_r2
    out.attrs['n_rows'] = len(common)
    return out


def period_dummies(years: pd.Series, breaks: Optional[list[int]] = None) -> pd.DataFrame:
    """Turn a year series into a DataFrame of period dummies.

    Default breaks: [1600, 1650, 1700, 1750, 1800, 1850, 1900].
    Dummy columns are named like 'period_1700_1749'.
    """
    breaks = breaks or [1600, 1650, 1700, 1750, 1800, 1850, 1900]
    out = {}
    for i in range(len(breaks) - 1):
        lo, hi = breaks[i], breaks[i + 1]
        out[f'period_{lo}_{hi - 1}'] = (
            (years >= lo) & (years < hi)
        ).astype('int8')
    return pd.DataFrame(out, index=years.index)


def load_genre_extras(
    ids: Iterable[str],
    *,
    recognized_only: bool = True,
    client=None,
) -> pd.DataFrame:
    """Load genre tags as binary indicators indexed by _id (NOT passage).

    Downstream: reindex to passages via a text→passage map.
    Columns are named like 'genre_novel', 'genre_epistolary'.

    Args:
        ids: text _ids to include
        recognized_only: filter to text_genre_tags.recognized=1 if column exists
        client: CH client
    """
    if client is None:
        import lltk
        client = lltk.db.client

    ids_list = list(set(ids))
    if not ids_list:
        return pd.DataFrame()

    escaped = ', '.join("'" + i.replace("'", "''") + "'" for i in ids_list)

    # Check if recognized column exists
    try:
        schema_df = client.query_df(
            "SELECT name FROM system.columns "
            "WHERE database='lltk' AND table='text_genre_tags'"
        )
        has_recognized = 'recognized' in schema_df['name'].values
    except Exception:
        has_recognized = False

    where = f"WHERE _id IN ({escaped})"
    if recognized_only and has_recognized:
        where += " AND recognized = 1"

    df = client.query_df(
        f"SELECT DISTINCT _id, tag FROM lltk.text_genre_tags {where}"
    )
    if df.empty:
        return pd.DataFrame()

    df = df.drop_duplicates(['_id', 'tag'])
    df['present'] = 1
    wide = df.pivot(index='_id', columns='tag', values='present').fillna(0)
    wide.columns = [f'genre_{c}' for c in wide.columns]
    return wide.astype('int8')
