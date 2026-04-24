"""Passage and text embedding utilities: fetch, pool, center."""

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


def fetch_passage_embeddings(
    ids: list[str],
    *,
    scheme: str = 'p500',
    client=None,
) -> pd.DataFrame:
    """Fetch passage embeddings from lltk.passage_embeddings.

    Returns DataFrame with columns: _id, seq, embedding.
    """
    if client is None:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host='localhost', port=8123, username='lltk', password='lltk',
        )
    return client.query_df(
        "SELECT _id, seq, embedding "
        "FROM lltk.passage_embeddings "
        "WHERE scheme = %(scheme)s AND _id IN %(ids)s",
        parameters={'scheme': scheme, 'ids': list(set(ids))},
    )


def mean_pool_to_text(
    emb_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Mean-pool per-passage embeddings to one L2-normalized vector per text.

    Returns (ids, X) where X is float32 with shape (n_texts, dim).
    """
    vecs: dict[str, list] = defaultdict(list)
    for _, row in emb_df.iterrows():
        vecs[row['_id']].append(row['embedding'])
    ids, X = [], []
    for _id, vs in vecs.items():
        m = np.array(vs, dtype=np.float32).mean(axis=0)
        norm = np.linalg.norm(m)
        if norm > 0:
            m /= norm
        ids.append(_id)
        X.append(m)
    return ids, np.array(X, dtype=np.float32)


def center_by_group(
    X: np.ndarray,
    groups: list,
) -> np.ndarray:
    """Mean-center embeddings within each group and L2-re-normalize.

    Suppresses the group-level signal (e.g. language) so remaining
    variance reflects shared structure.
    """
    X = X.copy()
    for g in set(groups):
        mask = np.array(groups) == g
        if not mask.any():
            continue
        X[mask] -= X[mask].mean(axis=0)
        norms = np.linalg.norm(X[mask], axis=1, keepdims=True)
        norms[norms == 0] = 1
        X[mask] /= norms
    return X
