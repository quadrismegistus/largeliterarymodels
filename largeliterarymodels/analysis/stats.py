"""Statistical primitives for corpus discrimination analysis.

Provides group_matrix, fisher_tests, and bh_fdr. These are generic functions
that operate on boolean DataFrames — no knowledge of tasks, schemas, or CH.

Requires scipy for fisher_tests.
"""

import numpy as np
import pandas as pd


def group_matrix(long_df, *, group_col, member_col, value_col=None):
    """Pivot a long DataFrame into a boolean member x group matrix."""
    if value_col is not None:
        mat = long_df.pivot_table(
            index=member_col, columns=group_col, values=value_col,
            aggfunc='max', fill_value=False,
        )
    else:
        long_df = long_df[[member_col, group_col]].drop_duplicates()
        long_df = long_df.assign(_present=True)
        mat = long_df.pivot_table(
            index=member_col, columns=group_col, values='_present',
            aggfunc='max', fill_value=False,
        )
    mat.columns.name = None
    mat.index.name = member_col
    return mat.astype(bool)


def fisher_tests(
    feature_matrix,
    group_matrix,
    *,
    min_group_n=30,
    min_feature_n=20,
    include_feature_pairs=False,
    cross_task_pairs_only=True,
):
    """Run Fisher exact tests for every (group, feature) pair."""
    from scipy.stats import fisher_exact

    shared_idx = feature_matrix.index.intersection(group_matrix.index)
    feat = feature_matrix.loc[shared_idx]
    grp = group_matrix.loc[shared_idx]

    feat = feat.loc[:, feat.sum() >= min_feature_n]
    grp = grp.loc[:, grp.sum() >= min_group_n]

    feat_arr = feat.values
    grp_arr = grp.values
    feat_cols = list(feat.columns)
    grp_cols = list(grp.columns)

    rows = []

    def _run_tests(g_arr, g_col, f_arr, f_cols):
        for fi, f_col in enumerate(f_cols):
            f_vec = f_arr[:, fi]
            for gi, g_col_name in enumerate(g_col):
                g_vec = g_arr[:, gi]
                a = int((g_vec & f_vec).sum())
                b = int((g_vec & ~f_vec).sum())
                c = int((~g_vec & f_vec).sum())
                d = int((~g_vec & ~f_vec).sum())
                if b == 0 and c == 0:
                    continue
                odds = (a * d) / (b * c) if (b * c) > 0 else float('inf')
                _, p = fisher_exact([[a, b], [c, d]], alternative='two-sided')
                rate_in = a / (a + b) if (a + b) > 0 else float('nan')
                rate_out = c / (c + d) if (c + d) > 0 else float('nan')
                rows.append({
                    'group': g_col_name,
                    'feature': f_col,
                    'a_group_feat': a,
                    'b_group_nofeat': b,
                    'c_nogroup_feat': c,
                    'd_nogroup_nofeat': d,
                    'rate_in_group': rate_in,
                    'rate_not_group': rate_out,
                    'odds_ratio': odds,
                    'p_value': p,
                })

    _run_tests(grp_arr, grp_cols, feat_arr, feat_cols)

    if include_feature_pairs:
        def _task_prefix(col):
            return col.split('.')[0] if '.' in col else col

        for gi, g_col_name in enumerate(feat_cols):
            g_vec = feat_arr[:, gi:gi+1]
            g_prefix = _task_prefix(g_col_name)
            candidate_feats = [
                (fi, fc) for fi, fc in enumerate(feat_cols)
                if fi > gi and (
                    not cross_task_pairs_only
                    or _task_prefix(fc) != g_prefix
                )
            ]
            if not candidate_feats:
                continue
            f_idx = [fi for fi, _ in candidate_feats]
            f_names = [fc for _, fc in candidate_feats]
            _run_tests(g_vec, [g_col_name], feat_arr[:, f_idx], f_names)

    if not rows:
        return pd.DataFrame(columns=[
            'group', 'feature', 'a_group_feat', 'b_group_nofeat',
            'c_nogroup_feat', 'd_nogroup_nofeat', 'rate_in_group',
            'rate_not_group', 'odds_ratio', 'p_value',
        ])

    return pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)


def bh_fdr(p_series, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    p = p_series.values.astype(float)
    n = len(p)
    if n == 0:
        return p_series.copy()

    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    q = p * n / ranks
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        if q_sorted[i] > q_sorted[i + 1]:
            q_sorted[i] = q_sorted[i + 1]
    q[order] = q_sorted
    q = np.clip(q, 0.0, 1.0)

    return pd.Series(q, index=p_series.index)


__all__ = ['fisher_tests', 'bh_fdr', 'group_matrix']
