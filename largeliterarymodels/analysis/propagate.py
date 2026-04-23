"""Propagate passage annotations to unlabeled passages via embedding classifiers.

Train per-field classifiers on labeled (passage, embedding) pairs, evaluate
with cross-validation, and optionally predict on the full 377K embedding pool.

Only writes predictions to CH after explicit confirmation — this module
reports accuracy first and lets the caller decide.

Usage:
    from largeliterarymodels.analysis.propagate import (
        train_classifiers, evaluate_classifiers, predict_all,
    )

    # Phase 1: evaluate
    report = evaluate_classifiers(
        task_name='passage-content', task_version=3,
        source_agent='qwen3.5-35b-a3b',
    )
    print(report)  # per-field accuracy, F1, AUC

    # Phase 2: predict (only if accuracy is good)
    predictions = predict_all(
        classifiers=report['classifiers'],
        min_accuracy=0.75,
    )

    # Phase 3: write (explicit, separate step)
    write_propagated(predictions, task_name='passage-content', task_version=3)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _load_labeled_embeddings(
    task_name: str,
    task_version: int,
    source_agent: str,
    is_prose_fiction: bool = True,
    include_lang: bool = True,
    client=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labeled passages with their embeddings (+ optional language feature).

    Returns:
        (labels_df, features_df) both indexed by (_id, scheme, seq).
        labels_df has task fields as columns.
        features_df has embedding dims as columns, plus 'lang_fr' (0/1) if
        include_lang=True. Language is a confounder — including it prevents
        classifiers from learning "French embedding region = X" instead of
        actual textual signal.
    """
    import lltk
    from largeliterarymodels.integrations import llmtasks

    if client is None:
        client = lltk.db.client

    labels = llmtasks.read_passage_annotations(
        task_name=task_name,
        task_version=task_version,
        source_agent=source_agent,
        use_latest_view=True,
        client=client,
    )
    if labels.empty:
        raise ValueError(f"No annotations for {task_name} v{task_version} agent={source_agent}")

    if is_prose_fiction and 'is_prose_fiction' in labels.columns:
        n_before = len(labels)
        labels = labels[labels['is_prose_fiction'] == True]
        log.info("prose filter: %d -> %d", n_before, len(labels))

    labels = labels.set_index(['_id', 'scheme', 'seq'])

    ids_unique = list(labels.index.get_level_values('_id').unique())
    escaped = ', '.join("'" + i.replace("'", "''") + "'" for i in ids_unique)
    emb_df = client.query_df(
        f"SELECT _id, scheme, seq, embedding "
        f"FROM lltk.passage_embeddings "
        f"WHERE _id IN ({escaped}) AND scheme = 'p500'"
    )
    emb_df = emb_df.set_index(['_id', 'scheme', 'seq'])

    common = labels.index.intersection(emb_df.index)
    log.info("labeled: %d, with embeddings: %d, overlap: %d",
             len(labels), len(emb_df), len(common))

    labels = labels.loc[common]
    features = pd.DataFrame(
        emb_df.loc[common, 'embedding'].tolist(),
        index=common,
    )

    if include_lang:
        lang_df = client.query_df(
            f"SELECT _id, lang FROM lltk.passages "
            f"WHERE _id IN ({escaped}) AND scheme = 'p500'"
        )
        if not lang_df.empty and 'lang' in lang_df.columns:
            lang_df = lang_df.set_index('_id')['lang']
            lang_map = lang_df.to_dict()
            ids = features.index.get_level_values('_id')
            features['lang_fr'] = [
                1 if lang_map.get(i, '').startswith('fr') else 0 for i in ids
            ]
            n_fr = features['lang_fr'].sum()
            log.info("language feature: %d FR, %d EN/other", n_fr, len(features) - n_fr)

    return labels, features


def _prepare_targets(labels: pd.DataFrame, schema) -> dict[str, pd.Series]:
    """Extract per-field binary targets from the labels DataFrame.

    Returns {field_name: binary_series} for all bool and list-expanded fields.
    Skips free-text fields (str), confidence, and ordinals.
    """
    import typing
    from .adapters import _coerce_to_list

    targets = {}
    for name, fld in schema.model_fields.items():
        t = fld.annotation
        origin = typing.get_origin(t)

        if t is bool:
            if name not in labels.columns:
                continue
            col = labels[name]
            targets[name] = col.apply(
                lambda x: 1 if (x is True or (isinstance(x, str) and x.lower() == 'true'))
                else 0
            ).astype(int)
        elif origin in (list, typing.List):
            if name not in labels.columns:
                continue
            parsed = labels[name].apply(_coerce_to_list)
            args = typing.get_args(t)
            if args:
                inner = args[0]
                inner_origin = typing.get_origin(inner)
                if inner_origin is typing.Literal:
                    values = list(typing.get_args(inner))
                    for v in values:
                        targets[f'{name}__{v}'] = parsed.apply(
                            lambda lst, v=v: 1 if v in lst else 0
                        ).astype(int)

    return targets


def evaluate_classifiers(
    task_name: str = 'passage-content',
    task_version: int = 3,
    source_agent: str = 'qwen3.5-35b-a3b',
    *,
    is_prose_fiction: bool = True,
    n_splits: int = 5,
    pca_components: Optional[int] = 100,
    min_positive_rate: float = 0.01,
    client=None,
) -> pd.DataFrame:
    """Train and evaluate per-field classifiers via stratified cross-validation.

    Returns a DataFrame with one row per field:
      field, accuracy, f1, roc_auc, positive_rate, n_samples, classifier

    The 'classifier' column holds the fitted sklearn model (trained on ALL
    data after CV — use for prediction only if metrics are acceptable).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from .registry import resolve_task_class

    task_class = resolve_task_class(task_name)
    labels, emb_matrix = _load_labeled_embeddings(
        task_name, task_version, source_agent,
        is_prose_fiction=is_prose_fiction, client=client,
    )
    targets = _prepare_targets(labels, task_class.schema)

    X = emb_matrix.values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = None
    if pca_components and X_scaled.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X_scaled)
        log.info("PCA: %d -> %d (%.1f%% variance)",
                 X_scaled.shape[1], pca_components,
                 100 * pca.explained_variance_ratio_.sum())
    else:
        X_reduced = X_scaled

    rows = []
    for field_name, y in targets.items():
        pos_rate = y.mean()
        if pos_rate < min_positive_rate or pos_rate > (1 - min_positive_rate):
            rows.append({
                'field': field_name, 'accuracy': np.nan, 'f1': np.nan,
                'roc_auc': np.nan, 'positive_rate': pos_rate,
                'n_samples': len(y), 'classifier': None,
                'note': 'skipped: near-constant',
            })
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in skf.split(X_reduced, y):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            clf.fit(X_reduced[train_idx], y.values[train_idx])
            y_pred = clf.predict(X_reduced[test_idx])
            y_prob = clf.predict_proba(X_reduced[test_idx])[:, 1]

            accs.append(accuracy_score(y.values[test_idx], y_pred))
            f1s.append(f1_score(y.values[test_idx], y_pred, zero_division=0))
            try:
                aucs.append(roc_auc_score(y.values[test_idx], y_prob))
            except ValueError:
                aucs.append(np.nan)

        # Refit on full data for prediction
        final_clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        final_clf.fit(X_reduced, y.values)

        rows.append({
            'field': field_name,
            'accuracy': np.mean(accs),
            'f1': np.mean(f1s),
            'roc_auc': np.mean(aucs),
            'positive_rate': pos_rate,
            'n_samples': len(y),
            'classifier': final_clf,
            'note': '',
        })

    report = pd.DataFrame(rows)
    report.attrs['scaler'] = scaler
    report.attrs['pca'] = pca
    report.attrs['pca_components'] = pca_components
    report.attrs['task_name'] = task_name
    report.attrs['task_version'] = task_version
    report.attrs['n_train'] = len(emb_matrix)
    report.attrs['include_lang'] = True
    return report


def calibrate_thresholds(
    report: pd.DataFrame,
    *,
    task_name: str = 'passage-content',
    task_version: int = 3,
    source_agent: str = 'qwen3.5-35b-a3b',
    min_precision: float = 0.95,
    n_splits: int = 5,
    pca_components: Optional[int] = 100,
    client=None,
) -> pd.DataFrame:
    """Find per-field probability thresholds that achieve target precision.

    Returns a DataFrame with columns:
      field, threshold, precision, recall, n_predicted_pos
    Only includes fields that achieve min_precision at some threshold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_score, recall_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from .registry import resolve_task_class

    task_class = resolve_task_class(task_name)
    labels, features = _load_labeled_embeddings(
        task_name, task_version, source_agent,
        is_prose_fiction=True, include_lang=True, client=client,
    )
    targets = _prepare_targets(labels, task_class.schema)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values.astype(np.float32))
    pca = None
    if pca_components and X_scaled.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X_scaled

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for field_name, y in targets.items():
        pos_rate = y.mean()
        if pos_rate < 0.01 or pos_rate > 0.99:
            continue

        all_probs = np.zeros(len(y))
        for train_idx, test_idx in skf.split(X_reduced, y):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            clf.fit(X_reduced[train_idx], y.values[train_idx])
            all_probs[test_idx] = clf.predict_proba(X_reduced[test_idx])[:, 1]

        for thresh in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
            preds = (all_probs >= thresh).astype(int)
            if preds.sum() == 0:
                continue
            prec = precision_score(y.values, preds, zero_division=0)
            rec = recall_score(y.values, preds, zero_division=0)
            if prec >= min_precision:
                rows.append({
                    'field': field_name,
                    'threshold': thresh,
                    'precision': prec,
                    'recall': rec,
                    'n_predicted_pos': int(preds.sum()),
                    'positive_rate': pos_rate,
                })
                break

    return pd.DataFrame(rows)


def predict_all(
    report: pd.DataFrame,
    *,
    thresholds: Optional[dict[str, float]] = None,
    min_accuracy: float = 0.75,
    min_f1: float = 0.0,
    batch_size: int = 50000,
    client=None,
) -> pd.DataFrame:
    """Predict on ALL passage embeddings for fields that pass thresholds.

    Args:
        report: output of evaluate_classifiers (has classifiers + scaler/pca in attrs)
        thresholds: {field_name: probability_threshold} — per-field thresholds
            from calibrate_thresholds(). If provided, uses these instead of 0.5
            and ignores min_accuracy/min_f1 (only predicts fields in this dict).
        min_accuracy: fallback filter if thresholds not provided
        min_f1: fallback filter if thresholds not provided
        batch_size: CH query batch size

    Returns:
        Wide DataFrame indexed by (_id, scheme, seq) with one column per
        predicted field (binary 0/1). Only includes fields passing thresholds.
    """
    import lltk
    if client is None:
        client = lltk.db.client

    scaler = report.attrs['scaler']
    pca = report.attrs.get('pca')

    if thresholds:
        field_names = [f for f in thresholds if f in report['field'].values]
        classifiers = dict(zip(report['field'], report['classifier']))
        classifiers = {f: classifiers[f] for f in field_names if classifiers.get(f) is not None}
        field_names = list(classifiers.keys())
    else:
        good_fields = report[
            (report['accuracy'] >= min_accuracy) &
            (report['f1'] >= min_f1) &
            (report['classifier'].notna())
        ]
        if good_fields.empty:
            log.warning("No fields pass thresholds")
            return pd.DataFrame()
        field_names = good_fields['field'].tolist()
        classifiers = dict(zip(good_fields['field'], good_fields['classifier']))
        thresholds = {f: 0.5 for f in field_names}

    log.info("predicting %d fields on full embedding pool", len(field_names))

    # Load all embeddings in batches
    total = client.query_df(
        "SELECT count() as n FROM lltk.passage_embeddings WHERE scheme = 'p500'"
    ).iloc[0][0]
    log.info("total embeddings to predict: %d", total)

    all_preds = []
    offset = 0
    while offset < total:
        batch = client.query_df(
            f"SELECT _id, scheme, seq, embedding "
            f"FROM lltk.passage_embeddings "
            f"WHERE scheme = 'p500' "
            f"ORDER BY _id, seq "
            f"LIMIT {batch_size} OFFSET {offset}"
        )
        if batch.empty:
            break

        idx = batch.set_index(['_id', 'scheme', 'seq']).index
        X = np.array(batch['embedding'].tolist(), dtype=np.float32)

        # Add language feature BEFORE scaling (must match training feature order)
        if report.attrs.get('include_lang', False):
            batch_ids = batch['_id'].unique().tolist()
            escaped_batch = ', '.join(
                "'" + i.replace("'", "''") + "'" for i in batch_ids
            )
            lang_df = client.query_df(
                f"SELECT DISTINCT _id, lang FROM lltk.passages "
                f"WHERE _id IN ({escaped_batch}) AND scheme = 'p500'"
            )
            lang_map = dict(zip(lang_df['_id'], lang_df['lang'])) if not lang_df.empty else {}
            lang_col = np.array([
                1 if lang_map.get(i, '').startswith('fr') else 0
                for i in batch['_id']
            ], dtype=np.float32).reshape(-1, 1)
            X = np.hstack([X, lang_col])

        X_scaled = scaler.transform(X)
        if pca is not None:
            X_scaled = pca.transform(X_scaled)

        pred_dict = {}
        for fname in field_names:
            probs = classifiers[fname].predict_proba(X_scaled)[:, 1]
            thresh = thresholds.get(fname, 0.5)
            pred_dict[fname] = (probs >= thresh).astype(int)

        pred_df = pd.DataFrame(pred_dict, index=idx)
        all_preds.append(pred_df)
        offset += batch_size
        log.info("predicted %d / %d", min(offset, total), total)

    return pd.concat(all_preds)


def write_propagated(
    predictions: pd.DataFrame,
    *,
    task_name: str = 'passage-content',
    task_version: int = 3,
    source_agent: str = 'e5-classifier',
    source_family: str = 'derived',
    run_id: Optional[str] = None,
    client=None,
) -> int:
    """Write propagated predictions to llmtasks.passage_annotations.

    Only call this after inspecting evaluate_classifiers() output
    and confirming accuracy is acceptable.

    Returns the number of rows written.
    """
    from datetime import datetime, timezone
    import lltk
    from largeliterarymodels.integrations.llmtasks import (
        PASSAGE_TABLE, ensure_schema,
    )

    if client is None:
        client = lltk.db.client

    ensure_schema(client=client)
    run_id = run_id or f'propagate-{datetime.now(timezone.utc).strftime("%Y%m%d")}'
    now = datetime.now(timezone.utc)

    rows = []
    for ((_id, scheme, seq), row) in predictions.iterrows():
        for field, value in row.items():
            rows.append({
                '_id': _id,
                'scheme': scheme,
                'seq': int(seq),
                'field': str(field),
                'value': 'true' if value == 1 else 'false',
                'source_family': source_family,
                'source_agent': source_agent,
                'task': task_name,
                'task_version': task_version,
                'run_id': run_id,
                'annotated_at': now,
                'meta': '{}',
            })

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    client.insert_df(PASSAGE_TABLE, df)
    log.info("wrote %d propagated rows to %s", len(df), PASSAGE_TABLE)
    return len(df)
