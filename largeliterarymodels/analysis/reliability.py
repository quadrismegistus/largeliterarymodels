"""Multi-agent ensemble reliability + majority-vote consensus.

Built on top of analysis.reader (load_task_annotations) and analysis.adapters
(classify_schema_fields). Handles N agents symmetrically — designed for
Sonnet+gemma+qwen++llama style ensembles.

Main API:
    load_agent_annotations(task, agents, task_version=2) → {agent: wide_df}
    per_field_trust(frames, schema, reference_agent) → DataFrame (field × agent)
    pairwise_agreement(frames, schema) → DataFrame (field × pair)
    majority_consensus(frames, schema,
                       trust_df=None, trust_threshold=None,
                       reference_agent=None) → (consensus_wide, tiers)
    write_consensus(consensus_wide, task_name, task_version,
                    ensemble_name='ensemble-maj', ...) → n_rows_inserted

Field-type handling for majority:
    bool      → majority True/False; ties → reference
    Literal   → mode; ties → reference
    list[L..] → per-label majority (each candidate label is a separate vote)
    other     → skipped (notes, confidence, etc.)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from .adapters import _coerce_to_list, classify_schema_fields
from .reader import load_task_annotations
from .registry import resolve_task_class


# ── Loading ────────────────────────────────────────────────────────────────

def load_agent_annotations(
    task_name: str,
    agents: Iterable[str],
    *,
    task_version: Optional[int] = None,
    ids: Optional[Iterable[str]] = None,
    client=None,
) -> dict[str, pd.DataFrame]:
    """Load the same task from multiple agents.

    Returns {agent: wide_df} where each wide_df is indexed by
    (_id, scheme, seq) and has one column per schema field.
    """
    frames: dict[str, pd.DataFrame] = {}
    for agent in agents:
        df = load_task_annotations(
            task_name,
            task_version=task_version,
            source_agent=agent,
            ids=list(ids) if ids else None,
            client=client,
        )
        if df.empty:
            raise ValueError(
                f"No CH rows for task={task_name!r} agent={agent!r} "
                f"version={task_version}"
            )
        frames[agent] = df.set_index(['_id', 'scheme', 'seq'])
    return frames


# ── Value normalization ────────────────────────────────────────────────────

def _norm_bool(v) -> Optional[bool]:
    # pd.isna catches NaN, NaT, None. Order matters — check BEFORE bool cast,
    # since Python bool(float('nan')) = True.
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):  # covers numpy.bool_ in older numpy; explicit below
        return bool(v)
    # numpy.bool_ on modern numpy is NOT a subclass of bool — catch via duck-type
    if hasattr(v, 'dtype') and str(getattr(v, 'dtype', '')) == 'bool':
        return bool(v)
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('true', '1'): return True
        if s in ('false', '0', ''): return False
    # Last resort: numpy scalar → try bool() directly
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return bool(v)
    except ImportError:
        pass
    return None


def _norm_list(v) -> tuple:
    return tuple(sorted(str(x) for x in _coerce_to_list(v)))


def _norm_scalar(v) -> str:
    if v is None:
        return ''
    if isinstance(v, float) and pd.isna(v):
        return ''
    return str(v)


def _field_value(row: pd.Series, field: str, kind: str):
    """Pull + normalize a single field value from a row."""
    v = row.get(field)
    if kind == 'bool':
        return _norm_bool(v)
    if kind == 'list':
        return _norm_list(v)
    if kind == 'enum':
        return _norm_scalar(v)
    return _norm_scalar(v)


# ── Per-field trust + pairwise agreement ───────────────────────────────────

def per_field_trust(
    frames: dict[str, pd.DataFrame],
    schema,
    reference_agent: str,
) -> pd.DataFrame:
    """Per-field agreement rate between each non-reference agent and the reference.

    For list[Literal] fields, agreement is exact-set match (strict).
    For bool/Literal fields, agreement is exact value match.

    Returns DataFrame with rows=fields, cols=non-reference agents, values=%match.
    """
    if reference_agent not in frames:
        raise ValueError(f"reference_agent={reference_agent!r} not in frames")

    lists, bools, enums, _ = classify_schema_fields(schema)
    fields = [(f, 'list') for f in lists] + \
             [(f, 'bool') for f in bools] + \
             [(f, 'enum') for f in enums]

    ref = frames[reference_agent]
    other_agents = [a for a in frames if a != reference_agent]

    rows = []
    for fname, kind in fields:
        row = {'field': fname, 'kind': kind}
        for agent in other_agents:
            other = frames[agent]
            common = ref.index.intersection(other.index)
            if len(common) == 0:
                row[agent] = float('nan')
                continue
            agree = 0
            for key in common:
                r_val = _field_value(ref.loc[key], fname, kind)
                o_val = _field_value(other.loc[key], fname, kind)
                if r_val == o_val:
                    agree += 1
            row[agent] = agree / len(common)
            row[f'{agent}_n'] = len(common)
        rows.append(row)

    return pd.DataFrame(rows).set_index('field')


def pairwise_agreement(
    frames: dict[str, pd.DataFrame],
    schema,
) -> pd.DataFrame:
    """All-pairs per-field agreement. Rows=fields, cols=agent pairs."""
    lists, bools, enums, _ = classify_schema_fields(schema)
    fields = [(f, 'list') for f in lists] + \
             [(f, 'bool') for f in bools] + \
             [(f, 'enum') for f in enums]

    agents = list(frames)
    pairs = [(a, b) for i, a in enumerate(agents) for b in agents[i+1:]]

    rows = []
    for fname, kind in fields:
        row = {'field': fname, 'kind': kind}
        for a, b in pairs:
            common = frames[a].index.intersection(frames[b].index)
            if len(common) == 0:
                row[f'{a}={b}'] = float('nan')
                continue
            agree = sum(
                _field_value(frames[a].loc[k], fname, kind)
                == _field_value(frames[b].loc[k], fname, kind)
                for k in common
            )
            row[f'{a}={b}'] = agree / len(common)
        rows.append(row)

    return pd.DataFrame(rows).set_index('field')


# ── Majority consensus ─────────────────────────────────────────────────────

def _majority_bool(votes: list[bool], ref: Optional[bool]) -> Optional[bool]:
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    c = sum(votes)
    n = len(votes)
    if c * 2 > n:
        return True
    if c * 2 < n:
        return False
    return ref if ref is not None else votes[0]


def _majority_enum(votes: list[str], ref: Optional[str]) -> Optional[str]:
    votes = [v for v in votes if v not in (None, '')]
    if not votes:
        return None
    counts = Counter(votes)
    top, n_top = counts.most_common(1)[0]
    ties = [v for v, c in counts.items() if c == n_top]
    if len(ties) == 1:
        return top
    if ref and ref in ties:
        return ref
    return sorted(ties)[0]  # deterministic fallback


def _majority_list(votes: list[tuple], ref: Optional[tuple]) -> tuple:
    """Per-label majority: each candidate label is voted True iff > half of
    the agents that annotated this passage included it."""
    votes = [v for v in votes if v is not None]
    if not votes:
        return ()
    all_labels = set()
    for v in votes:
        all_labels.update(v)
    n = len(votes)
    out = []
    for label in sorted(all_labels):
        votes_for = sum(1 for v in votes if label in v)
        if votes_for * 2 > n:
            out.append(label)
        elif votes_for * 2 == n and ref is not None and label in ref:
            out.append(label)
    return tuple(out)


def majority_consensus(
    frames: dict[str, pd.DataFrame],
    schema,
    *,
    reference_agent: Optional[str] = None,
    trust_df: Optional[pd.DataFrame] = None,
    trust_threshold: Optional[float] = None,
    field_exclusions: Optional[dict[str, list[str]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-passage consensus labels across agents.

    Args:
        frames: {agent: wide_df indexed by (_id, scheme, seq)}
        schema: Pydantic model class for the task
        reference_agent: used for tie-breaking and (with trust_threshold) for
            excluding low-trust agents per-field
        trust_df: output of per_field_trust(); rows=fields, cols=agents
        trust_threshold: if set with trust_df, exclude agents below this
            threshold per-field (e.g. drop qwen's vote on fields where it
            agrees with reference < 0.60)
        field_exclusions: {field: [agents]} explicit per-field agent blacklist.
            Applied ON TOP of the trust-threshold filter.

    Returns:
        (consensus_df, tiers_df)
        - consensus_df: wide DataFrame indexed by (_id, scheme, seq) with one
          column per schema field. Bool → True/False, enum → string, list → tuple.
        - tiers_df: same index, one column per field with value in
          {'unanimous', 'majority', 'no_consensus'} describing agreement level
          *across the agents actually counted* for that field.
    """
    lists, bools, enums, _ = classify_schema_fields(schema)
    kinds = {f: 'list' for f in lists}
    kinds.update({f: 'bool' for f in bools})
    kinds.update({f: 'enum' for f in enums})

    field_exclusions = field_exclusions or {}

    # Determine which agents count for each field.
    def agents_for_field(fname: str) -> list[str]:
        out = list(frames)
        if trust_df is not None and trust_threshold is not None and fname in trust_df.index:
            row = trust_df.loc[fname]
            out = [a for a in out
                   if a == reference_agent
                   or (a in row.index and pd.notna(row.get(a))
                       and row.get(a) >= trust_threshold)]
        if fname in field_exclusions:
            blacklist = set(field_exclusions[fname])
            out = [a for a in out if a not in blacklist]
        return out

    # Union of all passage keys.
    keys = set()
    for df in frames.values():
        keys.update(df.index)
    keys = sorted(keys)

    consensus_rows = {}
    tiers_rows = {}

    for key in keys:
        row = {}
        tiers = {}
        for fname, kind in kinds.items():
            active = agents_for_field(fname)
            votes = []
            for agent in active:
                df = frames[agent]
                if key not in df.index:
                    continue
                votes.append(_field_value(df.loc[key], fname, kind))

            ref_val = None
            if reference_agent and reference_agent in frames and key in frames[reference_agent].index:
                ref_val = _field_value(frames[reference_agent].loc[key], fname, kind)

            if kind == 'bool':
                val = _majority_bool(votes, ref_val)
            elif kind == 'enum':
                val = _majority_enum(votes, ref_val)
            else:  # list
                val = _majority_list(votes, ref_val)

            row[fname] = val

            # Agreement tier — among non-None votes cast for this field
            cast = [v for v in votes if v not in (None, ())]
            if len(cast) < 2:
                tiers[fname] = 'single_vote' if cast else 'no_vote'
            else:
                unique = set()
                for v in cast:
                    unique.add(v if not isinstance(v, tuple) else v)
                if len(unique) == 1:
                    tiers[fname] = 'unanimous'
                elif val is not None and val != () and any(
                    (v == val if not isinstance(val, tuple) else set(v) == set(val))
                    for v in cast
                ):
                    tiers[fname] = 'majority'
                else:
                    tiers[fname] = 'no_consensus'

        consensus_rows[key] = row
        tiers_rows[key] = tiers

    idx = pd.MultiIndex.from_tuples(keys, names=['_id', 'scheme', 'seq'])
    consensus_df = pd.DataFrame.from_dict(consensus_rows, orient='index')
    consensus_df.index = idx
    tiers_df = pd.DataFrame.from_dict(tiers_rows, orient='index')
    tiers_df.index = idx
    return consensus_df, tiers_df


# ── Disagreement / audit helpers ───────────────────────────────────────────

def flagged_for_audit(tiers_df: pd.DataFrame,
                      fields: Optional[list[str]] = None,
                      include_majority: bool = False) -> pd.Series:
    """Flag passages for prioritized audit based on consensus tier.

    Args:
        tiers_df: output of majority_consensus()
        fields: which fields to consider (None = all)
        include_majority: if True, also flag passages where the field landed
            in the 'majority' tier (i.e. 2-of-3 agreement, not unanimous).
            Useful for bool fields where 'no_consensus' is impossible with 3
            voters. Defaults to False — only 'no_consensus' (all-different
            on enum/list) is flagged.
    """
    cols = fields or list(tiers_df.columns)
    flag_values = {'no_consensus'}
    if include_majority:
        flag_values.add('majority')
    return tiers_df[cols].isin(flag_values).any(axis=1)


def audit_disagrees_with_reference(
    consensus_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    schema,
    fields: Optional[list[str]] = None,
) -> pd.Series:
    """Flag passages where consensus disagrees with the reference agent on
    ANY of the given fields. Typical use: prioritize Sonnet audit on
    ensemble-majority labels that contradict the cheap-Sonnet reference.

    Works for bool, enum, and list fields via schema introspection.
    """
    lists, bools, enums, _ = classify_schema_fields(schema)
    kinds = {f: 'list' for f in lists}
    kinds.update({f: 'bool' for f in bools})
    kinds.update({f: 'enum' for f in enums})

    check_fields = fields or list(kinds)
    common = consensus_df.index.intersection(reference_df.index)
    out = pd.Series(False, index=consensus_df.index)

    for k in common:
        for f in check_fields:
            if f not in kinds or f not in reference_df.columns:
                continue
            cv = consensus_df.loc[k, f]
            rv = _field_value(reference_df.loc[k], f, kinds[f])
            if cv is None or rv is None:
                continue
            if cv != rv:
                out[k] = True
                break
    return out


# ── Write consensus back to CH ─────────────────────────────────────────────

def _serialize_value(v) -> str:
    """Convert a consensus value to the CH String-column format.

    Mirrors integrations.llmtasks._serialize_value but accepts tuple-as-list.
    """
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v), ensure_ascii=False, default=str)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, default=str)
    if v is None:
        return ''
    return str(v)


def write_consensus(
    consensus_df: pd.DataFrame,
    *,
    task_name: str,
    task_version: int,
    ensemble_name: str = 'ensemble-maj',
    source_family: str = 'derived',
    run_id: Optional[str] = None,
    tiers_df: Optional[pd.DataFrame] = None,
    client=None,
    dry_run: bool = False,
) -> int:
    """Write consensus labels to passage_annotations as source_family='derived'.

    Args:
        consensus_df: output of majority_consensus; multi-indexed by
            (_id, scheme, seq), cols = field names.
        task_name: e.g. 'passage-form'
        task_version: integer; typically same as the source agents' version
        ensemble_name: becomes source_agent in CH (e.g. 'ensemble-maj3-trust60')
        source_family: 'derived' (default), distinguishes from 'llm'/'human'
        run_id: free-form identifier (saved with every row)
        tiers_df: if provided, each row's meta JSON includes the tier for
            that (passage, field) so downstream queries can filter unanimous
            vs majority vs no_consensus.
        dry_run: compute but skip INSERT.
    """
    from .. import integrations  # for PASSAGE_TABLE constant
    PASSAGE_TABLE = integrations.llmtasks.PASSAGE_TABLE
    if client is None:
        import lltk
        client = lltk.db.client

    if consensus_df.empty:
        return 0

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    field_cols = list(consensus_df.columns)

    for key, row in consensus_df.iterrows():
        _id, scheme, seq = key
        if _id is None or _id == '':
            continue
        for fname in field_cols:
            v = row[fname]
            if v is None or (isinstance(v, tuple) and len(v) == 0 and False):
                # keep empty lists (= passage checked but no labels); skip
                # genuinely missing values (None).
                if v is None:
                    continue
            tier = None
            if tiers_df is not None and fname in tiers_df.columns and key in tiers_df.index:
                tier = tiers_df.loc[key, fname]
            meta_obj = {'ensemble': ensemble_name}
            if tier is not None:
                meta_obj['tier'] = tier
            rows.append((
                str(_id), str(scheme), int(seq), fname,
                _serialize_value(v),
                source_family, ensemble_name, task_name, int(task_version),
                run_id or '', now,
                json.dumps(meta_obj, ensure_ascii=False),
            ))

    if not rows:
        return 0
    if dry_run:
        return len(rows)

    client.insert(
        PASSAGE_TABLE, rows,
        column_names=['_id', 'scheme', 'seq', 'field', 'value',
                      'source_family', 'source_agent', 'task', 'task_version',
                      'run_id', 'annotated_at', 'meta'],
    )
    return len(rows)
