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

Rank agreement (for ranking tasks, NOT the categorical functions above):
    kendall_w(rankings, pool=None) → dict (W, n_items, coverage, p, notes)
    pairwise_rank_correlation(rankings, method='spearman') → DataFrame
    rank_agreement_summary(per_item, pools=None) → DataFrame (item × W)

Field-type handling for majority:
    bool      → majority True/False; ties → reference
    Literal   → mode; ties → reference
    list[L..] → per-label majority (each candidate label is a separate vote)
    other     → skipped (notes, confidence, etc.)
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from .adapters import _coerce_to_list, classify_schema_fields
from .reader import load_task_annotations


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


def _is_even_split(cast: list, kind: str) -> bool:
    """True if the cast votes were an even split, i.e. the winning value had
    to be decided by the reference agent / deterministic fallback rather than
    an actual majority.

    bool: equal True/False counts. enum: top count shared by >= 2 values.
    list: any candidate label voted for by exactly half the voters.
    """
    if kind == 'bool':
        return sum(1 for v in cast if v) * 2 == len(cast)
    if kind == 'enum':
        counts = Counter(cast)
        top_n = counts.most_common(1)[0][1]
        return sum(1 for c in counts.values() if c == top_n) > 1
    if kind == 'list':
        all_labels = set()
        for v in cast:
            all_labels.update(v)
        n = len(cast)
        return any(
            sum(1 for v in cast if label in v) * 2 == n
            for label in all_labels
        )
    return False


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
          {'unanimous', 'majority', 'tie', 'no_consensus', 'single_vote',
          'no_vote'} describing agreement level *across the agents actually
          counted* for that field. 'tie' marks even splits (e.g. 2-2 bool
          votes) whose winning value was decided by the reference agent or a
          deterministic fallback rather than an actual majority.
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
            # Sorted agent order makes the no-reference tie fallback
            # (votes[0] in _majority_bool) deterministic instead of
            # depending on the frames dict's insertion order.
            active = sorted(agents_for_field(fname))
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
                unique = set(cast)
                if len(unique) == 1:
                    tiers[fname] = 'unanimous'
                elif _is_even_split(cast, kind):
                    # Even split: the value above was tie-broken, not voted
                    # in by a majority — don't dress it up as one.
                    tiers[fname] = 'tie'
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
            on enum/list) and 'tie' (even splits resolved by tie-breaking,
            e.g. 2-2 bool votes) are flagged.
    """
    cols = fields or list(tiers_df.columns)
    flag_values = {'no_consensus', 'tie'}
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
    from ..integrations import llmtasks  # for PASSAGE_TABLE constant
    PASSAGE_TABLE = llmtasks.PASSAGE_TABLE
    if client is None:
        from ._ch import _default_client
        client = _default_client()

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
            # Keep empty tuples (= passage checked but no labels); skip
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


# ── Rank agreement ─────────────────────────────────────────────────────────
#
# The functions above are categorical: they ask whether two coders produced
# the same value. Applied to a ranking that question is the wrong one, and
# quietly so — it scores rank 1 against rank 2 as exactly the same
# disagreement as rank 1 against rank 13, discarding the ordering that is the
# whole content of the instrument. A ranking task whose coders visibly agree
# can score mediocre on a categorical statistic, which inverts the conclusion
# rather than merely blurring it. The failure mode is not a function that
# errors; it is an existing function that returns a plausible number for a
# question you did not ask.
#
# For m coders ranking n items the statistic is Kendall's W (coefficient of
# concordance), with pairwise Spearman/Kendall for coder-to-coder detail.


def _as_tie_groups(entry, coder=None) -> list[list]:
    """Normalise one coder's ranking into ordered tie-groups, best first.

    Accepts either:
        ["kill", ["cry", "scream"], "run"]   ordered; a nested sequence is a tie
        {"kill": 1, "cry": 2, "scream": 2}   explicit ranks; equal rank = tie

    Every malformed input below used to produce a number rather than an error,
    which is the failure mode this module exists to avoid:

    - a bare string ranks its own characters;
    - a duplicated item overwrites its own midrank, so S is computed against a
      rank vector that does not sum to n(n+1)/2 and W can exceed 1;
    - a NaN rank compares false against everything including itself, so the
      tie-group it lands in — and hence the whole ranking — depends on dict
      insertion order;
    - a non-numeric rank raises `could not convert string to float` with no
      indication of which coder sent it.

    `coder` is only used to name the offender in the error message.
    """
    who = f'coder {coder!r}: ' if coder is not None else ''

    if isinstance(entry, (str, bytes)):
        raise ValueError(
            f'{who}ranking is a bare string {entry!r}; iterating it would rank '
            f'its individual characters. Pass a sequence of items.'
        )

    if isinstance(entry, dict):
        by_rank: dict[float, list] = {}
        for item, rank in entry.items():
            if rank is None:
                raise ValueError(
                    f'{who}item {item!r} has rank None. A missing rank must be '
                    f'omitted from the ranking, not recorded as None — the '
                    f'intersection is what handles partial coverage.'
                )
            try:
                value = float(rank)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'{who}item {item!r} has non-numeric rank {rank!r}'
                ) from exc
            if value != value:  # NaN
                raise ValueError(
                    f'{who}item {item!r} has rank NaN. NaN compares false '
                    f'against every rank including itself, so the tie-group it '
                    f'joins would depend on dict insertion order. Omit the '
                    f'item instead.'
                )
            by_rank.setdefault(value, []).append(item)
        groups = [sorted(by_rank[r], key=str) for r in sorted(by_rank)]
    else:
        groups = []
        for element in entry:
            if isinstance(element, (list, tuple, set, frozenset)):
                groups.append(sorted(element, key=str))
            else:
                groups.append([element])

    seen: set = set()
    for group in groups:
        for item in group:
            if item in seen:
                raise ValueError(
                    f'{who}item {item!r} appears more than once in the ranking. '
                    f'A repeated item overwrites its own midrank and can push W '
                    f'above 1; rank each item exactly once.'
                )
            seen.add(item)
    return groups


def _midranks(groups: list[list]) -> dict:
    """Assign midranks over ordered tie-groups: [a, [b, c], d] → 1, 2.5, 2.5, 4."""
    ranks, position = {}, 1
    for group in groups:
        mid = position + (len(group) - 1) / 2
        for item in group:
            ranks[item] = mid
        position += len(group)
    return ranks


def _restrict_and_rerank(groups: list[list], keep: set) -> tuple[dict, list[list]]:
    """Re-rank a coder over a subset of items.

    Ranks must be recomputed after intersecting, not filtered: a coder whose
    2nd and 5th choices survive is expressing ranks 1 and 2 over the surviving
    set, not 2 and 5. Filtering without re-ranking silently inflates S.
    """
    restricted = [[i for i in g if i in keep] for g in groups]
    restricted = [g for g in restricted if g]
    return _midranks(restricted), restricted


def _validate_pool(pool, n_distinct: int) -> int:
    """Check a declared candidate-pool size against what the coders did.

    An unvalidated pool is worse than no pool: `coverage` is the one number
    that stops a W over 4 items being quoted as a W over 15, and a pool smaller
    than the item set silently returns a coverage above 1 rather than failing.
    """
    try:
        as_int = int(pool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'pool must be a positive integer, got {pool!r}') from exc
    if as_int != pool or as_int <= 0:
        raise ValueError(f'pool must be a positive integer, got {pool!r}')
    if as_int < n_distinct:
        raise ValueError(
            f'pool={as_int} is smaller than the {n_distinct} distinct items the '
            f'coders ranked, which would report coverage above 1'
        )
    return as_int


def kendall_w(
    rankings: dict,
    pool: Optional[int] = None,
    min_items: int = 4,
) -> dict:
    """Kendall's W (coefficient of concordance) across coders' rankings.

    Ties are expected and corrected for. Coders may rank overlapping but
    non-identical subsets (items a coder judged unrankable simply do not
    appear); W is computed on the intersection, and `n_items` reports how many
    items that actually was. Read it — a coder who ranked 4 of a 15-word pool
    collapses the intersection to 4, and a W over 4 items is not comparable to
    a W over 15 no matter how high it is.

    `mean_spearman` is the mean of the pairwise Spearman correlations computed
    over the GLOBAL intersection — the same item set W itself uses. It is not
    derived from the textbook identity W = (1 + (m-1)r̄)/m, which holds only
    when every coder has the same tie structure and is silently wrong
    otherwise. Note that `pairwise_rank_correlation` uses PER-PAIR
    intersections instead, so its mean coefficient will differ from
    `mean_spearman` whenever coverage differs between coders; neither is wrong,
    they are answering questions about different item sets.

    Args:
        rankings: {coder: ranking}, each ranking in either form accepted by
            _as_tie_groups.
        pool: Optional size of the candidate pool the coders drew from. Must be
            a positive integer no smaller than the number of distinct items the
            coders actually ranked, else ValueError — a pool below that yields
            coverage > 1, which is not a coverage. When omitted, `coverage`
            falls back to intersection / union of all coders' items, so the
            restriction is always reported rather than opt-in.
        min_items: Refuse to return a W below this many items (default 4).
            Returns w=None with a note rather than a number that invites
            quotation.

    Returns:
        dict with keys: w, n_items, n_coders, coders, items, coverage,
        dropped_per_coder, ties_present, chi2, df, p_value, p_approximate,
        mean_spearman, notes, note. `w` is None whenever it could not be
        computed, and `notes` always says why. `notes` is the list; `note` is
        those notes joined with '; ', or None when there are none — several
        conditions can hold at once, so a single-slot note loses whichever one
        was written first.
    """
    coders = sorted(rankings)
    m = len(coders)
    grouped = {c: _as_tie_groups(rankings[c], coder=c) for c in coders}
    sets = {c: set(i for g in grouped[c] for i in g) for c in coders}

    union: set = set().union(*sets.values()) if sets else set()
    common = set.intersection(*(sets[c] for c in coders)) if coders else set()
    n = len(common)

    if pool is not None:
        pool = _validate_pool(pool, len(union))

    notes: list[str] = []
    out = {
        'w': None, 'n_items': n, 'n_coders': m, 'coders': tuple(coders),
        'items': tuple(sorted(common, key=str)),
        'coverage': (n / pool) if pool is not None
                    else (n / len(union) if union else None),
        # Distinct items, not tokens: a coder who repeated an item has not
        # thereby dropped one.
        'dropped_per_coder': {c: len(sets[c]) - n for c in coders},
        'ties_present': False,
        'chi2': None, 'df': None, 'p_value': None, 'p_approximate': None,
        'mean_spearman': None, 'notes': notes, 'note': None,
    }

    def finish():
        out['notes'] = list(notes)
        out['note'] = '; '.join(notes) if notes else None
        return out

    if m < 2:
        notes.append('W needs at least 2 coders')
        return finish()
    if n < 2:
        notes.append(f'only {n} item(s) common to all {m} coders')
        return finish()
    if n < min_items:
        notes.append(
            f'{n} common items is below min_items={min_items}; W over so few '
            f'items is unstable and not comparable to W over a full pool'
        )
        return finish()

    rank_maps, tie_correction = {}, 0.0
    flat_coders = []
    for c in coders:
        ranks, restricted = _restrict_and_rerank(grouped[c], common)
        rank_maps[c] = ranks
        # Tie correction must come off the RESTRICTED groups: a tie the
        # intersection has broken is no longer a tie in the ranking being
        # scored, and correcting for it shrinks the denominator, inflating W
        # above 1 in the limit.
        for group in restricted:
            t = len(group)
            if t > 1:
                out['ties_present'] = True
                tie_correction += t ** 3 - t
        if len(set(ranks.values())) < 2:
            flat_coders.append(c)

    if flat_coders:
        notes.append(
            'zero-variance coder(s) ' + ', '.join(repr(c) for c in flat_coders)
            + ' tied every common item; they carry no ordering information but '
              'still count toward m, dragging W toward its no-information value'
        )

    items = out['items']
    rank_sums = [sum(rank_maps[c][i] for c in coders) for i in items]
    mean_rank_sum = m * (n + 1) / 2
    S = sum((r - mean_rank_sum) ** 2 for r in rank_sums)

    denominator = m ** 2 * (n ** 3 - n) - m * tie_correction
    if denominator <= 0:
        notes.append('no rank variation to measure (every item tied)')
        return finish()

    w = 12 * S / denominator
    out['w'] = w

    from scipy import stats

    # Mean pairwise Spearman, computed rather than inferred. The identity
    # W = (1 + (m-1)r̄)/m assumes an identical tie structure across coders and
    # is off by ~0.05 on realistic tied data; it also reports 0.0 for a pair
    # whose correlation is undefined, which is a claim of independence rather
    # than of ignorance.
    coefficients, undefined = [], []
    for i, a in enumerate(coders):
        for b in coders[i + 1:]:
            xa = [rank_maps[a][it] for it in items]
            xb = [rank_maps[b][it] for it in items]
            if len(set(xa)) < 2 or len(set(xb)) < 2:
                undefined.append(f'{a}~{b}')
                continue
            coefficients.append(float(stats.spearmanr(xa, xb).statistic))
    out['mean_spearman'] = (
        sum(coefficients) / len(coefficients) if coefficients else None
    )
    if undefined:
        notes.append(
            'mean_spearman excludes ' + str(len(undefined))
            + ' undefined pair(s) (' + ', '.join(undefined)
            + ') where one coder had no rank variation'
        )

    chi2 = m * (n - 1) * w
    out['chi2'] = chi2
    out['df'] = n - 1
    out['p_value'] = float(stats.chi2.sf(chi2, n - 1))
    # The chi-square approximation is poor for short rankings, and — separately
    # — for two coders at any n, where m(n-1)W is not close to chi-square
    # because W is just a rescaled Spearman. Say so rather than let a p-value
    # be quoted at face value.
    out['p_approximate'] = bool(n <= 7 or m == 2)
    if n <= 7:
        notes.append(
            f'p from the chi-square approximation, unreliable at n={n} '
            f'(adequate above ~7 items)'
        )
    if m == 2:
        notes.append(
            'with 2 coders W is a rescaled Spearman correlation and the '
            'chi-square approximation on m(n-1)W is poor at any n; use the '
            'Spearman p-value from pairwise_rank_correlation instead'
        )
    return finish()


def pairwise_rank_correlation(
    rankings: dict,
    method: str = 'spearman',
    min_items: int = 3,
) -> pd.DataFrame:
    """All-pairs rank correlation between coders. Rows = coder pairs.

    Each pair uses its OWN intersection rather than the global one, so a single
    low-coverage coder does not shrink every other pair's n. This is the
    opposite convention from `kendall_w`, which is necessarily global: when
    coverage differs between coders the mean of this table's `coefficient`
    column will NOT equal `kendall_w(...)['mean_spearman']`, because the two
    are computed over different item sets. Compare them only when every coder
    ranked the same items.

    Kendall uses tau-b, which is tie-corrected; Spearman is computed on
    midranks.

    Args:
        rankings: {coder: ranking}, in either form accepted by _as_tie_groups.
        method: 'spearman' or 'kendall'.
        min_items: pairs sharing fewer items than this get coefficient=NaN and
            computable=False (default 3). Note that a coefficient at n=3 is
            barely one: untied Spearman over 3 items can only take the values
            1, 0.5, -0.5 and -1, so its p-value cannot go below 1/6 ≈ 0.167 and
            reporting significance there is meaningless. Raise min_items if the
            table is going to be read as evidence.

    Returns:
        DataFrame with columns: coder_a, coder_b, n, computable, coefficient,
        p_value. `computable` is False for pairs below min_items and for pairs
        where one coder tied everything (an undefined correlation) — filter on
        it explicitly rather than relying on pandas' .mean() to skip the NaNs,
        which hides how many pairs the mean rests on.
    """
    if method not in ('spearman', 'kendall'):
        raise ValueError("method must be 'spearman' or 'kendall'")
    from scipy import stats

    coders = sorted(rankings)
    grouped = {c: _as_tie_groups(rankings[c], coder=c) for c in coders}
    sets = {c: set(i for g in grouped[c] for i in g) for c in coders}

    rows = []
    for i, a in enumerate(coders):
        for b in coders[i + 1:]:
            common = sets[a] & sets[b]
            row = {'coder_a': a, 'coder_b': b, 'n': len(common),
                   'computable': False,
                   'coefficient': float('nan'), 'p_value': float('nan')}
            if len(common) >= min_items:
                ra, _ = _restrict_and_rerank(grouped[a], common)
                rb, _ = _restrict_and_rerank(grouped[b], common)
                items = sorted(common, key=str)
                xa = [ra[i2] for i2 in items]
                xb = [rb[i2] for i2 in items]
                if len(set(xa)) >= 2 and len(set(xb)) >= 2:
                    if method == 'spearman':
                        res = stats.spearmanr(xa, xb)
                    else:
                        res = stats.kendalltau(xa, xb, variant='b')
                    row['computable'] = True
                    row['coefficient'] = float(res.statistic)
                    row['p_value'] = float(res.pvalue)
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=['coder_a', 'coder_b', 'n', 'computable',
                     'coefficient', 'p_value'])
    return pd.DataFrame(rows)


#: Columns of rank_agreement_summary(), including when it is empty.
SUMMARY_COLUMNS = ('w', 'n_items', 'n_coders', 'coverage',
                   'ties_present', 'p_value', 'note')


def rank_agreement_summary(
    per_item: dict,
    pools: Optional[dict] = None,
    min_items: int = 4,
) -> pd.DataFrame:
    """Kendall's W per item across a corpus of ranking tasks.

    Args:
        per_item: {item_id: {coder: ranking}}.
        pools: Optional {item_id: pool_size} for per-item coverage.
        min_items: Passed to kendall_w.

    Returns:
        DataFrame indexed by item_id with the columns named in
        SUMMARY_COLUMNS: w, n_items, n_coders, coverage, ties_present, p_value
        and note. Items whose W could not be computed keep their row with w=NaN
        and the reason in `note`, rather than being dropped — a silently
        shorter table is how a coverage problem becomes invisible. An empty
        `per_item` returns an empty frame with those columns, not an error:
        callers concatenate these, and a zero-row corpus is a legitimate state.
    """
    rows = []
    for item_id, rankings in per_item.items():
        pool = pools.get(item_id) if pools else None
        r = kendall_w(rankings, pool=pool, min_items=min_items)
        rows.append({
            'item_id': item_id,
            'w': r['w'] if r['w'] is not None else float('nan'),
            'n_items': r['n_items'],
            'n_coders': r['n_coders'],
            'coverage': r['coverage'] if r['coverage'] is not None else float('nan'),
            'ties_present': r['ties_present'],
            'p_value': r['p_value'] if r['p_value'] is not None else float('nan'),
            'note': r['note'] or '',
        })
    if not rows:
        return pd.DataFrame(
            columns=list(SUMMARY_COLUMNS),
            index=pd.Index([], name='item_id'),
        )
    return pd.DataFrame(rows).set_index('item_id')
