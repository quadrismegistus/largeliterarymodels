"""CH reader for cross-task joint analysis.

Pulls passage annotations from llmtasks.passage_annotations via
largeliterarymodels.integrations.llmtasks.read_passage_annotations, unpacks
each task's wide DataFrame into a boolean feature matrix, and inner-joins
across tasks on (_id, scheme, seq).
"""

from typing import Iterable, Optional

import pandas as pd

from largeliterarymodels.integrations import llmtasks

from .adapters import wide_to_features
from .registry import prefix_for, resolve_task_class


def load_task_annotations(
    task_name: str,
    *,
    task_version: Optional[int] = None,
    source_agent: Optional[str] = None,
    ids: Optional[Iterable[str]] = None,
    client=None,
) -> pd.DataFrame:
    """Load a single task's annotations from CH in wide form.

    Returns a DataFrame with columns (_id, scheme, seq, <task fields>).
    Passages are at row-level; fields are at column-level.
    """
    wide = llmtasks.read_passage_annotations(
        ids=list(ids) if ids else None,
        task_name=task_name,
        task_version=task_version,
        source_agent=source_agent,
        use_latest_view=True,
        client=client,
    )
    return wide


def joint_feature_matrix(
    tasks: list[str],
    *,
    task_versions: Optional[dict[str, int]] = None,
    source_agents: Optional[dict[str, str]] = None,
    is_prose_fiction: bool = True,
    ids: Optional[Iterable[str]] = None,
    client=None,
) -> pd.DataFrame:
    """Build a joint boolean feature matrix across N tasks.

    For each task:
      1. Read wide annotation table from CH.
      2. Optionally filter to is_prose_fiction=True (if that field exists
         on the task's schema — currently on PassageContentTask V3+ and
         PassageFormTask).
      3. Unpack to boolean feature matrix with task-prefixed column names.

    Then inner-join feature matrices on (_id, scheme, seq). Final matrix has
    one row per passage that was annotated by ALL requested tasks.

    Args:
        tasks: CH task names to include (e.g. ['passage-content', 'passage-form'])
        task_versions: {task_name: int} — None = latest
        source_agents: {task_name: str} — None = all agents
        is_prose_fiction: if True, filter passages where the task reported
            is_prose_fiction=False. Applied per-task; intersection across
            tasks is the result.
        ids: restrict to these text _ids (optional)
        client: CH client override

    Returns:
        Boolean DataFrame indexed by (_id, scheme, seq). Columns are
        prefixed by task slug ('content.', 'form.', etc.).
    """
    task_versions = task_versions or {}
    source_agents = source_agents or {}

    frames: list[pd.DataFrame] = []
    for task_name in tasks:
        task_class = resolve_task_class(task_name)
        wide = load_task_annotations(
            task_name,
            task_version=task_versions.get(task_name),
            source_agent=source_agents.get(task_name),
            ids=ids,
            client=client,
        )
        if wide.empty:
            raise ValueError(
                f"No CH rows for task={task_name!r}, "
                f"version={task_versions.get(task_name)}, "
                f"agent={source_agents.get(task_name)}"
            )

        # Per-task is_prose_fiction filter (if the schema has that field)
        if is_prose_fiction and 'is_prose_fiction' in wide.columns:
            n_before = len(wide)
            wide = wide[wide['is_prose_fiction'] == True]
            n_after = len(wide)
            if n_after < n_before:
                print(
                    f"[{task_name}] filtered {n_before - n_after} "
                    f"non-prose passages ({n_after} kept)",
                    flush=True,
                )

        wide = wide.set_index(['_id', 'scheme', 'seq'])
        feats = wide_to_features(
            wide, task_class.schema, prefix=prefix_for(task_name)
        )
        frames.append(feats)

    if not frames:
        raise ValueError("No task data loaded.")

    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, how='inner')
    return result
