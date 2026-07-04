"""Registry: task name (as stored in CH) → Task class.

Maps the `task` column in llmtasks.passage_annotations to the Pydantic-bearing
Task class that defines its schema. The schema is introspected for
list/bool/enum field classification — no per-task adapter code required.
"""

TASK_REGISTRY: dict[str, str] = {
    # CH task-name → dotted-path importable Task class name.
    # Naming convention: Task.name with its 'classify_'/'score_' verb
    # stripped and underscores hyphenated ('classify_passage_content' →
    # 'passage-content'). If an ingest pipeline stores a different task
    # name, override with register_task().
    'passage-content':        'largeliterarymodels.tasks.classify_passage_content:PassageContentTask',
    'passage-form':           'largeliterarymodels.tasks.classify_passage_form:PassageFormTask',
    'passage-narrativity':    'largeliterarymodels.tasks.classify_passage_narrativity:PassageNarrativityTask',
    'passage-setting':        'largeliterarymodels.tasks.classify_passage_setting:PassageSettingTask',
    'emotion':                'largeliterarymodels.tasks.classify_emotion:EmotionTask',
    'emotion-zh':             'largeliterarymodels.tasks.classify_emotion_zh:EmotionTaskZh',
    'contradiction-response': 'largeliterarymodels.tasks.classify_contradiction_response:ContradictionResponseTask',
    'alignment-asymmetry':    'largeliterarymodels.tasks.score_alignment_asymmetry:AlignmentAsymmetryTask',
}


def register_task(ch_task_name: str, dotted_path: str) -> None:
    """Register a new task's Task class under its CH task-name.

    `dotted_path` is `module.path:ClassName`.
    """
    TASK_REGISTRY[ch_task_name] = dotted_path


def resolve_task_class(ch_task_name: str) -> type:
    """Import and return the Task class for a CH task name."""
    if ch_task_name not in TASK_REGISTRY:
        raise KeyError(
            f"No Task class registered for CH task {ch_task_name!r}. "
            f"Registered: {sorted(TASK_REGISTRY)}. "
            f"Call register_task() to add."
        )
    module_path, class_name = TASK_REGISTRY[ch_task_name].split(':')
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def prefix_for(ch_task_name: str) -> str:
    """Short prefix used for feature-matrix column names.

    'passage-content' → 'content.'; 'passage-form' → 'form.'.
    Falls back to the full task name if the 'passage-' prefix isn't present.
    """
    short = ch_task_name
    if short.startswith('passage-'):
        short = short[len('passage-'):]
    return f'{short.replace("-", "_")}.'
