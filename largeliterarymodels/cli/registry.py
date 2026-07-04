"""Task → adapter registry for the litmod CLI."""

from .adapters.passage import PassageAdapter
from .adapters.text import TextAdapter
from .adapters.word import WordAdapter
from .adapters.work import WorkAdapter


FAMILIES: dict[str, str] = {
    # passage family fetches text from lltk.passages via ClickHouse
    'PassageContentTask':       'passage',
    'PassageContentTaskV1':     'passage',
    'PassageFormTask':          'passage',
    'PassageTask':              'passage',
    'PassageNarrativityTask':   'passage',
    'PassageSettingTask':       'passage',
    # work family classifies from title/author/year metadata (no CH)
    'GenreTask':                'work',
    'GenreTaskLite':            'work',
    'MajorGenreTask':           'work',
    # text family annotates raw passage text supplied inline (no CH)
    'EmotionTask':              'text',
    'EmotionTaskZh':            'text',
    # word family keys on (word, pos) entries (no CH)
    'TranslationTask':          'word',
    # Future families — register when their adapters exist:
    # 'BibliographyTask':   'work_long',
    # 'CharacterIntroTask': 'character',
}


ADAPTERS = {
    'passage': PassageAdapter,
    'text': TextAdapter,
    'word': WordAdapter,
    'work': WorkAdapter,
}


def resolve(task_name: str):
    """Look up (Task class, adapter instance) for a registered task name."""
    if task_name not in FAMILIES:
        raise SystemExit(
            f"Unknown task: {task_name!r}. Registered: {sorted(FAMILIES)}"
        )
    family = FAMILIES[task_name]
    if family not in ADAPTERS:
        raise SystemExit(
            f"Task {task_name!r} → family {family!r} but no adapter is "
            f"registered for that family. Available: {sorted(ADAPTERS)}"
        )
    from largeliterarymodels import tasks as tasks_mod
    task_cls = getattr(tasks_mod, task_name)
    adapter = ADAPTERS[family]()
    return task_cls, adapter


def list_tasks() -> list[tuple[str, str, bool]]:
    """Return [(task_name, family, adapter_available), ...] sorted by name."""
    out = []
    for name, family in sorted(FAMILIES.items()):
        out.append((name, family, family in ADAPTERS))
    return out
