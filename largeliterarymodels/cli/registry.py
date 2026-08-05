"""Task → adapter registry for the litmod CLI."""

import hashlib
import importlib.util
import os
import sys

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


def _load_task_from_path(spec: str):
    """`path/to/file.py` or `path/to/file.py:ClassName` → (Task class, None).

    First-class support for instruments that live OUTSIDE this package —
    the registry names the package's own exemplar catalog, but research
    tasks belong in research repos (the lltk boundary, applied to the
    CLI), and they should not need registry membership to be
    administered. Only classes DEFINED in the file count: an imported
    base or a task imported for reference is not a candidate, or every
    file would ambiguously "define" Task itself. No adapter is returned
    — adapters map registered families; a file task works with the
    commands that need none (annotate, render) and Python for the rest.
    """
    path, _, cls_name = spec.partition(":")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise SystemExit(f"No such task file: {path!r}")
    from largeliterarymodels.task import Task
    mod_name = (f"litmod_taskfile_"
                f"{hashlib.sha1(path.encode()).hexdigest()[:8]}_"
                f"{os.path.splitext(os.path.basename(path))[0]}")
    spec_obj = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec_obj)
    # Registered before exec so the module's own classes introspect
    # normally (pydantic resolves __module__ at class creation).
    sys.modules[mod_name] = module
    try:
        spec_obj.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(mod_name, None)
        raise SystemExit(f"Could not import {path}: "
                         f"{type(e).__name__}: {e}") from e
    candidates = {name: obj for name, obj in vars(module).items()
                  if isinstance(obj, type) and issubclass(obj, Task)
                  and obj.__module__ == mod_name}
    if cls_name:
        obj = candidates.get(cls_name)
        if obj is None:
            raise SystemExit(
                f"{path} defines no Task subclass named {cls_name!r}. "
                f"Defined there: {sorted(candidates) or '(none)'}")
        return obj, None
    if len(candidates) == 1:
        return next(iter(candidates.values())), None
    if not candidates:
        raise SystemExit(
            f"No Task subclass is DEFINED in {path} (imported classes do "
            f"not count). Subclass largeliterarymodels.task.Task in the "
            f"file, or check the path.")
    raise SystemExit(
        f"{path} defines {len(candidates)} Task subclasses "
        f"({sorted(candidates)}) — pick one: {spec}:{sorted(candidates)[0]}")


def resolve(task_name: str):
    """(Task class, adapter instance) for a registered name — or a
    filesystem path: `path/to/task.py[:ClassName]` loads the class from
    the file, with adapter None (see _load_task_from_path)."""
    if (task_name.endswith(".py") or ".py:" in task_name
            or os.sep in task_name):
        return _load_task_from_path(task_name)
    if task_name not in FAMILIES:
        raise SystemExit(
            f"Unknown task: {task_name!r}. Registered: {sorted(FAMILIES)} "
            f"— or pass a file: path/to/task.py[:ClassName]"
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
