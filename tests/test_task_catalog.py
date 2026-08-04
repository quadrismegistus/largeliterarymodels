"""Catalog-wide invariants: every registered task is well-formed and every
few-shot example validates against its task's schema.

This is the drift detector: a schema change that invalidates an example, an
unregistered class, or a broken lazy import fails here before it fails in a
batch run.
"""

import inspect

import pytest

from largeliterarymodels import tasks as tasks_pkg
from largeliterarymodels.llm import _unwrap_schema
from largeliterarymodels.task import Task

ALL_EXPORT_NAMES = sorted(tasks_pkg._LAZY_IMPORTS)


def _task_classes():
    out = []
    for name in ALL_EXPORT_NAMES:
        obj = getattr(tasks_pkg, name)
        if inspect.isclass(obj) and issubclass(obj, Task) and obj is not Task:
            out.append(obj)
    return out


TASK_CLASSES = _task_classes()


@pytest.mark.parametrize("name", ALL_EXPORT_NAMES)
def test_lazy_export_resolves(name):
    assert getattr(tasks_pkg, name) is not None


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=lambda c: c.__name__)
def test_task_is_well_formed(task_cls):
    assert task_cls.system_prompt, f"{task_cls.__name__} has no system_prompt"
    # schema=None is legitimate for generation-style tasks (OCRCleanTask) and
    # SequentialTasks that parse raw output themselves (SocialNetworkTask);
    # anything schema-based must expose a real Pydantic model.
    if task_cls.schema is not None:
        _is_list, item_schema = _unwrap_schema(task_cls.schema)
        assert hasattr(item_schema, "model_fields"), (
            f"{task_cls.__name__}.schema is not a Pydantic model"
        )


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=lambda c: c.__name__)
def test_examples_validate_against_schema(task_cls):
    """Round-trip every few-shot example through model_validate."""
    for i, (inp, out) in enumerate(task_cls.examples or ()):
        assert isinstance(inp, str), (
            f"{task_cls.__name__}.examples[{i}] input is not a string"
        )
        items = out if isinstance(out, list) else [out]
        for item in items:
            if hasattr(item, "model_dump"):
                type(item).model_validate(item.model_dump())
            else:
                _is_list, item_schema = _unwrap_schema(task_cls.schema)
                item_schema.model_validate(item)


def test_task_names_unique_or_deliberately_shared():
    """Duplicate task names share one stash dir; only the known deliberate
    case (PassageContentTask V1/V2) is allowed.

    Dedupe by class IDENTITY, not by occurrence: a convenience alias
    (JudgeTask = JudgeTaskA) registers the same class under two export
    names, which is one stash dir and no hazard — counting it as a
    collision reported an annotation-mixing risk that did not exist.
    """
    by_name = {}
    for c in TASK_CLASSES:
        by_name.setdefault(c.name, set()).add(c)
    dupes = {n: [k.__name__ for k in ks]
             for n, ks in by_name.items() if len(ks) > 1}
    assert set(dupes) <= {"classify_passage_content"}, (
        f"unexpected shared task names (shared stash dirs): {dupes}"
    )
