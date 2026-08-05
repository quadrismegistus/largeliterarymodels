"""Human-annotation pipeline tests — the app had zero coverage until a
live smoke was the only way to answer "does it still work". These are
that smoke, made hermetic: storage round-trip on the jsonl engine, item
discovery from a real task stash, schema-generated form rendering, the
save loop through the app (including the 303-saved vs 200-error-page
distinction the smoke initially misread), and the kappa utility.
"""

import re

import pytest
from pydantic import BaseModel, Field

import largeliterarymodels.task as task_module
from hashstash import HashStash
from largeliterarymodels.llm import _make_key
from largeliterarymodels.task import Task


class Probe(BaseModel):
    is_dialogue: bool = Field(description="passage contains dialogue")
    tone: str = Field(description="overall tone")


@pytest.fixture
def task(tmp_path, monkeypatch):
    """A task whose human_stash (STASH_PATH-derived) lands in tmp and
    whose machine stash is an in-memory HashStash with two items."""
    monkeypatch.setattr(task_module, "STASH_PATH", str(tmp_path / "stash"))

    class T(Task):
        schema = Probe
        system_prompt = "Assess the passage."
        model = "claude-sonnet-4-6"
    T.name = "annotate_test_task"
    t = T()
    t._stash = HashStash(engine="memory").clear()
    for i, (text, val) in enumerate([("'Hello,' she said.", "true"),
                                     ("The moor lay silent.", "false")]):
        k = _make_key(text, t.model, "sys", 0.0, 64, schema_name="Probe",
                      metadata={"item": f"i{i}"})
        t._stash[k] = f'{{"is_dialogue": {val}, "tone": "flat"}}'
    return t


@pytest.fixture
def client(task):
    from starlette.testclient import TestClient
    from largeliterarymodels.annotate import create_app
    return TestClient(create_app(task, annotator="tester_a"))


class TestHumanStash:
    def test_round_trip_latest_wins(self, task):
        hs = task.human_stash("tester_a")
        hs["item-001"] = {"is_dialogue": True, "tone": "wry"}
        hs["item-001"] = {"is_dialogue": False, "tone": "wry"}  # edit
        assert hs["item-001"]["is_dialogue"] is False
        assert len(dict(hs.items())) == 1, "latest per key, not history"
        assert len(hs.df) == 1

    def test_annotators_are_isolated(self, task):
        task.human_stash("a")["k"] = {"is_dialogue": True, "tone": "x"}
        assert dict(task.human_stash("b").items()) == {}


class TestItemDiscovery:
    def test_items_found_and_deduped_by_prompt(self, task):
        from largeliterarymodels.annotate import _get_items
        items = _get_items(task)
        assert len(items) == 2
        assert all(i["llm_model"] == "claude-sonnet-4-6" for i in items)

    def test_multiple_models_collapse_to_one_item(self, task):
        from largeliterarymodels.annotate import _get_items
        k = _make_key("'Hello,' she said.", "gpt-4o-mini", "sys", 0.0, 64,
                      schema_name="Probe", metadata={"item": "i0"})
        task._stash[k] = '{"is_dialogue": false, "tone": "flat"}'
        items = _get_items(task)
        assert len(items) == 2, "same prompt, second model: same item"
        both = next(i for i in items if len(i["llm_results"]) == 2)
        assert set(both["llm_results"]) == {"claude-sonnet-4-6",
                                            "gpt-4o-mini"}


class TestAppLoop:
    def test_index_lists_items(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "annotate" in r.text.lower()

    def test_form_is_generated_from_the_schema(self, client):
        html = client.get("/annotate/0").text
        names = set(re.findall(r'name="([^"]+)"', html))
        assert {"human_is_dialogue", "human_tone"} <= names, \
            "human-side fields must be auto-generated from the schema"
        assert {"llm_is_dialogue", "llm_tone"} <= names, \
            "machine answers render beside them for comparison"

    def test_save_persists_and_redirects(self, task, client):
        html = client.get("/annotate/0").text
        names = set(re.findall(r'name="([^"]+)"', html))
        payload = {n: ("on" if "is_dialogue" in n else "wry")
                   for n in names if n.startswith("human_")}
        r = client.post("/save/0", data=payload, follow_redirects=False)
        assert r.status_code == 303, \
            "303 is the saved path; 200 is the validation-error page"
        [(key, saved)] = task.human_stash("tester_a").items()
        assert saved == {"is_dialogue": True, "tone": "wry"}

    def test_out_of_range_save_redirects_without_writing(self, task,
                                                         client):
        r = client.post("/save/99", data={}, follow_redirects=False)
        assert r.status_code == 303
        assert dict(task.human_stash("tester_a").items()) == {}


class TestAgreement:
    def test_inter_annotator_agreement_shape(self, task):
        from largeliterarymodels.annotate import inter_annotator_agreement
        for key in ("p1", "p2"):
            task.human_stash("a")[key] = {"is_dialogue": True,
                                          "tone": "wry"}
        task.human_stash("b")["p1"] = {"is_dialogue": True, "tone": "wry"}
        task.human_stash("b")["p2"] = {"is_dialogue": False,
                                       "tone": "dry"}
        df = inter_annotator_agreement(task, ["a", "b"])
        assert len(df) > 0
        fields = set(df["field"]) if "field" in df.columns else \
            set(df.index)
        assert {"is_dialogue", "tone"} <= fields
