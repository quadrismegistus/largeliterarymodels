"""The Survey class: keys, instrument identity, orchestration, read-back."""
import json

import pytest
from hashstash import HashStash

from largeliterarymodels import survey as survey_mod
from largeliterarymodels.answers import Answers
from largeliterarymodels.llm import _make_survey_key
from largeliterarymodels.questions import Choice, Noul, Score
from largeliterarymodels.survey import Survey

Q = {
    "abstract": Noul(instructions="Is the vocabulary abstract?"),
    "mode": Choice(instructions="Mode?",
                   criteria={"narrative": None, "didactic": None}),
    "degree": Score(instructions="How abstract?",
                    criteria=["concrete", "mixed", "abstract"]),
}


def _body(p=0.9):
    return {"model": "jev-1.13.0",
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "answers": {
                "abstract": {"type": "noul", "noul": p},
                "mode": {"type": "choice", "choice": "didactic",
                         "confidence": 0.8,
                         "probabilities": {"narrative": 0.2, "didactic": 0.8}},
                "degree": {"type": "score", "score": 1.9, "confidence": 0.9,
                           "legend": {"0": "concrete", "1": "mixed",
                                      "2": "abstract"},
                           "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}},
            }}


@pytest.fixture
def fake_provider(monkeypatch):
    calls = []

    def call(state, questions, model="jev-latest", usage_sink=None,
             raw_sink=None, **kw):
        calls.append({"state": state, "questions": questions, "model": model})
        body = _body()
        if usage_sink:
            usage_sink({"input_tokens": 100, "output_tokens": 10,
                        "cache_read_tokens": 0, "cache_write_tokens": 0,
                        "reasoning_tokens": 0, "reasoning_reported": False,
                        "reasoning_observed": False,
                        "response_model": "jev-1.13.0"})
        if raw_sink:
            raw_sink(body)
        return body

    import largeliterarymodels.providers as P
    monkeypatch.setattr(P, "call_typesafe_survey", call)
    return calls


def _survey(name="_t", **kw):
    class S(Survey):
        questions = Q
    S.name = name
    for k, v in kw.items():
        setattr(S, k, v)
    s = S()
    s._stash = HashStash(engine="memory").clear()
    return s


class TestKey:
    def test_carries_state_model_questions(self):
        s = _survey()
        _, _, key = s._prepare({"passage": "x"})
        assert set(key) == {"state", "model", "questions"}
        assert json.loads(key["state"]) == {"passage": "x"}
        assert key["model"] == "jev-latest"

    @pytest.mark.parametrize("absent", ["temperature", "max_tokens",
                                        "thinking", "top_p", "sampling",
                                        "schema"])
    def test_omits_what_was_never_sent(self, absent):
        # A field present in a key means that value governed the call. System
        # One accepts none of these, so carrying any would assert a pin the
        # API has no way to honour.
        _, _, key = _survey()._prepare({"passage": "x"})
        assert absent not in key

    def test_metadata_is_identity(self):
        s = _survey()
        _, _, a = s._prepare({"passage": "x"})
        _, _, b = s._prepare({"passage": "x"}, metadata={"id": 1})
        assert a != b and b["metadata"] == {"id": 1}

    def test_sample_separates_replicates(self):
        s = _survey()
        _, _, a = s._prepare({"passage": "x"}, sample=0)
        _, _, b = s._prepare({"passage": "x"}, sample=1)
        assert a != b
        _, _, none = s._prepare({"passage": "x"})
        assert "sample" not in none, "omitted when None, so nothing re-keys"

    def test_question_wording_change_rekeys(self):
        # The questionnaire-edit rule, on the Survey path: a description edit
        # is an edit to the instrument and must not serve a warm hit.
        a = _survey()._prepare({"passage": "x"})[2]
        s2 = _survey()
        s2.questions = {**Q, "abstract": Noul(instructions="Is it abstract? ")}
        assert s2._prepare({"passage": "x"})[2] != a

    def test_key_questions_are_the_administered_bytes(self):
        s = _survey()
        _, wire, key = s._prepare({"passage": "x"})
        assert json.loads(key["questions"]) == wire

    def test_state_change_rekeys(self):
        s = _survey()
        assert (s._prepare({"passage": "a"})[2]
                != s._prepare({"passage": "b"})[2])

    def test_builder_is_shared_with_llm(self):
        assert _make_survey_key("s", "m", "q") == {
            "state": "s", "model": "m", "questions": "q"}


class TestState:
    def test_mapping_passes_through(self):
        assert _survey().build_state({"passage": "x"}) == {"passage": "x"}

    def test_bare_string_is_named_not_sent_raw(self):
        # An unnamed blob is the one shape a question's `inspect` cannot
        # refer to.
        assert _survey().build_state("hello") == {"item": "hello"}


class TestInstrument:
    def test_text_is_the_wire_form(self):
        s = _survey()
        _, wire, _ = s._prepare({"passage": "x"})
        assert json.loads(s.instrument_text()) == wire

    def test_digest_stable_and_wording_sensitive(self):
        s, t = _survey(), _survey()
        assert s.instrument_sha256() == t.instrument_sha256()
        t.questions = {**Q, "abstract": Noul("Is it abstract?!")}
        assert t.instrument_sha256() != s.instrument_sha256()

    def test_administration_record_says_nothing_was_pinnable(self):
        rec = _survey().administration_record()
        assert rec["decoder_parameters"] is None
        assert "nothing is pinnable" in rec["decoder_parameters_note"]
        assert rec["n_questions"] == 3
        assert rec["instrument_sha256"]


class TestPerItemQuestions:
    def _s(self):
        class PerItem(Survey):
            name = "_per_item"

            def build_questions(self, item):
                return {w: Noul(instructions=f"Is {w} charged?")
                        for w in item["words"]}
        s = PerItem()
        s._stash = HashStash(engine="memory").clear()
        return s

    def test_instrument_methods_refuse_rather_than_fabricate(self):
        s = self._s()
        assert not s.has_static_instrument
        for fn in (s.instrument_text, s.instrument_sha256):
            with pytest.raises(ValueError, match="per item"):
                fn()

    def test_record_says_so_instead_of_a_digest(self):
        rec = self._s().administration_record()
        assert rec["per_item_questions"] is True
        assert rec["n_questions"] is None
        assert "instrument_sha256" not in rec

    def test_key_still_carries_the_questions_actually_asked(self):
        s = self._s()
        _, _, key = s._prepare({"words": ["beat", "kiss"]})
        assert set(json.loads(key["questions"])) == {"beat", "kiss"}

    def test_answer_keys_cannot_drift_from_what_was_asked(self, monkeypatch):
        # The structural property: the response is keyed by the ids we sent,
        # so the drop/invent/dupe/re-case class Task's list-field retry
        # exists for cannot occur.
        s = self._s()
        words = ["beat", "kiss", "Handcuff"]

        def call(state, questions, model="jev-latest", usage_sink=None,
                 raw_sink=None, **kw):
            return {"model": "m", "usage": {},
                    "answers": {q: {"type": "noul", "noul": 0.5}
                                for q in questions}}
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "call_typesafe_survey", call)
        a = s.run({"words": words})
        assert set(a) == set(words) and a.missing == ()


class TestRun:
    def test_returns_typed_answers(self, fake_provider):
        a = _survey().run({"passage": "x"})
        assert isinstance(a, Answers)
        assert a.p("abstract") == 0.9
        assert a["mode"].choice == "didactic"
        assert a["degree"].normalized == pytest.approx(0.95)

    def test_second_call_is_a_cache_hit(self, fake_provider):
        s = _survey()
        s.run({"passage": "x"})
        s.run({"passage": "x"})
        assert len(fake_provider) == 1

    def test_force_recomputes_and_appends(self, fake_provider):
        s = _survey()
        s.run({"passage": "x"})
        s.run({"passage": "x"}, force=True)
        assert len(fake_provider) == 2

    def test_metadata_changes_the_cell(self, fake_provider):
        s = _survey()
        s.run({"passage": "x"})
        s.run({"passage": "x"}, metadata={"id": 1})
        assert len(fake_provider) == 2, "metadata is part of the administration"

    def test_stash_holds_the_raw_body(self, fake_provider):
        s = _survey()
        s.run({"passage": "x"})
        [(_, stored)] = list(s.stash.items())
        assert stored["answers"]["abstract"]["noul"] == 0.9, \
            "raw, so a change to how an answer is read costs no call"


class TestMap:
    ITEMS = [{"passage": "a"}, {"passage": "b"}, {"passage": "c"}]

    def test_order_preserved(self, fake_provider):
        out = _survey().map(self.ITEMS, num_workers=2)
        assert len(out) == 3 and all(isinstance(a, Answers) for a in out)

    def test_warm_half_is_not_recalled(self, fake_provider):
        s = _survey()
        s.run(self.ITEMS[0])
        s.map(self.ITEMS, num_workers=1)
        assert len(fake_provider) == 3, "one warm, two cold"

    def test_usage_accumulates(self, fake_provider):
        s = _survey()
        s.map(self.ITEMS, num_workers=1)
        assert s.usage.report()["calls"] == 3

    def test_per_item_usage(self, fake_provider):
        s, pu = _survey(), {}
        s.map(self.ITEMS, num_workers=1, per_item_usage=pu)
        assert set(pu) == {0, 1, 2}
        assert all(r["input_tokens"] == 100 for r in pu.values())

    def test_failures_are_recorded_and_positional(self, monkeypatch):
        import largeliterarymodels.providers as P
        n = {"i": 0}

        def call(state, questions, **kw):
            n["i"] += 1
            if json.dumps(state).find('"b"') >= 0:
                raise RuntimeError("boom")
            return _body()
        monkeypatch.setattr(P, "call_typesafe_survey", call)
        s, errs = _survey(retries=0), {}
        out = s.map(self.ITEMS, num_workers=1, errors=errs)
        assert out[1] is None and set(errs) == {1}
        assert errs[1]["error"].startswith("RuntimeError")
        assert errs[1]["index"] == 1

    def test_metadata_list_reaches_the_key(self, fake_provider):
        s = _survey()
        s.map(self.ITEMS, num_workers=1,
              metadata_list=[{"id": i} for i in range(3)])
        metas = [k["metadata"] for k, _ in s.stash.items()]
        assert sorted(m["id"] for m in metas) == [0, 1, 2]

    def test_fail_fast_junk_raises(self, fake_provider):
        with pytest.raises(TypeError, match="fail_fast"):
            _survey().map(self.ITEMS, fail_fast="lots")


class TestReadBack:
    def test_results_yields_answers(self, fake_provider):
        s = _survey()
        s.map([{"passage": "a"}, {"passage": "b"}], num_workers=1)
        got = list(s.results)
        assert len(got) == 2 and all(isinstance(a, Answers) for _, a in got)

    def test_df_is_continuous_not_thresholded(self, fake_provider):
        s = _survey()
        s.map([{"passage": "a"}], num_workers=1, metadata_list=[{"id": "x"}])
        df = s.df
        assert df.loc[0, "abstract"] == 0.9, "the probability, not a bool"
        assert df.loc[0, "degree"] == 1.9, "the position, not a level"
        assert df.loc[0, "mode"] == "didactic"
        assert df.loc[0, "degree__conf"] == 0.9
        assert df.loc[0, "meta_id"] == "x"

    def test_df_has_no_bool_columns(self, fake_provider):
        s = _survey()
        s.map([{"passage": "a"}], num_workers=1)
        assert not any(str(d) == "bool" for d in s.df.dtypes)


class TestRepr:
    def test_names_the_instrument_size(self, fake_provider):
        assert "questions=3" in repr(_survey())


class TestAsTask:
    def _s(self):
        class S(Survey):
            name = "_cross"
            questions = {
                "abstract": Noul(instructions="Is it abstract?",
                                 criteria={"true": {"what": "qualities"},
                                           "false": {"what": "objects"}}),
                "mode": Choice(instructions="Mode?",
                               criteria={"narrative": "recounts events",
                                         "didactic": None}),
                "degree": Score(instructions="How abstract?",
                                criteria=["concrete", "mixed", "abstract"]),
            }
        return S()

    def test_types_compile(self):
        t = self._s().as_task()
        f = t.schema.model_fields
        assert f["abstract"].annotation is bool
        assert f["degree"].annotation is int
        assert set(getattr(f["mode"].annotation, "__args__", ())) == {
            "narrative", "didactic"}

    def test_score_bounds_come_from_the_level_count(self):
        t = self._s().as_task()
        meta = str(t.schema.model_fields["degree"].metadata)
        assert "0" in meta and "2" in meta
        with pytest.raises(Exception):
            t.schema.model_validate({"abstract": True, "mode": "narrative",
                                     "degree": 3})

    def test_question_ids_are_the_aliases_the_model_is_asked_for(self):
        # So the two arms of a comparison line up without a translation step.
        t = self._s().as_task()
        assert set(t.schema.model_json_schema()["properties"]) == {
            "abstract", "mode", "degree"}

    def test_criteria_reach_the_field_description(self):
        t = self._s().as_task()
        d = t.schema.model_fields["abstract"].description
        assert "TRUE when" in d and "FALSE when" in d and "qualities" in d

    def test_score_levels_are_numbered_in_the_description(self):
        d = self._s().as_task().schema.model_fields["degree"].description
        assert "0 = concrete" in d and "2 = abstract" in d

    def test_instrument_text_is_a_real_prompt(self):
        t = self._s().as_task()
        assert "independent questions" in t.instrument_text()
        assert t.instrument_sha256()

    def test_model_override(self):
        assert self._s().as_task(model="claude-sonnet-5").model \
            == "claude-sonnet-5"

    def test_per_item_questions_need_an_item(self):
        class PerItem(Survey):
            name = "_pi"

            def build_questions(self, item):
                return {w: Noul(instructions=f"Is {w} charged?")
                        for w in item["words"]}
        s = PerItem()
        with pytest.raises((TypeError, ValueError, KeyError)):
            s.as_task()
        t = s.as_task(item={"words": ["beat", "kiss"]})
        assert set(t.schema.model_json_schema()["properties"]) == {
            "beat", "kiss"}

    def test_non_identifier_ids_are_sanitised_but_keep_their_alias(self):
        class Colons(Survey):
            name = "_colons"
            questions = {"beat::scene": Score(instructions="x",
                                              criteria=["a", "b"]),
                         "beat::kind": Choice(instructions="y",
                                              criteria={"NONE": None,
                                                        "VIOLENT": None})}
        t = Colons().as_task()
        assert list(t.schema.model_fields) == ["beat__scene", "beat__kind"]
        obj = t.schema.model_validate({"beat::scene": 1,
                                       "beat::kind": "VIOLENT"})
        assert obj.model_dump(by_alias=True) == {"beat::scene": 1,
                                                 "beat::kind": "VIOLENT"}

    def test_alias_collision_refused_rather_than_merged(self):
        class Bad(Survey):
            name = "_bad"
            questions = {"a::b": Noul(instructions="x"),
                         "a__b": Noul(instructions="y")}
        with pytest.raises(ValueError, match="both sanitise"):
            Bad().as_task()

    def test_raw_dict_questions_refuse_to_compile(self):
        class Raw(Survey):
            name = "_raw"
            questions = {"q": {"type": "noul", "instructions": "x",
                               "weight": 2}}
        with pytest.raises(ValueError, match="raw dict"):
            Raw().as_task()

    def test_the_crossing_loses_the_probability_and_says_so(self):
        # The documented cost: a text model returns a bool where a Noul
        # returns a probability. If this ever stops being true the docstring
        # is wrong and someone should notice here.
        t = self._s().as_task()
        assert t.schema.model_fields["abstract"].annotation is bool
        assert "WHAT IS LOST" in Survey.as_task.__doc__
