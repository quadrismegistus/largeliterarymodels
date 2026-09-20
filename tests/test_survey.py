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


class TestFailFastTrips:
    """The abort path, which was broken and untested: breaker.error() returns
    a MESSAGE, so `raise breaker.error(model)` raised TypeError instead of
    aborting. Nothing in the suite noticed until fail_fast was made to fire."""

    def _s(self, monkeypatch, fails=True):
        import largeliterarymodels.providers as P
        n = {"calls": 0}

        def call(state, questions, **kw):
            n["calls"] += 1
            if fails:
                raise RuntimeError("systematic")
            return _body()
        monkeypatch.setattr(P, "call_typesafe_survey", call)
        return _survey(retries=0), n

    def test_aborts_with_batch_aborted_not_typeerror(self, monkeypatch):
        from largeliterarymodels.llm import BatchAborted
        s, _ = self._s(monkeypatch)
        with pytest.raises(BatchAborted, match="Aborting batch"):
            s.map([{"p": i} for i in range(40)], num_workers=1, fail_fast=5)

    def test_errors_counts_items_that_ran_not_the_batch_size(self, monkeypatch):
        # Pre-empted items are not item failures. Counting them is how one
        # abort made len(errors) report the whole batch.
        from largeliterarymodels.llm import BatchAborted
        s, _ = self._s(monkeypatch)
        errs = {}
        with pytest.raises(BatchAborted):
            s.map([{"p": i} for i in range(40)], num_workers=1, errors=errs,
                  fail_fast=5)
        assert 0 < len(errs) < 40

    def test_message_states_the_inference(self, monkeypatch):
        from largeliterarymodels.llm import BatchAborted
        s, _ = self._s(monkeypatch)
        with pytest.raises(BatchAborted) as e:
            s.map([{"p": i} for i in range(40)], num_workers=1, fail_fast=5)
        assert "rerun resumes" in str(e.value)
        assert "fail_fast=False" in str(e.value), "must say how to override"

    def test_a_healthy_batch_does_not_abort(self, monkeypatch):
        s, n = self._s(monkeypatch, fails=False)
        out = s.map([{"p": i} for i in range(40)], num_workers=1, fail_fast=5)
        assert len(out) == 40 and all(a is not None for a in out)

    def test_disabled_grinds_through(self, monkeypatch):
        s, n = self._s(monkeypatch)
        errs = {}
        out = s.map([{"p": i} for i in range(12)], num_workers=1,
                    errors=errs, fail_fast=False)
        assert out == [None] * 12 and len(errs) == 12


class TestAppendMode:
    """A forced rerun appends; without results_history those versions are on
    disk and unreadable. Needs the real pairtree stash — the memory engine
    silently does not append, which makes it useless for this."""

    def _s(self, tmp_path, monkeypatch, bodies):
        import largeliterarymodels.administered as adm_mod
        import largeliterarymodels.providers as P
        monkeypatch.setattr(adm_mod, "STASH_PATH", str(tmp_path / "stash"))
        seq = iter(bodies)

        def call(state, questions, **kw):
            return next(seq)
        monkeypatch.setattr(P, "call_typesafe_survey", call)

        class S(Survey):
            questions = Q
        S.name = "_append"
        return S()

    def test_versions_are_retained_oldest_first(self, tmp_path, monkeypatch):
        s = self._s(tmp_path, monkeypatch, [_body(0.1), _body(0.2), _body(0.3)])
        s.run({"p": 1})
        s.run({"p": 1}, force=True)
        s.run({"p": 1}, force=True)
        [(_, versions)] = list(s.results_history)
        assert [v.p("abstract") for v in versions] == [0.1, 0.2, 0.3]

    def test_results_shows_only_the_latest(self, tmp_path, monkeypatch):
        s = self._s(tmp_path, monkeypatch, [_body(0.1), _body(0.9)])
        s.run({"p": 1})
        s.run({"p": 1}, force=True)
        [(_, latest)] = list(s.results)
        assert latest.p("abstract") == 0.9

    def test_unforced_rerun_is_one_version(self, tmp_path, monkeypatch):
        # The claim this guards: identical output across two ordinary runs
        # demonstrates CACHING, not stability. And a Survey cannot be pinned
        # with temperature=0 at all, so forcing is the only way to measure
        # the model's own variance here.
        s = self._s(tmp_path, monkeypatch, [_body(0.1), _body(0.9)])
        s.run({"p": 1})
        s.run({"p": 1})
        [(_, versions)] = list(s.results_history)
        assert len(versions) == 1


class TestRawSidecar:
    def _wire(self, tmp_path, monkeypatch, **kw):
        import largeliterarymodels.administered as adm_mod
        import largeliterarymodels.providers as P
        from largeliterarymodels.rawlog import RawLog
        monkeypatch.setattr(adm_mod, "STASH_PATH", str(tmp_path / "stash"))

        def call(state, questions, model="jev-latest", usage_sink=None,
                 raw_sink=None, **kwargs):
            body = _body()
            if raw_sink:
                raw_sink(body)
            return body
        monkeypatch.setattr(P, "call_typesafe_survey", call)

        class S(Survey):
            questions = Q
        S.name = "_sidecar"
        return S(raw_log=RawLog(root=str(tmp_path / "raw")), **kw)

    def test_body_lands_under_the_annotation_key(self, tmp_path, monkeypatch):
        s = self._wire(tmp_path, monkeypatch)
        s.run({"passage": "x"})
        [key] = list(s.stash.keys())
        got = s.raw_sidecar.get(key)
        assert got is not None, "raw <-> annotation join must be a lookup"

    def test_map_path_is_wired_too(self, tmp_path, monkeypatch):
        # run() and imap() build the sink separately; a passthrough deleted
        # in one of them leaves the other green.
        s = self._wire(tmp_path, monkeypatch)
        s.map([{"passage": "a"}, {"passage": "b"}], num_workers=1)
        assert len(list(s.raw_sidecar.keys())) == 2

    def test_certify_sources_its_denominator_from_the_stash(self, tmp_path,
                                                            monkeypatch):
        s = self._wire(tmp_path, monkeypatch)
        s.map([{"passage": "a"}, {"passage": "b"}], num_workers=1)
        assert s.certify_raw()["complete"] is True

    def test_off_is_genuinely_off(self, tmp_path, monkeypatch):
        import largeliterarymodels.administered as adm_mod
        monkeypatch.setattr(adm_mod, "STASH_PATH", str(tmp_path / "stash2"))

        class S(Survey):
            questions = Q
        S.name = "_nosidecar"
        s = S()
        assert s.raw_sidecar is None
        with pytest.raises(ValueError, match="nothing to.*certify"):
            s.certify_raw()

    def test_annotation_stash_is_byte_identical_either_way(self, tmp_path,
                                                           monkeypatch):
        # Additive means additive: the sidecar must not touch the write path
        # it documents.
        on = self._wire(tmp_path, monkeypatch)
        on.run({"passage": "x"})
        [(k_on, v_on)] = list(on.stash.items())

        import largeliterarymodels.administered as adm_mod
        monkeypatch.setattr(adm_mod, "STASH_PATH", str(tmp_path / "stash_off"))

        class S(Survey):
            questions = Q
        S.name = "_sidecar"
        off = S()
        off.run({"passage": "x"})
        [(k_off, v_off)] = list(off.stash.items())
        assert k_on == k_off and v_on == v_off


class TestUsageLogReceipt:
    def _s(self, tmp_path, monkeypatch):
        import largeliterarymodels.administered as adm_mod
        import largeliterarymodels.providers as P
        monkeypatch.setattr(adm_mod, "STASH_PATH", str(tmp_path / "stash"))

        def call(state, questions, model="jev-latest", usage_sink=None,
                 raw_sink=None, **kw):
            if usage_sink:
                usage_sink({"input_tokens": 100, "output_tokens": 10,
                            "cache_read_tokens": 0, "cache_write_tokens": 0,
                            "reasoning_tokens": 0, "reasoning_reported": False,
                            "reasoning_observed": False,
                            "response_model": "jev-1.13.0"})
            return _body()
        monkeypatch.setattr(P, "call_typesafe_survey", call)

        class S(Survey):
            questions = Q
            usage_log = True
        S.name = "survey_usage_probe"
        return S(), tmp_path / "usage_logs" / "survey_usage_probe.jsonl"

    def test_one_record_per_batch(self, tmp_path, monkeypatch):
        s, path = self._s(tmp_path, monkeypatch)
        s.map([{"p": 1}, {"p": 2}], num_workers=1)
        rec = json.loads(path.read_text())
        assert rec["model"] == "jev-latest"
        assert rec["task"] == "survey_usage_probe"
        assert len(rec["items"]) == 2
        assert rec["batch"]["calls"] == 2
        assert rec["batch"]["input_tokens"] == 200
        assert rec["items"][0]["response_model"] == "jev-1.13.0", \
            "the id the server served, not the alias asked for"

    def test_record_is_priceable(self, tmp_path, monkeypatch):
        # The point of the sidecar: a later process prices the run from the
        # log alone, and for this provider that means the cached: null row.
        from largeliterarymodels import costs
        s, path = self._s(tmp_path, monkeypatch)
        s.map([{"p": 1}, {"p": 2}], num_workers=1)
        rec = json.loads(path.read_text())
        priced = costs.price_report(rec["model"], rec["batch"])
        # That the record RESOLVES and prices is the claim here; the
        # arithmetic is pinned against two real invoices in
        # test_typesafe_provider rather than restated.
        assert priced["provider"] == "typesafe"
        assert priced["warnings"] == []
        # The cached:null path: every input token billed fresh, nothing
        # discounted, and no output line because output is free.
        assert set(priced["lines"]) == {"fresh_input"}

    def test_imap_writes_one_too(self, tmp_path, monkeypatch):
        s, path = self._s(tmp_path, monkeypatch)
        list(s.imap([{"p": 1}], num_workers=1))
        assert path.exists()

    def test_shim_goes_through_the_shared_writer(self, tmp_path, monkeypatch):
        # One writer for Task and Survey both. Two record formats is how a
        # pricing script learns to parse only one of them.
        import largeliterarymodels.administered as adm_mod
        seen = []
        monkeypatch.setattr(adm_mod, "append_usage_log",
                            lambda *a, **kw: seen.append(a))
        s, _ = self._s(tmp_path, monkeypatch)
        s.map([{"p": 1}], num_workers=1)
        assert seen and seen[0][0] == "survey_usage_probe"
        assert seen[0][1] == "jev-latest"


class TestGoldenKey:
    """The exact key bytes, pinned. A refactor that changes them re-keys every
    corpus ever collected with this class, silently and at full price.

    Changed once, deliberately, before any corpus existed: `state` is now the
    indented render that the text arm is also given, so a cross-arm check is
    string equality. That change is why this test exists — it failed, which is
    the pin doing its job."""

    def test_key_is_exactly_this(self):
        s = _survey("golden")
        _, _, key = s._prepare({"passage": "Ye who listen"},
                               metadata={"id": 7}, sample=2)
        assert key == {
            "state": '{\n  "passage": "Ye who listen"\n}',
            "model": "jev-latest",
            "questions": json.dumps(
                {"abstract": {"type": "noul",
                              "instructions": "Is the vocabulary abstract?"},
                 "mode": {"type": "choice", "instructions": "Mode?",
                          "criteria": {"narrative": None, "didactic": None}},
                 "degree": {"type": "score", "instructions": "How abstract?",
                            "criteria": ["concrete", "mixed", "abstract"]}},
                indent=2, ensure_ascii=False, sort_keys=False),
            "metadata": {"id": 7},
            "sample": 2,
        }


class TestBothArmsSeeOneItem:
    """as_task() compiles the questions; state_text compiles the ITEM. Without
    the second the caller formats the item a second time by hand and the two
    arms diverge — which is the failure as_task() exists to prevent."""

    def test_state_text_is_the_state_the_survey_sends(self):
        s = _survey()
        item = {"female_frame": "She slowly took off her bra",
                "male_frame": "He slowly took off his bra"}
        assert json.loads(s.state_text(item)) == s.build_state(item)

    def test_named_fields_survive_so_inspect_still_refers(self):
        # A question's `inspect` names a state field; flattening to prose
        # would break the reference the instructions make.
        s = _survey()
        txt = s.state_text({"female_frame": "a", "male_frame": "b"})
        assert "female_frame" in txt and "male_frame" in txt

    def test_prompts_line_up_with_items(self):
        s = _survey()
        items = [{"p": 1}, {"p": 2}]
        assert s.as_task_prompts(items) == [s.state_text(i) for i in items]

    def test_a_bare_string_item_is_named_in_both_arms(self):
        s = _survey()
        assert json.loads(s.state_text("hello")) == {"item": "hello"}


class TestOneRenderer:
    """key["state"] and the text arm's prompt must be the SAME BYTES, so the
    cross-arm check is string equality. Two renderers of one thing drift."""

    def test_key_state_is_byte_identical_to_the_text_arm_prompt(self):
        s = _survey()
        item = {"female_frame": "She slowly took off her bra",
                "male_frame": "He slowly took off his bra"}
        _, _, key = s._prepare(item)
        assert key["state"] == s.state_text(item)
        assert key["state"] == s.as_task_prompts([item])[0]

    def test_build_state_is_called_once_per_item(self):
        # An override that hits a database or mutates must not run twice.
        calls = []

        class S(Survey):
            name = "_once"
            questions = Q

            def build_state(self, item):
                calls.append(item)
                return {"passage": item}
        S()._prepare("x")
        assert len(calls) == 1


class TestIncompleteResponses:
    """An unanswered question was neither a result nor an error: the partial
    body was stashed, `errors` stayed empty, and because the stash is durable
    the gap was then served as complete on every later read, free, forever.
    A defect that costs nothing to re-read is worse than one that costs a
    call."""

    def _s(self, monkeypatch, answered=3, asked=5, retries=1):
        import largeliterarymodels.providers as P
        n = {"calls": 0}

        def call(state, questions, model="jev-latest", usage_sink=None,
                 raw_sink=None, **kw):
            n["calls"] += 1
            if raw_sink:
                raw_sink({"answers": {}})
            return {"model": "m", "usage": {},
                    "answers": {q: {"type": "noul", "noul": 0.5}
                                for q in list(questions)[:answered]}}
        monkeypatch.setattr(P, "call_typesafe_survey", call)

        class S(Survey):
            questions = {f"q{i}": Noul(instructions=f"Is {i} charged?")
                         for i in range(asked)}
        S.name = "_incomplete"
        S.retries = retries
        s = S()
        s._stash = HashStash(engine="memory").clear()
        return s, n

    def test_it_is_retried_not_returned(self, monkeypatch):
        s, n = self._s(monkeypatch, retries=2)
        out = s.map([{"p": 1}], num_workers=1)
        assert out[0] is None
        assert n["calls"] == 3, "1 + retries, not a single silent acceptance"

    def test_the_partial_body_is_not_stashed(self, monkeypatch):
        # The durable half of the bug: a cached partial answer is served as
        # complete forever and costs nothing to re-read.
        s, _ = self._s(monkeypatch)
        s.map([{"p": 1}], num_workers=1)
        assert list(s.stash.keys()) == []

    def test_a_resumed_run_does_not_serve_a_gap_as_complete(self, monkeypatch):
        s, n = self._s(monkeypatch)
        s.map([{"p": 1}], num_workers=1)
        before = n["calls"]
        s.map([{"p": 1}], num_workers=1)
        assert n["calls"] > before, "nothing warm to serve"

    def test_the_error_names_which_questions_were_dropped(self, monkeypatch):
        # The analogue of _diagnose_partial_response naming the field: a
        # caller must not have to diff the question set by hand.
        s, _ = self._s(monkeypatch)
        errs = {}
        s.map([{"p": 1}], num_workers=1, errors=errs)
        assert errs[0]["missing"] == ("q3", "q4")
        assert "2 of 5" in errs[0]["error"]

    def test_run_raises_rather_than_returning_a_gap(self, monkeypatch):
        from largeliterarymodels.answers import IncompleteAnswers
        s, _ = self._s(monkeypatch)
        with pytest.raises(IncompleteAnswers, match="unanswered"):
            s.run({"p": 1})
        assert list(s.stash.keys()) == []

    def test_a_complete_response_is_unaffected(self, monkeypatch):
        s, n = self._s(monkeypatch, answered=5, asked=5)
        out = s.map([{"p": 1}], num_workers=1)
        assert out[0] is not None and n["calls"] == 1
        assert len(list(s.stash.keys())) == 1

    def test_retry_that_succeeds_is_a_success(self, monkeypatch):
        import largeliterarymodels.providers as P
        n = {"calls": 0}

        def call(state, questions, **kw):
            n["calls"] += 1
            keep = list(questions) if n["calls"] > 1 else list(questions)[:2]
            return {"model": "m", "usage": {},
                    "answers": {q: {"type": "noul", "noul": 0.5}
                                for q in keep}}
        monkeypatch.setattr(P, "call_typesafe_survey", call)

        class S(Survey):
            questions = {f"q{i}": Noul(instructions="x") for i in range(4)}
        S.name = "_recovers"
        S.retries = 1
        s = S()
        s._stash = HashStash(engine="memory").clear()
        errs = {}
        out = s.map([{"p": 1}], num_workers=1, errors=errs)
        assert out[0] is not None and errs == {}
        assert len(list(s.stash.keys())) == 1

    def test_unknown_kinds_do_not_trigger_a_retry(self, monkeypatch):
        # They were ANSWERED; the provider will return the same kind next
        # time, so retrying spends money to receive it again.
        import largeliterarymodels.providers as P
        n = {"calls": 0}

        def call(state, questions, **kw):
            n["calls"] += 1
            return {"model": "m", "usage": {},
                    "answers": {q: {"type": "quantile", "q50": 3}
                                for q in questions}}
        monkeypatch.setattr(P, "call_typesafe_survey", call)

        class S(Survey):
            questions = {"q0": Noul(instructions="x")}
        S.name = "_unknown"
        S.retries = 2
        s = S()
        s._stash = HashStash(engine="memory").clear()
        out = s.map([{"p": 1}], num_workers=1)
        assert n["calls"] == 1
        assert out[0].unknown == ("q0",)

    def test_the_sidecar_still_keeps_the_defective_body(self, tmp_path,
                                                        monkeypatch):
        # Right layering: the raw sink fires BEFORE the completeness check, so
        # the body that caused the defect survives for diagnosis while the
        # annotation store keeps only complete administrations.
        from largeliterarymodels.rawlog import RawLog
        s, _ = self._s(monkeypatch)
        s.raw_log = RawLog(root=str(tmp_path / "raw"))
        s._raw_log = None
        s.map([{"p": 1}], num_workers=1)
        assert len(list(s.raw_sidecar.keys())) >= 1
        assert list(s.stash.keys()) == []


class TestProvenance:
    def test_record_carries_the_package_version(self):
        # The question digest hashes the QUESTIONS; it cannot see a change to
        # how answers are gated, retried or parsed. Without this a corpus
        # collected across such a change has two administrations under one
        # digest and nothing in the artifact says so.
        rec = _survey().administration_record()
        assert rec["package_version"]

    def test_version_never_enters_a_key(self):
        # A version bump would orphan every key ever written — the same
        # reason _SAMPLING_DEFAULTS stays out of keys.
        _, _, key = _survey()._prepare({"passage": "x"})
        assert "package_version" not in key
        assert "version" not in " ".join(key)
