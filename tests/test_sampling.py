"""Decoder parameters (top_p and neighbours), replicate indices, and the
receipts that make a zero-call run legible.

The defect these pin: top_p was not a parameter of this package, and the
absence was not neutral. **kwargs reached call_deepseek and were forwarded
verbatim; call_anthropic, call_openai, call_google and call_local each
declared **kwargs and discarded them. A caller pinning top_p across four
vendors pinned one, silently, with identical cache keys either way.
"""

import json
from unittest.mock import patch

import pytest

import largeliterarymodels.providers as P
from largeliterarymodels.llm import (LLM, UsageTracker, _legacy_key_kwargs,
                                     _make_key, _sampling_fingerprint)
from tests.test_regressions import FakeStash, Out


DS = "deepseek/deepseek-v4-pro"


class TestKeyRecordsWhatGoverned:
    def test_unpinned_keys_are_byte_identical_to_the_old_schema(self):
        """The whole point of omit-when-unset: every key written before
        decoder parameters existed must still be reachable."""
        old = {"prompt": "p", "model": "m", "system_prompt": "s",
               "temperature": 0.0, "max_tokens": 4096}
        assert _make_key("p", "m", "s", 0.0, 4096) == old
        assert _make_key("p", "m", "s", 0.0, 4096,
                         sampling=None, sample=None) == old
        assert _make_key("p", "m", "s", 0.0, 4096, sampling={}) == old

    def test_pinned_values_are_flat_fields(self):
        """Flat, not a nested blob: 'which fields were pinned' has to be
        readable off the key, because that is the claim a methods note
        makes."""
        key = _make_key("p", "m", "s", 0.0, 4096,
                        sampling={"top_p": 0.9}, sample=2)
        assert key["top_p"] == 0.9
        assert key["sample"] == 2

    def test_a_value_that_cannot_govern_is_not_recorded(self):
        """DeepSeek's thinking mode accepts top_p and ignores it. A key
        recording top_p: 0.9 there asserts a pin that never applied — the
        same false methods claim effective_temperature exists to prevent."""
        _, _, governing = _sampling_fingerprint(
            DS, 0.0, {"thinking": None}, {"top_p": 0.9})
        assert governing == {}
        key = _make_key("p", DS, "s", None, 4096, sampling=governing)
        assert "top_p" not in key

        _, _, governing = _sampling_fingerprint(DS, 0.0, {}, {"top_p": 0.9})
        assert governing == {"top_p": 0.9}   # thinking disabled by default

    def test_unmeasured_families_raise_rather_than_guess(self):
        """Neither guess is safe: assume-governs plants a false pin,
        assume-ignored shares one cell between differently-sampled runs.
        Under ignorance the schema refuses."""
        for model in ("claude-sonnet-4-6", "gpt-5.4-mini",
                      "gemini-3.6-flash", "lmstudio/qwen3.5-35b"):
            with pytest.raises(P.SamplingUnmeasured):
                _sampling_fingerprint(model, 0.0, {}, {"top_p": 0.9})

    def test_the_raise_names_the_parameter_and_the_fix(self):
        with pytest.raises(P.SamplingUnmeasured) as e:
            P.effective_sampling("claude-sonnet-4-6", {"top_p": 0.9})
        msg = str(e.value)
        assert "top_p" in msg and "measured" in msg

    def test_legacy_fallback_is_off_once_anything_new_is_pinned(self):
        """A pre-schema entry cannot be a match for a pinned call: old code
        sent no decoder parameter and had no replicate index."""
        assert _legacy_key_kwargs("claude-cli/opus", 0.0, None, None) == \
            {"temperature": 0.0}
        assert _legacy_key_kwargs("claude-cli/opus", 0.0, None, None,
                                  {"top_p": 0.9}, None) is None
        assert _legacy_key_kwargs("claude-cli/opus", 0.0, None, None,
                                  None, 0) is None


class TestTheValueReachesTheWire:
    def _client(self, seen):
        class _Resp:
            model = DS.split("/")[-1]
            usage = None
            choices = [type("C", (), {"message": type("M", (), {
                "content": '{"x": 1}'})})]
        return type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {
                "create": staticmethod(
                    lambda **kw: (seen.append(kw), _Resp())[1])})})})

    def test_sync_sends_it(self, monkeypatch):
        seen = []
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        monkeypatch.setattr(P, "_cached_client",
                            lambda k, f: self._client(seen))
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        P.call_deepseek("hi", model=DS, sampling={"top_p": 0.9})
        assert seen[0]["top_p"] == 0.9

    def test_batch_sends_the_same_thing(self):
        """One filter for both transports. Two request builders is how a
        50,000-item batch ends up sampled differently from the probe that
        preceded it."""
        body, dropped = P.openai_request_body(
            "deepseek", DS, P.openai_messages("hi"),
            sampling={"top_p": 0.9})
        assert body["top_p"] == 0.9 and dropped == ()

    def test_an_ignored_value_is_reported_not_sent(self, monkeypatch):
        seen = []
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        monkeypatch.setattr(P, "_cached_client",
                            lambda k, f: self._client(seen))
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        monkeypatch.setattr(P, "_WARNED_NO_SAMPLING", set())
        usage = []
        P.call_deepseek("hi", model=DS, thinking=None,
                        sampling={"top_p": 0.9}, usage_sink=usage.append)
        assert "top_p" not in seen[0]
        assert "top_p" in usage[0]["dropped_params"]

    def test_strict_params_escalates_an_ignored_value(self, monkeypatch):
        monkeypatch.setenv("LITMOD_STRICT_PARAMS", "1")
        monkeypatch.setattr(P, "_WARNED_NO_SAMPLING", set())
        with pytest.raises(P.DroppedParameterError):
            P.effective_sampling(DS, {"top_p": 0.9}, thinking=None)


class TestSilentDropsAreNowAudible:
    """The original defect, from both directions."""

    def test_a_bare_sampling_kwarg_is_refused_everywhere(self):
        """On DeepSeek a bare top_p WOULD reach the API — and would not
        reach the key. That is the poisoning half of the same bug."""
        for model in (DS, "claude-sonnet-4-6", "gpt-5.4-mini",
                      "gemini-3.6-flash", "lmstudio/qwen3.5-35b"):
            fn = P.route_provider(model)
            with pytest.raises(ValueError, match="sampling="):
                fn("hi", model=model, top_p=0.9)

    def test_swallowed_kwargs_are_reported(self, monkeypatch):
        """Four of five routes accepted arbitrary kwargs and dropped them on
        the floor. Going nowhere is now audible."""
        monkeypatch.setattr(P, "_WARNED_SWALLOWED", set())
        names = P.swallowed_kwargs("anthropic", "claude-sonnet-4-6",
                                   {"nonesuch": 1, "cache_ttl": "1h"})
        assert names == ("nonesuch",)   # cache_ttl is framework-internal

    def test_swallowed_kwargs_raise_under_strict(self, monkeypatch):
        monkeypatch.setenv("LITMOD_STRICT_PARAMS", "1")
        monkeypatch.setattr(P, "_WARNED_SWALLOWED", set())
        with pytest.raises(P.DroppedParameterError):
            P.swallowed_kwargs("anthropic", "m", {"nonesuch": 1})

    def test_swallowed_names_reach_the_usage_record(self, monkeypatch):
        monkeypatch.setattr(P, "_WARNED_SWALLOWED", set())
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

        class _Resp:
            model = "claude-sonnet-4-6"
            stop_reason = "end_turn"
            content = [type("B", (), {"type": "text", "text": "ok"})()]
            usage = None
        client = type("C", (), {"messages": type("M", (), {
            "create": staticmethod(lambda **kw: _Resp())})})
        monkeypatch.setattr(P, "_cached_client", lambda k, f: client)
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        usage = []
        P.call_anthropic("hi", model="claude-sonnet-4-6",
                         usage_sink=usage.append, nonesuch=1)
        assert "nonesuch" in usage[0]["dropped_params"]


class TestReplicates:
    def test_identical_prompts_no_longer_collapse(self):
        """n samples at temperature 1 were one API call and n-1 cache hits,
        and the run reported one observation as n."""
        keys = {json.dumps(_make_key("p", "m", "s", 1.0, 64, sample=i),
                           sort_keys=True) for i in range(5)}
        assert len(keys) == 5

    def test_sample_is_its_own_field_not_metadata(self):
        """So it cannot collide with a real metadata dict."""
        key = _make_key("p", "m", "s", 1.0, 64, sample=3,
                        metadata={"text_id": "T1"})
        assert key["sample"] == 3
        assert key["metadata"] == {"text_id": "T1"}

    def test_replicates_each_make_a_call(self):
        stash = FakeStash()
        llm = LLM("claude-sonnet-4-6", temperature=1.0, stash=stash)
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 1}') as call:
            for i in range(3):
                llm.extract("the text", Out, system_prompt="sys", sample=i)
        assert call.call_count == 3


class TestZeroCallsIsNotOneThing:
    def test_all_cached_and_all_failed_are_distinguishable(self):
        cached = UsageTracker()
        failed = UsageTracker()
        failed.record_failure(RuntimeError("no API key"))
        assert cached.report()["calls"] == failed.report()["calls"] == 0
        assert cached.report()["failed_calls"] == 0
        assert failed.report()["failed_calls"] == 1
        assert "every attempt failed" in failed.summary_line()
        assert "nothing was sent" in cached.summary_line()

    def test_an_llm_run_records_its_own_failures(self, monkeypatch):
        """End to end through the real dispatch: the sink is wired by
        _provider_kwargs, so every path that calls a provider counts."""
        import largeliterarymodels.llm as M

        def _boom(**kw):
            raise RuntimeError("boom")
        monkeypatch.setattr(M, "route_provider", lambda m: _boom)
        llm = LLM("claude-sonnet-4-6", stash=FakeStash())
        with pytest.raises(ValueError, match="Failed to extract"):
            llm.extract("t", Out, system_prompt="s", retries=0)
        r = llm.usage.report()
        assert r["calls"] == 0 and r["failed_calls"] == 1
        assert r["failure_types"] == {"RuntimeError": 1}
        assert "every attempt failed" in llm.usage.summary_line()

    def test_dispatch_counts_a_raising_provider(self, monkeypatch):
        import largeliterarymodels.llm as M
        tracker = UsageTracker()
        monkeypatch.setattr(M, "route_provider",
                            lambda m: (_ for _ in ()).throw)
        with pytest.raises(Exception):  # noqa: B017 — any raise; the count is the point
            M._call_provider("p", "claude-sonnet-4-6",
                             failure_sink=tracker.record_failure)
        assert tracker.report()["failed_calls"] == 1


class TestUsageLogOnTheLLMPath:
    def test_llm_map_writes_a_receipt(self, tmp_path, monkeypatch):
        """A production run drove LLM.map directly and kept no durable
        record of a single token."""
        import largeliterarymodels.llm as M
        monkeypatch.setattr(M, "STASH_PATH", str(tmp_path / "stash"))
        llm = LLM("claude-sonnet-4-6", stash=FakeStash(), usage_log="probe")

        def fake(**kw):
            kw["usage_sink"]({"input_tokens": 10, "output_tokens": 4,
                              "response_model": "claude-sonnet-4-6"})
            return "text"
        with patch("largeliterarymodels.llm._call_provider", side_effect=fake):
            llm.map(["a", "b"])
        rows = (tmp_path / "usage_logs" / "probe.jsonl").read_text().strip()
        rec = json.loads(rows)
        assert rec["batch"]["input_tokens"] == 20
        assert len(rec["items"]) == 2

    def test_task_delegates_to_the_same_writer(self, monkeypatch):
        """One writer. Two record formats is how a pricing script learns to
        parse only one of them."""
        import largeliterarymodels.task as task_mod
        import largeliterarymodels.administered as adm_mod
        seen = []
        # Patched where the lookup happens: Task's receipt writer lives on
        # the shared _Administered base, so both Task and Survey go through
        # this one call. That is the "one writer" this test is about.
        monkeypatch.setattr(adm_mod, "append_usage_log",
                            lambda *a, **kw: seen.append((a, kw)))

        class T(task_mod.Task):
            schema = Out
        T()._append_usage_log(LLM("claude-sonnet-4-6", stash=FakeStash()),
                              {0: {"index": 0}}, None)
        assert seen and seen[0][0][0] == "T"


class TestCompanionParameters:
    """A route whose stack truncates unasked cannot be partially pinned.

    Measured adjacent to this package: transformers 5.4.0 applies an
    effective top_k=50 when the field is absent while every config surface
    reports None. Our local/ route speaks HTTP to llama.cpp/vLLM/LM
    Studio/Ollama rather than to generate(), so the number is not ours —
    but four backends behind one prefix, each with its own defaults, is the
    same hazard with more places to hide.
    """

    def test_local_refuses_a_partial_pin(self):
        with pytest.raises(P.SamplingUnmeasured, match="top_k"):
            P.effective_sampling("lmstudio/qwen3.5-35b", {"top_p": 0.9},
                                 allow_unmeasured=True)

    def test_local_accepts_an_explicit_companion(self):
        governing, dropped = P.effective_sampling(
            "lmstudio/qwen3.5-35b", {"top_p": 0.9, "top_k": 0},
            allow_unmeasured=True)
        assert governing == {"top_p": 0.9, "top_k": 0} and dropped == ()

    def test_routes_without_a_companion_are_unaffected(self):
        governing, _ = P.effective_sampling(DS, {"top_p": 0.9})
        assert governing == {"top_p": 0.9}


class TestProbeVerdict:
    """A null from an unvalidated instrument is not a measurement."""

    def _verdict(self, arms, param):
        from largeliterarymodels.cli.sampling_probe import _verdict
        return _verdict(arms, param)

    def _arms(self, supports):
        # supports: {value: n_distinct}; outputs faked to that cardinality
        return [(v, [f"o{i}" for i in range(n)], None)
                for v, n in supports.items()]

    def test_no_arm_moving_is_inconclusive_not_ignored(self):
        v, why, sens = self._verdict(
            self._arms({None: 9, 1.0: 9, 0.9: 9, 0.7: 9}), "top_p")
        assert v == "INCONCLUSIVE"
        assert "NOT evidence" in why
        assert sens.startswith("UNVALIDATED")

    def test_a_low_arm_moving_validates_the_instrument(self):
        v, _why, sens = self._verdict(
            self._arms({None: 30, 1.0: 30, 0.95: 30, 0.7: 4}), "top_p")
        assert v == "ACCEPTS"
        assert sens.startswith("VALIDATED")
        # The 0.95 null is now a bound rather than an absence — which is the
        # sentence that keeps a small vendor default from being written off.
        assert "bound" in sens

    def test_a_rejection_is_a_result(self):
        arms = [(None, ["a"], None), (1.0, [], ValueError("unsupported"))]
        v, why, _s = self._verdict(arms, "top_p")
        assert v == "REJECTS" and "one call settles it" in why


class TestRejectionScan:
    """Reading an undisclosed default off a family that rejects
    non-defaults: acceptance is then an equality test against a number the
    vendor does not publish. The premise is itself testable, and its
    failure is silent, so a negative control rides along.
    """

    def _scan(self, behaviour, monkeypatch):
        """behaviour: value -> True to accept, or an Exception to raise."""
        import largeliterarymodels.cli.sampling_probe as SP

        def fake(prompt, model=None, temperature=None, max_tokens=None,
                 sampling=None, allow_unmeasured=False):
            outcome = behaviour(sampling["top_p"])
            if isinstance(outcome, Exception):
                raise outcome
            return "42"
        monkeypatch.setattr(SP, "route_provider", lambda m: fake)
        return SP._rejection_scan("claude-opus-5", "top_p", "p", 1.0, 8)

    def test_one_acceptance_is_reported_without_interpretation(
            self, monkeypatch):
        bad = ValueError("top_p must equal the model default 1.0")
        verdict, detail, _rows = self._scan(
            lambda v: True if v == 1.0 else bad, monkeypatch)
        # Named for what was observed, not for what it might mean: the
        # earlier name DEFAULT_FOUND was true about the API and false about
        # the world, which is the shape that travels.
        assert verdict == "ONLY_VALUE_ACCEPTED"
        assert "top_p=1.0" in detail
        assert "deprecated" in detail.lower()

    def test_the_control_separates_not_settable_from_not_found(self,
                                                               monkeypatch):
        """Without the control, 'everything errored' is ambiguous between
        'the parameter is not settable' and 'the default is outside my
        candidate list' — and those want opposite constants."""
        bad = ValueError("Unsupported parameter: 'top_p'")
        verdict, detail, _rows = self._scan(lambda v: bad, monkeypatch)
        assert verdict == "NOT_SETTABLE"
        assert "_REJECTS" in detail or "rejects the PARAMETER" in detail

    def test_a_permissive_family_is_reported_as_such(self, monkeypatch):
        """If the control is accepted too, the equality premise is false and
        the trick does not apply — fall back to the support test."""
        verdict, detail, _rows = self._scan(lambda v: True, monkeypatch)
        assert verdict == "SETTABLE"
        assert "support test" in detail

    def test_error_signatures_ignore_the_echoed_value(self):
        """A message that echoes the rejected value would otherwise make
        every refusal look distinct, and the question is whether the API is
        refusing the VALUE or the PARAMETER."""
        from largeliterarymodels.cli.sampling_probe import _error_signature
        a = _error_signature(ValueError("invalid value 0.37 for 'top_p'"))
        b = _error_signature(ValueError("invalid value 1.0 for 'top_p'"))
        assert a == b


class TestMeasuredConstants:
    """Three Anthropic models, three regimes, measured 2026-08-21 by
    rejection scan. The route-wide table could not have expressed this, and
    a family claim from one member would have been wrong about the third.

    The correction that produced these: the Opus pattern (only 0.99
    accepted) was first read as "accepts iff equals the undisclosed
    default". The refusal text says "`top_p` is deprecated for this model",
    which is not an equality test — a surviving legacy value, a no-op and a
    default all produce identical acceptance evidence. The number licenses
    no claim.
    """

    def test_a_deprecated_parameter_is_never_sent(self):
        """Not even at the one value the API accepts. An
        accepted-and-ignored value in a key is the false pin the whole
        schema exists to prevent."""
        governing, dropped = P.effective_sampling("claude-opus-5",
                                                  {"top_p": 0.99})
        assert governing == {} and dropped == ("top_p",)

    def test_no_vendor_default_is_claimed_anywhere(self):
        """The table briefly held 0.99 for opus-5. Empty means UNMEASURED,
        never 1.0, and never a number that merely survived a deprecation."""
        assert P._SAMPLING_DEFAULTS == {}
        assert P.known_sampling_defaults("claude-opus-5") == {}

    def test_the_three_families_are_distinguished(self):
        assert P._sampling_state("anthropic", "top_p",
                                 "claude-opus-5") == P._DEPRECATED
        assert P._sampling_state("anthropic", "top_p",
                                 "claude-sonnet-5") == P._DEPRECATED
        assert P._sampling_state("anthropic", "top_p",
                                 "claude-haiku-4-5-20251001") is P._REJECTS

    def test_an_unprobed_anthropic_family_still_raises(self):
        """The measurement covers three models, not the route."""
        with pytest.raises(P.SamplingUnmeasured):
            P.effective_sampling("claude-opus-4-8", {"top_p": 0.9})

    def test_the_drop_reason_names_the_deprecation(self, monkeypatch):
        """A methods note written from dropped_params should not have to
        guess why the field is missing."""
        monkeypatch.setattr(P, "_WARNED_NO_SAMPLING", set())
        reason = P._dropped_reason("anthropic", "top_p", "claude-opus-5")
        assert "deprecated" in reason and "undisclosed" in reason

    def test_the_receipt_says_what_could_not_be_pinned(self):
        from largeliterarymodels.task import Task

        class T(Task):
            schema = Out
        rec = T().administration_record(model="claude-opus-5")
        assert rec["vendor_sampling_defaults"] == {}
        assert "top_p" in rec["sampling_unavailable"]
        assert T().administration_record(
            model="deepseek/deepseek-v4-pro")["sampling_unavailable"] == {}

    def test_settable_did_not_become_a_constant(self):
        """The scan found deepseek/openai/google accept every value INCLUDING
        the control, which proves settability and says nothing about whether
        the value governs — DeepSeek's thinking mode accepts and ignores all
        four. Those three stay unmeasured on purpose."""
        for model in ("gpt-5.4-nano", "gemini-3.5-flash-lite"):
            with pytest.raises(P.SamplingUnmeasured):
                P.effective_sampling(model, {"top_p": 0.9})


class TestRequestedIsNotApplied:
    """The receipt must separate what a run ASKED for from what governed.

    A three-model Anthropic arm is not homogeneous: haiku-4-5 honours a
    temperature that sonnet-5 and opus-5 discard, so "the Anthropic arm was
    temperature-pinned" can be true of one model and false of the next in
    the same corpus.
    """

    def _rec(self, model):
        from largeliterarymodels.task import Task

        class T(Task):
            schema = Out
            temperature = 0.0
        return T().administration_record(model=model)

    def test_the_two_fields_differ_exactly_where_a_claim_would_be_false(self):
        opus = self._rec("claude-opus-5")
        assert opus["temperature"] == 0.0
        assert opus["temperature_effective"] is None

    def test_within_one_vendor_the_answer_splits(self):
        assert self._rec("claude-sonnet-5")["temperature_effective"] is None
        assert self._rec(
            "claude-haiku-4-5-20251001")["temperature_effective"] == 0.0

    def test_a_model_that_honours_it_says_so(self):
        rec = self._rec("deepseek/deepseek-v4-pro")
        assert rec["temperature"] == rec["temperature_effective"] == 0.0


class TestProbingTemperature:
    """The parameter every Task sets was the one the probe could not
    measure. `_NO_TEMPERATURE_MODELS` is a denylist, so absence from it
    means 'not known to reject' — not 'accepts', and certainly not
    'applies'."""

    def test_temperature_is_probeable(self):
        from largeliterarymodels.cli.sampling_probe import _arm_kwargs
        # An UNSET arm sends no temperature at all, which is what an
        # unpinned run did.
        assert _arm_kwargs("temperature", None, 1.0) == {
            "temperature": None, "sampling": None}
        assert _arm_kwargs("temperature", 0.0, 1.0) == {
            "temperature": 0.0, "sampling": None}

    def test_a_sampling_param_still_holds_temperature_fixed(self):
        from largeliterarymodels.cli.sampling_probe import _arm_kwargs
        assert _arm_kwargs("top_p", 0.9, 1.0) == {
            "temperature": 1.0, "sampling": {"top_p": 0.9}}

    def test_temperature_arms_run_in_the_same_direction(self):
        """Lower value, smaller support — so the verdict logic is unchanged.
        And 0.0 against 1.0 is an enormous effect where top_p 0.95 against
        1.0 is a sliver, so this is the question the support test answers
        with power to spare."""
        from largeliterarymodels.cli.sampling_probe import (
            TEMPERATURE_VALUES, _verdict)
        assert min(TEMPERATURE_VALUES) == 0.0
        arms = [(None, ["a"] * 9, None), (1.0, [f"o{i}" for i in range(9)],
                                          None), (0.0, ["same"], None)]
        verdict, _why, sens = _verdict(arms, "temperature")
        assert verdict == "ACCEPTS" and sens.startswith("VALIDATED")


class TestRefusalTextIsNotTruncated:
    """The defect that cost a real finding: the scan printed refusals at 120
    characters, and the sentence that overturned the verdict — "`top_p` is
    deprecated for this model." — sat past the cut. The acceptance pattern
    could not distinguish a default from a deprecation; the message could,
    and the message was on screen and clipped.
    """

    LONG = ("Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': '`top_p` is deprecated "
            "for this model.'}}")

    def _run(self, capsys, monkeypatch):
        import largeliterarymodels.cli.sampling_probe as SP

        def fake(prompt, model=None, temperature=None, max_tokens=None,
                 sampling=None, allow_unmeasured=False):
            if sampling["top_p"] == 0.99:
                return "42"
            raise ValueError(self.LONG)
        monkeypatch.setattr(SP, "route_provider", lambda m: fake)
        args = type("A", (), {"yes": True, "prompt": None, "temperature": 1.0,
                              "max_tokens": 8, "values": None})()
        SP._run_rejection_scan(args, "claude-opus-5", "anthropic", "top_p")
        return capsys.readouterr().out

    def test_the_whole_message_is_printed(self, capsys, monkeypatch):
        out = self._run(capsys, monkeypatch)
        assert self.LONG in out, "refusal text must never be truncated"
        assert "deprecated" in out

    def test_no_constant_is_offered_for_an_ambiguous_pattern(self, capsys,
                                                             monkeypatch):
        """The verdict is named for what was observed. Printing a constant
        here would invite writing down the reading the evidence does not
        support."""
        out = self._run(capsys, monkeypatch)
        assert "ONLY_VALUE_ACCEPTED" in out
        assert "_ACCEPTS" not in out
        assert "No constant is printed" in out


class TestUnsupportedValuePhrasing:
    """gpt-5-nano refuses a pinned temperature with a DIFFERENT sentence
    from the one the repair loop knew: "Unsupported value: 'temperature'
    does not support 0.0 with this model." Matching only on "unsupported
    parameter" meant the loop never learned it and 6 of 6 calls died."""

    def test_both_phrasings_are_recognised(self):
        assert P._unsupported_param(
            ValueError("Unsupported parameter: 'temperature' is not supported"),
            "temperature")
        assert P._unsupported_param(
            ValueError("Unsupported value: 'temperature' does not support 0.0 "
                       "with this model. Only the default (1) is supported."),
            "temperature")

    def test_a_listed_supported_param_is_still_not_a_rejection(self):
        """The anchoring that the original phrasing needed must survive on
        the new one: the name has to be the FIRST quoted token, or an error
        listing what IS supported reads as rejecting all of it."""
        exc = ValueError("Unsupported value: 'top_p' does not support 0.5. "
                         "Supported parameters: 'temperature', 'seed'.")
        assert P._unsupported_param(exc, "top_p")
        assert not P._unsupported_param(exc, "temperature")

    def test_the_key_records_no_temperature_for_that_model(self):
        """The dynamic memo is per-process and cannot reach the cache key,
        so without a static entry the key would assert temperature 0.0 on a
        run that sampled at the model's default of 1 — a false pin."""
        assert P.effective_temperature("gpt-5-nano", 0.0) is None
        assert P.effective_temperature("gpt-5.4-mini", 0.0) == 0.0
        assert P.effective_temperature("deepseek/deepseek-v4-flash", 0.0) == 0.0
