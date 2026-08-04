"""Regression tests for audit fixes in the core library (2026-07).

Each test pins a specific repaired defect; see the named commit trailers in
git history for the original failure scenarios.
"""

import json
import warnings
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from largeliterarymodels.llm import LLM, _image_cache_id, _make_key, _parse_json_response
from largeliterarymodels.providers import (
    call_deepseek,
    check_api_keys,
    route_provider,
    _supports_temperature,
)
from largeliterarymodels.task import Task


class FakeStash(dict):
    """Dict-backed stash accepting dict keys, like HashStash."""

    def _k(self, key):
        return json.dumps(key, sort_keys=True, default=str)

    def __contains__(self, key):
        return dict.__contains__(self, self._k(key))

    def __getitem__(self, key):
        return dict.__getitem__(self, self._k(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, self._k(key), value)


class Out(BaseModel):
    x: int


class OutStrict(BaseModel):
    x: int
    y: str


class TestImageCacheKeys:
    """Image keys must be content-derived, not length/id-derived."""

    def test_equal_length_bytes_differ(self):
        k1 = _make_key("p", "m", images=[b"aaaa"])
        k2 = _make_key("p", "m", images=[b"bbbb"])
        assert k1["images"] != k2["images"]

    def test_bytes_key_stable(self):
        assert _image_cache_id(b"data") == _image_cache_id(b"data")

    def test_path_keys_unchanged(self):
        # paths keep keying by path so existing caches stay valid
        assert _make_key("p", "m", images=["page1.png"])["images"] == ["page1.png"]


class TestExtractRetrySemantics:
    def test_provider_error_consumes_retry(self):
        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated 529")
            return '{"x": 1}'

        llm = LLM("claude-sonnet-4-6", stash=FakeStash())
        with patch("largeliterarymodels.llm._call_provider", side_effect=flaky):
            assert llm.extract("p", Out, retries=1).x == 1
        assert calls["n"] == 2

    def test_stale_cache_falls_through_to_recompute(self):
        llm = LLM("claude-sonnet-4-6", stash=FakeStash())
        ck = {"k": 1}
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 1}'):
            llm.extract("p", Out, retries=0, cache_key=ck)
        # schema gained a required field; cached response no longer validates
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 2, "y": "z"}'):
            result = llm.extract("p", OutStrict, retries=0, cache_key=ck)
        assert result.y == "z"


class TestMapErrorHandling:
    def test_one_failure_does_not_abort_batch(self):
        def fail_bad(prompt=None, **kw):
            if prompt == "bad":
                raise RuntimeError("boom")
            return "ok:" + prompt

        llm = LLM("claude-sonnet-4-6", stash=FakeStash())
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=fail_bad):
            results = llm.map(["a", "bad", "c"], num_workers=2)
        assert results == ["ok:a", None, "ok:c"]


class TestExtractImapBatching:
    def test_duplicate_prompts_share_one_call(self):
        calls = {"n": 0}

        def count(**kw):
            calls["n"] += 1
            return '{"x": 5}'

        llm = LLM("claude-sonnet-4-6", stash=FakeStash())
        with patch("largeliterarymodels.llm._call_provider", side_effect=count):
            out = dict(llm.extract_imap(["same", "same"], Out, retries=0))
        assert calls["n"] == 1
        assert out[0].x == 5 and out[1].x == 5

    def test_generator_input_accepted(self):
        llm = LLM("claude-sonnet-4-6", stash=FakeStash())
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 7}'):
            out = dict(llm.extract_imap((p for p in ["g1", "g2"]), Out,
                                        retries=0))
        assert len(out) == 2


class TestJsonRepairPath:
    def test_malformed_but_repairable(self):
        # trailing comma + single quotes: unparseable by json.loads,
        # recoverable by json_repair
        raw = "{'x': 1,}"
        assert _parse_json_response(raw) == {"x": 1}

    def test_unrecoverable_raises(self):
        with pytest.raises(ValueError):
            _parse_json_response("complete nonsense with no structure")


class TestDeepseekWiring:
    def test_prefix_and_exact_names_route(self):
        assert route_provider("deepseek/deepseek-chat") is call_deepseek
        assert route_provider("deepseek-chat") is call_deepseek

    def test_bare_local_checkpoint_does_not_route_to_paid_api(self):
        with pytest.raises(ValueError):
            route_provider("deepseek-r1:8b")

    def test_local_prefix_still_wins(self):
        from largeliterarymodels.providers import call_local
        assert route_provider("ollama/deepseek-r1:8b") is call_local

    def test_images_rejected(self):
        with pytest.raises(ValueError):
            call_deepseek("hi", images=["x.png"])

    def test_key_check_knows_deepseek(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        assert "DEEPSEEK_API_KEY" in check_api_keys()

    def test_current_model_names_route(self):
        # Verified against GET /models 2026-07-30: these are the only two the
        # API lists. Retired aliases must keep routing so existing pins work.
        for m in ("deepseek-v4-flash", "deepseek-v4-pro"):
            assert route_provider(m) is call_deepseek
            assert route_provider(f"deepseek/{m}") is call_deepseek
        assert route_provider("deepseek-reasoner") is call_deepseek

    def test_advertised_models_are_not_retired_aliases(self, monkeypatch):
        # Pin the env: available_models() lists DeepSeek only when the key is
        # set, so without this the test's outcome depended on whose shell ran
        # it — "N passing" was a property of the machine, not the branch.
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        from largeliterarymodels.providers import _DEEPSEEK_LEGACY_ALIASES
        from largeliterarymodels.utils import available_models
        advertised = [m for m in available_models() if "deepseek" in m]
        assert advertised, "expected DeepSeek suggestions with a key set"
        for m in advertised:
            assert m.split("/")[-1] not in _DEEPSEEK_LEGACY_ALIASES

    def test_legacy_alias_warns_with_what_it_resolves_to(self, monkeypatch, caplog):
        """'deepseek-reasoner' silently serves flash; the caller must be told."""
        import largeliterarymodels.providers as P
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        monkeypatch.setattr(P, "_WARNED_DEEPSEEK_ALIASES", set())

        class _Resp:
            model = "deepseek-v4-flash"
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        class _Client:
            chat = type("Chat", (), {"completions": type(
                "Comp", (), {"create": staticmethod(lambda **kw: _Resp())})})

        monkeypatch.setattr(P, "_cached_client", lambda key, factory: _Client())
        with caplog.at_level("WARNING"):
            P.call_deepseek("hi", model="deepseek/deepseek-reasoner")
        text = caplog.text
        assert "deepseek-reasoner" in text and "deepseek-v4-flash" in text

    def test_resolved_model_logged_once_per_resolution(self, monkeypatch, caplog):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        with caplog.at_level("WARNING"):
            for _ in range(3):
                P._log_resolved_model("deepseek", "deepseek-chat",
                                      "deepseek-v4-flash")
        assert caplog.text.count("resolved server-side") == 1


class TestDeepSeekThinkingMode:
    """DeepSeek v4 reasons by default on BOTH tiers, at effort 'high'.

    Two things followed, and the second is why this is a regression class and
    not a cost note. Thinking bills as output — a live flash probe billed 107
    completion tokens of which 97 were reasoning — and thinking mode also
    accepts `temperature` and ignores it WITHOUT erroring, so the repair loop
    in _chat_completion (which can only see a 400) never recorded the drop.
    Runs were published as temperature-controlled that were not.
    """

    def _client(self, seen, reasoning_tokens=0):
        """A fake OpenAI-compatible client recording the kwargs it is sent."""
        class _Details:
            pass

        details = _Details()
        details.reasoning_tokens = reasoning_tokens

        class _Usage:
            prompt_tokens = 100
            completion_tokens = 107
            completion_tokens_details = details

        class _Resp:
            model = "deepseek-v4-flash"
            usage = _Usage()
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        def create(**kw):
            seen.append(kw)
            return _Resp()

        return type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(create)})})})

    def _patch(self, monkeypatch, seen, reasoning_tokens=0):
        import largeliterarymodels.providers as P
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        monkeypatch.delenv("LITMOD_STRICT_PARAMS", raising=False)
        monkeypatch.setattr(P, "_TOKEN_PARAM", {})
        monkeypatch.setattr(P, "_NO_TEMPERATURE", set())
        monkeypatch.setattr(P, "_WARNED_NO_TEMPERATURE", set())
        monkeypatch.setattr(P, "_WARNED_THINKING_NOT_DISABLED", set())
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        client = self._client(seen, reasoning_tokens)
        monkeypatch.setattr(P, "_cached_client", lambda key, factory: client)
        return P

    @pytest.mark.parametrize("model", ["deepseek/deepseek-v4-flash",
                                       "deepseek/deepseek-v4-pro"])
    def test_thinking_disabled_by_default_on_both_tiers(self, monkeypatch, model):
        seen = []
        P = self._patch(monkeypatch, seen)
        P.call_deepseek("hi", model=model)
        assert seen[-1]["extra_body"]["thinking"] == {"type": "disabled"}

    def test_temperature_is_sent_when_thinking_is_off(self, monkeypatch):
        """With thinking disabled the parameter takes effect, so send it."""
        seen = []
        P = self._patch(monkeypatch, seen)
        P.call_deepseek("hi", model="deepseek/deepseek-v4-pro", temperature=0.0)
        assert seen[-1]["temperature"] == 0.0

    def test_temperature_recorded_as_dropped_when_thinking_is_on(self, monkeypatch):
        """The silent-ignore case: no 400, so nothing else would catch it."""
        seen, recorded = [], []
        P = self._patch(monkeypatch, seen)
        P.call_deepseek("hi", model="deepseek/deepseek-v4-pro", temperature=0.0,
                        thinking=None, usage_sink=recorded.append)
        assert "temperature" not in seen[-1], "must not claim a pin that no-ops"
        assert recorded[-1]["dropped_params"] == ("temperature",)

    def test_drop_message_states_the_right_mechanism(self, monkeypatch, caplog):
        """The generic wording ('the API rejects it') is false here — DeepSeek
        accepts it and ignores it, which is why nothing caught it before."""
        seen = []
        P = self._patch(monkeypatch, seen)
        with caplog.at_level("WARNING"):
            P.call_deepseek("hi", model="deepseek/deepseek-v4-pro",
                            temperature=0.0, thinking=None)
        assert "rejects it" not in caplog.text
        assert "ignores them" in caplog.text

    def test_explicit_enable_is_honoured(self, monkeypatch):
        seen = []
        P = self._patch(monkeypatch, seen)
        P.call_deepseek("hi", model="deepseek/deepseek-v4-flash",
                        thinking={"type": "enabled"})
        assert seen[-1]["extra_body"]["thinking"] == {"type": "enabled"}

    def test_reasoning_tokens_are_read_from_the_response(self, monkeypatch):
        seen, recorded = [], []
        P = self._patch(monkeypatch, seen, reasoning_tokens=97)
        P.call_deepseek("hi", model="deepseek/deepseek-v4-pro", thinking=None,
                        temperature=None, usage_sink=recorded.append)
        assert recorded[-1]["reasoning_tokens"] == 97
        assert recorded[-1]["output_tokens"] == 107, "a breakdown, not a deduction"

    def test_warns_when_a_requested_disable_did_not_take(self, monkeypatch, caplog):
        """Accepted-and-ignored is the failure a 400 cannot announce."""
        seen = []
        P = self._patch(monkeypatch, seen, reasoning_tokens=97)
        with caplog.at_level("WARNING"):
            P.call_deepseek("hi", model="deepseek/deepseek-v4-pro")
        assert "reasoned anyway" in caplog.text
        assert "was NOT applied" in caplog.text

    def test_ignored_disable_lands_in_the_usage_record(self, monkeypatch):
        """The audit's finding must reach dropped_params, not just the log.

        An earlier version audited AFTER _chat_completion had flushed usage
        to the sink, so in the one scenario the audit exists to catch the
        warning said 'temperature was NOT applied' while the machine-readable
        record said it was — and a producer asserting on the record (the
        whole point of having one) read the run as clean.
        """
        seen, recorded = [], []
        P = self._patch(monkeypatch, seen, reasoning_tokens=97)
        P.call_deepseek("hi", model="deepseek/deepseek-v4-pro",
                        temperature=0.0, usage_sink=recorded.append)
        assert "temperature" in recorded[-1]["dropped_params"]
        assert recorded[-1]["reasoning_observed"] is True

    def test_ignored_disable_raises_under_strict(self, monkeypatch):
        """Strict mode must escalate the receipt, not only the request."""
        seen = []
        P = self._patch(monkeypatch, seen, reasoning_tokens=97)
        monkeypatch.setenv("LITMOD_STRICT_PARAMS", "1")
        with pytest.raises(Exception) as exc_info:
            P.call_deepseek("hi", model="deepseek/deepseek-v4-pro",
                            temperature=0.0, usage_sink=lambda u: None)
        assert "NOT applied" in str(exc_info.value)

    def test_silent_when_the_disable_worked(self, monkeypatch, caplog):
        seen = []
        P = self._patch(monkeypatch, seen, reasoning_tokens=0)
        with caplog.at_level("WARNING"):
            P.call_deepseek("hi", model="deepseek/deepseek-v4-pro")
        assert "reasoned anyway" not in caplog.text

    def test_caller_kwargs_reach_the_api(self, monkeypatch):
        seen = []
        P = self._patch(monkeypatch, seen)
        P.call_deepseek("hi", model="deepseek/deepseek-v4-pro",
                        thinking={"type": "enabled"}, temperature=None,
                        reasoning_effort="low")
        assert seen[-1]["reasoning_effort"] == "low"

    def test_framework_kwargs_do_not_reach_the_api(self, monkeypatch):
        """LLM._provider_kwargs injects cache_ttl into every call; DeepSeek
        400s on it. Forwarding kwargs blindly would break the whole provider."""
        seen = []
        P = self._patch(monkeypatch, seen)
        P.call_deepseek("hi", model="deepseek/deepseek-v4-pro", cache_ttl="1h")
        assert "cache_ttl" not in seen[-1]


class TestReasoningTokensAreVisible:
    """output=20,171 read as answer text when 91% of it was discarded CoT."""

    def test_summary_line_reports_the_share(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        u.record({"output_tokens": 107, "reasoning_tokens": 97})
        assert "reasoning=97" in u.summary_line()
        assert "91% of output" in u.summary_line()

    def test_absent_when_nothing_reasoned(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        u.record({"output_tokens": 107})
        assert "reasoning" not in u.summary_line()
        assert u.report()["reasoning_share"] == 0.0


class TestNoReasoningIsObservedNotAssumed:
    """A producer gating publication on 'thinking was off' needs a gate that
    can both pass and fail against the live API. Measured 2026-08-04: DeepSeek
    omits reasoning_content AND completion_tokens_details entirely when
    thinking is off, and returns both when it is on — so the fields' presence
    means reasoning, and demanding their presence as proof of absence would be
    a gate that never passes."""

    def _tracker(self):
        from largeliterarymodels.llm import UsageTracker
        return UsageTracker()

    def test_passes_on_the_real_thinking_off_shape(self):
        """The shape DeepSeek actually returns with thinking disabled: no
        reasoning fields at all. This must read as clean, or the gate is
        unusable on the provider it was written for."""
        u = self._tracker()
        for _ in range(3):
            u.record({"output_tokens": 15})
        assert u.no_reasoning_observed()
        assert u.unreported_calls == 3, "healthy state, not a fault"

    def test_fails_when_the_disable_stops_working(self):
        u = self._tracker()
        u.record({"output_tokens": 15})
        u.record({"output_tokens": 326, "reasoning_tokens": 306,
                  "reasoning_reported": True})
        assert not u.no_reasoning_observed()

    def test_not_observed_on_an_empty_run(self):
        """Zero calls is zero evidence, not a clean bill."""
        assert not self._tracker().no_reasoning_observed()

    def test_reasoning_content_alone_trips_the_audit(self, monkeypatch, caplog):
        """Second signal: if the token counter goes missing but the body still
        carries a chain of thought, the check must not fall silent — and the
        warning must speak in characters, not print a character count in
        token units (~4x overstatement in the sentence a methods note would
        quote)."""
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_WARNED_THINKING_NOT_DISABLED", set())
        msg = type("M", (), {"reasoning_content": "We need answer JSON.",
                             "content": "{}"})
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5,
                               "completion_tokens_details": None})
        resp = type("R", (), {"choices": [type("C", (), {"message": msg})],
                              "usage": usage, "model": "m"})
        u = P._usage_openai_compat(resp)
        assert u["reasoning_observed"] is True, \
            "the body must count as evidence even with no token split"
        with caplog.at_level("WARNING"):
            P._audit_thinking_disabled("deepseek", "m", u, resp)
        assert "reasoned anyway" in caplog.text
        assert "characters" in caplog.text
        assert "21 reasoning tokens" not in caplog.text

    def test_reported_zero_survives_a_round_trip_from_a_real_response(self,
                                                                     monkeypatch):
        """The provider layer must set reasoning_reported, or the whole check
        is inert no matter how the tracker behaves."""
        import largeliterarymodels.providers as P

        class _Details:
            reasoning_tokens = 0

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 15
            completion_tokens_details = _Details()

        u = P._usage_openai_compat(type("R", (), {"usage": _Usage()}))
        assert u["reasoning_reported"] is True and u["reasoning_tokens"] == 0

    def test_absent_details_are_not_reported_as_zero(self):
        import largeliterarymodels.providers as P

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 15          # no completion_tokens_details

        u = P._usage_openai_compat(type("R", (), {"usage": _Usage()}))
        assert u["reasoning_reported"] is False and u["reasoning_tokens"] == 0


class TestAnsweringModelIsRecorded:
    """`task.model` is what we ASKED for. Hosted APIs alias retired names onto
    current checkpoints, so an artifact recording only the request describes a
    coder that may never have run. The served id was already read for a warning
    and thrown away; it now reaches the usage sink as data."""

    def _resp(self, model, provider):
        if provider == "anthropic":
            u = type("U", (), {"input_tokens": 10, "output_tokens": 5,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0})()
            return type("R", (), {"model": model, "usage": u})()
        u = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5,
                           "completion_tokens_details": None})()
        return type("R", (), {"model": model, "usage": u})()

    def test_anthropic_reports_the_served_snapshot(self):
        import largeliterarymodels.providers as P
        u = P._usage_anthropic(self._resp("claude-sonnet-4-6-20260219",
                                          "anthropic"))
        assert u["response_model"] == "claude-sonnet-4-6-20260219"

    def test_openai_compat_reports_the_served_id(self):
        import largeliterarymodels.providers as P
        u = P._usage_openai_compat(self._resp("deepseek-v4-flash", "openai"))
        assert u["response_model"] == "deepseek-v4-flash"

    def test_google_uses_model_version_not_model(self):
        """Google spells it differently; reading `.model` would silently hold
        None for a whole provider and nothing would mark the gap."""
        import largeliterarymodels.providers as P
        m = type("M", (), {"prompt_token_count": 10,
                           "candidates_token_count": 5,
                           "cached_content_token_count": 0})()
        r = type("R", (), {"model_version": "gemini-2.5-pro-002",
                           "usage_metadata": m})()
        assert P._usage_google(r)["response_model"] == "gemini-2.5-pro-002"

    def test_tracker_counts_ids_rather_than_keeping_one(self):
        """A run can span two served ids; a scalar would read as uniformity."""
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        u.record({"output_tokens": 5, "response_model": "deepseek-v4-flash"})
        u.record({"output_tokens": 5, "response_model": "deepseek-v4-flash"})
        u.record({"output_tokens": 5, "response_model": "deepseek-v4-pro"})
        assert u.report()["response_models"] == {"deepseek-v4-flash": 2,
                                                 "deepseek-v4-pro": 1}

    def test_a_split_run_is_announced_in_the_summary(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        u.record({"output_tokens": 5, "response_model": "a"})
        u.record({"output_tokens": 5, "response_model": "b"})
        assert "SERVED BY 2 MODELS" in u.summary_line()

    def test_a_single_served_id_is_not_noise(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        u.record({"output_tokens": 5, "response_model": "a"})
        assert "SERVED BY" not in u.summary_line()
        assert u.report()["response_models"] == {"a": 1}


class TestThinkingBlockResponses:
    """Every current reasoning model puts a ThinkingBlock first in content;
    content[0].text raised AttributeError on every single call."""

    def _blocks(self, *kinds):
        out = []
        for k in kinds:
            b = type("Block", (), {})()
            b.type = k
            if k == "text":
                b.text = "the answer"
            elif k == "thinking":
                b.thinking = "deliberating"   # deliberately no .text
            out.append(b)
        return out

    def test_thinking_first_still_finds_text(self):
        from largeliterarymodels.providers import _response_text
        assert _response_text(self._blocks("thinking", "text"), "m") == "the answer"

    def test_plain_text_only(self):
        from largeliterarymodels.providers import _response_text
        assert _response_text(self._blocks("text"), "m") == "the answer"

    def test_thinking_block_has_no_text_attribute(self):
        # Guards the regression itself: if this ever gains .text the bug is
        # masked rather than fixed.
        blocks = self._blocks("thinking", "text")
        assert not hasattr(blocks[0], "text")

    def test_interleaved_text_blocks_are_concatenated(self):
        """[thinking, text, thinking, text] must yield the WHOLE answer.

        Returning only the first text block silently truncates the JSON,
        which surfaces downstream as a parse error blamed on the model and
        burns a billed retry.
        """
        from largeliterarymodels.providers import _response_text
        blocks = self._blocks("thinking", "text", "thinking", "text")
        blocks[1].text = '{"a": 1'
        blocks[3].text = "}"
        assert _response_text(blocks, "m") == '{"a": 1}'

    def test_dict_shaped_blocks_are_text_too(self):
        """A proxy or recorded response hands back dicts, not SDK objects;
        'text is present but reported absent' is the worst reading of it."""
        from largeliterarymodels.providers import _response_text
        assert _response_text([{"type": "text", "text": "hi"}], "m") == "hi"

    def test_no_text_block_raises_a_useful_error(self):
        from largeliterarymodels.providers import _response_text
        with pytest.raises(ValueError, match="no text block"):
            _response_text(self._blocks("thinking"), "claude-sonnet-5")

    def test_error_reports_stop_reason_instead_of_guessing(self):
        """The old message asserted max_tokens exhaustion for every empty
        response — refusals and tool-only turns included — from a function
        that could not see stop_reason at all."""
        from largeliterarymodels.providers import _response_text
        with pytest.raises(ValueError) as exc_info:
            _response_text(self._blocks("thinking"), "m", stop_reason="refusal")
        assert "stop_reason: 'refusal'" in str(exc_info.value)
        assert "raise max_tokens" not in str(exc_info.value)
        with pytest.raises(ValueError, match="raise max_tokens"):
            _response_text(self._blocks("thinking"), "m",
                           stop_reason="max_tokens")

    def test_thinking_disabled_by_default_where_it_would_run(self):
        from largeliterarymodels.providers import thinking_default
        # Measured 3.3x output tokens with thinking on; extraction discards it.
        assert thinking_default("claude-sonnet-5") == {"type": "disabled"}
        assert thinking_default("claude-opus-5") == {"type": "disabled"}

    def test_family_tags_do_not_match_higher_versions(self):
        """'sonnet-5' must not match a future 'sonnet-50' — a substring hit
        would hand the new model every constant measured for this one."""
        from largeliterarymodels.providers import (thinking_default,
                                                   _supports_temperature)
        assert thinking_default("claude-sonnet-50") is None
        assert thinking_default("claude-opus-50") is None
        assert _supports_temperature("claude-sonnet-50")

    def test_omitted_where_the_api_default_is_already_off(self):
        from largeliterarymodels.providers import thinking_default
        for m in ("claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-7"):
            assert thinking_default(m) is None

    def test_not_sent_where_disabling_is_rejected(self):
        from largeliterarymodels.providers import thinking_default
        # Fable/Mythos 400 on an explicit disable — must omit, not disable.
        assert thinking_default("claude-fable-5") is None
        assert thinking_default("claude-mythos-5") is None

    def test_opus_5_rejects_temperature(self):
        from largeliterarymodels.providers import _supports_temperature
        # 'opus-5' was missing from the list; it 400s on temperature.
        assert not _supports_temperature("claude-opus-5")
        assert not _supports_temperature("claude-sonnet-5")
        assert _supports_temperature("claude-sonnet-4-6")


class TestOpenAIParamHealing:
    """The gpt-5 tier renamed max_tokens; heal from the error, not a regex."""

    def _err(self, param, replacement):
        return Exception(
            f"Error code: 400 - {{'error': {{'message': \"Unsupported "
            f"parameter: '{param}' is not supported with this model. Use "
            f"'{replacement}' instead.\", 'type': 'invalid_request_error'}}}}"
        )

    def test_reads_replacement_from_the_error(self):
        from largeliterarymodels.providers import _healed_token_param
        exc = self._err("max_tokens", "max_completion_tokens")
        assert _healed_token_param(exc, "max_tokens") == "max_completion_tokens"

    def test_self_heals_a_future_rename(self):
        from largeliterarymodels.providers import _healed_token_param
        exc = self._err("max_completion_tokens", "output_token_limit")
        assert (_healed_token_param(exc, "max_completion_tokens")
                == "output_token_limit")

    def test_unrelated_errors_are_not_treated_as_renames(self):
        from largeliterarymodels.providers import _healed_token_param
        assert _healed_token_param(Exception("rate limit exceeded"),
                                   "max_tokens") is None
        assert _healed_token_param(self._err("temperature", "top_p"),
                                   "max_tokens") is None

    def test_heals_and_memoizes_against_a_fake_client(self, monkeypatch):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_TOKEN_PARAM", {})
        seen = []

        class _Resp:
            model = "gpt-5.4-mini-2026-03-17"
            usage = None
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        def create(**kw):
            seen.append(sorted(kw))
            if "max_tokens" in kw:
                raise self._err("max_tokens", "max_completion_tokens")
            return _Resp()

        client = type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(create)})})})
        msgs = [{"role": "user", "content": "hi"}]

        P._chat_completion(client, "openai", "gpt-5.4-mini", msgs, 0.0, 16)
        assert len(seen) == 2, "one rejected attempt, then the healed one"
        assert P._TOKEN_PARAM[("openai", "gpt-5.4-mini")] == "max_completion_tokens"

        # Second call must not re-pay the 400.
        seen.clear()
        P._chat_completion(client, "openai", "gpt-5.4-mini", msgs, 0.0, 16)
        assert len(seen) == 1

    def test_memo_is_per_provider_not_per_model_string(self, monkeypatch):
        """A local proxy serving 'gpt-5.4' must not poison the real OpenAI
        provider's memo for the process."""
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_TOKEN_PARAM",
                            {("local", "gpt-5.4"): "num_predict"})
        assert P._TOKEN_PARAM.get(("openai", "gpt-5.4"), "max_tokens") \
            == "max_tokens"

    def test_supported_parameter_lists_are_not_rejections(self):
        """The realistic OpenAI error for one param LISTS others as
        supported; reading those as rejections dropped `temperature`, filed
        a false audit record, and never repaired the actual param."""
        from largeliterarymodels.providers import _unsupported_param
        err = Exception(
            "Unsupported parameter: 'top_p' is not supported with this "
            "model. Supported parameters: 'temperature', "
            "'max_completion_tokens'."
        )
        assert _unsupported_param(err, "top_p")
        assert not _unsupported_param(err, "temperature")
        assert not _unsupported_param(err, "max_completion_tokens")

    def test_body_echo_cannot_poison_the_heal(self):
        """A gateway echoing the request body names innocent params; the
        heal must not read one of them as the replacement for a param the
        error never rejected."""
        from largeliterarymodels.providers import _healed_token_param
        err = Exception(
            "Unsupported parameter: 'temperature' is not supported. "
            "Request body: {'max_tokens': 4096}. Use 'top_p' instead."
        )
        assert _healed_token_param(err, "max_tokens") is None

    def test_oscillation_surfaces_the_api_error_not_a_bare_wrapper(self, monkeypatch):
        """An A -> B -> A repair suggestion is a wrong suggestion. The
        tried-set refuses the second A, so the API's own error propagates in
        two calls — an earlier version burned a third call and then raised
        'could not build an acceptable request' with the API's diagnostic
        discarded entirely (no cause, no message)."""
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_TOKEN_PARAM", {})
        calls = []

        def create(**kw):
            calls.append(kw)
            if "max_tokens" in kw:
                raise self._err("max_tokens", "max_completion_tokens")
            raise self._err("max_completion_tokens", "max_tokens")

        client = type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(create)})})})
        with pytest.raises(Exception, match="Unsupported parameter"):
            P._chat_completion(client, "openai", "oscillator",
                               [{"role": "user", "content": "hi"}], 0.0, 16)
        assert len(calls) == 2, "the repeat suggestion must not be retried"

    def test_drops_temperature_when_rejected(self, monkeypatch):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_TOKEN_PARAM", {})
        monkeypatch.setattr(P, "_NO_TEMPERATURE", set())
        seen = []

        class _Resp:
            model = "m"
            usage = None
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        def create(**kw):
            seen.append(sorted(kw))
            if "temperature" in kw:
                raise Exception("Unsupported parameter: 'temperature' is not "
                                "supported with this model.")
            return _Resp()

        client = type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(create)})})})
        P._chat_completion(client, "openai", "m",
                           [{"role": "user", "content": "hi"}], 0.0, 16)
        assert "temperature" not in seen[-1]
        assert ("openai", "m") in P._NO_TEMPERATURE

    def test_strict_mode_raises_on_every_call_not_once(self, monkeypatch):
        """The flag a methods claim rests on. An earlier version added the
        model to _NO_TEMPERATURE before reporting, so call 1 raised and
        calls 2..N silently dropped temperature — under the flag whose only
        job is to make that impossible. 999 of a 1,000-item batch completed
        unpinned."""
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_TOKEN_PARAM", {})
        monkeypatch.setattr(P, "_NO_TEMPERATURE", set())
        monkeypatch.setattr(P, "_WARNED_NO_TEMPERATURE", set())
        monkeypatch.setenv("LITMOD_STRICT_PARAMS", "1")

        class _Resp:
            model = "m"
            usage = None
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        def create(**kw):
            if "temperature" in kw:
                raise Exception("Unsupported parameter: 'temperature' is not "
                                "supported with this model.")
            return _Resp()

        client = type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(create)})})})
        msgs = [{"role": "user", "content": "hi"}]
        for call_n in range(3):
            with pytest.raises(P.DroppedParameterError):
                P._chat_completion(client, "openai", "m", msgs, 0.0, 16)


class TestDroppedParameterStrictMode:
    """A parameter that appears to apply and does not is the worst shape:
    'administered at temperature 0' reads as true with nothing to contradict it."""

    def test_warns_by_default(self, monkeypatch, caplog):
        import largeliterarymodels.providers as P
        monkeypatch.delenv("LITMOD_STRICT_PARAMS", raising=False)
        with caplog.at_level("WARNING"):
            P._report_dropped_param("anthropic", "claude-sonnet-5",
                                    "temperature", 0.0, set())
        assert "was NOT applied" in caplog.text
        assert "do not describe" in caplog.text

    def test_raises_under_strict_mode(self, monkeypatch):
        import largeliterarymodels.providers as P
        monkeypatch.setenv("LITMOD_STRICT_PARAMS", "1")
        with pytest.raises(P.DroppedParameterError, match="temperature"):
            P._report_dropped_param("anthropic", "claude-sonnet-5",
                                    "temperature", 0.0, set())

    def test_warns_once_per_model(self, monkeypatch, caplog):
        import largeliterarymodels.providers as P
        monkeypatch.delenv("LITMOD_STRICT_PARAMS", raising=False)
        warned = set()
        with caplog.at_level("WARNING"):
            for _ in range(4):
                P._report_dropped_param("anthropic", "claude-sonnet-5",
                                        "temperature", 0.0, warned)
        assert caplog.text.count("was NOT applied") == 1


class TestCacheFloor:
    def test_known_floors(self):
        from largeliterarymodels.providers import cache_minimum_tokens
        # Non-monotonic across generations, which is the trap.
        assert cache_minimum_tokens("claude-haiku-4-5") == 4096
        assert cache_minimum_tokens("claude-opus-4-7") == 2048
        assert cache_minimum_tokens("claude-sonnet-4-6") == 1024
        assert cache_minimum_tokens("claude-opus-5") == 512

    def test_warns_only_when_certainly_below(self, monkeypatch, caplog):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_WARNED_CACHE_FLOOR", set())
        with caplog.at_level("WARNING"):
            # 4000 chars is at most ~1600 tokens: below haiku's 4096 floor
            # at any plausible density.
            P._warn_if_below_cache_floor("claude-haiku-4-5", "x" * 4000)
        assert "will not cache" in caplog.text
        # The surprising part is the economics, so the message has to carry it:
        # padding up past the floor is cheaper, not more expensive.
        assert "CHEAPER" in caplog.text
        assert "409 tokens" in caplog.text  # 10% of the 4096 floor

    def test_silent_in_the_ambiguous_band(self, monkeypatch, caplog):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_WARNED_CACHE_FLOOR", set())
        with caplog.at_level("WARNING"):
            # A peer project's 12,558-char instrument counts 4,711 tokens and
            # caches fine on haiku; never tell them it doesn't.
            P._warn_if_below_cache_floor("claude-haiku-4-5", "x" * 12558)
        assert caplog.text == ""

    def test_silent_for_prompts_too_small_to_matter(self, monkeypatch, caplog):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_WARNED_CACHE_FLOOR", set())
        with caplog.at_level("WARNING"):
            P._warn_if_below_cache_floor("claude-haiku-4-5", "x" * 600)
        assert caplog.text == ""


class TestClaudeCliDisabled:
    def test_raises_without_opt_in(self, monkeypatch):
        from largeliterarymodels.providers import call_claude_cli
        monkeypatch.delenv("LITMOD_ALLOW_CLAUDE_CLI", raising=False)
        with pytest.raises(RuntimeError, match="not permit"):
            call_claude_cli("hi")


class TestClaudeCliRouting:
    def test_claude_cli_prefix(self):
        from largeliterarymodels.providers import call_claude_cli
        assert route_provider("claude-cli/sonnet") is call_claude_cli


class TestTemperatureFamilies:
    @pytest.mark.parametrize("model", [
        "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5",
        "claude-fable-5",
    ])
    def test_sampling_removed_families(self, model):
        assert not _supports_temperature(model)

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-6",
        "claude-haiku-4-5",
    ])
    def test_sampling_supported_families(self, model):
        assert _supports_temperature(model)


class TestTaskKwargs:
    def test_typo_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Task(modle="opus")
        assert any("modle" in str(x.message) for x in w)

    def test_model_kwarg_allowed_silently(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Task(model="opus")
        assert not w


class TestTaskResultsLatestOnly:
    """Rewritten cache keys (force=True reruns) must not double-count in
    Task.results / Task.df — append-mode stashes keep full history on disk,
    and bare items() yields every version (hashstash 1.0 review finding)."""

    def _task_with_rewritten_key(self, tmp_path):
        from hashstash import HashStash

        class T(Task):
            name = "results_dedupe_test"
            schema = Out

        task = T()
        task._stash = HashStash(str(tmp_path / "stash"), engine="pairtree",
                                append_mode=True)
        key = {"prompt": "p", "model": "m"}
        task._stash[key] = '{"x": 1}'
        task._stash[key] = '{"x": 2}'  # rewrite, e.g. force=True rerun
        return task

    def test_results_yields_latest_only(self, tmp_path):
        task = self._task_with_rewritten_key(tmp_path)
        results = list(task.results)
        assert len(results) == 1
        assert results[0][1].x == 2

    def test_df_does_not_double_count(self, tmp_path):
        task = self._task_with_rewritten_key(tmp_path)
        df = task.df
        assert len(df) == 1
        assert df.iloc[0]["x"] == 2


class TestAnthropicWiring:
    """call_anthropic against a fake client. The primary provider's entire
    new surface — cache_ttl, the thinking default, the dropped-temperature
    record, the usage field mapping — previously had zero tests: a typo in
    the cache_control dict would silently disable 1-hour caching (2x write
    cost, no error) with the whole suite green."""

    def _client(self, seen, blocks=None, stop_reason="end_turn"):
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 5,
                               "cache_read_input_tokens": 900,
                               "cache_creation_input_tokens": 0})()
        if blocks is None:
            think = type("B", (), {"type": "thinking", "thinking": "..."})()
            text = type("B", (), {"type": "text", "text": '{"x": 1}'})()
            blocks = [think, text]
        resp = type("R", (), {"model": "claude-sonnet-5-20260101",
                              "content": blocks, "usage": usage,
                              "stop_reason": stop_reason})()
        create = staticmethod(lambda **kw: (seen.append(kw), resp)[1])
        return type("C", (), {"messages": type("M", (), {"create": create})()})()

    def _call(self, monkeypatch, seen, model="claude-sonnet-5", **kw):
        import largeliterarymodels.providers as P
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.delenv("LITMOD_STRICT_PARAMS", raising=False)
        monkeypatch.setattr(P, "_WARNED_NO_TEMPERATURE", set())
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        monkeypatch.setattr(P, "_cached_client",
                            lambda key, factory: self._client(seen))
        return P.call_anthropic("hi", model=model, **kw)

    def test_text_read_through_a_thinking_first_response(self, monkeypatch):
        seen = []
        assert self._call(monkeypatch, seen) == '{"x": 1}'

    def test_cache_ttl_reaches_cache_control(self, monkeypatch):
        seen = []
        self._call(monkeypatch, seen, system_prompt="S" * 3000, cache_ttl="1h")
        assert seen[0]["system"][0]["cache_control"] == \
            {"type": "ephemeral", "ttl": "1h"}

    def test_thinking_disabled_by_default_on_sonnet_5(self, monkeypatch):
        seen = []
        self._call(monkeypatch, seen)
        assert seen[0]["thinking"] == {"type": "disabled"}

    def test_thinking_omitted_on_older_families(self, monkeypatch):
        seen = []
        self._call(monkeypatch, seen, model="claude-sonnet-4-6")
        assert "thinking" not in seen[0]

    def test_temperature_drop_reaches_the_usage_record(self, monkeypatch):
        seen, rec = [], []
        self._call(monkeypatch, seen, temperature=0.0, usage_sink=rec.append)
        assert "temperature" not in seen[0]
        assert rec[0]["dropped_params"] == ("temperature",)

    def test_usage_field_mapping(self, monkeypatch):
        """cache_read_input_tokens -> cache_read_tokens etc. A typo here
        zeroes cache_hit_rate on the primary provider, suite green."""
        seen, rec = [], []
        self._call(monkeypatch, seen, usage_sink=rec.append)
        assert rec[0]["cache_read_tokens"] == 900
        assert rec[0]["input_tokens"] == 10
        assert rec[0]["output_tokens"] == 5
        assert rec[0]["response_model"] == "claude-sonnet-5-20260101"

    def test_thinking_blocks_count_as_reasoning_observed(self, monkeypatch):
        """The Fable case: Anthropic prices no reasoning split, so the
        thinking BLOCKS are the receipt. A token-only gate passed every
        Anthropic run as clean — including the one family where thinking
        cannot be disabled."""
        from largeliterarymodels.llm import UsageTracker
        seen, rec = [], []
        self._call(monkeypatch, seen, usage_sink=rec.append)
        assert rec[0]["reasoning_observed"] is True
        u = UsageTracker()
        u.record(rec[0])
        assert not u.no_reasoning_observed(), \
            "a run with thinking blocks must fail the no-reasoning gate"

    def test_textonly_response_passes_the_gate(self, monkeypatch):
        from largeliterarymodels.llm import UsageTracker
        import largeliterarymodels.providers as P
        seen, rec = [], []
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        text = type("B", (), {"type": "text", "text": "{}"})()
        monkeypatch.setattr(
            P, "_cached_client",
            lambda key, factory: self._client(seen, blocks=[text]))
        P.call_anthropic("hi", model="claude-sonnet-5", temperature=None,
                         usage_sink=rec.append)
        u = UsageTracker()
        u.record(rec[0])
        assert u.no_reasoning_observed()


class TestDeepSeekUnknownIds:
    """Deciding a model's behaviour from its name is the recurring bug
    class. An earlier version keyed the disable on a tuple of known ids, so
    an unrecognised id got NO disable and — treating it as a thinking-mode
    run — had `temperature` withheld from the request entirely: worse than
    before the patch, and DroppedParameterError-on-every-call under strict."""

    def test_unknown_id_gets_the_disable_and_keeps_temperature(self, monkeypatch):
        import largeliterarymodels.providers as P
        seen = []

        class _Resp:
            model = "deepseek-v5-pro"
            usage = None
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        def create(**kw):
            seen.append(kw)
            return _Resp()

        client = type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(create)})})})
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        monkeypatch.setattr(P, "_cached_client", lambda key, factory: client)
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        P.call_deepseek("hi", model="deepseek/deepseek-v5-pro", temperature=0.0)
        assert seen[-1]["extra_body"]["thinking"] == {"type": "disabled"}
        assert seen[-1]["temperature"] == 0.0


class TestResponseModelSurvivesMissingUsage:
    """Local servers routinely omit the usage object — the runs where the
    served id is least predictable. Discarding response.model there held the
    provenance field at None for exactly the providers that need it, the
    same gap the Google fix was justified by."""

    def test_openai_compat(self):
        import largeliterarymodels.providers as P
        msg = type("M", (), {"content": "ok", "reasoning_content": None})
        r = type("R", (), {"model": "qwen3.5-35b", "usage": None,
                           "choices": [type("C", (), {"message": msg})]})
        u = P._usage_openai_compat(r)
        assert u["response_model"] == "qwen3.5-35b"

    def test_anthropic(self):
        import largeliterarymodels.providers as P
        r = type("R", (), {"model": "claude-sonnet-5-20260101", "usage": None,
                           "content": []})
        assert P._usage_anthropic(r)["response_model"] == \
            "claude-sonnet-5-20260101"


class TestGoogleThoughtTokens:
    def test_thoughts_map_to_reasoning(self):
        import largeliterarymodels.providers as P
        m = type("M", (), {"prompt_token_count": 100,
                           "candidates_token_count": 50,
                           "cached_content_token_count": 0,
                           "thoughts_token_count": 40})()
        r = type("R", (), {"model_version": "gemini-2.5-pro-002",
                           "usage_metadata": m})()
        u = P._usage_google(r)
        assert u["reasoning_tokens"] == 40
        assert u["reasoning_observed"] is True

    def test_cached_tokens_subtracted(self):
        """The cached-token subtraction was previously exercised only at 0."""
        import largeliterarymodels.providers as P
        m = type("M", (), {"prompt_token_count": 1000,
                           "candidates_token_count": 50,
                           "cached_content_token_count": 900,
                           "thoughts_token_count": None})()
        r = type("R", (), {"model_version": "g", "usage_metadata": m})()
        u = P._usage_google(r)
        assert u["input_tokens"] == 100 and u["cache_read_tokens"] == 900


class TestOpenAICompatCacheBranches:
    """OpenAI and DeepSeek spell cached tokens differently; if the fallback
    branch broke, cache_hit_rate reads 0% and input double-counts the cached
    prefix — a cost report wrong ~10x with no error."""

    def _resp(self, **usage_attrs):
        u = type("U", (), {"completion_tokens_details": None, **usage_attrs})()
        msg = type("M", (), {"content": "ok", "reasoning_content": None})
        return type("R", (), {"model": "m", "usage": u,
                              "choices": [type("C", (), {"message": msg})]})

    def test_deepseek_spelling(self):
        import largeliterarymodels.providers as P
        u = P._usage_openai_compat(self._resp(
            prompt_tokens=1000, completion_tokens=5,
            prompt_cache_hit_tokens=900))
        assert u["cache_read_tokens"] == 900 and u["input_tokens"] == 100

    def test_openai_spelling(self):
        import largeliterarymodels.providers as P
        details = type("D", (), {"cached_tokens": 900})()
        u = P._usage_openai_compat(self._resp(
            prompt_tokens=1000, completion_tokens=5,
            prompt_cache_hit_tokens=None, prompt_tokens_details=details))
        assert u["cache_read_tokens"] == 900 and u["input_tokens"] == 100


class TestFrameworkKwargCoupling:
    """Every kwarg LLM._provider_kwargs can inject must be stripped before
    the HTTP body — pinned by iterating the actual denylist, so the NEXT
    framework kwarg anyone adds fails here instead of 400ing DeepSeek."""

    def test_provider_kwargs_injections_are_all_denylisted(self):
        import largeliterarymodels.providers as P
        # cache_ttl is injected by LLM._provider_kwargs; usage_sink is a
        # named parameter of every provider; schema/schema_name are consumed
        # upstream. All four must stay out of the request body.
        assert "cache_ttl" in P._NON_API_KWARGS
        for k in ("schema", "schema_name", "thinking"):
            assert k in P._NON_API_KWARGS

    def test_denylisted_keys_never_reach_the_wire(self, monkeypatch):
        import largeliterarymodels.providers as P
        seen = []

        class _Resp:
            model = "deepseek-v4-pro"
            usage = None
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        client = type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {
                "create": staticmethod(lambda **kw: (seen.append(kw), _Resp())[1])})})})
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        monkeypatch.setattr(P, "_cached_client", lambda key, factory: client)
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        junk = {k: "x" for k in P._NON_API_KWARGS if k != "thinking"}
        P.call_deepseek("hi", model="deepseek/deepseek-v4-pro", **junk)
        for k in P._NON_API_KWARGS:
            assert k not in seen[-1]


class TestDroppedHintDedupe:
    """A caller's dropped_hint plus a live temperature 400 must record the
    drop once, not twice — the audit record is a count, and a double entry
    reads as two independent drops."""

    def test_hint_plus_rejection_counts_once(self, monkeypatch):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_TOKEN_PARAM", {})
        monkeypatch.setattr(P, "_NO_TEMPERATURE", set())
        monkeypatch.setattr(P, "_WARNED_NO_TEMPERATURE", set())
        monkeypatch.delenv("LITMOD_STRICT_PARAMS", raising=False)
        rec = []

        class _Resp:
            model = "m"
            usage = None
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]

        def create(**kw):
            if "temperature" in kw:
                raise Exception("Unsupported parameter: 'temperature' is "
                                "not supported with this model.")
            return _Resp()

        client = type("Client", (), {"chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(create)})})})
        P._chat_completion(client, "openai", "m",
                           [{"role": "user", "content": "hi"}], 0.0, 16,
                           usage_sink=rec.append,
                           dropped_hint=("temperature",))
        assert rec[0]["dropped_params"] == ("temperature",)


class TestGoogleThinkingDefault:
    """Gemini reasons by default (measured: 363 thought tokens on a
    two-field probe) and thoughts bill as output. Off by default where the
    API permits — and the two generations take DIFFERENT parameters
    (probed live: budget-0 on 3.6-flash is a generic INVALID_ARGUMENT;
    3.x wants thinking_level, whose "minimal" measured zero thoughts where
    the default thought 370). Cannot-disable families get the Fable
    arrangement — nothing sent, warned once — because healing a rejection
    into a thinking-on call would store its output under an off-claiming
    cache key."""

    def test_flash_25_gets_budget_zero(self):
        from largeliterarymodels.providers import google_thinking_setting
        assert google_thinking_setting("gemini-2.5-flash") == \
            ("thinking_budget", 0)

    def test_flash_3x_gets_level_minimal(self):
        """The £147 finding: gemini-3.6-flash thought on every one of
        14,520 calls because nothing was sent. Its off-equivalent is
        thinking_level='minimal', not budget-0 (a 400)."""
        from largeliterarymodels.providers import google_thinking_setting
        assert google_thinking_setting("gemini-3.6-flash") == \
            ("thinking_level", "minimal")
        assert google_thinking_setting("gemini-3.5-flash-lite") == \
            ("thinking_level", "minimal")

    def test_gemini_30_is_not_gemini_3x(self):
        from largeliterarymodels.providers import _is_gemini_3x
        assert _is_gemini_3x("gemini-3.6-flash")
        assert not _is_gemini_3x("gemini-30-flash")
        assert not _is_gemini_3x("gemini-2.5-flash")

    def test_cannot_disable_families_send_nothing_and_warn(self, monkeypatch,
                                                           caplog):
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_WARNED_GOOGLE_THINKING", set())
        with caplog.at_level("WARNING"):
            assert P.google_thinking_setting("gemini-2.5-pro") is None
            assert P.google_thinking_setting("gemini-3.1-pro-preview") is None
        assert "cannot express 'off'" in caplog.text

    def test_cross_provider_spellings_map_per_generation(self):
        from largeliterarymodels.providers import google_thinking_setting
        # 2.5: budget vocabulary.
        assert google_thinking_setting("gemini-2.5-flash", None) is None
        assert google_thinking_setting("gemini-2.5-flash", False) == \
            ("thinking_budget", 0)
        assert google_thinking_setting("gemini-2.5-flash",
                                       {"type": "disabled"}) == \
            ("thinking_budget", 0)
        assert google_thinking_setting("gemini-2.5-flash", 128) == \
            ("thinking_budget", 128)
        # 3.x: level vocabulary; the off spellings map to minimal.
        assert google_thinking_setting("gemini-3.6-flash", False) == \
            ("thinking_level", "minimal")
        assert google_thinking_setting("gemini-3.6-flash", "low") == \
            ("thinking_level", "low")
        assert google_thinking_setting("gemini-3.6-flash", None) is None
        with pytest.raises(ValueError):
            google_thinking_setting("gemini-2.5-flash", object())

    def test_wrong_generation_vocabulary_raises(self):
        """A deprecated budget on 3.x has documented 'unexpected
        performance'; a silently-degraded parameter is worse than an
        error. Levels on 2.5 do not exist at all."""
        from largeliterarymodels.providers import google_thinking_setting
        with pytest.raises(ValueError, match="thinking_level"):
            google_thinking_setting("gemini-3.6-flash", 128)
        with pytest.raises(ValueError, match="thinking_budget"):
            google_thinking_setting("gemini-2.5-flash", "minimal")

    def test_fingerprint_vocabulary(self):
        """Old flash entries are thinking-on output — they must orphan.
        Pro's behaviour is unchanged (thinking then, thinking now), so its
        keys must stay byte-stable. 3.x keys say level:minimal, never
        'disabled' — there is no off state for a key to claim."""
        from largeliterarymodels.providers import thinking_fingerprint
        assert thinking_fingerprint("gemini-2.5-flash") == "disabled"
        assert thinking_fingerprint("gemini-2.5-pro") is None
        assert thinking_fingerprint("gemini-2.5-flash", 128) == "budget:128"
        assert thinking_fingerprint("gemini-3.6-flash") == "level:minimal"
        assert thinking_fingerprint("gemini-3.6-flash", "low") == "level:low"
        assert thinking_fingerprint("gemini-3.1-pro-preview") is None

    def test_rejected_setting_is_loud_not_healed(self, monkeypatch):
        """Matched on what WE sent, not Google's error prose — the prose
        drifted between generations (2.5-pro names thinking mode;
        3.6-flash says only 'invalid argument')."""
        import largeliterarymodels.providers as P
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setattr(P, "_WARNED_GOOGLE_THINKING", set())

        class _Models:
            @staticmethod
            def generate_content(**kw):
                raise Exception(
                    "400 INVALID_ARGUMENT. Request contains an invalid "
                    "argument.")

        client = type("C", (), {"models": _Models()})
        monkeypatch.setattr(P, "_cached_client", lambda key, factory: client)
        with pytest.raises(RuntimeError,
                           match="_GOOGLE_THINKING_CANNOT_DISABLE"):
            P.call_google("hi", model="gemini-9.9-hypothetical")

    def test_level_minimal_reaches_the_config(self, monkeypatch):
        import largeliterarymodels.providers as P
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        seen = {}

        class _Meta:
            prompt_token_count = 10
            candidates_token_count = 5
            cached_content_token_count = 0
            thoughts_token_count = None

        class _Resp:
            model_version = "gemini-3.6-flash"
            usage_metadata = _Meta()
            text = "{}"

        class _Models:
            @staticmethod
            def generate_content(**kw):
                seen.update(kw)
                return _Resp()

        client = type("C", (), {"models": _Models()})
        monkeypatch.setattr(P, "_cached_client", lambda key, factory: client)
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        P.call_google("hi", model="gemini-3.6-flash")
        # The SDK coerces the string to its ThinkingLevel enum.
        level = seen["config"].thinking_config.thinking_level
        assert "MINIMAL" in str(level).upper()
        assert seen["config"].thinking_config.thinking_budget is None

    def test_budget_zero_reaches_the_config(self, monkeypatch):
        import largeliterarymodels.providers as P
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        seen = {}

        class _Meta:
            prompt_token_count = 10
            candidates_token_count = 5
            cached_content_token_count = 0
            thoughts_token_count = None

        class _Resp:
            model_version = "gemini-2.5-flash-002"
            usage_metadata = _Meta()
            text = "{}"

        class _Models:
            @staticmethod
            def generate_content(**kw):
                seen.update(kw)
                return _Resp()

        client = type("C", (), {"models": _Models()})
        monkeypatch.setattr(P, "_cached_client", lambda key, factory: client)
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        P.call_google("hi", model="gemini-2.5-flash")
        assert seen["config"].thinking_config.thinking_budget == 0


class TestClosingReviewFixes:
    """Defects found by the closing (post-fix-campaign) review."""

    def test_anthropic_thinking_budget_is_part_of_the_key(self):
        """{'type': 'enabled', 'budget_tokens': 1024} and 64000 are
        different administrations; collapsing both to 'enabled' served the
        low-budget output as a cache hit for the high-budget rerun."""
        from largeliterarymodels.providers import thinking_fingerprint
        low = thinking_fingerprint("claude-sonnet-5",
                                   {"type": "enabled", "budget_tokens": 1024})
        high = thinking_fingerprint("claude-sonnet-5",
                                    {"type": "enabled", "budget_tokens": 64000})
        assert low != high
        assert thinking_fingerprint("claude-sonnet-5") == "disabled"

    def test_anthropic_normalizes_cross_provider_off_spellings(self, monkeypatch):
        """thinking=False means off on DeepSeek and Google; on Anthropic it
        used to go on the wire verbatim (a 400) with the fingerprint
        recording the string 'False'."""
        import largeliterarymodels.providers as P
        seen = []
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 1,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0})()
        text = type("B", (), {"type": "text", "text": "{}"})()
        resp = type("R", (), {"model": "m", "content": [text],
                              "usage": usage, "stop_reason": "end_turn"})()
        create = staticmethod(lambda **kw: (seen.append(kw), resp)[1])
        client = type("C", (), {"messages": type("M", (), {"create": create})()})()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setattr(P, "_cached_client", lambda key, factory: client)
        monkeypatch.setattr(P, "_LOGGED_RESOLUTIONS", set())
        P.call_anthropic("hi", model="claude-sonnet-5", temperature=None,
                         thinking=False)
        assert seen[0]["thinking"] == {"type": "disabled"}
        assert P.thinking_fingerprint("claude-sonnet-5", False) == "disabled"

    def test_anthropic_enabled_without_budget_raises(self):
        """No defensible default budget to invent — raise, loudly."""
        from largeliterarymodels.providers import _normalize_thinking_anthropic
        with pytest.raises(ValueError, match="budget_tokens"):
            _normalize_thinking_anthropic("enabled")
        with pytest.raises(ValueError, match="budget_tokens"):
            _normalize_thinking_anthropic(True)

    def test_dropped_param_warnings_keyed_by_param(self, monkeypatch, caplog):
        """On claude-cli the temperature warning used to permanently
        suppress the max_tokens warning — the dedup set keyed on model
        alone, and whichever param reported first claimed the slot."""
        import largeliterarymodels.providers as P
        monkeypatch.delenv("LITMOD_STRICT_PARAMS", raising=False)
        warned = set()
        with caplog.at_level("WARNING"):
            P._report_dropped_param("claude-cli", "m", "temperature", 0.0,
                                    warned)
            P._report_dropped_param("claude-cli", "m", "max_tokens", 4096,
                                    warned)
        assert "`temperature`" in caplog.text
        assert "`max_tokens`" in caplog.text

    def test_fingerprinting_does_not_fire_cost_warnings(self, monkeypatch,
                                                        caplog):
        """Key computation may run for fully-cached items that never reach
        the API; a cost warning there charges the user for spend that is
        not happening."""
        import largeliterarymodels.providers as P
        monkeypatch.setattr(P, "_WARNED_THINKING", set())
        monkeypatch.setattr(P, "_WARNED_GOOGLE_THINKING", set())
        with caplog.at_level("WARNING"):
            P.thinking_fingerprint("claude-fable-5")
            P.thinking_fingerprint("gemini-2.5-pro")
        assert caplog.text == ""


class TestGeminiCacheFloorSurfaced:
    """A 3,906-token instrument on gemini-3.6-flash ran 14,520 times at
    full input price — ~130 tokens under the 4,096 implicit-caching floor,
    visible only on the invoice. The zero-reads warning now covers Gemini,
    so this shape surfaces in the first batch summary instead of a week
    later."""

    def test_gemini_floor_known(self):
        from largeliterarymodels.providers import cache_minimum_tokens
        assert cache_minimum_tokens("gemini-3.6-flash") == 4096
        assert cache_minimum_tokens("gemini-3.1-pro-preview") == 4096
        assert cache_minimum_tokens("gemini-2.5-flash") is None, \
            "2.5 floors are unmeasured — None, not a guess"

    def test_zero_reads_warns_on_gemini(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        for _ in range(5):
            u.record({"input_tokens": 3965, "output_tokens": 20,
                      "cache_read_tokens": 0})
        w = u.cache_warning("gemini-3.6-flash")
        assert w is not None and "4,096" in w

    def test_healthy_gemini_reads_stay_silent(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        for _ in range(5):
            u.record({"input_tokens": 200, "output_tokens": 20,
                      "cache_read_tokens": 4200})
        assert u.cache_warning("gemini-3.6-flash") is None
