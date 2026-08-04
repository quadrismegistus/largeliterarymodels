"""Tests for Task class and BibliographyTask (mocked API calls)."""

import pytest
from unittest.mock import patch
from pydantic import BaseModel, Field
from hashstash import HashStash
from largeliterarymodels.task import Task, _schema_repr


class Sentiment(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float


class TestTaskInit:
    def test_defaults(self):
        task = Task()
        assert task.name is None
        assert task.task_name == "Task"
        assert task.schema is None
        assert task.examples == ()  # immutable default

    def test_kwargs_override(self):
        task = Task(name="custom", retries=3)
        assert task.task_name == "custom"
        assert task.retries == 3

    def test_repr(self):
        task = Task()
        assert "Task" in repr(task)


class TestTaskStash:
    def test_stash_path_includes_name(self):
        task = Task(name="test_task")
        stash = task.stash
        assert "test_task" in str(stash)

    def test_stash_is_lazy(self):
        task = Task(name="lazy_test")
        assert task._stash is None
        _ = task.stash
        assert task._stash is not None

    def test_stash_is_cached(self):
        task = Task(name="cached_test")
        s1 = task.stash
        s2 = task.stash
        assert s1 is s2


class TestTaskSubclass:
    def test_subclass_with_schema(self):
        class SentimentTask(Task):
            name = "sentiment"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        assert task.name == "sentiment"
        assert task.schema is Sentiment
        assert "sentiment" in str(task.stash)

    def test_subclass_repr(self):
        class SentimentTask(Task):
            name = "sentiment"
            schema = Sentiment
        assert "Sentiment" in repr(SentimentTask())

    def test_list_schema_repr(self):
        class MultiTask(Task):
            name = "multi"
            schema = list[Sentiment]
        assert "list[Sentiment]" in repr(MultiTask())


class TestTaskRun:
    @patch("largeliterarymodels.llm._call_provider")
    def test_run_returns_validated_model(self, mock_call):
        mock_call.return_value = '{"sentiment": "positive", "confidence": 0.95}'

        class SentimentTask(Task):
            name = "sentiment_test_run"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        result = task.run("I love this!")
        assert isinstance(result, Sentiment)
        assert result.sentiment == "positive"

    def test_run_raises_without_schema(self):
        task = Task(name="no_schema")
        with pytest.raises(ValueError, match="no schema defined"):
            task.run("hello")

    @patch("largeliterarymodels.llm._call_provider")
    def test_run_with_model_override(self, mock_call):
        mock_call.return_value = '{"sentiment": "negative", "confidence": 0.8}'

        class SentimentTask(Task):
            name = "sentiment_test_model"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        task.run("I hate this!", model="gpt-4o-mini")
        assert mock_call.call_args[1]["model"] == "gpt-4o-mini"

    @patch("largeliterarymodels.llm._call_provider")
    def test_run_with_examples(self, mock_call):
        mock_call.return_value = '{"sentiment": "neutral", "confidence": 0.5}'

        class SentimentTask(Task):
            name = "sentiment_test_examples"
            schema = Sentiment
            system_prompt = "Assess sentiment."
            examples = [
                ("Great!", Sentiment(sentiment="positive", confidence=0.9)),
            ]

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        task.run("It's okay.")
        call_kwargs = mock_call.call_args[1]
        assert "Example 1" in call_kwargs["system_prompt"]


class TestTaskMap:
    @patch("largeliterarymodels.llm._call_provider")
    def test_map_returns_list(self, mock_call):
        mock_call.side_effect = [
            '{"sentiment": "positive", "confidence": 0.9}',
            '{"sentiment": "negative", "confidence": 0.8}',
        ]

        class SentimentTask(Task):
            name = "sentiment_test_map"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        results = task.map(["I love it", "I hate it"], num_workers=1)
        assert len(results) == 2
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"

    def test_map_raises_without_schema(self):
        task = Task(name="no_schema")
        with pytest.raises(ValueError, match="no schema defined"):
            task.map(["hello"])


class TestMapErrors:
    """A positional None says an item failed; `errors` says which and why."""

    def _task(self, name):
        class SentimentTask(Task):
            schema = Sentiment
            system_prompt = "Assess sentiment."
            retries = 0

        SentimentTask.name = name
        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        return task

    @patch("largeliterarymodels.llm._call_provider")
    def test_errors_dict_identifies_failed_item(self, mock_call):
        mock_call.side_effect = [
            '{"sentiment": "positive", "confidence": 0.9}',
            'not json at all',
            '{"sentiment": "neutral", "confidence": 0.5}',
        ]
        task = self._task("sentiment_test_errors")
        errors = {}
        results = task.map(
            ["a", "b", "c"], num_workers=1,
            metadata_list=[{"id": "x1"}, {"id": "x2"}, {"id": "x3"}],
            errors=errors,
        )
        assert results[0] is not None and results[2] is not None
        assert results[1] is None
        assert list(errors) == [1]
        entry = errors[1]
        assert entry["index"] == 1
        assert entry["metadata"] == {"id": "x2"}
        assert entry["attempts"] == 1
        assert "not json at all" in entry["raw"]
        assert isinstance(entry["exception"], Exception)

    @patch("largeliterarymodels.llm._call_provider")
    def test_errors_dict_empty_on_clean_run(self, mock_call):
        mock_call.return_value = '{"sentiment": "positive", "confidence": 0.9}'
        task = self._task("sentiment_test_errors_clean")
        errors = {}
        results = task.map(["a", "b"], num_workers=1, errors=errors)
        assert all(r is not None for r in results)
        assert errors == {}

    @patch("largeliterarymodels.llm._call_provider")
    def test_duplicate_prompts_each_get_an_entry(self, mock_call):
        # Duplicate prompts share one API call; both indices must be reported.
        mock_call.return_value = 'not json'
        task = self._task("sentiment_test_errors_dup")
        errors = {}
        results = task.map(["same", "same"], num_workers=1, errors=errors)
        assert results == [None, None]
        assert sorted(errors) == [0, 1]
        assert errors[1]["duplicate_of"] == 0

    @patch("largeliterarymodels.llm._call_provider")
    def test_map_without_errors_dict_is_unchanged(self, mock_call):
        mock_call.return_value = 'not json'
        task = self._task("sentiment_test_errors_optional")
        assert task.map(["a"], num_workers=1) == [None]


class TestUsageAndFailFast:
    """Token usage has to be a receipt, and a systematic failure has to stop
    rather than bill every retry."""

    def _task(self, name, **kw):
        class T(Task):
            schema = Sentiment
            system_prompt = "Assess sentiment."
        T.name = name
        task = T(**kw)
        task._stash = HashStash(engine="memory").clear()
        return task

    @patch("largeliterarymodels.llm._call_provider")
    def test_usage_is_recorded_from_the_provider(self, mock_call):
        def provider(**kw):
            kw["usage_sink"]({"input_tokens": 12, "output_tokens": 30,
                              "cache_read_tokens": 900, "cache_write_tokens": 0})
            return '{"sentiment": "positive", "confidence": 0.9}'
        mock_call.side_effect = provider

        task = self._task("usage_test")
        task.run("a")
        task.run("b")
        r = task.usage.report()
        assert r["calls"] == 2
        assert r["cache_read_tokens"] == 1800
        assert r["output_tokens"] == 60
        # prompt = fresh + read + write, not just input_tokens
        assert r["prompt_tokens"] == 2 * 912
        assert r["cache_hit_rate"] == pytest.approx(1800 / 1824)

    @patch("largeliterarymodels.llm._call_provider")
    def test_usage_accumulates_across_run_and_map(self, mock_call):
        def provider(**kw):
            kw["usage_sink"]({"input_tokens": 5, "output_tokens": 10})
            return '{"sentiment": "positive", "confidence": 0.9}'
        mock_call.side_effect = provider
        task = self._task("usage_accum")
        task.run("one")
        task.map(["two", "three"], num_workers=1)
        assert task.usage.calls == 3

    def test_cache_warning_fires_on_zero_observed_caching(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        for _ in range(4):
            u.record({"input_tokens": 3200, "output_tokens": 40})
        warning = u.cache_warning("claude-haiku-4-5")
        assert warning and "no prompt caching observed" in warning
        assert "4,096" in warning

    def test_cache_warning_silent_when_caching_works(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        for _ in range(4):
            u.record({"input_tokens": 10, "cache_read_tokens": 4711})
        assert u.cache_warning("claude-haiku-4-5") is None

    @patch("largeliterarymodels.llm._call_provider")
    def test_per_item_usage_attributes_tokens_to_items(self, mock_call):
        # Output length per item is a free difficulty signal — a longer answer
        # usually means the coder had more to say about that item.
        lengths = {0: 40, 1: 400, 2: 90}

        def provider(**kw):
            n = int(kw["prompt"].split()[-1])
            kw["usage_sink"]({"input_tokens": 5, "output_tokens": lengths[n],
                              "cache_read_tokens": 6863})
            return '{"sentiment": "positive", "confidence": 0.9}'
        mock_call.side_effect = provider

        task = self._task("per_item_test")
        per_item = {}
        task.map(["item 0", "item 1", "item 2"], num_workers=1,
                 per_item_usage=per_item)
        assert sorted(per_item) == [0, 1, 2]
        assert {i: e["output_tokens"] for i, e in per_item.items()} == lengths
        assert per_item[1]["cache_read_tokens"] == 6863
        assert all(e["calls"] == 1 for e in per_item.values())
        # and the aggregate still agrees with the per-item sum
        assert task.usage.output_tokens == sum(lengths.values())

    @patch("largeliterarymodels.llm._call_provider")
    def test_per_item_usage_accumulates_retries(self, mock_call):
        calls = {"n": 0}

        def provider(**kw):
            calls["n"] += 1
            kw["usage_sink"]({"output_tokens": 100})
            if calls["n"] == 1:
                return "not json"          # forces one retry
            return '{"sentiment": "positive", "confidence": 0.9}'
        mock_call.side_effect = provider

        task = self._task("per_item_retry", retries=2)
        per_item = {}
        task.map(["only item"], num_workers=1, per_item_usage=per_item)
        # Both the failed and successful attempt were billed to this item.
        assert per_item[0]["calls"] == 2
        assert per_item[0]["output_tokens"] == 200

    def test_dropped_params_are_recorded_not_just_logged(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        u.record({"output_tokens": 10, "dropped_params": ("temperature",)})
        u.record({"output_tokens": 10, "dropped_params": ("temperature",)})
        u.record({"output_tokens": 10})
        assert u.report()["dropped_params"] == {"temperature": 2}
        # and it is impossible to read the summary without seeing it
        assert "DROPPED PARAMS: temperature x2" in u.summary_line()

    def test_cache_warning_silent_for_non_anthropic(self):
        from largeliterarymodels.llm import UsageTracker
        u = UsageTracker()
        for _ in range(4):
            u.record({"input_tokens": 3200})
        assert u.cache_warning("openai/gpt-5.4-mini") is None

    @patch("largeliterarymodels.llm._call_provider")
    def test_total_failure_aborts_the_batch_quickly(self, mock_call):
        # A provider-shape mismatch fails every item the same way; grinding
        # through 156 items x 3 attempts is what this prevents.
        mock_call.side_effect = AttributeError(
            "'ThinkingBlock' object has no attribute 'text'")
        task = self._task("fail_fast_total", retries=2)
        with pytest.raises(RuntimeError, match="items to complete all failed"):
            task.map([f"item {i}" for i in range(200)], num_workers=1)
        # The unit is item OUTCOMES: floor=5 items, each capped at 2 billed
        # attempts by the identical-repeat cutoff — not 200 items x 3.
        assert mock_call.call_count <= 12, mock_call.call_count

    @patch("largeliterarymodels.llm._call_provider")
    def test_a_rare_tail_does_not_abort_a_healthy_run(self, mock_call):
        """The correction that matters: an absolute count read a 0.05% tail as
        a wiring fault and killed a run that would have finished at 99.9%."""
        ok = '{"sentiment": "positive", "confidence": 0.9}'

        def provider(**kw):
            n = int(kw["prompt"].split()[-1])
            if n in (7, 33, 61, 88, 119):     # 5 failures in 200 items
                raise ValueError("Input should be a valid dictionary")
            return ok
        mock_call.side_effect = provider

        task = self._task("fail_fast_tail", retries=0)
        results = task.map([f"item {i}" for i in range(200)], num_workers=1)
        assert sum(r is not None for r in results) == 195
        assert sum(r is None for r in results) == 5

    @patch("largeliterarymodels.llm._call_provider")
    def test_sustained_rate_aborts(self, mock_call):
        ok = '{"sentiment": "positive", "confidence": 0.9}'

        def provider(**kw):
            n = int(kw["prompt"].split()[-1])
            if n % 2 == 0:                     # 50%, well over the 20% bar
                raise ValueError("Input should be a valid dictionary")
            return ok
        mock_call.side_effect = provider

        task = self._task("fail_fast_rate", retries=0)
        with pytest.raises(RuntimeError, match="over the 20% threshold"):
            task.map([f"item {i}" for i in range(200)], num_workers=1)
        # Armed only after min_attempts, so it does not fire on item 5.
        assert mock_call.call_count >= 30

    @patch("largeliterarymodels.llm._call_provider")
    def test_thresholds_are_tunable(self, mock_call):
        mock_call.side_effect = ValueError("boom")
        task = self._task("fail_fast_tune", retries=0)
        with pytest.raises(RuntimeError):
            task.map([f"i{n}" for n in range(50)], num_workers=1,
                     fail_fast={"floor": 20})
        assert mock_call.call_count >= 20

    def test_breaker_states_its_own_inference(self):
        """A breaker that says only 'too many failures' sends the caller
        hunting for a bug that does not exist."""
        from largeliterarymodels.llm import _Breaker
        b = _Breaker(floor=2)
        for _ in range(2):
            b.record_attempt()
            b.record_failure(ValueError("bad"))
        msg = b.error("claude-sonnet-5")
        assert "Inference:" in msg
        assert "first 2 items to complete all failed" in msg
        assert "fail_fast=False" in msg

    @patch("largeliterarymodels.llm._call_provider")
    def test_identical_repeat_skips_the_remaining_retries(self, mock_call):
        mock_call.side_effect = ValueError("always the same parse failure")
        task = self._task("repeat_test", retries=3)
        with pytest.raises(ValueError, match="after 2 attempts"):
            task.run("x")  # single-item path
        # attempt 1 + one confirming retry, not 1 + 3.
        assert mock_call.call_count == 2

    @patch("largeliterarymodels.llm._call_provider")
    def test_differing_failures_still_retry(self, mock_call):
        mock_call.side_effect = [
            "not json at all",
            '{"sentiment": "positive", "confidence": 0.9}',
        ]
        task = self._task("differing_test", retries=2)
        assert task.run("x").sentiment == "positive"

    @patch("largeliterarymodels.llm._call_provider")
    def test_fail_fast_disabled_runs_every_item(self, mock_call):
        mock_call.side_effect = AttributeError("same every time")
        task = self._task("fail_fast_off", retries=0)
        results = task.map([f"i{n}" for n in range(6)], num_workers=1,
                           fail_fast=False)
        assert results == [None] * 6
        assert mock_call.call_count == 6


class TestResultsHistory:
    """A force= rerun appends rather than overwrites, so a model's own
    variance is recoverable — the noise floor a cached rerun cannot show."""

    @patch("largeliterarymodels.llm._call_provider")
    def test_force_reruns_are_all_retained(self, mock_call, tmp_path):
        class S(BaseModel):
            verdict: str

        class T(Task):
            name = "history_test"
            schema = S

        task = T()
        task._stash = HashStash(str(tmp_path / "s"), engine="pairtree",
                                append_mode=True)

        mock_call.side_effect = [
            '{"verdict": "yes"}', '{"verdict": "no"}', '{"verdict": "yes"}',
        ]
        task.run("item A")                 # cold
        task.run("item A")                 # cache hit — no new version
        task.run("item A", force=True)     # real recode
        task.run("item A", force=True)     # real recode
        assert mock_call.call_count == 3

        history = [versions for _key, versions in task.results_history]
        assert len(history) == 1, "one key, several versions"
        assert [v.verdict for v in history[0]] == ["yes", "no", "yes"]

        # `results` still shows one row per key — the latest.
        latest = [r.verdict for _, r in task.results]
        assert latest == ["yes"]

    @patch("largeliterarymodels.llm._call_provider")
    def test_single_version_keys_included(self, mock_call, tmp_path):
        class S(BaseModel):
            verdict: str

        class T(Task):
            name = "history_test_single"
            schema = S

        task = T()
        task._stash = HashStash(str(tmp_path / "s2"), engine="pairtree",
                                append_mode=True)
        mock_call.return_value = '{"verdict": "yes"}'
        task.run("only once")
        history = [versions for _key, versions in task.results_history]
        assert len(history) == 1 and len(history[0]) == 1


class TestPartialResponseDiagnosis:
    """Observed on three providers: the model returns the bare value of one
    field instead of the whole object. Pydantic says only 'expected dict, got
    list', so a generic invalid-JSON reprompt tells the model nothing."""

    def _schema(self):
        from typing import Literal

        class DisplacementRelation(BaseModel):
            relations: list[Literal["SEQUENCE", "SPECIFICITY", "CONTRAST"]]
            speech_act: str
            confidence: float
        return DisplacementRelation

    def test_identifies_the_field_from_a_bare_list(self):
        from largeliterarymodels.llm import _diagnose_partial_response
        # lacan's verbatim failing input_value
        assert _diagnose_partial_response(
            ["SEQUENCE", "SPECIFICITY"], self._schema()) == "relations"

    def test_a_proper_object_is_not_a_partial_response(self):
        from largeliterarymodels.llm import _diagnose_partial_response
        assert _diagnose_partial_response(
            {"relations": [], "speech_act": "x", "confidence": 0.5},
            self._schema()) is None

    def test_declines_to_guess_when_ambiguous(self):
        from largeliterarymodels.llm import _diagnose_partial_response

        class TwoLists(BaseModel):
            a: list[str]
            b: list[str]
        # A wrong field name in the reprompt is worse than none.
        assert _diagnose_partial_response(["x"], TwoLists) is None

    def test_list_schemas_are_not_flagged(self):
        from largeliterarymodels.llm import _diagnose_partial_response
        # For list[Model] a bare list is the expected shape.
        assert _diagnose_partial_response([{"sentiment": "x"}],
                                          list[Sentiment]) is None

    def test_retry_prompt_names_the_field(self):
        from largeliterarymodels.llm import _retry_prompt
        generic = _retry_prompt("ITEM")
        targeted = _retry_prompt("ITEM", partial_field="relations")
        assert "not valid JSON" in generic
        assert "`relations`" in targeted
        assert "COMPLETE object" in targeted
        assert "not valid JSON" not in targeted
        assert targeted.endswith("ITEM")

    @patch("largeliterarymodels.llm._call_provider")
    def test_the_targeted_reprompt_is_actually_sent(self, mock_call):
        schema = self._schema()

        class T(Task):
            name = "partial_reprompt"
        T.schema = schema
        task = T(retries=2)
        task._stash = HashStash(engine="memory").clear()

        mock_call.side_effect = [
            '["SEQUENCE", "SPECIFICITY"]',        # partial response
            '{"relations": ["SEQUENCE"], "speech_act": "directive", '
            '"confidence": 0.8}',
        ]
        result = task.run("some passage")
        assert result.speech_act == "directive"
        second_prompt = mock_call.call_args_list[1][1]["prompt"]
        assert "`relations`" in second_prompt


class TestRenderInstrument:
    """The instrument must be administrable outside the API path, without
    hand transcription, and byte-identical to what the API is sent."""

    def _task(self):
        class SentimentTask(Task):
            name = "sentiment_render"
            schema = Sentiment
            system_prompt = "Assess sentiment."
            examples = [
                ("Great!", Sentiment(sentiment="positive", confidence=0.9)),
                ("Awful.", Sentiment(sentiment="negative", confidence=0.85)),
            ]
        return SentimentTask()

    def test_contains_prompt_examples_schema_and_contract(self):
        text = self._task().render_instrument()
        assert "Assess sentiment." in text
        assert "Example 1 input:" in text and "Example 1 output:" in text
        assert "Example 2 input:" in text
        assert '"confidence"' in text          # JSON Schema of the model
        assert "ONLY valid JSON" in text       # output contract
        assert "positive" in text              # example output serialised

    @patch("largeliterarymodels.llm._call_provider")
    def test_byte_identical_to_administered_system_prompt(self, mock_call):
        """The whole point: no drift between rendered and administered."""
        mock_call.return_value = '{"sentiment": "positive", "confidence": 0.9}'
        task = self._task()
        task._stash = HashStash(engine="memory").clear()
        task.run("I love this!")
        sent = mock_call.call_args[1]["system_prompt"]
        assert task.instrument_text() == sent
        assert task.render_instrument() == sent

    def test_item_is_delimited_and_present(self):
        task = self._task()
        text = task.render_instrument(item="A sentence to judge.")
        assert task.ITEM_HEADER in text and task.ITEM_FOOTER in text
        assert "A sentence to judge." in text
        # bare instrument is a prefix of the item-bearing render
        assert text.startswith(task.instrument_text())

    def test_digest_is_opt_in_and_stable(self):
        task = self._task()
        assert "instrument_sha256" not in task.render_instrument()
        assert "instrument_sha256" in task.render_instrument(digest=True)
        assert task.instrument_sha256() == task.instrument_sha256()
        assert len(task.instrument_sha256()) == 64

    def test_digest_changes_when_instrument_changes(self):
        task = self._task()
        before = task.instrument_sha256()
        assert task.instrument_sha256(system_prompt="Judge tone.") != before
        assert task.instrument_sha256(examples=[]) != before

    def test_overrides_reach_the_rendered_text(self):
        task = self._task()
        text = task.render_instrument(system_prompt="Judge tone.", examples=[])
        assert "Judge tone." in text
        assert "Example 1 input:" not in text

    def test_raises_without_schema(self):
        task = Task(name="no_schema_render")
        with pytest.raises(ValueError, match="no schema defined"):
            task.render_instrument()

    def test_list_schema_renders_array_contract(self):
        class MultiTask(Task):
            name = "multi_render"
            schema = list[Sentiment]
            system_prompt = "Assess each."
        text = MultiTask().render_instrument()
        assert "JSON array" in text


class TestParseAndValidate:
    """Hand-administered responses get the same enforcement as tool calls."""

    def _task(self):
        class SentimentTask(Task):
            name = "sentiment_parse"
            schema = Sentiment
        return SentimentTask()

    def test_plain_json(self):
        r = self._task().parse_and_validate(
            '{"sentiment": "positive", "confidence": 0.9}')
        assert isinstance(r, Sentiment) and r.sentiment == "positive"

    def test_fenced_json_with_commentary(self):
        r = self._task().parse_and_validate(
            'Here is my annotation:\n```json\n'
            '{"sentiment": "negative", "confidence": 0.4}\n```\nHope that helps!')
        assert r.sentiment == "negative"

    def test_schema_envelope_is_unwrapped(self):
        r = self._task().parse_and_validate(
            '{"properties": {"sentiment": "neutral", "confidence": 0.5}}')
        assert r.sentiment == "neutral"

    def test_returns_none_on_garbage(self):
        assert self._task().parse_and_validate("I couldn't decide.") is None

    def test_returns_none_on_schema_violation(self):
        # parses as JSON but omits a required field
        assert self._task().parse_and_validate('{"sentiment": "positive"}') is None

    def test_strict_raises(self):
        with pytest.raises(Exception):
            self._task().parse_and_validate("I couldn't decide.", strict=True)

    def test_list_schema(self):
        class MultiTask(Task):
            name = "multi_parse"
            schema = list[Sentiment]
        r = MultiTask().parse_and_validate(
            '[{"sentiment": "positive", "confidence": 0.9}, '
            '{"sentiment": "negative", "confidence": 0.1}]')
        assert len(r) == 2 and r[1].sentiment == "negative"


class TestRetryPrompt:
    @patch("largeliterarymodels.llm._call_provider")
    def test_matches_the_api_paths_retry(self, mock_call):
        """The hand path's retry text must be the API path's retry text."""
        mock_call.side_effect = [
            'not json',
            '{"sentiment": "positive", "confidence": 0.9}',
        ]

        class SentimentTask(Task):
            name = "sentiment_retry"
            schema = Sentiment
            system_prompt = "Assess sentiment."
            retries = 1

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        task.run("I love this!")
        assert mock_call.call_count == 2
        assert mock_call.call_args_list[1][1]["prompt"] == \
            task.retry_prompt("I love this!")


class TestSchemaRepr:
    def test_none(self):
        assert _schema_repr(None) == "None"

    def test_plain(self):
        assert _schema_repr(Sentiment) == "Sentiment"

    def test_list(self):
        assert _schema_repr(list[Sentiment]) == "list[Sentiment]"


# --- BibliographyTask ---

class TestBibliographyTask:
    def test_import(self):
        from largeliterarymodels.tasks import BibliographyTask
        task = BibliographyTask()
        assert task.task_name == "BibliographyTask"
        assert task.retries == 2
        assert task.max_tokens == 16384

    def test_schema_is_list(self):
        from largeliterarymodels.tasks import BibliographyTask
        task = BibliographyTask()
        origin = getattr(task.schema, "__origin__", None)
        assert origin is list

    def test_has_examples(self):
        from largeliterarymodels.tasks import BibliographyTask
        task = BibliographyTask()
        assert len(task.examples) == 3
        for input_text, output in task.examples:
            assert isinstance(input_text, str)
            assert hasattr(output, "model_dump")

    def test_bibliography_entry_fields(self):
        from largeliterarymodels.tasks import BibliographyEntry
        entry = BibliographyEntry(
            author="GREENE, ROBERT",
            title="Greenes Never too late",
            year=1600,
        )
        assert entry.author == "GREENE, ROBERT"
        assert entry.is_translated is False
        assert entry.printer == ""

    def test_bibliography_entry_all_fields(self):
        from largeliterarymodels.tasks import BibliographyEntry
        entry = BibliographyEntry(
            author="BIDPAI",
            title="The morall philosophic of Doni",
            title_sub=": drawne out of the ancient writers",
            year=1601,
            edition="Second edition",
            id_biblio="STC 3054",
            is_translated=True,
            translated_from="",
            translator="Sir Thomas North",
            printer="S. Stafford",
            publisher="",
            bookseller="",
            notes_biblio="First edition in 1570.",
            notes="",
        )
        assert entry.is_translated is True
        assert entry.translator == "Sir Thomas North"

    @patch("largeliterarymodels.llm._call_provider")
    def test_bibliography_task_run(self, mock_call):
        from largeliterarymodels.tasks import BibliographyTask
        mock_call.return_value = '''[{
            "author": "DEKKER, THOMAS",
            "title": "The wonderfull yeare",
            "title_sub": "",
            "year": 1603,
            "edition": "",
            "id_biblio": "STC 6534",
            "is_translated": false,
            "translated_from": "",
            "translator": "",
            "printer": "T. Creede",
            "publisher": "",
            "bookseller": "",
            "notes_biblio": "First of three editions.",
            "notes": ""
        }]'''
        task = BibliographyTask()
        task._stash = HashStash(engine="memory").clear()
        entries = task.run("test chunk")
        assert len(entries) == 1
        assert entries[0].author == "DEKKER, THOMAS"
        assert entries[0].printer == "T. Creede"


class TestBreakerItemSemantics:
    """The breaker's unit is the item OUTCOME. Its two predecessors failed in
    opposite directions: a count killed a healthy tail; an attempts-rate
    double-counted retried items (nominal 20% aborted at 5-12% per-item) and
    its total-failure condition could not fire at all under concurrency."""

    def _task(self, name, retries=0):
        class T(Task):
            schema = Sentiment
            system_prompt = "Assess sentiment."
        T.name = name
        T.retries = retries
        task = T()
        task._stash = HashStash(engine="memory").clear()
        return task

    @patch("largeliterarymodels.llm._call_provider")
    def test_total_failure_fires_under_default_concurrency(self, mock_call):
        """The old condition (count == in-flight attempts) required every
        other worker to be between calls — at num_workers=4 it never fired
        and a 100%-broken batch burned 30 calls instead of ~5 items' worth."""
        mock_call.side_effect = AttributeError("same shape every time")
        task = self._task("breaker_conc", retries=0)
        with pytest.raises(RuntimeError, match="items to complete all failed"):
            task.map([f"i{n}" for n in range(300)], num_workers=4)
        assert mock_call.call_count <= 12, mock_call.call_count

    @patch("largeliterarymodels.llm._call_provider")
    def test_ten_percent_tail_completes(self, mock_call):
        """A uniform 10% per-item tail sits well under the 20% bar — the
        attempts-denominator version aborted this run 7 times in 8."""
        ok = '{"sentiment": "positive", "confidence": 0.9}'

        def provider(**kw):
            n = int(kw["prompt"].split()[-1])
            if n % 10 == 3:
                raise ValueError("Input should be a valid dictionary")
            return ok
        mock_call.side_effect = provider
        task = self._task("breaker_tail10", retries=1)
        results = task.map([f"item {n}" for n in range(400)], num_workers=1)
        assert sum(r is None for r in results) == 40

    @patch("largeliterarymodels.llm._call_provider")
    def test_burst_that_retries_into_success_is_a_success(self, mock_call):
        """Six consecutive 429s at batch start must not read as systematic:
        an item whose retry lands is a success, not half a failure."""
        ok = '{"sentiment": "positive", "confidence": 0.9}'
        seen = {}

        def provider(**kw):
            key = kw["prompt"].split()[-1]
            n = int(key)
            if n < 6 and seen.setdefault(key, 0) == 0:
                seen[key] += 1
                raise RuntimeError("429 rate limited")
            return ok
        mock_call.side_effect = provider
        task = self._task("breaker_burst", retries=1)
        results = task.map([f"item {n}" for n in range(100)], num_workers=1)
        assert all(r is not None for r in results)

    @patch("largeliterarymodels.llm._call_provider")
    def test_fail_fast_int_is_honoured_as_a_floor(self, mock_call):
        """fail_fast=50 was silently identical to fail_fast=5: the int's
        VALUE vanished into bool(fail_fast). The docstring promised the int."""
        mock_call.side_effect = AttributeError("same")
        task = self._task("breaker_int", retries=0)
        with pytest.raises(RuntimeError):
            task.map([f"i{n}" for n in range(80)], num_workers=1,
                     fail_fast=50)
        assert mock_call.call_count >= 50, mock_call.call_count

    @patch("largeliterarymodels.llm._call_provider")
    def test_fail_fast_empty_dict_means_defaults_not_disabled(self, mock_call):
        """{} is the natural spelling of 'use the defaults'; it used to
        evaluate falsy and switch the breaker OFF entirely."""
        mock_call.side_effect = AttributeError("same")
        task = self._task("breaker_empty", retries=0)
        with pytest.raises(RuntimeError):
            task.map([f"i{n}" for n in range(80)], num_workers=1,
                     fail_fast={})
        assert mock_call.call_count < 80

    def test_fail_fast_junk_raises(self):
        task = self._task("breaker_junk")
        with pytest.raises(TypeError, match="fail_fast"):
            list(task.imap(["a"], fail_fast="soon"))

    @patch("largeliterarymodels.llm._call_provider")
    def test_min_attempts_alias_still_means_min_outcomes(self, mock_call):
        mock_call.side_effect = ValueError("boom")
        task = self._task("breaker_alias", retries=0)
        with pytest.raises(RuntimeError):
            task.map([f"i{n}" for n in range(30)], num_workers=1,
                     fail_fast={"min_attempts": 10, "floor": 99})
        assert mock_call.call_count <= 15

    @patch("largeliterarymodels.llm._call_provider")
    def test_abort_fabricates_no_error_entries(self, mock_call):
        """After an abort, len(errors) must be the failure count. It used to
        include every never-attempted item as {'attempts': 0, 'exception':
        None} — 131 'failures' on a run with 3."""
        mock_call.side_effect = ValueError("boom")
        task = self._task("abort_errors", retries=0)
        errors = {}
        with pytest.raises(RuntimeError):
            task.map([f"i{n}" for n in range(60)], num_workers=4,
                     errors=errors)
        assert errors, "the real failures must still be recorded"
        assert all(e["exception"] is not None and e["attempts"] >= 1
                   for e in errors.values()), errors

    @patch("largeliterarymodels.llm._call_provider")
    def test_abort_carries_partial_results(self, mock_call):
        from largeliterarymodels.llm import BatchAborted
        ok = '{"sentiment": "positive", "confidence": 0.9}'

        def provider(**kw):
            n = int(kw["prompt"].split()[-1])
            if n >= 10:
                raise ValueError("boom")
            return ok
        mock_call.side_effect = provider
        task = self._task("abort_partial", retries=0)
        with pytest.raises(BatchAborted) as exc_info:
            task.map([f"item {n}" for n in range(60)], num_workers=1,
                     fail_fast={"min_outcomes": 15})
        # The exception carries what the consumer loop had yielded when the
        # trip landed; the FULL set of completed work is in the stash, which
        # is the resumability guarantee the abort message states.
        assert isinstance(exc_info.value.results, list)
        assert len(exc_info.value.results) == 60
        assert len(list(task.stash.keys())) == 10, \
            "every success must be recoverable from the stash"

    def test_signature_separates_kinds_not_indices(self):
        from largeliterarymodels.llm import _Breaker
        s = _Breaker.signature
        assert s(ValueError("index 42 bad")) == s(ValueError("index 7 bad"))
        assert s(ValueError("index 42 bad")) != s(TypeError("index 42 bad"))
        assert s(ValueError("index 42 bad")) != s(ValueError("wholly other"))

    def test_signature_survives_wide_schema_field_names(self):
        """120-char truncation collapsed every distinct missing-field error
        on a wide schema into one 'systematic' signature."""
        from largeliterarymodels.llm import _Breaker
        prefix = ("1 validation error for PassageForm\n  Field required "
                  "[type=missing, input_value={...}] " + "x" * 80)
        a = _Breaker.signature(ValueError(prefix + " field_alpha"))
        b = _Breaker.signature(ValueError(prefix + " field_beta"))
        assert a != b


class TestWarmCachePath:
    """The warm-cache branch (num_workers>1, >num_workers items, >2000-char
    system prompt) had never executed in CI — every prior test failed at
    least one gate — while production batches take it every time."""

    def _task(self, name, retries=0):
        class T(Task):
            schema = Sentiment
            system_prompt = "z" * 3000
        T.name = name
        T.retries = retries
        task = T()
        task._stash = HashStash(engine="memory").clear()
        return task

    @patch("largeliterarymodels.llm._call_provider")
    def test_every_index_yields_exactly_once(self, mock_call):
        ok = '{"sentiment": "positive", "confidence": 0.9}'
        mock_call.return_value = ok
        task = self._task("warm_all")
        results = task.map([f"u{n}" for n in range(12)], num_workers=4)
        assert all(r is not None for r in results) and len(results) == 12

    @patch("largeliterarymodels.llm._call_provider")
    def test_warm_item_duplicate_failure_recorded_for_both(self, mock_call):
        """Whether a failed item's duplicates appeared in `errors` used to
        depend on whether the item happened to be the warm one."""
        ok = '{"sentiment": "positive", "confidence": 0.9}'

        def provider(**kw):
            if kw["prompt"].endswith("dup"):
                raise ValueError("dup fails")
            return ok
        mock_call.side_effect = provider
        task = self._task("warm_dupes")
        errors = {}
        results = task.map(["dup", "dup"] + [f"u{n}" for n in range(20)],
                           num_workers=4, errors=errors, fail_fast=False)
        assert results[0] is None and results[1] is None
        assert sorted(errors) == [0, 1], sorted(errors)
        assert errors[1]["duplicate_of"] == 0

    @patch("largeliterarymodels.llm._call_provider")
    def test_duplicates_get_provenance_rows(self, mock_call):
        """A per-annotation provenance table must not have holes that read
        as 'unknown model' for rows that shared a de-duplicated call."""
        ok = '{"sentiment": "positive", "confidence": 0.9}'

        def provider(**kw):
            kw["usage_sink"]({"input_tokens": 5, "output_tokens": 7,
                              "response_model": "served-model-1"})
            return ok
        mock_call.side_effect = provider
        task = self._task("warm_prov")
        per_item = {}
        task.map(["dup", "dup"] + [f"u{n}" for n in range(20)],
                 num_workers=4, per_item_usage=per_item)
        assert per_item[1]["duplicate_of"] == 0
        assert per_item[1]["response_model"] == "served-model-1"
        assert per_item[1]["calls"] == 0, "no tokens were spent for the twin"
        assert per_item[0]["response_model"] == "served-model-1"


class TestStaleParsedDoesNotLeak:
    @patch("largeliterarymodels.llm._call_provider")
    def test_network_error_gets_the_generic_reprompt(self, mock_call):
        """Attempt 0 returns a bare list (partial diagnosis), attempt 1
        raises a network error. Attempt 2's reprompt must be the generic
        one — the stale `parsed` used to accuse the model of a bare-field
        response it never made on attempt 1."""
        prompts = []

        def provider(**kw):
            prompts.append(kw["prompt"])
            n = len(prompts)
            if n == 1:
                return '["positive", "negative"]'
            if n == 2:
                raise ConnectionError("reset by peer")
            return '{"sentiment": "positive", "confidence": 0.9}'
        mock_call.side_effect = provider

        class T(Task):
            schema = Sentiment
            system_prompt = "Assess."
        T.name = "stale_parsed"
        T.retries = 3
        task = T()
        task._stash = HashStash(engine="memory").clear()
        assert task.run("the item").sentiment == "positive"
        assert "not valid JSON" in prompts[2]
        assert "contained only the value" not in prompts[2]


class TestPermissiveFieldNotBlamed:
    def test_any_field_is_not_evidence(self):
        """A permissive field matches when nothing else does; naming it in
        the reprompt accuses the wrong field with confidence."""
        from typing import Any
        from largeliterarymodels.llm import _diagnose_partial_response

        class WithAny(BaseModel):
            themes: list[str]
            speaker: str
            extra: Any

        assert _diagnose_partial_response(7, WithAny) is None
        assert _diagnose_partial_response(
            [{"themes": ["a"], "speaker": "x"}], WithAny) is None
        # The targeted diagnosis still works when the match is genuine.
        assert _diagnose_partial_response(["a", "b"], WithAny) == "themes"


class TestCacheKeyProvenance:
    """The stash key is the durable artifact: thinking-on and thinking-off
    output must not share one, and a key must not assert a temperature the
    model rejected."""

    def _llm(self, model):
        from largeliterarymodels.llm import LLM
        from tests.test_regressions import FakeStash
        return LLM(model, stash=FakeStash())

    @patch("largeliterarymodels.llm._call_provider", return_value='{"x": 1}')
    def test_thinking_state_splits_the_key(self, mock_call):
        from tests.test_regressions import Out
        llm = self._llm("deepseek/deepseek-v4-pro")
        llm.extract("p", Out, retries=0)
        llm.extract("p", Out, retries=0, thinking={"type": "enabled"})
        assert mock_call.call_count == 2, \
            "an enabled-thinking rerun must not be served the disabled cache"
        assert len(llm.stash) == 2

    @patch("largeliterarymodels.llm._call_provider", return_value='{"x": 1}')
    def test_pre_thinking_families_keep_their_keys(self, mock_call):
        """No thinking parameter sent -> no thinking field in the key, so
        the existing annotation stock stays reachable."""
        from tests.test_regressions import Out
        llm = self._llm("lmstudio/qwen3.5-35b-a3b")
        llm.extract("p", Out, retries=0)
        (frozen,) = llm.stash.keys()
        assert '"thinking"' not in frozen

    @patch("largeliterarymodels.llm._call_provider", return_value='{"x": 1}')
    def test_rejected_temperature_is_not_asserted_in_the_key(self, mock_call):
        from tests.test_regressions import Out
        llm = self._llm("claude-sonnet-5")
        llm.extract("p", Out, retries=0, temperature=0.0)
        (frozen,) = llm.stash.keys()
        assert '"temperature": null' in frozen, frozen

    @patch("largeliterarymodels.llm._call_provider", return_value='{"x": 1}')
    def test_custom_cache_key_gains_the_coder_identity(self, mock_call):
        """Two models against the same work-unit key used to write one
        history — results_history then read cross-model disagreement as one
        model's own variance, under a docstring recommending exactly that
        reading."""
        from tests.test_regressions import Out
        from tests.test_regressions import FakeStash
        from largeliterarymodels.llm import LLM
        stash = FakeStash()
        LLM("claude-sonnet-4-6", stash=stash).extract(
            "p", Out, retries=0, cache_key={"id": "item1"})
        LLM("lmstudio/qwen3.5-35b-a3b", stash=stash).extract(
            "p", Out, retries=0, cache_key={"id": "item1"})
        assert len(stash) == 2, "two coders, two histories"

    def test_keys_that_already_carry_a_model_are_untouched(self):
        """SequentialTask chunk keys already include model (sometimes None);
        setdefault must leave them byte-identical or in-flight batch caches
        orphan mid-run."""
        from largeliterarymodels.llm import _custom_key
        original = {"task": "t", "text_id": "x", "chunk": 3, "model": None}
        assert _custom_key(dict(original), "claude-sonnet-5") == original


class TestBatchScopedCacheWarning:
    @patch("largeliterarymodels.llm._call_provider")
    def test_second_uncached_batch_still_warns(self, mock_call, caplog):
        """One well-cached batch used to immunise the Task against the
        no-caching warning for the rest of the process: the finally block
        read the LIFETIME tracker, whose early reads never went away."""
        ok = '{"sentiment": "positive", "confidence": 0.9}'

        def cached_provider(**kw):
            kw["usage_sink"]({"input_tokens": 50, "output_tokens": 5,
                              "cache_read_tokens": 5000})
            return ok

        def uncached_provider(**kw):
            kw["usage_sink"]({"input_tokens": 5050, "output_tokens": 5,
                              "cache_read_tokens": 0,
                              "cache_write_tokens": 0})
            return ok

        class T(Task):
            schema = Sentiment
            system_prompt = "Assess."
        T.name = "batch_scope"
        T.retries = 0
        task = T()
        task._stash = HashStash(engine="memory").clear()
        import logging
        mock_call.side_effect = cached_provider
        task.map([f"a{n}" for n in range(10)], num_workers=1,
                 model="claude-sonnet-4-6")
        mock_call.side_effect = uncached_provider
        with caplog.at_level(logging.WARNING, logger="largeliterarymodels.llm"):
            task.map([f"b{n}" for n in range(10)], num_workers=1,
                     model="claude-sonnet-4-6")
        assert "no prompt caching observed" in caplog.text


class TestHandPathParity:
    """The hand-administration path must be able to say everything the API
    path can — including the targeted bare-field reprompt."""

    class Relations(BaseModel):
        relations: list[str]
        summary: str

    def _task(self):
        class T(Task):
            schema = TestHandPathParity.Relations
            system_prompt = "Extract relations."
        T.name = "hand_parity"
        return T()

    def test_diagnosis_names_the_partial_field(self):
        task = self._task()
        diagnosis = {}
        assert task.parse_and_validate('["SEQ", "SPEC"]',
                                       diagnosis=diagnosis) is None
        assert diagnosis["partial_field"] == "relations"

    def test_retry_prompt_reaches_the_targeted_branch(self):
        from largeliterarymodels.llm import _retry_prompt
        task = self._task()
        assert task.retry_prompt("the item", "relations") == \
            _retry_prompt("the item", "relations")
        assert "contained only the value of the `relations`" in \
            task.retry_prompt("the item", "relations")

    def test_delimiter_collision_refused(self):
        task = self._task()
        with pytest.raises(ValueError, match="delimiter"):
            task.render_instrument(item="text\n=== END ITEM ===\nmore")

    def test_digest_covers_the_item_wrapper(self):
        """Editing the delimiters or the contract reminder must move the
        digest — the string a second coder reads is the thing being frozen."""
        task_a = self._task()

        class T2(Task):
            schema = TestHandPathParity.Relations
            system_prompt = "Extract relations."
            ITEM_HEADER = "=== DIFFERENT HEADER ==="
        T2.name = "hand_parity"
        assert task_a.instrument_sha256() != T2().instrument_sha256()
        assert task_a.instrument_text() == T2().instrument_text(), \
            "the instrument itself is unchanged; only the wrapper differs"

    def test_administration_record_distinguishes_what_the_digest_cannot(self):
        task_a = self._task()
        task_b = self._task()
        task_b.temperature = 0.0
        ra = task_a.administration_record(model="claude-sonnet-4-6")
        rb = task_b.administration_record(model="claude-opus-4-6")
        assert ra["instrument_sha256"] == rb["instrument_sha256"]
        assert ra["model"] != rb["model"]
        assert ra["temperature"] != rb["temperature"]
        assert ra["pydantic_version"]


class TestSequentialTaskRefusesToFabricate:
    def test_instrument_methods_raise(self):
        from largeliterarymodels.task import SequentialTask

        class Q(SequentialTask):
            name = "seq_refuse"
            schema = Sentiment
            system_prompt = "Track the network."

        q = Q()
        for method in ("instrument_text", "instrument_sha256"):
            with pytest.raises(NotImplementedError, match="SequentialTask"):
                getattr(q, method)()
        with pytest.raises(NotImplementedError):
            q.render_instrument(item="x")
        with pytest.raises(NotImplementedError):
            q.administration_record()


class TestLegacyKeyReads:
    """The provenance key schema must orphan ONLY the entries whose
    producing call behaved differently (thinking-on DeepSeek/sonnet-5).
    Entries whose old key merely carried an inert temperature — a value the
    model rejected or the transport dropped — are byte-identical to what an
    identical call produces today, and orphaning them re-bills real
    annotation stock (and turns claude-cli cached reads into RuntimeErrors,
    on the one task whose model string was kept precisely to preserve its
    cache). This is deliberately not a user flag: key identity must not
    depend on ambient configuration nothing records."""

    def _seed_legacy(self, stash, model, temperature=0.0):
        from largeliterarymodels.llm import (_build_extract_prompt, _make_key,
                                             _schema_name)
        from tests.test_regressions import Out
        full_system, user_prompt = _build_extract_prompt("p", Out)
        old_key = _make_key(user_prompt, model, full_system, temperature,
                            4096, schema_name=_schema_name(Out))
        stash[old_key] = '{"x": 42}'
        return old_key

    def _llm(self, model):
        from largeliterarymodels.llm import LLM
        from tests.test_regressions import FakeStash
        return LLM(model, stash=FakeStash())

    @patch("largeliterarymodels.llm._call_provider")
    def test_fable_reads_its_pre_schema_cache(self, mock_call):
        """Fable's thinking was on then and is on now (cannot disable) —
        the old entry is what an identical call produces today."""
        from tests.test_regressions import Out
        llm = self._llm("claude-fable-5")
        self._seed_legacy(llm.stash, "claude-fable-5")
        result = llm.extract("p", Out, retries=0, temperature=0.0)
        assert result.x == 42
        assert mock_call.call_count == 0, "must not re-bill the annotation"

    @patch("largeliterarymodels.llm._call_provider")
    def test_opus_4_7_reads_its_pre_schema_cache(self, mock_call):
        from tests.test_regressions import Out
        llm = self._llm("claude-opus-4-7")
        self._seed_legacy(llm.stash, "claude-opus-4-7")
        assert llm.extract("p", Out, retries=0, temperature=0.0).x == 42
        assert mock_call.call_count == 0

    def test_claude_cli_cached_reads_do_not_touch_the_disabled_provider(
            self, monkeypatch):
        """A cache miss on claude-cli raises through the disabled provider;
        the legacy read must satisfy it before routing happens at all."""
        from tests.test_regressions import Out
        monkeypatch.delenv("LITMOD_ALLOW_CLAUDE_CLI", raising=False)
        llm = self._llm("claude-cli/sonnet")
        self._seed_legacy(llm.stash, "claude-cli/sonnet")
        assert llm.extract("p", Out, retries=0, temperature=0.0).x == 42

    @patch("largeliterarymodels.llm._call_provider", return_value='{"x": 1}')
    def test_sonnet_5_thinking_on_entries_stay_orphaned(self, mock_call):
        """The old sonnet-5 entry is thinking-on output with uncontrolled
        sampling; the thinking-off call must NOT be served it."""
        from tests.test_regressions import Out
        llm = self._llm("claude-sonnet-5")
        self._seed_legacy(llm.stash, "claude-sonnet-5")
        assert llm.extract("p", Out, retries=0, temperature=0.0).x == 1
        assert mock_call.call_count == 1, "the orphan must not be served"

    @patch("largeliterarymodels.llm._call_provider", return_value='{"x": 1}')
    def test_deepseek_thinking_on_entries_stay_orphaned(self, mock_call):
        from tests.test_regressions import Out
        llm = self._llm("deepseek/deepseek-v4-pro")
        self._seed_legacy(llm.stash, "deepseek/deepseek-v4-pro",
                          temperature=0.7)
        assert llm.extract("p", Out, retries=0, temperature=0.7).x == 1
        assert mock_call.call_count == 1

    @patch("largeliterarymodels.llm._call_provider")
    def test_legacy_hit_is_copied_forward(self, mock_call):
        """One legacy read per item: the second read must hit the NEW key."""
        from tests.test_regressions import Out
        llm = self._llm("claude-fable-5")
        old_key = self._seed_legacy(llm.stash, "claude-fable-5")
        llm.extract("p", Out, retries=0, temperature=0.0)
        del llm.stash[llm.stash._k(old_key)]
        assert llm.extract("p", Out, retries=0, temperature=0.0).x == 42
        assert mock_call.call_count == 0

    @patch("largeliterarymodels.llm._call_provider")
    def test_batch_path_reads_legacy_too(self, mock_call):
        from tests.test_regressions import Out
        llm = self._llm("claude-opus-4-8")
        self._seed_legacy(llm.stash, "claude-opus-4-8")
        results = dict(llm.extract_imap(["p"], Out, retries=0,
                                        temperature=0.0))
        assert results[0].x == 42
        assert mock_call.call_count == 0

    @patch("largeliterarymodels.llm._call_provider", return_value='{"x": 1}')
    def test_custom_cache_keys_get_no_fallback(self, mock_call):
        """A bare pre-schema custom key could hold ANY model's annotation —
        that ambiguity is the defect, not an inert field."""
        from tests.test_regressions import Out, FakeStash
        from largeliterarymodels.llm import LLM
        stash = FakeStash()
        stash[{"id": "item1"}] = '{"x": 42}'
        llm = LLM("claude-fable-5", stash=stash)
        assert llm.extract("p", Out, retries=0,
                           cache_key={"id": "item1"}).x == 1
        assert mock_call.call_count == 1

    def test_deepseek_thinking_none_key_records_no_temperature(self):
        """thinking=None on DeepSeek means take the default, which reasons
        and ignores temperature — the key must not assert one. The
        anything-but-disabled condition, not enabled-only."""
        from largeliterarymodels.providers import effective_temperature
        assert effective_temperature("deepseek/deepseek-v4-pro", 0.7,
                                     None) is None
        assert effective_temperature("deepseek/deepseek-v4-pro", 0.7,
                                     "auto") == 0.7


class TestClosingReviewFixesLLM:
    def _task(self, name, system_prompt="Assess.", retries=0):
        class T(Task):
            schema = Sentiment
        T.name = name
        T.system_prompt = system_prompt
        T.retries = retries
        task = T()
        task._stash = HashStash(engine="memory").clear()
        return task

    @patch("largeliterarymodels.llm._call_provider")
    def test_fail_fast_zero_means_off(self, mock_call):
        """0 is the falsy spelling of 'off' (False and None both disable);
        as a floor it meant abort on the first failed item."""
        mock_call.side_effect = ValueError("boom")
        task = self._task("ff_zero")
        results = task.map([f"i{n}" for n in range(6)], num_workers=1,
                           fail_fast=0)
        assert results == [None] * 6
        assert mock_call.call_count == 6

    @patch("largeliterarymodels.llm._call_provider")
    def test_warm_path_abort_still_logs_the_batch_summary(self, mock_call,
                                                          caplog):
        """An abort raised in the warm-cache pre-flight used to skip the
        finally that logs the batch receipt — lost for exactly the runs
        that need it most."""
        import logging
        mock_call.side_effect = ValueError("boom")
        task = self._task("warm_abort_summary", system_prompt="z" * 3000)
        with caplog.at_level(logging.INFO, logger="largeliterarymodels.llm"):
            with pytest.raises(RuntimeError):
                task.map([f"i{n}" for n in range(12)], num_workers=4,
                         fail_fast=1)
        assert "extract_imap usage:" in caplog.text
