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
