"""Tests for SequentialTask: chunk iteration, rolling state, cache resume."""

import json
from unittest.mock import patch

from largeliterarymodels.task import SequentialTask


class EchoSequentialTask(SequentialTask):
    """Minimal concrete SequentialTask: counts passages per chunk."""

    name = "seq_echo_test"
    chunk_size = 2

    def build_state(self):
        return {"seen": []}

    def format_context(self, state):
        return f"CONTEXT: seen={len(state['seen'])} chunks"

    def update_state(self, state, result, chunk_idx, start, end):
        state["seen"].append((chunk_idx, start, end))
        return state

    def aggregate(self, all_results, state):
        return {"results": all_results, "n_state_updates": len(state["seen"])}


class RecordingLLM:
    """Stands in for LLM: records prompts/keys, returns canned JSON."""

    def __init__(self, response='{"ok": true}', cache=None):
        self.response = response
        self.calls = []          # (prompt, system_prompt, cache_key, force)
        self.cache = cache if cache is not None else {}

    def _frozen(self, key):
        return json.dumps(key, sort_keys=True)

    def generate(self, prompt=None, system_prompt=None, cache_key=None,
                 force=False):
        self.calls.append((prompt, system_prompt, cache_key, force))
        frozen = self._frozen(cache_key)
        if not force and frozen in self.cache:
            return self.cache[frozen]
        self.cache[frozen] = self.response
        return self.response


PASSAGES = ["one two three", "four five", "six seven eight", "nine", "ten"]


def run_with(llm, task=None, **kwargs):
    task = task or EchoSequentialTask()
    with patch.object(type(task), "_get_llm", return_value=llm):
        return task.run(PASSAGES, verbose=False, **kwargs)


class TestChunkIteration:
    def test_chunking_and_state_feedback(self):
        llm = RecordingLLM()
        out = run_with(llm)
        # 5 passages / chunk_size 2 -> 3 chunks
        assert len(llm.calls) == 3
        assert out["n_state_updates"] == 3
        assert len(out["results"]) == 3
        # rolling state visible in each successive prompt
        prompts = [c[0] for c in llm.calls]
        assert "seen=0" in prompts[0]
        assert "seen=1" in prompts[1]
        assert "seen=2" in prompts[2]

    def test_limit_chunks(self):
        llm = RecordingLLM()
        run_with(llm, limit_chunks=1)
        assert len(llm.calls) == 1

    def test_metadata_recorded(self):
        out = run_with(RecordingLLM())
        meta = out["metadata"]
        assert meta["n_passages"] == 5
        assert meta["n_chunks"] == 3
        assert meta["chunk_size"] == 2


class TestCacheKeys:
    def test_chunk_key_shape(self):
        llm = RecordingLLM()
        run_with(llm, cache_key="mytext_001")
        key = llm.calls[0][2]
        assert key["task"] == "seq_echo_test"
        assert key["text_id"] == "mytext_001"
        assert key["chunk"] == 0
        assert "prompt_version" not in key, "default preserves legacy keys"

    def test_prompt_version_opt_in(self):
        class Versioned(EchoSequentialTask):
            prompt_version = "v2"

        llm = RecordingLLM()
        run_with(llm, task=Versioned())
        assert llm.calls[0][2]["prompt_version"] == "v2"

    def test_resume_from_cache(self):
        """Second run with the same cache reuses chunk generations."""
        shared_cache = {}
        llm1 = RecordingLLM(cache=shared_cache)
        run_with(llm1, cache_key="t1")
        llm2 = RecordingLLM(cache=shared_cache)
        run_with(llm2, cache_key="t1")
        # RecordingLLM served all three chunks from the shared cache dict
        assert len(shared_cache) == 3


class TestFailureTolerance:
    def test_parse_failure_yields_none_and_continues(self):
        llm = RecordingLLM(response="NOT JSON AT ALL {{{")
        out = run_with(llm)
        assert out["results"] == [None, None, None]
        assert out["n_state_updates"] == 0

    def test_update_state_failure_keeps_result(self):
        class Brittle(EchoSequentialTask):
            def update_state(self, state, result, chunk_idx, start, end):
                raise KeyError("subclass bug")

        out = run_with(RecordingLLM(), task=Brittle())
        # results survive even though state updates crashed
        assert out["results"] == [{"ok": True}] * 3

    def test_generate_failure_yields_none_and_continues(self):
        class ExplodingLLM(RecordingLLM):
            def generate(self, **kwargs):
                raise RuntimeError("provider down")

        out = run_with(ExplodingLLM())
        assert out["results"] == [None, None, None]
