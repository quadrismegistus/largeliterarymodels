"""Batch transport tests — fake adapters, no paid calls.

The constraint that decides whether the feature ships is KEY IDENTITY: a
batch-computed annotation must be indistinguishable from a streamed one,
warm-readable by either path. Everything else is money-safety (the
ledger) and receipt fidelity.
"""

import json

import pytest
from unittest.mock import patch
from pydantic import BaseModel

import largeliterarymodels.batch as B
from largeliterarymodels.llm import LLM
from tests.test_regressions import FakeStash, Out


class FakeAdapter:
    """Scriptable stand-in for a provider batch endpoint."""

    provider = "anthropic"

    def __init__(self, respond=None, fail_cids=(), omit_cids=()):
        self.submitted = []
        self.respond = respond or (lambda cid, req: '{"x": 1}')
        self.fail_cids = set(fail_cids)
        self.omit_cids = set(omit_cids)
        self.batch_ids = []

    def submit(self, requests):
        self.submitted.append(requests)
        bid = f"fake-batch-{len(self.batch_ids)}"
        self.batch_ids.append((bid, requests))
        return bid

    def is_done(self, batch_id):
        return True

    def results(self, batch_id):
        _, requests = next(b for b in self.batch_ids if b[0] == batch_id)
        for cid, req in requests:
            if cid in self.omit_cids:
                continue
            if cid in self.fail_cids:
                yield cid, False, None, None, "errored"
                continue
            usage = {"input_tokens": 10, "output_tokens": 5,
                     "cache_read_tokens": 0, "cache_write_tokens": 0,
                     "reasoning_tokens": 0, "reasoning_reported": False,
                     "reasoning_observed": False,
                     "response_model": "served-batch-model"}
            yield cid, True, self.respond(cid, req), usage, None


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger" / "ledger.jsonl"


def _llm():
    return LLM("claude-sonnet-4-6", temperature=0.0, max_tokens=64,
               stash=FakeStash())


def _run(llm, fake, prompts, monkeypatch, **kw):
    monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
    kw.setdefault("probe", False)
    kw.setdefault("poll_interval", 0)
    return B.extract_batch(llm, prompts, Out, system_prompt="sys", **kw)


class TestKeyIdentity:
    def test_batch_results_serve_the_streaming_path_warm(self, ledger,
                                                         monkeypatch):
        """The identity guarantee, both directions: batch writes what
        streaming reads. Zero provider calls on the re-read."""
        llm = _llm()
        fake = FakeAdapter()
        results = _run(llm, fake, ["a", "b", "c"], monkeypatch)
        assert all(r is not None and r.x == 1 for r in results)
        with patch("largeliterarymodels.llm._call_provider") as mock_call:
            again = llm.extract_map(["a", "b", "c"], Out,
                                    system_prompt="sys")
        assert [r.x for r in again] == [1, 1, 1]
        assert mock_call.call_count == 0

    def test_streamed_items_are_not_resubmitted(self, ledger, monkeypatch):
        """The other direction: a half-warm stash batches only its cold
        half."""
        llm = _llm()
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 9}'):
            llm.extract("a", Out, system_prompt="sys", retries=0)
        fake = FakeAdapter()
        results = _run(llm, fake, ["a", "b"], monkeypatch)
        assert results[0].x == 9, "the streamed annotation must be served"
        assert len(fake.submitted[0]) == 1, "only the cold item submits"

    def test_duplicates_share_one_request(self, ledger, monkeypatch):
        llm = _llm()
        fake = FakeAdapter()
        results = _run(llm, fake, ["same", "same", "other"], monkeypatch)
        assert len(fake.submitted[0]) == 2
        assert results[0].x == results[1].x == 1


class TestLedger:
    def test_open_batch_is_resumed_not_resubmitted(self, ledger,
                                                   monkeypatch, caplog):
        """The money-safety core: same items, second call, one
        submission."""
        llm = _llm()
        fake = FakeAdapter()
        _run(llm, fake, ["a", "b"], monkeypatch)
        # Simulate the first run dying between submit and collect: reopen
        # the ledger row (collect appended 'closed'; drop that line).
        lines = [l for l in open(ledger).read().splitlines()
                 if json.loads(l)["status"] != "closed"]
        open(ledger, "w").write("\n".join(lines) + "\n")
        llm2 = _llm()  # fresh stash: items are cold again
        results = _run(llm2, fake, ["a", "b"], monkeypatch)
        assert len(fake.submitted) == 1, \
            "the open batch must be resumed, never resubmitted"
        assert "resuming OPEN batch" in caplog.text
        assert all(r.x == 1 for r in results)

    def test_submitting_without_id_stops_loudly(self, ledger, monkeypatch):
        """The one ambiguous state guesses nothing: the provider may hold
        a live, billable batch."""
        llm = _llm()
        fake = FakeAdapter()

        class Dies(FakeAdapter):
            def submit(self, requests):
                raise KeyboardInterrupt("process dies mid-submit")

        with pytest.raises(KeyboardInterrupt):
            _run(llm, Dies(), ["a", "b"], monkeypatch)
        with pytest.raises(B.AmbiguousBatchState):
            _run(llm, fake, ["a", "b"], monkeypatch)

    def test_handle_survives_process_death(self, ledger, monkeypatch):
        """wait=False, 'die', reconstruct from the ledger, collect."""
        llm = _llm()
        fake = FakeAdapter()
        handle = _run(llm, fake, ["a", "b"], monkeypatch, wait=False)
        assert isinstance(handle, B.BatchHandle)
        fresh = B.BatchHandle.from_ledger(handle.batch_id)
        fresh._adapter = fake
        llm2 = _llm()
        got = fresh.collect(llm2, Out)
        assert sum(v is not None for v in got.values()) == 2
        assert llm2.stash  # results landed in the fresh process's stash

    def test_force_resubmits_deliberately(self, ledger, monkeypatch):
        llm = _llm()
        fake = FakeAdapter()
        _run(llm, fake, ["a"], monkeypatch)
        _run(llm, fake, ["a"], monkeypatch, force=True)
        assert len(fake.submitted) == 2


class TestReceiptsAndFallback:
    def test_per_item_receipts_carry_transport_and_model(self, ledger,
                                                         monkeypatch):
        llm = _llm()
        fake = FakeAdapter()
        per_item = {}
        _run(llm, fake, ["a", "b"], monkeypatch, per_item_usage=per_item)
        assert per_item[0]["transport"] == "batch"
        assert per_item[0]["response_model"] == "served-batch-model"
        assert llm.usage.report()["response_models"] == \
            {"served-batch-model": 2}

    def test_errored_items_fall_back_to_sync_at_list(self, ledger,
                                                     monkeypatch):
        llm = _llm()
        fake = FakeAdapter()
        # Fail whichever cid corresponds to prompt "bad".
        fake.respond = lambda cid, req: '{"x": 2}'

        def submit(requests):
            # Mark the second request as the failing one.
            fake.fail_cids = {requests[1][0]}
            return FakeAdapter.submit(fake, requests)
        fake.submit = submit
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 7}') as mock_call:
            results = _run(llm, fake, ["good", "bad"], monkeypatch)
        assert results[0].x == 2
        assert results[1].x == 7, "the failed item must be recovered sync"
        assert mock_call.call_count >= 1

    def test_invalid_json_result_uses_sync_retry_machinery(self, ledger,
                                                           monkeypatch):
        llm = _llm()
        fake = FakeAdapter(respond=lambda cid, req: "not json at all")
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 3}'):
            results = _run(llm, fake, ["a"], monkeypatch)
        assert results[0].x == 3

    def test_missing_results_are_reported_not_invented(self, ledger,
                                                       monkeypatch):
        llm = _llm()
        fake = FakeAdapter()

        def submit(requests):
            fake.omit_cids = {requests[0][0]}
            return FakeAdapter.submit(fake, requests)
        fake.submit = submit
        errors = {}
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=RuntimeError("sync also down")):
            results = _run(llm, fake, ["a", "b"], monkeypatch,
                           errors=errors)
        assert results[0] is None
        assert errors[0]["error"].startswith("batch returned no result")


class TestProbeFirst:
    def test_probe_runs_sync_before_submission(self, ledger, monkeypatch):
        llm = _llm()
        fake = FakeAdapter()
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 5}') as mock_call:
            results = _run(llm, fake, ["a", "b", "c"], monkeypatch,
                           probe=True)
        assert mock_call.call_count == 1
        assert results[0].x == 5
        assert len(fake.submitted[0]) == 2, "probe item is not resubmitted"

    def test_probe_failure_aborts_before_money(self, ledger, monkeypatch):
        llm = _llm()
        fake = FakeAdapter()
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=RuntimeError("rejected param")):
            with pytest.raises(ValueError):
                _run(llm, fake, ["a", "b"], monkeypatch, probe=True)
        assert not fake.submitted, "nothing may be submitted after a "\
            "failed probe"


class TestRefusals:
    def test_deepseek_raises(self):
        llm = LLM("deepseek/deepseek-v4-pro", stash=FakeStash())
        with pytest.raises(ValueError, match="NO batch API"):
            B.extract_batch(llm, ["a"], Out)

    def test_local_raises(self):
        llm = LLM("lmstudio/qwen3.5-35b-a3b", stash=FakeStash())
        with pytest.raises(ValueError, match="local endpoint"):
            B.extract_batch(llm, ["a"], Out)

    def test_oversized_wait_false_refused(self, ledger, monkeypatch):
        monkeypatch.setitem(B._CHUNK_LIMITS, "anthropic", 2)
        llm = _llm()
        fake = FakeAdapter()
        with pytest.raises(ValueError, match="single chunk"):
            _run(llm, fake, ["a", "b", "c"], monkeypatch, wait=False)

    def test_chunking_splits_and_completes(self, ledger, monkeypatch):
        monkeypatch.setitem(B._CHUNK_LIMITS, "anthropic", 2)
        llm = _llm()
        fake = FakeAdapter()
        results = _run(llm, fake, ["a", "b", "c", "d", "e"], monkeypatch)
        assert len(fake.submitted) == 3
        assert all(r is not None for r in results)


class TestTaskSurface:
    def test_task_map_batch_true_routes_to_extract_batch(self, ledger,
                                                         monkeypatch):
        from hashstash import HashStash
        from largeliterarymodels.task import Task
        from pydantic import Field

        class S(BaseModel):
            x: int

        class T(Task):
            schema = S
            system_prompt = "sys"
            retries = 0
            model = "claude-sonnet-4-6"
        T.name = "batch_surface"
        task = T()
        task._stash = HashStash(engine="memory").clear()
        fake = FakeAdapter()
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
        results = task.map(["a", "b"], batch=True, probe=False)
        assert [r.x for r in results] == [1, 1]
        assert len(fake.submitted) == 1

    def test_batch_with_images_refused(self):
        from largeliterarymodels.task import Task

        class S(BaseModel):
            x: int

        class T(Task):
            schema = S
            system_prompt = "sys"
        T.name = "batch_images"
        with pytest.raises(ValueError, match="images"):
            T().map(["a"], batch=True, images_list=[["x.png"]])
