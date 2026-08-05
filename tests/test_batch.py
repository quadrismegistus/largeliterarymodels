"""Batch transport tests — fake adapters, no paid calls.

The constraint that decides whether the feature ships is KEY IDENTITY: a
batch-computed annotation must be indistinguishable from a streamed one,
warm-readable by either path — INCLUDING the probe and fallback items,
which is where the first design broke it. Money-safety tests run against
PERSISTENT stashes with the default probe on: the review showed a fresh
stash plus probe=False masked every resume defect at once.
"""

import json
import threading

import pytest
from unittest.mock import patch
from pydantic import BaseModel

import largeliterarymodels.batch as B
from largeliterarymodels.llm import LLM
from tests.test_regressions import FakeStash, Out

OKJSON = '{"x": 1}'


class FakeAdapter:
    """Scriptable stand-in for a provider batch endpoint. Replays results
    in SUBMISSION order like Google (order param respected), so
    order-correlation bugs are falsifiable here, unlike the first fake."""

    provider = "anthropic"

    def __init__(self, respond=None):
        self.submitted = []
        self.respond = respond or (lambda cid, req: OKJSON)
        self.fail_cids = set()
        self.omit_cids = set()
        self.batches = {}
        self.done = {}
        self.subs = {}
        self.reconcile_mode = "absent"

    def request_bytes(self, req):
        return len(json.dumps(req[1], default=str))

    def submit(self, requests, sub=None):
        bid = f"fake-batch-{len(self.batches)}"
        self.submitted.append(requests)
        self.batches[bid] = list(requests)
        self.done[bid] = True
        self.subs[bid] = sub
        return bid

    def find_sub(self, sub, cids, since_ts):
        """Realistic for the tagged providers: a batch that reached the
        endpoint is discoverable by its sub tag, and a completed scan
        with no match is definitive absence. reconcile_mode='candidates'
        scripts the one inconclusive case (Anthropic in-progress batches
        cannot be content-checked)."""
        for bid, s in self.subs.items():
            if s is not None and s == sub:
                return ("found", bid)
        if self.reconcile_mode == "candidates":
            return ("candidates", [("fake-cand", len(list(cids)))])
        return ("absent", None)

    def is_done(self, batch_id):
        return self.done[batch_id]

    def results(self, batch_id, order=None):
        reqs = dict(self.batches[batch_id])
        replay = order if order is not None else [c for c, _ in
                                                 self.batches[batch_id]]
        for cid in replay:
            req = reqs.get(cid)
            if cid in self.omit_cids:
                continue
            if cid in self.fail_cids:
                yield cid, False, None, None, "errored: boom"
                continue
            usage = {"input_tokens": 10, "output_tokens": 5,
                     "cache_read_tokens": 0, "cache_write_tokens": 0,
                     "reasoning_tokens": 0, "reasoning_reported": False,
                     "reasoning_observed": False,
                     "response_model": "served-batch-model"}
            yield cid, True, self.respond(cid, req), usage, None


@pytest.fixture
def ledger(tmp_path):
    return B._Ledger(root=str(tmp_path / "ledger"))


def _llm(stash=None):
    return LLM("claude-sonnet-4-6", temperature=0.0, max_tokens=64,
               stash=stash if stash is not None else FakeStash())


def _run(llm, fake, prompts, monkeypatch, ledger=None, **kw):
    monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
    kw.setdefault("poll_interval", 0)
    return B.extract_batch(llm, prompts, Out, system_prompt="sys",
                           ledger=ledger, **kw)


def _age_submission(ledger, sub, seconds=3600):
    """Backdate a submission's cid records past the fresh window."""
    rec = ledger.get_sub(sub)
    for cid in rec["cids"]:
        st = dict(ledger.get_cid(cid))
        ts = st.pop("ts")
        ledger.set_cid(cid, ts=ts - seconds, **st)


def _sub_from_exc(excinfo):
    return str(excinfo.value).rsplit("submission id: ", 1)[1].strip()


def _sync_provider(**kw):
    """A well-behaved provider for probe/fallback sync calls."""
    if "usage_sink" in kw and kw["usage_sink"]:
        kw["usage_sink"]({"input_tokens": 20, "output_tokens": 8})
    return '{"x": 7}'


class TestKeyIdentity:
    def test_all_three_transports_share_keys(self, ledger, monkeypatch):
        """Batch items, the PROBE item, and a FALLBACK item must all land
        under streaming-path keys: a warm extract_map re-read serves
        everything with zero provider calls — with metadata_list set,
        which is what every litmod run passes and what broke the probe's
        key in review."""
        stash = FakeStash()
        llm = _llm(stash)
        fake = FakeAdapter()

        def submit(requests, sub=None):
            fake.fail_cids = {requests[0][0]}  # first batch item falls back
            return FakeAdapter.submit(fake, requests, sub=sub)
        fake.submit = submit
        metas = [{"text_id": f"T{i}"} for i in range(4)]
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            results = _run(llm, fake, ["a", "b", "c", "d"], monkeypatch,
                           ledger=ledger, probe=True, metadata_list=metas)
        assert all(r is not None for r in results)
        with patch("largeliterarymodels.llm._call_provider") as mock_call:
            again = llm.extract_map(["a", "b", "c", "d"], Out,
                                    system_prompt="sys",
                                    metadata_list=metas)
        assert all(r is not None for r in again)
        assert mock_call.call_count == 0, \
            "probe/fallback/batch items must all be warm under streaming keys"

    def test_fallback_administers_the_same_instrument(self, ledger,
                                                      monkeypatch):
        """The fallback re-runs with the ALREADY-BUILT instrument: the
        first design re-wrapped it, appending a second contract block —
        the item was administered under a different instrument than its
        batchmates and stashed under an unreachable key."""
        llm = _llm()
        fake = FakeAdapter(respond=lambda cid, req: "not json")
        seen_prompts = []

        def provider(**kw):
            seen_prompts.append(kw["system_prompt"])
            return OKJSON
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=provider):
            _run(llm, fake, ["a"], monkeypatch, ledger=ledger, probe=False)
        assert len(seen_prompts) == 1
        contract = "You must respond with ONLY valid JSON"
        assert seen_prompts[0].count(contract) == 1, \
            "the contract block must appear exactly once"

    def test_streamed_items_are_not_resubmitted(self, ledger, monkeypatch):
        llm = _llm()
        with patch("largeliterarymodels.llm._call_provider",
                   return_value='{"x": 9}'):
            llm.extract("a", Out, system_prompt="sys", retries=0)
        fake = FakeAdapter()
        results = _run(llm, fake, ["a", "b"], monkeypatch, ledger=ledger,
                       probe=False)
        assert results[0].x == 9
        assert len(fake.submitted[0]) == 1


class TestLedgerMoneySafety:
    """All against PERSISTENT stashes with the default probe — the
    configurations the first test suite masked."""

    def test_resume_after_partial_progress_with_probe_on(self, ledger,
                                                         monkeypatch):
        """The design-killer from review: the probe shifts the cold set,
        so a set-fingerprint ledger never matched again and resubmitted
        the live batch. Per-item resolution must resume it."""
        stash = FakeStash()
        fake = FakeAdapter()
        fake.done = {}

        def submit(requests, sub=None):
            bid = FakeAdapter.submit(fake, requests, sub=sub)
            fake.done[bid] = False   # still processing when run 1 'dies'
            return bid
        fake.submit = submit
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            handle = _run(_llm(stash), fake, ["a", "b", "c"], monkeypatch,
                          ledger=ledger, probe=True, wait=False)
        assert len(fake.submitted) == 1
        # Run 2: same PERSISTENT stash (probe item now warm — the cold set
        # has shifted), batch now done.
        fake.done[handle.batch_id] = True
        fake.submit = lambda requests, sub=None: (_ for _ in ()).throw(
            AssertionError("resubmitted a live batch"))
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            results = _run(_llm(stash), fake, ["a", "b", "c"], monkeypatch,
                           ledger=ledger, probe=True)
        assert all(r is not None for r in results)
        assert len(fake.submitted) == 1, "one submission, ever"

    def test_subset_rerun_resumes_not_resubmits(self, ledger, monkeypatch):
        """A filtered manifest (--limit) is a SUBSET of the submitted set;
        the set-fingerprint design resubmitted it."""
        stash = FakeStash()
        fake = FakeAdapter()
        handle = _run(_llm(stash), fake, ["a", "b", "c"], monkeypatch,
                      ledger=ledger, probe=False, wait=False)
        fake.submit = lambda requests, sub=None: (_ for _ in ()).throw(
            AssertionError("resubmitted"))
        results = _run(_llm(stash), fake, ["a", "b"], monkeypatch,
                       ledger=ledger, probe=False)
        assert all(r is not None for r in results)

    def test_died_before_accept_reconciles_to_resubmit(self, ledger,
                                                       monkeypatch):
        """Crash window (1): die after marking cids 'submitting', before
        the provider accepted anything. A fresh row still stops (could
        be a live process); a stale one is reconciled against the
        provider — nothing found, so the items are abandoned and the
        rerun resubmits, instead of being stuck behind an operator
        prompt for a batch that never existed."""
        stash = FakeStash()
        fake = FakeAdapter()

        class Dies(FakeAdapter):
            def submit(self, requests, sub=None):
                raise KeyboardInterrupt("dies mid-submit")

        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            with pytest.raises(KeyboardInterrupt):
                _run(_llm(stash), Dies(), ["a", "b", "c"], monkeypatch,
                     ledger=ledger, probe=True)
        # A fresh row reads as another process; age it past the window.
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            with pytest.raises(B.BatchInProgress) as exc:
                _run(_llm(stash), fake, ["a", "b", "c"], monkeypatch,
                     ledger=ledger, probe=True)
        _age_submission(ledger, _sub_from_exc(exc))
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            results = _run(_llm(stash), fake, ["a", "b", "c"], monkeypatch,
                           ledger=ledger, probe=True)
        assert all(r is not None for r in results)
        assert len(fake.submitted) == 1, "reconcile abandoned, rerun submits"

    def test_fresh_submitting_row_reads_as_another_process(self, ledger,
                                                           monkeypatch):
        stash = FakeStash()

        class Dies(FakeAdapter):
            def submit(self, requests, sub=None):
                raise KeyboardInterrupt("dies mid-submit")

        with pytest.raises(KeyboardInterrupt):
            _run(_llm(stash), Dies(), ["a", "b"], monkeypatch,
                 ledger=ledger, probe=False)
        with pytest.raises(B.BatchInProgress):
            _run(_llm(stash), FakeAdapter(), ["a", "b"], monkeypatch,
                 ledger=ledger, probe=False)

    def test_unresolvable_ambiguity_survives_force(self, ledger,
                                                   monkeypatch):
        """force is cache discipline, not permission to double-bill:
        when reconciliation is inconclusive (the Anthropic in-progress
        case), the loud stop still fires through force=True, and the
        message carries what the lookup DID find."""
        stash = FakeStash()

        class Dies(FakeAdapter):
            def submit(self, requests, sub=None):
                raise KeyboardInterrupt("dies mid-submit")

        with pytest.raises(KeyboardInterrupt):
            _run(_llm(stash), Dies(), ["a"], monkeypatch, ledger=ledger,
                 probe=False)
        fake = FakeAdapter()
        fake.reconcile_mode = "candidates"
        with pytest.raises(B.BatchInProgress) as exc:
            _run(_llm(stash), fake, ["a"], monkeypatch,
                 ledger=ledger, probe=False)
        _age_submission(ledger, _sub_from_exc(exc))
        with pytest.raises(B.AmbiguousBatchState, match="inconclusive"):
            _run(_llm(stash), fake, ["a"], monkeypatch,
                 ledger=ledger, probe=False, force=True)

    def test_operator_resolution_lines_work(self, ledger, monkeypatch):
        """The AmbiguousBatchState instructions must actually function:
        an 'abandoned' line clears the block and the rerun submits."""
        stash = FakeStash()

        class Dies(FakeAdapter):
            def submit(self, requests, sub=None):
                raise KeyboardInterrupt("dies")

        with pytest.raises(KeyboardInterrupt):
            _run(_llm(stash), Dies(), ["a"], monkeypatch, ledger=ledger,
                 probe=False)
        with pytest.raises(B.BatchInProgress) as exc:
            _run(_llm(stash), FakeAdapter(), ["a"], monkeypatch,
                 ledger=ledger, probe=False)
        sub = _sub_from_exc(exc)
        _age_submission(ledger, sub)
        ledger.abandon(sub)
        fake = FakeAdapter()
        results = _run(_llm(stash), fake, ["a"], monkeypatch,
                       ledger=ledger, probe=False)
        assert results[0].x == 1 and len(fake.submitted) == 1

    def test_stray_garbage_in_ledger_dir_does_not_block(self, ledger,
                                                        monkeypatch):
        """Pairtree has no torn-line failure class (one file per version,
        atomic rename — hashstash seat's SIGKILL receipts); a stray temp
        file from a killed writer must be invisible."""
        import os
        stash = FakeStash()
        fake = FakeAdapter()
        _run(_llm(stash), fake, ["a"], monkeypatch, ledger=ledger,
             probe=False)
        with open(os.path.join(ledger.root, "garbage.tmp.99999"), "w") as f:
            f.write("{torn nonsense")
        results = _run(_llm(stash), fake, ["b"], monkeypatch,
                       ledger=ledger, probe=False)
        assert results[0].x == 1

    def test_concurrent_processes_submit_once(self, ledger, monkeypatch):
        """Two threads, same items: the lock's read-decide-append must
        yield one submission and one loud stop."""
        stash_a, stash_b = FakeStash(), FakeStash()
        fake = FakeAdapter()
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
        barrier = threading.Barrier(2)
        outcomes = []

        real_lock = ledger.lock

        def slow_lock():
            barrier.wait(timeout=5)
            return real_lock()
        ledger.lock = slow_lock

        def go(stash):
            try:
                B.extract_batch(_llm(stash), ["a", "b"], Out,
                                system_prompt="sys", probe=False,
                                poll_interval=0, ledger=ledger)
                outcomes.append("ran")
            except B.BatchInProgress:
                outcomes.append("stopped")
        t1 = threading.Thread(target=go, args=(stash_a,))
        t2 = threading.Thread(target=go, args=(stash_b,))
        t1.start(); t2.start(); t1.join(); t2.join()
        assert len(fake.submitted) == 1
        assert sorted(outcomes) == ["ran", "stopped"]

    def test_collect_refuses_a_live_batch(self, ledger, monkeypatch):
        """Collecting an in-flight batch recorded every item missing and
        closed the row on a LIVE batch — whose items a rerun then
        double-billed."""
        stash = FakeStash()
        fake = FakeAdapter()
        fake.done = {}

        def submit(requests, sub=None):
            bid = FakeAdapter.submit(fake, requests, sub=sub)
            fake.done[bid] = False
            return bid
        fake.submit = submit
        handle = _run(_llm(stash), fake, ["a"], monkeypatch, ledger=ledger,
                      probe=False, wait=False)
        with pytest.raises(RuntimeError, match="still processing"):
            handle.collect(_llm(stash), Out)
        states = ledger.states_for(list(handle.cid_to_key))
        assert all(s["state"] == "open" for s in states.values())

    def test_recollecting_a_closed_batch_raises(self, ledger, monkeypatch):
        stash = FakeStash()
        fake = FakeAdapter()
        handle = _run(_llm(stash), fake, ["a"], monkeypatch, ledger=ledger,
                      probe=False, wait=False)
        handle.collect(_llm(stash), Out)
        with pytest.raises(ValueError, match="already collected"):
            B.BatchHandle.from_ledger(handle.batch_id, ledger=ledger)

    def test_handle_survives_process_death_through_real_adapter_path(
            self, ledger, monkeypatch):
        """from_ledger -> adapter() -> collect, with an order-sensitive
        fake: the first suite injected the adapter directly, bypassing
        the order reconstruction where Google corruption lived."""
        stash = FakeStash()
        fake = FakeAdapter(respond=lambda cid, req: json.dumps(
            {"x": int(req["params"]["messages"][0]["content"][-1])}
            if "params" in req else {"x": 0}))
        # Distinguishable per-item responses keyed by prompt digit.
        fake.respond = lambda cid, req: json.dumps(
            {"x": int(dict(fake.batches[
                next(b for b in fake.batches)])[cid]["messages"][-1]
                ["content"][-1])})
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
        handle = B.extract_batch(_llm(stash), ["item 0", "item 1", "item 2"],
                                 Out, system_prompt="sys", probe=False,
                                 wait=False, ledger=ledger)
        fresh = B.BatchHandle.from_ledger(handle.batch_id, ledger=ledger)
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
        llm2 = _llm(FakeStash())
        got = fresh.collect(llm2, Out)
        # A from_ledger handle has no run-local indices; submission order
        # (persisted) is the positional truth.
        by_pos = {fresh.order.index(c): v for c, v in got.items()}
        assert {i: v.x for i, v in by_pos.items()} == \
            {0: 0, 1: 1, 2: 2}, "results must map to THEIR items"

    def test_all_errored_batch_aborts_instead_of_repaying_corpus(
            self, ledger, monkeypatch):
        """A systematically failed batch must not convert into an
        unbounded full-price sync run."""
        stash = FakeStash()
        fake = FakeAdapter()

        def submit(requests, sub=None):
            fake.fail_cids = {c for c, _ in requests}
            return FakeAdapter.submit(fake, requests, sub=sub)
        fake.submit = submit
        sync_calls = []
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=lambda **kw: sync_calls.append(1) or OKJSON):
            with pytest.raises(RuntimeError, match="systematic"):
                _run(_llm(stash), fake, [f"i{n}" for n in range(20)],
                     monkeypatch, ledger=ledger, probe=False)
        assert len(sync_calls) < 8, \
            f"{len(sync_calls)} sync calls — the corpus was being repaid"
        submitted_cids = [c for c, _ in fake.submitted[0]]
        states = ledger.states_for(submitted_cids)
        assert all(s["state"] == "open" for s in states.values()), \
            "the records must stay open: results remain on the provider"


class TestReceipts:
    def test_transport_split_reaches_pricing(self, ledger, monkeypatch):
        """Probe at list + batch at 50% must price as a mixed run, not a
        uniformly discounted one."""
        from largeliterarymodels import costs
        llm = _llm()
        fake = FakeAdapter()
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            _run(llm, fake, ["a", "b", "c"], monkeypatch, ledger=ledger,
                 probe=True)
        rep = llm.usage.report()
        assert set(rep["by_transport"]) == {"batch", "sync-probe"}
        est = costs.price_report("claude-sonnet-4-6", rep, batch=True)
        assert any("mixed-transport" in w for w in est["warnings"])
        assert "sync_transports_at_list" in est["lines"]

    def test_dropped_params_reach_the_batch_record(self, ledger,
                                                   monkeypatch):
        """sonnet-5 rejects temperature; the batch receipt must say so,
        per item, exactly as the streaming path does."""
        llm = LLM("claude-sonnet-5", temperature=0.0, max_tokens=64,
                  stash=FakeStash())
        fake = FakeAdapter()
        _run(llm, fake, ["a", "b"], monkeypatch, ledger=ledger, probe=False)
        assert llm.usage.report()["dropped_params"] == {"temperature": 2}

    def test_probe_and_fallback_rows_in_per_item(self, ledger, monkeypatch):
        llm = _llm()
        fake = FakeAdapter()

        def submit(requests, sub=None):
            fake.fail_cids = {requests[0][0]}
            return FakeAdapter.submit(fake, requests, sub=sub)
        fake.submit = submit
        per_item = {}
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_sync_provider):
            _run(llm, fake, ["a", "b", "c"], monkeypatch, ledger=ledger,
                 probe=True, per_item_usage=per_item)
        transports = {i: e.get("transport") for i, e in per_item.items()}
        assert transports[0] == "sync-probe"
        assert "sync-fallback" in transports.values()
        assert "batch" in transports.values()


class TestSurface:
    def test_wait_false_fully_warm_returns_a_handle(self, ledger,
                                                    monkeypatch):
        llm = _llm()
        with patch("largeliterarymodels.llm._call_provider",
                   return_value=OKJSON):
            llm.extract("a", Out, system_prompt="sys", retries=0)
        handle = _run(llm, FakeAdapter(), ["a"], monkeypatch, ledger=ledger,
                      probe=False, wait=False)
        assert isinstance(handle, B.BatchHandle)
        handle.wait()
        assert handle.collect(llm, Out) == {}

    def test_rejected_kwargs_are_loud(self, ledger):
        llm = _llm()
        with pytest.raises(TypeError, match="fail_fast"):
            B.extract_batch(llm, ["a"], Out, fail_fast=5, ledger=ledger)
        with pytest.raises(ValueError, match="images"):
            B.extract_batch(llm, ["a"], Out, images=["x.png"],
                            ledger=ledger)

    def test_deepseek_and_local_raise_before_any_work(self):
        with pytest.raises(ValueError, match="NO batch API"):
            B.extract_batch(LLM("deepseek/deepseek-v4-pro",
                                stash=FakeStash()), ["a"], Out)
        with pytest.raises(ValueError, match="local endpoint"):
            B.extract_batch(LLM("lmstudio/qwen3.5-35b-a3b",
                                stash=FakeStash()), ["a"], Out)

    def test_fully_warm_run_needs_no_api_key(self, ledger, monkeypatch):
        """A warm rerun reads the stash; demanding credentials for a read
        was a review finding."""
        llm = _llm()
        with patch("largeliterarymodels.llm._call_provider",
                   return_value=OKJSON):
            llm.extract("a", Out, system_prompt="sys", retries=0)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # No adapter monkeypatch: construction would raise on the missing
        # key, so reaching the result proves it was never constructed.
        results = B.extract_batch(llm, ["a"], Out, system_prompt="sys",
                                  probe=False, ledger=ledger)
        assert results[0].x == 1

    def test_chunks_by_bytes_not_just_count(self, ledger, monkeypatch):
        """Count limits alone were 5-60x too generous for real
        instruments: a 25 KB instrument rides in every request."""
        llm = _llm()
        fake = FakeAdapter()
        monkeypatch.setitem(B._LIMITS, "anthropic",
                            {"count": 1000, "bytes": 3000})
        big = "y" * 900
        results = _run(llm, fake, [big + str(n) for n in range(8)],
                       monkeypatch, ledger=ledger, probe=False)
        assert len(fake.submitted) > 1, "must split on bytes"
        assert all(r is not None for r in results)

    def test_task_map_batch_true_with_retries_override(self, ledger,
                                                       monkeypatch):
        """task.map(batch=True, retries=2) used to TypeError on the
        double-passed kwarg."""
        from hashstash import HashStash
        from largeliterarymodels.task import Task

        class S(BaseModel):
            x: int

        class T(Task):
            schema = S
            system_prompt = "sys"
            retries = 0
            model = "claude-sonnet-4-6"
        T.name = "batch_surface_v2"
        task = T()
        task._stash = HashStash(engine="memory").clear()
        fake = FakeAdapter()
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
        results = task.map(["a", "b"], batch=True, probe=False, retries=2,
                           ledger=ledger)
        assert [r.x for r in results] == [1, 1]


class TestReconcile:
    """The two crash windows the lock cannot close — it deliberately does
    not span the network submit — settled by asking the provider, which
    is the only party that knows whether the batch exists."""

    def test_died_after_accept_attaches_and_resumes(self, ledger,
                                                    monkeypatch):
        """Crash window (2), the double-bill one: the provider ACCEPTED
        the batch but the process died before the id was written back.
        Blind reclaim would resubmit paid work; the reconciled rerun
        must find the tagged batch and resume it. The fake's submit
        raises unconditionally after run 1, so any resubmission attempt
        is an exception, not a silent pass."""
        stash = FakeStash()

        class DiesAfterAccept(FakeAdapter):
            def submit(self, requests, sub=None):
                FakeAdapter.submit(self, requests, sub=sub)
                raise KeyboardInterrupt("died after the provider accepted")

        fake = DiesAfterAccept()
        with pytest.raises(KeyboardInterrupt):
            _run(_llm(stash), fake, ["a", "b"], monkeypatch,
                 ledger=ledger, probe=False)
        assert len(fake.submitted) == 1, "the batch EXISTS provider-side"
        with pytest.raises(B.BatchInProgress) as exc:
            _run(_llm(stash), fake, ["a", "b"], monkeypatch,
                 ledger=ledger, probe=False)
        _age_submission(ledger, _sub_from_exc(exc))
        results = _run(_llm(stash), fake, ["a", "b"], monkeypatch,
                       ledger=ledger, probe=False)
        assert all(r is not None for r in results)
        assert len(fake.submitted) == 1, "reconciled and resumed, not resold"

    def test_reconcile_is_definitive_where_tags_exist(self, ledger):
        """reconcile() itself: absent -> abandon (items resubmittable),
        found-by-tag -> attach (items open under the recovered id)."""
        fake = FakeAdapter()

        def plant(sub):
            ledger.set_sub(sub, ["c1", "c2"], "anthropic",
                           "claude-sonnet-4-6")
            for cid in ("c1", "c2"):
                ledger.set_cid(cid, state="submitting", sub=sub,
                               batch_id=None, provider="anthropic",
                               key={"k": cid})
        plant("sub-x")
        verdict, detail = B.reconcile("sub-x", ledger=ledger, adapter=fake)
        assert (verdict, detail) == ("abandoned", None)
        assert ledger.get_cid("c1")["state"] == "abandoned"
        bid = fake.submit([("c1", {}), ("c2", {})], sub="sub-y")
        plant("sub-y")
        verdict, detail = B.reconcile("sub-y", ledger=ledger, adapter=fake)
        assert (verdict, detail) == ("attached", bid)
        assert ledger.get_cid("c2")["state"] == "open"
        assert ledger.get_cid("c2")["batch_id"] == bid
        assert ledger.get_cid("c2")["key"] == {"k": "c2"}, \
            "attach must preserve the stash key material"

    def test_reconcile_failure_falls_back_to_the_loud_stop(self, ledger,
                                                           monkeypatch):
        """A dead network or missing key during reconciliation must not
        crash the resolution pass — it degrades to AmbiguousBatchState
        with the failure named."""
        stash = FakeStash()

        class Dies(FakeAdapter):
            def submit(self, requests, sub=None):
                raise KeyboardInterrupt("dies")

        with pytest.raises(KeyboardInterrupt):
            _run(_llm(stash), Dies(), ["a"], monkeypatch, ledger=ledger,
                 probe=False)
        fake = FakeAdapter()
        fake.find_sub = lambda *a: (_ for _ in ()).throw(
            ConnectionError("provider unreachable"))
        with pytest.raises(B.BatchInProgress) as exc:
            _run(_llm(stash), fake, ["a"], monkeypatch, ledger=ledger,
                 probe=False)
        _age_submission(ledger, _sub_from_exc(exc))
        with pytest.raises(B.AmbiguousBatchState,
                           match="auto-reconcile failed"):
            _run(_llm(stash), fake, ["a"], monkeypatch, ledger=ledger,
                 probe=False)


class TestCustomIdIdentity:
    """The Anthropic reconcile path identifies batches by CONTENT — our
    deterministic custom_ids — which is only safe because the hashed key
    covers the FULL request identity. If any request-shaping parameter
    escaped the hash, rerunning the same items under a changed model or
    prompt would reuse the old cids and content-matching could attach
    the new run to the old batch: previous model's output labelled as
    the new run, no exception anywhere. (Hazard named by the hashstash
    seat; these tests are the pin.)"""

    def test_every_request_shaping_field_changes_the_cid(self):
        from largeliterarymodels.llm import _make_key
        base = dict(prompt="p", model="claude-sonnet-4-6",
                    system_prompt="built instrument bytes",
                    temperature=0.0, max_tokens=64,
                    schema_name="Out", metadata=None, thinking=None)
        cid = B._custom_id(_make_key(**base))
        assert cid == B._custom_id(_make_key(**base)), "deterministic"
        variants = [dict(base, prompt="q"),
                    dict(base, model="claude-opus-4-7"),
                    dict(base, system_prompt="different instrument"),
                    dict(base, temperature=0.7),
                    dict(base, temperature=None),
                    dict(base, max_tokens=128),
                    dict(base, schema_name="Other"),
                    dict(base, metadata={"text_id": "T1"}),
                    dict(base, thinking="enabled:budget:1024")]
        cids = [B._custom_id(_make_key(**v)) for v in variants]
        assert len({cid, *cids}) == len(variants) + 1, \
            "a request-shaping field escaped the custom_id hash"

    def test_changed_model_rerun_does_not_attach_to_old_batch(
            self, ledger, monkeypatch):
        """Two-run drill: run 1 completes on model A; run 2 on model B
        with the SAME items dies mid-submit and goes stale. An
        Anthropic-style content-matching reconcile must NOT find run 1's
        ended batch (the cids differ, because model is in the key) — it
        must abandon and resubmit, and run 2's results must be model
        B's, not a relabelling of run 1's."""
        stash = FakeStash()

        class ContentMatch(FakeAdapter):
            """find_sub the way the real Anthropic adapter works: no
            tag, membership check on ended batches, absent otherwise."""
            def find_sub(self, sub, cids, since_ts):
                want = set(cids)
                for bid, reqs in self.batches.items():
                    if reqs and reqs[0][0] in want:
                        return ("found", bid)
                return ("absent", None)

        fake = ContentMatch(respond=lambda cid, req: json.dumps(
            {"x": 1 if "sonnet" in req.get("model", "") else 2}))
        run1 = _run(_llm(stash), fake, ["a", "b"], monkeypatch,
                    ledger=ledger, probe=False)
        assert [r.x for r in run1] == [1, 1]

        llm_b = LLM("claude-opus-4-7", temperature=0.0, max_tokens=64,
                    stash=stash)

        class DiesOnce(ContentMatch):
            armed = True

            def submit(self, requests, sub=None):
                if DiesOnce.armed:
                    DiesOnce.armed = False
                    raise KeyboardInterrupt("dies mid-submit")
                return FakeAdapter.submit(self, requests, sub=sub)
        fake.__class__ = DiesOnce
        with pytest.raises(KeyboardInterrupt):
            _run(llm_b, fake, ["a", "b"], monkeypatch, ledger=ledger,
                 probe=False)
        with pytest.raises(B.BatchInProgress) as exc:
            _run(llm_b, fake, ["a", "b"], monkeypatch, ledger=ledger,
                 probe=False)
        _age_submission(ledger, _sub_from_exc(exc))
        run2 = _run(llm_b, fake, ["a", "b"], monkeypatch, ledger=ledger,
                    probe=False)
        assert [r.x for r in run2] == [2, 2], \
            "run 2 must carry model B's output, not run 1's relabelled"
        assert len(fake.batches) == 2, \
            "reconcile must abandon and resubmit, not attach across models"


class TestLockDesign:
    def test_submit_lock_is_global_not_chunk_shaped(self, ledger):
        """Design pin: the submit lock's identity must not depend on what
        is being submitted. Per-cid locks deadlock under differently-
        ordered overlapping sets (blocking flock, no timeout); per-chunk
        locks race on shared cids because chunk boundaries shift with the
        cold set. If this test fails, someone 'optimized' the lock —
        re-read the comment on _SUBMIT_LOCK before shipping it."""
        import largeliterarymodels.batch as B2
        assert B2._Ledger._SUBMIT_LOCK == "__batch_submit__"
        # Two ledgers over the same root contend on the same lock file
        # regardless of any chunk identity.
        other = B2._Ledger(root=ledger.root)
        lk = ledger.lock()
        try:
            import threading
            acquired = threading.Event()

            def try_other():
                lk2 = other.lock()
                acquired.set()
                B2._Ledger.unlock(lk2)
            t = threading.Thread(target=try_other, daemon=True)
            t.start()
            t.join(timeout=0.3)
            assert not acquired.is_set(), \
                "second locker must block while the first holds"
        finally:
            B2._Ledger.unlock(lk)
