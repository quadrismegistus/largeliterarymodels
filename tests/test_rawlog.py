"""Raw-response sidecar tests.

The constraint that decides whether this ships is WRITE-PATH ISOLATION
(malign-logits seat): with raw_log unset, a run must construct no sink,
do no serialization, and touch no path this feature adds — an additive
feature that shares a write path is additive only until it is not. The
positive tests then pin the join (sidecar entries land under the SAME
key as the annotation) and the receipt discipline (a sidecar failure
never fails the run it documents).
"""

import json
import types

import pytest
from unittest.mock import patch

import largeliterarymodels.batch as B
import largeliterarymodels.providers as P
from largeliterarymodels.llm import LLM
from largeliterarymodels.rawlog import RawLog
from tests.test_batch import FakeAdapter, _sync_provider
from tests.test_regressions import FakeStash, Out

OKJSON = '{"x": 1}'


@pytest.fixture
def rawlog(tmp_path):
    return RawLog(root=str(tmp_path / "raw_responses"))


def _llm(stash=None, raw_log=None):
    return LLM("claude-sonnet-4-6", temperature=0.0, max_tokens=64,
               stash=stash if stash is not None else FakeStash(),
               raw_log=raw_log)


def _provider_that_sinks(body):
    """Stand-in provider honouring the raw_sink contract."""
    def fake(**kw):
        if kw.get("raw_sink") is not None:
            kw["raw_sink"](body)
        return OKJSON
    return fake


class TestWritePathIsolation:
    def test_off_means_no_sink_reaches_the_provider(self):
        """raw_log unset: the provider must receive raw_sink=None, so it
        skips serialization entirely — off is not 'sink to nowhere', it
        is no sink at all."""
        llm = _llm()
        assert llm.raw_log is None
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return OKJSON
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=fake):
            llm.extract("a", Out, system_prompt="sys", retries=0)
        assert "raw_sink" in seen and seen["raw_sink"] is None

    def test_annotation_stash_identical_with_and_without_sidecar(
            self, rawlog):
        """The sidecar must never alter what a run writes to the
        annotation stash: same provider replies, byte-identical stash
        contents either way."""
        stash_off, stash_on = FakeStash(), FakeStash()
        provider = _provider_that_sinks({"body": "whole"})
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=provider):
            _llm(stash_off).extract("a", Out, system_prompt="sys",
                                    retries=0)
            _llm(stash_on, raw_log=rawlog).extract(
                "a", Out, system_prompt="sys", retries=0)
        assert dict(stash_off) == dict(stash_on)
        assert len(list(rawlog.keys())) == 1, \
            "the only extra write goes to the sidecar root"


class TestCaptureAndJoin:
    def test_sidecar_entry_lands_under_the_annotation_key(self, rawlog):
        stash = FakeStash()
        llm = _llm(stash, raw_log=rawlog)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_provider_that_sinks({"choices": ["raw"]})):
            llm.extract("a", Out, system_prompt="sys", retries=0)
        [ann_key] = list(stash.keys())  # FakeStash canonicalises to JSON
        [raw_key] = [json.dumps(k, sort_keys=True, default=str)
                     for k in rawlog.keys()]
        assert ann_key == raw_key, \
            "raw <-> annotation joins must be a lookup, not a correlation"
        env = rawlog.get(json.loads(raw_key))
        assert env["body"] == {"choices": ["raw"]}
        assert env["transport"] == "sync"
        assert env["model"] == "claude-sonnet-4-6"

    def test_retries_append_versions_under_one_key(self, rawlog):
        llm = _llm(raw_log=rawlog)
        replies = iter(["not json", OKJSON])

        def flaky(**kw):
            if kw.get("raw_sink") is not None:
                kw["raw_sink"]({"attempt_body": True})
            return next(replies)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=flaky):
            llm.extract("a", Out, system_prompt="sys", retries=1)
        [key] = list(rawlog.keys())
        assert len(rawlog.history(key)) == 2, \
            "every attempt's body is kept, oldest first"

    def test_sidecar_failure_never_fails_the_run_but_is_counted(
            self, rawlog):
        """Never-fail alone makes completeness unverifiable: a missing
        receipt is indistinguishable from a call never made. The failure
        must be COUNTED so 'we have the bodies' is a claim about
        receipt()['failed'] == 0, not about an absence of error lines."""
        llm = _llm(raw_log=rawlog)
        # Patch the CLASS: dunder lookup bypasses instance attributes, so
        # an instance-level boom never fires — the first version of this
        # test passed vacuously with the write succeeding underneath it.
        # (llm.stash is a FakeStash, so only the sidecar store is hit.)
        with patch.object(type(rawlog.stash), "__setitem__",
                          side_effect=OSError("disk full")):
            with patch("largeliterarymodels.llm._call_provider",
                       side_effect=_provider_that_sinks({"b": 1})):
                result = llm.extract("a", Out, system_prompt="sys",
                                     retries=0)
        assert result.x == 1, "a receipt failure must not fail the run"
        rec = rawlog.receipt()
        assert rec["failed"] == 1 and rec["recorded"] == 0
        assert "disk full" in rec["errors"][0]

    def test_failed_calls_never_pollute_the_write_fault_count(
            self, rawlog):
        """receipt()['failed'] must count SIDECAR faults only: a call
        that dies at the transport never reaches record(), so the two
        kinds of 'missing' (no body to record vs body dropped) are
        separable in-process — the separation malign-logits asked
        whether the receipt already provides. It does; this pins it."""
        llm = _llm(raw_log=rawlog)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=ConnectionError("provider down")):
            with pytest.raises(Exception):
                llm.extract("a", Out, system_prompt="sys", retries=0)
        assert rawlog.receipt() == {"recorded": 0, "failed": 0,
                                    "errors": []}, \
            "a transport failure is not a sidecar fault"
        # Warm hit: annotation served from stash, no call, no body —
        # missing from any audit, but still not a write fault.
        stash = FakeStash()
        llm2 = _llm(stash, raw_log=rawlog)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_provider_that_sinks({"b": 1})):
            llm2.extract("w", Out, system_prompt="sys", retries=0)
        before = rawlog.receipt()
        with patch("largeliterarymodels.llm._call_provider") as never:
            llm2.extract("w", Out, system_prompt="sys", retries=0)
        assert never.call_count == 0 and rawlog.receipt() == before, \
            "a warm hit records nothing and fails nothing"

    def test_extraction_failures_still_leave_their_bodies(self, rawlog):
        """The sink fires before parsing: an item whose extraction
        failed leaves the raw record of the failure — the sidecar's
        whole point, exercised on the least convenient path."""
        llm = _llm(raw_log=rawlog)

        def junk(**kw):
            if kw.get("raw_sink") is not None:
                kw["raw_sink"]({"the_actual_reply": "not json at all"})
            return "not json at all"
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=junk):
            with pytest.raises(Exception):
                llm.extract("a", Out, system_prompt="sys", retries=0)
        assert rawlog.receipt()["recorded"] == 1
        [key] = list(rawlog.keys())
        assert rawlog.get(key)["body"]["the_actual_reply"] == \
            "not json at all"

    def test_audit_makes_coverage_a_runnable_statement(self, rawlog):
        """'The sidecar has N of N' must be checkable against a scoped
        key set, with the missing keys named."""
        keys = [{"prompt": p, "model": "m"} for p in ("a", "b", "c")]
        rawlog.record(keys[0], {"x": 1})
        rawlog.record(keys[2], {"x": 3})
        report = rawlog.audit(keys)
        assert (report["total"], report["present"]) == (3, 2)
        assert report["missing"] == [keys[1]]
        assert rawlog.receipt() == {"recorded": 2, "failed": 0,
                                    "errors": []}


class TestReceiptRetention:
    """A run fires many times (resumption is the normal case), so the
    run-level claim quantifies over firings — receipts must be RETAINED,
    and the claim must not silently weaken when they are not."""

    def test_receipts_survive_the_process_per_firing(self, rawlog):
        rawlog.record({"k": 1}, {"b": 1})
        rawlog.flush_receipt()
        second = RawLog(root=rawlog.root)  # a later firing, same store
        second.record({"k": 2}, {"b": 2})
        second.flush_receipt()
        second.flush_receipt()  # repeat flushes: latest per firing wins
        rows = {r["firing"]: r for r in second.receipts()}
        assert len(rows) == 2, "one retained receipt per firing"
        assert sorted(r["recorded"] for r in rows.values()) == [1, 1]

    def test_imap_boundary_flushes_automatically(self, rawlog):
        llm = _llm(raw_log=rawlog)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_provider_that_sinks({"b": 1})):
            llm.extract_map(["a", "b"], Out, system_prompt="sys",
                            retries=0)
        [row] = rawlog.receipts()
        assert row["recorded"] == 2 and row["failed"] == 0

    def test_a_counted_failure_is_flushed_immediately(self, rawlog):
        """The row that says bodies were dropped must not wait for a
        batch boundary a crashing process never reaches. Body writes
        fail here while the receipt row still lands — the partial-outage
        case; a store-wide outage is certify()'s job, not this flush's."""
        orig = type(rawlog.stash).__setitem__

        def body_writes_fail(self, key, value, _orig=orig):
            if key != RawLog._RECEIPTS_KEY:
                raise OSError("body store full")
            return _orig(self, key, value)
        with patch.object(type(rawlog.stash), "__setitem__",
                          body_writes_fail):
            rawlog.record({"k": 1}, {"b": 1})
        later = RawLog(root=rawlog.root)  # fresh process, durable read
        [row] = later.receipts()
        assert row["failed"] == 1 and "body store full" in row["errors"][0]

    def test_certify_complete_is_retention_independent(self, rawlog):
        """Presence proves completeness with ZERO receipts retained —
        lost receipts cannot undermine a store that has every body."""
        keys = [{"k": i} for i in range(3)]
        for k in keys:
            rawlog.record(k, {"b": 1})
        # No flush anywhere: simulates every firing dying pre-boundary.
        cert = RawLog(root=rawlog.root).certify(keys)
        assert cert["complete"] is True
        assert cert["firings_retained"] == 0

    def test_certify_against_self_derived_keys_is_a_tautology(
            self, rawlog):
        """Executable warning, not a feature. Keys derived FROM the
        sidecar make certify() check a set against itself: a dropped
        body takes its key with it and never appears in the
        denominator, so complete=True is guaranteed by construction
        rather than earned — a guard taking its threshold from the
        artifact it guards. The claim is a cross-check only when keys
        come from an independent record (the annotation stash, the
        ledger, the input manifest). Exhibited here so nobody
        rediscovers it as a surprise."""
        rawlog.record({"k": 0}, {"b": 1})
        with patch.object(type(rawlog.stash), "__setitem__",
                          side_effect=OSError("down")):
            rawlog.record({"k": 1}, {"b": 1})  # dropped; key never lands
        # (The failure-triggered flush ran INSIDE the outage and was lost
        # with the body — the correlated-failure caveat, live. Store back
        # up: retain the count, as the crash-recovery path would.)
        rawlog.flush_receipt()
        tautology = rawlog.certify(rawlog.keys())  # self-derived scope
        assert tautology["complete"] is True, \
            "the drop is invisible to a denominator the sidecar chose"
        honest = rawlog.certify([{"k": 0}, {"k": 1}])  # independent scope
        assert honest["complete"] is False
        assert honest["known_drops"] == 1, \
            "the independent denominator sees what the self-check cannot"

    def test_certify_treats_unaccounted_absence_as_dropped(self, rawlog):
        """Missing beyond what retained receipts explain must surface as
        unaccounted — never fold into 'clean': retention bounds what
        absence can be EXPLAINED, not what presence PROVES."""
        keys = [{"k": i} for i in range(4)]
        rawlog.record(keys[0], {"b": 1})
        rawlog.record(keys[1], {"b": 1})
        with patch.object(type(rawlog.stash), "__setitem__",
                          side_effect=OSError("down")):
            rawlog.record(keys[2], {"b": 1})  # counted, receipt lost too
        rawlog.flush_receipt()  # store back up: failed=1 now retained
        cert = rawlog.certify(keys)
        assert cert["complete"] is False
        assert (cert["present"], len(cert["missing"])) == (2, 2)
        assert cert["known_drops"] == 1, "the retained receipt explains one"
        assert cert["unaccounted"] == 1, \
            "the other missing key is suspect, not clean"


class TestProviderThreading:
    def test_chat_completion_serializes_to_the_sink(self):
        """The shared OpenAI-compat path must hand the sink a serialized
        body, not an SDK object."""
        response = types.SimpleNamespace(
            model="m", choices=[], usage=None,
            model_dump=lambda mode=None: {"id": "resp-1", "choices": []})
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: response)))
        got = []
        P._chat_completion(client, "openai", "gpt-4o-mini", [],
                           0.0, 64, raw_sink=got.append)
        assert got == [{"id": "resp-1", "choices": []}]

    def test_serialize_response_fallbacks(self):
        assert P.serialize_response({"already": "plain"}) == \
            {"already": "plain"}

        class Dumps:
            def model_dump(self, mode=None):
                return {"mode": mode}
        assert P.serialize_response(Dumps()) == {"mode": "json"}

        class Hostile:
            def __repr__(self):
                return "<hostile>"
        assert P.serialize_response(Hostile()) == \
            {"unserialisable": "<hostile>"}


class TestBatchCapture:
    def test_batch_probe_and_fallback_all_reach_the_sidecar(
            self, tmp_path, monkeypatch, rawlog):
        """All three transports of one batch run must leave raw bodies,
        labelled to agree with the usage rows."""
        ledger = B._Ledger(root=str(tmp_path / "ledger"))
        stash = FakeStash()
        llm = _llm(stash, raw_log=rawlog)
        fake = FakeAdapter()

        def submit(requests, sub=None):
            fake.fail_cids = {requests[0][0]}  # first batch item -> fallback
            return FakeAdapter.submit(fake, requests, sub=sub)
        fake.submit = submit
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)

        def provider(**kw):
            if kw.get("raw_sink") is not None:
                kw["raw_sink"]({"sync_body": True})
            return _sync_provider(**kw)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=provider):
            results = B.extract_batch(llm, ["a", "b", "c"], Out,
                                      system_prompt="sys", probe=True,
                                      poll_interval=0, ledger=ledger)
        assert all(r is not None for r in results)
        transports = sorted(env["transport"] for env in
                            (rawlog.get(k) for k in rawlog.keys()))
        assert transports == ["batch", "sync-fallback", "sync-probe"]
        batch_env = [rawlog.get(k) for k in rawlog.keys()
                     if rawlog.get(k)["transport"] == "batch"]
        assert batch_env[0]["body"] == {"fake_body": next(
            c for c, _ in fake.submitted[0]
            if c not in fake.fail_cids)}

    def test_batch_off_asks_adapter_for_no_raw(self, tmp_path,
                                               monkeypatch):
        """With the sidecar off, collect must pass want_raw=False so
        adapters skip serialization — isolation on the batch path too."""
        ledger = B._Ledger(root=str(tmp_path / "ledger"))
        llm = _llm()
        fake = FakeAdapter()
        seen = {}
        orig = FakeAdapter.results

        def spy(self, batch_id, order=None, want_raw=False):
            seen["want_raw"] = want_raw
            return orig(self, batch_id, order=order, want_raw=want_raw)
        fake.results = types.MethodType(spy, fake)
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
        B.extract_batch(llm, ["a"], Out, system_prompt="sys", probe=False,
                        poll_interval=0, ledger=ledger)
        assert seen["want_raw"] is False
