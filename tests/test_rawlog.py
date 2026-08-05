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

    def test_retries_append_versions_oldest_first_with_attempts(
            self, rawlog):
        """Distinct bodies, so ordering is OBSERVABLE (the first draft
        recorded the same body twice and 'oldest first' was unpinned —
        review finding), and attempt markers make history()
        self-describing: attempt 1 was administered under a retry
        prompt the key does not show."""
        llm = _llm(raw_log=rawlog)
        replies = iter([("not json", 1), (OKJSON, 2)])

        def flaky(**kw):
            text, n = next(replies)
            if kw.get("raw_sink") is not None:
                kw["raw_sink"]({"reply": n})
            return text
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=flaky):
            llm.extract("a", Out, system_prompt="sys", retries=1)
        [key] = list(rawlog.keys())
        hist = rawlog.history(key)
        assert [h["body"]["reply"] for h in hist] == [1, 2], \
            "oldest first, and genuinely ordered"
        assert [h["attempt"] for h in hist] == [0, 1]

    def test_generate_and_map_paths_carry_the_sink(self, rawlog):
        """generate() is every SequentialTask chunk's path and map() is
        OCRCleanTask's plural path — both were review findings: map had
        NO sink (5,000 pages would record zero bodies, receipt clean,
        indistinguishable from sidecar-off) and neither was pinned."""
        llm = _llm(raw_log=rawlog)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_provider_that_sinks({"g": 1})):
            llm.generate("gen prompt", system_prompt="sys")
            llm.map(["m1", "m2"], system_prompt="sys", num_workers=2)
        assert len(list(rawlog.keys())) == 3
        assert rawlog.receipt()["recorded"] == 3
        [row] = rawlog.receipts()
        assert row["recorded"] == 3, \
            "map's boundary must flush the firing receipt"
        env = next(rawlog.get(k) for k in rawlog.keys())
        assert env["provider"] == "anthropic", \
            "sync envelopes must carry provider identity too"

    def test_task_level_raw_log_wiring(self, rawlog):
        """Task(raw_log=...) is the documented public entry point and
        deleting its passthrough left the whole suite green (mutation
        finding) — this is the pin. Also pins the public accessor and
        certify_raw sourcing its denominator from the annotation stash."""
        from largeliterarymodels.task import Task

        class S(Out):
            pass

        class T(Task):
            schema = Out
            system_prompt = "sys"
            retries = 0
            model = "claude-sonnet-4-6"
        T.name = "rawlog_wiring_test"
        task = T(raw_log=rawlog, temperature=0.0, max_tokens=64)
        # A real stash: certify_raw sources its denominator from
        # stash.keys(), which must yield the original dict keys.
        from hashstash import HashStash
        task._stash = HashStash(engine="memory").clear()
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_provider_that_sinks({"t": 1})):
            task.run("one")
            task.map(["two", "three"])
        assert rawlog.receipt()["recorded"] == 3
        assert task.raw_sidecar is rawlog
        cert = task.certify_raw()
        assert cert["complete"] is True and cert["total"] == 3
        with pytest.raises(ValueError, match="no raw-response sidecar"):
            T(temperature=0.0, max_tokens=64).certify_raw()

    def test_caller_supplied_raw_sink_wins_without_typeerror(self, rawlog):
        """Pre-PR a raw_sink kwarg was harmlessly absorbed; the explicit
        parameter made it a TypeError on generate/extract (review
        finding). setdefault discipline: the caller's sink wins."""
        llm = _llm(raw_log=rawlog)
        mine = []
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=_provider_that_sinks({"b": 1})):
            llm.generate("g", system_prompt="sys", raw_sink=mine.append)
            llm.extract("e", Out, system_prompt="sys", retries=0,
                        raw_sink=mine.append)
        assert mine == [{"b": 1}, {"b": 1}]
        assert rawlog.receipt()["recorded"] == 0, \
            "the instance sink must not double-record"

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
                                    "errors": [], "dropped_keys": []}, \
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
                                    "errors": [], "dropped_keys": []}


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

    def test_latest_per_firing_actually_wins(self, rawlog):
        """Mutating latest-wins to first-wins left the suite green
        (review finding): the old test flushed twice with nothing
        changing. Now the counters MOVE between flushes and the
        retained row must show the later state."""
        rawlog.record({"k": 1}, {"b": 1})
        rawlog.flush_receipt()
        with patch.object(type(rawlog.stash), "__setitem__",
                          side_effect=OSError("down")):
            rawlog.record({"k": 2}, {"b": 2})  # in-outage flush lost too
        rawlog.flush_receipt()  # store back up: the later snapshot lands
        [row] = rawlog.receipts()
        assert (row["recorded"], row["failed"]) == (1, 1), \
            "the LATER snapshot must win, not the first"

    def test_fork_re_mints_the_firing_and_resets_counters(self, rawlog):
        """A fork()ed child inherits the parent's counters and firing
        id; flushing them as its own double-counts the parent's work
        and collapses two firings into one row (review finding —
        ProcessPoolExecutor is the HPC path). Simulated via the stored
        pid, which is what the guard actually reads."""
        rawlog.record({"k": 1}, {"b": 1})
        parent_firing = rawlog._firing
        rawlog.flush_receipt()
        rawlog._pid = rawlog._pid - 1  # what the child observes
        rawlog.record({"k": 2}, {"b": 2})
        assert rawlog._firing != parent_firing
        assert rawlog.receipt()["recorded"] == 1, \
            "the child starts from zero — parent work is not re-counted"
        rawlog.flush_receipt()
        rows = {r["firing"]: r["recorded"] for r in rawlog.receipts()}
        assert rows[parent_firing] == 1 and len(rows) == 2

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

    def test_another_runs_drops_cannot_absolve_this_runs_missing(
            self, rawlog):
        """The review wave's sharpest finding: known_drops summed
        GLOBALLY, so one old outage permanently raised the explanation
        budget and unaccounted — the alarm — disarmed with store age.
        Attribution is per-KEY now: run X's drops explain only run X's
        keys."""
        x_keys = [{"run": "x", "i": i} for i in range(3)]
        with patch.object(type(rawlog.stash), "__setitem__",
                          side_effect=OSError("down")):
            for k in x_keys:
                rawlog.record(k, {"b": 1})
        rawlog.flush_receipt()  # 3 drops retained, attributed to x-keys
        y_keys = [{"run": "y", "i": i} for i in range(2)]
        rawlog.record(y_keys[0], {"b": 1})  # y's other key: never written
        cert = rawlog.certify(y_keys)
        assert cert["complete"] is False
        assert cert["known_drops"] == 0, \
            "x's outage must not explain y's missing body"
        assert cert["unaccounted"] == 1
        assert cert["drops_all_firings"] == 3, \
            "the global count survives as context, never as arithmetic"
        assert rawlog.certify(x_keys)["known_drops"] == 3

    def test_since_makes_presence_administration_level(self, rawlog):
        """A forced rerun whose bodies were ALL dropped certified
        complete=True on the previous run's bodies (review blocker):
        presence was key-level. since=<run start> is the honest form —
        and with per-key attribution the rerun's drops are explained,
        not unaccounted."""
        keys = [{"k": i} for i in range(2)]
        for k in keys:
            rawlog.record(k, {"b": "run1"})
        run2_start = __import__("time").time()
        with patch.object(type(rawlog.stash), "__setitem__",
                          side_effect=OSError("down")):
            for k in keys:
                rawlog.record(k, {"b": "run2"})
        key_level = rawlog.certify(keys)
        assert key_level["complete"] is True, \
            "documented default: ANY administration's body counts"
        run_level = rawlog.certify(keys, since=run2_start)
        assert run_level["complete"] is False
        assert len(run_level["missing"]) == 2
        assert run_level["known_drops"] == 2 and \
            run_level["unaccounted"] == 0, \
            "this run's drops explain this run's missing, exactly"

    def test_degraded_and_corrupt_bodies_do_not_certify(self, rawlog):
        """An empty body and an unserialisable marker are presence
        without evidence; a corrupt entry is neither — and one corrupt
        entry must not deny certification of the rest (review
        findings)."""
        good, empty, marker, broken = ({"k": i} for i in range(4))
        rawlog.record(good, {"real": "body"})
        rawlog.record(empty, {})
        rawlog.record(marker, {"unserialisable": "<Response 0x1>"})
        rawlog.record(broken, {"will": "corrupt"})
        real_get = rawlog.get
        rawlog.get = lambda k: (_ for _ in ()).throw(
            RuntimeError("LZ4 decode failed")) if k == broken \
            else real_get(k)
        cert = rawlog.certify([good, empty, marker, broken])
        assert cert["complete"] is False
        assert cert["present"] == 1
        assert cert["degraded"] == [empty, marker]
        assert cert["corrupt"] == [broken], \
            "corrupt is reported per-key, not raised over the whole audit"


class TestProviderThreading:
    def test_no_serialization_work_when_sink_is_none(self):
        """The PR's headline isolation claim, previously untested:
        hoisting serialize_response above the None guard left the suite
        green (review finding). Providers must do ZERO raw work for
        raw_sink=None — a bomb in serialize_response must not detonate."""
        response = types.SimpleNamespace(
            model="m", choices=[], usage=None,
            model_dump=lambda mode=None: {"id": "r", "choices": []})
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: response)))
        with patch.object(P, "serialize_response",
                          side_effect=AssertionError("serialized with "
                                                     "sink off")):
            P._chat_completion(client, "openai", "gpt-4o-mini", [],
                               0.0, 64, raw_sink=None)
            P._sink_raw(None, response)

    def test_sink_failure_is_structural_not_sink_dependent(self):
        """'Never fails the run' must hold even for a hostile sink and a
        raising serializer — the guard lives in the provider, not in
        RawLog.record (review finding: the sink call was unguarded, so
        a serialize_response error consumed a retry)."""
        response = types.SimpleNamespace(
            model="m", choices=[], usage=None,
            model_dump=lambda mode=None: {"id": "r", "choices": []})
        P._sink_raw(lambda body: (_ for _ in ()).throw(
            OSError("hostile sink")), response)  # must not raise
        with patch.object(P, "serialize_response",
                          side_effect=RuntimeError("serializer bug")):
            P._sink_raw(lambda body: None, response)  # must not raise

    def test_concurrent_records_count_exactly(self, rawlog):
        """The class finally earns its name: 8 threads recording
        concurrently must neither lose counts nor corrupt the store."""
        import threading as th
        errs = []

        def worker(n):
            try:
                for i in range(25):
                    rawlog.record({"w": n, "i": i}, {"b": 1})
            except Exception as e:  # noqa: BLE001
                errs.append(e)
        threads = [th.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errs
        assert rawlog.receipt()["recorded"] == 200
        assert len(list(rawlog.keys())) == 200

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
        # The fallback item's FAILED batch body is retained beneath the
        # retry's — get() shows the retry, history() shows both.
        fb_key = next(k for k in rawlog.keys()
                      if rawlog.get(k)["transport"] == "sync-fallback")
        fb_hist = rawlog.history(fb_key)
        assert [h["transport"] for h in fb_hist] == \
            ["batch", "sync-fallback"]
        assert "errored" in fb_hist[0]["body"], \
            "the provider's error payload is the drift evidence"
        assert rawlog.receipts(), \
            "collect must flush the firing receipt"
        batch_env = [rawlog.get(k) for k in rawlog.keys()
                     if rawlog.get(k)["transport"] == "batch"]
        assert batch_env[0]["body"] == {"fake_body": next(
            c for c, _ in fake.submitted[0]
            if c not in fake.fail_cids)}

    def test_breaker_abort_still_retains_the_firing_receipt(
            self, tmp_path, monkeypatch, rawlog):
        """The abort path leaves the ledger row open for a resume — the
        firing whose counters most need retaining. The flush was not in
        a finally and the breaker's raise jumped it (review finding,
        found independently by two reviewers)."""
        ledger = B._Ledger(root=str(tmp_path / "ledger"))
        llm = _llm(FakeStash(), raw_log=rawlog)
        fake = FakeAdapter()

        def submit(requests, sub=None):
            fake.fail_cids = {c for c, _ in requests}
            return FakeAdapter.submit(fake, requests, sub=sub)
        fake.submit = submit
        monkeypatch.setattr(B, "_adapter_for", lambda m, t=None: fake)
        with patch("largeliterarymodels.llm._call_provider",
                   side_effect=lambda **kw: (_ for _ in ()).throw(
                       ConnectionError("sync down too"))):
            with pytest.raises(RuntimeError, match="systematic"):
                B.extract_batch(llm, [f"i{n}" for n in range(12)], Out,
                                system_prompt="sys", probe=False,
                                poll_interval=0, ledger=ledger)
        assert rawlog.receipts(), \
            "the aborted collection must still retain its receipt"

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
