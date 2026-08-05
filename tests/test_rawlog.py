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
        assert len(list(rawlog.stash.keys())) == 1, \
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
                     for k in rawlog.stash.keys()]
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
        [key] = list(rawlog.stash.keys())
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
                            (rawlog.get(k) for k in rawlog.stash.keys()))
        assert transports == ["batch", "sync-fallback", "sync-probe"]
        batch_env = [rawlog.get(k) for k in rawlog.stash.keys()
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
