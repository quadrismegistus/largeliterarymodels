"""Batch transport: 50% pricing on Anthropic, OpenAI and Google.

The endpoints were never the hard part. This module reconciles batch
submission with what llm already guarantees:

  * KEY IDENTITY — a batch-computed annotation writes to the stash under
    exactly the key the streaming path would have used, INCLUDING the
    sync-fallback path (which re-runs items with the already-built
    instrument via extract(prebuilt=True) rather than letting the
    contract block double-wrap) and the probe (which carries the item's
    metadata so its key matches).
  * RECEIPTS — per-item usage with every field the streaming path
    records, tagged by transport ("batch" / "sync-probe" /
    "sync-fallback"); dropped params reach the machine-readable record
    per item; UsageTracker splits token sums by transport so
    costs.price_report can price a mixed run honestly.
  * MONEY-SAFETY — resolution is PER ITEM: every item's custom_id is a
    deterministic hash of its stash key, and the ledger maps custom_ids
    to provider batch ids. A rerun over any overlapping set — subsets,
    supersets, after partial progress, after the probe shifted the cold
    set — finds its open items and RESUMES their batches. (The first
    design fingerprinted the whole cold set, which shrinks as work
    completes, so resume only worked when nothing had progressed —
    found by adversarial review before any money was lost to it.)
    The ledger is written before submission under a file lock; a row
    that says "submitting" with no batch id stops the next run loudly.

DeepSeek has no batch API (verified absent, not assumed); local
endpoints have nothing to discount — both raise.
"""

import hashlib
import json
import logging
import os
import time

from . import providers as P
from .llm import (STASH_PATH, _Breaker, _build_extract_prompt, _make_key,
                  _parse_json_response, _sampling_fingerprint, _schema_name,
                  _stash_read, _legacy_key_kwargs, _validate_parsed)

log = logging.getLogger(__name__)

LEDGER_DIR = os.path.join(os.path.dirname(STASH_PATH), "batch_ledger")

# Per-provider submission limits: requests AND payload bytes, with a
# safety margin. Count limits alone were 5-60x too generous for real
# instruments (a 25 KB instrument rides in EVERY request), which would
# have failed loudly at submit — after the ledger row was written.
_LIMITS = {
    "anthropic": {"count": 100_000, "bytes": 180 * 1024 * 1024},
    "openai": {"count": 50_000, "bytes": 140 * 1024 * 1024},
    "google": {"count": 5_000, "bytes": 14 * 1024 * 1024},
}

# A 'submitting' row younger than this may belong to a LIVE process mid-
# submission (the lock is not held across the network call); older, the
# process is presumed dead and the state is genuinely ambiguous.
_SUBMITTING_FRESH_SECONDS = 600


class AmbiguousBatchState(RuntimeError):
    """A ledger record says 'submitting' with no batch id, and it is old.

    The submitting process died inside the submission call: the provider
    may or may not hold a live, billable batch. The run already tried to
    settle this automatically — reconcile() asks the provider, using the
    sub id tagged into each submission — so seeing this error means the
    lookup itself failed or was inconclusive (details appended below).
    Retry it, or check the provider's console for a batch created around
    the record's timestamp, then resolve by hand:

        from largeliterarymodels.batch import reconcile, _Ledger
        reconcile("<sub id>")                                  # retry lookup
        _Ledger().attach("<sub id>", "<provider batch id>")   # it exists
        _Ledger().abandon("<sub id>")                          # it does not

    (attach makes the next run resume it; abandon makes the next run
    resubmit.) This error is raised regardless of force=True: force is
    cache discipline, not permission to double-bill.

    Seen on one machine but not another over the same ledger? Check NTP
    before the ledger: the lookup's time window compares provider batch
    timestamps against the local submission record, and clock skew fails
    SAFE — a legitimate candidate gets dropped and you land here (a
    false stop), never a wrong attach.
    """


class BatchInProgress(RuntimeError):
    """Another process appears to be submitting these items right now
    (a 'submitting' row younger than 10 minutes). Wait for it to finish
    — its batch will cover these items — or, if it is known dead,
    resolve the row as described in AmbiguousBatchState."""


def _custom_id(key):
    canon = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:48]


# ---------------------------------------------------------------------------
# Ledger v2 — small append-only rows (NO key material), per-item
# resolution, file-locked read-decide-append, torn-line tolerant. Key
# material lives in one sidecar file per submission, written before the
# submitting row references it.
# ---------------------------------------------------------------------------

class _Ledger:
    """Money-critical bookkeeping on a SIBLING-root pairtree HashStash.

    Design settled with the hashstash seat, from their multi-process test
    receipts rather than doc reading:

      * a SIBLING root, never a sub-stash: parent.clear() rmtree-s its own
        dirname and takes any nested sub-stash with it — a cache cleanup
        must not be able to vaporise a billing record. A ledger does not
        share a lifetime with a cache.
      * PAIRTREE, not jsonl: one file per version via tmp+atomic-rename,
        so the torn-line failure class does not exist (their receipt: a
        SIGKILLed writer lost zero of 5,802 versions; a planted garbage
        .tmp is invisible to readers).
      * append_mode gives the state progression for free: setting the
        same cid again appends a version; get() is latest-wins, and
        get_all(all_results=True) is the audit trail (submitting ->
        open -> closed, with _written_at).
      * read-decide-append is NOT atomic on any engine — their 8-process
        race double-billed 8/8 without a lock — so the decide phase runs
        under stash.key_lock(), hashstash's public cross-process lock.

    Stated limits, theirs verbatim: the lock is a LOCAL flock — one
    machine only; two hosts (or NFS) submitting the same items are not
    protected. And there is no fsync, so an OS crash (not a process
    crash) can lose page-cached rows. Single-machine submission
    discipline per task is the operating assumption.
    """

    # ONE global lock per ledger, deliberately — not per-cid and not
    # per-chunk. Per-cid would mean acquiring N flocks in sequence: not
    # atomic (another process can be mid-sequence on an overlapping set)
    # and a deadlock under different acquisition orders, since the
    # underlying flock is blocking with no timeout. Per-chunk sounds
    # cleaner but reintroduces the first hazard here, because chunk
    # boundaries are NOT stable identities — chunking depends on the
    # cold set, which shifts between runs, so two processes with
    # overlapping items chunked differently would hold different locks
    # while racing the same cids. The cost of global is serializing a
    # milliseconds-long scan-and-mark section that never spans the
    # network submit; the benefit is that the critical section's
    # identity cannot drift out from under it. (Hazard analysis from
    # the hashstash seat; the choice and its rationale are ours.)
    _SUBMIT_LOCK = "__batch_submit__"

    def __init__(self, root=None):
        from hashstash import HashStash
        self.root = root or LEDGER_DIR
        self.stash = HashStash(self.root, engine="pairtree",
                               append_mode=True)

    # -- records -----------------------------------------------------------
    # cid          -> {"state", "sub", "batch_id", "provider", "ts", "key"}
    # ("sub", s)   -> {"cids": [...], "provider", "model"}
    # ("batch", b) -> {"order": [...], "sub", "model", "provider",
    #                  "dropped": [...]}

    def lock(self):
        ctx = self.stash.key_lock(self._SUBMIT_LOCK)
        ctx.__enter__()
        return ctx

    @staticmethod
    def unlock(ctx):
        ctx.__exit__(None, None, None)

    def set_cid(self, cid, ts=None, **fields):
        self.stash[cid] = dict(fields, ts=ts if ts is not None
                               else time.time())

    def get_cid(self, cid):
        try:
            return self.stash[cid]
        except KeyError:
            return None

    def history(self, cid):
        """Every recorded state for a cid, oldest first — the audit trail."""
        try:
            return self.stash.get_all(cid, all_results=True)
        except (KeyError, TypeError):
            return []

    def states_for(self, cids):
        """cid -> latest record, for exactly the cids asked about.

        O(len(cids)) point reads — the previous JSONL design re-parsed
        the ENTIRE ledger on every call, and one 50k batch made that
        file gigabytes.
        """
        out = {}
        for cid in cids:
            rec = self.get_cid(cid)
            if rec is not None:
                out[cid] = rec
        return out

    def set_sub(self, sub, cids, provider, model):
        self.stash[("sub", sub)] = {"cids": list(cids),
                                    "provider": provider, "model": model,
                                    "ts": time.time()}

    def get_sub(self, sub):
        try:
            return self.stash[("sub", sub)]
        except KeyError:
            return None

    def set_batch(self, batch_id, **fields):
        self.stash[("batch", batch_id)] = dict(fields, ts=time.time())

    def get_batch(self, batch_id):
        try:
            return self.stash[("batch", batch_id)]
        except KeyError:
            return None

    # -- operator resolutions (the AmbiguousBatchState instructions) ------

    def abandon(self, sub):
        """Operator: the died-mid-submit batch does NOT exist upstream —
        clear its items so the next run resubmits them."""
        rec = self.get_sub(sub)
        if rec is None:
            raise ValueError(f"no submission {sub!r} in the ledger")
        for cid in rec["cids"]:
            self.set_cid(cid, state="abandoned", sub=sub, batch_id=None,
                         provider=rec.get("provider"))
        return len(rec["cids"])

    def attach(self, sub, batch_id):
        """Operator: the died-mid-submit batch DOES exist upstream under
        `batch_id` — attach it so the next run resumes instead of
        resubmitting. Restores per-cid order from the submission record."""
        rec = self.get_sub(sub)
        if rec is None:
            raise ValueError(f"no submission {sub!r} in the ledger")
        for cid in rec["cids"]:
            old = self.get_cid(cid) or {}
            self.set_cid(cid, state="open", sub=sub, batch_id=batch_id,
                         provider=rec.get("provider"), key=old.get("key"))
        self.set_batch(batch_id, order=rec["cids"], sub=sub,
                       model=rec.get("model"), provider=rec.get("provider"),
                       dropped=[])
        return len(rec["cids"])


# ---------------------------------------------------------------------------
# Provider adapters — plain-dict results; SDK objects never escape.
# ---------------------------------------------------------------------------

class _AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, timeout=None):
        from anthropic import Anthropic
        api_key = P._get_key("ANTHROPIC_API_KEY")
        self.client = P._cached_client(
            ("anthropic", api_key, timeout),
            lambda: Anthropic(api_key=api_key) if timeout is None
            else Anthropic(api_key=api_key, timeout=timeout))

    def request_bytes(self, req):
        return len(json.dumps(req[1], default=str))

    def submit(self, requests, sub=None):
        batch = self.client.messages.batches.create(requests=[
            {"custom_id": cid, "params": params} for cid, params in requests
        ])
        return batch.id

    def is_done(self, batch_id):
        return self.client.messages.batches.retrieve(
            batch_id).processing_status == "ended"

    def find_sub(self, sub, cids, since_ts):
        """Reconcile a died-mid-submit window against the provider.

        Anthropic offers no batch-level tag, so identification is by
        CONTENT: our custom_ids are deterministic, so an ended batch
        whose result ids intersect ours is ours, definitively. An
        in-progress batch cannot be opened to check — it is reported as
        a candidate (matching request count, created after the
        submission record) rather than guessed at.
        """
        cids = set(cids)
        candidates = []
        for b in self.client.messages.batches.list(limit=20):
            created = getattr(b, "created_at", None)
            ts = created.timestamp() if hasattr(created, "timestamp") else 0
            if ts and ts < since_ts - 60:
                continue
            if b.processing_status == "ended":
                for r in self.client.messages.batches.results(b.id):
                    if r.custom_id in cids:
                        return ("found", b.id)
                    break  # one id decides: batches don't interleave subs
            else:
                total = getattr(getattr(b, "request_counts", None),
                                "processing", None)
                candidates.append((b.id, total))
        return ("candidates", candidates) if candidates else ("absent", None)

    def results(self, batch_id, order=None, want_raw=False):
        for r in self.client.messages.batches.results(batch_id):
            kind = r.result.type
            if kind == "succeeded":
                msg = r.result.message
                u = P._usage_anthropic(msg)
                raw = P.serialize_response(msg) if want_raw else None
                try:
                    text = P._response_text(
                        msg.content, getattr(msg, "model", "?"),
                        stop_reason=getattr(msg, "stop_reason", None))
                    yield r.custom_id, True, text, u, None, raw
                except ValueError as e:
                    yield r.custom_id, False, None, u, str(e), raw
            else:
                detail = str(getattr(r.result, "error", "") or "")[:200]
                raw = P.serialize_response(r.result) if want_raw else None
                yield r.custom_id, False, None, None, f"{kind}: {detail}", raw


class _OpenAIAdapter:
    provider = "openai"

    def __init__(self, timeout=None):
        from openai import OpenAI
        api_key = P._get_key("OPENAI_API_KEY")
        self.client = P._cached_client(
            ("openai", api_key, timeout),
            lambda: OpenAI(api_key=api_key) if timeout is None
            else OpenAI(api_key=api_key, timeout=timeout))

    @staticmethod
    def _line(cid, body):
        return json.dumps({"custom_id": cid, "method": "POST",
                           "url": "/v1/chat/completions", "body": body})

    def request_bytes(self, req):
        return len(self._line(*req)) + 1

    def submit(self, requests, sub=None):
        import io
        buf = io.BytesIO()
        for cid, body in requests:
            buf.write(self._line(cid, body).encode("utf-8"))
            buf.write(b"\n")
        buf.seek(0)
        buf.name = "batch.jsonl"
        f = self.client.files.create(file=buf, purpose="batch")
        # The sub id rides in batch metadata so a died-mid-submit window
        # is reconcilable by ASKING the provider, not guessing. (A crash
        # between files.create and batches.create leaves an orphan file
        # but no billable batch — find_sub correctly reports absent.)
        batch = self.client.batches.create(
            input_file_id=f.id, endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"litmod_sub": sub} if sub else None)
        return batch.id

    def find_sub(self, sub, cids, since_ts):
        """Definitive either way: submissions are tagged in batch
        metadata, so a match is ours and a completed scan with no match
        means no billable batch exists."""
        for b in self.client.batches.list(limit=50):
            if (getattr(b, "metadata", None) or {}).get(
                    "litmod_sub") == sub:
                return ("found", b.id)
            created = getattr(b, "created_at", 0) or 0
            if created and created < since_ts - 60:
                break  # newest-first: past the window, nothing older is ours
        return ("absent", None)

    def is_done(self, batch_id):
        return self.client.batches.retrieve(batch_id).status in (
            "completed", "failed", "expired", "cancelled")

    def _lines(self, file_id):
        if not file_id:
            return
        content = self.client.files.content(file_id)
        text = content.text if hasattr(content, "text") else \
            content.read().decode("utf-8")
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)

    def results(self, batch_id, order=None, want_raw=False):
        batch = self.client.batches.retrieve(batch_id)
        for line in self._lines(getattr(batch, "output_file_id", None)):
            body = (line.get("response") or {}).get("body") or {}
            u = _usage_from_openai_dict(body)
            raw = body if want_raw else None
            choices = body.get("choices") or []
            text = (choices[0].get("message") or {}).get("content") \
                if choices else None
            if text is not None:
                yield line["custom_id"], True, text, u, None, raw
            else:
                yield line["custom_id"], False, None, u, "no content", raw
        for line in self._lines(getattr(batch, "error_file_id", None)):
            payload = ((line.get("response") or {}).get("body")
                       or line.get("error") or {})
            err = json.dumps(payload)[:300]
            yield line["custom_id"], False, None, None, err, \
                (payload if want_raw else None)


class _GoogleAdapter:
    provider = "google"

    def __init__(self, timeout=None):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        self.client = P._cached_client(("google", api_key, timeout),
                                       lambda: genai.Client(api_key=api_key))

    def request_bytes(self, req):
        r = req[1]
        contents = r["contents"]
        size = len(contents) if isinstance(contents, str) else 4096
        sysi = getattr(r["config"], "system_instruction", None) or ""
        return size + len(sysi) + 512

    def submit(self, requests, sub=None):
        model = requests[0][1]["model"]
        inline = [{"contents": r["contents"], "config": r["config"]}
                  for _, r in requests]
        # display_name carries the sub id (not a timestamp): it is the
        # only provider-visible field we control, and it is what makes a
        # died-mid-submit window reconcilable via find_sub.
        job = self.client.batches.create(
            model=model, src=inline,
            config={"display_name": f"litmod-{sub}" if sub
                    else f"litmod-{int(time.time())}"})
        return job.name

    def find_sub(self, sub, cids, since_ts):
        """Definitive either way: the sub id is the display_name."""
        want = f"litmod-{sub}"
        for job in self.client.batches.list(config={"page_size": 50}):
            if getattr(job, "display_name", None) == want:
                return ("found", job.name)
        return ("absent", None)

    def is_done(self, batch_id):
        state = str(self.client.batches.get(name=batch_id).state)
        return any(t in state for t in
                   ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"))

    def results(self, batch_id, order=None, want_raw=False):
        """Google inline responses correlate by SUBMISSION ORDER. The
        order comes from the caller (persisted in the submission sidecar)
        — never re-derived: sorted(cids) is submission order with
        probability 1/n!, and a mis-zip writes every annotation under
        another item's key, silently."""
        if order is None:
            raise RuntimeError(
                "google batch results need the persisted submission order "
                "— collect via a handle built from the ledger sidecar.")
        job = self.client.batches.get(name=batch_id)
        responses = getattr(job.dest, "inlined_responses", None) or []
        for cid, item in zip(order, responses):
            resp = getattr(item, "response", None)
            if resp is None:
                err = str(getattr(item, "error", "no response"))[:300]
                raw = (P.serialize_response(getattr(item, "error", None))
                       if want_raw else None)
                yield cid, False, None, None, err, raw
                continue
            u = P._usage_google(resp)
            raw = P.serialize_response(resp) if want_raw else None
            text = getattr(resp, "text", None)
            if text is not None:
                yield cid, True, text, u, None, raw
            else:
                yield cid, False, None, u, "no text part", raw


def _usage_from_openai_dict(body):
    u = body.get("usage") or {}
    details = u.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    prompt = u.get("prompt_tokens", 0) or 0
    return P._usage(
        input_tokens=max(0, prompt - (cached or 0)),
        output_tokens=u.get("completion_tokens", 0),
        cache_read_tokens=cached or 0,
        reasoning_tokens=reasoning or 0,
        reasoning_reported=reasoning is not None,
        reasoning_observed=bool(reasoning),
        response_model=body.get("model"),
    )


def _validate_batchable(model):
    m = model.lower()
    if P._routes_to_local(m):
        raise ValueError(
            f"{model!r} is a local endpoint: there is no batch API and "
            f"nothing to discount — use the concurrent path (num_workers=)."
        )
    if P._routes_to_deepseek(m):
        raise ValueError(
            f"{model!r}: DeepSeek has NO batch API (verified absent from "
            f"their pricing page). Run without batch=True; costs.price "
            f"already refuses to invent the discount."
        )


def _adapter_for(model, timeout=None):
    _validate_batchable(model)
    m = model.lower()
    if P._routes_to_anthropic(m):
        return _AnthropicAdapter(timeout)
    if P._routes_to_google(m):
        return _GoogleAdapter(timeout)
    return _OpenAIAdapter(timeout)


def reconcile(sub, ledger=None, adapter=None, timeout=None):
    """Resolve a died-mid-submit submission by ASKING the provider.

    The lock deliberately does not span the network submit, which leaves
    two crash windows it cannot close (hashstash seat's analysis): die
    after marking cids 'submitting' but before the API accepted (no
    batch exists — safe to resubmit), or die after it accepted but
    before the batch id was written back (a LIVE billable batch with no
    ledger record — resubmitting double-bills). The two are
    indistinguishable client-side, so the ledger alone must stop; but
    the provider knows which happened. Submissions are tagged
    provider-visibly (OpenAI batch metadata, Google display_name;
    Anthropic by deterministic custom_id content-match on ended
    batches), so this lookup is definitive where tags exist, and
    candidate-listing where they don't.

    Returns ("attached", batch_id) — the batch exists, the ledger now
    resumes it; ("abandoned", None) — no batch exists, the next run
    resubmits; or ("candidates", [...]) — inconclusive (untaggable
    in-progress Anthropic batches), left for the operator.
    """
    ledger = ledger or _Ledger()
    rec = ledger.get_sub(sub)
    if rec is None:
        raise ValueError(f"no submission {sub!r} in the ledger under "
                         f"{ledger.root}")
    adapter = adapter or _adapter_for(rec["model"], timeout)
    verdict, detail = adapter.find_sub(sub, rec["cids"],
                                       rec.get("ts") or 0)
    if verdict == "found":
        ledger.attach(sub, detail)
        log.warning("batch reconcile: %s DID reach %s as %s — attached; "
                    "the next run resumes it", sub, adapter.provider, detail)
        return ("attached", detail)
    if verdict == "absent":
        ledger.abandon(sub)
        log.warning("batch reconcile: %s never reached %s — abandoned; "
                    "the next run resubmits its items", sub,
                    adapter.provider)
        return ("abandoned", None)
    return ("candidates", detail)


def _build_request(model, prompt, full_system, temperature, max_tokens,
                   thinking, cache_ttl):
    """(request, dropped) via the SAME builders the sync path calls."""
    m = model.lower()
    if P._routes_to_anthropic(m):
        return P.anthropic_request_params(
            prompt, model=model, system_prompt=full_system,
            temperature=temperature, max_tokens=max_tokens,
            cache_ttl=cache_ttl, thinking=thinking)
    if P._routes_to_google(m):
        gm, contents, config, _setting = P.google_request(
            prompt, model=model, system_prompt=full_system,
            temperature=temperature, max_tokens=max_tokens,
            thinking=thinking)
        return {"model": gm, "contents": contents, "config": config}, ()
    return P.openai_request_body(
        "openai", model, P.openai_messages(prompt, full_system),
        temperature=temperature, max_tokens=max_tokens)


def _record_item(per_item_usage, idx, usage, transport):
    entry = per_item_usage.setdefault(idx, {
        "index": idx, "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "reasoning_tokens": 0, "response_model": None,
    })
    entry["calls"] += 1
    entry["transport"] = transport
    for k in ("input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens", "reasoning_tokens"):
        entry[k] += usage.get(k, 0)
    if usage.get("response_model"):
        entry["response_model"] = usage["response_model"]


class BatchHandle:
    """One submitted batch, reconstructable in any later process.

    Collect only after the batch is done: collect() refuses to run (and
    refuses to close the ledger row) on an in-flight batch — closing a
    live batch and resubmitting its items is a double bill. The usual
    flow after a crash:

        handle = BatchHandle.from_ledger(batch_id)   # or its sub id
        handle.wait()
        handle.collect(llm, schema)
        # then rerun extract_batch warm to assemble the full result list
    """

    def __init__(self, provider, model, batch_id, sub, cid_to_key,
                 cid_to_index, order, dropped=(), adapter=None, ledger=None):
        self.provider = provider
        self.model = model
        self.batch_id = batch_id
        self.sub = sub
        self.cid_to_key = cid_to_key
        self.cid_to_index = cid_to_index
        self.order = order
        self.dropped = tuple(dropped or ())
        self._adapter = adapter
        self._ledger = ledger or _Ledger()

    @classmethod
    def from_ledger(cls, batch_id_or_sub, timeout=None, ledger=None):
        ledger = ledger or _Ledger()
        meta = ledger.get_batch(batch_id_or_sub)
        batch_id = batch_id_or_sub if meta else None
        if meta is None:
            # Maybe a submission id: resolve through its cids' states.
            sub_rec = ledger.get_sub(batch_id_or_sub)
            if sub_rec is None:
                raise ValueError(
                    f"no ledger record for {batch_id_or_sub!r} under "
                    f"{ledger.root}")
            states = ledger.states_for(sub_rec["cids"])
            open_ids = {s.get("batch_id") for s in states.values()
                        if s.get("state") == "open"}
            if not open_ids:
                if any(s.get("state") == "submitting"
                       for s in states.values()):
                    raise AmbiguousBatchState(
                        AmbiguousBatchState.__doc__
                        + f"\n\nsubmission id: {batch_id_or_sub}")
                raise ValueError(
                    f"submission {batch_id_or_sub!r} has no open batch — "
                    f"already collected or abandoned.")
            batch_id = open_ids.pop()
            meta = ledger.get_batch(batch_id)
        sub = meta["sub"]
        cids = meta["order"] or ledger.get_sub(sub)["cids"]
        states = ledger.states_for(cids)
        if states and all(s.get("state") == "closed"
                          for s in states.values()):
            raise ValueError(
                f"batch {batch_id_or_sub!r} is already collected — its "
                f"results are in the stash; a warm run serves them.")
        cid_to_key = {c: s.get("key") for c, s in states.items()
                      if s.get("key") is not None}
        return cls(meta.get("provider"), meta.get("model"), batch_id, sub,
                   cid_to_key, {}, meta.get("order"),
                   meta.get("dropped", ()), ledger=ledger)

    def adapter(self, timeout=None):
        if self._adapter is None:
            self._adapter = _adapter_for(self.model, timeout)
        return self._adapter

    def is_done(self):
        return self.adapter().is_done(self.batch_id)

    def wait(self, poll_interval=60, timeout=None):
        start = time.time()
        while not self.is_done():
            if timeout and time.time() - start > timeout:
                raise TimeoutError(
                    f"batch {self.batch_id} not done after {timeout}s — "
                    f"still running server-side; collect later with "
                    f"BatchHandle.from_ledger({self.batch_id!r})")
            time.sleep(poll_interval)

    def collect(self, llm, schema, retries=1, errors=None,
                per_item_usage=None, fallback_cap=None, **sync_kwargs):
        a = self.adapter()
        if not a.is_done(self.batch_id):
            raise RuntimeError(
                f"batch {self.batch_id} is still processing — collecting "
                f"now would record every item as missing and close the "
                f"ledger row on a LIVE batch, whose items a rerun would "
                f"then resubmit and double-bill. wait() first.")
        # A systematically failed batch must not convert into an
        # unbounded full-price sync run: the breaker counts fallback
        # outcomes, and a trip aborts collection with the row left open.
        breaker = _Breaker(floor=(fallback_cap or 5))
        got = set()
        want_raw = getattr(llm, "raw_log", None) is not None
        results = {}
        try:
            return self._collect_inner(
                a, llm, schema, retries, errors, per_item_usage, breaker,
                got, results, want_raw, **sync_kwargs)
        finally:
            # Firing boundary in a FINALLY, unlike its first draft: the
            # breaker abort deliberately leaves the ledger row open for
            # a resume, and that is exactly the firing whose counters
            # most need retaining.
            if want_raw:
                llm.raw_log.flush_receipt()

    def _collect_inner(self, a, llm, schema, retries, errors,
                       per_item_usage, breaker, got, results, want_raw,
                       **sync_kwargs):
        n_ok, n_fallback = 0, 0
        first_error = None
        for cid, ok, text, usage, err, raw in a.results(
                self.batch_id, order=self.order, want_raw=want_raw):
            key = self.cid_to_key.get(cid)
            if key is None:
                log.warning("batch %s: result for unknown custom_id %s",
                            self.batch_id, cid)
                continue
            got.add(cid)
            idx = self.cid_to_index.get(cid)
            if want_raw and raw is not None:
                llm.raw_log.record(key, raw, transport="batch",
                                   model=self.model, provider=self.provider)
            parsed_ok = False
            if ok and text is not None:
                try:
                    results[cid] = _validate_parsed(
                        _parse_json_response(text), schema)
                    llm.stash[key] = text
                    parsed_ok = True
                    n_ok += 1
                    breaker.record_success()
                except Exception as e:  # noqa: BLE001 — falls back below
                    err = f"{type(e).__name__}: {e}"
            if usage is not None:
                u = dict(usage, transport="batch")
                if self.dropped:
                    u["dropped_params"] = self.dropped
                llm.usage.record(u)
                if per_item_usage is not None and idx is not None:
                    _record_item(per_item_usage, idx, u, "batch")
            if parsed_ok:
                continue
            first_error = first_error or err
            if breaker.record_failure(
                    RuntimeError(str(err or "batch item failed")[:300])):
                raise RuntimeError(
                    f"batch {self.batch_id}: {breaker.tripped_reason} — "
                    f"first error: {first_error!r}. This looks systematic; "
                    f"a sync fallback would re-bill the whole batch at "
                    f"list price, so collection stopped. The batch's "
                    f"results remain on the provider and the ledger row "
                    f"stays open: fix the cause, then collect again "
                    f"(already-parsed items are in the stash).")
            n_fallback += 1
            log.warning("batch %s: item %s failed (%s) — sync fallback at "
                        "list price", self.batch_id, cid,
                        str(err or "?")[:200])
            fb_usage = []

            def fb_sink(u, _fb=fb_usage):
                u = dict(u, transport="sync-fallback")
                llm.usage.record(u)
                _fb.append(u)
            try:
                results[cid] = llm.extract(
                    prompt=key["prompt"], schema=schema,
                    system_prompt=key["system_prompt"], prebuilt=True,
                    temperature=key["temperature"],
                    max_tokens=key["max_tokens"],
                    metadata=key.get("metadata"),
                    retries=retries, usage_sink=fb_sink,
                    raw_transport="sync-fallback", **sync_kwargs)
                if per_item_usage is not None and idx is not None:
                    for u in fb_usage:
                        _record_item(per_item_usage, idx, u,
                                     "sync-fallback")
            except Exception as e:  # noqa: BLE001
                results[cid] = None
                if errors is not None and idx is not None:
                    errors[idx] = {
                        "index": idx, "error": f"{type(e).__name__}: {e}",
                        "exception": e, "attempts": 1 + retries,
                        "metadata": key.get("metadata"),
                        "prompt_head": (key.get("prompt") or "")[:200],
                        "raw": "",
                        "transport": "batch+sync-fallback",
                    }
        missing = set(self.cid_to_key) - got
        if missing and not got:
            raise RuntimeError(
                f"batch {self.batch_id} returned NO results for its "
                f"{len(missing)} items — it may have failed or expired "
                f"wholesale. Not closing the ledger row; inspect the batch "
                f"on the provider console before rerunning.")
        for cid in missing:
            idx = self.cid_to_index.get(cid)
            results[cid] = None
            if errors is not None and idx is not None:
                key = self.cid_to_key.get(cid) or {}
                errors[idx] = {
                    "index": idx,
                    "error": "batch returned no result for this item",
                    "exception": None, "attempts": 0,
                    "metadata": key.get("metadata"),
                    "prompt_head": (key.get("prompt") or "")[:200],
                    "raw": "", "transport": "batch",
                }
        for cid in self.cid_to_key:
            self._ledger.set_cid(cid, state="closed", sub=self.sub,
                                 batch_id=self.batch_id,
                                 provider=self.provider)
        self._ledger.set_batch(self.batch_id, order=self.order,
                               sub=self.sub, model=self.model,
                               provider=self.provider,
                               dropped=list(self.dropped),
                               closed=True, n_ok=n_ok,
                               n_fallback=n_fallback,
                               n_missing=len(missing))
        if n_fallback or missing:
            log.warning(
                "batch %s: %d ok, %d sync fallbacks (billed at list), %d "
                "missing — the effective discount on this batch is below "
                "50%% accordingly", self.batch_id, n_ok, n_fallback,
                len(missing))
        return results


class CompletedHandle(BatchHandle):
    """Returned by extract_batch(wait=False) when nothing needed
    submitting (fully warm, or the probe consumed the only cold item):
    a handle-shaped object so the caller's contract holds, with nothing
    to wait for and nothing to collect — the results are in the stash."""

    def __init__(self):
        super().__init__(None, None, None, None, {}, {}, None)

    def is_done(self):
        return True

    def wait(self, poll_interval=60, timeout=None):
        return None

    def collect(self, llm, schema, **kwargs):
        return {}


_BATCH_REJECTED_KWARGS = ("fail_fast", "num_workers", "verbose", "cache_key",
                          "images_list", "warm_cache", "raw_transport")


def extract_batch(llm, prompts, schema, system_prompt=None, examples=None,
                  temperature=None, max_tokens=None, metadata_list=None,
                  images=None, force=False, retries=1, probe=True, wait=True,
                  poll_interval=60, timeout=None, errors=None,
                  per_item_usage=None, cache_ttl=None, ledger=None,
                  **kwargs):
    """Batch-transport analogue of LLM.extract_map. Same keys, same
    receipts, half the price on the items the batch serves.

    See the module docstring for the guarantees. wait=False returns a
    BatchHandle (always — a fully-warm input returns a CompletedHandle),
    single chunk only.
    """
    for k in _BATCH_REJECTED_KWARGS:
        if k in kwargs:
            raise TypeError(
                f"extract_batch() does not take {k!r}: it has no meaning "
                f"on the batch transport (see the concurrent path).")
    if images:
        raise ValueError("the batch path does not carry images (payload "
                         "size); use the concurrent path for image tasks.")
    prompts = list(prompts)
    temperature = temperature if temperature is not None else llm.temperature
    max_tokens = max_tokens if max_tokens is not None else llm.max_tokens
    model = llm.model
    _validate_batchable(model)  # deepseek/local refuse before any work,
    adapter = None              # but the client (and its API-key demand)
    ledger = ledger or _Ledger()  # is built only if something must submit
    # — a fully-warm rerun must not require credentials the read needs
    # nothing for.

    full_system, _ = _build_extract_prompt(
        "", schema, system_prompt=system_prompt, examples=examples)
    s_name = _schema_name(schema)
    eff_temp, thinking_fp = _sampling_fingerprint(model, temperature, kwargs)
    legacy = _legacy_key_kwargs(model, temperature, eff_temp, thinking_fp)

    if P._routes_to_anthropic(model.lower()) and cache_ttl is None \
            and full_system:
        # Gated on the BUILT instrument, not the raw system_prompt arg: a
        # schema-only task has a cacheable instrument even when the caller
        # passed no system_prompt, and a 5-minute TTL on a batch that
        # outlives it by hours forfeits every read.
        cache_ttl = "1h"

    results = [None] * len(prompts)
    to_compute = []
    seen = {}
    dup_of = {}
    for i, prompt in enumerate(prompts):
        metadata = metadata_list[i] if metadata_list else None
        key = _make_key(prompt, model, full_system, eff_temp, max_tokens,
                        schema_name=s_name, metadata=metadata,
                        thinking=thinking_fp)
        legacy_key = None if legacy is None else _make_key(
            prompt, model, full_system, legacy["temperature"], max_tokens,
            schema_name=s_name, metadata=metadata)
        hit, cached = (False, None)
        if not force:
            hit, cached = _stash_read(llm.stash, key, legacy_key, model)
        if hit:
            try:
                results[i] = _validate_parsed(
                    _parse_json_response(cached), schema)
                continue
            except Exception:  # noqa: BLE001 — recompute below
                pass
        cid = _custom_id(key)
        if cid in seen:
            dup_of.setdefault(seen[cid], []).append(i)
            continue
        seen[cid] = i
        to_compute.append((i, cid, prompt, key, metadata))

    # Per-item ledger resolution: open items resume their batches;
    # submitting items stop (fresh -> another process; stale ->
    # ambiguous); everything else submits fresh.
    cids_all = [item[1] for item in to_compute]
    states = ledger.states_for(cids_all)
    now = time.time()
    # Died-mid-submit reconciliation: a STALE 'submitting' row is first
    # resolved against the provider itself — the ledger alone cannot
    # know whether the death fell before the API accepted (no batch,
    # safe to resubmit) or after (a live billable batch with no id
    # recorded, where resubmitting double-bills), but the provider can.
    # Only what the lookup cannot settle still raises.
    stale_subs = {st.get("sub") for st in states.values()
                  if st.get("state") == "submitting" and st.get("sub")
                  and now - st["ts"] >= _SUBMITTING_FRESH_SECONDS}
    unresolved = {}
    for sub_id in sorted(stale_subs):
        try:
            verdict, detail = reconcile(sub_id, ledger=ledger)
        except Exception as e:  # noqa: BLE001 — fall back to the loud stop
            unresolved[sub_id] = f"auto-reconcile failed: {e}"
            continue
        if verdict == "candidates":
            unresolved[sub_id] = ("auto-reconcile inconclusive — "
                                  f"in-progress candidate batches "
                                  f"(id, request_count): {detail}")
    if stale_subs:
        states = ledger.states_for(cids_all)
    resume_by_batch = {}
    fresh = []
    for item in to_compute:
        st = states.get(item[1])
        if st and st["state"] == "submitting":
            # Checked BEFORE force: force is cache discipline, not
            # permission to double-bill through the one ambiguous state.
            if now - st["ts"] < _SUBMITTING_FRESH_SECONDS:
                raise BatchInProgress(
                    BatchInProgress.__doc__
                    + f"\n\nsubmission id: {st.get('sub')}")
            raise AmbiguousBatchState(
                AmbiguousBatchState.__doc__
                + f"\n\nsubmission id: {st.get('sub')}"
                + "\n" + unresolved.get(st.get("sub"), ""))
        if st is None or st["state"] in ("closed", "abandoned") or force:
            if force and st and st["state"] == "open":
                log.warning(
                    "batch: force=True resubmits %s… already live in "
                    "batch %s — that batch's spend is orphaned, "
                    "deliberately", item[1][:12], st["batch_id"])
            fresh.append(item)
        elif st["state"] == "open":
            resume_by_batch.setdefault(st["batch_id"], []).append(item)

    # PROBE AFTER RESOLUTION, from the fresh set only. Probing before
    # resolution sync-billed one item per rerun that was already sitting
    # in an open (or ambiguous) batch — silently draining an
    # AmbiguousBatchState one paid call at a time instead of stopping,
    # and double-paying items whose results the batch already holds.
    if probe and fresh:
        i, cid, prompt, key, metadata = fresh[0]
        log.info("batch probe: 1 sync call before committing %d requests",
                 len(fresh) - 1)

        def probe_sink(u):
            u = dict(u, transport="sync-probe")
            llm.usage.record(u)
            if per_item_usage is not None:
                _record_item(per_item_usage, i, u, "sync-probe")
        results[i] = llm.extract(
            prompt=prompt, schema=schema, system_prompt=system_prompt,
            examples=examples, temperature=temperature,
            max_tokens=max_tokens, retries=retries, force=force,
            cache_ttl=cache_ttl, metadata=metadata, usage_sink=probe_sink,
            raw_transport="sync-probe", **kwargs)
        fresh = fresh[1:]

    handles = []
    for batch_id, items in resume_by_batch.items():
        log.warning("batch: %d items are live in open batch %s — resuming "
                    "it instead of resubmitting", len(items), batch_id)
        handle = BatchHandle.from_ledger(batch_id, ledger=ledger)
        # The sidecar's indices are the ORIGINAL submission's positions;
        # THIS run's items sit at different indices (subsets, reordered
        # manifests). Remap by cid, or a 2-item rerun of a 3-item batch
        # indexes off the end of its own results list.
        handle.cid_to_index = {c[1]: c[0] for c in items}
        handles.append(handle)

    if fresh:
        adapter = _adapter_for(model)
        # Build all requests first (build-time side effects: dropped-param
        # reporting, strict-mode, cache-floor warnings — before money).
        built = []
        dropped_all = ()
        for i, cid, prompt, key, metadata in fresh:
            req, dropped = _build_request(model, prompt, full_system,
                                          temperature, max_tokens,
                                          kwargs.get("thinking", "auto"),
                                          cache_ttl)
            dropped_all = dropped or dropped_all
            built.append((i, cid, prompt, key, req))
        # Chunk by count AND bytes.
        limits = _LIMITS[adapter.provider]
        chunks = []
        cur, cur_bytes = [], 0
        for item in built:
            b = adapter.request_bytes((item[1], item[4]))
            if cur and (len(cur) >= limits["count"]
                        or cur_bytes + b > limits["bytes"]):
                chunks.append(cur)
                cur, cur_bytes = [], 0
            cur.append(item)
            cur_bytes += b
        if cur:
            chunks.append(cur)
        if not wait and (len(chunks) > 1 or handles):
            raise ValueError(
                f"wait=False supports a single new chunk and no resumed "
                f"batches ({len(chunks)} chunks, {len(handles)} resumes "
                f"here) — split the input or use wait=True.")

        for chunk in chunks:
            sub = f"sub-{int(time.time() * 1e6)}-{os.getpid()}"
            cids = [c[1] for c in chunk]
            ledger.set_sub(sub, cids, adapter.provider, model)
            # Read-decide-append under hashstash's cross-process key_lock:
            # the flock on individual appends covers only the append, and
            # an unlocked read-decide race measured 8/8 double-submissions
            # in the hashstash seat's 8-process test.
            lk = ledger.lock()
            try:
                latest = ledger.states_for(cids)
                clash = [
                    c for c in cids
                    if latest.get(c, {}).get("state") == "submitting"
                    or (latest.get(c, {}).get("state") == "open"
                        and not force)
                ]
                if clash:
                    raise BatchInProgress(
                        BatchInProgress.__doc__ + f"\n\nsubmission id: "
                        f"{latest[clash[0]].get('sub')}")
                for i, cid, prompt, key, req in chunk:
                    ledger.set_cid(cid, state="submitting", sub=sub,
                                   batch_id=None,
                                   provider=adapter.provider, key=key)
            finally:
                _Ledger.unlock(lk)
            batch_id = adapter.submit([(c[1], c[4]) for c in chunk],
                                      sub=sub)
            for i, cid, prompt, key, req in chunk:
                ledger.set_cid(cid, state="open", sub=sub,
                               batch_id=batch_id,
                               provider=adapter.provider, key=key)
            ledger.set_batch(batch_id, order=cids, sub=sub, model=model,
                             provider=adapter.provider,
                             dropped=list(dropped_all))
            log.info("batch: submitted %d requests to %s as %s",
                     len(chunk), adapter.provider, batch_id)
            handles.append(BatchHandle(
                adapter.provider, model, batch_id, sub,
                {c[1]: c[3] for c in chunk}, {c[1]: c[0] for c in chunk},
                cids, dropped=dropped_all, adapter=adapter, ledger=ledger))

    if not wait:
        return handles[0] if handles else CompletedHandle()

    for handle in handles:
        handle.wait(poll_interval=poll_interval, timeout=timeout)
        by_cid = handle.collect(llm, schema, retries=retries, errors=errors,
                                per_item_usage=per_item_usage)
        for cid, value in by_cid.items():
            idx = handle.cid_to_index.get(cid)
            if idx is not None:
                results[int(idx)] = value
    for i, twins in dup_of.items():
        for j in twins:
            results[j] = results[i]
    return results
