"""Batch transport: 50% pricing on Anthropic, OpenAI and Google.

The endpoints were never the hard part. This module exists to reconcile
batch submission with what llm already guarantees:

  * KEY IDENTITY — a batch-computed annotation writes to the stash under
    exactly the key the streaming path would have used (transport is not
    part of an administration's identity), so warm reads serve batch
    results and a half-warm stash batches only its cold half.
  * RECEIPTS — per-item usage lands in UsageTracker/per_item_usage with
    every field the streaming path records (reasoning evidence,
    response_model, dropped params), tagged "transport": "batch".
  * MONEY-SAFETY — the ledger is written BEFORE submission, and a later
    run whose items overlap an open batch RESUMES it. The one ambiguous
    state (a ledger row that says "submitting" with no batch id — the
    process died inside the submission call) stops loudly for the
    operator: silently resubmitting a possibly-live batch is the
    money-burning failure this design exists to prevent.

DeepSeek has no batch API (verified absent from their pricing page, not
assumed) and local endpoints have nothing to discount — both raise.

Effective discount is ~48-49%, not 50%: items whose batch result fails to
parse fall back to the sync path at list price (~1-in-2,000 measured on
list-typed schemas), and the one probe item runs sync by design.
"""

import hashlib
import json
import os
import time

import logging

from . import providers as P
from .llm import (STASH_PATH, UsageTracker, _build_extract_prompt,
                  _make_key, _parse_json_response, _sampling_fingerprint,
                  _schema_name, _stash_read, _legacy_key_kwargs,
                  _validate_parsed)

log = logging.getLogger(__name__)

LEDGER_DIR = os.path.join(os.path.dirname(STASH_PATH), "batch_ledger")

# Per-provider submission limits (requests per batch). Google's is
# byte-bound (20 MB inline), approximated conservatively by count and
# checked by size at build time.
_CHUNK_LIMITS = {"anthropic": 100_000, "openai": 50_000, "google": 5_000}


class AmbiguousBatchState(RuntimeError):
    """A ledger row says 'submitting' with no batch id.

    The process died inside the submission call: the provider may or may
    not hold a live, billable batch. Nothing here can know, so nothing
    here guesses — check the provider's console/API for a batch created
    around the row's timestamp, then either mark the row closed (add a
    line with status 'closed' and the same fingerprint) or attach the id
    (status 'open', batch_id filled in) and rerun to resume it.
    """


def _custom_id(key):
    """Deterministic per-item id: same key, same id, on every run —
    which is what makes resubmission detectable at all."""
    canon = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:48]


def _fingerprint(custom_ids):
    return hashlib.sha256(
        "".join(sorted(custom_ids)).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Ledger — append-only JSONL; the latest row per fingerprint is the state.
# ---------------------------------------------------------------------------

def _ledger_path():
    os.makedirs(LEDGER_DIR, exist_ok=True)
    return os.path.join(LEDGER_DIR, "ledger.jsonl")


def _ledger_append(row):
    row = dict(row, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(_ledger_path(), "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def _ledger_state():
    """fingerprint -> latest row."""
    path = _ledger_path()
    state = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                state[row["fingerprint"]] = row
    return state


# ---------------------------------------------------------------------------
# Provider adapters — the only code that touches a batch endpoint. Duck-
# typed so tests substitute fakes; each returns plain dicts, never SDK
# objects, so collect() has one shape to reason about.
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

    def submit(self, requests):
        batch = self.client.messages.batches.create(requests=[
            {"custom_id": cid, "params": params} for cid, params in requests
        ])
        return batch.id

    def is_done(self, batch_id):
        return self.client.messages.batches.retrieve(
            batch_id).processing_status == "ended"

    def results(self, batch_id):
        """Yield (custom_id, ok, text, usage_dict, error_str)."""
        for r in self.client.messages.batches.results(batch_id):
            kind = r.result.type
            if kind == "succeeded":
                msg = r.result.message
                u = P._usage_anthropic(msg)
                try:
                    text = P._response_text(
                        msg.content, getattr(msg, "model", "?"),
                        stop_reason=getattr(msg, "stop_reason", None))
                    yield r.custom_id, True, text, u, None
                except ValueError as e:
                    yield r.custom_id, False, None, u, str(e)
            else:
                yield r.custom_id, False, None, None, kind


class _OpenAIAdapter:
    provider = "openai"

    def __init__(self, timeout=None):
        from openai import OpenAI
        api_key = P._get_key("OPENAI_API_KEY")
        self.client = P._cached_client(
            ("openai", api_key, timeout),
            lambda: OpenAI(api_key=api_key) if timeout is None
            else OpenAI(api_key=api_key, timeout=timeout))

    def submit(self, requests):
        lines = "\n".join(
            json.dumps({"custom_id": cid, "method": "POST",
                        "url": "/v1/chat/completions", "body": body})
            for cid, body in requests)
        f = self.client.files.create(file=lines.encode("utf-8"),
                                     purpose="batch")
        batch = self.client.batches.create(
            input_file_id=f.id, endpoint="/v1/chat/completions",
            completion_window="24h")
        return batch.id

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

    def results(self, batch_id):
        batch = self.client.batches.retrieve(batch_id)
        for line in self._lines(getattr(batch, "output_file_id", None)):
            body = (line.get("response") or {}).get("body") or {}
            u = _usage_from_openai_dict(body)
            choices = body.get("choices") or []
            text = (choices[0].get("message") or {}).get("content") \
                if choices else None
            if text is not None:
                yield line["custom_id"], True, text, u, None
            else:
                yield line["custom_id"], False, None, u, "no content"
        for line in self._lines(getattr(batch, "error_file_id", None)):
            err = json.dumps((line.get("response") or {}).get("body")
                             or line.get("error") or {})[:300]
            yield line["custom_id"], False, None, None, err


class _GoogleAdapter:
    provider = "google"

    def __init__(self, timeout=None):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        self.client = P._cached_client(("google", api_key, timeout),
                                       lambda: genai.Client(api_key=api_key))
        self._model = None

    def submit(self, requests):
        """Google inline requests correlate by ORDER, not custom_id, so
        the (cid, request) list order is the contract collect() relies on."""
        self._order = [cid for cid, _ in requests]
        self._model = requests[0][1]["model"] if requests else None
        inline = []
        for _, req in requests:
            inline.append({"contents": req["contents"],
                           "config": req["config"]})
        job = self.client.batches.create(
            model=self._model, src=inline,
            config={"display_name": f"litmod-{_fingerprint(self._order)}"})
        return job.name

    def is_done(self, batch_id):
        state = str(self.client.batches.get(name=batch_id).state)
        return any(t in state for t in
                   ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"))

    def results(self, batch_id):
        job = self.client.batches.get(name=batch_id)
        responses = getattr(job.dest, "inlined_responses", None) or []
        for cid, item in zip(self._order, responses):
            resp = getattr(item, "response", None)
            if resp is None:
                err = str(getattr(item, "error", "no response"))[:300]
                yield cid, False, None, None, err
                continue
            u = P._usage_google(resp)
            text = getattr(resp, "text", None)
            if text is not None:
                yield cid, True, text, u, None
            else:
                yield cid, False, None, u, "no text part"


def _usage_from_openai_dict(body):
    """The OpenAI batch output is plain JSON, not SDK objects; map its
    usage dict into the same normalised shape the sync path records."""
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
        response_model=body.get("model"),
    )


def _adapter_for(model, timeout=None):
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
    if P._routes_to_anthropic(m):
        return _AnthropicAdapter(timeout)
    if P._routes_to_google(m):
        return _GoogleAdapter(timeout)
    return _OpenAIAdapter(timeout)


def _build_request(model, prompt, full_system, temperature, max_tokens,
                   thinking, cache_ttl):
    """One batch request via the SAME builders the sync path calls."""
    m = model.lower()
    if P._routes_to_anthropic(m):
        params, dropped = P.anthropic_request_params(
            prompt, model=model, system_prompt=full_system,
            temperature=temperature, max_tokens=max_tokens,
            cache_ttl=cache_ttl, thinking=thinking)
        return params, dropped
    if P._routes_to_google(m):
        gm, contents, config, _setting = P.google_request(
            prompt, model=model, system_prompt=full_system,
            temperature=temperature, max_tokens=max_tokens,
            thinking=thinking)
        return {"model": gm, "contents": contents, "config": config}, ()
    body = P.openai_request_body(
        "openai", model, P.openai_messages(prompt, full_system),
        temperature=temperature, max_tokens=max_tokens)
    return body, ()


class BatchHandle:
    """A submitted batch, reconstructable after process death.

    handle = extract_batch(..., wait=False)
    ...process dies, new process...
    handle = BatchHandle.from_ledger(batch_id_or_fingerprint)
    results = handle.collect(llm, schema, ...)
    """

    def __init__(self, provider, model, batch_id, fingerprint, cid_to_key,
                 cid_to_index, adapter=None):
        self.provider = provider
        self.model = model
        self.batch_id = batch_id
        self.fingerprint = fingerprint
        self.cid_to_key = cid_to_key       # custom_id -> key dict
        self.cid_to_index = cid_to_index   # custom_id -> original index
        self._adapter = adapter

    @classmethod
    def from_ledger(cls, batch_id_or_fingerprint, timeout=None):
        for fp, row in _ledger_state().items():
            if batch_id_or_fingerprint in (fp, row.get("batch_id")):
                if row["status"] == "submitting":
                    raise AmbiguousBatchState(AmbiguousBatchState.__doc__)
                return cls(
                    row["provider"], row["model"], row["batch_id"], fp,
                    {c: json.loads(k) for c, k in row["custom_ids"].items()},
                    row.get("indices", {}),
                )
        raise ValueError(
            f"no ledger row for {batch_id_or_fingerprint!r} in "
            f"{_ledger_path()}")

    def adapter(self, timeout=None):
        if self._adapter is None:
            self._adapter = _adapter_for(self.model, timeout)
            if isinstance(self._adapter, _GoogleAdapter):
                # Google correlates by order; restore it from the ledger.
                self._adapter._order = sorted(self.cid_to_key)
        return self._adapter

    def wait(self, poll_interval=60, timeout=None):
        start = time.time()
        a = self.adapter()
        while not a.is_done(self.batch_id):
            if timeout and time.time() - start > timeout:
                raise TimeoutError(
                    f"batch {self.batch_id} not done after {timeout}s — it "
                    f"is still running server-side; collect later with "
                    f"BatchHandle.from_ledger({self.batch_id!r})")
            time.sleep(poll_interval)

    def collect(self, llm, schema, retries=1, errors=None,
                per_item_usage=None, **sync_kwargs):
        """Write results to the stash under the ORIGINAL keys, feed the
        receipts, and fall back to the sync path for failures."""
        a = self.adapter()
        got = set()
        n_ok, n_fallback = 0, 0
        results = {}
        for cid, ok, text, usage, err in a.results(self.batch_id):
            key = self.cid_to_key.get(cid)
            if key is None:
                log.warning("batch %s: result for unknown custom_id %s",
                            self.batch_id, cid)
                continue
            got.add(cid)
            idx = self.cid_to_index.get(cid)
            parsed_ok = False
            if ok and text is not None:
                try:
                    results[cid] = _validate_parsed(
                        _parse_json_response(text), schema)
                    llm.stash[key] = text
                    parsed_ok = True
                    n_ok += 1
                except Exception as e:  # noqa: BLE001 — falls back below
                    err = f"{type(e).__name__}: {e}"
            if usage is not None:
                u = dict(usage, transport="batch")
                llm.usage.record(u)
                if per_item_usage is not None and idx is not None:
                    _record_item(per_item_usage, idx, u)
            if not parsed_ok:
                n_fallback += 1
                log.warning("batch %s: item %s failed (%s) — falling back "
                            "to the sync path at list price",
                            self.batch_id, cid, (err or "?")[:200])
                try:
                    results[cid] = llm.extract(
                        # The key was built from these exact fields, so the
                        # sync fallback recomputes the identical key and
                        # its retry machinery (partial-field reprompts
                        # included) takes over.
                        prompt=key["prompt"], schema=schema,
                        system_prompt=key["system_prompt"],
                        temperature=key["temperature"],
                        max_tokens=key["max_tokens"],
                        retries=retries, **sync_kwargs)
                except Exception as e:  # noqa: BLE001
                    results[cid] = None
                    if errors is not None and idx is not None:
                        errors[idx] = {
                            "index": idx, "error": f"{type(e).__name__}: {e}",
                            "exception": e, "attempts": 1 + retries,
                            "transport": "batch+sync-fallback",
                        }
        missing = set(self.cid_to_key) - got
        for cid in missing:
            idx = self.cid_to_index.get(cid)
            results[cid] = None
            if errors is not None and idx is not None:
                errors[idx] = {
                    "index": idx,
                    "error": "batch returned no result for this item",
                    "exception": None, "attempts": 0, "transport": "batch",
                }
        _ledger_append({"status": "closed", "fingerprint": self.fingerprint,
                        "provider": self.provider, "model": self.model,
                        "batch_id": self.batch_id, "custom_ids": {},
                        "n_ok": n_ok, "n_fallback": n_fallback,
                        "n_missing": len(missing)})
        if n_fallback or missing:
            log.warning(
                "batch %s: %d ok, %d sync fallbacks (billed at list), %d "
                "missing — the effective discount on this batch is below "
                "50%% accordingly", self.batch_id, n_ok, n_fallback,
                len(missing))
        return results


def _record_item(per_item_usage, idx, usage):
    entry = per_item_usage.setdefault(idx, {
        "index": idx, "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "reasoning_tokens": 0, "response_model": None,
    })
    entry["calls"] += 1
    entry["transport"] = "batch"
    for k in ("input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens", "reasoning_tokens"):
        entry[k] += usage.get(k, 0)
    if usage.get("response_model"):
        entry["response_model"] = usage["response_model"]


def extract_batch(llm, prompts, schema, system_prompt=None, examples=None,
                  temperature=None, max_tokens=None, metadata_list=None,
                  force=False, retries=1, probe=True, wait=True,
                  poll_interval=60, timeout=None, errors=None,
                  per_item_usage=None, cache_ttl=None, **kwargs):
    """Batch-transport analogue of LLM.extract_map. Same keys, same
    receipts, half the price on the items that succeed.

    probe=True runs the FIRST cold item through the sync path before
    anything is submitted: it catches rejected parameters where the batch
    has no repair loop (and warms the OpenAI param memo the request
    builder consults), warms the prompt cache, and is fail_fast's only
    meaningful moment — after submission the spend is committed.

    wait=False returns a BatchHandle instead of results (single chunk
    only); collect later, in this process or another.
    """
    prompts = list(prompts)
    temperature = temperature if temperature is not None else llm.temperature
    max_tokens = max_tokens if max_tokens is not None else llm.max_tokens
    model = llm.model
    _adapter_for(model)  # raises early for deepseek/local, before any work

    full_system, _ = _build_extract_prompt(
        "", schema, system_prompt=system_prompt, examples=examples)
    s_name = _schema_name(schema)
    eff_temp, thinking_fp = _sampling_fingerprint(model, temperature, kwargs)
    legacy = _legacy_key_kwargs(model, temperature, eff_temp, thinking_fp)

    if P._routes_to_anthropic(model.lower()) and cache_ttl is None \
            and system_prompt:
        # Anthropic's own guidance for batches: entries outlive the 5-minute
        # TTL while the batch processes; 1h writes cost 2x once and read at
        # 0.1x thousands of times.
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
        to_compute.append((i, cid, prompt, key))

    if probe and to_compute:
        i, cid, prompt, key = to_compute[0]
        log.info("batch probe: 1 sync call before committing %d requests",
                 len(to_compute) - 1)
        results[i] = llm.extract(
            prompt=prompt, schema=schema, system_prompt=system_prompt,
            examples=examples, temperature=temperature,
            max_tokens=max_tokens, retries=retries, force=force,
            cache_ttl=cache_ttl, **kwargs)
        to_compute = to_compute[1:]

    if not to_compute:
        _fan_out_duplicates(results, dup_of)
        return results

    chunk_limit = _CHUNK_LIMITS[
        "anthropic" if P._routes_to_anthropic(model.lower())
        else "google" if P._routes_to_google(model.lower()) else "openai"]
    if not wait and len(to_compute) > chunk_limit:
        raise ValueError(
            f"{len(to_compute)} items exceed one {model} batch "
            f"({chunk_limit}); wait=False supports a single chunk — "
            f"split the input or use wait=True.")

    state = _ledger_state()
    all_handles = []
    for start in range(0, len(to_compute), chunk_limit):
        chunk = to_compute[start:start + chunk_limit]
        cids = [c[1] for c in chunk]
        fp = _fingerprint(cids)
        row = state.get(fp)
        if row and not force:
            if row["status"] == "submitting":
                raise AmbiguousBatchState(AmbiguousBatchState.__doc__)
            if row["status"] == "open":
                log.warning(
                    "batch: resuming OPEN batch %s (%d items) from the "
                    "ledger instead of resubmitting — the money was "
                    "already committed", row["batch_id"], len(chunk))
                all_handles.append(BatchHandle(
                    row["provider"], model, row["batch_id"], fp,
                    {c[1]: c[3] for c in chunk},
                    {c[1]: c[0] for c in chunk}))
                continue
        requests = []
        for i, cid, prompt, key in chunk:
            req, _dropped = _build_request(model, prompt, full_system,
                                           temperature, max_tokens,
                                           kwargs.get("thinking", "auto"),
                                           cache_ttl)
            requests.append((cid, req))
        adapter = _adapter_for(model, timeout)
        _ledger_append({
            "status": "submitting", "fingerprint": fp,
            "provider": adapter.provider, "model": model, "batch_id": None,
            "n": len(chunk),
            "custom_ids": {c[1]: json.dumps(c[3], sort_keys=True,
                                            default=str) for c in chunk},
            "indices": {c[1]: c[0] for c in chunk},
        })
        batch_id = adapter.submit(requests)
        _ledger_append({
            "status": "open", "fingerprint": fp,
            "provider": adapter.provider, "model": model,
            "batch_id": batch_id, "n": len(chunk),
            "custom_ids": {c[1]: json.dumps(c[3], sort_keys=True,
                                            default=str) for c in chunk},
            "indices": {c[1]: c[0] for c in chunk},
        })
        log.info("batch: submitted %d requests to %s as %s",
                 len(chunk), adapter.provider, batch_id)
        all_handles.append(BatchHandle(
            adapter.provider, model, batch_id, fp,
            {c[1]: c[3] for c in chunk}, {c[1]: c[0] for c in chunk},
            adapter=adapter))

    if not wait:
        return all_handles[0]

    for handle in all_handles:
        handle.wait(poll_interval=poll_interval, timeout=timeout)
        by_cid = handle.collect(llm, schema, retries=retries, errors=errors,
                                per_item_usage=per_item_usage, **kwargs)
        for cid, value in by_cid.items():
            idx = handle.cid_to_index.get(cid)
            if idx is not None:
                results[idx] = value
    _fan_out_duplicates(results, dup_of)
    return results


def _fan_out_duplicates(results, dup_of):
    for i, twins in dup_of.items():
        for j in twins:
            results[j] = results[i]
