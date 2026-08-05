"""Core LLM class: unified interface for text generation with HashStash caching."""

import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashstash import HashStash
from tqdm import tqdm
from .providers import (route_provider, _load_image_bytes,
                        cache_minimum_tokens, effective_temperature,
                        thinking_fingerprint)

log = logging.getLogger(__name__)

# Model constants (rolling aliases; keep in sync with cli/models.py MODEL_TAGS).
# These are the single most rot-prone lines in the package: a constant naming a
# retired generation is how `deepseek-chat` quietly served the cheap tier and
# how `gpt-4o`-era defaults met the gpt-5 max_tokens rename. `litmod doctor`
# exists to catch that; run it after editing here.
CLAUDE_OPUS = "claude-opus-4-7"
CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_HAIKU = "claude-haiku-4-5"
GPT_5 = "gpt-5.4"
GPT_5_MINI = "gpt-5.4-mini"
GPT_5_NANO = "gpt-5.4-nano"
# Retained: some cached annotations were keyed to these strings.
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GEMINI_PRO = "gemini-2.5-pro"
GEMINI_FLASH = "gemini-2.5-flash"

DEFAULT_MODEL = CLAUDE_SONNET
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096


def _data_dir(env=None, pkg_parent=None):
    """Root for ALL persistent state — stash, batch ledger, raw sidecars,
    usage logs, human annotations. Everything derives from STASH_PATH,
    so this one function decides where a run's money-backed artifacts
    live. Resolution order:

    1. LITMOD_DATA_DIR — explicit project root, always wins.
    2. A NON-EMPTY package-relative data/ dir OUTSIDE site-packages —
       the clone/editable workflow, unchanged: an existing repo's 9 GB
       of annotation history stays exactly where it is.
    3. ~/.largeliterarymodels/data — the durable default.

    A package-relative dir inside site-packages is NEVER used, even if
    it has data: the old derivation was relative to __file__, so a
    plain pip install silently pointed the stash, the batch ledger and
    the raw sidecars into the venv — where the run reported success,
    certify() said complete, and the next --force-reinstall deleted
    the lot, ledger included (lacan seat's field report). A path that
    pip owns is not storage.
    """
    env = env if env is not None else os.getenv("LITMOD_DATA_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if pkg_parent is None:
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_data = os.path.join(pkg_parent, "data")
    parts = set(os.path.normpath(os.path.abspath(pkg_data)).split(os.sep))
    in_site = bool(parts & {"site-packages", "dist-packages"})

    def _non_empty(p):
        try:
            with os.scandir(p) as it:
                return next(it, None) is not None
        except OSError:
            return False

    if not in_site and _non_empty(pkg_data):
        return pkg_data
    home = os.path.join(os.path.expanduser("~"),
                        ".largeliterarymodels", "data")
    if in_site and _non_empty(pkg_data):
        log.warning(
            "largeliterarymodels: found data at %s — INSIDE site-packages, "
            "where the next reinstall deletes it. Using %s instead. If "
            "that site-packages data matters (a stash, batch ledger or "
            "raw sidecar written by an earlier run of this install), copy "
            "it out NOW, and set LITMOD_DATA_DIR if your real data root "
            "is elsewhere (e.g. a repo clone's data/ directory).",
            pkg_data, home)
    return home


STASH_PATH = os.path.join(_data_dir(), "stash")


def _call_provider(prompt, model, system_prompt=None, temperature=DEFAULT_TEMPERATURE,
                   max_tokens=DEFAULT_MAX_TOKENS, images=None, usage_sink=None,
                   raw_sink=None, **kwargs):
    """Dispatch a prompt to the appropriate provider. Used as the cacheable function."""
    provider_fn = route_provider(model)
    return provider_fn(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        images=images,
        usage_sink=usage_sink,
        raw_sink=raw_sink,
        **kwargs,
    )


class UsageTracker:
    """Thread-safe accumulator for provider token usage.

    Providers return a bare string, so without this the token counts — and in
    particular whether the prompt cache is actually being read — are
    unobservable. A batch that silently stopped caching emits byte-identical
    output at roughly ten times the input price, so "the cache is working"
    has to be a measurement rather than an assumption.

    Usage:
        results = task.map(prompts)
        print(task.usage.summary_line())
        task.usage.report()['cache_hit_rate']
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        # Callable concurrently with record(): take the same lock, or a
        # half-reset tracker reports calls from one run against tokens from
        # another. __init__ creates the lock before its reset() call.
        with self._lock:
            self._reset_unlocked()

    def _reset_unlocked(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        # A subset of output_tokens, not an addition to it: providers bill
        # chain-of-thought as output. Tracked because on an extract call it is
        # spend on text we parse as JSON and discard, and because a provider
        # that reasons is also a provider that ignores `temperature`.
        self.reasoning_tokens = 0
        # Calls whose provider reported a reasoning-token COUNT — diagnostic,
        # for telling one provider's silence from another's zero.
        self.reasoning_reported_calls = 0
        # Calls that showed ANY evidence of reasoning: a nonzero token count,
        # a reasoning_content body (DeepSeek), a thinking block in the
        # content (Anthropic), thought tokens (Google). This is the gate's
        # counter — the token count alone is one signal, and it is the one
        # that goes missing (Anthropic prices no split at all, so a
        # token-only gate passed Fable runs as clean).
        self.reasoning_observed_calls = 0
        # Served model id -> calls it answered. A COUNTER, not a scalar: one
        # run can legitimately span two ids (a rolling alias re-resolving
        # mid-batch, a retry landing on a different snapshot), and a scalar
        # would report whichever came first and read as uniformity.
        self.response_models = {}
        # transport -> token sums (see record); "sync" when untagged.
        self.by_transport = {}
        # param name -> number of calls it was dropped from. A parameter that
        # looks applied and is not can falsify a methods claim silently, so the
        # run keeps a checkable record of the omission rather than only a log
        # line someone had to be suspicious enough to read.
        self.dropped_params = {}

    def record(self, usage):
        with self._lock:
            self.calls += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.cache_read_tokens += usage.get("cache_read_tokens", 0)
            self.cache_write_tokens += usage.get("cache_write_tokens", 0)
            self.reasoning_tokens += usage.get("reasoning_tokens", 0)
            if usage.get("reasoning_reported"):
                self.reasoning_reported_calls += 1
            if usage.get("reasoning_observed") or usage.get("reasoning_tokens"):
                self.reasoning_observed_calls += 1
            # Per-transport token split. A mixed batch run (batch items at
            # 50%, probe + fallbacks at list) cannot be priced from summed
            # totals — price_report(batch=True) was discounting the
            # list-billed tokens too, a systematic under-estimate.
            transport = usage.get("transport", "sync")
            by_t = self.by_transport.setdefault(transport, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
            })
            by_t["calls"] += 1
            for k in ("input_tokens", "output_tokens", "cache_read_tokens",
                      "cache_write_tokens"):
                by_t[k] += usage.get(k, 0)
            served = usage.get("response_model")
            if served:
                self.response_models[served] = \
                    self.response_models.get(served, 0) + 1
            for param in usage.get("dropped_params", ()):
                self.dropped_params[param] = self.dropped_params.get(param, 0) + 1

    @property
    def prompt_tokens(self):
        """Full prompt size: uncached + cache reads + cache writes.

        Providers report `input_tokens` as the uncached remainder only, so this
        is the number to compare against a token count of the prompt.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def report(self):
        # Under the lock: record() bumps calls before tokens, so an unlocked
        # read racing a record() returns a torn snapshot — calls from one
        # state of the world, tokens from another.
        with self._lock:
            prompt = self.input_tokens + self.cache_read_tokens \
                + self.cache_write_tokens
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "prompt_tokens": prompt,
                "cache_hit_rate": (self.cache_read_tokens / prompt) if prompt else 0.0,
                "reasoning_tokens": self.reasoning_tokens,
                "reasoning_share": (self.reasoning_tokens / self.output_tokens)
                                   if self.output_tokens else 0.0,
                "reasoning_reported_calls": self.reasoning_reported_calls,
                "reasoning_observed_calls": self.reasoning_observed_calls,
                "response_models": dict(self.response_models),
                "by_transport": {t: dict(v)
                                 for t, v in self.by_transport.items()},
                "dropped_params": dict(self.dropped_params),
            }

    def summary_line(self):
        r = self.report()
        line = (
            f"usage: {r['calls']} calls  "
            f"prompt={r['prompt_tokens']:,} (cache_read={r['cache_read_tokens']:,}, "
            f"write={r['cache_write_tokens']:,}, fresh={r['input_tokens']:,}) "
            f"output={r['output_tokens']:,}  "
            f"cache_hit_rate={r['cache_hit_rate']:.1%}"
        )
        if r["reasoning_tokens"]:
            line += (f"  reasoning={r['reasoning_tokens']:,} "
                     f"({r['reasoning_share']:.0%} of output)")
        if len(r["response_models"]) > 1:
            # One run, two servers. Worth the line: it means the run's
            # annotations do not share a single coder.
            served = ", ".join(f"{k} x{v}" for k, v in
                               sorted(r["response_models"].items()))
            line += f"  SERVED BY {len(r['response_models'])} MODELS: {served}"
        if r["dropped_params"]:
            dropped = ", ".join(f"{k} x{v}" for k, v in
                                sorted(r["dropped_params"].items()))
            line += f"  DROPPED PARAMS: {dropped}"
        return line

    def no_reasoning_observed(self):
        """True if this run made calls and none showed evidence of reasoning.

        For a producer gating publication on "thinking was off". Evidence is
        multi-signal, per provider: the reasoning-token count and the
        reasoning_content body (DeepSeek — measured 2026-08-04, both absent
        with thinking off, both present with it on), thinking blocks in the
        content (Anthropic, which prices no token split — a token-only gate
        passed every Anthropic run as clean, including Fable, where thinking
        cannot be disabled), and thought tokens (Google).

        Absence of every signal with calls made reads as clean. That is why
        this is `observed` and not `confirmed`: the honest claim is that we
        looked for reasoning in every response and found none, not that the
        provider certified its absence. If a disable silently stops working,
        the signals reappear and this goes False — the failure worth
        catching, since the same flip also un-pins `temperature`.

        Stated limit: a model that interleaves its deliberation into the
        answer text itself (local qwen-style <think> output with no
        structured field) leaves none of these signals and reads as clean.
        `unreported_calls` says how many calls carried no token count either
        way.
        """
        with self._lock:
            return (self.calls > 0 and self.reasoning_observed_calls == 0
                    and self.reasoning_tokens == 0)

    @property
    def unreported_calls(self):
        """Calls whose response carried no reasoning-token field at all.

        Diagnostic, not a gate — on DeepSeek with thinking off this equals
        `calls`, which is the healthy state. Useful for telling one provider's
        silence from another's zero when comparing across backends.
        """
        return self.calls - self.reasoning_reported_calls

    def cache_warning(self, model, system_prompt=None):
        """A warning string if caching plainly is not happening, else None.

        Evidence-based, unlike a character-count estimate. Two distinct
        failure shapes, and the second is the more expensive one:

          * reads=0, writes=0 — the prefix never cached at all, usually
            because it sits below the model's floor (declined silently);
          * reads=0, writes>0 — every call wrote a NEW entry and none read
            one back, i.e. the prefix differs call to call. Each call pays
            the ~1.25x write premium with no reads to pay it back. An
            earlier version returned None here because "some cache activity"
            looked healthy — the exact diagnosis its own message text
            offered was the one it could never reach.

        Covers Anthropic AND Gemini: Gemini's implicit caching has its own
        silent floor, and the miss it hides is the same shape — a
        3,906-token instrument ran 14,520 times at full input price, ~130
        tokens under the 4,096 minimum, visible only on the invoice.
        Gemini reports cache reads only (no write counter), so the
        writes-only branch never fires there; zero reads is the signal.
        """
        m = model.lower()
        anthropic_like = "claude" in m
        google_like = "gemini" in m or m.startswith("google/")
        if self.calls < 3 or not (anthropic_like or google_like):
            return None
        if self.cache_read_tokens:
            return None
        per_call = self.prompt_tokens / max(1, self.calls)
        if per_call < 200:  # trivially small prompts; nothing to cache
            return None
        if self.cache_write_tokens:
            return (
                f"every call wrote a fresh cache entry and none read one "
                f"({self.calls} calls, cache_write="
                f"{self.cache_write_tokens:,}, cache_read=0). The cacheable "
                f"prefix differs between calls, so each pays the ~1.25x "
                f"write premium with no reads to pay it back — check for a "
                f"timestamp, counter, or per-item text in the system prompt."
            )
        floor = cache_minimum_tokens(model)
        floor_text = (
            f"{model!r} declines to cache a prefix below {floor:,} tokens "
            f"and does so silently" if floor is not None else
            f"{model!r} has no measured cache floor in this package — "
            f"measure it, or count tokens against the family's floors"
        )
        return (
            f"no prompt caching observed across {self.calls} calls to {model!r} "
            f"(~{per_call:,.0f} prompt tokens each, cache_read=0, "
            f"cache_write=0). {floor_text}, so every call is paying "
            f"full input price. Either the prompt is under the floor or "
            f"caching is not being requested."
        )

    def __repr__(self):
        return f"UsageTracker({self.summary_line()})"


class BatchAborted(RuntimeError):
    """A batch was stopped by fail_fast.

    Results already computed are cached in the stash — a rerun after fixing
    the fault recovers them for free. `results` carries the partial list when
    raised from extract_map (None from the streaming imap path, which has no
    list to attach); `errors` is the same dict the caller passed in, if any.
    """

    def __init__(self, message, results=None, errors=None):
        super().__init__(message)
        self.results = results
        self.errors = errors


class _Breaker:
    """Stops a batch whose failures are systematic rather than a tail.

    A retry loop assumes failures are transient. A provider-shape mismatch is
    not: the same exception recurs on every item and every attempt, and each
    attempt is billed — thinking tokens bill as output — so a broken batch can
    spend real money producing zero rows.

    The unit is the ITEM OUTCOME, not the attempt. Both predecessors of this
    design failed, in opposite directions:

      * an absolute count (abort after N failures) killed a real ~1,500-item
        run whose failures were a sparse recoverable tail. (Reported by the
        run's author; the figures attached to the incident did not
        reconstruct cleanly at review, but the shape of the failure — a
        count cannot tell a fault from a tail — does not depend on them.)
      * a rate over ATTEMPTS double-counted retried items: a failing item
        contributes 1 + retries failed attempts where a succeeding item
        contributes one success, so the nominal 20% threshold aborted
        healthy runs measured at a 5–12% per-item failure rate — and its
        total-failure condition compared a per-signature count against
        in-flight attempts, so under num_workers > 1 it could not fire at
        all, and the abort message under-reported its own evidence (77% for
        a batch failing at 100%).

    Counting only FINAL item outcomes fixes all three at once, and gives
    transient-burst tolerance for free: an item that fails a 429 and retries
    into success is a success, not half a failure.

    Conditions (either trips):

      * total failure — the first `floor`-plus completed items ALL failed
        with one signature. A wiring fault fails 100% from the first call,
        so this fires after ~floor items (each costing at most 1 + retries
        billed calls, fewer when the identical-repeat cutoff bites).
      * sustained rate — one signature's final failures exceed `rate` of
        completed items, once `min_outcomes` items have completed.
    """

    def __init__(self, min_outcomes=30, rate=0.2, floor=5, enabled=True,
                 min_attempts=None):
        # min_attempts is the retired name for min_outcomes, honoured so an
        # existing fail_fast={'min_attempts': N} keeps its meaning.
        self.min_outcomes = min_attempts if min_attempts is not None else min_outcomes
        self.rate = rate
        self.floor = floor
        self.enabled = enabled
        self._lock = threading.Lock()
        self._failures = {}
        self.outcomes = 0
        self.total_failures = 0
        self.attempts = 0
        self.tripped_signature = None
        self.tripped_reason = None

    @staticmethod
    def signature(exc):
        # Type plus a truncated, digit-stripped message: enough to tell
        # "same bug every time" from "different bad JSON every time". 300
        # chars, not 120: a pydantic error on a wide schema names its fields
        # past the shorter cut, and truncating before them collapsed every
        # distinct missing-field failure into one "systematic" signature.
        text = re.sub(r"\d+", "#", str(exc))[:300]
        return f"{type(exc).__name__}: {text}"

    def record_attempt(self):
        """Count one billed provider attempt, for the abort message only."""
        with self._lock:
            self.attempts += 1

    def record_success(self):
        """Count an item that reached a valid result."""
        if not self.enabled:
            return
        with self._lock:
            self.outcomes += 1

    def record_failure(self, exc):
        """Count an item's FINAL failure; return True once the batch should stop."""
        if not self.enabled:
            return False
        sig = self.signature(exc)
        with self._lock:
            self.outcomes += 1
            self.total_failures += 1
            count = self._failures[sig] = self._failures.get(sig, 0) + 1
            if self.tripped_signature is None:
                if count >= self.floor and count == self.outcomes:
                    self.tripped_signature = sig
                    self.tripped_reason = (
                        f"the first {self.outcomes} items to complete all "
                        f"failed this way ({self.attempts} billed attempts)"
                    )
                elif (self.outcomes >= self.min_outcomes
                        and count / self.outcomes > self.rate):
                    self.tripped_signature = sig
                    self.tripped_reason = (
                        f"{count} of {self.outcomes} completed items "
                        f"({count / self.outcomes:.0%}) failed with one "
                        f"signature, over the {self.rate:.0%} threshold"
                    )
            return self.tripped_signature is not None

    @property
    def tripped(self):
        return self.tripped_signature is not None

    def error(self, model):
        # State the inference, not just the verdict: naming why it believes
        # this is systematic is what lets a caller falsify it in two minutes
        # instead of hunting for a bug that isn't there.
        return (
            f"Aborting batch on model {model!r}: {self.tripped_reason} — "
            f"{self.tripped_signature}. Inference: item failures at this rate "
            f"are systematic rather than a recoverable tail, and every retry "
            f"is billed, so the run is stopped rather than grinding through "
            f"the remaining items. Completed results are cached; a rerun "
            f"resumes where this stopped. If the inference is wrong — a rare "
            f"per-item parse failure can look like this — rerun with "
            f"fail_fast=False, or loosen it with fail_fast={{'rate': 0.5}}."
        )


def _image_cache_id(img):
    """Content-derived cache identifier for one image input.

    Paths are used as-is; bytes and PIL images are hashed so two images of
    equal size never share a key and PIL images cache-hit across runs.
    """
    if isinstance(img, str):
        return img
    if isinstance(img, bytes):
        return f"<bytes:{hashlib.md5(img).hexdigest()}>"
    data, _ = _load_image_bytes(img)
    return f"<image:{hashlib.md5(data).hexdigest()}>"


def _make_key(prompt, model, system_prompt=None, temperature=DEFAULT_TEMPERATURE,
              max_tokens=DEFAULT_MAX_TOKENS, schema_name=None, images=None,
              metadata=None, thinking=None):
    """Build the dict used as a HashStash key.

    Args:
        metadata: Optional dict of user-defined metadata (e.g. page_number,
                  source_file). Stored in the key for retrieval via task.df
                  but does not affect the LLM call. THIS MAKES METADATA
                  PART OF THE ADMINISTRATION'S IDENTITY, and the failure
                  mode is silent and costs money: a resumed or repeated
                  run must pass byte-identical metadata or every item
                  re-keys and re-pays at full price while cache_hit_rate
                  honestly reports 0% (lacan seat's field report: 24
                  items became 48 keys). Deliberate — the stash value is
                  raw text, so the key is metadata's only durable home —
                  and unchangeable now without orphaning every
                  metadata-bearing key, which is the norm for litmod
                  runs. Treat metadata as identity, not decoration.
        schema_name: A DISCRIMINATOR, not the schema's identity carrier.
                  On the extract path the built instrument — which
                  embeds the schema's full JSON spec, descriptions
                  included — is this key's system_prompt, so a field
                  OR description change re-keys (same sign as
                  metadata: costs money, receipts honest; for a
                  research instrument a "cosmetic" edit to the
                  questionnaire is an edit to the questionnaire).
                  Custom cache_key dicts are the exception: they carry
                  schema by NAME only, so a questionnaire change does
                  NOT re-key there and a warm read serves answers to a
                  question that no longer exists (malign seat's
                  mirror-image report). The sanctioned mechanisms:
                  SequentialTask.prompt_version (enters the chunk key;
                  for revisions of the same construct) or rename the
                  class (for a different construct). One validation
                  backstop and its stated limit: warm hits re-validate
                  against the CURRENT schema and recompute on failure,
                  but pydantic ignores extra fields, so a pure field
                  REMOVAL leaves old rows validating — the one drift
                  validation cannot see.
        images: Optional list of images. Paths are keyed as-is; bytes and
                PIL images are keyed by content hash (bytes are not stored).
        thinking: The resolved thinking state ("disabled"/"enabled") when a
                  thinking parameter will actually be sent, else None. None
                  omits the field, which keeps every pre-thinking cache key
                  byte-stable; a value marks the calls whose thinking state
                  is part of their identity — thinking-on and thinking-off
                  output must never share a key, or a rerun hands back one
                  as the other with nothing in the record saying so.
    """
    key = {
        "prompt": prompt,
        "model": model,
        "system_prompt": system_prompt,
        # The temperature that will GOVERN the call, not the one requested:
        # callers pass effective_temperature(...), which is None on models
        # that reject or ignore it. The stash key is the durable artifact;
        # recording temperature: 0.0 on a model that never applied it plants
        # a false methods claim in every later read.
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if schema_name:
        key["schema"] = schema_name
    if images:
        key["images"] = [_image_cache_id(img) for img in images]
    if metadata:
        key["metadata"] = metadata
    if thinking is not None:
        key["thinking"] = thinking
    return key


def _sampling_fingerprint(model, temperature, kwargs):
    """(effective_temperature, thinking_fingerprint) for a call's stash key.

    Resolved from static provider knowledge only — a rejection discovered
    dynamically mid-run must not change keys between one call and the next.
    """
    thinking = kwargs.get("thinking", "auto")
    return (effective_temperature(model, temperature, thinking),
            thinking_fingerprint(model, thinking))


_WARNED_LEGACY_READS = set()


def _legacy_key_kwargs(model, temperature, eff_temp, thinking_fp):
    """The old-schema key fields to ALSO try on a read miss, or None.

    The provenance key schema (effective temperature, thinking state) orphans
    two kinds of pre-schema entries, and only one deserves it:

      * entries whose producing call BEHAVED differently — DeepSeek and
        sonnet-5/opus-5 extract output from when thinking ran by default.
        Serving those under a thinking-off key is the poisoning the schema
        exists to prevent. No fallback, on purpose.
      * entries whose key merely CARRIED an inert field — a temperature the
        model rejected (opus-4-7/4-8, fable, claude-cli) or ignored. The
        annotation is byte-for-byte what an identical call would produce
        today; orphaning it re-bills real annotation stock for nothing, and
        on the disabled claude-cli path converts a cached read into a hard
        RuntimeError.

    The predicate separating them is exact: this call sends a thinking
    parameter iff thinking_fp is not None, and old code never sent one — so
    when thinking_fp is None the old call and this call hit the same API
    default and behaved identically, and the only key difference is the
    recorded-but-inert temperature. This is deliberately NOT a user flag:
    key identity must not depend on ambient configuration, and a flag's
    state is recorded nowhere — the fallback is applied exactly where it is
    provably safe and nowhere else.
    """
    if thinking_fp is not None or eff_temp == temperature:
        return None
    return {"temperature": temperature}


def _stash_read(stash, key, legacy_key, model):
    """(hit, value) for `key`, falling back to a safe legacy-schema key.

    A legacy hit is copied forward under the new key, so the migration is
    one read per item and the legacy schema ages out of the hot path.
    """
    if key in stash:
        return True, stash[key]
    if legacy_key is not None and legacy_key in stash:
        value = stash[legacy_key]
        stash[key] = value
        if model not in _WARNED_LEGACY_READS:
            _WARNED_LEGACY_READS.add(model)
            log.info(
                "%s: served from a pre-provenance-schema cache key (the old "
                "key recorded a temperature the model never applied; the "
                "annotation itself is identical) and copied forward under "
                "the new key. This is once-per-item; no action needed.",
                model,
            )
        return True, value
    return False, None


def _custom_key(cache_key, model, schema_name=None):
    """A caller-supplied stash key, completed with the coder's identity.

    A cache_key names the WORK UNIT (text_id, chunk, ...); without the model
    in it, two models' annotations read back as one history — df shows
    model="" and results_history reads cross-model disagreement as one
    model's own variance. setdefault, not overwrite: keys that already carry
    a model (SequentialTask's chunk keys do) stay byte-identical, so
    existing caches remain reachable.
    """
    key = dict(cache_key)
    key.setdefault("model", model)
    if schema_name:
        key.setdefault("schema", schema_name)
    return key


def _schema_to_json_spec(schema):
    """Convert a Pydantic model (or list[Model]) to a JSON schema description for the prompt."""
    is_list, item_schema = _unwrap_schema(schema)
    json_schema = item_schema.model_json_schema()
    schema_json = json.dumps(json_schema, indent=2)
    if is_list:
        return f"a JSON array of objects, where each object matches this schema:\n{schema_json}"
    else:
        return f"a JSON object matching this schema:\n{schema_json}"


def _unwrap_schema(schema):
    """Unwrap list[Model] into (True, Model) or (False, schema)."""
    origin = getattr(schema, "__origin__", None)
    if origin is list:
        args = schema.__args__
        return True, args[0]
    return False, schema


def _schema_name(schema):
    """Get a stable name for a schema, handling list[Model]."""
    is_list, item_schema = _unwrap_schema(schema)
    name = item_schema.__name__
    return f"list[{name}]" if is_list else name


def _format_examples(examples, schema):
    """Format few-shot examples into prompt text."""
    if not examples:
        return ""
    parts = []
    for i, (input_text, output) in enumerate(examples, 1):
        if hasattr(output, "model_dump_json"):
            output_json = output.model_dump_json(indent=2)
        elif isinstance(output, dict):
            output_json = json.dumps(output, indent=2)
        elif isinstance(output, list):
            output_json = json.dumps(
                [o.model_dump() if hasattr(o, "model_dump") else o for o in output],
                indent=2,
            )
        else:
            output_json = str(output)
        parts.append(f"Example {i} input:\n{input_text}\n\nExample {i} output:\n{output_json}")
    return "\n\n---\n\n".join(parts)


def _build_extract_prompt(prompt, schema, system_prompt=None, examples=None):
    """Build the full system prompt and user prompt for structured extraction."""
    schema_spec = _schema_to_json_spec(schema)

    system_parts = []
    if system_prompt:
        system_parts.append(system_prompt)
    system_parts.append(
        f"You must respond with ONLY valid JSON matching the following specification — "
        f"no markdown fencing, no commentary, no extra text.\n\n"
        f"Respond with {schema_spec}"
    )

    examples_text = _format_examples(examples, schema)
    if examples_text:
        system_parts.append(f"Here are some examples:\n\n{examples_text}")

    full_system = "\n\n".join(system_parts)
    return full_system, prompt


def _value_fits_field(value, field):
    """True if `value` validates against a single field's annotation."""
    from pydantic import TypeAdapter
    try:
        TypeAdapter(field.annotation).validate_python(value)
        return True
    except Exception:
        return False


def _is_permissive(annotation):
    """True for annotations that validate anything (Any, object).

    A permissive field matching a stray value is not evidence the model
    returned that field's bare value — it is the field that matches when
    nothing else does, so counting it turns the exactly-one-match rule into
    a guess and the reprompt into an accusation about the wrong field.
    """
    from typing import Any
    return annotation is Any or annotation is object


def _diagnose_partial_response(parsed, schema):
    """Name the field whose bare value the model returned, if that's what it did.

    Observed across Anthropic, OpenAI and DeepSeek: instead of the whole
    object, the model emits the value of one field —
    ``['SEQUENCE', 'SPECIFICITY']`` rather than
    ``{"relations": ["SEQUENCE", ...], ...}``. Pydantic reports only "expected
    dict, got list", so a generic invalid-JSON reprompt asks the model to guess
    what it did wrong when the caller can already tell it.

    Returns None unless exactly one field fits — an ambiguous match is a guess,
    and a wrong field name in the reprompt is worse than no field name.
    """
    if isinstance(parsed, dict) or parsed is None:
        return None
    is_list, item_schema = _unwrap_schema(schema)
    if is_list:
        # A bare list is the expected shape here, so it is not evidence of a
        # partial response.
        return None
    try:
        fields = item_schema.model_fields
    except Exception:
        return None
    matches = [name for name, f in fields.items()
               if not _is_permissive(f.annotation)
               and _value_fits_field(parsed, f)]
    return matches[0] if len(matches) == 1 else None


def _retry_prompt(prompt, partial_field=None):
    """The reprompt sent after a response failed to parse/validate.

    One definition shared by extract, extract_imap, and Task.retry_prompt, so
    a hand-administered retry matches the API path's retry exactly.

    Args:
        partial_field: When the model returned only one field's value (see
            _diagnose_partial_response), naming it converts "that wasn't valid
            JSON" — true but useless, since the JSON parsed fine — into the
            actual defect.
    """
    if partial_field:
        complaint = (
            f"Your previous response contained only the value of the "
            f"`{partial_field}` field. Return the COMPLETE object with every "
            f"required field, not the value of a single field."
        )
    else:
        complaint = (
            "Your previous response was not valid JSON. "
            "Return ONLY valid JSON matching the schema, nothing else."
        )
    return f"{complaint}\n\n{prompt}"


def _parse_json_response(text):
    """Extract JSON from an LLM response, handling markdown fencing and surrounding text.

    Falls back to json_repair for common malformations (e.g. missing opening
    quotes on string values — observed with qwen3.5 on large multi-field schemas).
    """
    text = text.strip()
    # strip markdown fencing
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # try as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find first [ or { and match to last ] or }
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    # Last resort: json_repair for malformed output from local models.
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        # repair_json returns '' when completely unrecoverable.
        if repaired not in ('', None):
            return repaired
    except ImportError:
        pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def _validate_parsed(data, schema):
    """Validate parsed JSON against the Pydantic schema."""
    is_list, item_schema = _unwrap_schema(schema)
    data = _unwrap_envelopes(data, item_schema)
    if is_list:
        if not isinstance(data, list):
            data = [data]
        return [item_schema.model_validate(_unwrap_envelopes(item, item_schema))
                for item in data]
    else:
        return item_schema.model_validate(data)


def _unwrap_envelopes(data, item_schema):
    """Apply all known output-envelope unwraps in sequence."""
    data = _unwrap_schema_envelope(data, item_schema)
    data = _unwrap_per_field_envelope(data)
    return data


def _unwrap_schema_envelope(data, item_schema):
    """Some models (observed: gemma4) echo the JSON-schema structure back as
    an output envelope: {"properties": {...actual fields...}}. Detect and
    unwrap when the inner dict clearly matches the schema better than the
    outer one."""
    if not (isinstance(data, dict) and "properties" in data
            and isinstance(data["properties"], dict)):
        return data
    try:
        expected = set(item_schema.model_fields.keys())
    except Exception:
        return data
    outer_match = len(set(data.keys()) & expected)
    inner_match = len(set(data["properties"].keys()) & expected)
    if inner_match > outer_match and inner_match > 0:
        return data["properties"]
    return data


def _unwrap_per_field_envelope(data):
    """Some models (observed: llama-3.1-70b on large schemas) wrap each field
    value as a JSON-schema field descriptor containing a 'value' key — e.g.
    {"type": "boolean", "value": false} or {"description": "...", "title": "...",
    "type": "boolean", "value": false}. Detect when every value in the dict
    has both 'type' and 'value' keys and unwrap to the bare values."""
    if not isinstance(data, dict) or not data:
        return data
    for v in data.values():
        if not (isinstance(v, dict) and "type" in v and "value" in v):
            return data
    return {k: v["value"] for k, v in data.items()}


class LLM:
    """Unified LLM interface with automatic caching via HashStash.

    Usage:
        llm = LLM("claude-sonnet-4-20250514")
        text = llm.generate("What is the plot of Pamela?")

        # cached: identical calls return instantly
        text2 = llm.generate("What is the plot of Pamela?")

        # structured extraction
        from pydantic import BaseModel, Field
        class Character(BaseModel):
            name: str
            role: str = Field(description="Role in the narrative")
        characters = llm.extract("Describe the characters in Pamela.",
                                  schema=list[Character])

        # with images
        llm = LLM("gemini-2.5-flash")
        text = llm.generate("Describe this page.", images=["page1.png"])
    """

    def __init__(self, model=DEFAULT_MODEL, system_prompt=None, temperature=DEFAULT_TEMPERATURE,
                 max_tokens=DEFAULT_MAX_TOKENS, stash=None, cache_ttl=None,
                 usage=None, raw_log=None):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self.usage = usage if usage is not None else UsageTracker()
        self.stash = stash if stash is not None else HashStash(
            STASH_PATH, engine="pairtree", append_mode=True,
        )
        # Raw-response sidecar (rawlog.RawLog), opt-in: None/False = off.
        # Off is genuinely off — no sink is constructed, providers skip
        # serialization, and no sidecar path runs. Not part of any stash
        # key: recording what came back does not change what came back.
        from .rawlog import RawLog
        self.raw_log = RawLog.resolve(raw_log)

    def _raw_sink(self, key, transport, attempt=None):
        """Sink recording a serialized provider body under `key`, or None
        when the sidecar is off (None means providers do no raw work)."""
        if self.raw_log is None:
            return None
        provider = route_provider(self.model).__name__.replace("call_", "")
        return lambda body: self.raw_log.record(
            key, body, transport=transport, model=self.model,
            provider=provider, attempt=attempt)

    def _provider_kwargs(self, kwargs):
        """Per-call provider options this instance contributes.

        cache_ttl and usage_sink are instance-level and deliberately not part
        of the stash key: neither changes the response text.
        """
        out = dict(kwargs)
        out.setdefault("usage_sink", self.usage.record)
        if self.cache_ttl is not None:
            out.setdefault("cache_ttl", self.cache_ttl)
        return out

    def _resolve(self, system_prompt=None, temperature=None, max_tokens=None):
        """Resolve per-call overrides against instance defaults."""
        return (
            system_prompt if system_prompt is not None else self.system_prompt,
            temperature if temperature is not None else self.temperature,
            max_tokens if max_tokens is not None else self.max_tokens,
        )

    def generate(self, prompt, system_prompt=None, temperature=None,
                 max_tokens=None, images=None, metadata=None, force=False,
                 cache_key=None, **kwargs):
        """Generate text from the LLM, with caching.

        Args:
            prompt: The user prompt.
            system_prompt: Override instance system_prompt for this call.
            temperature: Override instance temperature for this call.
            max_tokens: Override instance max_tokens for this call.
            images: List of images (file paths, bytes, or PIL Images).
            metadata: Dict of user-defined metadata to store with the cache entry.
            force: If True, bypass cache and force a new generation.
            cache_key: Optional dict to override the auto-generated stash key.
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: The generated text.
        """
        system_prompt, temperature, max_tokens = self._resolve(
            system_prompt, temperature, max_tokens,
        )
        eff_temp, thinking_fp = _sampling_fingerprint(
            self.model, temperature, kwargs)
        legacy_key = None
        if cache_key is not None:
            # No legacy fallback for custom keys: a bare pre-schema custom
            # key could hold ANY model's annotation — that ambiguity is the
            # defect, not an inert field.
            key = _custom_key(cache_key, self.model)
        else:
            key = _make_key(prompt, self.model, system_prompt, eff_temp,
                            max_tokens, images=images, metadata=metadata,
                            thinking=thinking_fp)
            legacy = _legacy_key_kwargs(self.model, temperature, eff_temp,
                                        thinking_fp)
            if legacy is not None:
                legacy_key = _make_key(prompt, self.model, system_prompt,
                                       legacy["temperature"], max_tokens,
                                       images=images, metadata=metadata)

        if not force:
            hit, value = _stash_read(self.stash, key, legacy_key, self.model)
            if hit:
                return value

        call_kwargs = self._provider_kwargs(kwargs)
        call_kwargs.setdefault("raw_sink", self._raw_sink(key, "sync"))
        result = _call_provider(
            prompt=prompt,
            model=self.model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            images=images,
            **call_kwargs,
        )
        self.stash[key] = result
        return result

    def extract(self, prompt, schema, system_prompt=None, examples=None,
                temperature=None, max_tokens=None, images=None, metadata=None,
                force=False, retries=1, cache_key=None, prebuilt=False,
                raw_transport=None, **kwargs):
        """Extract structured data from text using a Pydantic schema.

        Args:
            prompt: The input text to extract from.
            schema: A Pydantic BaseModel class, or list[BaseModel] for multiple items.
            system_prompt: Domain-specific instructions prepended to the schema prompt.
            examples: Few-shot examples as list of (input_str, output) tuples.
                      Output can be a BaseModel instance, dict, or list thereof.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            images: List of images (file paths, bytes, or PIL Images).
            metadata: Dict of user-defined metadata to store with the cache entry.
            force: If True, bypass cache.
            retries: Number of retries on malformed JSON (default 1).
            cache_key: Optional dict to use as the stash key instead of the
                       auto-generated prompt-based key. Useful for sequential
                       pipelines where the prompt varies but the identity of
                       the work unit is stable (e.g. text_id + passage_seq).
            raw_transport: Transport label for raw-response sidecar entries
                       (default "sync"). The batch path passes "sync-probe" /
                       "sync-fallback" so sidecar envelopes agree with the
                       usage rows. No effect unless the LLM has raw_log set.
            **kwargs: Additional provider-specific arguments.

        Returns:
            A validated Pydantic model instance (or list of instances).
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        if prebuilt:
            # The caller holds an ALREADY-BUILT (instrument, user prompt)
            # pair — the batch fallback path re-running an item whose key
            # stores the full instrument as its system_prompt. Rebuilding
            # would append a second contract block: the item would be
            # administered under a different instrument than its batchmates
            # and stashed under an unreachable key, re-billed forever.
            full_system, user_prompt = system_prompt, prompt
        else:
            full_system, user_prompt = _build_extract_prompt(
                prompt, schema, system_prompt=system_prompt, examples=examples,
            )
        s_name = _schema_name(schema)
        eff_temp, thinking_fp = _sampling_fingerprint(
            self.model, temperature, kwargs)
        legacy_key = None
        if cache_key is not None:
            key = _custom_key(cache_key, self.model, schema_name=s_name)
        else:
            key = _make_key(user_prompt, self.model, full_system, eff_temp,
                            max_tokens, schema_name=s_name, images=images,
                            metadata=metadata, thinking=thinking_fp)
            legacy = _legacy_key_kwargs(self.model, temperature, eff_temp,
                                        thinking_fp)
            if legacy is not None:
                legacy_key = _make_key(user_prompt, self.model, full_system,
                                       legacy["temperature"], max_tokens,
                                       schema_name=s_name, images=images,
                                       metadata=metadata)

        hit, cached = (False, None)
        if not force:
            hit, cached = _stash_read(self.stash, key, legacy_key, self.model)
        if hit:
            if isinstance(cached, str):
                try:
                    return _validate_parsed(_parse_json_response(cached), schema)
                except Exception as e:
                    # Schema changed since the response was cached — fall
                    # through and recompute (matches extract_imap behavior).
                    log.warning(
                        "extract: cached response for %s no longer parses/"
                        "validates (%s); recomputing", s_name, e,
                    )
            else:
                return cached

        last_error = None
        raw = None
        attempts = 0
        partial_field = None
        for attempt in range(1 + retries):
            attempts += 1
            if attempt == 0:
                call_system = full_system
                call_prompt = user_prompt
            else:
                log.warning(
                    "extract retry %d/%d for %s (model=%s): %s",
                    attempt, retries, s_name, self.model, last_error,
                )
                call_system = full_system
                call_prompt = _retry_prompt(user_prompt, partial_field)

            # Provider call inside the try so transient network/API errors
            # consume a retry instead of aborting (parity with extract_imap).
            # `parsed` is re-bound every attempt: reading it via locals() let
            # a previous attempt's parse leak into this attempt's diagnosis,
            # so a network error was reprompted as "you returned only the
            # value of the X field" — a complaint about JSON that this
            # attempt never produced.
            parsed = None
            call_kwargs = self._provider_kwargs(kwargs)
            call_kwargs.setdefault(
                "raw_sink",
                self._raw_sink(key, raw_transport or "sync",
                               attempt=attempt))
            try:
                raw = _call_provider(
                    prompt=call_prompt,
                    model=self.model,
                    system_prompt=call_system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    images=images,
                    **call_kwargs,
                )
                parsed = _parse_json_response(raw)
                result = _validate_parsed(parsed, schema)
                self.stash[key] = raw
                return result
            except Exception as e:
                partial_field = _diagnose_partial_response(parsed, schema)
                # An identical repeat is deterministic — a further billed
                # attempt cannot succeed, so stop rather than burn it, unless
                # the diagnosis gives the retry something new to say.
                if (last_error is not None and not partial_field
                        and _Breaker.signature(e) == _Breaker.signature(last_error)):
                    last_error = e
                    break
                last_error = e
                continue

        # Report attempts actually made, not the configured ceiling — they
        # differ whenever an identical repeat cut the retries short.
        log.error(
            "extract giving up on %s after %d attempts (model=%s). "
            "Last error: %s. Raw (truncated): %s",
            s_name, attempts, self.model, last_error, (raw or '')[:400],
        )
        raise ValueError(
            f"Failed to extract valid {s_name} after {attempts} attempts "
            f"(retries={retries}; a repeat of the identical error stops early, "
            f"since a further billed attempt cannot succeed). "
            f"Last error: {last_error}. "
            f"Raw response (truncated): {(raw or '')[:400]}"
        )

    def map(self, prompts, system_prompt=None, temperature=None,
            max_tokens=None, images_list=None, metadata_list=None,
            num_workers=4, force=False, errors=None, **kwargs):
        """Generate text for multiple prompts, with caching and parallelism.

        Args:
            prompts: List of prompt strings.
            system_prompt: Override instance system_prompt.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            images_list: List of image lists, one per prompt (or None).
            metadata_list: List of metadata dicts, one per prompt (or None).
            num_workers: Number of parallel threads (default 4).
            force: If True, bypass cache and force new generations.
            errors: Optional dict for per-item failure diagnostics; see
                ``extract_imap``.
            **kwargs: Additional provider-specific arguments.

        Returns:
            list[str | None]: Generated texts in the same order as prompts.
                Entries are None for prompts whose provider call failed
                after logging (the rest of the batch still completes).
        """
        system_prompt, temperature, max_tokens = self._resolve(
            system_prompt, temperature, max_tokens,
        )

        results = [None] * len(prompts)
        to_compute = []

        eff_temp, thinking_fp = _sampling_fingerprint(
            self.model, temperature, kwargs)
        legacy = _legacy_key_kwargs(self.model, temperature, eff_temp,
                                    thinking_fp)
        for i, prompt in enumerate(prompts):
            images = images_list[i] if images_list else None
            metadata = metadata_list[i] if metadata_list else None
            key = _make_key(prompt, self.model, system_prompt, eff_temp,
                            max_tokens, images=images, metadata=metadata,
                            thinking=thinking_fp)
            legacy_key = None if legacy is None else _make_key(
                prompt, self.model, system_prompt, legacy["temperature"],
                max_tokens, images=images, metadata=metadata)
            hit, value = (False, None)
            if not force:
                hit, value = _stash_read(self.stash, key, legacy_key,
                                         self.model)
            if hit:
                results[i] = value
            else:
                to_compute.append((i, prompt, key, images))

        total = len(prompts)
        fresh = len(to_compute)
        cached = total - fresh
        if total >= 10:
            log.info("generate_map: %d/%d cached, %d API calls needed (model=%s)",
                     cached, total, fresh, self.model)

        if not to_compute:
            return results

        def _do_one(item):
            i, prompt, key, images = item
            call_kwargs = self._provider_kwargs(kwargs)
            call_kwargs.setdefault("raw_sink", self._raw_sink(key, "sync"))
            try:
                result = _call_provider(
                    prompt=prompt,
                    model=self.model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    images=images,
                    **call_kwargs,
                )
            except Exception as e:
                log.error("map: prompt %d failed (model=%s): %s",
                          i, self.model, e)
                if errors is not None:
                    errors[i] = {
                        "index": i,
                        "error": f"{type(e).__name__}: {e}",
                        "exception": e,
                        "attempts": 1,
                        "metadata": metadata_list[i] if metadata_list else None,
                        "prompt_head": (prompt or "")[:200],
                        "raw": "",
                    }
                return i, None
            self.stash[key] = result
            return i, result

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_do_one, item): item for item in to_compute}
                for future in tqdm(as_completed(futures), total=len(to_compute), desc=f"Generating ({self.model})"):
                    i, result = future.result()
                    results[i] = result
        finally:
            # Firing boundary, same discipline as extract_imap's. The
            # context manager above joined the workers, so the flushed
            # counters are final.
            if self.raw_log is not None:
                self.raw_log.flush_receipt()

        return results

    def extract_imap(self, prompts, schema, system_prompt=None, examples=None,
                     temperature=None, max_tokens=None, images_list=None,
                     metadata_list=None, num_workers=4,
                     force=False, retries=1, verbose=False, errors=None,
                     fail_fast=5, warm_cache=True, per_item_usage=None,
                     **kwargs):
        """Extract structured data from multiple prompts, yielding as each completes.

        Yields (index, result) tuples in completion order — cached items first,
        then API results as threads finish. Each result is cached to the stash
        the moment it completes, so partial runs are resumable.

        Args:
            prompts: List of input texts.
            schema: Pydantic BaseModel class or list[BaseModel].
            system_prompt: Domain-specific instructions.
            examples: Few-shot examples as list of (input_str, output) tuples.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            images_list: List of image lists, one per prompt (or None).
            metadata_list: List of metadata dicts, one per prompt (or None).
            num_workers: Number of parallel threads (default 4).
            force: If True, bypass cache.
            retries: Number of retries on malformed JSON (default 1).
            verbose: If True, print a compact per-call summary as each result
                lands (plays nicely with tqdm via tqdm.write). If a callable,
                use it as a custom formatter — signature
                (i: int, prompt: str, metadata: dict|None, result) -> str.
            errors: Optional dict. A yielded None says only that *some* item
                failed; pass a dict and each failure is recorded as
                errors[index] = {'index', 'error', 'exception', 'attempts',
                'metadata', 'prompt_head', 'raw'} (plus 'duplicate_of' when
                the item shared a de-duplicated call). Only ATTEMPTED
                failures are recorded: items a fail_fast abort prevented
                from starting yield None with no entry, so `len(errors)`
                counts real failures rather than the batch size. Written
                from worker threads, so read it after the iteration
                finishes.
            fail_fast: Stops a batch whose item failures are systematic.
                  * dict — _Breaker settings, e.g. {'rate': 0.5},
                    {'floor': 10}; {} means the defaults.
                  * int N — shorthand for {'floor': N}: abort once the first
                    N completed items all failed identically.
                  * True — the defaults; False/None — disabled.
                Anything else raises TypeError — an earlier version read
                every non-dict truthy value as merely "enabled", so
                fail_fast=50 behaved exactly like fail_fast=5 and {} (the
                natural spelling of "defaults") disabled the breaker.
                Aborts raise BatchAborted; completed results are already in
                the stash, so a rerun resumes rather than repays.
            warm_cache: Run the first uncached item alone before fanning out
                (default True). A prompt-cache entry is not readable until the
                write completes, so N parallel first calls otherwise each pay
                the write premium instead of one write plus N-1 cheap reads.
            per_item_usage: Optional dict for per-item token counts, keyed by
                index: {'index', 'calls', 'input_tokens', 'output_tokens',
                'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens',
                'response_model'}. Retries accumulate into the same entry, so
                output_tokens is an item's total billed output; response_model
                is from the last attempt that reported one. Useful as a free
                difficulty signal — a longer answer on an item usually means
                the coder had more to say about it — with no need to
                re-tokenise the text. Only live provider calls are recorded;
                locally-cached items never reach a provider and get no entry.
                An index that shared another index's de-duplicated call gets a
                zero-token entry with 'duplicate_of' pointing at the index
                whose call produced its annotation.
            **kwargs: Additional provider-specific arguments.

        Yields:
            tuple: (index, result) where result is a validated Pydantic model
                instance (or list thereof), or None on failure.
        """
        prompts = list(prompts)
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        full_system, _ = _build_extract_prompt(
            "", schema, system_prompt=system_prompt, examples=examples,
        )
        s_name = _schema_name(schema)

        def _default_verbose_line(i, prompt, metadata, result, from_cache=False):
            meta_str = ""
            if isinstance(metadata, dict) and metadata:
                meta_str = " ".join(f"{k}={v}" for k, v in metadata.items() if v not in ("", None))
            try:
                if isinstance(result, list):
                    payload = f"[list x{len(result)}]"
                    if result:
                        payload += " " + ", ".join(
                            f"{k}={v!r}" for k, v in list(result[0].model_dump().items())[:3]
                        )
                else:
                    payload = ", ".join(
                        f"{k}={v!r}" for k, v in list(result.model_dump().items())[:4]
                    )
            except Exception:
                payload = str(result)[:120]
            prompt_head = (prompt or "").splitlines()[0][:60]
            tag = "⊛" if from_cache else "→"
            return f"[{i:>5}] {meta_str}  {tag} {payload}  ({prompt_head!r})"

        def _emit_verbose(i, prompt, metadata, result, from_cache):
            try:
                line = (verbose(i, prompt, metadata, result) if callable(verbose)
                        else _default_verbose_line(i, prompt, metadata, result, from_cache))
                tqdm.write(line)
            except Exception as e:
                tqdm.write(f"[{i}] <verbose formatter error: {e}>")

        to_compute = []
        seen_keys = {}   # frozen key -> first index submitted
        dup_of = {}      # first index -> [duplicate indices awaiting its result]

        eff_temp, thinking_fp = _sampling_fingerprint(
            self.model, temperature, kwargs)
        legacy = _legacy_key_kwargs(self.model, temperature, eff_temp,
                                    thinking_fp)
        for i, prompt in enumerate(prompts):
            images = images_list[i] if images_list else None
            metadata = metadata_list[i] if metadata_list else None
            key = _make_key(prompt, self.model, full_system, eff_temp,
                            max_tokens, schema_name=s_name, images=images,
                            metadata=metadata, thinking=thinking_fp)
            legacy_key = None if legacy is None else _make_key(
                prompt, self.model, full_system, legacy["temperature"],
                max_tokens, schema_name=s_name, images=images,
                metadata=metadata)
            hit, cached = (False, None)
            if not force:
                hit, cached = _stash_read(self.stash, key, legacy_key,
                                          self.model)
            if hit:
                if isinstance(cached, str):
                    try:
                        result = _validate_parsed(_parse_json_response(cached), schema)
                        if verbose:
                            _emit_verbose(i, prompt, metadata, result, from_cache=True)
                        yield i, result
                        continue
                    except Exception:
                        pass
                else:
                    if verbose:
                        _emit_verbose(i, prompt, metadata, cached, from_cache=True)
                    yield i, cached
                    continue
            # Duplicate prompts in one batch share a single API call.
            frozen = json.dumps(key, sort_keys=True, default=str)
            if frozen in seen_keys:
                dup_of.setdefault(seen_keys[frozen], []).append(i)
                continue
            seen_keys[frozen] = i
            to_compute.append((i, prompt, key, images))

        total = len(prompts)
        fresh = len(to_compute)
        n_cached = total - fresh
        if total >= 10:
            log.info("extract_imap: %d/%d cached, %d API calls needed (model=%s)",
                     n_cached, total, fresh, self.model)
            if n_cached == 0 and fresh >= 100:
                has_old_entries = next(iter(self.stash.items()), None) is not None
                if has_old_entries:
                    log.warning(
                        "extract_imap: 0/%d cached despite existing entries in %s's stash. "
                        "System prompt, examples, schema, temperature, or max_tokens may "
                        "have changed since the last run — previous cache keys are unreachable.",
                        total, s_name,
                    )

        if not to_compute:
            return

        def _record_error(i, prompt, exc, attempts, raw):
            """Record one item's failure so the caller can tell which None is which."""
            if errors is None:
                return
            errors[i] = {
                "index": i,
                "error": f"{type(exc).__name__}: {exc}",
                "exception": exc,
                "attempts": attempts,
                "metadata": metadata_list[i] if metadata_list else None,
                "prompt_head": (prompt or "")[:200],
                "raw": (raw or "")[:1000],
            }

        if fail_fast is None or fail_fast is False:
            breaker = _Breaker(enabled=False)
        elif fail_fast is True:
            breaker = _Breaker()
        elif isinstance(fail_fast, dict):
            breaker = _Breaker(**fail_fast)
        elif isinstance(fail_fast, int):
            if fail_fast == 0:
                # 0 reads as "off" under the falsy convention (False and
                # None both disable); as a floor it would mean abort on the
                # FIRST failed item — near-tightest, from the spelling of
                # loosest. Honour the convention.
                breaker = _Breaker(enabled=False)
            else:
                # An int is a loosening/tightening of WHEN aborting is
                # allowed, so it must gate both conditions: floor alone
                # would let the 20%-rate condition fire at 30 outcomes and
                # silently override a caller's fail_fast=50.
                breaker = _Breaker(floor=fail_fast,
                                   min_outcomes=max(30, fail_fast))
        else:
            raise TypeError(
                f"fail_fast={fail_fast!r}: pass a dict of _Breaker settings, "
                f"an int floor, True for defaults, or False to disable."
            )

        # Batch-scoped tracker beside the Task-lifetime one. The finally
        # block's summary and cache warning describe THIS batch: the
        # lifetime tracker's history meant one well-cached batch immunised a
        # Task against the no-caching warning for the rest of the process,
        # and the logged hit rate averaged every batch since construction.
        batch_usage = UsageTracker()

        def _item_sink(i):
            """Usage sink that attributes one call's tokens to item `i`.

            Retries add further calls for the same item, so entries accumulate
            rather than overwrite — output_tokens for an item is its total
            billed output, which is what a cost or length analysis wants.
            """
            def sink(usage):
                self.usage.record(usage)
                batch_usage.record(usage)
                if per_item_usage is None:
                    return
                entry = per_item_usage.setdefault(i, {
                    "index": i, "calls": 0, "input_tokens": 0,
                    "output_tokens": 0, "cache_read_tokens": 0,
                    "cache_write_tokens": 0, "reasoning_tokens": 0,
                    "response_model": None,
                })
                entry["calls"] += 1
                for k in ("input_tokens", "output_tokens", "cache_read_tokens",
                          "cache_write_tokens", "reasoning_tokens"):
                    entry[k] += usage.get(k, 0)
                # Per-annotation provenance: which model answered for THIS
                # item. A run-level counter cannot attribute a mid-run alias
                # flip to particular annotations, which is the whole question
                # a provenance record exists to answer. Retries: the last
                # attempt that REPORTED a served id wins — the kept text is
                # the last attempt's, and an attempt whose response carried
                # no usage object must not blank an id an earlier one saw.
                if usage.get("response_model"):
                    entry["response_model"] = usage["response_model"]
            return sink

        def _do_one(item):
            i, prompt, key, images = item
            last_error = None
            raw = None
            attempts = 0
            partial_field = None
            call_kwargs = dict(kwargs)
            call_kwargs["usage_sink"] = _item_sink(i)
            for attempt in range(1 + retries):
                if breaker.tripped:
                    # Another item already proved this failure is systematic.
                    break
                call_prompt = prompt
                if attempt > 0:
                    log.warning(
                        "extract_imap retry %d/%d for prompt %d (model=%s): %s",
                        attempt, retries, i, self.model, last_error,
                    )
                    call_prompt = _retry_prompt(prompt, partial_field)
                attempts += 1
                breaker.record_attempt()
                # `parsed` is re-bound every attempt (see extract): a stale
                # binding turned a network error into a reprompt about JSON
                # this attempt never produced.
                parsed = None
                # Rebuilt per attempt for the attempt marker; a
                # caller-supplied raw_sink in kwargs wins, matching
                # usage_sink's setdefault discipline.
                if "raw_sink" not in kwargs:
                    call_kwargs["raw_sink"] = self._raw_sink(
                        key, "sync", attempt=attempt)
                try:
                    raw = _call_provider(
                        prompt=call_prompt,
                        model=self.model,
                        system_prompt=full_system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        images=images,
                        **self._provider_kwargs(call_kwargs),
                    )
                    parsed = _parse_json_response(raw)
                    result = _validate_parsed(parsed, schema)
                    self.stash[key] = raw
                    breaker.record_success()
                    return i, result
                except Exception as e:
                    # Diagnose the specific "returned one field's value"
                    # failure so the retry can name it instead of complaining
                    # about JSON that actually parsed fine.
                    partial_field = _diagnose_partial_response(parsed, schema)
                    # A repeat of the identical error on the same item is
                    # deterministic; a further attempt costs money and cannot
                    # succeed — unless we now have something new to say.
                    repeat = (last_error is not None
                              and _Breaker.signature(e) == _Breaker.signature(last_error))
                    last_error = e
                    if repeat and not partial_field:
                        break
                    continue
            if attempts == 0:
                # The breaker tripped before this item ever ran. Not an item
                # failure: no error entry, no log line. An earlier version
                # recorded these as failures with attempts=0, so one abort
                # made len(errors) report the batch size — 131 "failures" on
                # a run with 3.
                return i, None
            # The item's FINAL outcome is a failure; only now does it count
            # toward the breaker's rate — an item that retried into success
            # is a success.
            breaker.record_failure(last_error)
            log.error(
                "extract_imap giving up on prompt %d after %d attempts (model=%s). "
                "Last error: %s. Raw (truncated): %s",
                i, attempts, self.model, last_error,
                (raw or '')[:400],
            )
            _record_error(i, prompt, last_error, attempts, raw)
            return i, None

        def _fan_out(i, result):
            """Yield an item's result to it and every duplicate index.

            One definition for the warm-cache pre-flight and the executor
            loop: when they diverged, whether a failed item's duplicates
            appeared in `errors` depended on whether the item happened to be
            the warm one.
            """
            for j in (i, *dup_of.get(i, ())):
                if j != i:
                    # Duplicates of a failed item failed too — give each its
                    # own entry so `errors` is keyed by every failed index.
                    if errors is not None and i in errors:
                        errors[j] = {
                            **errors[i], "index": j, "duplicate_of": i,
                            "metadata": metadata_list[j] if metadata_list else None,
                        }
                    # And a provenance row: j's annotation came from i's
                    # call. Zero tokens (nothing extra was billed), but the
                    # served id and the pointer, so a per-annotation table
                    # has no holes that read as "unknown model".
                    if per_item_usage is not None and i in per_item_usage:
                        per_item_usage[j] = {
                            "index": j, "calls": 0, "input_tokens": 0,
                            "output_tokens": 0, "cache_read_tokens": 0,
                            "cache_write_tokens": 0, "reasoning_tokens": 0,
                            "response_model":
                                per_item_usage[i].get("response_model"),
                            "duplicate_of": i,
                        }
                if verbose:
                    _emit_verbose(j, prompts[j],
                                  metadata_list[j] if metadata_list else None,
                                  result, from_cache=(j != i))
                yield j, result

        # Warm the prompt cache on one item before fanning out. The entry is
        # not readable until its write lands, so without this the first
        # num_workers calls each pay the ~1.25x write premium on the shared
        # system block rather than one write plus cheap reads.
        try:
            if (warm_cache and num_workers > 1
                    and len(to_compute) > num_workers
                    and len(full_system) > 2000):
                first, to_compute = to_compute[0], to_compute[1:]
                i, result = _do_one(first)
                yield from _fan_out(i, result)
                if breaker.tripped:
                    raise BatchAborted(breaker.error(self.model),
                                       errors=errors)

            executor = ThreadPoolExecutor(max_workers=num_workers)
            try:
                futures = {executor.submit(_do_one, item): item
                           for item in to_compute}
                for future in tqdm(as_completed(futures),
                                   total=len(to_compute),
                                   desc=f"Extracting {s_name} ({self.model})"):
                    i, result = future.result()
                    yield from _fan_out(i, result)
                    if breaker.tripped:
                        raise BatchAborted(breaker.error(self.model),
                                           errors=errors)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        finally:
            # Batch boundary: retain this firing's sidecar coverage
            # counters durably. Runs fire many times (resumption is the
            # normal case), so the run-level claim is a conjunction over
            # firings — a receipt that dies with the process cannot join
            # it. Join stragglers first: shutdown(wait=False) above lets
            # already-running futures continue, and a receipt flushed
            # under them records counters that are still moving.
            if self.raw_log is not None:
                if "executor" in locals():
                    executor.shutdown(wait=True)
                self.raw_log.flush_receipt()
            # The summary covers the warm-cache pre-flight too: an abort
            # raised there previously skipped this block entirely, losing
            # the batch receipt for exactly the runs that need it most.
            if total >= 10:
                # THIS batch's numbers, not the Task-lifetime aggregate.
                log.info("extract_imap %s", batch_usage.summary_line())
                warning = batch_usage.cache_warning(self.model)
                if warning:
                    log.warning("extract_imap: %s", warning)

    def extract_map(self, prompts, schema, system_prompt=None, examples=None,
                    temperature=None, max_tokens=None, images_list=None,
                    metadata_list=None, num_workers=4,
                    force=False, retries=1, verbose=False, errors=None,
                    fail_fast=5, warm_cache=True, per_item_usage=None,
                    **kwargs):
        """Extract structured data from multiple prompts, with caching and parallelism.

        Like extract_imap but collects all results into a list in prompt order.

        Args:
            errors: Optional dict for per-item failure diagnostics; see
                ``extract_imap``.
            fail_fast: Abort on repeated identical failures; see ``extract_imap``.
            warm_cache: Warm the prompt cache before fanning out; see
                ``extract_imap``.

        Returns:
            list: Validated Pydantic model instances (or lists thereof) in prompt order.

        Raises:
            BatchAborted: when fail_fast stops the batch. The exception
                carries the partial results computed so far (`.results`) —
                they are also in the stash, so a rerun after fixing the
                fault resumes rather than repays.
        """
        results = [None] * len(prompts)
        try:
            for i, result in self.extract_imap(
                prompts, schema, system_prompt=system_prompt, examples=examples,
                temperature=temperature, max_tokens=max_tokens,
                images_list=images_list, metadata_list=metadata_list,
                num_workers=num_workers, force=force, retries=retries,
                verbose=verbose, errors=errors, fail_fast=fail_fast,
                warm_cache=warm_cache, per_item_usage=per_item_usage, **kwargs,
            ):
                results[i] = result
        except BatchAborted as e:
            e.results = results
            raise
        return results

    def extract_batch(self, prompts, schema, **kwargs):
        """Batch-transport extract_map: 50% pricing on providers with a
        batch API, same stash keys, same receipts. See
        largeliterarymodels.batch.extract_batch for the contract (ledger
        semantics, probe-first, wait=False handles)."""
        from .batch import extract_batch
        return extract_batch(self, prompts, schema, **kwargs)

    def __repr__(self):
        return f"LLM(model={self.model!r})"
