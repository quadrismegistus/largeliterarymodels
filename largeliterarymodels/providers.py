"""LLM provider backends: Anthropic, OpenAI, Google GenAI, Claude CLI.

Each provider function takes a standard set of arguments and returns the
response text as a string. No litellm — direct SDK calls only.

Supports multimodal inputs via the `images` parameter: a list of file paths,
bytes, or PIL Image objects.
"""

import base64
import io
import logging
import os
import re

log = logging.getLogger(__name__)

# SDK clients are memoized per (provider, credentials, base_url, timeout):
# each client owns an httpx connection pool, and constructing one per call
# churns TCP connections/file descriptors under parallel batch runs.
_CLIENT_CACHE = {}


def _cached_client(cache_key, factory):
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = factory()
        _CLIENT_CACHE[cache_key] = client
    return client


def _get_key(env_var):
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Missing {env_var} in environment")
    return key


# DeepSeek's hosted API models. Bare names like 'deepseek-r1:8b' are local
# checkpoints and must NOT route to the paid API — use an explicit
# 'ollama/'/'lmstudio/' prefix for those.
#
# Verified against GET https://api.deepseek.com/models (2026-07-30): the API
# now lists ONLY deepseek-v4-flash and deepseek-v4-pro. The older names still
# resolve server-side, but both land on FLASH — including 'deepseek-reasoner',
# so asking for a specific tier silently gets you the cheap one, with no error.
# They stay routable so existing pins (and their cache keys) keep working, and
# warn once per process instead.
#
# Corrected 2026-08-04: this comment used to add that resolving to flash
# "silently gets you a NON-reasoning model". That was wrong, and wrong in the
# expensive direction — flash thinks, and thinks by default. See the thinking
# section below. Do not reason about a tier's behaviour from its name.
_DEEPSEEK_CURRENT_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
_DEEPSEEK_LEGACY_ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
}
_DEEPSEEK_API_MODELS = _DEEPSEEK_CURRENT_MODELS + tuple(_DEEPSEEK_LEGACY_ALIASES)

# (provider, requested, resolved) triples already logged, so a batch of N
# calls reports the resolution once rather than N times.
_LOGGED_RESOLUTIONS = set()

# DeepSeek legacy aliases already warned about (call_deepseek). Its own set:
# an earlier version parked bare model strings in _LOGGED_RESOLUTIONS beside
# the triples — no collision was possible, but a set with two meanings is a
# bug waiting for a third.
_WARNED_DEEPSEEK_ALIASES = set()


def _log_resolved_model(provider, requested, resolved):
    """Log the model id the server actually served, once per resolution.

    Hosted APIs quietly alias retired model names onto current checkpoints,
    so the id you asked for is not always the id you were billed for or the
    one that produced your annotations.

    Severity is split by what the mismatch means. A rolling alias resolving
    to its own dated snapshot (claude-sonnet-4-6 -> ...-20260219) is the
    routine case and logs at INFO; warning on it buries the mismatch this
    function exists to surface — a name landing on a DIFFERENT model, the
    deepseek-chat -> v4-flash shape — under hundreds of lines of the normal
    one.
    """
    if not resolved:
        return
    triple = (provider, requested, resolved)
    if triple in _LOGGED_RESOLUTIONS:
        return
    _LOGGED_RESOLUTIONS.add(triple)
    if resolved == requested:
        log.info("%s: serving model %r", provider, requested)
    elif resolved.startswith(requested):
        log.info(
            "%s: requested model %r resolved to dated snapshot %r — record "
            "the resolved id as the model of record",
            provider, requested, resolved,
        )
    else:
        log.warning(
            "%s: requested model %r resolved server-side to %r — a DIFFERENT "
            "model name, not a dated snapshot of the requested one. Record "
            "the resolved id, not the alias, as the model of record",
            provider, requested, resolved,
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible request repair
#
# Providers rename request parameters between model generations. The gpt-5
# tier renamed max_tokens -> max_completion_tokens and 400s on the old name,
# which means one stale constant takes out every call to a whole model family.
# Rather than branch on a version regex — which needs editing at each rename,
# and is what let this break in the first place — read the replacement out of
# the API's own error message and remember it for the process.
# ---------------------------------------------------------------------------

# Keyed by (provider, model), not the bare model string: a local proxy
# serving a model under a hosted provider's name must not poison the hosted
# provider's memo for the rest of the process.
_TOKEN_PARAM = {}        # (provider, model) -> output-length parameter name
_NO_TEMPERATURE = set()  # (provider, model) pairs that reject `temperature`

_PARAM_HINT_RE = re.compile(r"[Uu]se '([A-Za-z_][A-Za-z0-9_]*)' instead")


def _unsupported_param(exc, name):
    """True if `exc` is the API rejecting parameter `name`.

    `name` must be the FIRST quoted token after the "unsupported parameter"
    phrase. A bare substring check reads rejection errors that go on to LIST
    the supported parameters — "Unsupported parameter: 'top_p' ... Supported
    parameters: 'temperature', ..." — as rejections of every parameter they
    mention, which here meant dropping `temperature` permanently, filing a
    false do-not-call-this-temperature-controlled audit record, and never
    repairing the parameter actually at fault.
    """
    return re.search(
        r"unsupported parameter[^']*'%s'" % re.escape(name),
        str(exc), re.IGNORECASE,
    ) is not None


def _healed_token_param(exc, current):
    """The parameter name the API says to use instead of `current`, or None.

    Reading the replacement from the error self-heals the next rename too.
    """
    if not _unsupported_param(exc, current):
        return None
    match = _PARAM_HINT_RE.search(str(exc))
    if match and match.group(1) != current:
        return match.group(1)
    return None


# Kwargs the framework passes between its own layers, which must never be
# forwarded into an HTTP request body. `cache_ttl` is the live example: it is
# an Anthropic prompt-cache concept that LLM._provider_kwargs injects into
# every call, and an OpenAI-compatible endpoint 400s on it.
_NON_API_KWARGS = frozenset({"cache_ttl", "thinking", "schema", "schema_name"})


def _api_kwargs(kwargs):
    """Caller kwargs minus the framework-internal ones."""
    return {k: v for k, v in kwargs.items() if k not in _NON_API_KWARGS}


def openai_messages(prompt, system_prompt=None, images=None):
    """The messages list exactly as the OpenAI-compatible sync paths build it.

    Shared by call_openai, call_deepseek (text-only) and call_local, and by
    the batch path — one constructor, so the transports cannot drift.
    """
    if images:
        content = []
        for img in images:
            data, mime = _load_image_bytes(img)
            b64 = base64.b64encode(data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _openai_attempt_kwargs(param, send_temperature, temperature, max_tokens,
                           extra):
    """One attempt's request kwargs — the single constructor _chat_completion
    builds every attempt from and openai_request_body wraps for batch."""
    kwargs = dict(extra)
    kwargs[param] = max_tokens
    if send_temperature and temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def openai_request_body(provider, model, messages, temperature=0.7,
                        max_tokens=4096, **extra):
    """A complete chat-completions body for one BATCH request line.

    Consults the same repair memos the sync path learns from — which is
    why the batch path probes one item sync first: a cold memo builds
    max_tokens for a gpt-5 model and 50,000 requests fail at once where
    the sync loop would have healed the first 400.
    """
    memo = (provider, _strip_prefix(model))
    body = _openai_attempt_kwargs(
        _TOKEN_PARAM.get(memo, "max_tokens"),
        memo not in _NO_TEMPERATURE,
        temperature, max_tokens, extra,
    )
    body["model"] = _strip_prefix(model)
    body["messages"] = messages
    return body


def _chat_completion(client, provider, model, messages, temperature, max_tokens,
                     usage_sink=None, dropped_hint=(), usage_filter=None,
                     **extra):
    """POST an OpenAI-compatible chat completion, repairing rejected params.

    Shared by the OpenAI, DeepSeek, and local-endpoint providers so a rename
    on one is fixed for all three. Repairs are memoized per (provider, model),
    so the 400 is paid once per process rather than once per call.

    dropped_hint: parameters the CALLER already determined cannot take effect,
        to be merged into this call's audit record. Needed because the repair
        loop below can only detect a parameter the API rejects loudly; a
        parameter the API accepts and ignores leaves no trace in the response.
    usage_filter: optional callable (usage_dict, response) -> usage_dict, run
        before the sink. This is how a caller audits the RESPONSE and amends
        the machine-readable record — a post-hoc audit that only logs leaves
        the usage record asserting the opposite of what the warning says.
    """
    memo = (provider, model)
    param = _TOKEN_PARAM.get(memo, "max_tokens")
    send_temperature = memo not in _NO_TEMPERATURE
    dropped = list(dropped_hint)
    if not send_temperature and temperature is not None:
        # Known-rejected from an earlier call. Re-report EVERY call: under
        # LITMOD_STRICT_PARAMS the first call raising and the next thousand
        # sailing through is the exact silence the flag exists to forbid.
        _report_dropped_param(provider, model, "temperature",
                              temperature, _WARNED_NO_TEMPERATURE)
        if "temperature" not in dropped:
            dropped.append("temperature")

    # At most one repair per rejected parameter, then give up and raise. The
    # tried-set breaks repair oscillation (A -> B -> A from a proxy echoing
    # request bodies): a name suggested twice is a wrong suggestion.
    tried = {param}
    last_exc = None
    for _ in range(3):
        kwargs = _openai_attempt_kwargs(param, send_temperature, temperature,
                                        max_tokens, extra)
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **kwargs,
            )
        except Exception as e:
            last_exc = e
            healed = _healed_token_param(e, param)
            if healed is not None and healed not in tried:
                log.warning(
                    "%s: model %r rejects %r; retrying with %r and remembering "
                    "it for this process", provider, model, param, healed,
                )
                tried.add(healed)
                param = healed
                continue
            if send_temperature and _unsupported_param(e, "temperature"):
                _NO_TEMPERATURE.add(memo)
                send_temperature = False
                if "temperature" not in dropped:
                    dropped.append("temperature")
                # Raises under LITMOD_STRICT_PARAMS rather than proceeding
                # with a pin the caller believes is in effect.
                _report_dropped_param(provider, model, "temperature",
                                      temperature, _WARNED_NO_TEMPERATURE)
                continue
            raise
        _TOKEN_PARAM[memo] = param
        _log_resolved_model(provider, model, getattr(response, "model", None))
        if usage_sink is not None:
            u = _usage_openai_compat(response)
            if dropped:
                u["dropped_params"] = tuple(dropped)
            if usage_filter is not None:
                u = usage_filter(u, response) or u
            usage_sink(u)
        return response

    # Chain the API's own error: it is the one artifact this whole design
    # rests on, and a bare "could not build a request" strips exactly the
    # diagnostic a caller needs to falsify the repair loop's inference.
    raise RuntimeError(
        f"{provider}: could not build an acceptable request for model "
        f"{model!r} after trying output-length parameter names {sorted(tried)}. "
        f"Last API error: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Usage reporting
#
# Providers return a bare string, so token counts would otherwise be
# unobservable. A caller needs them to *prove* a prompt-cache hit rather than
# assume one: a batch that silently stopped caching produces byte-identical
# output at ~10x the input cost. Each provider normalises its own usage object
# into the same four keys and hands it to `usage_sink`.
# ---------------------------------------------------------------------------

def _usage(input_tokens=0, output_tokens=0, cache_read_tokens=0,
           cache_write_tokens=0, reasoning_tokens=0, reasoning_reported=False,
           reasoning_observed=False, response_model=None):
    """Normalised token counts.

    `reasoning_tokens` is a BREAKDOWN of output_tokens, not an addition to it:
    providers bill chain-of-thought as output, so subtracting it here would
    understate the bill. It is tracked separately because on an extraction
    call it measures spend on text the caller structurally cannot read — we
    parse the JSON and discard everything else — which is the difference
    between an expensive model and a wasted one.
    """
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cache_read_tokens": int(cache_read_tokens or 0),
        "cache_write_tokens": int(cache_write_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        # Whether the provider reported the token-count field AT ALL on this
        # call. Diagnostic — distinguishes one provider's silence from
        # another's zero when comparing across backends.
        "reasoning_reported": bool(reasoning_reported),
        # Whether this call showed ANY evidence of reasoning: a nonzero token
        # count, a reasoning_content body (DeepSeek), a thinking block in the
        # response content (Anthropic), or thought tokens (Google). This is
        # the gate's field, and it is deliberately multi-signal — one field
        # going missing must not be able to silence a publication gate.
        "reasoning_observed": bool(reasoning_observed) or bool(reasoning_tokens),
        # The model id the SERVER says produced this response, as opposed to
        # the one we asked for. Hosted APIs alias retired names onto current
        # checkpoints, so the two differ routinely and only this one describes
        # what actually answered. _log_resolved_model has warned about the
        # mismatch for a while; this carries it as data so an artifact can
        # record provenance instead of a log line nobody kept.
        #
        # It is a SELF-REPORT, not a verification: it cannot detect a provider
        # that serves one checkpoint and names another. "The id the server
        # reported having served" is the strongest honest reading.
        "response_model": response_model,
    }


class DroppedParameterError(RuntimeError):
    """A request parameter the caller set could not be sent to this model."""


def _report_dropped_param(provider, model, param, value, warned,
                          reason="the API rejects it on this family"):
    """Record that `param` was dropped, loudly, and raise under strict mode.

    A parameter that appears to apply and does not is the worst failure shape
    this package has: 'administered at temperature 0' reads as true, nothing in
    the output contradicts it, and the claim can be published. A warning is
    only read by someone already suspicious, so LITMOD_STRICT_PARAMS=1 turns
    this into an error. It is not the default because every Task sets a
    temperature, so erroring by default would stop the newest Anthropic models
    from running at all — see UsageTracker.dropped_params for the per-run audit
    record that makes the omission checkable after the fact either way.
    """
    message = (
        f"{provider}: {model!r} does not accept `{param}` ({reason}), so "
        f"{param}={value!r} was NOT applied. Sampling is not pinned for this "
        f"model — do not describe these runs as {param}-controlled."
    )
    if os.getenv("LITMOD_STRICT_PARAMS"):
        raise DroppedParameterError(
            message + " Raised because LITMOD_STRICT_PARAMS is set; pass a "
            f"model that accepts `{param}`, or stop setting it."
        )
    # Keyed by (model, param): keyed on model alone, whichever parameter was
    # reported first permanently suppressed the warning for every other —
    # on claude-cli the temperature warning silenced the max_tokens one.
    if (model, param) not in warned:
        warned.add((model, param))
        log.warning("%s", message)


def _block_type(block):
    """Content-block type across SDK objects and plain dicts."""
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _usage_anthropic(response):
    # Anthropic reports no reasoning-token split, but the content blocks are
    # themselves the receipt: a model that thought returns a thinking block.
    # Without this, the no-reasoning gate returns a clean bill on every
    # Anthropic run — including Fable, where thinking cannot be disabled.
    blocks = getattr(response, "content", None) or []
    thought = any(_block_type(b) in ("thinking", "redacted_thinking")
                  for b in blocks)
    served = getattr(response, "model", None)
    u = getattr(response, "usage", None)
    if u is None:
        # No usage object is not no provenance: the served id and the
        # thinking blocks are still on the response.
        return _usage(reasoning_observed=thought, response_model=served)
    return _usage(
        input_tokens=getattr(u, "input_tokens", 0),
        output_tokens=getattr(u, "output_tokens", 0),
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0),
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0),
        reasoning_observed=thought,
        response_model=served,
    )


def _usage_openai_compat(response):
    """Normalise an OpenAI-compatible usage object.

    Covers OpenAI (`prompt_tokens_details.cached_tokens`), DeepSeek
    (`prompt_cache_hit_tokens`), and local servers (usually neither).
    Note these report cache *reads* only — there is no write counter, so
    cache_write_tokens stays 0 and hit rate is read/(read+uncached input).
    """
    served = getattr(response, "model", None)
    # Second reasoning signal, independent of the token counter: the
    # chain-of-thought body itself. Measured live on DeepSeek both appear
    # and disappear together; the redundancy is for the day they do not.
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    reasoning_body = getattr(message, "reasoning_content", None)
    u = getattr(response, "usage", None)
    if u is None:
        # Local servers routinely omit usage; the served id and any
        # reasoning body are still on the response.
        return _usage(reasoning_observed=bool(reasoning_body),
                      response_model=served)
    cached = getattr(u, "prompt_cache_hit_tokens", None)
    if cached is None:
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
    prompt = getattr(u, "prompt_tokens", 0) or 0
    # completion_tokens is the whole billed output INCLUDING chain-of-thought.
    # Reading the split matters: a DeepSeek v4 probe billed 107 completion
    # tokens of which 97 were reasoning, so a usage line reporting only the
    # total describes 91% of its own number as answer text when it was not.
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    return _usage(
        # Report uncached input only, matching Anthropic's semantics, so the
        # four keys sum to the true prompt size on every provider.
        input_tokens=max(0, prompt - (cached or 0)),
        output_tokens=getattr(u, "completion_tokens", 0),
        cache_read_tokens=cached,
        reasoning_tokens=reasoning or 0,
        reasoning_reported=reasoning is not None,
        reasoning_observed=bool(reasoning) or bool(reasoning_body),
        response_model=served,
    )


def _usage_google(response):
    # Google spells the served id `model_version`, not `model` — a provenance
    # field that silently held None for one of four providers would be worse
    # than not having one, since nothing would mark the gap.
    served = getattr(response, "model_version", None)
    m = getattr(response, "usage_metadata", None)
    if m is None:
        return _usage(response_model=served)
    cached = getattr(m, "cached_content_token_count", 0) or 0
    prompt = getattr(m, "prompt_token_count", 0) or 0
    # Gemini thinking models report their deliberation as thoughts_token_count
    # — and unlike DeepSeek's completion_tokens, candidates_token_count does
    # NOT include it. Measured live (2026-08-04, doctor probe): 14 candidate
    # tokens beside 363 thought tokens, so reporting candidates alone as
    # output_tokens understates the billed output 26x and put the reasoning
    # share at 2593%. Sum them here so output_tokens means the same thing on
    # every provider: everything billed at the output rate, of which
    # reasoning_tokens is the subset spent deliberating.
    thoughts = getattr(m, "thoughts_token_count", None)
    return _usage(
        input_tokens=max(0, prompt - cached),
        output_tokens=(getattr(m, "candidates_token_count", 0) or 0)
        + (thoughts or 0),
        cache_read_tokens=cached,
        reasoning_tokens=thoughts or 0,
        reasoning_reported=thoughts is not None,
        response_model=served,
    )


# ---------------------------------------------------------------------------
# Prompt-cache minimums
#
# Anthropic will not cache a prefix below a per-model token floor, and it
# declines SILENTLY: no error, cache_creation_input_tokens 0, every call at
# full input price. The floor is not monotonic across generations, so a
# prompt that caches on Sonnet may not on Haiku.
#
# Measured 2026-07-30 against claude-haiku-4-5: a 12,558-char system block of
# our own instrument text (3,222 tokens) cached 0 on two consecutive identical
# calls, while a 19,189-char block (5,013 tokens) wrote once and read once.
#
# Do NOT estimate tokens from character count to decide this. Measured density
# ranges from 2.7 to 3.9 chars/token across real instruments — a peer project's
# 12,558-char instrument counts 4,711 tokens where ours counts 3,222, so a
# chars/4 rule calls one of them wrong. Cache behaviour is detected from
# observed usage instead (see llm.UsageTracker); the check below only fires
# when a prompt is short enough to be below the floor at ANY plausible density.
# ---------------------------------------------------------------------------

def _family_match(model_lower, tag):
    """True if `tag` names this model's family.

    A version tag is complete only when not followed by another digit:
    'sonnet-5' must not match a future 'sonnet-50', or that model silently
    inherits every constant measured for this one.
    """
    i = model_lower.find(tag)
    while i != -1:
        j = i + len(tag)
        if j >= len(model_lower) or not model_lower[j].isdigit():
            return True
        i = model_lower.find(tag, i + 1)
    return False


_CACHE_MIN_TOKENS = (
    ("haiku-4-5", 4096),
    ("opus-4-6", 4096),
    ("opus-4-5", 4096),
    ("opus-4-7", 2048),
    ("opus-5", 512),
    ("fable-5", 512),
    ("mythos-5", 512),
)
# The 1024 floor was MEASURED on these families; it is not a fallback for
# models nobody has measured (cache_minimum_tokens returns None there).
_CACHE_MIN_DEFAULT_FAMILIES = ("sonnet-5", "sonnet-4-6", "sonnet-4-5",
                               "opus-4-8", "opus-4-1")
_CACHE_MIN_DEFAULT = 1024

# Gemini implicit-caching minimums. 3.x measured in the field: a
# 3,906-token prefix never cached across 14,520 calls (~$76 of full-price
# input); 4,096 is the documented minimum it sat just under. 2.5-era
# floors are unmeasured here — None, not a guess.
_GOOGLE_CACHE_MIN_TOKENS = (
    ("gemini-3", 4096),
)


# Densest chars/token seen on real instrument text. Used only to bound an
# estimate from above, never to produce one.
_CHARS_PER_TOKEN_DENSE = 2.5


def cache_minimum_tokens(model):
    """Minimum cacheable prompt-prefix length, in tokens, for `model`.

    Below this the Anthropic API silently declines to cache. For an exact
    check, count the real tokens:

        client.messages.count_tokens(model=..., system=task.instrument_text(),
                                     messages=[{"role": "user", "content": "x"}])

    then compare against this value. Do not estimate from character count.

    Returns None for a model with no measured floor — including every
    non-Anthropic model, where the concept does not apply. Unknown must stay
    unknown: the whole premise of this table is that the floor is
    non-monotonic across generations, so a default here is a guess dressed
    as a measurement, and the failure it invites (a future model inheriting
    a floor four times lower than its real one) is silent by construction.
    Matching is by family with a version boundary and longest-tag-wins, so
    'opus-4-5' beats 'opus-5' wherever it sits in the table and neither
    matches a future 'opus-50'.
    """
    m = model.lower()
    if _routes_to_google(m):
        # Gemini implicit caching has its own floor, and a miss is silent
        # in exactly the Anthropic way: a 3,906-token instrument on
        # gemini-3.6-flash ran 14,520 times at full input price, ~130
        # tokens under the 4,096 minimum, and nothing said so until the
        # invoice did (reported by the malign-logits seat, 2026-08-04).
        for tag, minimum in _GOOGLE_CACHE_MIN_TOKENS:
            if _family_match(m, tag):
                return minimum
        return None
    if "claude" not in m and not m.startswith("anthropic/"):
        return None
    matches = [(tag, minimum) for tag, minimum in _CACHE_MIN_TOKENS
               if _family_match(m, tag)]
    if matches:
        return max(matches, key=lambda pair: len(pair[0]))[1]
    for tag in _CACHE_MIN_DEFAULT_FAMILIES:
        if _family_match(m, tag):
            return _CACHE_MIN_DEFAULT
    return None


_WARNED_CACHE_FLOOR = set()


def _warn_if_below_cache_floor(model, system_prompt):
    """Warn once when a system block cannot possibly reach the cache floor.

    Deliberately conservative: it fires only when the prompt is below the
    floor even at the densest tokenisation we have measured, so it never
    tells a caller their working cache is broken. Prompts in the ambiguous
    band are caught after the fact by the no-caching-observed check in
    llm.UsageTracker, which reads real usage rather than guessing.
    """
    # Short prompts are below any plausible floor, but the send path still
    # marks them cacheable (the API just declines) — the early return here is
    # only about warning noise, not about whether caching was requested.
    if len(system_prompt) < 2000:
        return
    floor = cache_minimum_tokens(model)
    if floor is None:
        # No measured floor for this model. Guessing one would either cry
        # wolf or reassure falsely; the observed-usage check in
        # llm.UsageTracker.cache_warning catches the real outcome either way.
        return
    upper_bound = len(system_prompt) / _CHARS_PER_TOKEN_DENSE
    if upper_bound >= floor:
        return
    key = (model, floor)
    if key in _WARNED_CACHE_FLOOR:
        return
    _WARNED_CACHE_FLOOR.add(key)
    log.warning(
        "anthropic: system prompt is at most ~%d tokens (%d chars) but %r will "
        "not cache a prefix below %d tokens — it declines silently, so every "
        "call pays full input price. Note the inversion this creates: a cached "
        "prefix bills at ~0.1x, so any prompt over ~%d tokens is CHEAPER "
        "padded up past the floor than left short of it, and a shorter, "
        "better-written instrument can cost several times more per call than a "
        "longer one on this model. Options: pad past the floor (more few-shot "
        "examples pay for themselves twice), use a model with a lower floor "
        "(Sonnet/Opus 4.8 = 1024, Opus 5 = 512), or accept the uncached cost.",
        int(upper_bound), len(system_prompt), model, floor, int(floor * 0.1),
    )


# Routing predicates, shared between route_provider and the cache-key
# fingerprint helpers so the two cannot drift: a model that routes to a
# provider must be fingerprinted by that provider's rules.

def _routes_to_local(model_lower):
    return model_lower.startswith(("local/", "ollama/", "vllm/", "lmstudio/"))


def _routes_to_claude_cli(model_lower):
    return model_lower.startswith("claude-cli/")


def _routes_to_anthropic(model_lower):
    return (not _routes_to_claude_cli(model_lower)
            and not _routes_to_local(model_lower)
            and ("claude" in model_lower
                 or model_lower.startswith("anthropic/")))


def _routes_to_deepseek(model_lower):
    return (not _routes_to_local(model_lower)
            and (model_lower.startswith("deepseek/")
                 or model_lower in _DEEPSEEK_API_MODELS))


def _routes_to_google(model_lower):
    return (not _routes_to_local(model_lower)
            and ("gemini" in model_lower
                 or model_lower.startswith("google/")))


def route_provider(model):
    """Return the appropriate provider function for a model string."""
    model_lower = model.lower()
    if _routes_to_local(model_lower):
        return call_local
    if _routes_to_claude_cli(model_lower):
        return call_claude_cli
    if _routes_to_anthropic(model_lower):
        return call_anthropic
    elif _routes_to_deepseek(model_lower):
        return call_deepseek
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or model_lower.startswith("openai/"):
        return call_openai
    elif _routes_to_google(model_lower):
        return call_google
    else:
        raise ValueError(
            f"Cannot determine provider for model '{model}'. "
            f"Model name should contain 'claude', 'gpt', or 'gemini', "
            f"or use a prefix like 'anthropic/', 'openai/', 'google/', 'deepseek/', 'claude-cli/', or 'local/'."
        )


def _strip_prefix(model):
    """Remove provider prefix like 'anthropic/' or 'openai/' from model name."""
    for prefix in ("anthropic/", "openai/", "google/", "deepseek/",
                   "claude-cli/", "local/", "ollama/", "vllm/", "lmstudio/"):
        if model.lower().startswith(prefix):
            return model[len(prefix):]
    return model


def _load_image_bytes(image):
    """Convert an image (path, bytes, or PIL Image) to (bytes, mime_type)."""
    if isinstance(image, str):
        # File path
        with open(image, "rb") as f:
            data = f.read()
        ext = os.path.splitext(image)[1].lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        }
        return data, mime_map.get(ext, "image/png")
    elif isinstance(image, bytes):
        return image, "image/png"
    else:
        # PIL Image
        buf = io.BytesIO()
        fmt = getattr(image, "format", "PNG") or "PNG"
        image.save(buf, format=fmt)
        mime = f"image/{fmt.lower()}"
        return buf.getvalue(), mime


# Model families where the API rejects sampling params (temperature/top_p).
# Substring check against the stripped model id; extend as new families ship.
# 'opus-5' was missing here and would have 400'd the same way gpt-5 did on
# max_tokens — the whole class of bug is a constant that knows only the model
# generation it was written against.
# Probed 2026-08-04 to rule out a suspected measurement artifact: these
# were first measured rejecting `temperature` while thinking was on by
# default, and thinking mode requires temperature=1 — so the rejection
# could have been thinking's, not the model's. It is not: sonnet-5 and
# opus-5 both 400 with an explicit thinking={"type": "disabled"} —
# "`temperature` is deprecated for this model." The constant is a fact
# about the model family, not about the thinking default.
_NO_TEMPERATURE_MODELS = ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5",
                          "fable", "mythos")

_WARNED_NO_TEMPERATURE = set()


def _supports_temperature(model):
    model_lower = model.lower()
    return not any(_family_match(model_lower, tag)
                   for tag in _NO_TEMPERATURE_MODELS)


# ---------------------------------------------------------------------------
# Extended thinking on extraction calls
#
# Models where thinking runs when the `thinking` parameter is omitted. On
# these, a structured-extraction call pays for a deliberation the caller
# structurally cannot read — extract parses the output as JSON and discards
# everything else — so the old default was wrong, not merely expensive.
#
# Measured 2026-07-30, claude-sonnet-5, PassageFormTask instrument (6,863
# tokens), identical prompt:
#
#     thinking omitted   1,124 and 1,134 output tokens   blocks [thinking, text]
#     thinking disabled  288, 301, 342, 347, 375         blocks [text]
#
# 3.3x on the output line at the means, and output dominates once the prompt
# cache is warm. Re-verify rather than trust this:
#
#     from largeliterarymodels.llm import _build_extract_prompt
#     system, _ = _build_extract_prompt("", task.schema,
#                                       system_prompt=task.system_prompt,
#                                       examples=task.examples)
#     # then one client.messages.create with thinking={"type": "disabled"}
#     # and one without, and compare usage.output_tokens.
#
# Fable/Mythos cannot be turned off: an explicit {"type": "disabled"} is a
# 400 there, so those pay for thinking unavoidably.
# ---------------------------------------------------------------------------

_THINKING_ON_BY_DEFAULT = ("sonnet-5", "opus-5", "fable", "mythos")
_THINKING_CANNOT_DISABLE = ("fable", "mythos")

_WARNED_THINKING = set()


def thinking_default(model, _warn=True):
    """The `thinking` value to send when a caller expresses no preference.

    Returns {"type": "disabled"} for models that would otherwise think, None
    to omit the parameter (older families, where omitting already means off).
    _warn=False suppresses the cannot-disable cost warning — cache-key
    fingerprinting calls this for items that may never reach the API.
    """
    m = model.lower()
    if not any(_family_match(m, tag) for tag in _THINKING_ON_BY_DEFAULT):
        return None
    if any(_family_match(m, tag) for tag in _THINKING_CANNOT_DISABLE):
        if _warn and model not in _WARNED_THINKING:
            _WARNED_THINKING.add(model)
            log.warning(
                "anthropic: %r has thinking permanently on (an explicit "
                "disable is rejected), so extraction calls bill thinking as "
                "output tokens. Measured ~3.3x the output of a "
                "thinking-disabled call — use sonnet-5/opus-5 if that matters.",
                model,
            )
        return None
    return {"type": "disabled"}


# ---------------------------------------------------------------------------
# Extended thinking on the OpenAI-compatible providers (DeepSeek)
#
# DeepSeek v4 ships thinking ENABLED BY DEFAULT at effort "high", on BOTH
# deepseek-v4-flash and deepseek-v4-pro — the docs state it without a
# per-model exception. Measured on a live flash call (2026-08-04, two-field
# probe schema): completion_tokens=107 of which reasoning_tokens=97. 91% of
# the billed output was chain-of-thought. On a full instrument the same seat
# measured ~1,700 output tokens/call.
#
# Thinking costs this package twice, and the second cost is the serious one:
#
#   1. Output tokens, which dominate once the prompt cache is warm. An extract
#      call parses the JSON and discards everything else, so the deliberation
#      is unreadable by construction — the old default was wrong, not merely
#      expensive.
#
#   2. SAMPLING CONTROL. Per the docs: "Thinking mode does not support the
#      temperature, top_p, presence_penalty, or frequency_penalty parameters.
#      Please note that, for compatibility with existing software, setting
#      these parameters will not trigger an error but will also have no
#      effect." So temperature=0 was accepted, silently discarded, and — with
#      no 400 to catch — never reached _NO_TEMPERATURE or the dropped_params
#      audit record. Every DeepSeek annotation taken in thinking mode is
#      uncontrolled sampling that reads as temperature-pinned. That is the
#      exact failure shape _report_dropped_param exists to prevent, arriving
#      through the one door that machinery could not watch.
#
# Disabling repairs both. OpenAI-format spelling, via extra_body:
#     extra_body={"thinking": {"type": "disabled"}}
# https://api-docs.deepseek.com/guides/thinking_mode/
#
# Not extended to call_openai: the gpt-5 tier reasons too, but its knob is a
# top-level reasoning_effort with a different vocabulary, and nobody has
# measured it here. reasoning_tokens is captured for every OpenAI-compatible
# provider regardless, so that gap shows up as a number rather than silence.
# ---------------------------------------------------------------------------

_THINKING_DISABLED_OPENAI = {"type": "disabled"}

_WARNED_THINKING_NOT_DISABLED = set()


def deepseek_thinking_default(model):
    """The `thinking` value to send when a caller expresses no preference.

    Disabled for EVERY model on the DeepSeek route, known ids and future
    ones alike. An earlier version keyed this on a tuple of known ids, which
    meant an unrecognised id got no disable — and, worse, the temperature
    handling then treated it as a thinking-mode run and withheld
    `temperature` from the request entirely. Deciding a model's behaviour
    from its name is this package's recurring bug class; send the disable
    and let the response-side audit catch a model that ignores it.
    """
    return dict(_THINKING_DISABLED_OPENAI)


def _normalize_thinking_anthropic(thinking):
    """Canonicalise a caller's thinking argument for the Anthropic API.

    The cross-provider "off" spellings must mean off here too — an earlier
    version forwarded thinking=False verbatim onto the wire (a 400) while
    the same spelling disabled thinking on DeepSeek and Google. "enabled"
    without a budget raises rather than inventing one: Anthropic's enable
    requires budget_tokens, and a silently chosen default budget would be a
    sampling-policy decision smuggled in as a spelling convenience.
    """
    if thinking is None:
        return None
    if thinking is False or (isinstance(thinking, str)
                             and thinking.lower() == "disabled"):
        return {"type": "disabled"}
    if thinking is True or (isinstance(thinking, str)
                            and thinking.lower() == "enabled"):
        raise ValueError(
            "anthropic thinking must be an explicit dict with budget_tokens "
            "(e.g. {'type': 'enabled', 'budget_tokens': 4096}) — there is no "
            "defensible default budget to invent."
        )
    if isinstance(thinking, dict) and "type" in thinking:
        return dict(thinking)
    raise ValueError(
        f"thinking={thinking!r} is not a recognised value for an Anthropic "
        f"model. Use 'auto', None, False/'disabled', or a dict with a "
        f"'type' key."
    )


def _normalize_thinking(thinking):
    """Canonicalise a caller's thinking argument for the OpenAI-format API.

    The natural spellings of "off" must all mean off: an earlier version
    forwarded `thinking=False` verbatim (an undocumented value on the wire)
    while ALSO treating it as thinking-on for the temperature audit — wrong
    in both directions at once. None means "send nothing".
    """
    if thinking is None:
        return None
    if thinking is False:
        return dict(_THINKING_DISABLED_OPENAI)
    if thinking is True:
        return {"type": "enabled"}
    if isinstance(thinking, str) and thinking.lower() in ("disabled", "enabled"):
        return {"type": thinking.lower()}
    if isinstance(thinking, dict) and "type" in thinking:
        return dict(thinking)
    raise ValueError(
        f"thinking={thinking!r} is not a recognised value. Use 'auto', None, "
        f"True/False, 'enabled'/'disabled', or a dict with a 'type' key."
    )


def _audit_thinking_disabled(provider, model, usage, response=None,
                             requested_temperature=None):
    """Check that a requested thinking-disable actually took effect, and
    make the usage record tell the truth when it did not.

    A 400 is the easy case. The dangerous case is a provider that accepts
    {"type": "disabled"} and reasons anyway — the request looks right, the
    cost is unchanged, and the run's temperature claim is still false. The
    only honest check is the receipt in the response: the reasoning-token
    count and the reasoning_content body, either of which counts (one field
    going missing must not silence the check).

    Returns the usage dict, amended: when the disable was ignored,
    `temperature` joins dropped_params — DeepSeek's docs say thinking mode
    accepts and ignores sampling params — so the machine-readable record
    agrees with the warning instead of contradicting it. Runs as a
    usage_filter inside _chat_completion, BEFORE the sink sees the dict; an
    earlier version audited after the sink had flushed, which meant the one
    failure it existed to catch produced a log line and a clean audit record.
    Raises DroppedParameterError under LITMOD_STRICT_PARAMS.
    """
    ignored = usage.get("reasoning_observed") or usage.get("reasoning_tokens", 0)
    if not ignored:
        return usage
    if requested_temperature is not None:
        dropped = tuple(usage.get("dropped_params", ()))
        if "temperature" not in dropped:
            usage["dropped_params"] = dropped + ("temperature",)
        _report_dropped_param(
            provider, model, "temperature", requested_temperature,
            _WARNED_NO_TEMPERATURE,
            reason="the requested thinking disable was ignored, and thinking "
                   "mode accepts sampling params without applying them",
        )
    if model not in _WARNED_THINKING_NOT_DISABLED:
        _WARNED_THINKING_NOT_DISABLED.add(model)
        tokens = usage.get("reasoning_tokens", 0)
        if tokens:
            magnitude = f"{tokens} reasoning tokens billed as output"
        else:
            body = None
            try:
                body = getattr(response.choices[0].message,
                               "reasoning_content", None)
            except (AttributeError, IndexError, TypeError):
                pass
            # Characters, not tokens — the counter was absent, and printing
            # a character count in token units overstates ~4x in the exact
            # sentence a methods note would quote.
            magnitude = (f"a reasoning_content body of ~{len(body)} "
                         f"characters (no token count reported)"
                         if body else "reasoning evidence in the response")
        log.warning(
            "%s: asked %r for thinking={'type': 'disabled'} and it reasoned "
            "anyway (%s). The disable is not taking effect on this model — "
            "treat these runs as thinking-mode runs; `temperature` was "
            "recorded as dropped. Re-check the spelling against "
            "https://api-docs.deepseek.com/guides/thinking_mode/",
            provider, model, magnitude,
        )
    return usage


# Gemini families where thinking_budget=0 is a 400: "Budget 0 is invalid.
# This model only works in thinking mode." (probed live 2026-08-04 on
# 2.5-pro AND 3.1-pro-preview; 2.5-flash accepts 0 and emits no thoughts).
# Static list, not a dynamic heal, on purpose: the cache-key fingerprint
# must be decided before the call, and healing a rejected disable into a
# thinking-on call would store thinking-on output under a thinking-off key
# — the exact poisoning the fingerprint exists to prevent. A future model
# joining this family fails LOUDLY with the API's own message plus a
# pointer here; loud beats a silently wrong provenance record.
_GOOGLE_THINKING_CANNOT_DISABLE = ("gemini-2.5-pro", "gemini-3.1-pro")

_WARNED_GOOGLE_THINKING = set()


_GOOGLE_THINKING_LEVELS = ("minimal", "low", "medium", "high")


def _is_gemini_3x(model):
    """Gemini 3.x uses thinking_level; thinking_budget is deprecated there
    and a zero budget is a 400. Version-boundary matched: 'gemini-3' must
    not claim a future 'gemini-30'."""
    return _family_match(_strip_prefix(model).lower(), "gemini-3")


def google_thinking_setting(model, thinking="auto", _warn=True):
    """The ThinkingConfig field to send for a Gemini call, or None.

    Returns ("thinking_budget", int) for the 2.5 generation,
    ("thinking_level", str) for 3.x, or None to send nothing. The two
    generations take DIFFERENT parameters — probed live 2026-08-04:
    thinking_budget=0 on gemini-3.6-flash is a generic INVALID_ARGUMENT
    (budget survives only as a deprecated nonzero back-compat field), and
    per Google's docs 3.x cannot fully disable thinking; thinking_level
    "minimal" is the documented off-equivalent ("matches the 'no thinking'
    setting for most queries"), measured at zero reported thoughts on our
    probes where the API default thought 370 tokens. When thoughts do
    occur under minimal — documented as possible on complex items — the
    usage receipts (reasoning_tokens, the no_reasoning_observed gate)
    carry it; that is a documented behaviour, not a broken disable, so it
    is not warned about the way an ignored budget-0 is.

    "auto": 2.5 -> budget 0; 3.x -> level "minimal"; cannot-disable
    families -> nothing sent plus a once-per-model cost warning (the Fable
    arrangement). Explicit spellings: a level string on 3.x; an int budget
    on 2.5 (rejected on 3.x — deprecated there, and a silently-degraded
    parameter is worse than an error); the cross-provider off spellings
    (False/'disabled'/{"type": "disabled"}) map to the generation's
    off-equivalent; True/'enabled'/None take the API default.
    _warn=False suppresses the cost warning for key-fingerprinting calls.
    """
    is_3x = _is_gemini_3x(model)
    off = ("thinking_level", "minimal") if is_3x else ("thinking_budget", 0)
    if thinking == "auto":
        m = _strip_prefix(model).lower()
        if any(_family_match(m, tag)
               for tag in _GOOGLE_THINKING_CANNOT_DISABLE):
            if _warn and model not in _WARNED_GOOGLE_THINKING:
                _WARNED_GOOGLE_THINKING.add(model)
                log.warning(
                    "google: %r only works in thinking mode (its minimum "
                    "thinking level/budget cannot express 'off'), so "
                    "extraction calls bill thoughts as output — measured "
                    "~400 thought tokens per two-field probe. Use "
                    "gemini-2.5-flash or a 3.x flash tier if that matters.",
                    model,
                )
            return None
        return off
    if thinking is None:
        return None
    if isinstance(thinking, bool):
        return None if thinking else off
    if isinstance(thinking, int):
        if is_3x:
            raise ValueError(
                f"thinking={thinking!r}: Gemini 3.x takes thinking_level "
                f"('minimal'/'low'/'medium'/'high'), not a token budget — "
                f"the budget survives only as a deprecated field with "
                f"documented 'unexpected performance'. Pass a level."
            )
        return ("thinking_budget", thinking)
    if isinstance(thinking, str) and thinking.lower() in _GOOGLE_THINKING_LEVELS:
        if not is_3x:
            raise ValueError(
                f"thinking={thinking!r}: the Gemini 2.5 generation takes an "
                f"int thinking_budget, not a level string."
            )
        return ("thinking_level", thinking.lower())
    if isinstance(thinking, str) and thinking.lower() in ("disabled", "enabled"):
        return off if thinking.lower() == "disabled" else None
    if isinstance(thinking, dict) and "type" in thinking:
        return off if thinking.get("type") == "disabled" else None
    raise ValueError(
        f"thinking={thinking!r} is not a recognised value for a Gemini "
        f"model. Use 'auto', None, True/False, 'enabled'/'disabled', a "
        f"level string (3.x), or an int thinking_budget (2.5)."
    )


def thinking_fingerprint(model, thinking="auto"):
    """The thinking state a call with these arguments will request, for
    cache-keying — or None when no thinking parameter would be sent.

    The cache key must distinguish thinking-on from thinking-off output:
    they are different text, differently sampled (DeepSeek ignores
    `temperature` in thinking mode), and a rerun without force= would
    otherwise hand back one as the other with nothing in the request or the
    response saying so. None — the pre-thinking families, and providers with
    no thinking parameter — keeps those cache keys byte-stable, so the vast
    majority of existing annotations stay reachable; a non-None value marks
    exactly the calls whose thinking state is part of their identity.
    """
    m = model.lower()
    if _routes_to_google(m):
        setting = google_thinking_setting(model, thinking, _warn=False)
        if setting is None:
            return None
        param, value = setting
        if param == "thinking_budget":
            return "disabled" if value == 0 else f"budget:{value}"
        # thinking_level: never "disabled" — 3.x has no off state to claim,
        # and a key must not assert one the model cannot deliver.
        return f"level:{value}"
    resolved = None
    if _routes_to_anthropic(m):
        resolved = (thinking_default(model, _warn=False) if thinking == "auto"
                    else _normalize_thinking_anthropic(thinking))
    elif _routes_to_deepseek(m):
        resolved = (deepseek_thinking_default(model) if thinking == "auto"
                    else _normalize_thinking(thinking))
    if resolved is None:
        return None
    if isinstance(resolved, dict):
        kind = resolved.get("type", str(sorted(resolved.items())))
        # budget_tokens is part of the output's identity: a 1,024-token and
        # a 64,000-token deliberation are different administrations, and
        # collapsing them to "enabled" served one as a cache hit for the
        # other with nothing recording the mismatch.
        budget = resolved.get("budget_tokens")
        return f"{kind}:budget:{budget}" if budget is not None else kind
    return str(resolved)


def effective_temperature(model, temperature, thinking="auto"):
    """The temperature that will actually govern sampling, for cache keys.

    None when the model is statically known to reject it (the Anthropic
    families above), to ignore it (DeepSeek in thinking mode, per their
    docs), or to discard it (the claude-cli path). The stash key is the
    durable artifact — a key recording temperature: 0.0 on a model that
    never applied it carries a false methods claim into every later read.
    Static knowledge only: rejections discovered dynamically mid-run must
    not change keys between one call and the next.
    """
    if temperature is None:
        return None
    m = model.lower()
    if _routes_to_claude_cli(m):
        return None
    if _routes_to_anthropic(m) and not _supports_temperature(model):
        return None
    if _routes_to_deepseek(m) and thinking_fingerprint(model, thinking) != "disabled":
        # Anything short of an explicit disable leaves DeepSeek thinking —
        # including thinking=None (send nothing, take the default, which
        # reasons) — and thinking mode ignores temperature. `!= "disabled"`,
        # not `== "enabled"`: the None case is the one a key would otherwise
        # record a temperature for.
        return None
    return temperature


def _response_text(content, model, stop_reason=None):
    """Concatenate every text block from an Anthropic response, in order.

    Never index content[0]: every current reasoning model (Sonnet 5, Opus 5,
    Fable 5 — thinking is on by default) puts a ThinkingBlock first, and
    content[0].text raises AttributeError on every single call. Tool use and
    server-tool results also occupy leading positions.

    ALL text blocks, not the first: interleaved thinking can emit
    [thinking, text, thinking, text], and returning only the first block
    silently truncates the JSON — which then surfaces downstream as a parse
    error blamed on the model and burns a billed retry. Blocks may also be
    plain dicts (a proxy, a recorded response), so type access goes through
    _block_type rather than attribute access alone.
    """
    texts = []
    for block in content:
        if _block_type(block) == "text":
            texts.append(block["text"] if isinstance(block, dict)
                         else block.text)
    if texts:
        return "".join(texts)
    # No text at all: report what we did get, and only diagnose max_tokens
    # exhaustion when stop_reason actually says so — an error message that
    # guesses a cause it cannot check sends the caller down the wrong road.
    kinds = [_block_type(b) or type(b).__name__ for b in content]
    detail = f"(blocks: {kinds}, stop_reason: {stop_reason!r})"
    if stop_reason == "max_tokens":
        detail += (" — the response hit max_tokens before any text block; "
                   "raise max_tokens")
    raise ValueError(
        f"anthropic: response from {model!r} contained no text block {detail}"
    )


def anthropic_request_params(prompt, model="claude-sonnet-4-6",
                             system_prompt=None, temperature=0.7,
                             max_tokens=4096, images=None, cache_ttl=None,
                             thinking="auto"):
    """(api_kwargs, dropped_params) — the EXACT request the sync path sends.

    One constructor for both transports: the sync path calls this and then
    client.messages.create(**api_kwargs); the batch path calls this per
    item and submits {"custom_id": ..., "params": api_kwargs}. Two request
    builders is how sync and batch drift apart — the recurring bug class,
    in a shape a unit test cannot catch.

    Side-effects are deliberate and belong at BUILD time on both paths:
    _report_dropped_param fires (and raises under LITMOD_STRICT_PARAMS)
    when the family rejects temperature — constructing 10,000 batch
    requests with a pin that will not apply should fail before money
    moves, not after — and the cache-floor warning fires once per model.
    """
    model = _strip_prefix(model)

    # Build content blocks
    if images:
        content = []
        for img in images:
            data, mime = _load_image_bytes(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(data).decode("utf-8"),
                },
            })
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt

    api_kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    # Newer model families reject sampling params entirely; skip when unsupported.
    dropped = []
    if temperature is not None and _supports_temperature(model):
        api_kwargs["temperature"] = temperature
    elif temperature is not None:
        _report_dropped_param("anthropic", model, "temperature", temperature,
                              _WARNED_NO_TEMPERATURE)
        dropped.append("temperature")
    # Mark system (which includes few-shot examples per llm._build_extract_prompt)
    # as cacheable. Task batches reuse the same system across hundreds-to-thousands
    # of calls; the per-call user message is tiny. Caching cuts input cost ~10x on
    # cache hits. Below the model's cache threshold Anthropic silently skips.
    if system_prompt:
        cache_control = {"type": "ephemeral"}
        if cache_ttl:
            cache_control["ttl"] = cache_ttl
        api_kwargs["system"] = [{
            "type": "text",
            "text": system_prompt,
            "cache_control": cache_control,
        }]
        _warn_if_below_cache_floor(model, system_prompt)

    thinking_param = (thinking_default(model) if thinking == "auto"
                      else _normalize_thinking_anthropic(thinking))
    if thinking_param is not None:
        api_kwargs["thinking"] = thinking_param
    return api_kwargs, tuple(dropped)


def call_anthropic(prompt, model="claude-sonnet-4-6", system_prompt=None,
                   temperature=0.7, max_tokens=4096, images=None,
                   timeout=None, cache_ttl=None, usage_sink=None,
                   thinking="auto", **kwargs):
    """Call Anthropic's Claude API directly.

    Args:
        thinking: "auto" (default) disables extended thinking on families
            where it would otherwise run — see thinking_default. Pass None to
            send nothing and take the API default, or an explicit dict such as
            {"type": "adaptive"} to request it.
        cache_ttl: Prompt-cache lifetime for the system block: None for the
            5-minute default, or "1h". Caveat measured in the field: when a
            5-minute entry for the same prefix is already warm, a "1h" request
            reads it rather than establishing a 1-hour entry — so switching
            TTL mid-batch does not extend the existing entry. The 1-hour write
            also costs 2x base input vs 1.25x for 5 minutes, so it only pays
            back across three or more reads.
        usage_sink: Optional callable receiving a normalised usage dict.
    """
    from anthropic import Anthropic

    api_key = _get_key("ANTHROPIC_API_KEY")
    client = _cached_client(
        ("anthropic", api_key, timeout),
        lambda: Anthropic(api_key=api_key) if timeout is None
        else Anthropic(api_key=api_key, timeout=timeout),
    )
    api_kwargs, dropped = anthropic_request_params(
        prompt, model=model, system_prompt=system_prompt,
        temperature=temperature, max_tokens=max_tokens, images=images,
        cache_ttl=cache_ttl, thinking=thinking,
    )
    model = api_kwargs["model"]
    dropped = list(dropped)

    response = client.messages.create(**api_kwargs)
    # Rolling aliases (claude-sonnet-4-6) resolve to dated snapshots; log which.
    _log_resolved_model("anthropic", model, getattr(response, "model", None))
    if usage_sink is not None:
        u = _usage_anthropic(response)
        if dropped:
            u["dropped_params"] = tuple(dropped)
        usage_sink(u)
    return _response_text(response.content, model,
                          stop_reason=getattr(response, "stop_reason", None))


def call_claude_cli(prompt, model="claude-cli/opus", system_prompt=None,
                    temperature=0.7, max_tokens=4096, images=None, **kwargs):
    """DEPRECATED: call Claude by shelling out to the `claude` CLI.

    Disabled by default. Anthropic does not permit using the Claude Code CLI
    as a programmatic backend, so this raises unless the caller sets
    LITMOD_ALLOW_CLAUDE_CLI=1 to acknowledge that. Use `claude-*` (Anthropic
    API) model strings instead.

    The escape hatch exists only so an in-flight run can be finished, not as
    a supported path. MajorGenreTask still carries `claude-cli/sonnet` as its
    default model string — deliberately left in place, since `model` is part
    of the HashStash key and re-pinning it would orphan its annotations — so
    that task needs an explicit `model=` override or the env var.

    Model string after prefix selects the model:
        claude-cli/opus   → --model claude-opus-4-6
        claude-cli/sonnet → --model claude-sonnet-4-6
        claude-cli/haiku  → --model claude-haiku-4-5
        claude-cli/<full> → --model <full>  (pass-through)
    """
    import json
    import shutil
    import subprocess

    if not os.getenv("LITMOD_ALLOW_CLAUDE_CLI"):
        raise RuntimeError(
            "The claude-cli provider is disabled: Anthropic does not permit "
            "using the Claude Code CLI as a programmatic backend. Pass an "
            "Anthropic API model instead (e.g. model='claude-sonnet-4-6'). "
            "To finish an in-flight run anyway, set LITMOD_ALLOW_CLAUDE_CLI=1."
        )

    # This path builds a command line with NO sampling or length controls.
    # Accepting those parameters silently is the "administered at temperature
    # 0 reads as true" shape — the CLI never saw them. Record the drop
    # (raises under LITMOD_STRICT_PARAMS); images cannot be passed at all.
    if images:
        raise ValueError("claude-cli cannot pass images; use the API path.")
    if temperature is not None:
        _report_dropped_param(
            "claude-cli", model, "temperature", temperature,
            _WARNED_NO_TEMPERATURE,
            reason="the CLI transport has no temperature flag; the value "
                   "never reaches the model",
        )
    if max_tokens is not None:
        _report_dropped_param(
            "claude-cli", model, "max_tokens", max_tokens,
            _WARNED_NO_TEMPERATURE,
            reason="the CLI transport has no output-length flag; the value "
                   "never reaches the model",
        )

    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "Claude CLI not found. Install from https://claude.com/claude-code"
        )

    model_name = _strip_prefix(model)
    model_map = {
        "opus": "claude-opus-4-6",
        "sonnet": "claude-sonnet-4-6",
        "haiku": "claude-haiku-4-5",
    }
    model_name = model_map.get(model_name, model_name)

    cmd = [
        claude_bin, "-p", "--bare",
        "--output-format", "json",
        "--model", model_name,
    ]

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"<system>\n{system_prompt}\n</system>\n\n{prompt}"

    result = subprocess.run(
        cmd, input=full_prompt,
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): "
            f"{result.stderr[:500]}"
        )

    try:
        data = json.loads(result.stdout)
        return data.get("result", result.stdout)
    except json.JSONDecodeError:
        return result.stdout


def call_openai(prompt, model="gpt-5.4-mini", system_prompt=None,
                temperature=0.7, max_tokens=4096, images=None,
                timeout=None, usage_sink=None, **kwargs):
    """Call OpenAI's API directly.

    The gpt-5 tier renamed max_tokens to max_completion_tokens and 400s on
    the old name; _chat_completion repairs that from the error message.
    """
    from openai import OpenAI

    api_key = _get_key("OPENAI_API_KEY")
    client = _cached_client(
        ("openai", api_key, timeout),
        lambda: OpenAI(api_key=api_key) if timeout is None
        else OpenAI(api_key=api_key, timeout=timeout),
    )
    model = _strip_prefix(model)
    messages = openai_messages(prompt, system_prompt, images)

    response = _chat_completion(
        client, "openai", model, messages, temperature, max_tokens,
        usage_sink=usage_sink,
    )
    return response.choices[0].message.content


def call_deepseek(prompt, model="deepseek/deepseek-v4-pro", system_prompt=None,
                  temperature=0.7, max_tokens=4096, images=None,
                  timeout=None, usage_sink=None, thinking="auto",
                  extra_body=None, **kwargs):
    """Call DeepSeek's API (OpenAI-compatible, text-only).

    Args:
        thinking: "auto" (default) disables thinking on models that would
            otherwise reason — see deepseek_thinking_default and the section
            above it for why that is the right default on an extract path.
            Pass None to send nothing and take the API default (which
            reasons, at effort "high"), or an explicit dict such as
            {"type": "enabled"} to request it.
        extra_body: Merged into the request's extra_body. The `thinking` key
            set from the argument above wins on conflict.
        **kwargs: Forwarded verbatim as top-level request parameters — this is
            the route for `reasoning_effort`. Framework-internal keys
            (_NON_API_KWARGS) are stripped rather than sent.

    Note that with thinking enabled, DeepSeek accepts and ignores
    `temperature`; this records that as a dropped parameter rather than
    letting a sampling claim stand on a parameter that had no effect.
    """
    from openai import OpenAI

    if images:
        raise ValueError(
            "DeepSeek's chat API is text-only; images are not supported."
        )

    api_key = _get_key("DEEPSEEK_API_KEY")
    client = _cached_client(
        ("deepseek", api_key, timeout),
        lambda: OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        if timeout is None
        else OpenAI(api_key=api_key, base_url="https://api.deepseek.com",
                    timeout=timeout),
    )
    model = _strip_prefix(model)
    if model in _DEEPSEEK_LEGACY_ALIASES and model not in _WARNED_DEEPSEEK_ALIASES:
        log.warning(
            "DeepSeek model %r is a retired name kept alive as a server-side "
            "alias; it resolves to %r (the cheap tier). Pin one of %s "
            "explicitly. Note that re-pinning changes the cache key, so "
            "already-annotated items will re-run.",
            model, _DEEPSEEK_LEGACY_ALIASES[model], list(_DEEPSEEK_CURRENT_MODELS),
        )
        _WARNED_DEEPSEEK_ALIASES.add(model)

    messages = openai_messages(prompt, system_prompt)

    body = dict(extra_body or {})
    thinking_param = (deepseek_thinking_default(model) if thinking == "auto"
                      else _normalize_thinking(thinking))
    if thinking_param is not None:
        body["thinking"] = thinking_param

    # Thinking mode accepts temperature and ignores it (docs, quoted above),
    # so there is no 400 for _chat_completion's repair loop to catch. Decide
    # here, from what we are about to send, and record it.
    sent = body.get("thinking")
    reasons = not (isinstance(sent, dict) and sent.get("type") == "disabled")
    dropped_hint = ()
    requested_temperature = temperature
    if reasons and temperature is not None:
        _report_dropped_param(
            "deepseek", model, "temperature", temperature,
            _WARNED_NO_TEMPERATURE,
            reason="thinking mode accepts sampling params and ignores them, "
                   "without erroring — so nothing else would have caught this",
        )
        dropped_hint = ("temperature",)
        temperature = None

    # The audit runs as a usage_filter so its findings land IN the usage
    # record the sink flushes, not merely in a log line after the record has
    # already been written clean.
    def _audit_filter(u, resp):
        if reasons:
            return u
        return _audit_thinking_disabled(
            "deepseek", model, u, resp,
            requested_temperature=requested_temperature,
        )

    response = _chat_completion(
        client, "deepseek", model, messages, temperature, max_tokens,
        usage_sink=usage_sink, dropped_hint=dropped_hint,
        usage_filter=_audit_filter,
        **({"extra_body": body} if body else {}),
        **_api_kwargs(kwargs),
    )
    if usage_sink is None and not reasons:
        # No sink means the filter never ran; the warning (and the strict-
        # mode escalation) must still fire off the response itself.
        _audit_thinking_disabled(
            "deepseek", model, _usage_openai_compat(response), response,
            requested_temperature=requested_temperature,
        )
    return response.choices[0].message.content


def google_request(prompt, model="gemini-3.1-pro-preview", system_prompt=None,
                   temperature=0.7, max_tokens=4096, thinking="auto",
                   images=None):
    """(model, contents, GenerateContentConfig, thinking_setting) — the
    EXACT request the sync path sends, shared with the batch path so the
    two transports cannot drift. The thinking_setting rides along because
    the caller's post-response audit (budget-0 accepted-and-ignored) and
    the loud-rejection wrap both key off what was actually sent.
    """
    from google.genai import types

    model = _strip_prefix(model)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system_prompt:
        config.system_instruction = system_prompt
    setting = google_thinking_setting(model, thinking)
    if setting is not None:
        param, value = setting
        config.thinking_config = types.ThinkingConfig(**{param: value})

    if images:
        parts = []
        for img in images:
            data, mime = _load_image_bytes(img)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        parts.append(types.Part.from_text(text=prompt))
        contents = parts
    else:
        contents = prompt
    return model, contents, config, setting


def call_google(prompt, model="gemini-3.1-pro-preview", system_prompt=None,
                temperature=0.7, max_tokens=4096, images=None,
                timeout=None, usage_sink=None, thinking="auto", **kwargs):
    """Call Google's GenAI API directly.

    Args:
        thinking: "auto" (default) sends thinking_budget=0 on families that
            accept it (gemini-2.5-flash) — Gemini reasons by default, and on
            an extract call the thoughts are text we parse as JSON and
            discard. Families that reject a zero budget (2.5-pro, 3.1-pro)
            get nothing sent and a once-per-model cost warning, like Fable.
            Pass None for the API default, an int for an explicit budget,
            or the cross-provider 'disabled'/'enabled' spellings.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment")

    def _make_google_client():
        if timeout is None:
            return genai.Client(api_key=api_key)
        # google-genai takes timeout in milliseconds via http_options.
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    client = _cached_client(("google", api_key, timeout), _make_google_client)
    model, contents, config, setting = google_request(
        prompt, model=model, system_prompt=system_prompt,
        temperature=temperature, max_tokens=max_tokens, thinking=thinking,
        images=images,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as e:
        if setting is not None and "INVALID_ARGUMENT" in str(e):
            # Matched on what WE sent, not on Google's error prose — the
            # prose already drifted once between generations (2.5-pro says
            # "only works in thinking mode"; 3.6-flash says only "invalid
            # argument"). Deliberately loud, not healed: retrying without
            # the setting would run thinking-on and store the output under
            # a thinking-off/minimal cache key — a silently wrong
            # provenance record, which is worse than this error.
            raise RuntimeError(
                f"google: {model!r} rejected {setting[0]}={setting[1]!r}. "
                f"If this model cannot express 'off', it belongs in "
                f"providers._GOOGLE_THINKING_CANNOT_DISABLE (thinking then "
                f"stays on, warned once, billed as output); if it is a new "
                f"generation, its parameter vocabulary may have changed — "
                f"probe it before adding constants. Original error: {e}"
            ) from e
        raise
    _log_resolved_model("google", model,
                        getattr(response, "model_version", None))
    if setting == ("thinking_budget", 0):
        # Budget-0 accepted-and-ignored is a broken disable and warns.
        # thinking_level="minimal" is NOT audited here: the docs say the
        # model "may reason very minimally for complex tasks" under it, so
        # thoughts there are documented behaviour — the usage receipts and
        # the no_reasoning_observed gate carry them per run.
        m = getattr(response, "usage_metadata", None)
        thoughts = getattr(m, "thoughts_token_count", None) if m else None
        if thoughts and model not in _WARNED_THINKING_NOT_DISABLED:
            _WARNED_THINKING_NOT_DISABLED.add(model)
            log.warning(
                "google: asked %r for thinking_budget=0 and it thought "
                "anyway (%d thought tokens billed as output). Treat these "
                "runs as thinking-mode runs.", model, thoughts,
            )
    if usage_sink is not None:
        usage_sink(_usage_google(response))
    text = response.text
    if text is None:
        # A thinking model can spend the whole max_output_tokens budget on
        # thoughts and return no text part; response.text is then None and
        # the caller's .strip() raises a bare AttributeError three frames
        # away from the cause. Found live: gemini-2.5-pro burned all 512
        # probe tokens thinking. Report the actual finish state instead.
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates \
            else None
        m = getattr(response, "usage_metadata", None)
        thoughts = getattr(m, "thoughts_token_count", None) if m else None
        raise ValueError(
            f"google: response from {model!r} contained no text "
            f"(finish_reason: {finish!r}, thought tokens: {thoughts}). "
            + ("The thinking budget consumed max_output_tokens before any "
               "answer text — raise max_tokens."
               if thoughts and str(finish) and "MAX_TOKENS" in str(finish)
               else "")
        )
    return text


_LOCAL_BACKEND_DEFAULTS = {
    "ollama":   "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm":     "http://localhost:8000/v1",
    # "local/" has no canonical backend; falls through to LOCAL_BASE_URL
    # or Ollama's port as the final default.
}


def _resolve_local_base_url(model: str) -> str:
    """Pick the base URL for a local/OSS model string.

    Priority:
      1. Per-backend env override (OLLAMA_BASE_URL, LMSTUDIO_BASE_URL, VLLM_BASE_URL)
      2. LOCAL_BASE_URL (legacy global override)
      3. Per-backend default (lmstudio→1234, ollama→11434, vllm→8000)
      4. Ollama default as final fallback

    The routing is prefix-pinned so `lmstudio/...` always hits LM Studio and
    `ollama/...` always hits Ollama, even when both servers are running.
    """
    prefix = None
    model_lower = model.lower()
    for p in _LOCAL_BACKEND_DEFAULTS:
        if model_lower.startswith(p + "/"):
            prefix = p
            break

    if prefix:
        per_backend = os.getenv(f"{prefix.upper()}_BASE_URL")
        if per_backend:
            return per_backend

    global_override = os.getenv("LOCAL_BASE_URL")
    if global_override:
        return global_override

    if prefix and _LOCAL_BACKEND_DEFAULTS.get(prefix):
        return _LOCAL_BACKEND_DEFAULTS[prefix]

    return "http://localhost:11434/v1"


def call_local(prompt, model="llama3.3", system_prompt=None,
               temperature=0.7, max_tokens=4096, images=None,
               timeout=None, usage_sink=None, **kwargs):
    """Call a local OpenAI-compatible API (Ollama, vLLM, LM Studio, llama.cpp server).

    Routing is prefix-pinned: `lmstudio/<model>` always hits LM Studio (port
    1234), `ollama/<model>` always hits Ollama (11434), `vllm/<model>` always
    hits vLLM (8000). Override any of them with the corresponding
    `<BACKEND>_BASE_URL` env var. `LOCAL_BASE_URL` still works as a global
    override that wins over the per-backend defaults.

    No API key required; the OpenAI SDK needs a non-empty string so we pass 'local'.

    Quality caveat: open-weight models are meaningfully below API-tier Claude
    and GPT for structured extraction with multilingual content, specialist
    literary knowledge, and strict JSON compliance. Treat as a complement
    (validation passes, dev iteration, cost-free experimentation) rather than
    a drop-in replacement for GenreTask / TranslationTask / PassageTask.
    """
    from openai import OpenAI

    base_url = _resolve_local_base_url(model)
    client = _cached_client(
        ("local", base_url, timeout),
        lambda: OpenAI(api_key="local", base_url=base_url) if timeout is None
        else OpenAI(api_key="local", base_url=base_url, timeout=timeout),
    )
    model = _strip_prefix(model)
    messages = openai_messages(prompt, system_prompt, images)

    # Disable thinking mode for qwen3.5+ which defaults to reasoning — otherwise
    # max_tokens gets burned in `reasoning_content` leaving empty `content`. The
    # OpenAI-compat layer forwards this to Qwen's chat template.
    extra_body = {"cache_prompt": True}
    if "qwen" in model.lower():
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        response = _chat_completion(
            client, "local", model, messages, temperature, max_tokens,
            usage_sink=usage_sink, extra_body=extra_body,
        )
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg or "econnrefused" in msg:
            raise RuntimeError(
                f"Local inference server at {base_url} is not reachable. "
                f"Is the expected backend running? Override via "
                f"OLLAMA_BASE_URL / LMSTUDIO_BASE_URL / VLLM_BASE_URL / "
                f"LOCAL_BASE_URL env if using a different host/port."
            ) from e
        raise
    # _chat_completion already logged the resolved id — a local server answers
    # with whatever checkpoint is loaded, not always the one requested.
    return response.choices[0].message.content


def check_api_keys(verbose=False):
    """Check which provider API keys are available in the environment."""
    keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
    }
    available = {k: v for k, v in keys.items() if v}
    if verbose:
        for k, v in keys.items():
            status = "+" if v else "X"
            print(f"  {status} {k}")
    return available


def set_api_keys():
    """Interactively set API keys (safe for Colab — keys stay in memory only).

    Prompts for each provider key. Press Enter to skip. Uses getpass to
    mask input where available (Colab, terminals), falls back to input().
    Keys are set as environment variables for the current process only.
    """
    try:
        from getpass import getpass
        ask = getpass
    except ImportError:
        ask = input

    providers = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("OPENAI_API_KEY", "OpenAI (GPT)"),
        ("GEMINI_API_KEY", "Google (Gemini)"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
    ]
    for env_var, label in providers:
        existing = os.getenv(env_var)
        if existing:
            print(f"  + {label}: already set")
            continue
        val = ask(f"  {label} API key (Enter to skip): ").strip()
        if val:
            os.environ[env_var] = val
            print(f"  + {label}: set")
        else:
            print(f"  - {label}: skipped")
