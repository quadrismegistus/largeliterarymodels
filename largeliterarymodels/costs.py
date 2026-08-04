"""Cost estimation across every provider this package routes to.

Reads the packaged multi-provider table (model_pricing.json — Anthropic,
OpenAI, DeepSeek, Google, fetched with source URLs), prices measured
workloads, and integrates with UsageTracker so a completed run is priced
in one line:

    from largeliterarymodels import costs
    results = task.map(prompts)
    costs.print_report(task.model, task.usage.report())

    # counterfactuals: the same measured workload on other models
    costs.price(task.model, fresh=500_000, cached=18_000_000, output=650_000)
    costs.compare(fresh=500_000, cached=18_000_000, output=650_000)

A price is not a quote. Pricing a workload measured on model A against
model B is a counterfactual — output volume and tokenization both differ,
and across Anthropic's 4.7 tokenizer boundary the same text is ~30% more
tokens. Models that cannot stop reasoning bill hidden thought tokens as
output, so a non-reasoning workload priced against them is a FLOOR; that
flag is derived from this package's measured thinking behaviour
(providers.py), not from a parallel constant that can drift from it.

Validated against a real invoice: Registration P's gpt-4o-mini arm,
predicted $1.8511 from measured tokens, billed $1.86 (0.05% off). That
check is pinned in the test suite as a known-answer gate.
"""

import datetime
import json
import re
from pathlib import Path

PRICING_FILE = Path(__file__).parent / "model_pricing.json"
M = 1_000_000

_PROVIDERS = ("anthropic", "openai", "deepseek", "google")

_pricing = None


def _load():
    global _pricing
    if _pricing is None:
        with open(PRICING_FILE) as f:
            _pricing = json.load(f)
    return _pricing


def pricing_date():
    """The date the table was fetched — print it beside every estimate.

    A pricing table is a constants file, which is this package's most
    reliable source of rot: the previous one was Anthropic-only, last
    touched 2026-04-27, and could price none of a three-provider run.
    """
    return _load().get("fetched", "unknown")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

_DATED_VARIANT_RE = re.compile(r"^(?P<base>.+)-post-(?P<date>\d{4}-\d{2}-\d{2})$")


def _is_local(model):
    return model.lower().startswith(("local/", "ollama/", "vllm/", "lmstudio/"))


def resolve(model, on=None):
    """(provider, table_name, rates) for a model string, date-aware.

    Accepts the ids this package actually uses: provider-prefixed
    ('anthropic/claude-sonnet-5'), bare, aliased ('sonnet'), and dated
    snapshots served back by providers ('claude-sonnet-5-20260101' falls
    back to its longest table prefix).

    `on` (a date or ISO string, default today) selects between dated
    pricing rows: sonnet-5's introductory rate has a sibling row named
    '<model>-post-2026-09-01', and pricing an August run with September
    prices — or a September projection with August prices — is a silent
    44% error in one direction or the other.
    """
    p = _load()
    if on is None:
        on = datetime.date.today()
    elif isinstance(on, str):
        on = datetime.date.fromisoformat(on)

    name = model
    for prefix in ("anthropic/", "openai/", "deepseek/", "google/"):
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    name = p.get("aliases", {}).get(model, p.get("aliases", {}).get(name, name))

    def _find(n):
        for prov in _PROVIDERS:
            if n in p.get(prov, {}):
                return prov, n
        return None

    hit = _find(name)
    if hit is None:
        # Dated snapshot ids (claude-sonnet-5-20260101) and similar: longest
        # table name that prefixes the requested id.
        candidates = [(prov, n) for prov in _PROVIDERS for n in p.get(prov, {})
                      if name.startswith(n) and not _DATED_VARIANT_RE.match(n)]
        if candidates:
            hit = max(candidates, key=lambda pn: len(pn[1]))
    if hit is None:
        raise ValueError(
            f"unknown model {model!r} — not in model_pricing.json (fetched "
            f"{pricing_date()}). Local models are priced as zero; hosted "
            f"models need a row with a source URL, not a guess."
        )
    prov, base = hit

    # Date-variant selection: '<base>-post-YYYY-MM-DD' rows supersede the
    # base row from their date onward.
    chosen, chosen_from = base, None
    for n in p[prov]:
        mrow = _DATED_VARIANT_RE.match(n)
        if mrow and mrow.group("base") == base:
            start = datetime.date.fromisoformat(mrow.group("date"))
            if on >= start and (chosen_from is None or start > chosen_from):
                chosen, chosen_from = n, start
    return prov, chosen, p[prov][chosen]


def _expiry_warning(provider, name, on):
    """A warning when a dated sibling row takes effect within 30 days."""
    p = _load()
    for n in p.get(provider, {}):
        mrow = _DATED_VARIANT_RE.match(n)
        if mrow and mrow.group("base") == name:
            start = datetime.date.fromisoformat(mrow.group("date"))
            days = (start - on).days
            if 0 < days <= 30:
                future = p[provider][n]
                return (f"{name} pricing changes on {start.isoformat()} "
                        f"({days} days away): input ${future['input']}/M, "
                        f"output ${future['output']}/M. Re-price anything "
                        f"scheduled past that date.")
    return None


def thinking_unavoidable(model):
    """True where measured behaviour says thinking cannot be turned off.

    Derived from providers.py's constants — the same knowledge the call
    path uses — rather than from the pricing table's `reasoning` flag,
    which describes the VENDOR's product tiering. The two disagree
    exactly where it costs money: the table marks claude-fable-5
    reasoning=false (it is not a reasoning-tier product), but its
    thinking cannot be disabled and bills as output all the same.
    """
    from .providers import (_THINKING_CANNOT_DISABLE,
                            _GOOGLE_THINKING_CANNOT_DISABLE, _family_match,
                            _strip_prefix)
    m = _strip_prefix(model).lower()
    return any(_family_match(m, tag) for tag in
               _THINKING_CANNOT_DISABLE + _GOOGLE_THINKING_CANNOT_DISABLE)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def cache_floor(model):
    """Minimum cacheable prefix (tokens) for `model`, or None if unmeasured.

    Sourced from providers.cache_minimum_tokens — the same constants the
    call path warns from — NOT stored in the pricing table, so there is
    one set of floors to rot instead of two. Anthropic floors are
    measured and corroborated against the caching docs; Gemini 3.x's
    4,096 comes from a field report (a 3,906-token instrument that never
    cached across 14,520 calls). OpenAI/DeepSeek floors are unmeasured
    here and return None — unknown, not zero.
    """
    from .providers import cache_minimum_tokens
    return cache_minimum_tokens(model)


def price(model, fresh=0, cached=0, output=0, cache_write_5m=0,
          cache_write_1h=0, batch=False, on=None, times=1,
          prefix_tokens=None):
    """USD for one measured workload on `model`, with its warnings attached.

    Args:
        fresh: uncached input tokens.
        cached: cache-READ input tokens. On models with no cache tier
            (`cached: null` in the table) these bill at the FULL input
            rate — the single largest cost driver on high-cache workloads
            and the easiest thing to miss: 18.4M cached tokens that cost
            $2.76 on gpt-4o-mini would cost $552 on a null-cache model.
        output: output tokens, INCLUDING any reasoning/thought tokens
            (that is how every provider bills them, and how
            UsageTracker.report() counts them).
        cache_write_5m / cache_write_1h: cache-write tokens by TTL
            (Anthropic; elsewhere writes bill as plain input and the
            table's nulls make that arithmetic automatic).
        batch: apply the provider's batch discount where one exists
            (anthropic/openai/google 0.5; deepseek has NO batch API and
            prices unchanged — a 'batch run' there is a plan, not a
            discount).
        on: date for dated pricing rows (default today).
        times: multiply the whole workload (e.g. 3 coder arms).
        prefix_tokens: size of the cacheable prefix this workload ASSUMES
            will cache (the instrument: system prompt + examples). Makes
            a prospective estimate cache-aware: providers decline to
            cache below a per-model floor SILENTLY, so a plan whose
            prefix sits under it pays full input price on every call
            while the estimate says otherwise. When the prefix is below
            the model's known floor, the cached/write lines are re-billed
            at the full input rate and the warning states the padding
            economics — past the floor, a cached prefix bills at ~0.1x,
            so padding UP is routinely cheaper than staying short.
            Retrospective pricing (price_report) does not need this: the
            measured reads already tell the truth.

    Returns dict: usd, usd_list (undiscounted), lines (component
    breakdown), warnings (list of strings a methods note should read),
    provider, model (the table row used), pricing_date.
    """
    if _is_local(model):
        return {
            "provider": "local", "model": model, "usd": 0.0, "usd_list": 0.0,
            "lines": {}, "pricing_date": pricing_date(),
            "warnings": ["local model: $0 API-side; electricity and "
                         "hardware are not modeled."],
        }
    if on is None:
        on = datetime.date.today()
    elif isinstance(on, str):
        on = datetime.date.fromisoformat(on)
    p = _load()
    provider, name, r = resolve(model, on=on)
    warnings = []

    cached_rate = r["cached"]
    if cached_rate is None:
        cached_rate = r["input"]
        if cached:
            warnings.append(
                f"{name} has NO cache tier: {cached:,} cached-read tokens "
                f"bill at the full input rate (${r['input']}/M). This is "
                f"not a small discount being missed; it is the largest "
                f"line on high-cache workloads."
            )
    w5 = r.get("cache_write_5m")
    w1 = r.get("cache_write_1h")
    w5 = r["input"] if w5 is None else w5
    w1 = r["input"] if w1 is None else w1

    if prefix_tokens is not None and (cached or cache_write_5m or cache_write_1h):
        floor = cache_floor(model)
        if floor is not None and prefix_tokens < floor:
            # The assumed caching will not happen: below the floor the
            # provider declines silently and every token bills as fresh
            # input. Re-bill the cache lines honestly rather than let the
            # estimate assert a discount reality will not grant.
            warnings.append(
                f"cacheable prefix ~{prefix_tokens:,} tokens is BELOW "
                f"{name}'s {floor:,}-token cache floor — the provider will "
                f"decline to cache it, silently, and every call pays full "
                f"input price. Cache lines re-billed at ${r['input']}/M. "
                f"Past the floor a cached read bills ~10x cheaper, so "
                f"PADDING the instrument up by ~{floor - prefix_tokens:,} "
                f"tokens (more few-shot examples) is usually cheaper than "
                f"leaving it short."
            )
            cached_rate = r["input"]
            w5 = w1 = r["input"]
        elif floor is None:
            warnings.append(
                f"{name} has no measured cache floor in this package — the "
                f"cached-token pricing assumes the ~{prefix_tokens:,}-token "
                f"prefix caches; verify with a two-call probe before "
                f"trusting it at scale."
            )

    disc = 1.0
    if batch:
        d = p.get("batch_discount", {}).get(provider)
        if d:
            disc = 1.0 - d
        else:
            warnings.append(
                f"{provider} has no batch API/discount — batch pricing "
                f"requested but list rates apply."
            )

    lines = {
        "fresh_input": fresh * r["input"] / M,
        "cached_reads": cached * cached_rate / M,
        "cache_writes_5m": cache_write_5m * w5 / M,
        "cache_writes_1h": cache_write_1h * w1 / M,
        "output": output * r["output"] / M,
    }
    lines = {k: round(v * times, 6) for k, v in lines.items() if v}
    usd_list = round(sum(lines.values()), 4)
    usd = round(usd_list * disc, 4)

    if r.get("reasoning") or thinking_unavoidable(model):
        warnings.append(
            f"{name} bills reasoning/thought tokens as output and cannot "
            f"stop reasoning — an output estimate from a non-reasoning "
            f"workload is a FLOOR, not an estimate."
        )
    exp = _expiry_warning(provider, name, on)
    if exp:
        warnings.append(exp)
    if r.get("note"):
        warnings.append(f"table note: {r['note']}")

    return {"provider": provider, "model": name, "usd": usd,
            "usd_list": usd_list, "lines": lines, "warnings": warnings,
            "pricing_date": pricing_date()}


def price_report(model, report, batch=False, on=None):
    """Price a UsageTracker.report() dict (or a per_item_usage entry).

    The tracker's keys map directly: input_tokens is the uncached
    remainder (fresh), cache_read_tokens the reads, cache_write_tokens the
    writes (priced at the 5m TTL — the default the call path requests),
    output_tokens the full billed output including reasoning. Reasoning
    share, dropped params, and served models ride along as warnings, so
    the dollar figure arrives next to the receipts that qualify it.
    """
    est = price(
        model,
        fresh=report.get("input_tokens", 0),
        cached=report.get("cache_read_tokens", 0),
        output=report.get("output_tokens", 0),
        cache_write_5m=report.get("cache_write_tokens", 0),
        batch=batch, on=on,
    )
    reasoning = report.get("reasoning_tokens", 0)
    if reasoning:
        share = reasoning / max(1, report.get("output_tokens", 1))
        est["warnings"].append(
            f"{reasoning:,} of the output tokens ({share:.0%}) were "
            f"reasoning — text the extract path parses as JSON and "
            f"discards. If that share surprises you, check the thinking "
            f"defaults before the next run, not after it."
        )
    dropped = report.get("dropped_params") or {}
    if dropped:
        est["warnings"].append(
            f"dropped params on this run: {dropped} — do not describe it "
            f"as controlled on those parameters."
        )
    served = report.get("response_models") or {}
    if len(served) > 1:
        est["warnings"].append(
            f"run was served by {len(served)} model ids: {served} — price "
            f"and provenance are both per-id questions now."
        )
    return est


def print_report(model, report, batch=False, on=None):
    """price_report, formatted for a terminal."""
    est = price_report(model, report, batch=batch, on=on)
    print(f"{est['provider']}/{est['model']}"
          f"{'  [batch]' if batch else ''}: ${est['usd']:.4f}"
          f"   (prices fetched {est['pricing_date']})")
    for k, v in est["lines"].items():
        print(f"    {k:<16} ${v:.4f}")
    for w in est["warnings"]:
        print(f"    ! {w}")
    return est


def compare(fresh=0, cached=0, output=0, cache_write_5m=0, batch=False,
            on=None, providers=None, times=1):
    """Price one workload against every model in the table, cheapest first.

    Counterfactual by construction — see the module docstring. Reasoning
    floors are marked; `cached: null` models surface with the cached
    volume priced at full input rate rather than hidden.
    """
    p = _load()
    rows = []
    for prov in providers or _PROVIDERS:
        for name in p.get(prov, {}):
            if _DATED_VARIANT_RE.match(name):
                continue
            est = price(name, fresh=fresh, cached=cached, output=output,
                        cache_write_5m=cache_write_5m, batch=batch, on=on,
                        times=times)
            rows.append(est)
    rows.sort(key=lambda e: e["usd"])
    return rows


# ---------------------------------------------------------------------------
# Back-compat surface (previous Anthropic-only module)
# ---------------------------------------------------------------------------

def resolve_model(model):
    """Previous API: the resolved table name for a model string."""
    return resolve(model)[1]


def estimate(model, input_tokens, output_tokens, n_calls=1, cached_tokens=0):
    """Previous API: n_calls of (input_tokens, output_tokens) where
    cached_tokens of each call's input hit the prompt cache — one write on
    the first call, reads on the rest. Now provider-agnostic.
    """
    fresh_per_call = input_tokens - cached_tokens
    est = price(
        model,
        fresh=fresh_per_call * n_calls,
        cached=cached_tokens * max(0, n_calls - 1),
        cache_write_5m=cached_tokens if n_calls >= 1 and cached_tokens else 0,
        output=output_tokens * n_calls,
        # The per-call cached portion IS the prefix, so the old API gets
        # floor-awareness for free: a plan whose instrument sits under the
        # model's floor re-bills at full rate and says so, instead of
        # promising a discount the provider will silently decline.
        prefix_tokens=cached_tokens if cached_tokens else None,
    )
    no_cache = price(model, fresh=input_tokens * n_calls,
                     output=output_tokens * n_calls)
    lines = est["lines"]
    return {
        "model": est["model"],
        "n_calls": n_calls,
        "input_tokens_per_call": input_tokens,
        "cached_tokens_per_call": cached_tokens,
        "output_tokens_per_call": output_tokens,
        "cache_write": round(lines.get("cache_writes_5m", 0.0), 4),
        "cache_hits": round(lines.get("cached_reads", 0.0), 4),
        "uncached_input": round(lines.get("fresh_input", 0.0), 4),
        "output": round(lines.get("output", 0.0), 4),
        "total": est["usd"],
        "without_cache": no_cache["usd"],
        "cache_savings": round(no_cache["usd"] - est["usd"], 4),
        "warnings": est["warnings"],
    }


def print_estimate(input_tokens, output_tokens, n_calls=1, cached_tokens=0,
                   models=None):
    """Previous API: cost comparison across models (default the Anthropic
    trio; any table model or alias works now)."""
    if models is None:
        models = ["haiku", "sonnet", "opus"]
    print(f"Cost estimate: {n_calls:,} calls × {input_tokens:,} input "
          f"({cached_tokens:,} cached) + {output_tokens:,} output tokens "
          f"(prices fetched {pricing_date()})\n")
    print(f"{'Model':<28s} {'Total':>9s} {'w/o cache':>10s} {'Savings':>8s}")
    print("-" * 60)
    for model in models:
        try:
            e = estimate(model, input_tokens, output_tokens, n_calls,
                         cached_tokens)
            flag = " *floor" if any("FLOOR" in w for w in e["warnings"]) else ""
            print(f"{e['model']:<28s} ${e['total']:>8.2f} "
                  f"${e['without_cache']:>9.2f} ${e['cache_savings']:>7.2f}"
                  f"{flag}")
        except ValueError as ex:
            print(f"{model:<28s} {ex}")


def count_tokens(text):
    """Count tokens using cl100k_base — an APPROXIMATION for every current
    model, and ~30% under for Claude 4.7+ (newer tokenizer). Use the
    provider's count_tokens endpoint when the number matters."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def dry_run(task_class, input_dir, output_tokens=200, models=None, limit=0):
    """Measure token counts from a task's actual prompts, then estimate.

    Unchanged behaviour from the previous module, now priced against the
    multi-provider table.
    """
    import glob
    import json as _json

    files = sorted(glob.glob(f"{input_dir}/*.json"))
    if limit:
        files = files[:limit]

    task = task_class()
    system_prompt = getattr(task, "system_prompt", "")
    examples_text = ""
    if getattr(task, "examples", None):
        for inp, out in task.examples:
            examples_text += str(inp) + str(
                out.model_dump_json() if hasattr(out, "model_dump_json")
                else out)
    cached_tokens = count_tokens(system_prompt + examples_text)

    prompt_tokens = []
    for f in files:
        with open(f) as fh:
            sn = _json.load(fh)
        prompt = task_class.format_input(sn)
        prompt_tokens.append(count_tokens(prompt))

    n = len(prompt_tokens)
    if n == 0:
        print("No files found.")
        return
    avg_prompt = sum(prompt_tokens) // n
    print(f"{n} inputs; cached prefix ~{cached_tokens:,} tokens, prompt "
          f"~{avg_prompt:,} avg (cl100k approximation — see count_tokens)")
    print_estimate(cached_tokens + avg_prompt, output_tokens, n_calls=n,
                   cached_tokens=cached_tokens, models=models)
