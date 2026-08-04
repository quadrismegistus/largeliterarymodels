"""`litmod doctor` — prove each provider still works on current models.

Motivation: three provider bugs shipped at once, and none was findable by
reading the code, because in every case the code was correct about the API
generation it was written against:

  - deepseek: 'deepseek-chat' silently resolved to the cheap tier
  - openai:   max_tokens 400s across the whole gpt-5 tier
  - anthropic: response.content[0].text raises on any model that thinks

A unit test cannot catch this class — the mismatch is between our constants
and a live API. So: probe each provider live with a two-field schema and
assert a valid parse.

Honest scope, learned the hard way: a hard failure (FAIL) catches the
max_tokens and content[0] class. The deepseek class — a wrong model served,
a parameter accepted-and-ignored, thinking billed silently — produces a
perfectly valid parse, so a PASS/FAIL doctor structurally cannot see it.
That is what WARN is for: after a passing parse, the probe inspects the
usage receipt (dropped params, reasoning tokens, the served model id) and
surfaces anything a methods note would need to know. A fourth bug found
2026-08-04 announced itself only in that receipt: DeepSeek v4 reasons by
default on both tiers and accepts-and-ignores `temperature` while doing so.

Each provider is probed on a cheap tier, a current frontier tier, and the
package's own per-provider DEFAULT model — a retired default is exactly the
drift this file exists to catch, and it was the one tier the matrix did not
cover.
"""

import inspect
import os
import sys
import time

from pydantic import BaseModel, Field


class DoctorProbe(BaseModel):
    """Deliberately tiny: two fields, two types, no optionals."""

    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0.0 to 1.0")


PROBE_TEXT = "The garden was ruined, and she found she did not mind at all."

# One cheap and one current frontier model per provider. Update when a tier
# ships — that edit is the point of this file existing. The "default" tier is
# derived from each call_* signature at probe time, not listed here, so it
# cannot drift from the code it checks.
PROVIDER_MATRIX = {
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "cheap": "claude-haiku-4-5",
        "frontier": "claude-sonnet-5",
        "call": "call_anthropic",
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "cheap": "openai/gpt-5.4-nano",
        "frontier": "openai/gpt-5.4",
        "call": "call_openai",
    },
    "google": {
        "env": "GEMINI_API_KEY",
        "cheap": "gemini-2.5-flash",
        "frontier": "gemini-2.5-pro",
        "call": "call_google",
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "cheap": "deepseek/deepseek-v4-flash",
        "frontier": "deepseek/deepseek-v4-pro",
        "call": "call_deepseek",
    },
    # Local endpoints are opt-in: a stopped server is not a provider bug.
    "local": {
        "env": None,
        "cheap": "lmstudio/qwen3.5-35b-a3b",
        "frontier": None,
        "call": None,
        "opt_in": True,
    },
}


def _default_model(call_name):
    """The model a bare call to this provider function would use."""
    if not call_name:
        return None
    from .. import providers
    fn = getattr(providers, call_name, None)
    if fn is None:
        return None
    param = inspect.signature(fn).parameters.get("model")
    return param.default if param is not None else None


def _probe(model, timeout):
    """Run one extraction. Returns (status, detail, elapsed_seconds).

    status: "PASS" | "WARN" | "FAIL". WARN is a valid parse whose usage
    receipt shows something a PASS would otherwise bury: a dropped
    parameter, reasoning billed as output, or the server naming a different
    model than the one requested. Burying those in a detail string is the
    "a warning is only read by someone already suspicious" failure this
    package keeps re-finding; the probe's job is to surface it as a status.
    """
    from hashstash import HashStash

    from ..llm import LLM

    # Strict mode turns a dropped parameter into an exception. For a run,
    # that is the caller's choice; for the doctor it converts a finding this
    # probe exists to REPORT into a crash mid-diagnosis. Park it.
    strict = os.environ.pop("LITMOD_STRICT_PARAMS", None)
    t0 = time.time()
    try:
        # 2048, not 512: thinking models (gemini-2.5-pro cannot disable it)
        # bill their deliberation against max_output_tokens, and 512 was
        # measured being consumed entirely by thoughts — the probe then fails
        # on budget, which reads as a provider bug that isn't there.
        llm = LLM(model=model, temperature=0.0, max_tokens=2048,
                  # Ephemeral stash: probes are diagnostics, not annotations,
                  # and force=True bypasses only the read — the write still
                  # landed in the production stash.
                  stash=HashStash(engine="memory"))
        try:
            result = llm.extract(
                prompt=PROBE_TEXT,
                schema=DoctorProbe,
                system_prompt="Assess the sentiment of the passage.",
                retries=0,
                force=True,
                timeout=timeout,
            )
            # Inside the try: LLM.usage and result are probe evidence, and an
            # attribute error HERE must read as this probe failing, not as
            # the doctor crashing with the remaining providers unprobed.
            report = llm.usage.report()
            detail = (f"sentiment={result.sentiment!r} "
                      f"confidence={result.confidence} | "
                      f"{llm.usage.summary_line()}")
        except Exception as e:  # noqa: BLE001 — reporting every failure is the job
            return "FAIL", f"{type(e).__name__}: {str(e)[:160]}", time.time() - t0

        warnings = []
        if report["dropped_params"]:
            warnings.append(f"dropped params: {report['dropped_params']}")
        if report["reasoning_tokens"] or report["reasoning_observed_calls"]:
            warnings.append(
                f"reasoning observed ({report['reasoning_tokens']} tokens "
                f"billed as output)")
        from ..providers import _strip_prefix
        requested = _strip_prefix(model)
        for served in report["response_models"]:
            if served != requested and not served.startswith(requested):
                warnings.append(
                    f"served by {served!r}, a different model name than "
                    f"requested — record the served id")
        if warnings:
            return "WARN", "; ".join(warnings) + " | " + detail, time.time() - t0
        return "PASS", detail, time.time() - t0
    finally:
        if strict is not None:
            os.environ["LITMOD_STRICT_PARAMS"] = strict


def cmd_doctor(args) -> int:
    from ..providers import check_api_keys

    keys = check_api_keys()
    wanted = ([p.strip() for p in args.provider.split(',') if p.strip()]
              if args.provider else list(PROVIDER_MATRIX))
    unknown = [p for p in wanted if p not in PROVIDER_MATRIX]
    if unknown:
        raise SystemExit(f"unknown provider(s): {unknown}. "
                         f"Known: {sorted(PROVIDER_MATRIX)}")

    tiers = ["cheap", "frontier", "default"]
    if args.cheap_only:
        tiers = ["cheap"]

    failures, warnings, skipped, passed = [], [], [], []
    for name in wanted:
        spec = PROVIDER_MATRIX[name]
        if spec.get("opt_in") and not args.include_local:
            skipped.append(f"{name} (pass --include-local to probe)")
            continue
        if spec["env"] and spec["env"] not in keys:
            skipped.append(f"{name} (no {spec['env']})")
            continue

        print(f"\n{name}", file=sys.stderr, flush=True)
        probed = set()
        for tier in tiers:
            model = (spec.get(tier) if tier != "default"
                     else _default_model(spec.get("call")))
            if not model or model in probed:
                continue
            probed.add(model)
            # Newline before probing: provider warnings log to stderr too, and
            # a half-written line would be split by them.
            print(f"  {tier:<9} {model}", file=sys.stderr, flush=True)
            status, detail, elapsed = _probe(model, args.timeout)
            print(f"            {status} ({elapsed:.1f}s)  {detail}",
                  file=sys.stderr, flush=True)
            bucket = {"PASS": passed, "WARN": warnings,
                      "FAIL": failures}[status]
            bucket.append(f"{name}/{tier} {model}")

    # The summary is the product; it goes to stdout so `litmod doctor >
    # report.txt` captures it (progress and provider logs stay on stderr).
    print(f"\n{'=' * 68}")
    print(f"passed {len(passed)}   warned {len(warnings)}   "
          f"failed {len(failures)}   skipped {len(skipped)}")
    for s in skipped:
        print(f"  skipped: {s}")
    for w in warnings:
        print(f"  WARNED:  {w}")
    for f in failures:
        print(f"  FAILED:  {f}")
    if failures:
        print("\nA frontier-tier failure usually means a constant in "
              "providers.py knows only an older model generation.")
    if not passed and not warnings and not failures:
        # Probing nothing is not health. A doctor that exits 0 because every
        # provider was skipped goes green forever in a CI whose key env var
        # was renamed — the least detectable possible failure of a checkup.
        print("\nprobed NOTHING — every provider was skipped. That is not a "
              "clean bill of health; check the API-key environment variables.")
        return 1
    return 1 if failures else 0
