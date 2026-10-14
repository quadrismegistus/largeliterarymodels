"""`litmod probe-sampling` — measure whether a decoder parameter GOVERNS.

Why this exists: `providers.effective_sampling` refuses to pin `top_p` (or
its neighbours) on a family nobody has measured, because both available
guesses fail silently and in opposite directions. A refusal is only
defensible if resolving it is cheap, so this is the command that produces
the constant.

What it measures, and why not the obvious thing:

  * NOT acceptance. A 200 response is not evidence a value governed —
    DeepSeek's thinking mode accepts all four sampling parameters and
    ignores them, per its own docs, which is the failure this package
    already pays for elsewhere.
  * NOT distributional distance between arms. At n=100 a distance test has
    almost no power against a default of 0.95, and its null reads as "the
    parameter did nothing" — a false no-op in the one direction that is
    unrecoverable, because it exonerates a corpus that was never pinned.
  * SUPPORT SIZE. Truncation's signature is that outputs become
    *impossible*, not merely rarer: a token past the nucleus boundary has
    probability exactly zero. So count DISTINCT outputs (types, not
    densities) on a flat-continuation prompt. The collapse is visible at
    small n.

Two conditions the design depends on, both easy to get wrong:

  * temperature fixed and ABOVE ZERO in every arm, or the support collapses
    for a reason that has nothing to do with top_p and every family reads as
    fully truncating.
  * never the schema path. JSON-constrained output is low-entropy by
    construction and shows a ceiling effect that reads as a no-op.

Nothing here touches the stash. That is what makes `allow_unmeasured=True`
safe: the refusal protects the durable artifact, and a probe produces none.
"""

import sys

from ..providers import (SAMPLING_PARAMS, _SAMPLING_SUPPORT, provider_route,
                         route_provider)

#: A prompt with a large, near-flat continuation set. The point is many
#: roughly-equiprobable answers, so that removing tail mass removes TYPES.
FLAT_PROMPT = ("Pick a random integer between 1 and 100. "
               "Reply with the number only, no other words.")

DEFAULT_VALUES = (1.0, 0.95, 0.9, 0.7)

#: `temperature` is not in SAMPLING_PARAMS — it is a first-class argument of
#: every call_*, governed by _NO_TEMPERATURE_MODELS rather than by
#: _SAMPLING_SUPPORT — but it is the parameter every Task sets, so a probe
#: that cannot measure it leaves the most-claimed field unmeasured. Its arms
#: run in the same direction as top_p's (lower value, smaller support), so
#: the verdict logic is unchanged; only the defaults differ, and they are
#: wide because temperature 0 against 1 is an enormous effect where top_p
#: 0.95 against 1.0 is a sliver. This is the one question the support test
#: answers with power to spare.
TEMPERATURE_VALUES = (1.0, 0.5, 0.0)


def _normalise(text):
    return (text or "").strip().lower()


def _arm_kwargs(param, value, temperature):
    """The call kwargs for one arm.

    `temperature` is a first-class argument rather than a member of the
    sampling dict, so the UNSET arm for it means sending no temperature at
    all — which is what an unpinned run did.
    """
    if param == "temperature":
        return {"temperature": value, "sampling": None}
    return {"temperature": temperature,
            "sampling": None if value is None else {param: value}}


def _one_arm(model, param, value, n, prompt, temperature, max_tokens):
    """(outputs, error) for n calls at one parameter value.

    `value is None` is the UNSET arm — the vendor's own default, which for
    three of four vendors is undocumented and for one is unknowable. That
    arm is the whole reason to run this: it says what an unpinned corpus
    actually ran at.
    """
    fn = route_provider(model)
    outputs = []
    for _ in range(n):
        try:
            outputs.append(_normalise(fn(
                prompt, model=model, max_tokens=max_tokens,
                allow_unmeasured=True,
                **_arm_kwargs(param, value, temperature),
            )))
        except Exception as e:  # noqa: BLE001
            # A rejection is a RESULT, not a crash: it is the _REJECTS
            # constant, learned in one call rather than n.
            return outputs, e
    return outputs, None


def _verdict(arms, param):
    """(verdict, why, sensitivity) — read the constant off the arms, and
    refuse to over-read them.

    The third value is the part that keeps a null honest. A no-move at
    top_p=0.95 is uninformative on its own: 0.95 removes a sliver of TAIL
    mass and n draws mostly reveal the head, so the test can fail to see a
    real effect. The LOW arms are therefore a positive control on the
    instrument — if 0.7 collapses support, the test demonstrably has
    sensitivity on this prompt, and a null at 0.95 becomes a BOUND on the
    effect rather than an absence of one. If no arm moves at all, nothing
    has been measured, and saying so is the whole job.
    """
    if any(err is not None for _v, _o, err in arms):
        rejected = [v for v, _o, err in arms if err is not None]
        return ("REJECTS",
                f"the API refused {param}={rejected[0]!r}; one call settles "
                f"it. If the family also rejects temperature it admits no "
                f"sampling control at all, and the constant should say so.",
                None)
    supports = {v: len(set(o)) for v, o, _e in arms}
    pinned = {v: s for v, s in supports.items() if v is not None}
    if not pinned:
        return ("INCONCLUSIVE", "no pinned arm ran.", None)
    top = max(pinned)
    moved = [v for v, s in pinned.items() if s != supports[top]]
    if not moved:
        return ("INCONCLUSIVE",
                "support did not move across ANY pinned arm, including the "
                "lowest. That is NOT evidence the parameter was ignored: "
                "an unvalidated instrument and an inert parameter look "
                "identical here. Most likely the prompt's continuation set "
                "is smaller than the nucleus at every value tested — widen "
                "the prompt, lower the floor arm, or raise n. Do not write "
                "a constant from this.",
                "UNVALIDATED — no arm moved, so the test has not shown it "
                "can detect truncation on this prompt at all.")
    lo = min(pinned)
    sensitivity = (f"VALIDATED — {param}={lo} moved support "
                   f"{supports[top]} -> {supports[lo]}, so the test can "
                   f"detect truncation on this prompt. A null at a HIGHER "
                   f"value is now a bound on the effect, not an absence.")
    if supports[lo] < supports[top]:
        return ("ACCEPTS",
                f"support collapses as {param} falls, which is what a "
                f"parameter that governs looks like.",
                sensitivity)
    return ("INCONCLUSIVE",
            "support moved but not monotonically with the parameter; read "
            "the arms before writing a constant.",
            sensitivity)


def _default_match(arms):
    """Which explicit value the UNSET arm's support resembles."""
    unset = next((o for v, o, _e in arms if v is None), None)
    if unset is None:
        return None
    target = len(set(unset))
    pinned = [(v, len(set(o))) for v, o, _e in arms if v is not None]
    if not pinned:
        return None
    best, _ = min(pinned, key=lambda vs: abs(vs[1] - target))
    return best, target


#: Candidate defaults for the rejection scan, most likely first, plus a
#: control that is in range and certainly NOT anybody's default.
DEFAULT_CANDIDATES = (1.0, 0.999, 0.99, 0.95, 0.9)
ABSURD_CONTROL = 0.37


def _error_signature(exc):
    """Coarse identity of an API error, for comparing one refusal to
    another. Type plus the first line, with digits and quoted values
    stripped, so 'invalid value 0.37' and 'invalid value 1.0' compare
    equal — the question is whether the API is refusing the VALUE or the
    PARAMETER, and a message that echoes the value would otherwise make
    every refusal look distinct."""
    import re
    head = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    head = re.sub(r"[0-9]+(\.[0-9]+)?", "N", head)
    head = re.sub(r"'[^']*'", "'X'", head)
    return (type(exc).__name__, head[:160])


def _rejection_scan(model, param, prompt, temperature, max_tokens,
                    candidates=DEFAULT_CANDIDATES):
    """Read an UNDISCLOSED default off a family that rejects non-defaults.

    Where an API accepts a value iff it equals the default, acceptance IS
    an equality test against a number the vendor does not publish — so the
    default is readable in under ten calls, with no sampling and no power
    calculation. Anthropic's Opus line is the live case: it publishes no
    top_p default for any model AND rejects non-default values, which reads
    as unknowable and is in fact the most directly measurable of the four.

    The premise is itself testable, and its failure is silent, so the scan
    carries a NEGATIVE CONTROL (an in-range value that is certainly not a
    default). Without it, "every candidate errored" is ambiguous between
    'the parameter is not settable at all' and 'the default is not in my
    candidate list' — and those want opposite constants.

    Returns (verdict, detail, rows).
    """
    fn = route_provider(model)
    rows = []
    for value in list(candidates) + [ABSURD_CONTROL]:
        try:
            fn(prompt, model=model, max_tokens=max_tokens,
               allow_unmeasured=True,
               **_arm_kwargs(param, value, temperature))
            rows.append((value, True, None))
        except Exception as e:  # noqa: BLE001
            rows.append((value, False, e))

    control_ok = rows[-1][1]
    accepted = [v for v, ok, _e in rows[:-1] if ok]
    sigs = {_error_signature(e) for _v, ok, e in rows if not ok and e}

    if accepted and control_ok:
        return ("SETTABLE", (
            f"{param} was accepted at several values including the control, "
            f"so this family does NOT reject non-defaults and the equality "
            f"trick does not apply here. The parameter is settable; measure "
            f"whether it GOVERNS with the support test (drop "
            f"--rejection-scan)."), rows)
    if len(accepted) == 1 and not control_ok:
        return ("ONLY_VALUE_ACCEPTED", (
            f"exactly one value was accepted ({param}={accepted[0]}) and "
            f"everything else including the control was refused. READ THE "
            f"REFUSAL TEXT BELOW BEFORE INTERPRETING THIS. The pattern is "
            f"consistent with accepts-iff-default — in which case "
            f"{accepted[0]} is the undisclosed default and is what an "
            f"unpinned corpus ran at — and equally consistent with a "
            f"surviving legacy value or a no-op under a DEPRECATED "
            f"parameter, which is if anything likelier, since a parameter "
            f"that does nothing has no reason to reject anything. The "
            f"acceptance pattern alone does not distinguish them; the error "
            f"message usually does."), rows)
    if not accepted and not control_ok and len(sigs) == 1:
        return ("NOT_SETTABLE", (
            "every value was refused with one error shape, the control "
            "included. The premise is wrong: this family rejects the "
            "PARAMETER, not non-default values. The constant is _REJECTS, "
            "and if it also rejects temperature the family admits no "
            "sampling control at all."), rows)
    if not accepted:
        return ("INCONCLUSIVE", (
            "every candidate was refused but the refusals differ in shape, "
            "so 'parameter not settable' and 'default outside the candidate "
            "list' are not yet separated. Read the errors below and widen "
            "the list before writing a constant."), rows)
    return ("INCONCLUSIVE",
            "mixed acceptance without a clean signature; read the rows.",
            rows)


def _run_rejection_scan(args, model, route, param):
    """The under-ten-calls read of an undisclosed default. Cheap enough
    that it is worth trying BEFORE spending n on any restrictive family."""
    cands = list(DEFAULT_CANDIDATES) + [ABSURD_CONTROL]
    print(f"probe-sampling --rejection-scan: {model} ({route}), {param}")
    print(f"  {len(cands)} calls: "
          f"{', '.join(str(c) for c in DEFAULT_CANDIDATES)}, "
          f"and {ABSURD_CONTROL} as a negative control.")
    print("  Reads the default off a family that accepts a value only when "
          "it equals it.")
    if not args.yes:
        try:
            if input("  proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("  aborted.")
                return 1
        except EOFError:
            print("  no tty and --yes not given; aborted.", file=sys.stderr)
            return 1

    verdict, detail, rows = _rejection_scan(
        model, param, args.prompt or FLAT_PROMPT, args.temperature,
        args.max_tokens)
    for value, ok, err in rows:
        tag = "control" if value == ABSURD_CONTROL else ""
        if ok:
            print(f"  {param}={value:<6} ACCEPTED  {tag}")
        else:
            print(f"  {param}={value:<6} refused   {tag} "
                  f"{type(err).__name__}")
    print(f"\n  VERDICT: {verdict} — {detail}")

    # The refusal text IN FULL, never truncated. The one time this was
    # printed at 120 characters, the sentence that overturned the verdict
    # was past the cut: an "undisclosed default" was really
    # "`top_p` is deprecated for this model." The evidence that decides the
    # interpretation is in the message, so the message is not a detail.
    seen = set()
    for value, ok, err in rows:
        if ok or err is None:
            continue
        text = str(err).strip()
        if text in seen:
            continue
        seen.add(text)
        print(f"\n  refusal text ({param}={value}), verbatim:\n"
              f"    {text}")

    if verdict == "ONLY_VALUE_ACCEPTED":
        print("\n  No constant is printed for this verdict on purpose. "
              "Decide from the refusal text above which reading it "
              "supports, and record the TEXT beside whichever constant "
              "you write.")
    elif verdict == "NOT_SETTABLE":
        print(f"\n  Constant to add in providers._SAMPLING_SUPPORT:\n"
              f'    "{route}": {{"{param}": _REJECTS}},')
    return 0


def cmd_probe_sampling(args) -> int:
    model = args.model
    param = args.param
    if param not in SAMPLING_PARAMS and param != "temperature":
        print(f"unknown parameter {param!r}; known: temperature, "
              f"{', '.join(SAMPLING_PARAMS)}", file=sys.stderr)
        return 2
    default_values = (TEMPERATURE_VALUES if param == "temperature"
                      else DEFAULT_VALUES)
    values = ([float(v) for v in args.values.split(",")]
              if args.values else list(default_values))
    arms_to_run = [None] + values          # unset first: it is the question
    total = len(arms_to_run) * args.n

    route = provider_route(model)
    known = _SAMPLING_SUPPORT.get(route, {}).get(param)
    if args.rejection_scan:
        return _run_rejection_scan(args, model, route, param)
    print(f"probe-sampling: {model} ({route}), {param}")
    print(f"  arms: unset, {', '.join(str(v) for v in values)}   n={args.n}")
    print(f"  temperature={args.temperature} (fixed and above zero, or the "
          f"support collapses for the wrong reason)")
    print(f"  {total} live calls, nothing cached, nothing stashed.")
    if known:
        print(f"  NOTE: {route}/{param} is already recorded as {known!r}.")
    if not args.yes:
        try:
            if input("  proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("  aborted.")
                return 1
        except EOFError:
            print("  no tty and --yes not given; aborted.", file=sys.stderr)
            return 1

    arms = []
    for value in arms_to_run:
        label = "unset" if value is None else f"{param}={value}"
        outputs, err = _one_arm(model, param, value, args.n,
                                args.prompt or FLAT_PROMPT,
                                args.temperature, args.max_tokens)
        arms.append((value, outputs, err))
        if err is not None:
            print(f"  {label:>16}: REJECTED — {type(err).__name__}: "
                  f"{str(err)[:120]}")
            break
        distinct = len(set(outputs))
        print(f"  {label:>16}: {distinct:>3} distinct / {len(outputs)} "
              f"samples   e.g. {sorted(set(outputs))[:6]}")

    verdict, why, sensitivity = _verdict(arms, param)
    print(f"\n  VERDICT: {verdict} — {why}")
    if sensitivity:
        print(f"  INSTRUMENT: {sensitivity}")
    if route == "local" and param != "top_k":
        print(f"  NOTE: on the local route, probe top_k too and pin it in "
              f"the same call. These backends apply their own top_k unasked "
              f"and every config surface can report None while it is in "
              f"force, so a {param}-only pin describes a call that did not "
              f"happen (see providers._SAMPLING_COMPANIONS).")

    match = _default_match(arms)
    if match and verdict == "ACCEPTS":
        best, target = match
        print(f"  The unset arm's support ({target}) is closest to "
              f"{param}={best}. Read that as a bracket, not a measurement: "
              f"it says what an unpinned run resembles, which is the number "
              f"an already-collected corpus needs.")

    if verdict in ("ACCEPTS", "REJECTS"):
        if param == "temperature":
            # temperature lives in its own denylist, and the asymmetry is
            # the point: absence from a DENYLIST is not a measurement, it
            # is only "not known to reject". That is exactly the gap this
            # probe closes, so say which fact was established.
            print("\n  temperature is governed by "
                  "providers._NO_TEMPERATURE_MODELS (a denylist), not by "
                  "_SAMPLING_SUPPORT.")
            if verdict == "REJECTS":
                print("  Add the family tag to _NO_TEMPERATURE_MODELS with "
                      "the date and this command line.")
            else:
                print(f"  MEASURED: {model} both accepts temperature AND "
                      f"varies with it, so a run that set it was "
                      f"temperature-controlled. Absence from the denylist "
                      f"had only meant 'not known to reject'; this is the "
                      f"positive fact. Record it beside the tuple.")
        else:
            state = "_ACCEPTS" if verdict == "ACCEPTS" else "_REJECTS"
            print(f"\n  Constant to add in providers._SAMPLING_SUPPORT, "
                  f"with the date and this command line beside it:\n"
                  f'    "{route}": {{"{param}": {state}}},')
    return 0
