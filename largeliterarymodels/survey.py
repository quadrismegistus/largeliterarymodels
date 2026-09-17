"""Survey: administer typed questions, get typed answers.

The sibling of `Task`. A Task sends a system prompt built from a Pydantic
schema and parses text back; a Survey sends state plus a dict of typed
questions and gets typed answers back, each carrying a probability.

    class Abstraction(Survey):
        name = "abstraction"
        questions = {
            "concrete_bespeaks_abstract": Noul(
                instructions={"question": "Does the passage bind concrete "
                                          "particulars to abstract meanings?",
                              "inspect": "passage"},
                criteria={"true": {"what": "..."}, "false": {"what": "..."}}),
            "abstractness": Score(instructions="How abstract is the passage?",
                                  criteria=["concrete", "mixed", "abstract"]),
        }

    s = Abstraction()
    answers = s.run({"passage": text})
    answers.p("concrete_bespeaks_abstract")      # 0.91, not True

## WHY NOT A MODE OF TASK

Compiling a schema down to questions throws away the reason to be here.
`bool` -> Noul -> threshold at 0.5 -> `bool` pays for a probability and
stores a boolean, and the probability is the whole gain (see answers.py for
the measurement). The compile also does not work on the cases that gain
most: `Charge.words: list[Word]` became tractable because each candidate
word became a question KEY, and that is not derivable from the schema — it
needs the candidate list supplied at call time.

The reverse direction IS total, which is why questions are the source of
truth here and `as_task()` (still to come) will run the other way.

## PER-ITEM QUESTION SETS

`questions` may be a dict (one instrument for the corpus) or built per item
by overriding `build_questions`. The per-item case is what makes the
list-valued shape work: ask one question per candidate and the response is
keyed by the candidates you sent, so the rater cannot drop, invent,
duplicate or re-case one. That failure class — the one `Task`'s list-field
retry exists to catch — is structurally absent.

Where the question set is per item there is no static instrument, so
`instrument_text()`/`instrument_sha256()` refuse rather than fabricate, the
same way `SequentialTask`'s do. The cache key is unaffected: it always
carries the questions ACTUALLY administered for that item.
"""

import json
import logging

from .administered import _Administered
from .answers import parse as parse_answers
from .llm import (
    BatchAborted, UsageTracker, _make_survey_key, _record_item_usage,
    _resolve_breaker, _stash_read,
)
from .questions import canonical, questions_sha256, to_wire

log = logging.getLogger(__name__)

DEFAULT_SURVEY_MODEL = "jev-latest"


class Survey(_Administered):
    """A question set administered to items, with its own cache."""

    name = None
    #: dict[str, Noul|Choice|Score], or None when build_questions is overridden.
    questions = None
    model = DEFAULT_SURVEY_MODEL
    #: Transport retries. Low by default because there is no JSON to fail to
    #: parse — a Survey failure is a transport failure, not a malformed answer.
    retries = 1
    num_workers = 8

    _KNOWN_ATTRS_HINT = ("name, questions, model, retries, num_workers, "
                         "usage_log, raw_log")

    # ---- the instrument -------------------------------------------------

    def build_questions(self, item):
        """The question set for one item. Override for per-item questions."""
        if self.questions is None:
            raise ValueError(
                f"{type(self).__name__} has no `questions` and does not "
                f"override build_questions()."
            )
        return self.questions

    def build_state(self, item):
        """The state for one item — the content the questions are asked of.

        A mapping is passed through, so named fields reach the model as named
        fields. A bare string is wrapped rather than sent raw, because an
        unnamed blob is the one shape a question's `inspect` field cannot
        refer to.
        """
        if isinstance(item, str):
            return {"item": item}
        return dict(item)

    @property
    def has_static_instrument(self):
        return type(self).build_questions is Survey.build_questions \
            and self.questions is not None

    def _no_static_instrument(self, what):
        raise ValueError(
            f"{type(self).__name__}.{what} is unavailable: this Survey builds "
            f"its questions per item (build_questions is overridden), so there "
            f"is no single instrument to render. The questions actually "
            f"administered for an item are in that item's cache key; "
            f"questions_sha256(self.build_questions(item)) digests them."
        )

    def instrument_text(self):
        """The administered bytes: the canonical question set.

        Byte-identical to what is sent, because the wire form IS the
        canonical form — see questions.py. The counterpart of
        Task.instrument_text().
        """
        if not self.has_static_instrument:
            self._no_static_instrument("instrument_text()")
        return canonical(self.questions)

    def instrument_sha256(self):
        if not self.has_static_instrument:
            self._no_static_instrument("instrument_sha256()")
        return questions_sha256(self.questions)

    def administration_record(self, model=None):
        """Instrument digest plus what the digest deliberately omits.

        Thinner than Task's by exactly the fields System One does not have:
        no temperature, no max_tokens, no sampling, no thinking. That absence
        is recorded rather than left to be inferred, because "the methods
        note could not say what it ran at" and "there was nothing to run it
        at" are different claims.
        """
        rec = {
            "survey": self.task_name,
            "class": type(self).__name__,
            "model": model or self.model,
            "n_questions": (len(self.questions)
                            if self.has_static_instrument else None),
            "per_item_questions": not self.has_static_instrument,
            "decoder_parameters": None,
            "decoder_parameters_note": (
                "System One accepts none: no temperature, top_p, penalties, "
                "seed or thinking control. Nothing was pinned because nothing "
                "is pinnable."),
        }
        if self.has_static_instrument:
            rec["instrument_sha256"] = self.instrument_sha256()
        return rec

    # ---- administration -------------------------------------------------

    def _prepare(self, item, metadata=None, sample=None, model=None):
        """(state_json, questions_wire, questions_json, key) for one item."""
        state = self.build_state(item)
        wire = to_wire(self.build_questions(item))
        state_json = json.dumps(state, ensure_ascii=False, sort_keys=False)
        questions_json = json.dumps(wire, indent=2, ensure_ascii=False,
                                    sort_keys=False)
        key = _make_survey_key(state_json, model or self.model, questions_json,
                               metadata=metadata, sample=sample)
        return state, wire, key

    def run(self, item, model=None, metadata=None, sample=None, force=False,
            **kwargs):
        """Administer the question set to one item; return `Answers`."""
        from . import providers

        model = model or self.model
        state, wire, key = self._prepare(item, metadata, sample, model)

        if not force:
            hit, cached = _stash_read(self.stash, key, None, model)
            if hit:
                return parse_answers(cached, asked=wire)

        sidecar = self._resolved_raw_log()
        raw_sink = None
        if sidecar is not None:
            raw_sink = lambda body: sidecar.record(  # noqa: E731
                key, body, transport="sync", model=model, provider="typesafe")

        body = providers.call_typesafe_survey(
            state=state, questions=wire, model=model,
            usage_sink=self.usage.record, raw_sink=raw_sink, **kwargs)
        self.stash[key] = body
        return parse_answers(body, asked=wire)

    def imap(self, items, model=None, metadata_list=None, num_workers=None,
             force=False, verbose=False, errors=None, per_item_usage=None,
             fail_fast=5, samples=None, **kwargs):
        """Yield (index, Answers) as each completes; cached items first.

        Each result is stashed the moment it completes, so a partial run is
        resumable. Failures yield (index, None) and are recorded in `errors`.
        """
        llm_model = model or self.model
        items = list(items)
        if not self.usage_log:
            yield from self._imap(items, llm_model, metadata_list,
                                  num_workers, force, verbose, errors,
                                  per_item_usage, fail_fast, samples, kwargs)
            return

        shim = _UsageShim(llm_model, self.usage)

        def _run(log_errors, log_items):
            yield from self._imap(items, llm_model, metadata_list,
                                  num_workers, force, verbose, log_errors,
                                  log_items, fail_fast, samples, kwargs)

        yield from self._with_usage_log(shim, metadata_list, errors,
                                        per_item_usage, _run)

    def _imap(self, items, model, metadata_list, num_workers, force, verbose,
              errors, per_item_usage, fail_fast, samples, kwargs):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from . import providers

        num_workers = num_workers or self.num_workers
        breaker = _resolve_breaker(fail_fast)
        batch_usage = UsageTracker()
        sidecar = self._resolved_raw_log()

        prepared, pending = [], []
        for i, item in enumerate(items):
            meta = metadata_list[i] if metadata_list else None
            sample = samples[i] if samples else None
            state, wire, key = self._prepare(item, meta, sample, model)
            prepared.append((i, state, wire, key))

        # Cache pass first, so a resumed run yields its warm half immediately
        # and only the cold half reaches a worker. No cache WARMING pass: that
        # exists on the extract path to avoid N parallel prompt-cache write
        # premiums, and this provider has no cache tier to write to.
        for i, state, wire, key in prepared:
            if not force:
                hit, cached = _stash_read(self.stash, key, None, model)
                if hit:
                    if verbose:
                        log.info("[%d] cached", i)
                    yield i, parse_answers(cached, asked=wire)
                    continue
            pending.append((i, state, wire, key))

        if not pending:
            return

        def _sink(i):
            def sink(usage):
                self.usage.record(usage)
                batch_usage.record(usage)
                if per_item_usage is not None:
                    _record_item_usage(per_item_usage, i, usage)
            return sink

        def _do_one(task):
            i, state, wire, key = task
            last_error = None
            for attempt in range(1 + self.retries):
                if breaker.tripped:
                    raise breaker.error(model)
                breaker.record_attempt()
                try:
                    raw_sink = None
                    if sidecar is not None:
                        raw_sink = lambda body, k=key: sidecar.record(  # noqa: E731
                            k, body, transport="sync", model=model,
                            provider="typesafe", attempt=attempt)
                    body = providers.call_typesafe_survey(
                        state=state, questions=wire, model=model,
                        usage_sink=_sink(i), raw_sink=raw_sink, **kwargs)
                    self.stash[key] = body
                    breaker.record_success()
                    return i, parse_answers(body, asked=wire)
                except Exception as e:  # noqa: BLE001 — recorded per item
                    last_error = e
            breaker.record_failure(last_error)
            if errors is not None:
                errors[i] = {
                    "index": i,
                    "error": f"{type(last_error).__name__}: {last_error}",
                    "exception": last_error,
                    "attempts": 1 + self.retries,
                    "metadata": metadata_list[i] if metadata_list else None,
                }
            return i, None

        executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            futures = {executor.submit(_do_one, t): t for t in pending}
            aborted = None
            for fut in as_completed(futures):
                try:
                    i, result = fut.result()
                except BatchAborted as e:
                    aborted = e
                    continue
                if verbose:
                    log.info("[%d] %s", i, "ok" if result is not None else "FAILED")
                yield i, result
            if aborted is not None:
                raise aborted
        finally:
            executor.shutdown(wait=False)
            if verbose:
                log.info("batch: %s", batch_usage.summary_line())
            if sidecar is not None:
                sidecar.flush_receipt()

    def map(self, items, **kwargs):
        """Administer to every item; return a list of `Answers` in order."""
        items = list(items)
        out = [None] * len(items)
        for i, result in self.imap(items, **kwargs):
            out[i] = result
        return out

    # ---- reading back ---------------------------------------------------

    @property
    def results(self):
        """Iterate (key_dict, Answers), latest per key."""
        try:
            pairs = self.stash.items(all_results=False)
        except TypeError:
            latest = {}
            for key, body in self.stash.items():
                latest[json.dumps(key, sort_keys=True, default=str)] = (key, body)
            pairs = latest.values()
        for key, body in pairs:
            try:
                yield key, parse_answers(body)
            except Exception:  # noqa: BLE001 — a bad row must not stop the walk
                continue

    @property
    def df(self):
        """One row per administration; one column per question.

        Values are the CONTINUOUS answer — a probability for a Noul, a
        position for a Score, the label for a Choice — with `<q>__conf`
        beside the two that carry a confidence. Nothing is thresholded here;
        the cut belongs to the analysis, not to the store.
        """
        import pandas as pd

        rows = []
        for key, answers in self.results:
            row = {"model": key.get("model", "") if isinstance(key, dict) else ""}
            if isinstance(key, dict):
                if key.get("sample") is not None:
                    row["sample"] = key["sample"]
                meta = key.get("metadata")
                if isinstance(meta, dict):
                    for k, v in meta.items():
                        row[f"meta_{k}"] = v
            for qid, a in answers.items():
                row[qid] = a.value
                conf = getattr(a, "confidence", None)
                if conf is not None:
                    row[f"{qid}__conf"] = conf
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self):
        n = len(self.questions) if self.has_static_instrument else "per-item"
        return (f"<{type(self).__name__} name={self.task_name!r} "
                f"model={self.model!r} questions={n}>")


class _UsageShim:
    """What `_Administered._append_usage_log` needs from an LLM.

    The receipt writer takes `llm.model` and `llm.usage.report()`; a Survey
    has both but is not an LLM. A shim rather than a second writer, because
    two record formats is how a pricing script learns to parse only one.
    """

    def __init__(self, model, usage):
        self.model = model
        self.usage = usage
