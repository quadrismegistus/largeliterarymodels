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
                    # Another item already proved this failure systematic.
                    # NOT an item failure: no error entry and no attempt
                    # recorded, because counting pre-empted items as failures
                    # is how one abort made len(errors) report the batch size
                    # rather than the number of items that actually ran.
                    return i, None
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
            for fut in as_completed(futures):
                i, result = fut.result()
                if verbose:
                    log.info("[%d] %s", i, "ok" if result is not None else "FAILED")
                yield i, result
            if breaker.tripped:
                # Raised after draining, so completed work has been yielded
                # and stashed: a rerun resumes rather than restarting. errors
                # carries only the items that actually ran and failed.
                raise BatchAborted(breaker.error(model), errors=errors)
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
    def results_history(self):
        """Iterate (key_dict, [Answers, ...]) — every retained version per key.

        The stash is append-mode, so a key re-run with force=True holds one
        entry per run, oldest first; `results` and `df` show only the latest.
        Without this the versions a forced rerun paid for are on disk and
        unreadable.

        The use case is the model's own variance. Note it is the ONLY way to
        measure that here: a Survey cannot be pinned with temperature=0
        because System One accepts no decoder parameters at all, so
        "identical output across two runs" is a statement about the cache
        until the versions have been forced and compared.
        """
        for key in self.stash.keys():
            try:
                bodies = self.stash.get_all(key)
            except (AttributeError, TypeError):
                bodies = [self.stash[key]]
            if not isinstance(bodies, list):
                bodies = [bodies]
            versions = []
            for body in bodies:
                try:
                    versions.append(parse_answers(body))
                except Exception:  # noqa: BLE001 — a bad version must not
                    continue       # hide the good ones beside it
            if versions:
                yield key, versions

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

    # ---- crossing to a text model ---------------------------------------

    def as_task(self, item=None, model=None, name=None):
        """Compile this question set into a `Task` a text model can answer.

        The direction that works. Questions are the smaller language: every
        Noul is a bool, every Choice a Literal, every Score an ordered int,
        and the instructions and criteria become the field descriptions. The
        reverse compile — schema to questions — does not, which is why this
        class does not offer it.

        The use case is cross-provider replication of ONE instrument: run the
        same questions on System One and on Sonnet and compare. Doing that by
        hand means writing the instrument twice and hoping, which is how two
        arms of a comparison end up measuring two things.

        WHAT IS LOST, AND IT IS THE POINT OF THE CLASS. A text model returns
        a bool where a Noul returns a probability. The compiled Task cannot
        express `noul: 0.62`, so the continuous measure — the thing that
        bought a CI three times tighter — does not survive the crossing. Use
        this to get a comparison arm, not to replace the Survey.

        Question ids become field ALIASES, so the JSON the model is asked for
        is keyed exactly as the Survey's answers are, and
        `result.model_dump(by_alias=True)` lines the two arms up directly.
        Ids that are not Python identifiers (`word::scene`) get a sanitised
        field name and keep the id as the alias.
        """
        import re

        from pydantic import ConfigDict, Field, create_model

        from .task import Task

        qs = self.questions if self.has_static_instrument \
            else self.build_questions(item)
        if qs is None:
            raise ValueError(
                f"{type(self).__name__}.as_task() needs a question set: pass "
                f"item= so build_questions() can produce one.")

        fields, seen = {}, {}
        for qid, q in qs.items():
            fname = re.sub(r"\W", "_", qid)
            if not fname or fname[0].isdigit():
                fname = f"q_{fname}"
            if fname in seen:
                raise ValueError(
                    f"question ids {seen[fname]!r} and {qid!r} both sanitise "
                    f"to the field name {fname!r}. Rename one: a compiled "
                    f"Task cannot keep them apart, and silently merging two "
                    f"questions is worse than refusing."
                )
            seen[fname] = qid
            fields[fname] = _compile_question(qid, q, Field)

        Compiled = create_model(
            name or f"{type(self).__name__}Annotation",
            __config__=ConfigDict(populate_by_name=True),
            **fields,
        )

        survey = self

        class _CompiledTask(Task):
            schema = Compiled
            system_prompt = _COMPILED_SYSTEM_PROMPT
            examples = ()

        _CompiledTask.name = name or f"{survey.task_name}_as_task"
        if model:
            _CompiledTask.model = model
        _CompiledTask.__doc__ = (
            f"{survey.task_name!r} compiled from typed questions. Field "
            f"aliases are the question ids; model_dump(by_alias=True) lines "
            f"this up with Survey answers."
        )
        return _CompiledTask()

    def __repr__(self):
        n = len(self.questions) if self.has_static_instrument else "per-item"
        return (f"<{type(self).__name__} name={self.task_name!r} "
                f"model={self.model!r} questions={n}>")


_COMPILED_SYSTEM_PROMPT = (
    "You are answering a fixed set of independent questions about the item "
    "you are given.\n\n"
    "Answer each question ON ITS OWN against the item. The questions do not "
    "depend on one another and an answer to one is not evidence for "
    "another.\n\n"
    "Boolean fields are yes/no judgements: answer true only where the "
    "described condition holds.\n\n"
    "Integer fields are positions on an ordered scale whose levels are "
    "described in the field's description, counting from 0. Pick the level "
    "whose description fits the item best.\n\n"
    "Where a field description names what a true and a false answer mean, or "
    "what each option covers and does not cover, treat those as the "
    "definition and not as a hint."
)


def _describe(spec, indent=""):
    """Render an instruction or criterion into prompt prose.

    Structured specs are conventions rather than a schema (see questions.py),
    so this renders whatever keys are present instead of looking for known
    ones — a convention this function did not anticipate must still reach the
    model rather than being dropped on the way.
    """
    if spec is None:
        return ""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (list, tuple)):
        return "; ".join(_describe(x) for x in spec)
    if isinstance(spec, dict):
        return " ".join(f"{k}: {_describe(v)}" for k, v in spec.items() if v)
    return str(spec)


def _compile_question(qid, q, Field):
    """One question -> a (type, FieldInfo) pair for create_model."""
    from typing import Literal

    from .questions import Choice, Noul, Score

    if isinstance(q, Noul):
        desc = _describe(q.instructions)
        if q.criteria:
            for k in ("true", "false"):
                if k in q.criteria:
                    desc += f"\n{k.upper()} when {_describe(q.criteria[k])}"
        return (bool, Field(default=False, alias=qid, description=desc))

    if isinstance(q, Choice):
        opts = tuple(q.criteria)
        desc = _describe(q.instructions)
        for opt, spec in q.criteria.items():
            rendered = _describe(spec)
            desc += f"\n{opt}" + (f" = {rendered}" if rendered else "")
        return (Literal[opts], Field(default=opts[0], alias=qid,
                                     description=desc))

    if isinstance(q, Score):
        n = len(q.criteria)
        desc = _describe(q.instructions) + f"\nAn integer 0-{n - 1}:"
        for i, lv in enumerate(q.criteria):
            desc += f"\n{i} = {_describe(lv)}"
        return (int, Field(default=0, ge=0, le=n - 1, alias=qid,
                           description=desc))

    raise ValueError(
        f"question {qid!r} is a raw dict, which as_task() cannot compile: "
        f"the escape hatch carries fields these types do not model, so "
        f"there is no schema field to make from it. Express it as a Noul, "
        f"Choice or Score to cross to a text model."
    )


class _UsageShim:
    """What `_Administered._append_usage_log` needs from an LLM.

    The receipt writer takes `llm.model` and `llm.usage.report()`; a Survey
    has both but is not an LLM. A shim rather than a second writer, because
    two record formats is how a pricing script learns to parse only one.
    """

    def __init__(self, model, usage):
        self.model = model
        self.usage = usage
