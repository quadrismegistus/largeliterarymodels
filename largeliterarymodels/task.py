"""Task class: reusable structured extraction tasks with their own cache."""

import hashlib
import json
import logging
import os
from hashstash import HashStash
from .llm import (
    LLM, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, STASH_PATH,
    UsageTracker, _build_extract_prompt, _diagnose_partial_response,
    _parse_json_response, _retry_prompt, _validate_parsed, _unwrap_schema,
)

log = logging.getLogger(__name__)


def _merge_per_item(dst, src_items):
    """Accumulate batch-local per-item rows into a caller's dict.

    Preserves the documented accumulation semantics of a reused
    per_item_usage dict while letting the usage_log collect into a fresh
    one — the direct-write arrangement double-counted: batch 2's receipt
    carried batch 1's tokens.
    """
    for i, entry in src_items.items():
        row = dst.setdefault(i, {
            "index": i, "calls": 0, "input_tokens": 0,
            "output_tokens": 0, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "reasoning_tokens": 0,
            "response_model": None,
        })
        for k in ("calls", "input_tokens", "output_tokens",
                  "cache_read_tokens", "cache_write_tokens",
                  "reasoning_tokens"):
            row[k] = row.get(k, 0) + entry.get(k, 0)
        if entry.get("response_model"):
            row["response_model"] = entry["response_model"]
        if "duplicate_of" in entry:
            row["duplicate_of"] = entry["duplicate_of"]


class Task:
    """A reusable structured extraction task.

    Bundles together a Pydantic schema, system prompt, few-shot examples,
    and retry config. Each task gets its own HashStash subdirectory.

    Subclass to define a task:

        class BechdelTask(Task):
            name = "bechdel"
            schema = BechdelResult
            system_prompt = "You are a literary critic assessing the Bechdel test..."
            examples = [
                ("INT. HOUSE...", BechdelResult(...)),
            ]

    Then use:

        task = BechdelTask()
        result = task.run(scene_text)
        results = task.map(scenes)

        # with images and metadata
        result = task.run("Extract entries from this page.",
                          images=["page1.png"],
                          metadata={"page": 1, "source": "mish_biblio.pdf"})

    Schema note: a list-typed field carries a small reliability tax. Observed
    across Anthropic, OpenAI and DeepSeek, a model occasionally returns the
    bare value of a list field instead of the whole object — e.g.
    ``["SEQUENCE", "SPECIFICITY"]`` rather than ``{"relations": [...], ...}``.
    Rates are low (1 in ~2,000 on one measured run) and a retry recovers it,
    since llm._diagnose_partial_response names the offending field in the
    reprompt rather than complaining about JSON that parsed fine. Worth knowing
    when choosing between one list field and several booleans, and worth
    budgeting a retry for.
    """

    name = None  # defaults to class name if not set
    schema = None
    system_prompt = None
    examples = ()  # immutable default; subclasses may use lists
    retries = 1
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS
    # Prompt-cache lifetime for the instrument: None = 5-minute default,
    # "1h" for long or resumed runs. The 1-hour write costs 2x base input vs
    # 1.25x, so it pays back only across three or more reads — and a "1h"
    # request reads an already-warm 5-minute entry instead of upgrading it,
    # so set this before a run rather than partway through.
    cache_ttl = None
    # Opt-in durable usage receipts. The stash stores only the response
    # text, so token counts — and with them any post-hoc price estimate —
    # die with the process. usage_log=True appends one JSONL record per
    # map() batch (run report + per-item rows: tokens, reasoning,
    # response_model) to data/usage_logs/<task_name>.jsonl. Additive: no
    # stash value or key is touched, old entries simply have no log rows.
    usage_log = False

    # Attributes settable via __init__ kwargs even when the class doesn't
    # declare them (subclasses commonly set `model` as a class attribute,
    # but the base class leaves it to _get_llm's DEFAULT_MODEL fallback).
    _DYNAMIC_ATTRS = {'model'}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(type(self), k) and k not in self._DYNAMIC_ATTRS:
                import warnings
                warnings.warn(
                    f"{type(self).__name__}({k}=...): unknown attribute — "
                    f"possible typo? Known: name, schema, system_prompt, "
                    f"examples, retries, temperature, max_tokens, model.",
                    stacklevel=2,
                )
            setattr(self, k, v)
        self._stash = None
        self._human_stashes = {}
        self._usage = None

    @property
    def task_name(self):
        return self.name or self.__class__.__name__

    @property
    def stash(self):
        if self._stash is None:
            stash_dir = os.path.join(STASH_PATH, self.task_name)
            self._stash = HashStash(stash_dir, engine="pairtree", append_mode=True)
        return self._stash

    def human_stash(self, annotator: str = 'default'):
        """JSONL-backed stash for human annotations by this annotator.

        Uses hashstash flat mode: each write appends a plain-JSON line with
        dict fields inlined at top level (greppable, jq-queryable). Append-
        only on disk (full edit history preserved); reads return the latest
        value per key.

        Usage:
            stash = task.human_stash('ryan')
            stash[item_key] = {'field1': True, ...}   # append edit
            stash[item_key]                            # latest dict for key
            stash.items()                              # {key: latest_value}
            stash.df                                   # all history as DataFrame

        Files live under data/stash/_human_annotations/<task>/<annotator>/
        jsonl.hashstash.raw/data.jsonl.
        """
        if annotator not in self._human_stashes:
            root = os.path.join(
                STASH_PATH, '_human_annotations', self.task_name, annotator,
            )
            # hashstash >= 0.4 defaults jsonl engine to flat/raw/no-b64 —
            # no flags needed.
            self._human_stashes[annotator] = HashStash(
                root_dir=root, engine='jsonl',
            )
        return self._human_stashes[annotator]

    @property
    def usage(self):
        """Token usage accumulated across this task's calls.

        A receipt rather than an assumption: a run whose prompt cache silently
        stopped working produces identical output at roughly ten times the
        input price, and thinking tokens bill as output. Read after a run:

            results = task.map(prompts)
            print(task.usage.summary_line())
            task.usage.report()['cache_hit_rate']

        Counts only live provider calls — cache hits from the local stash never
        reach a provider, so a fully-cached run reports zero.
        """
        if self._usage is None:
            self._usage = UsageTracker()
        return self._usage

    def _get_llm(self, model=None):
        """Get an LLM instance using this task's stash."""
        return LLM(
            model=model or getattr(self, 'model', None) or DEFAULT_MODEL,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stash=self.stash,
            cache_ttl=self.cache_ttl,
            # One tracker per Task, shared by every LLM it builds, so counts
            # accumulate across run/map calls instead of resetting per call.
            usage=self.usage,
        )

    def run(self, prompt, model=None, system_prompt=None, examples=None,
            images=None, metadata=None, force=False, **kwargs):
        """Extract structured data from a single input.

        Args:
            prompt: The input text to extract from.
            model: Override the default model.
            system_prompt: Override the task's system_prompt.
            examples: Override the task's few-shot examples.
            images: List of images (file paths, bytes, or PIL Images).
            metadata: Dict of user-defined metadata (e.g. page_number).
            force: Bypass cache.
            **kwargs: Additional arguments passed to LLM.extract().

        Returns:
            Validated Pydantic model instance (or list thereof).
        """
        if self.schema is None:
            raise ValueError(f"Task '{self.name}' has no schema defined.")
        llm = self._get_llm(model)
        return llm.extract(
            prompt=prompt,
            schema=self.schema,
            system_prompt=system_prompt or self.system_prompt,
            examples=examples if examples is not None else self.examples,
            images=images,
            metadata=metadata,
            retries=self.retries,
            force=force,
            **kwargs,
        )

    def _imap_kwargs(self, system_prompt=None, examples=None,
                     images_list=None, metadata_list=None,
                     num_workers=4, force=False, verbose=False,
                     errors=None, per_item_usage=None, **kwargs):
        """Build kwargs for extract_imap/extract_map."""
        return dict(
            schema=self.schema,
            system_prompt=system_prompt or self.system_prompt,
            examples=examples if examples is not None else self.examples,
            images_list=images_list,
            metadata_list=metadata_list,
            num_workers=num_workers,
            retries=self.retries,
            force=force,
            verbose=verbose,
            errors=errors,
            per_item_usage=per_item_usage,
            **kwargs,
        )

    def imap(self, prompts, model=None, system_prompt=None, examples=None,
             images_list=None, metadata_list=None,
             num_workers=4, force=False, verbose=False, errors=None,
             per_item_usage=None, **kwargs):
        """Extract structured data, yielding (index, result) as each completes.

        Cached items yield first, then API results in completion order.
        Each result is cached the moment it completes, so partial runs
        are resumable.

        Args:
            errors: Optional dict. Failed items are recorded as
                ``errors[index] = {...}`` — see ``map`` for the shape.
        """
        if self.schema is None:
            raise ValueError(f"Task '{self.name}' has no schema defined.")
        llm = self._get_llm(model)
        if not self.usage_log:
            yield from llm.extract_imap(
                prompts=prompts,
                **self._imap_kwargs(system_prompt=system_prompt,
                                    examples=examples,
                                    images_list=images_list,
                                    metadata_list=metadata_list,
                                    num_workers=num_workers, force=force,
                                    verbose=verbose, errors=errors,
                                    per_item_usage=per_item_usage, **kwargs),
            )
            return
        # usage_log covers imap too — it is the resumable long-run path,
        # which is exactly where a durable receipt matters most. Same
        # fresh-dict-then-merge shape as map(); the receipt lands in the
        # finally so a consumer that stops iterating early still gets one
        # for the calls that were made.
        prompts = list(prompts)
        log_items = {}
        log_errors = errors if errors is not None else {}
        try:
            yield from llm.extract_imap(
                prompts=prompts,
                **self._imap_kwargs(system_prompt=system_prompt,
                                    examples=examples,
                                    images_list=images_list,
                                    metadata_list=metadata_list,
                                    num_workers=num_workers, force=force,
                                    verbose=verbose, errors=log_errors,
                                    per_item_usage=log_items, **kwargs),
            )
        finally:
            try:
                self._append_usage_log(llm, log_items, metadata_list,
                                       log_errors)
            except Exception as e:  # noqa: BLE001
                log.error("usage_log: failed to append receipt (%s: %s)",
                          type(e).__name__, e)
            if per_item_usage is not None:
                _merge_per_item(per_item_usage, log_items)

    def map(self, prompts, model=None, system_prompt=None, examples=None,
            images_list=None, metadata_list=None,
            num_workers=4, force=False, verbose=False, errors=None,
            per_item_usage=None, batch=False, **kwargs):
        """Extract structured data from multiple inputs, with parallelism.

        Like imap but collects all results into a list in prompt order.

        Args:
            errors: Optional dict for per-item failure diagnostics. A None in
                the returned list is positional and otherwise opaque; pass a
                dict here and each failed index gets an entry with keys
                ``index``, ``error``, ``exception``, ``attempts``,
                ``metadata``, ``prompt_head``, ``raw`` (and ``duplicate_of``
                when the item shared a de-duplicated call):

                    errors = {}
                    results = task.map(prompts, metadata_list=metas,
                                       errors=errors)
                    for i, e in errors.items():
                        print(i, e['metadata'], e['error'])

                Only failures are recorded, so ``errors`` stays empty on a
                clean run and ``len(errors)`` is the failure count.
        """
        if self.schema is None:
            raise ValueError(f"Task '{self.name}' has no schema defined.")
        llm = self._get_llm(model)
        # With usage_log on, per-item usage is ALWAYS collected into a fresh
        # internal dict, then merged into the caller's afterwards. Logging
        # directly from a caller-supplied dict double-counted: callers may
        # reuse one dict across batches (accumulation is its documented
        # behaviour), and a record claiming to describe one batch carried
        # every previous batch's tokens inside it.
        log_items = {} if self.usage_log else None
        pass_items = log_items if self.usage_log else per_item_usage
        log_errors = errors if errors is not None else (
            {} if self.usage_log else None)
        if batch:
            # The batch transport: 50% pricing where the provider offers
            # it, identical stash keys, ledger-safe submission. See
            # largeliterarymodels.batch.extract_batch for the contract.
            if images_list:
                raise ValueError(
                    "batch=True does not carry images (payload size); use "
                    "the concurrent path for image tasks.")
            results = llm.extract_batch(
                prompts, self.schema,
                system_prompt=system_prompt or self.system_prompt,
                examples=examples if examples is not None else self.examples,
                metadata_list=metadata_list, force=force,
                retries=self.retries, errors=log_errors,
                per_item_usage=pass_items, **kwargs,
            )
        else:
            results = llm.extract_map(
                prompts=prompts,
                **self._imap_kwargs(system_prompt=system_prompt,
                                    examples=examples,
                                    images_list=images_list,
                                    metadata_list=metadata_list,
                                    num_workers=num_workers, force=force,
                                    verbose=verbose, errors=log_errors,
                                    per_item_usage=pass_items, **kwargs),
            )
        if self.usage_log:
            # A receipt must never fail the run it receipts: the results
            # exist, the stash holds them, and an unwritable log line is a
            # diagnostic, not a crash.
            try:
                self._append_usage_log(llm, log_items, metadata_list,
                                       log_errors)
            except Exception as e:  # noqa: BLE001
                log.error("usage_log: failed to append receipt (%s: %s) — "
                          "the run itself succeeded and the stash holds "
                          "its results", type(e).__name__, e)
            if per_item_usage is not None:
                _merge_per_item(per_item_usage, log_items)
        return results

    def _append_usage_log(self, llm, per_item, metadata_list, errors=None):
        """Append this batch's usage receipts as one JSONL record.

        The stash stores only response text, so token counts (and any
        post-hoc price estimate) are otherwise unrecoverable once the
        process exits — Registration P's pricing worked only because that
        producer kept its own artifact. One record per batch: timestamp,
        model, the batch report, and per-item rows keyed by index with
        metadata attached where given. Fully-cached batches log a report
        of zeros — a receipt that nothing was billed is still a receipt.
        """
        import time

        log_dir = os.path.join(os.path.dirname(STASH_PATH), "usage_logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{self.task_name}.jsonl")
        items = []
        for i, entry in sorted((per_item or {}).items()):
            row = dict(entry)
            if metadata_list and i < len(metadata_list) and metadata_list[i]:
                row["metadata"] = metadata_list[i]
            if errors and i in errors:
                # Spend on an item that produced no annotation is still
                # spend, but a per-annotation cost table must not read it
                # as an annotation's price.
                row["failed"] = True
            items.append(row)
        # Batch totals from the per-item rows, not the Task-lifetime
        # tracker: a record claiming to describe THIS batch must not carry
        # every previous batch's tokens inside it.
        token_keys = ("calls", "input_tokens", "output_tokens",
                      "cache_read_tokens", "cache_write_tokens",
                      "reasoning_tokens")
        batch = {k: sum(e.get(k, 0) for e in items) for k in token_keys}
        served = {}
        for e in items:
            rm = e.get("response_model")
            # Rows that made no call (de-duplicated twins) carry a copied
            # response_model for provenance, but counting them made
            # response_models sum to 3 beside calls: 2 in the same dict.
            if rm and e.get("calls", 0) > 0:
                served[rm] = served.get(rm, 0) + 1
        batch["response_models"] = served
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task": self.task_name,
            "model": llm.model,
            "batch": batch,
            "items": items,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        log.info("usage_log: appended %d item rows to %s", len(items), path)

    # ------------------------------------------------------------------
    # Instrument serialisation
    #
    # An annotation scheme that only one code path can administer is not
    # frozen, only committed: cross-provider replication, a subagent second
    # coder, or a human on paper all require the instrument as text. These
    # methods delegate to llm._build_extract_prompt — the same function the
    # API path calls — so the rendered instrument cannot drift from the
    # administered one. Do not re-render the parts here.
    # ------------------------------------------------------------------

    ITEM_HEADER = "=== ITEM TO ANNOTATE ==="
    ITEM_FOOTER = "=== END ITEM ==="
    # The one-line contract reminder appended after the item block. A class
    # constant, not a literal in render_instrument, because instrument_sha256
    # covers it: the digest must hash the same string the renderer emits.
    CONTRACT_REMINDER = "Respond with ONLY the JSON described above."

    def _require_schema(self):
        if self.schema is None:
            raise ValueError(f"Task '{self.task_name}' has no schema defined.")

    def instrument_text(self, system_prompt=None, examples=None):
        """The instrument as one string, byte-identical to what the API sees.

        Returns exactly the system prompt ``Task.run``/``Task.map`` send:
        the task's system_prompt, the output contract, the JSON Schema of the
        Pydantic model, and the few-shot examples as labelled
        ``Example N input:`` / ``Example N output:`` pairs.

        Byte-identity is the point — a second coder (another provider, a
        subagent, a human) administered this string received the same
        instrument as the API model, not a transcription of it.
        """
        self._require_schema()
        full_system, _ = _build_extract_prompt(
            "",
            self.schema,
            system_prompt=system_prompt or self.system_prompt,
            examples=examples if examples is not None else self.examples,
        )
        return full_system

    def instrument_sha256(self, system_prompt=None, examples=None):
        """SHA-256 of everything a second coder reads except the item itself.

        Covers ``instrument_text`` PLUS the item-block wrapper — the
        delimiters and the contract reminder ``render_instrument`` appends.
        An earlier digest covered the instrument alone, so the delimiters
        and reminder could be edited without the digest moving: the string
        handed to a second coder was not the string the digest described.

        Record this to freeze the scheme: two runs claiming to administer
        the same instrument should carry the same digest.

        Two omissions are deliberate, and a methods note needs both stated:
        the digest embeds the Pydantic-rendered JSON schema verbatim, so a
        pydantic upgrade can change it with no change to the scheme
        (conservative — a false alarm, never false reassurance); and it
        covers no administration parameters — model, temperature,
        max_tokens are the same digest. ``administration_record()`` carries
        those alongside it.
        """
        wrapper = (f"{self.ITEM_HEADER}\n\n{self.ITEM_FOOTER}\n\n"
                   f"{self.CONTRACT_REMINDER}")
        text = self.instrument_text(system_prompt=system_prompt,
                                    examples=examples) + "\n\n" + wrapper
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def administration_record(self, model=None):
        """The reproducibility receipt for one administration of this task.

        The instrument digest plus the parameters the digest deliberately
        omits — what a methods note needs so "the same instrument" and "the
        same administration" stay distinct claims. Pin the model explicitly
        when the run overrides the task default.
        """
        import pydantic
        return {
            "task": self.task_name,
            "schema": _schema_repr(self.schema),
            "instrument_sha256": self.instrument_sha256(),
            "model": model or getattr(self, "model", None) or DEFAULT_MODEL,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retries": self.retries,
            # The digest embeds model_json_schema() output, which can change
            # across pydantic versions with no change to the scheme — this
            # is the field that explains a digest mismatch between two runs
            # of the same instrument.
            "pydantic_version": pydantic.VERSION,
        }

    def render_instrument(self, item=None, system_prompt=None, examples=None,
                          digest=False):
        """Serialise the instrument (optionally with one item) to one string.

        Suitable for pasting to another provider, handing to a subagent, or
        printing for a human coder.

        With no ``item`` the return value is exactly ``instrument_text()``.
        With an ``item`` it appends the item between ``ITEM_HEADER`` /
        ``ITEM_FOOTER`` delimiters plus a one-line contract reminder — the
        API path sends the item as a separate user turn, so flattening a
        conversation into one string needs a delimiter the turn boundary
        provided for free.

        Args:
            item: The input text to annotate (a prompt string as ``run``
                would take it). None renders the bare instrument.
            system_prompt: Override the task's system_prompt.
            examples: Override the task's few-shot examples.
            digest: Append a provenance footer (task name, schema,
                instrument sha256). Provenance metadata, not part of the
                instrument — off by default so the default output stays
                byte-exact.

        Returns:
            str: The self-contained instrument.
        """
        if item is not None:
            for delim in (self.ITEM_HEADER, self.ITEM_FOOTER):
                if delim in item:
                    raise ValueError(
                        f"item contains the delimiter {delim!r}: the "
                        f"flattened form cannot represent it unambiguously "
                        f"(a second coder would see two item boundaries). "
                        f"Pre-process the item, or administer via the API "
                        f"path, which sends the item as its own turn."
                    )
        parts = [self.instrument_text(system_prompt=system_prompt,
                                      examples=examples)]
        if item is not None:
            parts.append(
                f"{self.ITEM_HEADER}\n{item}\n{self.ITEM_FOOTER}\n\n"
                f"{self.CONTRACT_REMINDER}"
            )
        if digest:
            parts.append(
                f"[instrument provenance — not part of the instrument]\n"
                f"task: {self.task_name}\n"
                f"schema: {_schema_repr(self.schema)}\n"
                f"instrument_sha256: "
                f"{self.instrument_sha256(system_prompt=system_prompt, examples=examples)}"
            )
        return "\n\n".join(parts)

    def parse_and_validate(self, text, strict=False, diagnosis=None):
        """Parse a hand-administered response into a validated schema object.

        Applies the same pipeline the tool-call path gets for free: markdown
        de-fencing, brace matching, json_repair fallback, the known
        output-envelope unwraps, then Pydantic validation.

        Args:
            text: The raw response text from whatever administered the
                instrument (another provider, a subagent, a typed-up human
                annotation).
            strict: Raise instead of returning None on failure. Use when you
                want the reason; the default swallows it into a log warning.
            diagnosis: Optional dict, populated on failure with 'error' and
                'partial_field' — the same diagnosis the API path computes
                (see _diagnose_partial_response). Pass it through to
                ``retry_prompt``, or the hand path can only ever send the
                generic reprompt: a coder that returned a bare list would be
                told "that was not valid JSON" about JSON that parsed fine.

        Returns:
            A validated Pydantic instance (or list thereof), or None if the
            text could not be parsed or did not validate.
        """
        self._require_schema()
        parsed = None
        try:
            parsed = _parse_json_response(text)
            return _validate_parsed(parsed, self.schema)
        except Exception as e:
            if diagnosis is not None:
                diagnosis["error"] = f"{type(e).__name__}: {e}"
                diagnosis["partial_field"] = _diagnose_partial_response(
                    parsed, self.schema)
            if strict:
                raise
            log.warning(
                "%s.parse_and_validate failed (%s): %s",
                self.task_name, e, (text or "")[:200],
            )
            return None

    def retry_prompt(self, item, partial_field=None):
        """The reprompt the API path uses after an invalid response.

        Administering the instrument by hand otherwise loses the retry half
        of the retry semantics. Send this in place of the item block's item
        when ``parse_and_validate`` returns None, passing the
        ``partial_field`` its ``diagnosis`` reported so the targeted branch
        — "you returned only the value of the X field" — is reachable by
        hand exactly as it is by API.
        """
        return _retry_prompt(item, partial_field)

    @property
    def results(self):
        """Iterate over cached (key_dict, parsed_result) pairs, latest per key.

        The stash is append-mode: a key rewritten (e.g. via force=True) keeps
        every version on disk. Under hashstash 1.0 items() already collapses
        to latest-per-key, but ask explicitly so .df cannot double-count
        rewritten keys if that default changes back; fall back to last-wins
        dedup for stash engines without the kwarg. Use `results_history` to
        see the versions this discards.

        Yields:
            tuple: (key_dict, validated pydantic object or list thereof)
        """
        try:
            pairs = self.stash.items(all_results=False)
        except TypeError:
            latest = {}
            for key, raw in self.stash.items():
                latest[json.dumps(key, sort_keys=True, default=str)] = (key, raw)
            pairs = latest.values()
        for key, raw in pairs:
            if not isinstance(raw, str):
                continue
            try:
                parsed = _parse_json_response(raw)
                result = _validate_parsed(parsed, self.schema)
                yield key, result
            except Exception:
                continue

    @property
    def results_history(self):
        """Iterate (key_dict, [result, ...]) — every retained version per key.

        Append-mode keeps each rewrite of a key rather than overwriting, so a
        key re-run with force=True holds one entry per run, oldest first.
        `results` and `df` show only the latest; this exposes the rest.

        The use case is measuring a model's own variance: run the same items
        N times with force=True, then read the versions back. Without force=
        a repeat call is a cache hit, so identical output across two ordinary
        runs demonstrates caching, not determinism — temperature=0 is not
        evidence of a stable annotation until it has been forced.

        Keys with only one version are included (as a 1-element list), so
        `{k: v for k, v in task.results_history if len(v) > 1}` isolates the
        items that were actually re-run.

        Yields:
            tuple: (key_dict, list of validated pydantic objects)
        """
        for key in self.stash.keys():
            try:
                raws = self.stash.get_all(key)
            except (AttributeError, TypeError):
                raws = [self.stash[key]]
            if not isinstance(raws, list):
                raws = [raws]
            versions = []
            for raw in raws:
                if not isinstance(raw, str):
                    continue
                try:
                    versions.append(
                        _validate_parsed(_parse_json_response(raw), self.schema))
                except Exception:
                    continue
            if versions:
                yield key, versions

    @property
    def df(self):
        """Build a DataFrame from all cached results.

        For list[Model] schemas, each item becomes its own row.
        Key metadata (model, prompt snippet, temperature) and user-defined
        metadata are included as columns.
        """
        import pandas as pd
        rows = []
        is_list, item_schema = _unwrap_schema(self.schema)
        for key, result in self.results:
            meta = {}
            if isinstance(key, dict):
                meta["model"] = key.get("model", "")
                meta["temperature"] = key.get("temperature", "")
                prompt = key.get("prompt", "")
                meta["prompt"] = prompt[:200] if isinstance(prompt, str) else str(prompt)[:200]
                # Include user-defined metadata
                user_meta = key.get("metadata")
                if isinstance(user_meta, dict):
                    for k, v in user_meta.items():
                        meta[f"meta_{k}"] = v

            items = result if is_list else [result]
            for item in items:
                row = {**meta, **item.model_dump()}
                rows.append(row)

        return pd.DataFrame(rows)

    def annotate(self, port=8989, annotator='default', host='127.0.0.1'):
        """Launch a web app for human annotation of this task's cached items.

        The app generates form fields from the Pydantic schema, shows the
        LLM's annotation alongside for comparison, and saves human annotations
        to a JSONL file per annotator. A /compare page shows inter-annotator
        agreement statistics.

        Args:
            port: Port to serve on.
            annotator: Annotator ID (each gets their own JSONL file).
            host: Host to bind to.
        """
        from .annotate import run_annotator
        run_annotator(self, port=port, annotator=annotator, host=host)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.task_name!r}, schema={_schema_repr(self.schema)})"


class SequentialTask(Task):
    """Base class for tasks that process a text chunk-by-chunk with
    feedforward state (summaries, character registers, etc.).

    Unlike Task.run() which handles a single prompt, SequentialTask.run()
    processes a full text by splitting it into chunks of passages, maintaining
    rolling state across chunks, and aggregating the results.

    Subclass and implement:
        - build_state(): return initial state dict
        - format_context(state): format state for the prompt
        - parse_response(raw): parse LLM output into structured result
        - update_state(state, result, chunk_idx, start, end): update state from result
        - aggregate(all_results, state): combine chunk results into final output
    """

    chunk_size = 10
    max_tokens = 8192
    # Bump (e.g. to 'v2') when system_prompt/format_context change materially:
    # folds into the chunk cache key so stale generations aren't served.
    # None preserves legacy keys, so existing caches — including in-flight
    # batch runs — stay valid.
    prompt_version = None

    def _no_static_instrument(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} is a SequentialTask: each chunk's "
            f"prompt is built from rolling state (format_context) plus the "
            f"chunk's passages, and the model is called via generate(), not "
            f"the extract path. There is no single static instrument to "
            f"render — the inherited implementation would happily serialise "
            f"(and hash, and hand to a second coder) a schema-and-contract "
            f"string this task never sends."
        )

    # A SequentialTask with a schema attribute would otherwise inherit
    # instrument methods that FABRICATE: verified against a schema-carrying
    # subclass, instrument_text() returned the bare system prompt plus a
    # full extract-path contract block that generate() never administers,
    # and instrument_sha256() froze it. Refuse loudly instead.
    def instrument_text(self, system_prompt=None, examples=None):
        self._no_static_instrument()

    def instrument_sha256(self, system_prompt=None, examples=None):
        self._no_static_instrument()

    def render_instrument(self, item=None, system_prompt=None, examples=None,
                          digest=False):
        self._no_static_instrument()

    def administration_record(self, model=None):
        self._no_static_instrument()

    def build_state(self):
        """Initialize the rolling state. Override in subclasses."""
        return {}

    def format_context(self, state):
        """Format the rolling state as a prompt prefix. Override in subclasses."""
        raise NotImplementedError

    def format_passages(self, passages_df, start_idx):
        """Format a chunk of passages for the prompt."""
        parts = []
        for i, (_, row) in enumerate(passages_df.iterrows()):
            pnum = start_idx + i
            parts.append(f"--- P{pnum:03d} ({row['n_words']} words) ---")
            parts.append(row['text'])
            parts.append("")
        return '\n'.join(parts)

    def parse_response(self, raw):
        """Parse raw LLM output into a structured result dict. Override in subclasses."""
        import re
        json_text = raw.strip()
        if json_text.startswith('```'):
            json_text = re.sub(r'^```(?:json)?\s*', '', json_text)
            json_text = re.sub(r'\s*```\s*$', '', json_text)
        return json.loads(json_text)

    def update_state(self, state, result, chunk_idx, start, end):
        """Update rolling state from the chunk's output. Override in subclasses."""
        raise NotImplementedError

    def aggregate(self, all_results, state):
        """Combine all chunk results into a final output dict. Override in subclasses."""
        raise NotImplementedError

    @staticmethod
    def _load_passages(source, passage_size=500):
        """Load passages from a file path or list of strings.

        Returns:
            tuple: (pd.DataFrame with 'text' and 'n_words' columns, source_label)
        """
        import pandas as pd
        if isinstance(source, list):
            rows = [{'text': t, 'n_words': len(t.split()), 'seq': i}
                    for i, t in enumerate(source)]
            return pd.DataFrame(rows), 'list'

        if isinstance(source, str):
            import os
            if os.path.isfile(source):
                with open(source) as f:
                    full_text = f.read()
                words = full_text.split()
                passages = []
                for i in range(0, len(words), passage_size):
                    chunk = ' '.join(words[i:i + passage_size])
                    passages.append({
                        'text': chunk, 'n_words': len(words[i:i + passage_size]),
                        'seq': len(passages),
                    })
                return pd.DataFrame(passages), os.path.basename(source)

        raise ValueError(
            f"source must be a list of strings or a path to a .txt file, "
            f"got {type(source).__name__}: {str(source)[:80]}"
        )

    def run(self, source, model=None, chunk_size=None, limit_chunks=0,
            force=False, verbose=True, save=None, source_label=None,
            cache_key=None):
        """Process a full text chunk-by-chunk with feedforward state.

        Args:
            source: One of:
                - path to a .txt file (auto-chunked into ~500-word passages)
                - list of passage strings
            model: Override the default model.
            chunk_size: Override the default chunk size.
            limit_chunks: Stop after N chunks (0=all).
            force: Bypass cache.
            verbose: Print progress to stderr.
            save: Path to save JSON output (or True for auto-naming).
            source_label: Human-readable label for progress output.
            cache_key: Stable identifier for caching (e.g. text ID).
                If not provided, falls back to source_label.

        Returns:
            dict: Aggregated results from all chunks.
        """
        import sys
        import time

        chunk_size = chunk_size or self.chunk_size
        model = model or getattr(self, 'model', None) or DEFAULT_MODEL

        pdf, auto_label = self._load_passages(source)
        source_label = source_label or cache_key or auto_label
        cache_key = cache_key or source_label
        n_chunks = (len(pdf) + chunk_size - 1) // chunk_size
        if limit_chunks:
            n_chunks = min(n_chunks, limit_chunks)

        if verbose:
            print(f"Model: {model}", file=sys.stderr)
            print(f"Text: {source_label}", file=sys.stderr)
            print(f"Passages: {len(pdf)}, Chunk size: {chunk_size}, "
                  f"Chunks: {n_chunks}", file=sys.stderr)

        llm = self._get_llm(model)
        state = self.build_state()
        all_results = []

        t0 = time.time()
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(pdf))
            chunk_df = pdf.iloc[start:end]

            context = self.format_context(state)
            passages_text = self.format_passages(chunk_df, start)
            prompt = context + "\n\n" + f"PASSAGES:\n{passages_text}"

            chunk_cache_key = {
                'task': self.task_name, 'text_id': cache_key,
                'chunk': chunk_idx, 'model': model,
                'chunk_size': chunk_size,
            }
            if self.prompt_version is not None:
                chunk_cache_key['prompt_version'] = self.prompt_version

            try:
                raw = llm.generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    cache_key=chunk_cache_key,
                    force=force,
                )
            except Exception as e:
                if verbose:
                    print(f"  [Chunk {chunk_idx:02d}] FAILED: {e!s:.100s}",
                          file=sys.stderr)
                all_results.append(None)
                continue

            try:
                result = self.parse_response(raw)
            except Exception as e:
                if verbose:
                    print(f"  [Chunk {chunk_idx:02d}] PARSE FAILED: {e!s:.80s}",
                          file=sys.stderr)
                all_results.append(None)
                continue

            # A bad chunk result must not kill a multi-hour run: keep the
            # parsed result but carry the previous state forward.
            try:
                state = self.update_state(state, result, chunk_idx, start, end)
            except Exception as e:
                if verbose:
                    print(f"  [Chunk {chunk_idx:02d}] STATE UPDATE FAILED: "
                          f"{e!s:.80s}", file=sys.stderr)
            all_results.append(result)

            if verbose:
                elapsed = time.time() - t0
                self.log_chunk(chunk_idx, start, end, elapsed, state, result)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nDone: {n_chunks} chunks in {elapsed:.0f}s "
                  f"({elapsed/max(1,n_chunks):.1f}s/chunk)", file=sys.stderr)

        output = self.aggregate(all_results, state)
        output['metadata'] = {
            'source': source_label,
            'model': model,
            'schema_version': f'{self.task_name}_v1',
            'n_passages': len(pdf),
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'elapsed_seconds': elapsed,
        }

        if save:
            self._save_result(output, save, source_label, model)

        return output

    @staticmethod
    def model_slug(model):
        return model.split('/')[-1].lower().replace('.', '').replace(' ', '_')

    def _save_result(self, output, save, source_label, model):
        """Save aggregated result to JSON."""
        if save is True:
            m_slug = self.model_slug(model)
            save = self._resolve_save_path(source_label, m_slug)
        os.makedirs(os.path.dirname(save), exist_ok=True)
        with open(save, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        import sys
        print(f"Saved to {save}", file=sys.stderr)

    def _resolve_save_path(self, source_label, model_slug):
        source_slug = source_label.replace('/', '_').replace(' ', '_').strip('_')
        return os.path.normpath(os.path.join(
            STASH_PATH, '..', f'{self.task_name}_{source_slug}_{model_slug}.json',
        ))

    def log_chunk(self, chunk_idx, start, end, elapsed, state, result):
        """Print per-chunk progress. Override for custom logging."""
        import sys
        print(f"  [Chunk {chunk_idx:02d}] P{start:03d}-P{end-1:03d}  "
              f"{elapsed:6.1f}s", file=sys.stderr)


def _schema_repr(schema):
    if schema is None:
        return "None"
    origin = getattr(schema, "__origin__", None)
    if origin is list:
        return f"list[{schema.__args__[0].__name__}]"
    return schema.__name__
