"""What a Task and a Survey share: the administration record-keeping.

Two things can be administered to a corpus in this package. A `Task` sends a
system prompt built from a Pydantic schema and parses text back. A `Survey`
sends typed questions and gets typed answers back. They have almost nothing
in common at the instrument — one has prompt bytes and a schema, the other
has a question set and no prompt at all — and everything in common at the
bookkeeping: where results are stashed, how tokens are counted, where the
raw bodies go, and how a durable receipt gets written.

This base is that bookkeeping and nothing else. It was deliberately
extracted AFTER both subclasses existed rather than designed up front,
because a base class derived from one example is a guess about the second.

What is NOT here, and why:

- `run`/`imap`/`map`. The signatures differ (prompts and images against
  state and questions) and the orchestration — threads, cache reads,
  retries, fail_fast — already lives in `LLM`, not here. Only the
  usage-receipt wrapper around it is shared, and that is `_with_usage_log`.
- `results`/`results_history`/`df`. Both subclasses have them and neither
  implementation is portable: a Task parses text against a schema, a Survey
  parses a JSON body into typed answers.
- `temperature`, `max_tokens`, `sampling`, `schema`, `system_prompt`,
  `examples`. All text-path. A System One model has no decoder parameters
  at all, and inheriting a `temperature` attribute it can never honour is
  how a key ends up asserting a pin that never applied.
"""

import logging
import os

from hashstash import HashStash

from .llm import STASH_PATH, UsageTracker, append_usage_log

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


class _Administered:
    """Stash, usage, raw sidecar and receipts, shared by Task and Survey."""

    #: Stash subdirectory and usage-log filename. Defaults to the class name.
    name = None

    # Prompt-cache lifetime for the instrument where the provider has one:
    # None = the provider default, "1h" for long or resumed runs. Kept on the
    # base because it is passed through to LLM either way; a provider with no
    # cache tier (System One) ignores it, and `cached: null` in the pricing
    # table is what keeps the cost estimate honest there.
    cache_ttl = None

    # Opt-in durable usage receipts. The stash stores only the response body,
    # so token counts — and with them any post-hoc price estimate — die with
    # the process. usage_log=True appends one JSONL record per map() batch to
    # data/usage_logs/<name>.jsonl. Additive: no stash value or key is
    # touched, and older entries simply have no log rows.
    usage_log = False

    # Opt-in raw-response sidecar (rawlog.RawLog), keyed by the SAME key as
    # the annotation. Additive: off constructs no sink and touches no write
    # path.
    raw_log = False

    #: Attributes settable via __init__ kwargs even where the class does not
    #: declare them. Subclasses commonly set `model` as a class attribute,
    #: but the bases leave it to the LLM's DEFAULT_MODEL fallback.
    _DYNAMIC_ATTRS = {'model'}

    #: Named in the unknown-attribute warning, so the hint lists the
    #: attributes THIS subclass actually has rather than the other one's.
    _KNOWN_ATTRS_HINT = "name, model"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(type(self), k) and k not in self._DYNAMIC_ATTRS:
                import warnings
                warnings.warn(
                    f"{type(self).__name__}({k}=...): unknown attribute — "
                    f"possible typo? Known: {self._KNOWN_ATTRS_HINT}.",
                    stacklevel=2,
                )
            setattr(self, k, v)
        self._stash = None
        self._human_stashes = {}
        self._usage = None
        self._raw_log = None

    @property
    def task_name(self):
        return self.name or self.__class__.__name__

    @property
    def stash(self):
        if self._stash is None:
            stash_dir = os.path.join(STASH_PATH, self.task_name)
            self._stash = HashStash(stash_dir, engine="pairtree",
                                    append_mode=True)
        return self._stash

    @property
    def usage(self):
        """Token usage accumulated across this instrument's calls.

        A receipt rather than an assumption: a run whose prompt cache
        silently stopped working produces identical output at roughly ten
        times the input price. Counts only live provider calls — stash hits
        never reach a provider, so a fully-cached run reports zero, and
        `summary_line()` says which kind of zero it is.
        """
        if self._usage is None:
            self._usage = UsageTracker()
        return self._usage

    def _resolved_raw_log(self):
        """Resolve raw_log once, so every LLM this object builds shares one
        sidecar handle instead of re-opening the store."""
        if self._raw_log is None and self.raw_log:
            from .rawlog import RawLog
            self._raw_log = RawLog.resolve(self.raw_log)
        return self._raw_log

    @property
    def raw_sidecar(self):
        """The resolved RawLog store, or None when the sidecar is off.

        The public accessor: `.raw_log` is the *configuration* (bool, path or
        RawLog) and has no methods when set with True.
        """
        return self._resolved_raw_log()

    def certify_raw(self, keys=None, since=None):
        """Certify sidecar coverage against this object's OWN stash — the
        independent record certify() requires. Passing keys derived from the
        sidecar itself is the tautology its docstring warns about; sourcing
        the denominator from the annotation store makes the correct call the
        easy one.

        keys: annotation keys to scope to (default: every key in the stash —
        fine for an instrument-sized claim, too broad for a run-sized one;
        pass the run's keys for that).
        """
        sidecar = self._resolved_raw_log()
        if sidecar is None:
            raise ValueError(
                f"{type(self).__name__} {self.task_name!r} has no "
                f"raw-response sidecar (raw_log is falsy) — nothing to "
                f"certify.")
        if keys is None:
            keys = list(self.stash.keys())
        return sidecar.certify(keys, since=since)

    def _append_usage_log(self, llm, per_item, metadata_list, errors=None):
        """Append this batch's usage receipts as one JSONL record.

        The record format lives in llm.append_usage_log, where the calls are.
        """
        append_usage_log(self.task_name, llm.model, per_item, metadata_list,
                         errors, stash_path=STASH_PATH)

    def _with_usage_log(self, llm, metadata_list, errors, per_item_usage,
                        run):
        """Run `run(log_errors, log_items)` and write a receipt afterwards.

        A generator wrapper, so it serves imap directly and map through it.
        The receipt lands in a `finally` because imap is the resumable
        long-run path: a consumer that stops iterating early still made the
        calls, and a receipt that only survives a clean exit is a receipt
        that goes missing exactly when the run was interesting. A failure to
        write one logs and never fails the run it documents.
        """
        log_items = {}
        log_errors = errors if errors is not None else {}
        try:
            yield from run(log_errors, log_items)
        finally:
            try:
                self._append_usage_log(llm, log_items, metadata_list,
                                       log_errors)
            except Exception as e:  # noqa: BLE001 — a receipt must not fail
                log.error("usage_log: failed to append receipt (%s: %s)",
                          type(e).__name__, e)
            if per_item_usage is not None:
                _merge_per_item(per_item_usage, log_items)
