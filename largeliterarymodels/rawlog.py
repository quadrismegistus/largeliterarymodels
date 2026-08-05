"""Raw-response sidecar: the provider's serialized reply, kept whole.

The annotation stash stores only the answer text, and usage receipts
store only the fields the normalizers knew to extract when they were
written. Every provider-drift bug this package has shipped was diagnosed
from a raw response — and each time, the raw response had already been
discarded at parse time, so the diagnosis needed fresh probes. The
sidecar is insurance against the unanticipated field: thinking blocks,
safety ratings, fingerprints, whatever tomorrow's question needs.

Design, settled by the same arguments that placed the batch ledger:

* SIBLING root, never inside the annotation stash: raw bodies are
  provenance with a different lifetime from a cache — clearing or
  rebuilding the cache must not take the record of what the provider
  actually said. And values stay small: thinking-bearing bodies run
  3-20x the answer text.
* Keyed by the SAME dict as the annotation, so raw <-> annotation joins
  are a lookup, not a correlation. append_mode makes retries and forced
  reruns append versions — every attempt's body is kept, oldest first.
* OPT-IN and write-path isolated: with raw_log unset, no sink is
  constructed, providers skip serialization entirely, and nothing in
  this module runs. A non-sidecar run takes no path this feature touches
  (an additive feature that shares a write path is additive only until
  it is not — malign-logits seat). Pinned by test.
* A sidecar write failure must never fail the run it documents: record()
  logs the error and returns. Same discipline as usage_log.
"""
import logging
import os
import threading
import time

log = logging.getLogger(__name__)


def _default_root():
    from .llm import STASH_PATH
    return os.path.join(os.path.dirname(STASH_PATH), "raw_responses")


class RawLog:
    """Append-mode store of serialized provider responses, by stash key."""

    def __init__(self, root=None):
        from hashstash import HashStash
        self.root = root or _default_root()
        self.stash = HashStash(self.root, engine="pairtree",
                               append_mode=True)
        # Coverage counters (thread-safe: imap fans out). Never-fail
        # writes make the run safe but would make the sidecar's own
        # completeness unverifiable — a missing receipt is
        # indistinguishable from a call never made (malign-logits seat).
        # A sidecar that KNOWS how many it missed is still additive; one
        # that only logged something is not.
        self._lock = threading.Lock()
        self.recorded = 0
        self.failed = 0
        self._errors = []
        # One firing = one process's use of this store. Real runs fire
        # many times — resumption is the normal case, not the exception
        # (malign-logits seat: their registration fired five times) — so
        # a run-level claim quantifies over firings, and per-firing
        # receipts must be RETAINED, not just correct. flush_receipt()
        # persists this firing's counters; certify() composes the claim.
        self._firing = f"{os.getpid()}-{int(time.time() * 1e6)}"

    _RECEIPTS_KEY = ("__receipts__",)

    @classmethod
    def resolve(cls, value):
        """Normalise the raw_log argument: falsy -> None (off), True ->
        default root, str/path -> that root, RawLog -> itself."""
        if not value:
            return None
        if isinstance(value, cls):
            return value
        if value is True:
            return cls()
        return cls(root=str(value))

    def record(self, key, body, transport="sync", model=None,
               provider=None):
        """Append one serialized response under the annotation's key.

        Never raises: the sidecar documents a run, so a failure to
        document must not become a failure of the run. But it COUNTS:
        read receipt() before claiming this run's bodies are all here.
        Returns True if the entry landed.
        """
        try:
            self.stash[key] = {
                "ts": time.time(),
                "transport": transport,
                "model": model,
                "provider": provider,
                "body": body,
            }
        except Exception as e:  # noqa: BLE001 — receipt-write discipline
            with self._lock:
                self.failed += 1
                if len(self._errors) < 10:
                    self._errors.append(f"{type(e).__name__}: {e}")
            log.error("raw_log: failed to record response (%s: %s) — the "
                      "run continues; only the sidecar entry is lost "
                      "(failure COUNTED: receipt()['failed'] is now %d)",
                      type(e).__name__, e, self.failed)
            # Persist the failure count NOW, not at the batch boundary: a
            # crash later in the batch must not lose the one row that
            # says bodies were dropped. Correlated-failure caveat: if the
            # whole store is down this flush fails too (logged) — which
            # is why certify() treats unexplained absence as dropped
            # rather than trusting receipt presence.
            self.flush_receipt()
            return False
        with self._lock:
            self.recorded += 1
        return True

    def flush_receipt(self):
        """Durably retain this firing's counters in the store itself.

        Called automatically when a failure is counted and at batch
        boundaries (extract_imap's finally, batch collect); safe to call
        repeatedly — snapshots append under one reserved key and
        receipts() keeps the latest per firing. Never raises.
        """
        with self._lock:
            snap = {"firing": self._firing, "ts": time.time(),
                    "recorded": self.recorded, "failed": self.failed,
                    "errors": list(self._errors)}
        try:
            self.stash[self._RECEIPTS_KEY] = snap
        except Exception as e:  # noqa: BLE001
            log.error("raw_log: could not retain firing receipt (%s: %s)",
                      type(e).__name__, e)

    def receipts(self):
        """All retained firing receipts, latest snapshot per firing.

        The durable side of receipt(): one entry per process that ever
        wrote to this store and reached a flush point. A firing that
        died before any flush left no row — certify() accounts for that
        by never treating absence of receipts as absence of drops.
        """
        try:
            rows = self.stash.get_all(self._RECEIPTS_KEY,
                                      all_results=True) or []
        except (KeyError, TypeError):
            return []
        latest = {}
        for row in rows:
            if isinstance(row, dict) and row.get("firing"):
                latest[row["firing"]] = row
        return list(latest.values())

    def certify(self, keys):
        """The composed 'no bodies were dropped' claim, quantified
        correctly over firings — one call, so the correct statement is
        easier to make than the incorrect one.

        complete=True is RETENTION-INDEPENDENT: every scoped key has a
        body in the store, proven by presence, so lost receipts cannot
        undermine it. complete=False lists the missing; retained
        receipts explain at most known_drops of them, and the remainder
        is unaccounted — treat unaccounted as dropped, never as clean:
        receipt retention bounds what absence can be EXPLAINED, never
        what presence PROVES.
        """
        report = self.audit(keys)
        recs = self.receipts()
        known_drops = sum(r.get("failed", 0) for r in recs)
        return dict(report, complete=not report["missing"],
                    firings_retained=len(recs), known_drops=known_drops,
                    unaccounted=max(0, len(report["missing"]) - known_drops))

    def receipt(self):
        """This process's coverage: {'recorded': N, 'failed': M, ...}.

        'The sidecar has the bodies' is a claim about M being zero —
        state it from this, not from an absence of error lines.

        M counts SIDECAR faults only, by construction: record() is
        reached exclusively with a body in hand, after the provider
        returned — a call that failed at the transport never fires the
        sink, so it cannot inflate this count. (Note the converse
        bonus: the sink fires before parsing, so items whose extraction
        failed still leave their bodies here — the raw record of the
        failure.)
        """
        with self._lock:
            return {"recorded": self.recorded, "failed": self.failed,
                    "errors": list(self._errors)}

    def audit(self, keys):
        """Durable, post-hoc coverage over a SCOPED key set: which of
        these annotation keys have at least one sidecar body?

        The scoping is the point — over an unscoped historical stash,
        'missing' conflates 'the write failed' with 'the sidecar was
        off when this was annotated', which are different claims. Pass
        the keys of the run you are making the claim about (e.g. the
        annotation stash keys of one batch) and 'present == total' is
        a statement someone can run rather than infer.

        RESIDUAL AMBIGUITY, stated rather than discovered: even scoped,
        a missing key alone cannot say WHY it is missing — the sidecar
        write failed, or no body ever reached the sink (a warm cache
        hit made no call; a transport-level failure returned nothing;
        another process annotated it with the sidecar off). This audit
        is durable and cannot separate those. The IN-PROCESS receipt()
        can, exactly: its failed count contains only write faults (see
        receipt()), so the run-end claim has two parts — failed == 0
        says no body was dropped, and this audit says which scoped keys
        have bodies at all. State both; neither substitutes for the
        other.

        Returns {'total': N, 'present': n, 'missing': [keys...]}.
        """
        keys = list(keys)
        missing = [k for k in keys if self.get(k) is None]
        return {"total": len(keys), "present": len(keys) - len(missing),
                "missing": missing}

    def get(self, key):
        """Latest recorded envelope for an annotation key, or None."""
        try:
            return self.stash[key]
        except KeyError:
            return None

    def keys(self):
        """Annotation keys with bodies here — reserved bookkeeping keys
        (the firing receipts) excluded, so iterating the sidecar never
        hands a reader a row that is not a response envelope."""
        for k in self.stash.keys():
            if k != self._RECEIPTS_KEY:
                yield k

    def history(self, key):
        """Every recorded envelope for a key, oldest first — retries,
        forced reruns, and transport changes all append."""
        try:
            return self.stash.get_all(key, all_results=True) or []
        except (KeyError, TypeError):
            return []
