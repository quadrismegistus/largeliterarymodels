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
  On a batch-fallback item get() returns the RETRY's body; history() is
  required to see the batch body that failed.
* OPT-IN and write-path isolated: with raw_log unset, no sink is
  constructed, providers skip serialization entirely, and nothing in
  this module runs. A non-sidecar run takes no path this feature touches
  (an additive feature that shares a write path is additive only until
  it is not — malign-logits seat). Pinned by test.
* A sidecar write failure must never fail the run it documents: record()
  logs the error and returns. Same discipline as usage_log.

Stated limit, same as the batch ledger's: pairtree writes are atomic
renames with no fsync, so an OS crash (not a process crash) can lose a
page-cached body that record() counted as a success.
"""
import json
import logging
import os
import threading
import time
import uuid

log = logging.getLogger(__name__)

# Per-firing bound on retained dropped-key material: enough to attribute
# any plausible partial outage, small enough that a pathological one
# cannot balloon the receipt row.
_MAX_DROPPED_KEYS = 200


def _default_root():
    from .llm import STASH_PATH
    return os.path.join(os.path.dirname(STASH_PATH), "raw_responses")


def _canon(key):
    return json.dumps(key, sort_keys=True, default=str)


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
        # that only logged something is not. Drops are attributed to
        # KEYS, not just counted: a count from one run must never
        # "explain" another run's missing bodies.
        self._lock = threading.Lock()
        self.recorded = 0
        self.failed = 0
        self._errors = []
        self._dropped_keys = []
        # One firing = one process's use of this store. Real runs fire
        # many times — resumption is the normal case, not the exception
        # (malign-logits seat: their registration fired five times) — so
        # a run-level claim quantifies over firings, and per-firing
        # receipts must be RETAINED, not just correct. flush_receipt()
        # persists this firing's counters; certify() composes the claim.
        # The uuid guards against pid-reuse collisions; the stored pid
        # detects fork(): a child inherits the parent's counters and
        # firing id, and flushing them as its own double-counts the
        # parent's work — on pid change the child re-mints and resets.
        self._pid = os.getpid()
        self._firing = self._mint_firing()

    @staticmethod
    def _mint_firing():
        return (f"{os.getpid()}-{int(time.time() * 1e6)}"
                f"-{uuid.uuid4().hex[:8]}")

    _RECEIPTS_KEY = ("__receipts__",)

    def _check_fork_locked(self):
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self._firing = self._mint_firing()
            self.recorded = 0
            self.failed = 0
            self._errors = []
            self._dropped_keys = []

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
               provider=None, attempt=None):
        """Append one serialized response under the annotation's key.

        Never raises: the sidecar documents a run, so a failure to
        document must not become a failure of the run. But it COUNTS,
        and it attributes — the failed KEY goes into this firing's
        receipt, so certify() can distinguish this run's drops from
        another run's. Returns True if the entry landed.
        """
        if isinstance(key, tuple) and tuple(key) == self._RECEIPTS_KEY:
            raise ValueError("record() must not write under the reserved "
                             "receipts key")
        with self._lock:
            self._check_fork_locked()
        envelope = {
            "ts": time.time(),
            "transport": transport,
            "model": model,
            "provider": provider,
            "body": body,
        }
        if attempt is not None:
            # Retries are administered under a modified prompt the key
            # does not show; the attempt index makes history()
            # self-describing.
            envelope["attempt"] = attempt
        try:
            self.stash[key] = envelope
        except Exception as e:  # noqa: BLE001 — receipt-write discipline
            with self._lock:
                self.failed += 1
                failed_now = self.failed
                if len(self._errors) < 10:
                    self._errors.append(f"{type(e).__name__}: {e}")
                if len(self._dropped_keys) < _MAX_DROPPED_KEYS:
                    self._dropped_keys.append(_canon(key))
            log.error("raw_log: failed to record response (%s: %s) — the "
                      "run continues; only the sidecar entry is lost "
                      "(failure COUNTED: receipt()['failed'] is now %d)",
                      type(e).__name__, e, failed_now)
            # Persist the failure NOW, not at the batch boundary: a crash
            # later in the batch must not lose the row that says bodies
            # were dropped. Rate-limited so a full outage does not double
            # its own I/O per item. Correlated-failure caveat: if the
            # whole store is down this flush fails too (logged) — which
            # is why certify() treats unexplained absence as dropped
            # rather than trusting receipt presence.
            if failed_now <= 3 or failed_now % 25 == 0:
                self.flush_receipt()
            return False
        with self._lock:
            self.recorded += 1
        return True

    def flush_receipt(self):
        """Durably retain this firing's counters in the store itself.

        Called automatically when a failure is counted and at batch
        boundaries (extract_imap's and LLM.map's finally, batch
        collect's finally); safe to call repeatedly — snapshots append
        under one reserved key and receipts() keeps the latest per
        firing. Never raises. The lock is held across the write so
        concurrent flushes cannot land out of snapshot order.
        """
        with self._lock:
            self._check_fork_locked()
            snap = {"firing": self._firing, "ts": time.time(),
                    "recorded": self.recorded, "failed": self.failed,
                    "errors": list(self._errors),
                    "dropped_keys": list(self._dropped_keys),
                    "dropped_keys_truncated":
                        self.failed > len(self._dropped_keys)}
            try:
                self.stash[self._RECEIPTS_KEY] = snap
            except Exception as e:  # noqa: BLE001
                log.error("raw_log: could not retain firing receipt "
                          "(%s: %s)", type(e).__name__, e)

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

    def receipt(self):
        """This process's coverage: {'recorded': N, 'failed': M,
        'errors': [...], 'dropped_keys': [...]}.

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
                    "errors": list(self._errors),
                    "dropped_keys": list(self._dropped_keys)}

    def audit(self, keys, since=None):
        """Durable, post-hoc coverage over a SCOPED key set.

        The scoping is the point — over an unscoped historical stash,
        'missing' conflates 'the write failed' with 'the sidecar was
        off when this was annotated', which are different claims. Pass
        the keys of the run you are making the claim about (e.g. the
        annotation stash keys of one batch) and 'present == total' is
        a statement someone can run rather than infer.

        PRESENCE IS KEY-LEVEL BY DEFAULT, not administration-level: a
        key counts as present if ANY administration ever left a body —
        a forced rerun whose bodies were all dropped still reads
        present on the previous run's bodies. Pass `since=<run start
        timestamp>` to require the latest body to postdate it, which
        makes presence mean 'a body from THIS administration'.

        RESIDUAL AMBIGUITY, stated rather than discovered: even scoped,
        a missing key alone cannot say WHY it is missing — the sidecar
        write failed, or no body ever reached the sink (a warm cache
        hit made no call; a transport-level failure returned nothing;
        another process annotated it with the sidecar off). certify()
        separates the first cause from the rest via per-key drop
        attribution in the firing receipts.

        Returns {'total', 'present', 'missing': [keys],
        'degraded': [keys], 'corrupt': [keys]} — degraded means an
        envelope exists but its body is empty or the unserialisable
        fallback (evidence-free); corrupt means the entry cannot be
        read at all. One corrupt entry must not deny certification of
        the other 9,999.
        """
        keys = list(keys)
        missing, degraded, corrupt = [], [], []
        for k in keys:
            try:
                env = self.get(k)
            except Exception:  # noqa: BLE001 — decode/storage failure
                corrupt.append(k)
                continue
            if env is None or (since is not None
                               and env.get("ts", 0) < since):
                missing.append(k)
                continue
            body = env.get("body")
            if not body or (isinstance(body, dict)
                            and set(body) == {"unserialisable"}):
                degraded.append(k)
        present = len(keys) - len(missing) - len(degraded) - len(corrupt)
        return {"total": len(keys), "present": present,
                "missing": missing, "degraded": degraded,
                "corrupt": corrupt}

    def certify(self, keys, since=None):
        """The composed 'no bodies were dropped' claim, quantified
        correctly over firings — one call, so the correct statement is
        easier to make than the incorrect one.

        complete=True requires a USABLE body for every scoped key: not
        missing, not degraded (empty/unserialisable), not corrupt. It
        is RETENTION-INDEPENDENT — presence proves it, so lost receipts
        cannot undermine it. Note the default presence semantics are
        key-level (see audit()); pass `since=<run start>` for the
        administration-level claim, which is the honest form after a
        forced rerun.

        Absence is explained by ATTRIBUTION, never by arithmetic on
        global counts: a missing key counts as known_drops only if some
        retained firing receipt (or this process) recorded THAT KEY as
        dropped. A count from an unrelated run's outage can therefore
        never absolve this run's missing bodies — unaccounted is the
        alarm, and it must not disarm with store age. Treat unaccounted
        as dropped, never as clean: retention bounds what absence can
        be EXPLAINED, never what presence PROVES.

        KEY PROVENANCE IS THE CLAIM'S FOUNDATION: `keys` must come from
        a record INDEPENDENT of this sidecar — the annotation stash
        (the record of what was completed), the batch ledger, the input
        manifest. Never pass keys derived from the sidecar itself
        (RawLog.keys() or anything downstream of it): a dropped body
        takes its key with it and never appears in the denominator, so
        certify would be checking a set against itself and
        complete=True would be guaranteed by construction rather than
        earned — a guard taking its threshold from the artifact it
        guards (malign-logits seat). The tautology is exhibited,
        deliberately, in the test suite.
        """
        report = self.audit(keys, since=since)
        recs = self.receipts()
        dropped = set()
        truncated = False
        for r in recs:
            dropped.update(r.get("dropped_keys", ()))
            truncated = truncated or r.get("dropped_keys_truncated", False)
        # This firing's drops count even before any flush landed.
        mine = self.receipt()
        dropped.update(mine["dropped_keys"])
        truncated = truncated or mine["failed"] > len(mine["dropped_keys"])
        explained = [k for k in report["missing"] if _canon(k) in dropped]
        return dict(
            report,
            complete=not (report["missing"] or report["degraded"]
                          or report["corrupt"]),
            firings_retained=len(recs),
            known_drops=len(explained),
            unaccounted=len(report["missing"]) - len(explained),
            # Context, deliberately NOT netted against this run's
            # missing: attribution may be incomplete when a receipt hit
            # the per-firing dropped-key bound.
            drops_all_firings=sum(r.get("failed", 0) for r in recs),
            attribution_truncated=truncated,
        )

    def get(self, key):
        """Latest recorded envelope for an annotation key, or None.

        Raises on storage/decode corruption — audit() catches that
        per-key and reports it as 'corrupt' rather than aborting."""
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
        forced reruns, and transport changes all append. On a
        batch-fallback item this is the only way to see the failed
        batch body: get() returns the retry's."""
        try:
            return self.stash.get_all(key, all_results=True) or []
        except (KeyError, TypeError):
            return []
