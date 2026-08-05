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
        document must not become a failure of the run.
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
            log.error("raw_log: failed to record response (%s: %s) — the "
                      "run continues; only the sidecar entry is lost",
                      type(e).__name__, e)

    def get(self, key):
        """Latest recorded envelope for an annotation key, or None."""
        try:
            return self.stash[key]
        except KeyError:
            return None

    def history(self, key):
        """Every recorded envelope for a key, oldest first — retries,
        forced reruns, and transport changes all append."""
        try:
            return self.stash.get_all(key, all_results=True) or []
        except (KeyError, TypeError):
            return []
