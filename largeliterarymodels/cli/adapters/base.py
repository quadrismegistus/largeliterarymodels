"""TaskAdapter protocol.

An adapter binds a *task family* (passage, work, word, character) to the
I/O shape that family needs: canonical smoke fixtures, input loading, and
prompt construction. Tasks themselves stay pure — adapters live here so
test-data + I/O don't leak into production Task classes.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskAdapter(Protocol):
    """Contract for a task-family adapter.

    Implementations live in `largeliterarymodels/cli/adapters/<family>.py`.
    """

    family: str

    def fixtures(self) -> list[dict]:
        """Return canonical smoke-test records. Each record must be
        directly consumable by `build_prompt`."""
        ...

    def build_prompt(self, record: dict) -> tuple[str, dict]:
        """Build (prompt_text, metadata_dict) for one record."""
        ...

    def load_input(self, source: str) -> list[dict]:
        """Load records from an input source (CSV path, etc.). Each record
        must be consumable by `build_prompt`. Deferred to M2; stubs may
        raise NotImplementedError."""
        ...
