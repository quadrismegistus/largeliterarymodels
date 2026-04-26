"""Smoke test: GenreTask via claude-cli/sonnet.

Run outside Claude Code:
    python scripts/test_claude_cli_genre.py
"""
import time
from largeliterarymodels.tasks import GenreTask, format_text_for_classification

task = GenreTask()

tests = [
    dict(title="Pamela", author_norm="richardson", year=1740),
    dict(title="Robinson Crusoe", author_norm="defoe", year=1719),
    dict(title="The Castle of Otranto", author_norm="walpole", year=1764),
]

for t in tests:
    prompt = format_text_for_classification(**t)
    t0 = time.time()
    result = task.run(prompt, model="claude-cli/sonnet", force=True)
    elapsed = time.time() - t0
    print(f"{t['title']:30} → {result.genre:10} {result.genre_raw:30} "
          f"conf={result.confidence:.1f}  {elapsed:.1f}s")
