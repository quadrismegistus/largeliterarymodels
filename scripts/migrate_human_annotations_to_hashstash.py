"""One-time migration: legacy human-annotation JSONL files → JSONLHashStash.

Legacy format (2026-04-11 to 2026-04-21):
    data/stash/_human_annotations/<task>/<annotator>.jsonl
    each line: {"_annotation_key": "...", "_annotator": "...", <fields>...}

New format (2026-04-21+, via task.human_stash(annotator)):
    data/stash/_human_annotations/<task>/<annotator>/jsonl.hashstash.raw/data.jsonl
    each line: {"__key__": "<item_key>", <fields>..., "__written_at__": ...}

Idempotent: rescans on rerun. Already-migrated annotators (where the
annotator dir exists) are skipped. Legacy .jsonl files get renamed to
.jsonl.legacy once migrated.

Usage:
    python scripts/migrate_human_annotations_to_hashstash.py
    python scripts/migrate_human_annotations_to_hashstash.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

from hashstash import HashStash

ANNOTATIONS_ROOT = Path(
    '/Users/rj416/github/largeliterarymodels/data/stash/_human_annotations'
)


def migrate_one(jsonl_file: Path, dry_run: bool = False) -> int:
    task_dir = jsonl_file.parent
    annotator = jsonl_file.stem
    new_stash_dir = task_dir / annotator

    if new_stash_dir.exists() and new_stash_dir.is_dir():
        print(f"  SKIP  {task_dir.name}/{annotator} — new stash dir exists",
              file=sys.stderr)
        return 0

    entries = []
    with open(jsonl_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  SKIP line in {jsonl_file}: {e}", file=sys.stderr)

    if not entries:
        print(f"  EMPTY {task_dir.name}/{annotator} — nothing to migrate")
        return 0

    if dry_run:
        print(f"  WOULD migrate {task_dir.name}/{annotator} "
              f"({len(entries)} entries → new stash)")
        return len(entries)

    stash = HashStash(root_dir=str(new_stash_dir), engine='jsonl')
    n = 0
    for entry in entries:
        key = entry.pop('_annotation_key', None)
        entry.pop('_annotator', None)
        if not key:
            continue
        stash[key] = entry
        n += 1

    legacy = jsonl_file.with_suffix('.jsonl.legacy')
    jsonl_file.rename(legacy)
    print(f"  OK    {task_dir.name}/{annotator}: migrated {n} entries "
          f"(legacy file → {legacy.name})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would happen without writing')
    args = ap.parse_args()

    if not ANNOTATIONS_ROOT.exists():
        print(f"No annotations dir at {ANNOTATIONS_ROOT}")
        return

    total = 0
    for task_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        for jsonl in sorted(task_dir.glob('*.jsonl')):
            total += migrate_one(jsonl, dry_run=args.dry_run)

    verb = 'would migrate' if args.dry_run else 'migrated'
    print(f"\nDone: {verb} {total} entries total.")


if __name__ == '__main__':
    main()
