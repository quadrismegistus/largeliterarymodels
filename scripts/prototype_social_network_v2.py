"""Prototype V2: streaming social-network extraction with stable IDs,
character merging, and active-window register pruning.

Usage:
    uv run python scripts/prototype_social_network_v2.py
    uv run python scripts/prototype_social_network_v2.py --text-id _chadwyck/Eighteenth-Century_Fiction/defoe.06
    uv run python scripts/prototype_social_network_v2.py --model sonnet
"""

import argparse
import json
import re
import sys
import time

import lltk
import pandas as pd
from pydantic import BaseModel, Field

from largeliterarymodels.llm import LLM

sys.stdout.reconfigure(line_buffering=True)

MODEL_TABLE = {
    'qwen': 'lmstudio/qwen3.5-35b-a3b',
    'gemma': 'lmstudio/gemma-4-31b-it',
    'sonnet': 'claude-sonnet-4-6',
}

SUMMARY_WORD_LIMIT = 400

SYSTEM_PROMPT = """\
You are annotating a novel passage by passage in sequence. You will receive:
1. The COMPLETE character register with stable IDs (every character seen so far)
2. A narrative summary so far
3. The next passage of text

Your job for each passage:
- List which characters (by ID) are ACTIVE in this passage
- Add NEW characters not yet in the register, with the next available ID
- MERGE characters if you realize two IDs refer to the same person
- List social RELATIONS visible in this passage
- Update the narrative summary (must stay under 400 words, covering the whole story)

CRITICAL — character identity:
- When in doubt, CREATE A NEW CHARACTER. Merging later is easy; splitting a \
wrongly-reused ID is impossible. Do NOT reuse an existing ID unless you are \
confident this is the same person.
- Different unnamed characters ("a lady", "my husband") at different points in \
the story are usually DIFFERENT PEOPLE. Give each a new ID.
- Only MERGE when the text makes the identity explicit (e.g. "Dalinda, the \
Shepherdess from the masquerade" confirms two entries are one person).
- Characters may be referred to by description, title, or role rather than name. \
Track them with whatever label the text uses.

Summary rules:
- summary_update is a CUMULATIVE summary of the ENTIRE STORY from the beginning \
through the current passage. It is NOT a summary of just this passage.
- Incorporate the new passage's events into the existing NARRATIVE SUMMARY SO FAR.
- Keep it UNDER 400 WORDS. Compress older events to make room for newer ones.
- Use character IDs (C01, C02...) when referring to characters.

Other rules:
- Use character IDs (C01, C02...) in all output fields, not names
- Only report relations with evidence in THIS passage
"""


class NewCharacter(BaseModel):
    id: str = Field(description="Next available ID, e.g. C07")
    name: str = Field(description="Name or identifying description as used in text")
    gender: str = Field(description="male, female, or unknown")
    social_class: str = Field(default="unknown", description="e.g. nobility, gentry, merchant, servant, criminal, unknown")
    notes: str = Field(default="", description="Aliases, brief identification")


class MergeCharacters(BaseModel):
    keep_id: str = Field(description="ID to keep (the primary)")
    remove_id: str = Field(default="", description="ID to merge into keep_id")
    reason: str = Field(default="", description="Why these are the same person")


class Relation(BaseModel):
    char_a: str = Field(description="Character ID (e.g. C01)")
    char_b: str = Field(description="Character ID (e.g. C03)")
    relation_type: str = Field(description="kinship, marriage, courtship, service, friendship, enmity, alliance, economic, other")
    detail: str = Field(default="", description="Brief specification")


class PassageSocialAnnotation(BaseModel):
    characters_active: list[str] = Field(description="IDs of characters active in this passage")
    new_characters: list[NewCharacter] = Field(default_factory=list)
    merge_characters: list[MergeCharacters] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    summary_update: str = Field(description="Updated narrative summary, UNDER 400 words")


class CharacterRegister:
    def __init__(self):
        self.characters = {}  # id -> dict
        self.next_id = 1

    def add(self, name, gender, social_class, notes, passage):
        cid = f"C{self.next_id:02d}"
        self.next_id += 1
        self.characters[cid] = {
            'id': cid, 'name': name, 'gender': gender,
            'social_class': social_class, 'notes': notes,
            'first_seen': passage, 'last_seen': passage,
            'aliases': [],
        }
        return cid

    def add_with_id(self, cid, name, gender, social_class, notes, passage):
        num = int(cid[1:])
        if num >= self.next_id:
            self.next_id = num + 1
        self.characters[cid] = {
            'id': cid, 'name': name, 'gender': gender,
            'social_class': social_class, 'notes': notes,
            'first_seen': passage, 'last_seen': passage,
            'aliases': [],
        }
        return cid

    def touch(self, cid, passage):
        if cid in self.characters:
            self.characters[cid]['last_seen'] = passage

    def merge(self, keep_id, remove_id, reason=""):
        if keep_id not in self.characters or remove_id not in self.characters:
            return
        removed = self.characters.pop(remove_id)
        keeper = self.characters[keep_id]
        keeper['aliases'].append(removed['name'])
        if removed.get('notes'):
            keeper['notes'] = (keeper.get('notes', '') + '; ' + removed['notes']).strip('; ')
        keeper['first_seen'] = min(keeper['first_seen'], removed['first_seen'])
        keeper['last_seen'] = max(keeper['last_seen'], removed['last_seen'])

    def format_register(self):
        if not self.characters:
            return "(No characters registered yet)"
        lines = []
        for cid, c in sorted(self.characters.items()):
            line = f"{cid}: {c['name']} ({c['gender']}, {c['social_class']})"
            if c['aliases']:
                line += f" — also: {', '.join(c['aliases'])}"
            if c.get('notes'):
                line += f" — {c['notes']}"
            lines.append(line)
        return '\n'.join(lines)

    def all_as_list(self):
        return list(self.characters.values())


def truncate_summary(summary, limit=SUMMARY_WORD_LIMIT):
    words = summary.split()
    if len(words) <= limit:
        return summary
    # keep last N words (most recent events are most important)
    return '... ' + ' '.join(words[-limit:])


def format_prompt(passage_text, passage_num, total, register, summary):
    parts = [f"PASSAGE {passage_num + 1} of {total}", ""]

    reg_text = register.format_register()
    parts.append(f"CHARACTER REGISTER (complete):\n{reg_text}")
    parts.append("")

    if summary:
        parts.append(f"NARRATIVE SUMMARY SO FAR:\n{summary}")
        parts.append("")

    parts.append(f"PASSAGE TEXT:\n{passage_text}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen', choices=list(MODEL_TABLE.keys()))
    parser.add_argument('--text-id', default='_chadwyck/Eighteenth-Century_Fiction/defoe.06')
    parser.add_argument('--limit', type=int, default=0, help='Stop after N passages (0=all)')
    args = parser.parse_args()

    model = MODEL_TABLE[args.model]
    print(f"Model: {model}", file=sys.stderr)
    print(f"Text: {args.text_id}", file=sys.stderr)

    pdf = lltk.db.get_passages([args.text_id])
    pdf = pdf.sort_values('seq').reset_index(drop=True)
    if args.limit:
        pdf = pdf.head(args.limit)
    print(f"Passages: {len(pdf)}", file=sys.stderr)

    register = CharacterRegister()
    summary = ""
    all_relations = []
    all_merges = []

    llm = LLM(model=model, temperature=0.2, max_tokens=4096)

    t0 = time.time()
    for i, (_, row) in enumerate(pdf.iterrows()):
        prompt = format_prompt(
            row['text'], i, len(pdf),
            register, summary,
        )

        try:
            result = llm.extract(
                prompt=prompt,
                schema=PassageSocialAnnotation,
                system_prompt=SYSTEM_PROMPT,
                retries=4,
                cache_key={'task': 'social_network_v3', 'text_id': args.text_id,
                           'seq': int(row['seq']), 'model': model},
            )
        except (ValueError, Exception) as e:
            print(f"  [P{i:03d}] FAILED: {e!s:.80s}", file=sys.stderr)
            continue

        if result is None:
            print(f"  [P{i:03d}] FAILED: None result", file=sys.stderr)
            continue

        # Apply merges first (skip malformed ones)
        for m in result.merge_characters:
            if not m.keep_id or not m.remove_id:
                continue
            register.merge(m.keep_id, m.remove_id, m.reason)
            all_merges.append({
                'passage': i,
                'keep_id': m.keep_id,
                'remove_id': m.remove_id,
                'reason': m.reason,
            })

        # Add new characters
        for nc in result.new_characters:
            register.add_with_id(nc.id, nc.name, nc.gender, nc.social_class, nc.notes, i)

        # Touch active characters
        for cid in result.characters_active:
            register.touch(cid, i)

        # Update summary with truncation
        summary = truncate_summary(result.summary_update, SUMMARY_WORD_LIMIT)

        # Collect relations
        for rel in result.relations:
            all_relations.append({
                'passage': i,
                'char_a': rel.char_a,
                'char_b': rel.char_b,
                'type': rel.relation_type,
                'detail': rel.detail,
            })

        elapsed = time.time() - t0
        n_total = len(register.characters)
        active_ids = ', '.join(result.characters_active[:6])
        new_str = ', '.join(f"{nc.id}={nc.name}" for nc in result.new_characters)
        merge_str = ', '.join(f"{m.remove_id}→{m.keep_id}" for m in result.merge_characters)
        n_rels = len(result.relations)
        summary_words = len(result.summary_update.split())

        # Relation details for this passage
        rel_str = '; '.join(
            f"{r.char_a}→{r.char_b}({r.relation_type})" for r in result.relations
        )
        # Summary preview (first 120 chars)
        summary_preview = result.summary_update[:120].replace('\n', ' ')

        status = f"  [P{i:03d}] {elapsed:6.1f}s  chars={n_total}  active=[{active_ids}]"
        if new_str:
            status += f"  NEW=[{new_str}]"
        if merge_str:
            status += f"  MERGE=[{merge_str}]"
        status += f"\n         rels: {rel_str}" if rel_str else ""
        status += f"\n         summary ({summary_words}w): {summary_preview}..."
        print(status, file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {len(pdf)} passages in {elapsed:.0f}s ({elapsed/len(pdf):.1f}s/passage)",
          file=sys.stderr)

    # Final report
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"FINAL CHARACTER REGISTER ({len(register.characters)} characters):", file=sys.stderr)
    for c in sorted(register.all_as_list(), key=lambda x: x['first_seen']):
        aliases = f" (also: {', '.join(c['aliases'])})" if c['aliases'] else ""
        print(f"  {c['id']:5s} {c['name']:35s} {c['gender']:8s} {c['social_class']:15s} "
              f"P{c['first_seen']:03d}-P{c['last_seen']:03d}{aliases}",
              file=sys.stderr)

    if all_merges:
        print(f"\nMERGES ({len(all_merges)}):", file=sys.stderr)
        for m in all_merges:
            print(f"  P{m['passage']:03d}: {m['remove_id']} → {m['keep_id']} ({m['reason']})", file=sys.stderr)

    print(f"\nALL RELATIONS ({len(all_relations)} edges):", file=sys.stderr)
    # summarize by type
    from collections import Counter
    type_counts = Counter(r['type'] for r in all_relations)
    for t, n in type_counts.most_common():
        print(f"  {t:15s}: {n}", file=sys.stderr)

    print(f"\nFINAL SUMMARY ({len(summary.split())} words):", file=sys.stderr)
    print(f"  {summary[:500]}...", file=sys.stderr)

    # JSON output
    output = {
        'text_id': args.text_id,
        'model': model,
        'n_passages': len(pdf),
        'elapsed_seconds': elapsed,
        'character_register': register.all_as_list(),
        'merges': all_merges,
        'relations': all_relations,
        'final_summary': summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
