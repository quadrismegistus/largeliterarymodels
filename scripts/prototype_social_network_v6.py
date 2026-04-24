"""Prototype V6: chunked passages (10 at a time), flat JSON output.

Each call processes 10 passages (~5K words). The model receives:
- A deduplicated character register (code-side merge of same_as)
- All accumulated relations
- Cumulative summary
- 10 passages of raw text

Output is flat JSON: characters, relations (including same_as), summary.

Usage:
    uv run python scripts/prototype_social_network_v6.py --limit-chunks 3
    uv run python scripts/prototype_social_network_v6.py --model sonnet
"""

import argparse
import json
import re
import sys
import time

import lltk

from largeliterarymodels.llm import LLM

sys.stdout.reconfigure(line_buffering=True)

MODEL_TABLE = {
    'qwen': 'lmstudio/qwen3.5-35b-a3b',
    'qwen36': 'lmstudio/qwen/qwen3.6-27b',
    'gemma': 'lmstudio/gemma-4-31b-it',
    'sonnet': 'claude-sonnet-4-6',
}

CHUNK_SIZE = 10
SUMMARY_WORD_LIMIT = 400

RELATION_TYPES = [
    'parent_of', 'child_of', 'sibling_of', 'spouse_of', 'married',
    'courted', 'courted_by', 'attracted_to', 'rejected',
    'serves', 'served_by', 'employed_by', 'patron_of',
    'friend_of', 'confidante_of', 'enemy_of', 'rival_of',
    'allied_with', 'betrayed', 'deceived',
    'indebted_to', 'inherited_from',
    'same_as',
]

SYSTEM_PROMPT = """\
You are extracting a social network from a novel, processing 10 passages at a time.

You receive:
1. A CHARACTER REGISTER (all characters identified so far, with IDs)
2. A RELATION LIST (all social relations found so far)
3. A NARRATIVE SUMMARY of the story so far
4. The next 10 PASSAGES of text (labelled with passage numbers)

Return valid JSON with these fields:

{
  "new_characters": [
    {"id": "C12", "name": "the Linen-Draper", "gender": "male", "class": "merchant",
     "notes": "Moll's second husband"}
  ],
  "relations": [
    {"a": "C01", "b": "C12", "type": "spouse_of", "passage": "P042",
     "detail": "married after short courtship"},
    {"a": "C12", "b": "C01", "type": "same_as", "passage": "P045",
     "detail": "the Draper is revealed to be the gentleman from Bath"}
  ],
  "summary": "Cumulative summary of the entire story so far, under 400 words."
}

## Character rules

- Every PERSON mentioned gets an ID. Servants, magistrates, unnamed ladies — everyone.
- NOT objects, places, or abstract concepts. Only people.
- Check the register before creating. If a character matches someone existing, use \
their ID. If unsure, create a new ID and emit a same_as relation later.
- Use the next available ID shown in the register header.

## Relation types (closed vocabulary)

Kinship: parent_of, child_of, sibling_of, spouse_of, married
Courtship: courted, courted_by, attracted_to, rejected
Service: serves, served_by, employed_by, patron_of
Social: friend_of, confidante_of, enemy_of, rival_of, allied_with, betrayed, deceived
Economic: indebted_to, inherited_from
Identity: same_as (two IDs are the same person)

Use ONLY these types. Tag each relation with the passage number where it's evidenced.

## Summary rules

The summary is CUMULATIVE — the entire story from the beginning through these passages. \
Not a summary of just these 10 passages. Under 400 words. Compress older events \
to make room for newer ones.

Return ONLY valid JSON. No commentary before or after.
"""


class CharacterRegister:
    def __init__(self):
        self.characters = {}  # id -> dict
        self.next_id = 1
        self.merged = {}  # removed_id -> kept_id

    def add(self, char_dict):
        cid = char_dict['id']
        num = int(cid[1:])
        if num >= self.next_id:
            self.next_id = num + 1
        self.characters[cid] = {
            'id': cid,
            'name': char_dict.get('name', '?'),
            'gender': char_dict.get('gender', 'unknown'),
            'class': char_dict.get('class', 'unknown'),
            'notes': char_dict.get('notes', ''),
            'aliases': [],
        }

    def apply_same_as(self, a_id, b_id):
        if a_id not in self.characters and b_id not in self.characters:
            return
        if a_id not in self.characters:
            a_id, b_id = b_id, a_id
        if b_id not in self.characters:
            self.merged[b_id] = a_id
            return
        # Keep the lower-numbered ID
        keep, remove = (a_id, b_id) if int(a_id[1:]) < int(b_id[1:]) else (b_id, a_id)
        removed = self.characters.pop(remove, None)
        if removed and keep in self.characters:
            keeper = self.characters[keep]
            if removed['name'] not in keeper.get('aliases', []):
                keeper.setdefault('aliases', []).append(removed['name'])
        self.merged[remove] = keep

    def resolve_id(self, cid):
        while cid in self.merged:
            cid = self.merged[cid]
        return cid

    def format_for_prompt(self):
        if not self.characters:
            return f"(No characters yet. Next available ID: C01)"
        lines = [f"Next available ID: C{self.next_id:02d}", ""]
        for cid in sorted(self.characters.keys(), key=lambda x: int(x[1:])):
            c = self.characters[cid]
            line = f"{cid}: {c['name']} ({c['gender']}, {c['class']})"
            if c.get('aliases'):
                line += f" — also: {', '.join(c['aliases'])}"
            if c.get('notes'):
                line += f" — {c['notes']}"
            lines.append(line)
        return '\n'.join(lines)

    def all_as_list(self):
        return list(self.characters.values())


def format_relations_for_prompt(all_relations):
    if not all_relations:
        return "(No relations yet)"
    lines = []
    for r in all_relations:
        line = f"{r['a']} {r['type']} {r['b']}"
        if r.get('detail'):
            line += f" — {r['detail']}"
        lines.append(line)
    return '\n'.join(lines)


def format_passages(passages_df, start_idx):
    parts = []
    for i, (_, row) in enumerate(passages_df.iterrows()):
        pnum = start_idx + i
        parts.append(f"--- PASSAGE P{pnum:03d} ({row['n_words']} words) ---")
        parts.append(row['text'])
        parts.append("")
    return '\n'.join(parts)


def truncate_summary(summary, limit=SUMMARY_WORD_LIMIT):
    words = summary.split()
    if len(words) <= limit:
        return summary
    return '... ' + ' '.join(words[-limit:])


def format_prompt(passages_text, register, relations, summary):
    parts = []
    parts.append(f"CHARACTER REGISTER:\n{register.format_for_prompt()}")
    parts.append("")
    parts.append(f"RELATIONS:\n{format_relations_for_prompt(relations)}")
    parts.append("")
    if summary:
        parts.append(f"NARRATIVE SUMMARY SO FAR:\n{summary}")
        parts.append("")
    parts.append(f"PASSAGES TO PROCESS:\n{passages_text}")
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen', choices=list(MODEL_TABLE.keys()))
    parser.add_argument('--text-id', default='_chadwyck/Eighteenth-Century_Fiction/defoe.06')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--limit-chunks', type=int, default=0,
                        help='Stop after N chunks (0=all)')
    args = parser.parse_args()

    model = MODEL_TABLE[args.model]
    print(f"Model: {model}", file=sys.stderr)
    print(f"Text: {args.text_id}", file=sys.stderr)

    pdf = lltk.db.get_passages([args.text_id])
    pdf = pdf.sort_values('seq').reset_index(drop=True)
    n_chunks = (len(pdf) + args.chunk_size - 1) // args.chunk_size
    if args.limit_chunks:
        n_chunks = min(n_chunks, args.limit_chunks)
    print(f"Passages: {len(pdf)}, Chunk size: {args.chunk_size}, Chunks: {n_chunks}",
          file=sys.stderr)

    llm = LLM(model=model, temperature=0.2, max_tokens=8192)
    register = CharacterRegister()
    all_relations = []
    summary = ""

    t0 = time.time()
    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(pdf))
        chunk_df = pdf.iloc[start:end]

        passages_text = format_passages(chunk_df, start)
        prompt = format_prompt(passages_text, register, all_relations, summary)

        cache_key = {'task': 'social_network_v6', 'text_id': args.text_id,
                     'chunk': chunk_idx, 'model': model, 'chunk_size': args.chunk_size}

        try:
            raw = llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                cache_key=cache_key,
            )
        except Exception as e:
            print(f"  [Chunk {chunk_idx}] FAILED: {e!s:.100s}", file=sys.stderr)
            continue

        if raw is None:
            print(f"  [Chunk {chunk_idx}] FAILED: None", file=sys.stderr)
            continue

        # Parse JSON from response (strip any markdown fencing)
        json_text = raw.strip()
        if json_text.startswith('```'):
            json_text = re.sub(r'^```(?:json)?\s*', '', json_text)
            json_text = re.sub(r'\s*```\s*$', '', json_text)

        try:
            result = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"  [Chunk {chunk_idx}] JSON PARSE FAILED: {e!s:.80s}", file=sys.stderr)
            print(f"    Raw: {raw[:200]}", file=sys.stderr)
            continue

        # Process new characters
        new_chars = result.get('new_characters', [])
        for c in new_chars:
            if 'id' in c:
                register.add(c)

        # Process relations
        new_rels = result.get('relations', [])
        valid_rels = []
        for r in new_rels:
            if not r.get('a') or not r.get('b') or not r.get('type'):
                continue
            r['a'] = register.resolve_id(r['a'])
            r['b'] = register.resolve_id(r['b'])
            if r['type'] == 'same_as':
                register.apply_same_as(r['a'], r['b'])
            valid_rels.append(r)
        all_relations.extend(valid_rels)

        # Update summary
        new_summary = result.get('summary', '')
        if new_summary:
            summary = truncate_summary(new_summary, SUMMARY_WORD_LIMIT)

        elapsed = time.time() - t0
        n_chars = len(register.characters)
        n_rels = len(all_relations)
        new_names = ', '.join(f"{c.get('id','?')}={c.get('name','?')}" for c in new_chars[:4])
        if len(new_chars) > 4:
            new_names += f" (+{len(new_chars)-4})"
        same_as = [r for r in new_rels if r.get('type') == 'same_as']
        social_rels = [r for r in valid_rels if r.get('type') != 'same_as']
        summary_preview = summary[:140].replace('\n', ' ')

        status = f"  [Chunk {chunk_idx:02d}] P{start:03d}-P{end-1:03d}  {elapsed:6.1f}s  {n_chars} chars, {n_rels} rels"
        if new_names:
            status += f"\n         new: {new_names}"
        if same_as:
            merge_strs = [r['a'] + '=' + r['b'] for r in same_as]
            status += f"\n         merges: {'; '.join(merge_strs)}"
        if social_rels:
            rel_strs = [r['a'] + ' ' + r['type'] + ' ' + r['b'] for r in social_rels[:5]]
            rel_preview = '; '.join(rel_strs)
            if len(social_rels) > 5:
                rel_preview += f" (+{len(social_rels)-5})"
            status += f"\n         rels: {rel_preview}"
        status += f"\n         summary: {summary_preview}..."
        print(status, file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {n_chunks} chunks in {elapsed:.0f}s ({elapsed/n_chunks:.1f}s/chunk)",
          file=sys.stderr)

    # Final report
    n_chars = len(register.characters)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"CHARACTERS ({n_chars}):", file=sys.stderr)
    for c in sorted(register.all_as_list(), key=lambda x: int(x['id'][1:])):
        aliases = f" (also: {', '.join(c['aliases'])})" if c.get('aliases') else ""
        print(f"  {c['id']:5s} {c['name']:40s} {c['gender']:8s} {c['class']:12s}{aliases}",
              file=sys.stderr)

    from collections import Counter
    rel_types = Counter(r['type'] for r in all_relations)
    print(f"\nRELATIONS ({len(all_relations)}):", file=sys.stderr)
    for rtype, n in rel_types.most_common():
        print(f"  {rtype:25s}: {n}", file=sys.stderr)

    print(f"\nFINAL SUMMARY ({len(summary.split())}w):", file=sys.stderr)
    print(f"  {summary[:400]}...", file=sys.stderr)

    # JSON output
    output = {
        'text_id': args.text_id,
        'model': model,
        'n_passages': min(n_chunks * args.chunk_size, len(pdf)),
        'n_chunks': n_chunks,
        'elapsed_seconds': elapsed,
        'characters': register.all_as_list(),
        'relations': all_relations,
        'final_summary': summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
