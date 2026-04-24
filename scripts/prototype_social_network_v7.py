"""Prototype V7: chunked passages, summary chain feedback.

Each call processes 10 passages. The model receives:
- A flat character register (ID: name, gender, class)
- A chain of per-chunk 100-word summaries (using character IDs inline)
- 10 passages of raw text

No relations fed back — relations are output only.

Usage:
    uv run python scripts/prototype_social_network_v7.py --limit-chunks 3
    uv run python scripts/prototype_social_network_v7.py --model sonnet
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
1. A CHARACTER REGISTER (ID: name, gender, class — all characters so far)
2. STORY SO FAR — a chain of short summaries from previous chunks, using character IDs
3. The next 10 PASSAGES of text

Return valid JSON with these fields:

{
  "new_characters": [
    {"id": "C12", "name": "the Linen-Draper", "gender": "male", "class": "merchant",
     "notes": "Moll's second husband"}
  ],
  "relations": [
    {"a": "C01", "b": "C12", "type": "spouse_of", "passage": "P042",
     "detail": "married after short courtship"},
    {"a": "C05", "b": "C02", "type": "same_as", "passage": "P045",
     "detail": "the gentleman from Bath is revealed to be the same as C02"}
  ],
  "chunk_summary": "A 100-word summary of JUST these 10 passages. Use character IDs \
inline: e.g. 'Moll (C01) married the Draper (C12) but he squandered their fortune.'"
}

## Character rules

- Every PERSON mentioned gets an ID. Not objects or places — only people.
- Check the register before creating. Use existing IDs where possible.
- When unsure if two are the same person, create a new ID and emit same_as later.
- Next available ID is shown in the register header.

## Relation types (closed vocabulary)

Kinship: parent_of, child_of, sibling_of, spouse_of, married
Courtship: courted, courted_by, attracted_to, rejected
Service: serves, served_by, employed_by, patron_of
Social: friend_of, confidante_of, enemy_of, rival_of, allied_with, betrayed, deceived
Economic: indebted_to, inherited_from
Identity: same_as (two IDs are the same person)

Use ONLY these types. Tag each relation with the passage number where it's evidenced.

## chunk_summary rules

- Summarize ONLY these 10 passages, not the whole story.
- Around 100 words (80-120 is fine).
- Use character IDs inline: "Moll (C01) discovered that her husband (C14) was her brother."
- Focus on social events: who met, married, betrayed, served, deceived whom.

Return ONLY valid JSON. No commentary before or after.
"""


class CharacterRegister:
    def __init__(self):
        self.characters = {}
        self.next_id = 1
        self.merged = {}

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
            return "Next available ID: C01\n(No characters yet — this is the first chunk)"
        lines = [f"Next available ID: C{self.next_id:02d}"]
        for cid in sorted(self.characters.keys(), key=lambda x: int(x[1:])):
            c = self.characters[cid]
            line = f"{cid}: {c['name']} ({c['gender']}, {c['class']})"
            if c.get('aliases'):
                line += f" / {', '.join(c['aliases'])}"
            lines.append(line)
        return '\n'.join(lines)

    def all_as_list(self):
        return list(self.characters.values())


def format_story_so_far(summaries):
    if not summaries:
        return "(No previous chunks — this is the beginning of the novel)"
    parts = []
    for start_p, end_p, text in summaries:
        parts.append(f"[P{start_p:03d}-P{end_p:03d}] {text}")
    return '\n\n'.join(parts)


def format_passages(passages_df, start_idx):
    parts = []
    for i, (_, row) in enumerate(passages_df.iterrows()):
        pnum = start_idx + i
        parts.append(f"--- P{pnum:03d} ({row['n_words']} words) ---")
        parts.append(row['text'])
        parts.append("")
    return '\n'.join(parts)


def format_prompt(passages_text, register, summaries):
    parts = []
    parts.append(f"CHARACTER REGISTER:\n{register.format_for_prompt()}")
    parts.append("")
    parts.append(f"STORY SO FAR:\n{format_story_so_far(summaries)}")
    parts.append("")
    parts.append(f"PASSAGES:\n{passages_text}")
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen36', choices=list(MODEL_TABLE.keys()))
    parser.add_argument('--text-id', default='_chadwyck/Eighteenth-Century_Fiction/defoe.06')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--limit-chunks', type=int, default=0)
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
    summaries = []  # list of (start_p, end_p, summary_text)

    t0 = time.time()
    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(pdf))
        chunk_df = pdf.iloc[start:end]

        passages_text = format_passages(chunk_df, start)
        prompt = format_prompt(passages_text, register, summaries)

        cache_key = {'task': 'social_network_v7', 'text_id': args.text_id,
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

        json_text = raw.strip()
        if json_text.startswith('```'):
            json_text = re.sub(r'^```(?:json)?\s*', '', json_text)
            json_text = re.sub(r'\s*```\s*$', '', json_text)

        try:
            result = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"  [Chunk {chunk_idx}] JSON PARSE FAILED: {e!s:.80s}", file=sys.stderr)
            print(f"    Raw: {raw[:300]}", file=sys.stderr)
            continue

        # New characters
        for c in result.get('new_characters', []):
            if 'id' in c:
                register.add(c)

        # Relations
        new_rels = result.get('relations', [])
        for r in new_rels:
            if not r.get('a') or not r.get('b') or not r.get('type'):
                continue
            r['a'] = register.resolve_id(r['a'])
            r['b'] = register.resolve_id(r['b'])
            if r['type'] == 'same_as':
                register.apply_same_as(r['a'], r['b'])
            all_relations.append(r)

        # Chunk summary
        chunk_summary = result.get('chunk_summary', '')
        if chunk_summary:
            summaries.append((start, end - 1, chunk_summary))

        elapsed = time.time() - t0
        n_chars = len(register.characters)
        n_rels = len(all_relations)
        new_names = ', '.join(
            c.get('id', '?') + '=' + c.get('name', '?')
            for c in result.get('new_characters', [])[:4]
        )
        if len(result.get('new_characters', [])) > 4:
            new_names += f" (+{len(result['new_characters'])-4})"
        same_as_rels = [r for r in new_rels if r.get('type') == 'same_as']
        social_rels = [r for r in new_rels if r.get('type') != 'same_as'
                       and r.get('a') and r.get('b')]

        status = f"  [Chunk {chunk_idx:02d}] P{start:03d}-P{end-1:03d}  {elapsed:6.1f}s  {n_chars} chars, {n_rels} rels"
        if new_names:
            status += f"\n         new: {new_names}"
        if same_as_rels:
            merge_strs = [r['a'] + '=' + r['b'] for r in same_as_rels]
            status += f"\n         merges: {'; '.join(merge_strs)}"
        if social_rels:
            rel_strs = [r['a'] + ' ' + r['type'] + ' ' + r['b'] for r in social_rels[:5]]
            rel_preview = '; '.join(rel_strs)
            if len(social_rels) > 5:
                rel_preview += f" (+{len(social_rels)-5})"
            status += f"\n         rels: {rel_preview}"
        if chunk_summary:
            status += f"\n         summary: {chunk_summary[:160]}..."
        print(status, file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {n_chunks} chunks in {elapsed:.0f}s ({elapsed/n_chunks:.1f}s/chunk)",
          file=sys.stderr)

    # Final report
    n_chars = len(register.characters)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"CHARACTERS ({n_chars}):", file=sys.stderr)
    for c in sorted(register.all_as_list(), key=lambda x: int(x['id'][1:])):
        aliases = f" / {', '.join(c['aliases'])}" if c.get('aliases') else ""
        print(f"  {c['id']:5s} {c['name']:40s} {c['gender']:8s} {c['class']:12s}{aliases}",
              file=sys.stderr)

    from collections import Counter
    rel_types = Counter(r['type'] for r in all_relations)
    print(f"\nRELATIONS ({len(all_relations)}):", file=sys.stderr)
    for rtype, n in rel_types.most_common():
        print(f"  {rtype:25s}: {n}", file=sys.stderr)

    print(f"\nSTORY CHAIN ({len(summaries)} summaries):", file=sys.stderr)
    for start_p, end_p, text in summaries:
        print(f"  [P{start_p:03d}-P{end_p:03d}] {text[:120]}...", file=sys.stderr)

    output = {
        'text_id': args.text_id,
        'model': model,
        'n_passages': min(n_chunks * args.chunk_size, len(pdf)),
        'n_chunks': n_chunks,
        'elapsed_seconds': elapsed,
        'characters': register.all_as_list(),
        'relations': all_relations,
        'summaries': [{'start': s, 'end': e, 'text': t} for s, e, t in summaries],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
