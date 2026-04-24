"""Prototype V3: triple-based social network extraction.

Output is a flat list of (subject, predicate, object) triples per passage.
No Pydantic schema — just plain text generation + parsing.

Usage:
    uv run python scripts/prototype_social_network_v3.py
    uv run python scripts/prototype_social_network_v3.py --model sonnet --limit 30
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
    'gemma': 'lmstudio/gemma-4-31b-it',
    'sonnet': 'claude-sonnet-4-6',
}

SUMMARY_WORD_LIMIT = 400

SYSTEM_PROMPT = """\
You are annotating a novel passage by passage. You receive:
1. All triples emitted so far (the knowledge graph built from previous passages)
2. A narrative summary of the story so far
3. The next passage of text

Emit new triples for THIS passage, then an updated cumulative summary.

## Triple format

Each triple is one line: (subject, predicate, object)

Subjects and objects are either character IDs (C01, C02...) or quoted strings.
Predicates come from this vocabulary:

Character identity:
  introduced_as — first description used in text: (C01, introduced_as, "the Nurse")
  also_known_as — alias or later name: (C01, also_known_as, "Mrs Betty")
  same_as — merge two IDs: (C05, same_as, C02)
  gender — male/female/unknown
  class — nobility/gentry/merchant/professional/servant/criminal/clergy/unknown

Relations (directional, from subject to object):
  parent_of, child_of, sibling_of, spouse_of, married — kinship
  courted_by, courted, attracted_to — courtship
  serves, served_by, employed_by — service
  friend_of, confidante_of — friendship
  enemy_of, rival_of, deceived, betrayed — enmity
  indebted_to, paid, inherited_from, traded_with — economic

Narrative:
  appeared_in — character active in this passage: (C01, appeared_in, P005)
  died_in, born_in, left_in, arrived_in — life events

You may use other predicates if none of the above fit, but prefer the vocabulary.

## Rules

- EVERY person mentioned in the passage gets a character ID. Servants, \
magistrates, unnamed ladies — everyone. Create liberally. A character mentioned \
once still gets an ID.
- Use the next available ID (check the existing triples to find the highest).
- If you realize two IDs are the same person, emit (Cnew, same_as, Cold).
- Emit appeared_in for every character active in the passage.
- The summary must be CUMULATIVE — the entire story so far, not just this passage. \
Under 400 words. Compress older events to make room.

## Output format

TRIPLES:
(C01, appeared_in, P003)
(C04, introduced_as, "the Elder Brother")
(C04, gender, male)
(C04, courted, C01)
...

SUMMARY:
The cumulative narrative summary here...
"""


def parse_triples(text):
    triples = []
    for line in text.split('\n'):
        line = line.strip()
        m = re.match(r'^\((.+?),\s*(.+?),\s*(.+?)\)\s*$', line)
        if m:
            triples.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return triples


def parse_output(text):
    triples_section = ''
    summary_section = ''

    if 'SUMMARY:' in text:
        parts = text.split('SUMMARY:', 1)
        triples_section = parts[0]
        summary_section = parts[1].strip()
    elif 'TRIPLES:' in text:
        triples_section = text
    else:
        triples_section = text

    if 'TRIPLES:' in triples_section:
        triples_section = triples_section.split('TRIPLES:', 1)[1]

    triples = parse_triples(triples_section)
    return triples, summary_section


def format_triples_for_prompt(all_triples, max_lines=300):
    if not all_triples:
        return "(No triples yet)"
    lines = [f"({s}, {p}, {o})" for s, p, o in all_triples]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        return f"(... {len(all_triples) - max_lines} earlier triples omitted ...)\n" + '\n'.join(lines)
    return '\n'.join(lines)


def truncate_summary(summary, limit=SUMMARY_WORD_LIMIT):
    words = summary.split()
    if len(words) <= limit:
        return summary
    return '... ' + ' '.join(words[-limit:])


def format_prompt(passage_text, passage_num, total, all_triples, summary):
    parts = [f"PASSAGE {passage_num + 1} of {total}", ""]

    parts.append(f"KNOWLEDGE GRAPH SO FAR:\n{format_triples_for_prompt(all_triples)}")
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
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    model = MODEL_TABLE[args.model]
    print(f"Model: {model}", file=sys.stderr)
    print(f"Text: {args.text_id}", file=sys.stderr)

    pdf = lltk.db.get_passages([args.text_id])
    pdf = pdf.sort_values('seq').reset_index(drop=True)
    if args.limit:
        pdf = pdf.head(args.limit)
    print(f"Passages: {len(pdf)}", file=sys.stderr)

    llm = LLM(model=model, temperature=0.2, max_tokens=4096)

    all_triples = []
    summary = ""

    t0 = time.time()
    for i, (_, row) in enumerate(pdf.iterrows()):
        prompt = format_prompt(row['text'], i, len(pdf), all_triples, summary)

        cache_key = {'task': 'social_network_v3_triples', 'text_id': args.text_id,
                     'seq': int(row['seq']), 'model': model}

        raw = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            cache_key=cache_key,
        )

        if raw is None:
            print(f"  [P{i:03d}] FAILED", file=sys.stderr)
            continue

        new_triples, new_summary = parse_output(raw)

        if new_summary:
            summary = truncate_summary(new_summary, SUMMARY_WORD_LIMIT)

        all_triples.extend(new_triples)

        elapsed = time.time() - t0

        # Display
        n_chars = len(set(s for s, p, o in all_triples if re.match(r'^C\d+$', s)))
        new_ids = [s for s, p, o in new_triples if p == 'introduced_as']
        merges = [(s, o) for s, p, o in new_triples if p == 'same_as']
        rels = [(s, p, o) for s, p, o in new_triples
                if p not in ('introduced_as', 'also_known_as', 'same_as',
                             'gender', 'class', 'appeared_in', 'born_in', 'died_in')]
        summary_preview = summary[:140].replace('\n', ' ')

        status = f"  [P{i:03d}] {elapsed:6.1f}s  graph={len(all_triples)} triples, {n_chars} chars"
        if new_ids:
            status += f"  NEW=[{', '.join(new_ids)}]"
        if merges:
            status += f"  MERGE=[{'; '.join(f'{s}={o}' for s,o in merges)}]"
        if rels:
            rel_str = '; '.join(f"{s} {p} {o}" for s, p, o in rels[:4])
            if len(rels) > 4:
                rel_str += f" (+{len(rels)-4} more)"
            status += f"\n         rels: {rel_str}"
        status += f"\n         summary: {summary_preview}..."
        print(status, file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {len(pdf)} passages in {elapsed:.0f}s ({elapsed/len(pdf):.1f}s/passage)",
          file=sys.stderr)

    # Final stats
    n_chars = len(set(s for s, p, o in all_triples if re.match(r'^C\d+$', s)))
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"GRAPH: {len(all_triples)} triples, {n_chars} characters", file=sys.stderr)

    # Character roster
    char_names = {}
    for s, p, o in all_triples:
        if p == 'introduced_as':
            char_names[s] = o
        elif p == 'also_known_as' and s in char_names:
            char_names[s] += f" / {o}"
    print(f"\nCHARACTERS:", file=sys.stderr)
    for cid in sorted(char_names.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999):
        print(f"  {cid:5s} {char_names[cid]}", file=sys.stderr)

    # Relation type counts
    from collections import Counter
    pred_counts = Counter(p for _, p, _ in all_triples)
    print(f"\nPREDICATE COUNTS:", file=sys.stderr)
    for pred, n in pred_counts.most_common(20):
        print(f"  {pred:25s}: {n}", file=sys.stderr)

    print(f"\nFINAL SUMMARY ({len(summary.split())}w):", file=sys.stderr)
    print(f"  {summary[:300]}...", file=sys.stderr)

    # JSON output
    output = {
        'text_id': args.text_id,
        'model': model,
        'n_passages': len(pdf),
        'elapsed_seconds': elapsed,
        'triples': [{'s': s, 'p': p, 'o': o} for s, p, o in all_triples],
        'final_summary': summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
