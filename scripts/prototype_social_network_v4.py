"""Prototype V4: two-tier triple extraction.

Tier 1 (structural): closed predicate vocabulary, fed back as the social graph.
Tier 2 (descriptive): open vocabulary, captured but not fed back.

Usage:
    uv run python scripts/prototype_social_network_v4.py --limit 30
    uv run python scripts/prototype_social_network_v4.py --model sonnet
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

TIER1_PREDICATES = {
    # identity
    'introduced_as', 'also_known_as', 'same_as', 'gender', 'class',
    # kinship
    'parent_of', 'child_of', 'sibling_of', 'spouse_of', 'married',
    # courtship
    'courted', 'courted_by', 'attracted_to', 'rejected',
    # service
    'serves', 'served_by', 'employed_by', 'patron_of',
    # social
    'friend_of', 'confidante_of', 'enemy_of', 'rival_of',
    'allied_with', 'betrayed', 'deceived',
    # economic
    'indebted_to', 'inherited_from',
    # presence
    'appeared_in',
}

SYSTEM_PROMPT = """\
You are annotating a novel passage by passage. You receive:
1. The SOCIAL GRAPH so far (structural triples from previous passages)
2. A narrative summary of the story so far
3. The next passage of text

You emit two kinds of triples, then an updated summary.

## Tier 1: STRUCTURAL triples (social graph)

These use a CLOSED predicate vocabulary. Format: (subject, predicate, object)
Subjects and objects are character IDs (C01, C02...) or quoted strings.

Identity predicates:
  introduced_as — (C01, introduced_as, "the Nurse")
  also_known_as — (C01, also_known_as, "Mrs Betty")
  same_as — merge IDs: (C05, same_as, C02)
  gender — (C01, gender, male)
  class — (C01, class, servant)  [nobility/gentry/merchant/professional/servant/criminal/clergy/unknown]

Relation predicates (between characters):
  parent_of, child_of, sibling_of, spouse_of, married
  courted, courted_by, attracted_to, rejected
  serves, served_by, employed_by, patron_of
  friend_of, confidante_of, enemy_of, rival_of
  allied_with, betrayed, deceived
  indebted_to, inherited_from

Presence:
  appeared_in — (C01, appeared_in, P005)

USE ONLY THESE PREDICATES for Tier 1. No others.

## Tier 2: DESCRIPTIVE triples (free-form)

Capture anything else interesting: traits, speech, actions, emotions, \
occupations, age, events. Use any predicates you like. Quads are fine \
for speech or directed actions: (C01, said_to, C02, "the quoted words").

Examples:
  (C02, age, "17-18")
  (C02, profession, "lace-mender")
  (C02, described_as, "a Woman debauch'd from her Youth")
  (C48, said_to, C49, "why exclaim so about Fortune?")
  (C02, learned, "French")
  (C02, stole, "a gold Watch")

## Rules

- EVERY person mentioned gets a character ID. Create liberally.
- Use the next available ID (check the graph for the highest).
- Emit appeared_in for every character active in the passage.
- The summary must be CUMULATIVE (entire story so far), under 400 words.

## Output format

TIER 1:
(C01, appeared_in, P003)
(C04, introduced_as, "the Elder Brother")
(C04, courted, C01)

TIER 2:
(C04, described_as, "young, gay, knew the Town")
(C04, said_to, C01, "you are very pretty")

SUMMARY:
The cumulative narrative summary here...
"""


def parse_triples(text):
    triples = []
    for line in text.split('\n'):
        line = line.strip()
        # Match 3 or 4 element tuples
        m = re.match(r'^\((.+?),\s*(.+?),\s*(.+?)(?:,\s*(.+?))?\)\s*$', line)
        if m:
            s, p, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            ctx = m.group(4).strip() if m.group(4) else None
            triples.append((s, p, o, ctx))
    return triples


def parse_output(text):
    tier1, tier2, summary = [], [], ''

    # Split sections
    sections = re.split(r'^(TIER 1|TIER 2|SUMMARY):?\s*$', text, flags=re.MULTILINE)

    current = None
    for section in sections:
        section_stripped = section.strip()
        if section_stripped == 'TIER 1':
            current = 'tier1'
            continue
        elif section_stripped == 'TIER 2':
            current = 'tier2'
            continue
        elif section_stripped == 'SUMMARY':
            current = 'summary'
            continue

        if current == 'tier1':
            tier1.extend(parse_triples(section))
        elif current == 'tier2':
            tier2.extend(parse_triples(section))
        elif current == 'summary':
            summary = section.strip()

    # If no sections found, try to parse everything as triples
    if not tier1 and not tier2 and not summary:
        all_triples = parse_triples(text)
        for t in all_triples:
            if t[1] in TIER1_PREDICATES:
                tier1.append(t)
            else:
                tier2.append(t)
        # Try to find summary at the end
        for line in text.split('\n'):
            if len(line.strip()) > 100 and not line.strip().startswith('('):
                summary = line.strip()

    return tier1, tier2, summary


def format_graph_for_prompt(tier1_triples, max_lines=200):
    if not tier1_triples:
        return "(Empty graph — this is the first passage)"
    lines = []
    for s, p, o, ctx in tier1_triples:
        lines.append(f"({s}, {p}, {o})")
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        return f"(... {len(tier1_triples) - max_lines} earlier triples omitted ...)\n" + '\n'.join(lines)
    return '\n'.join(lines)


def truncate_summary(summary, limit=SUMMARY_WORD_LIMIT):
    words = summary.split()
    if len(words) <= limit:
        return summary
    return '... ' + ' '.join(words[-limit:])


def format_prompt(passage_text, passage_num, total, tier1_triples, summary):
    parts = [f"PASSAGE {passage_num + 1} of {total}", ""]
    parts.append(f"SOCIAL GRAPH SO FAR:\n{format_graph_for_prompt(tier1_triples)}")
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

    all_tier1 = []
    all_tier2 = []
    summary = ""

    t0 = time.time()
    for i, (_, row) in enumerate(pdf.iterrows()):
        prompt = format_prompt(row['text'], i, len(pdf), all_tier1, summary)

        cache_key = {'task': 'social_network_v4', 'text_id': args.text_id,
                     'seq': int(row['seq']), 'model': model}

        try:
            raw = llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                cache_key=cache_key,
            )
        except Exception as e:
            print(f"  [P{i:03d}] FAILED: {e!s:.80s}", file=sys.stderr)
            continue

        if raw is None:
            print(f"  [P{i:03d}] FAILED: None", file=sys.stderr)
            continue

        new_tier1, new_tier2, new_summary = parse_output(raw)

        if new_summary:
            summary = truncate_summary(new_summary, SUMMARY_WORD_LIMIT)

        all_tier1.extend(new_tier1)
        all_tier2.extend(new_tier2)

        elapsed = time.time() - t0

        # Stats
        n_chars = len(set(s for s, p, o, _ in all_tier1 if re.match(r'^C\d+$', s)))
        new_ids = [(s, o) for s, p, o, _ in new_tier1 if p == 'introduced_as']
        merges = [(s, o) for s, p, o, _ in new_tier1 if p == 'same_as']
        struct_rels = [(s, p, o) for s, p, o, _ in new_tier1
                       if p not in ('introduced_as', 'also_known_as', 'same_as',
                                    'gender', 'class', 'appeared_in')]
        summary_preview = summary[:140].replace('\n', ' ')

        status = f"  [P{i:03d}] {elapsed:6.1f}s  graph={len(all_tier1)}t1+{len(all_tier2)}t2  {n_chars} chars"
        if new_ids:
            id_str = ', '.join(f"{s}={o}" for s, o in new_ids)
            status += f"  NEW=[{id_str}]"
        if merges:
            status += f"  MERGE=[{'; '.join(f'{s}={o}' for s,o in merges)}]"
        if struct_rels:
            rel_str = '; '.join(f"{s} {p} {o}" for s, p, o in struct_rels[:4])
            if len(struct_rels) > 4:
                rel_str += f" (+{len(struct_rels)-4})"
            status += f"\n         graph: {rel_str}"
        if new_tier2:
            t2_preview = '; '.join(
                f"{s} {p} {o}" for s, p, o, _ in new_tier2[:3]
            )
            if len(new_tier2) > 3:
                t2_preview += f" (+{len(new_tier2)-3})"
            status += f"\n         color: {t2_preview}"
        status += f"\n         summary: {summary_preview}..."
        print(status, file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {len(pdf)} passages in {elapsed:.0f}s ({elapsed/len(pdf):.1f}s/passage)",
          file=sys.stderr)

    # Final report
    n_chars = len(set(s for s, p, o, _ in all_tier1 if re.match(r'^C\d+$', s)))
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"GRAPH: {len(all_tier1)} structural + {len(all_tier2)} descriptive triples, {n_chars} characters",
          file=sys.stderr)

    char_names = {}
    for s, p, o, _ in all_tier1:
        if p == 'introduced_as':
            char_names[s] = o
        elif p == 'also_known_as' and s in char_names:
            char_names[s] += f" / {o}"
    print(f"\nCHARACTERS:", file=sys.stderr)
    for cid in sorted(char_names.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999):
        print(f"  {cid:5s} {char_names[cid]}", file=sys.stderr)

    from collections import Counter
    print(f"\nTIER 1 PREDICATES:", file=sys.stderr)
    for pred, n in Counter(p for _, p, _, _ in all_tier1).most_common():
        print(f"  {pred:25s}: {n}", file=sys.stderr)
    print(f"\nTIER 2 PREDICATES (top 20):", file=sys.stderr)
    for pred, n in Counter(p for _, p, _, _ in all_tier2).most_common(20):
        print(f"  {pred:25s}: {n}", file=sys.stderr)

    print(f"\nFINAL SUMMARY ({len(summary.split())}w):", file=sys.stderr)
    print(f"  {summary[:400]}...", file=sys.stderr)

    output = {
        'text_id': args.text_id,
        'model': model,
        'n_passages': len(pdf),
        'elapsed_seconds': elapsed,
        'tier1': [{'s': s, 'p': p, 'o': o, 'ctx': c} for s, p, o, c in all_tier1],
        'tier2': [{'s': s, 'p': p, 'o': o, 'ctx': c} for s, p, o, c in all_tier2],
        'final_summary': summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
