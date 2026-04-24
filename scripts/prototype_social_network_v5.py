"""Prototype V5: refined two-tier triple extraction.

Fixes from V4:
- Skip paratextual passages (dedications, prefaces) or mark them
- Strict tier 1 filtering: only accept exact predicate matches
- Better duplicate prevention: show highest ID in prompt
- Clearer instructions about character vs non-character entities

Usage:
    uv run python scripts/prototype_social_network_v5.py --limit 30
    uv run python scripts/prototype_social_network_v5.py --model sonnet
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

TIER1_PREDICATES = frozenset({
    'introduced_as', 'also_known_as', 'same_as', 'gender', 'class',
    'parent_of', 'child_of', 'sibling_of', 'spouse_of', 'married',
    'courted', 'courted_by', 'attracted_to', 'rejected',
    'serves', 'served_by', 'employed_by', 'patron_of',
    'friend_of', 'confidante_of', 'enemy_of', 'rival_of',
    'allied_with', 'betrayed', 'deceived',
    'indebted_to', 'inherited_from',
    'appeared_in',
})

SYSTEM_PROMPT = """\
You are annotating a novel passage by passage. You receive:
1. The SOCIAL GRAPH so far (structural triples from previous passages)
2. A narrative summary of the story so far
3. The next passage of text

You emit two kinds of triples, then an updated summary.

## Tier 1: STRUCTURAL triples (social graph)

Format: (subject, predicate, object)
Subjects and objects are character IDs (C01, C02...) or quoted strings.

CLOSED predicate vocabulary — use ONLY these:

  introduced_as  — first appearance: (C01, introduced_as, "Mrs Betty")
  also_known_as  — later alias: (C01, also_known_as, "Moll Flanders")
  same_as        — merge two IDs: (C05, same_as, C02)
  gender         — male / female / unknown
  class          — nobility / gentry / merchant / professional / servant / criminal / clergy / unknown

  parent_of    child_of     sibling_of   spouse_of   married
  courted      courted_by   attracted_to rejected
  serves       served_by    employed_by  patron_of
  friend_of    confidante_of enemy_of    rival_of
  allied_with  betrayed     deceived
  indebted_to  inherited_from

  appeared_in  — character is active: (C01, appeared_in, P005)

NO OTHER predicates in Tier 1.

## Tier 2: DESCRIPTIVE triples (free-form)

Capture anything else interesting about the passage. Open vocabulary. \
Quads are fine: (C01, said_to, C02, "the quoted words").

Examples:
  (C02, age, "17-18")
  (C02, profession, "lace-mender")
  (C02, described_as, "a Woman debauch'd from her Youth")
  (C43, said_to, C44, "beauty is a portion")

## Character identity rules

- Characters are PEOPLE who act in the story. Not abstract concepts, \
not "the Reader," not "the Pen." If a passage is a preface addressed \
to readers, the characters are the author and any people mentioned, \
not the readers themselves.
- EVERY person who acts or is acted upon gets an ID. Minor characters \
(servants, magistrates, unnamed ladies) all get IDs.
- CHECK THE GRAPH before creating a new ID. The current highest ID is \
shown. If a character matches someone already in the graph, use their \
existing ID. Do NOT re-introduce a character who already has an ID.
- When in doubt whether two descriptions refer to the same person, \
create a new ID and merge later with same_as.
- Use appeared_in for every character active in the passage.

## Summary rules

summary_update is CUMULATIVE — the entire story from passage 1 to now. \
NOT a summary of just this passage. Incorporate new events into the \
existing summary. Under 400 words. Compress older events to make room.

## Output format

TIER 1:
(C01, appeared_in, P003)
(C04, introduced_as, "the Elder Brother")
(C04, gender, male)
(C04, class, gentry)
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
        m = re.match(r'^\((.+?),\s*(.+?),\s*(.+?)(?:,\s*(.+?))?\)\s*$', line)
        if m:
            s, p, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            ctx = m.group(4).strip() if m.group(4) else None
            triples.append((s, p, o, ctx))
    return triples


def parse_output(text):
    tier1_raw, tier2, summary = [], [], ''

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
            tier1_raw.extend(parse_triples(section))
        elif current == 'tier2':
            tier2.extend(parse_triples(section))
        elif current == 'summary':
            summary = section.strip()

    if not tier1_raw and not tier2 and not summary:
        all_triples = parse_triples(text)
        for t in all_triples:
            if t[1] in TIER1_PREDICATES:
                tier1_raw.append(t)
            else:
                tier2.append(t)
        for line in text.split('\n'):
            if len(line.strip()) > 100 and not line.strip().startswith('('):
                summary = line.strip()

    # Strict filter: only allow exact tier 1 predicates
    tier1 = [t for t in tier1_raw if t[1] in TIER1_PREDICATES]
    # Move rejected tier1 triples to tier2
    tier2.extend(t for t in tier1_raw if t[1] not in TIER1_PREDICATES)

    return tier1, tier2, summary


def get_highest_id(triples):
    max_id = 0
    for s, p, o, _ in triples:
        for token in (s, o):
            m = re.match(r'^C(\d+)$', str(token))
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id


def format_graph_for_prompt(tier1_triples):
    if not tier1_triples:
        return "(Empty graph — this is the first passage)\nNext available ID: C01"
    lines = []
    for s, p, o, ctx in tier1_triples:
        lines.append(f"({s}, {p}, {o})")
    highest = get_highest_id(tier1_triples)
    header = f"Next available ID: C{highest + 1:02d}\n"
    if len(lines) > 250:
        header += f"(showing last 250 of {len(lines)} triples)\n"
        lines = lines[-250:]
    return header + '\n'.join(lines)


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

        cache_key = {'task': 'social_network_v5', 'text_id': args.text_id,
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

        n_chars = len(set(s for s, p, o, _ in all_tier1 if re.match(r'^C\d+$', s)))
        new_ids = [(s, o) for s, p, o, _ in new_tier1 if p == 'introduced_as']
        merges = [(s, o) for s, p, o, _ in new_tier1 if p == 'same_as']
        struct_rels = [(s, p, o) for s, p, o, _ in new_tier1
                       if p not in ('introduced_as', 'also_known_as', 'same_as',
                                    'gender', 'class', 'appeared_in')]
        summary_preview = summary[:140].replace('\n', ' ')

        status = f"  [P{i:03d}] {elapsed:6.1f}s  graph={len(all_tier1)}t1+{len(all_tier2)}t2  {n_chars} chars"
        if new_ids:
            id_str = ', '.join(f"{s}={o}" for s, o in new_ids[:4])
            if len(new_ids) > 4:
                id_str += f" (+{len(new_ids)-4})"
            status += f"  NEW=[{id_str}]"
        if merges:
            status += f"  MERGE=[{'; '.join(f'{s}={o}' for s,o in merges)}]"
        if struct_rels:
            rel_str = '; '.join(f"{s} {p} {o}" for s, p, o in struct_rels[:3])
            if len(struct_rels) > 3:
                rel_str += f" (+{len(struct_rels)-3})"
            status += f"\n         graph: {rel_str}"
        if new_tier2[:2]:
            t2_preview = '; '.join(f"({s}, {p}, {o})" for s, p, o, _ in new_tier2[:2])
            status += f"\n         color: {t2_preview}"
        status += f"\n         summary: {summary_preview}..."
        print(status, file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {len(pdf)} passages in {elapsed:.0f}s ({elapsed/len(pdf):.1f}s/passage)",
          file=sys.stderr)

    n_chars = len(set(s for s, p, o, _ in all_tier1 if re.match(r'^C\d+$', s)))
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"GRAPH: {len(all_tier1)} structural + {len(all_tier2)} descriptive, {n_chars} characters",
          file=sys.stderr)

    char_names = {}
    for s, p, o, _ in all_tier1:
        if p == 'introduced_as':
            if s not in char_names:
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
    print(f"  {summary[:500]}...", file=sys.stderr)

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
