"""Prototype: streaming social-network extraction from sequential passages.

Each passage gets the accumulated character register + a rolling narrative
summary (capped at ~200 words). The LLM returns characters/relations active
in that passage plus an updated summary.

Usage:
    uv run python scripts/prototype_social_network.py
    uv run python scripts/prototype_social_network.py --model sonnet
"""

import argparse
import json
import sys
import time

import lltk
import pandas as pd
from pydantic import BaseModel, Field

from largeliterarymodels.llm import LLM

sys.stdout.reconfigure(line_buffering=True)

TEXT_ID = '_chadwyck/Eighteenth-Century_Fiction/haywood.13'

MODEL_TABLE = {
    'qwen': 'lmstudio/qwen3.5-35b-a3b',
    'gemma': 'lmstudio/gemma-4-31b-it',
    'sonnet': 'claude-sonnet-4-6',
}

SYSTEM_PROMPT = """\
You are annotating a novel passage by passage in sequence. You will receive:
1. A character register (all characters identified so far)
2. A narrative summary so far (what has happened in the story)
3. The next passage of text

Your job:
- Identify any NEW characters introduced in this passage (not already in the register)
- List which known characters are ACTIVE (mentioned or relevant) in this passage
- List social RELATIONS visible in this passage (kinship, marriage, courtship, \
service, friendship, enmity, alliance, economic, or other)
- Update the narrative summary to incorporate this passage's events

Rules:
- Only report relations that have textual evidence in THIS passage
- The narrative summary must be under 200 words and cover the whole story so far
- For new characters, give their name as used in the text; note aliases if apparent
- If a character is referred to by description only ("a young lady"), still add them \
with that description as their name; update the name later if revealed
- Relations are directional: "a" acts toward "b" (e.g. a courts b, a serves b)
"""


class NewCharacter(BaseModel):
    name: str = Field(description="Name or identifying description")
    gender: str = Field(description="male, female, or unknown")
    social_class: str = Field(description="e.g. nobility, gentry, servant, clergy, unknown")
    notes: str = Field(default="", description="Aliases, brief identification")


class Relation(BaseModel):
    char_a: str = Field(description="Name of character A (as in register)")
    char_b: str = Field(description="Name of character B (as in register)")
    relation_type: str = Field(description="kinship, marriage, courtship, service, friendship, enmity, alliance, economic, other")
    detail: str = Field(default="", description="Brief specification, e.g. 'father-daughter', 'unrequited'")


class PassageSocialAnnotation(BaseModel):
    characters_active: list[str] = Field(description="Names of known characters active in this passage")
    new_characters: list[NewCharacter] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    summary_update: str = Field(description="Updated narrative summary (under 200 words, covers whole story so far)")


def format_prompt(passage_text, passage_num, total, character_register, summary):
    parts = []

    parts.append(f"PASSAGE {passage_num + 1} of {total}")
    parts.append("")

    if character_register:
        parts.append("CHARACTER REGISTER:")
        for c in character_register:
            line = f"- {c['name']} ({c['gender']}, {c['social_class']})"
            if c.get('notes'):
                line += f" — {c['notes']}"
            parts.append(line)
        parts.append("")

    if summary:
        parts.append(f"NARRATIVE SUMMARY SO FAR:\n{summary}")
        parts.append("")

    parts.append(f"PASSAGE TEXT:\n{passage_text}")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen', choices=list(MODEL_TABLE.keys()))
    parser.add_argument('--text-id', default=TEXT_ID)
    args = parser.parse_args()

    model = MODEL_TABLE[args.model]
    print(f"Model: {model}", file=sys.stderr)
    print(f"Text: {args.text_id}", file=sys.stderr)

    pdf = lltk.db.get_passages([args.text_id])
    pdf = pdf.sort_values('seq').reset_index(drop=True)
    print(f"Passages: {len(pdf)}", file=sys.stderr)

    character_register = []
    summary = ""
    all_relations = []
    llm = LLM(model=model, temperature=0.2, max_tokens=4096)

    t0 = time.time()
    for i, (_, row) in enumerate(pdf.iterrows()):
        prompt = format_prompt(
            row['text'], i, len(pdf),
            character_register, summary,
        )

        result = llm.extract(
            prompt=prompt,
            schema=PassageSocialAnnotation,
            system_prompt=SYSTEM_PROMPT,
        )

        if result is None:
            print(f"  [P{i:02d}] FAILED", file=sys.stderr)
            continue

        for nc in result.new_characters:
            character_register.append({
                'name': nc.name,
                'gender': nc.gender,
                'social_class': nc.social_class,
                'notes': nc.notes,
                'first_seen': i,
            })

        summary = result.summary_update

        for rel in result.relations:
            all_relations.append({
                'passage': i,
                'char_a': rel.char_a,
                'char_b': rel.char_b,
                'type': rel.relation_type,
                'detail': rel.detail,
            })

        elapsed = time.time() - t0
        active_str = ', '.join(result.characters_active[:5])
        new_str = ', '.join(nc.name for nc in result.new_characters)
        rel_str = '; '.join(f"{r.char_a}→{r.char_b}({r.relation_type})" for r in result.relations)

        print(f"  [P{i:02d}] {elapsed:5.1f}s  active=[{active_str}]  "
              f"new=[{new_str}]  rels=[{rel_str}]",
              file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {len(pdf)} passages in {elapsed:.0f}s ({elapsed/len(pdf):.1f}s/passage)",
          file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"FINAL CHARACTER REGISTER ({len(character_register)} characters):", file=sys.stderr)
    for c in character_register:
        print(f"  {c['name']:30s} {c['gender']:8s} {c['social_class']:15s} (first P{c['first_seen']:02d}) {c.get('notes','')}", file=sys.stderr)

    print(f"\nALL RELATIONS ({len(all_relations)} edges):", file=sys.stderr)
    for r in all_relations:
        print(f"  P{r['passage']:02d}: {r['char_a']:20s} → {r['char_b']:20s} [{r['type']:12s}] {r['detail']}", file=sys.stderr)

    print(f"\nFINAL SUMMARY:", file=sys.stderr)
    print(f"  {summary}", file=sys.stderr)

    output = {
        'text_id': args.text_id,
        'model': model,
        'n_passages': len(pdf),
        'elapsed_seconds': elapsed,
        'character_register': character_register,
        'relations': all_relations,
        'final_summary': summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
