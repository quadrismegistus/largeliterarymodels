"""Export annotated passages as a nicely-formatted markdown file for reading.

Joins task cache (qwen PassageContentTask results) with the pilot manifest
and pulls passage text + genre_raw from lltk. Shuffles randomly and writes
to a single .md file for browsing.

Usage:
    python scripts/export_passage_markdown.py \\
        --out ~/Dropbox/Prof/Books/AbsLitHist/data/passage_annotations_preview.md \\
        --n 50
"""

import argparse
import os
import random
import sys

import lltk
import pandas as pd
from largeliterarymodels.tasks import PassageContentTask


SCENE_FLAGS = ['has_battle_or_violence', 'has_courtship', 'has_travel_or_journey',
               'has_domestic_routine', 'has_moral_reflection', 'has_sexual_encounter',
               'has_religious_content', 'has_deception_or_intrigue',
               'has_legal_or_financial']
SETTING_FLAGS = ['setting_domestic_interior', 'setting_grand_estate',
                 'setting_wilderness', 'setting_urban_street',
                 'setting_confined_space', 'setting_rural_outdoor']
CHARACTER_FLAGS = ['characters_noble_aristocratic', 'characters_gentry_or_middling',
                   'characters_laboring_or_servant', 'characters_clergy',
                   'characters_criminal_or_underclass', 'has_male_characters',
                   'has_female_characters', 'introduces_new_character']
FANTASTICAL_FLAGS = ['has_supernatural', 'has_ghost_or_haunting',
                     'has_prophecy_or_omen', 'has_monster_or_creature',
                     'has_dream_or_vision', 'has_allegorical_personification']
THREAT_FLAGS = ['threat_physical', 'threat_supernatural', 'threat_social_or_reputational']
EMOTION_FLAGS = ['emotion_extreme', 'emotion_restrained_or_ironic']
FORM_FLAGS = ['has_character_thoughts', 'has_free_indirect_or_monologue',
              'is_dramatized_scene', 'is_narrative_summary', 'is_epistolary',
              'contains_embedded_letter']

GROUPS = [
    ('Scene', SCENE_FLAGS),
    ('Setting', SETTING_FLAGS),
    ('Characters', CHARACTER_FLAGS),
    ('Fantastical', FANTASTICAL_FLAGS),
    ('Threat', THREAT_FLAGS),
    ('Emotion', EMOTION_FLAGS),
    ('Form/Interiority', FORM_FLAGS),
]


def get_genre_raw(ids):
    """Pull genre_raw annotations (most-recent per source) for each _id."""
    if not ids:
        return {}
    id_list = ", ".join(f"'{i}'" for i in ids)
    sql = f"""
        SELECT _id, source, argMax(value, annotated_at) AS genre_raw
        FROM lltk.annotations
        WHERE field='genre_raw' AND _id IN ({id_list})
        GROUP BY _id, source
    """
    try:
        df = lltk.db.query(sql)
    except Exception as e:
        print(f"  genre_raw lookup failed: {e}", file=sys.stderr)
        return {}
    out = {}
    for tid, g in df.groupby('_id'):
        parts = [f"{r['source']}: {r['genre_raw']}" for _, r in g.iterrows()
                 if r['genre_raw']]
        out[tid] = parts
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True,
                   help='Output markdown file path')
    p.add_argument('--n', type=int, default=50,
                   help='Number of passages to include (stratified across tag_label)')
    p.add_argument('--seed', type=int, default=7,
                   help='Shuffle seed')
    p.add_argument('--manifest', default='/Users/rj416/github/largeliterarymodels/data/pilot_2026-04_passage_content_sample.csv')
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest)
    label_by_id = dict(zip(manifest['_id'], manifest['tag_label']))
    title_by_id = dict(zip(manifest['_id'], manifest['title']))
    author_by_id = dict(zip(manifest['_id'], manifest['author']))
    year_by_id = dict(zip(manifest['_id'], manifest['year']))

    task = PassageContentTask()
    df = task.df
    qwen = df[df['model'].str.contains('qwen3.5', na=False)].copy()
    man_pair = set(zip(manifest['_id'], 'p500:' + manifest['seq'].astype(str)))
    in_pilot = qwen[[(r['meta__id'], r['meta_section_id']) in man_pair
                     for _, r in qwen.iterrows()]].copy()
    in_pilot['tag_label'] = in_pilot['meta__id'].map(label_by_id)
    print(f"joined: {len(in_pilot)} annotated passages", file=sys.stderr)

    # Stratified shuffle: ~equal counts per tag_label
    rng = random.Random(args.seed)
    buckets = {}
    for _, r in in_pilot.iterrows():
        buckets.setdefault(r['tag_label'], []).append(r.to_dict())
    per_bucket = max(1, args.n // max(1, len(buckets)))
    picked = []
    for label, rows in buckets.items():
        rng.shuffle(rows)
        picked.extend(rows[:per_bucket])
    rng.shuffle(picked)
    picked = picked[:args.n]
    print(f"picked: {len(picked)} passages ({per_bucket}/tag)", file=sys.stderr)

    # Pull passage text + genre_raw
    ids_needed = list({p['meta__id'] for p in picked})
    pdf = lltk.db.get_passages(ids_needed)
    text_by_pair = {(r['_id'], int(r['seq'])): r['text']
                    for _, r in pdf.iterrows()}
    genre_raw_by_id = get_genre_raw(ids_needed)

    # Write markdown
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(f"# Passage content annotations — preview\n\n")
        f.write(f"**Source**: `lltk.passages` (scheme=p500) annotated by `qwen3.5-35b-a3b` "
                f"via `PassageContentTask` (39 binary content flags).\n\n")
        f.write(f"**Sample**: {len(picked)} random passages across "
                f"romance / novel_only / contrast(amatory+picaresque) from the "
                f"[2026-04 pilot manifest]"
                f"(../../github/largeliterarymodels/data/pilot_2026-04_passage_content_sample.csv).\n\n")
        f.write(f"Read linearly to develop an intuitive feel for what PassageContentTask "
                f"picks up, misses, and over-reaches.\n\n")
        f.write("---\n\n")

        for i, row in enumerate(picked, 1):
            tid = row['meta__id']
            seq = int(row['meta_section_id'].replace('p500:', ''))
            title = title_by_id.get(tid, '') or '(untitled)'
            author = author_by_id.get(tid, '') or '(anon)'
            year = year_by_id.get(tid)
            year_s = str(int(year)) if pd.notna(year) else '—'
            tag = row.get('tag_label', '')
            text = text_by_pair.get((tid, seq), '(text not found)')

            f.write(f"## {i}. {title} ({year_s}) — `{tag}`\n\n")
            f.write(f"- **Author**: {author}\n")
            f.write(f"- **_id**: `{tid}`, **seq**: {seq}\n")
            gr = genre_raw_by_id.get(tid, [])
            if gr:
                f.write(f"- **genre_raw** (from `lltk.annotations`):\n")
                for g in gr:
                    f.write(f"    - {g}\n")
            else:
                f.write(f"- **genre_raw**: —\n")
            f.write(f"- **confidence**: {row.get('confidence', '')}\n\n")

            f.write("### Passage\n\n")
            # Blockquote the passage — preserve paragraph structure
            for line in str(text).split('\n'):
                f.write(f"> {line}\n")
            f.write("\n")

            f.write("### Annotations\n\n")
            for group_name, flags in GROUPS:
                trues = [flag for flag in flags if row.get(flag) is True]
                if trues:
                    display = [flag.replace(f'has_','').replace('setting_','').replace('characters_','').replace('threat_','').replace('emotion_','').replace('is_','').replace('contains_','')
                               for flag in trues]
                    f.write(f"- **{group_name}**: {', '.join(display)}\n")
            summary = row.get('passage_summary', '') or ''
            notes = row.get('notes', '') or ''
            if summary:
                f.write(f"\n**Summary**: {summary}\n")
            if notes:
                f.write(f"\n**Notes**: {notes}\n")
            f.write("\n---\n\n")

    print(f"Wrote {len(picked)} passages to {args.out}", file=sys.stderr)
    print(f"File size: {os.path.getsize(args.out):,} bytes", file=sys.stderr)


if __name__ == '__main__':
    main()
