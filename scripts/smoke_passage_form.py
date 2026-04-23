"""Smoke-test PassageFormTask on 3 fixed passages across models.

Mirrors smoke_passage_content.py but targets the narratological/interpretive
task. Best run with frontier models (Sonnet/Opus) — several fields require
interpretive judgment that locals handle unreliably.

Usage:
    python scripts/smoke_passage_form.py --model sonnet
    python scripts/smoke_passage_form.py --model opus
    python scripts/smoke_passage_form.py --model qwen3.5-35b-a3b  # local sanity check
"""

import argparse
import sys

import lltk
from largeliterarymodels.tasks import PassageFormTask, format_passage


SAMPLE = [
    {
        '_id': '_chadwyck/Early_English_Prose_Fiction/ee01010.02',
        'seq': 25,
        'title': 'Alcander and Philocrates',
        'author': 'Anon.',
        'year': 1696,
    },
    {
        '_id': '_chadwyck/Eighteenth-Century_Fiction/amory.01',
        'seq': 25,
        'title': 'John Buncle',
        'author': 'Amory, Thomas',
        'year': 1756,
    },
    {
        '_id': '_chadwyck/Nineteenth-Century_Fiction/ncf0302.01',
        'seq': 25,
        'title': 'Hermsprong; or, Man As He Is Not',
        'author': 'Bage, Robert',
        'year': 1796,
    },
]


def resolve_model(tag):
    table = {
        'e2b':             'ollama/gemma4:e2b',
        'e4b':             'ollama/gemma4:e4b',
        '31b':             'ollama/gemma4:31b',
        'gguf-31b':        'lmstudio/gemma-4-31b-it',
        'qwen3.5-35b-a3b': 'lmstudio/qwen3.5-35b-a3b',
        'llama-70b':       'lmstudio/meta-llama-3.1-70b-instruct',
        'sonnet':          'claude-sonnet-4-6',
        'opus':            'claude-opus-4-7',
        'gemini-flash':    'gemini-2.5-flash',
        'gemini-pro':      'gemini-2.5-pro',
    }
    if tag not in table:
        raise SystemExit(f"Unknown model tag: {tag}. Choices: {list(table)}")
    return table[tag]


def fetch_passages():
    ids = [s['_id'] for s in SAMPLE]
    df = lltk.db.get_passages(ids)
    lookup = {}
    for _, r in df.iterrows():
        lookup[(r['_id'], int(r['seq']))] = r['text']
    out = []
    for s in SAMPLE:
        key = (s['_id'], s['seq'])
        if key not in lookup:
            raise SystemExit(f"Missing passage {key} — re-run `lltk db-passages`?")
        out.append({**s, 'text': lookup[key]})
    return out


def print_result(p, result):
    print(f"\n{'='*70}\n{p['_id']}  seq={p['seq']}  ({p['title']}, {p['year']})\n{'='*70}",
          flush=True)
    d = result.model_dump()
    # Lists
    for k in ('narration_modes', 'interiority_focus', 'emotional_register'):
        if k in d:
            print(f"  {k}: {d[k]}", flush=True)
    # Scalar enums
    for k in ('story_time_span', 'distance_traveled', 'narrative_frequency',
              'narrate_vs_describe', 'mood', 'physicality_level'):
        if k in d:
            print(f"  {k}: {d[k]}", flush=True)
    # Bool axes
    bools = ('voices_heterogeneous', 'has_reality_effect', 'concrete_bespeaks_abstract',
             'abstractions_as_agents', 'characters_known_by_reputation',
             'characters_known_by_physical_appearance', 'uses_nominalization')
    true_bools = [k for k in bools if d.get(k) is True]
    false_bools = [k for k in bools if d.get(k) is False]
    if true_bools:
        print(f"  TRUE: {', '.join(true_bools)}", flush=True)
    if false_bools:
        print(f"  (False: {', '.join(false_bools)})", flush=True)
    if d.get('notes'):
        print(f"  notes: {d['notes']}", flush=True)
    print(f"  confidence: {d.get('confidence')}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['e2b', 'e4b', '31b', 'gguf-31b',
                                 'qwen3.5-35b-a3b', 'llama-70b',
                                 'sonnet', 'opus',
                                 'gemini-flash', 'gemini-pro'])
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    model = resolve_model(args.model)
    print(f"model={model} num_workers={args.num_workers}", file=sys.stderr, flush=True)

    passages = fetch_passages()
    task = PassageFormTask()

    if args.num_workers <= 1:
        for p in passages:
            prompt, meta = format_passage(
                p['text'],
                title=p['title'], author=p['author'], year=p['year'],
                _id=p['_id'], section_id=f"p500:{p['seq']}",
            )
            result = task.run(prompt, model=model, metadata=meta)
            print_result(p, result)
    else:
        prompts, metas = [], []
        for p in passages:
            prompt, meta = format_passage(
                p['text'],
                title=p['title'], author=p['author'], year=p['year'],
                _id=p['_id'], section_id=f"p500:{p['seq']}",
            )
            prompts.append(prompt)
            metas.append(meta)
        results = task.map(prompts, model=model, metadata_list=metas,
                           num_workers=args.num_workers, verbose=True)
        for p, result in zip(passages, results):
            print_result(p, result)


if __name__ == '__main__':
    main()
