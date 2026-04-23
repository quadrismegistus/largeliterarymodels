"""Smoke-test PassageContentTask on 3 fixed passages across models.

Passages are hard-coded (_id, seq) so every model run annotates the same three.
Metadata (title/author/year) is also hard-coded to avoid a separate lookup.

Usage:
    python scripts/smoke_passage_content.py --model opus
    python scripts/smoke_passage_content.py --model sonnet
    python scripts/smoke_passage_content.py --model llama-70b
    python scripts/smoke_passage_content.py --model 31b
"""

import argparse
import sys

import lltk
from largeliterarymodels.tasks import PassageContentTask, format_passage


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
    for k, v in d.items():
        if isinstance(v, list):
            print(f"  {k}: {v}", flush=True)
        elif v is True:
            print(f"  {k}: True", flush=True)
    bool_fields = [k for k, v in d.items() if isinstance(v, bool)]
    false_bools = [k for k in bool_fields if d[k] is False]
    if false_bools:
        print(f"  (False bools: {', '.join(false_bools)})", flush=True)
    print(f"  summary: {d.get('passage_summary','')}", flush=True)
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
    parser.add_argument('--num-workers', type=int, default=1,
                        help='task.map worker count. >1 uses task.map() for parallelism. '
                             'Requires provider to support parallel requests '
                             '(LM Studio/Ollama must be configured with matching parallel slots).')
    args = parser.parse_args()

    model = resolve_model(args.model)
    print(f"model={model} num_workers={args.num_workers}", file=sys.stderr, flush=True)

    passages = fetch_passages()
    task = PassageContentTask()

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
