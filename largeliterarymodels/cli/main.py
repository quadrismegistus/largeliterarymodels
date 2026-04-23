"""litmod CLI entry point.

Subcommands:
    litmod ls
    litmod show     <TaskName>
    litmod smoke    <TaskName> --model M[,M2,...]
    litmod run      <TaskName> --input CSV --model M [--output CSV]
    litmod annotate <TaskName> [--annotator name] [--port N]
"""

import argparse
import json
import os
import random
import sys
import time

from .models import resolve_model
from .output import compare_print, header_for, pretty_print
from .registry import list_tasks, resolve


def cmd_ls(_args) -> int:
    rows = list_tasks()
    if not rows:
        print("(no tasks registered)", file=sys.stderr)
        return 0
    width = max(len(r[0]) for r in rows)
    print(f"{'TASK'.ljust(width)}  FAMILY    ADAPTER")
    for name, family, ok in rows:
        mark = 'yes' if ok else 'MISSING'
        print(f"{name.ljust(width)}  {family:<8}  {mark}")
    return 0


def cmd_show(args) -> int:
    task_cls, adapter = resolve(args.task)
    print(f"task:   {args.task}")
    print(f"family: {adapter.family}")
    print(f"schema: {task_cls.schema.__name__}")
    print()
    print("--- JSON schema ---")
    print(json.dumps(task_cls.schema.model_json_schema(), indent=2))
    print()
    print("--- fixtures ---")
    try:
        fx = adapter.fixtures()
    except Exception as e:  # noqa: BLE001
        print(f"(failed to load fixtures: {e})")
        return 0
    for r in fx:
        preview = {k: (v if k != 'text' else f"<{len(v)} chars>")
                   for k, v in r.items()}
        print(f"  {preview}")
    return 0


def _run_model(task, prompts, metas, model_id, num_workers):
    """Run one model over all prompts, return list[Result|None]."""
    if num_workers <= 1 or len(prompts) == 1:
        results = []
        for p, m in zip(prompts, metas):
            try:
                results.append(task.run(p, model=model_id, metadata=m))
            except Exception as e:  # noqa: BLE001
                print(f"  failed on one prompt: {e}", file=sys.stderr,
                      flush=True)
                results.append(None)
        return results
    return task.map(prompts, model=model_id, metadata_list=metas,
                    num_workers=num_workers, verbose=True)


def cmd_smoke(args) -> int:
    task_cls, adapter = resolve(args.task)
    tags = [t.strip() for t in args.model.split(',') if t.strip()]
    if not tags:
        raise SystemExit("--model is required")
    resolved = [(tag, resolve_model(tag)) for tag in tags]

    print(f"task={args.task} models={[t for t,_ in resolved]} "
          f"num_workers={args.num_workers}", file=sys.stderr, flush=True)

    records = adapter.fixtures()
    task = task_cls()

    prompts, metas = [], []
    for r in records:
        p, m = adapter.build_prompt(r)
        prompts.append(p)
        metas.append(m)

    results_by_model: dict[str, list] = {}
    for tag, full_id in resolved:
        if len(resolved) > 1:
            print(f"\n--- running {tag} ({full_id}) ---",
                  file=sys.stderr, flush=True)
        results_by_model[tag] = _run_model(
            task, prompts, metas, full_id, args.num_workers)

    if len(resolved) == 1:
        tag = tags[0]
        for r, result in zip(records, results_by_model[tag]):
            if result is None:
                print(f"\n[FAILED] {header_for(r)}", flush=True)
                continue
            pretty_print(result, header_for(r))
    else:
        for i, r in enumerate(records):
            per_model = {tag: results_by_model[tag][i] for tag in tags}
            compare_print(per_model, header_for(r))
    return 0


def cmd_run(args) -> int:
    import pandas as pd

    task_cls, adapter = resolve(args.task)
    model = resolve_model(args.model)
    print(f"task={args.task} model={model} num_workers={args.num_workers} "
          f"input={args.input}", file=sys.stderr, flush=True)

    records = adapter.load_input(args.input)
    if not records:
        raise SystemExit("No records loaded from input.")
    print(f"Loaded {len(records)} records", file=sys.stderr, flush=True)

    if args.limit and args.limit > 0:
        records = records[: args.limit]
        print(f"Limiting to first {len(records)} records", file=sys.stderr,
              flush=True)

    prompts, metas = [], []
    for r in records:
        p, m = adapter.build_prompt(r)
        prompts.append(p)
        metas.append(m)

    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        order = list(range(len(prompts)))
        rng.shuffle(order)
        prompts = [prompts[i] for i in order]
        metas = [metas[i] for i in order]
        records = [records[i] for i in order]

    task = task_cls()
    t0 = time.time()
    results = task.map(prompts, model=model, metadata_list=metas,
                       num_workers=args.num_workers, verbose=True)
    elapsed = time.time() - t0
    print(f"\ntask.map done: {len(results)} in {elapsed/60:.1f} min "
          f"({elapsed/max(1,len(results)):.2f}s/record)",
          file=sys.stderr, flush=True)

    out_path = args.output or _default_output_path(args.task, args.model)
    rows, n_failed = [], 0
    for rec, result in zip(records, results):
        base = {k: v for k, v in rec.items() if k != 'text'}
        if result is None:
            n_failed += 1
            rows.append({**base, '_failed': True})
            continue
        rows.append({**base, **result.model_dump(), '_failed': False})

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path} (failed: {n_failed})",
          file=sys.stderr, flush=True)
    return 0


def cmd_annotate(args) -> int:
    from largeliterarymodels.annotate import run_annotator, load_manifest_keys
    task_cls, _adapter = resolve(args.task)
    task = task_cls()
    only_keys = load_manifest_keys(args.manifest) if args.manifest else None
    run_annotator(task,
                  port=args.port,
                  annotator=args.annotator,
                  host=args.host,
                  only_keys=only_keys)
    return 0


def _default_output_path(task_name: str, model_tag: str) -> str:
    slug = model_tag.replace('.', '').replace(':', '-').replace('/', '-')
    return os.path.join('data', f'litmod_run_{task_name}_{slug}.csv')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='litmod',
                                description='Large-literary-models CLI.')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('ls', help='list registered tasks').set_defaults(
        func=cmd_ls)

    sp = sub.add_parser('show', help='show task schema + fixtures')
    sp.add_argument('task')
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser('smoke', help='run task on fixtures')
    sp.add_argument('task')
    sp.add_argument('--model', required=True,
                    help='short tag (sonnet, opus, qwen-35b, ...) or '
                         'full ID. Comma-separate multiple models for '
                         'side-by-side comparison: --model sonnet,opus')
    sp.add_argument('--num-workers', type=int, default=1)
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser('annotate',
                        help='serve human annotation web app for a task')
    sp.add_argument('task')
    sp.add_argument('--annotator', default='default',
                    help='annotator identifier (used as JSONL filename suffix)')
    sp.add_argument('--port', type=int, default=8989)
    sp.add_argument('--host', default='127.0.0.1')
    sp.add_argument('--manifest', default=None,
                    help='optional CSV path (must have _id + seq cols) '
                         'to restrict annotatable items to a specific '
                         'manifest — e.g. balanced100')
    sp.set_defaults(func=cmd_annotate)

    sp = sub.add_parser('run', help='run task over a manifest CSV')
    sp.add_argument('task')
    sp.add_argument('--input', required=True,
                    help='path to manifest CSV (adapter decides required cols)')
    sp.add_argument('--model', required=True,
                    help='short tag (sonnet, opus, qwen-35b, ...) or full ID')
    sp.add_argument('--num-workers', type=int, default=4)
    sp.add_argument('--output', default=None,
                    help='output CSV path. Default: data/litmod_run_<task>_<model>.csv')
    sp.add_argument('--limit', type=int, default=0,
                    help='debug: run only the first N records (0 = all)')
    sp.add_argument('--shuffle-seed', type=int, default=42,
                    help='shuffle prompt order (deterministic). Use None to disable.')
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
