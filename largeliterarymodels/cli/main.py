"""litmod CLI entry point.

Subcommands:
    litmod ls
    litmod show     <TaskName>
    litmod doctor   [--provider anthropic,openai,...] [--cheap-only]
    litmod render   <TaskName> [--item TEXT | --item-file F | --fixture]
    litmod smoke    <TaskName> --model M[,M2,...]
    litmod run      <TaskName> --input CSV --model M [--output CSV]
    litmod annotate <TaskName> [--annotator name] [--port N]
    litmod cloud    <launch|setup|upload|run|status|download|stop|sync|attach|cancel|log|ssh>
"""

import argparse
import json
import os
import random
import sys
import time

from .cloud import SUMMARY_TASK_MAP
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
    print(f"family: {adapter.family if adapter else '(file task — no adapter)'}")
    print(f"schema: {task_cls.schema.__name__}")
    print()
    print("--- JSON schema ---")
    print(json.dumps(task_cls.schema.model_json_schema(), indent=2))
    print()
    print("--- fixtures ---")
    if adapter is None:
        print("(file tasks have no fixture adapter)")
        return 0
    try:
        fx = adapter.fixtures()
    except (SystemExit, Exception) as e:  # noqa: BLE001 — adapters raise SystemExit on missing data
        print(f"(failed to load fixtures: {e})")
        return 0
    for r in fx:
        preview = {k: (v if k != 'text' else f"<{len(v)} chars>")
                   for k, v in r.items()}
        print(f"  {preview}")
    return 0


def _cmd_doctor(args) -> int:
    # Imported lazily: pulls in provider SDKs, which `litmod ls` shouldn't pay for.
    from .doctor import cmd_doctor
    return cmd_doctor(args)


def cmd_render(args) -> int:
    """Print the task's instrument as one self-contained string.

    For administering the identical scheme outside the API path: another
    provider, a subagent, or a human coder on paper. The DEFAULT output is
    byte-identical to the API system prompt — an earlier version defaulted
    the provenance footer ON, so the natural invocation shipped a
    non-byte-exact instrument to exactly the audience byte-exactness was
    built for.
    """
    try:
        task_cls, adapter = resolve(args.task)
    except SystemExit:
        # The CLI registry maps a curated dozen tasks to adapters, which is
        # narrower than the task package itself — and the tasks most in need
        # of rendering are often the newest, unregistered ones. Fall back to
        # the package: the instrument needs no adapter, only --fixture does.
        import largeliterarymodels.tasks as tasks_pkg
        from largeliterarymodels.task import Task
        task_cls = getattr(tasks_pkg, args.task, None)
        if not (isinstance(task_cls, type) and issubclass(task_cls, Task)):
            raise
        adapter = None

    task = task_cls()

    item = None
    if args.item:
        item = args.item
    elif args.item_file:
        with open(args.item_file) as f:
            item = f.read()
    elif args.fixture:
        if adapter is None:
            raise SystemExit(
                f"{args.task} has no registered adapter, so --fixture has "
                f"nothing to draw from; pass --item or --item-file."
            )
        try:
            records = adapter.fixtures()
        except (SystemExit, Exception) as e:  # noqa: BLE001
            raise SystemExit(f"could not load fixtures: {e}")
        if not records:
            raise SystemExit("adapter returned no fixtures")
        item, _meta = adapter.build_prompt(records[0])

    text = task.render_instrument(item=item, digest=args.digest)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(text + "\n")
        print(f"wrote {len(text)} chars to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_price(args) -> int:
    """Price a measured workload: one model, or the whole table ranked.

    Prospective estimates are cache-aware when --prefix-tokens is given —
    below the model's known cache floor the discount is refused rather
    than promised. Every line prints the table's fetch date: a pricing
    table is a constants file, and constants rot.
    """
    from largeliterarymodels import costs

    if args.times < 1:
        print(f"--times {args.times}: a workload runs a positive number of "
              f"times; a negative price is not a refund.", file=sys.stderr)
        return 2

    kw = dict(fresh=args.fresh, cached=args.cached, output=args.output,
              cache_write_5m=args.cache_write, batch=args.batch,
              on=args.on, times=args.times,
              prefix_tokens=args.prefix_tokens)

    if args.model:
        try:
            est = costs.price(args.model, **kw)
        except ValueError as e:
            # The module's error names the table and its fetch date; a
            # traceback buries that under frames nobody asked for.
            print(str(e), file=sys.stderr)
            return 1
        print(f"{est['provider']}/{est['model']}"
              f"{'  [batch]' if args.batch else ''}: ${est['usd']:.4f}"
              f"   (prices fetched {est['pricing_date']})")
        for k, v in est["lines"].items():
            print(f"    {k:<16} ${v:.4f}")
        for w in est["warnings"]:
            print(f"    ! {w}")
        return 0

    rows = costs.compare(providers=[args.provider] if args.provider else None,
                         **kw)
    floored = sum(1 for est in rows
                  if any("BELOW" in w for w in est["warnings"]))
    print(f"workload: {args.fresh:,} fresh + {args.cached:,} cached input, "
          f"{args.output:,} output"
          + (f", x{args.times}" if args.times > 1 else "")
          + f"   (prices fetched {costs.pricing_date()})")
    if args.batch:
        print("BATCH pricing where the provider offers it "
              "(deepseek does not).")
    print(f"\n  {'provider':<10} {'model':<32} {'COST':>10}")
    print("  " + "-" * 56)
    for est in rows:
        floor = any("FLOOR" in w for w in est["warnings"])
        under = any("BELOW" in w for w in est["warnings"])
        print(f"  {est['provider']:<10} {est['model']:<32} "
              f"{est['usd']:>10.2f}{' *floor' if floor else ''}"
              f"{' !no-cache' if under else ''}")
    print("\n  *floor: the model cannot stop reasoning; a non-reasoning "
          "workload priced against it is a floor, not an estimate.")
    if floored:
        print(f"  !no-cache: prefix under this model's cache floor "
              f"({floored} rows re-billed at full input rate — floors are "
              f"non-monotonic, which is exactly what reorders this table).")
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
    if adapter is None:
        raise SystemExit(
            f"{args.task}: a task loaded from a file has no fixture "
            f"adapter, and smoke runs on fixtures. Use `litmod annotate` "
            f"or `render` with it, or smoke from Python: "
            f"TaskClass().run(text).")
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
    if adapter is None:
        raise SystemExit(
            f"{args.task}: a task loaded from a file has no input "
            f"adapter, and run needs one to load records. Run it from "
            f"Python (task.map with your own manifest), or register an "
            f"adapter family.")
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

    # Shuffle for even progress/ETA across eras, but write output rows in
    # the original manifest order.
    order = list(range(len(prompts)))
    if not args.no_shuffle:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(order)
        prompts = [prompts[i] for i in order]
        metas = [metas[i] for i in order]

    task = task_cls()
    t0 = time.time()
    if getattr(args, "batch", False):
        results = task.map(prompts, model=model, metadata_list=metas,
                           batch=True)
    else:
        results = task.map(prompts, model=model, metadata_list=metas,
                           num_workers=args.num_workers, verbose=True)
    elapsed = time.time() - t0

    unshuffled = [None] * len(results)
    for pos, orig in enumerate(order):
        unshuffled[orig] = results[pos]
    results = unshuffled
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


def cmd_batch(args) -> int:
    """Run a summary-based task over a directory of social network JSONs."""
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    task_map = SUMMARY_TASK_MAP
    if args.task not in task_map:
        raise SystemExit(f"Unknown task: {args.task}. Available: {list(task_map.keys())}")

    from largeliterarymodels import tasks as task_mod
    task_class = getattr(task_mod, task_map[args.task])
    model = resolve_model(args.model) if args.model else task_class.model

    output_dir = args.output or os.path.join('data', f'{args.task}_output')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(Path(args.input).glob('*.json'))
    if args.limit:
        files = files[:args.limit]

    def model_slug(m):
        return m.replace('/', '-').replace('.', '')

    print(f"Task: {args.task} ({task_map[args.task]})", file=sys.stderr)
    print(f"Files: {len(files)}", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)

    if args.dry_run:
        slug = model_slug(model)
        existing = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
        n_skip = sum(1 for f in files
                     if f"{_json.load(open(f)).get('metadata', {}).get('_canonical_id', f.stem).replace('/', '_')}_{slug}.json" in existing)
        print(f"Would process {len(files) - n_skip}, skip {n_skip}", file=sys.stderr)
        print(file=sys.stderr)
        from largeliterarymodels.costs import dry_run as cost_dry_run
        cost_dry_run(task_class, args.input, output_tokens=200)
        return 0

    task = task_class()
    slug = model_slug(model)

    def run_one(fpath):
        with open(fpath) as fh:
            sn = _json.load(fh)
        cid = sn.get('metadata', {}).get('_canonical_id', '')
        if not cid:
            cid = Path(fpath).stem
        out_name = f"{cid.replace('/', '_')}_{slug}.json"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path) and not args.force:
            return 'skip', cid, 0
        prompt = task_class.format_input(sn)
        if not prompt.strip() or len(prompt) < 50:
            return 'empty', cid, 0
        t0 = time.time()
        try:
            result = task.run(prompt, model=model, force=args.force)
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR {cid}: {e} ({elapsed:.1f}s)", flush=True)
            return 'error', cid, elapsed
        out = {
            'metadata': {'source': cid, '_canonical_id': cid,
                         'model': model, 'task': args.task},
            **result.model_dump(),
        }
        with open(out_path, 'w') as fh:
            _json.dump(out, fh, indent=2)
        return 'ok', cid, elapsed

    counts = {'ok': 0, 'skip': 0, 'error': 0, 'empty': 0}
    total_time = 0
    done = 0
    n_workers = args.num_workers

    if n_workers > 1:
        futures = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for fpath in files:
                futures[pool.submit(run_one, fpath)] = fpath
            for fut in as_completed(futures):
                done += 1
                status, cid, elapsed = fut.result()
                counts[status] += 1
                if status == 'ok':
                    total_time += elapsed
                    avg = total_time / counts['ok']
                    remaining = (len(files) - done) * avg / n_workers
                    print(f"  [{done}/{len(files)}] {cid} ({elapsed:.1f}s, "
                          f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
                elif status == 'skip' and done % 100 == 0:
                    print(f"  [{done}/{len(files)}] skipping...", flush=True)
    else:
        for i, fpath in enumerate(files):
            status, cid, elapsed = run_one(fpath)
            counts[status] += 1
            if status == 'ok':
                total_time += elapsed
                avg = total_time / counts['ok']
                remaining = (len(files) - i - 1) * avg
                print(f"  [{i+1}/{len(files)}] {cid} ({elapsed:.1f}s, "
                      f"avg {avg:.1f}s, ~{remaining/60:.0f}min left)", flush=True)
            elif status == 'skip' and (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(files)}] skipping...", flush=True)

    print(f"\nDone: {counts['ok']} ok, {counts['skip']} skipped, "
          f"{counts['error']} errors, {counts['empty']} empty", file=sys.stderr)
    print(f"Total API time: {total_time:.0f}s ({total_time/60:.1f}min)",
          file=sys.stderr)
    return 0


def cmd_cloud(args) -> int:
    """Delegate to the Vast.ai cloud manager."""
    from .cloud import main as cloud_main
    cloud_argv = [args.cloud_command] + args.cloud_args
    if args.yes:
        cloud_argv = ['--yes'] + cloud_argv
    cloud_main(cloud_argv)
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
    sp.add_argument('task', help='registered task name, or '
                    'path/to/task.py[:ClassName]')
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser(
        'doctor',
        help='probe each provider on a cheap and a current frontier model')
    sp.add_argument('--provider', default=None,
                    help='comma-separated subset (anthropic,openai,google,'
                         'deepseek,local)')
    sp.add_argument('--cheap-only', action='store_true',
                    help='skip the frontier-tier probes')
    sp.add_argument('--include-local', action='store_true',
                    help='also probe the local endpoint (must be running)')
    sp.add_argument('--timeout', type=float, default=120.0)
    sp.set_defaults(func=_cmd_doctor)

    sp = sub.add_parser(
        'render',
        help='print the task instrument as one self-contained string')
    sp.add_argument('task', help='registered task name, or path/to/task.py[:ClassName]')
    sp.add_argument('--item', default=None,
                    help='item text to append as the item to annotate')
    sp.add_argument('--item-file', default=None,
                    help='read the item to annotate from this file')
    sp.add_argument('--fixture', action='store_true',
                    help="use the adapter's first fixture as the item")
    sp.add_argument('--digest', action='store_true',
                    help='append the provenance footer (instrument sha256). '
                         'Off by default: the default output is byte-'
                         'identical to the API system prompt, so it can be '
                         'piped to a second coder as-is')
    sp.add_argument('--output', '-o', default=None, help='write to file')
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser(
        'price',
        help='price a measured workload against one model or the table')
    sp.add_argument('--fresh', type=int, default=0,
                    help='uncached input tokens')
    sp.add_argument('--cached', type=int, default=0,
                    help='cache-read input tokens')
    sp.add_argument('--output', type=int, default=0, help='output tokens')
    sp.add_argument('--cache-write', type=int, default=0,
                    help='cache-write tokens (5m TTL)')
    sp.add_argument('--prefix-tokens', type=int, default=None,
                    help='cacheable-prefix size; enables the cache-floor '
                         'check on prospective estimates')
    sp.add_argument('--model', default=None,
                    help='price one model (tag, alias, or full id) instead '
                         'of ranking the table')
    sp.add_argument('--provider',
                    choices=('anthropic', 'openai', 'deepseek', 'google'))
    sp.add_argument('--batch', action='store_true',
                    help='apply batch discounts where the provider has one')
    sp.add_argument('--times', type=int, default=1,
                    help='multiply the workload (e.g. 3 coder arms)')
    sp.add_argument('--on', default=None,
                    help='pricing date YYYY-MM-DD (dated rows, e.g. '
                         'sonnet-5 introductory pricing)')
    sp.set_defaults(func=cmd_price)

    sp = sub.add_parser('smoke', help='run task on fixtures')
    sp.add_argument('task', help='registered task name, or path/to/task.py[:ClassName]')
    sp.add_argument('--model', required=True,
                    help='short tag (sonnet, opus, qwen-35b, ...) or '
                         'full ID. Comma-separate multiple models for '
                         'side-by-side comparison: --model sonnet,opus')
    sp.add_argument('--num-workers', type=int, default=1)
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser('annotate',
                        help='serve human annotation web app for a task')
    sp.add_argument('task', help='registered task name, or path/to/task.py[:ClassName]')
    sp.add_argument('--annotator', default='default',
                    help='annotator identifier (used as JSONL filename suffix)')
    sp.add_argument('--port', type=int, default=8989)
    sp.add_argument('--host', default='127.0.0.1')
    sp.add_argument('--manifest', default=None,
                    help='optional CSV path (must have _id + seq cols) '
                         'to restrict annotatable items to a specific '
                         'manifest — e.g. balanced100')
    sp.set_defaults(func=cmd_annotate)

    sp = sub.add_parser('batch',
                        help='run summary-based task over social network exports')
    sp.add_argument('task', choices=sorted(SUMMARY_TASK_MAP))
    sp.add_argument('--input', '-i', required=True,
                    help='dir of social network JSONs')
    sp.add_argument('--output', '-o', default=None,
                    help='output dir (default: data/<task>_output/)')
    sp.add_argument('--model', '-m', default=None,
                    help='model tag or full ID (default: task default)')
    sp.add_argument('--num-workers', '-w', type=int, default=1)
    sp.add_argument('--force', action='store_true')
    sp.add_argument('--dry-run', action='store_true',
                    help='show cost estimate without running')
    sp.add_argument('--limit', type=int, default=0)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser('cloud',
                        help='manage Vast.ai GPU instances')
    sp.add_argument('--yes', '-y', action='store_true',
                    help='skip confirmation prompts')
    sp.add_argument('cloud_command',
                    help='cloud subcommand: launch|setup|upload|run|status|'
                         'download|stop|sync|attach|cancel|log|ssh '
                         '(validated by the cloud parser)')
    sp.add_argument('cloud_args', nargs=argparse.REMAINDER, help='arguments for subcommand')
    sp.set_defaults(func=cmd_cloud)

    sp = sub.add_parser('run', help='run task over a manifest CSV')
    sp.add_argument('task', help='registered task name, or path/to/task.py[:ClassName]')
    sp.add_argument('--input', required=True,
                    help='path to manifest CSV (adapter decides required cols)')
    sp.add_argument('--model', required=True,
                    help='short tag (sonnet, opus, qwen-35b, ...) or full ID')
    sp.add_argument('--num-workers', type=int, default=4)
    sp.add_argument('--batch', action='store_true',
                    help='submit via the provider batch API (50%% pricing '
                         'on anthropic/openai/google; blocks until the '
                         'batch completes, ledger-safe against resubmission)')
    sp.add_argument('--output', default=None,
                    help='output CSV path. Default: data/litmod_run_<task>_<model>.csv')
    sp.add_argument('--limit', type=int, default=0,
                    help='debug: run only the first N records (0 = all)')
    sp.add_argument('--shuffle-seed', type=int, default=42,
                    help='seed for deterministic prompt-order shuffle '
                         '(output CSV keeps manifest order)')
    sp.add_argument('--no-shuffle', action='store_true',
                    help='process prompts in manifest order')
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
