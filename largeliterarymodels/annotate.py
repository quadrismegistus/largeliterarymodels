"""Generic human annotation web app for any Task.

FastHTML + MonsterUI + fh-pydantic-form. The task's Pydantic schema auto-
generates form widgets (text, select, multiselect pills for list[Literal],
checkboxes, sliders). The LLM's annotation is shown side-by-side as a
read-only reference via fh-pydantic-form's ComparisonForm — click the copy
arrows to pull an LLM value into the human form.

Annotations persist via `task.human_stash(annotator)` (JSONL flat mode).

Usage:
    task.annotate(port=8989)
    # or CLI: python -m largeliterarymodels.annotate classify_passage_form
"""

import re

import fasthtml.common as fh
import monsterui.all as mui
from fh_pydantic_form import (
    ComparisonForm,
    PydanticForm,
    comparison_form_js,
    list_manipulation_js,
)

from .llm import _parse_json_response, _validate_parsed


# ── Data loading ──────────────────────────────────────────────────────────

def _load_annotations(task, annotator: str) -> dict:
    return dict(task.human_stash(annotator).items())


def _save_annotation(task, annotator: str, item_key: str, annotation: dict):
    task.human_stash(annotator)[item_key] = annotation


_HEADER_RE = re.compile(r'^(Title|Author|Year):\s*(.+?)\s*$', re.MULTILINE)


def _parse_header(prompt: str) -> dict:
    """Pull Title/Author/Year lines out of a format_passage-style header."""
    return {m.group(1).lower(): m.group(2) for m in _HEADER_RE.finditer(prompt[:1000])}


def _parse_id(_id: str) -> tuple[str, str]:
    """Best-effort (author, title) extraction from an lltk-style _id.

    Examples:
        _canon_fiction/Anon.1001Nights.1914        → (Anon, 1001Nights)
        _canon_fiction/Tatius.Clitopho and Leucippe → (Tatius, Clitopho and Leucippe)
        _chadwyck/Fic/defoe.08                     → (defoe, 08)
        _earlyprint/A68436                         → ('', A68436)
    """
    tail = str(_id).split('/')[-1]
    parts = tail.split('.', 1)
    if len(parts) == 2:
        author, rest = parts
        rest = re.sub(r'\.\d{3,4}$', '', rest)  # strip trailing .YYYY
        return author, rest
    return '', tail


def _get_items(task, prefer_model: str | None = None,
               only_keys: set | None = None) -> list[dict]:
    """Pull annotatable items from a task's cache, deduped by item_key.

    Each item collects ALL per-model LLM results seen for that passage. The
    `llm_result` / `llm_model` fields default to `prefer_model` if present,
    else the most-frequent model in the cache overall, else first-seen.

    Args:
        task: Task instance with cached results
        prefer_model: default LLM model to show in the comparison view
        only_keys: optional set of (_id, seq_int) tuples — restricts items
            to only those in the set. Useful for staging a specific manifest.

    Returns list of dicts: key, prompt, metadata, llm_result, llm_model,
    llm_results (dict[model→result]), author, title, year.
    """
    from collections import Counter, defaultdict

    # Pre-count model frequency so we can pick a sensible default.
    model_counts: Counter[str] = Counter()
    by_key: dict[str, dict] = {}
    llm_by_key: dict[str, dict[str, object]] = defaultdict(dict)

    for cache_key, raw in task.stash.items():
        if not isinstance(raw, str) or not isinstance(cache_key, dict):
            continue
        try:
            parsed = _parse_json_response(raw)
            result = _validate_parsed(parsed, task.schema)
        except Exception:
            continue

        prompt = cache_key.get('prompt', '')
        metadata = cache_key.get('metadata', {}) or {}
        model = cache_key.get('model', 'unknown')

        if metadata.get('_id') and metadata.get('section_id'):
            item_key = f"{metadata['_id']}::{metadata['section_id']}"
        else:
            item_key = str(hash(prompt))[:12]

        # Manifest filter: extract (id, seq_int) and check against only_keys
        if only_keys is not None:
            _id = metadata.get('_id')
            sid = str(metadata.get('section_id', ''))
            try:
                seq = int(sid.split(':')[-1]) if ':' in sid else int(sid)
            except (ValueError, TypeError):
                continue
            if (_id, seq) not in only_keys:
                continue

        if isinstance(result, list):
            result = result[0] if result else None

        llm_by_key[item_key][model] = result
        model_counts[model] += 1

        if item_key in by_key:
            continue
        header = _parse_header(prompt)
        author_id, title_id = _parse_id(metadata.get('_id', ''))
        by_key[item_key] = {
            'key': item_key,
            'prompt': prompt,
            'metadata': metadata,
            'author': header.get('author') or author_id,
            'title': header.get('title') or title_id,
            'year': str(header.get('year') or metadata.get('year') or ''),
        }

    default_model = prefer_model or (model_counts.most_common(1)[0][0]
                                     if model_counts else None)

    items = []
    for key, base in by_key.items():
        results = llm_by_key[key]
        chosen = default_model if default_model in results else next(iter(results), None)
        items.append({
            **base,
            'llm_results': results,
            'llm_model': chosen,
            'llm_result': results.get(chosen) if chosen else None,
        })
    return items


# ── App ───────────────────────────────────────────────────────────────────

def create_app(task, annotator='default', only_keys: set | None = None):
    """Create the FastHTML annotation app.

    Args:
        task: Task instance.
        annotator: identifier for this annotator (separate stash per person).
        only_keys: optional set of (_id, seq_int) tuples to restrict
            annotatable items (e.g. load a manifest CSV and filter).
    """
    app, rt = fh.fast_app(
        hdrs=[
            mui.Theme.blue.headers(),
            list_manipulation_js(),
            comparison_form_js(),
        ],
        pico=False,
    )

    items = _get_items(task, only_keys=only_keys)
    schema = task.schema

    # Base forms — register routes once; per-request we clone via
    # with_initial_values() to populate defaults.
    human_form = PydanticForm('human', schema, spacing='compact')
    llm_form = PydanticForm('llm', schema, spacing='compact', disabled=True)
    base_comparison = ComparisonForm(
        name='anno',
        left_form=human_form,
        right_form=llm_form,
        left_label='👤 Human (you)',
        right_label='🤖 LLM',
        copy_left=False,
        copy_right=True,  # arrow on LLM side → copy into human form
    )
    base_comparison.register_routes(app)

    # ── Routes ────────────────────────────────────────────────────────

    @rt('/')
    def get():
        done = _load_annotations(task, annotator)
        n_done = len(done)
        n_total = len(items)

        rows = []
        for i, it in enumerate(items):
            is_done = it['key'] in done
            status_text = 'done' if is_done else 'todo'
            badge = (
                mui.Label(status_text, cls=mui.LabelT.primary) if is_done
                else mui.Label(status_text, cls=mui.LabelT.secondary)
            )
            section = it['metadata'].get('section_id', '') or ''
            rows.append(fh.Tr(
                fh.Td(str(i), cls='py-1 px-2 text-gray-500 text-xs'),
                fh.Td(it['year'] or '', cls='py-1 px-2 text-sm',
                      data_sort=str(int(it['year'])) if str(it['year']).lstrip('-').isdigit() else '0'),
                fh.Td(it['author'], cls='py-1 px-2 text-sm'),
                fh.Td(it['title'][:60], cls='py-1 px-2 text-sm',
                      uk_tooltip=it['title'] if len(it['title']) > 60 else None),
                fh.Td(section, cls='py-1 px-2 text-xs text-gray-500'),
                fh.Td(badge, cls='py-1 px-2',
                      data_sort='1' if is_done else '0'),
                data_href=f'/annotate/{i}',
                data_status=status_text,
                cls='border-b hover:bg-blue-50 cursor-pointer',
            ))

        sort_js = fh.Script("""
(function(){
  const tbl = document.getElementById('items-table');
  if (!tbl) return;
  const tbody = tbl.querySelector('tbody');

  function val(td) {
    return td.dataset.sort !== undefined ? td.dataset.sort : td.textContent.trim();
  }
  function sortBy(col, numeric) {
    const asc = tbody.dataset.sortCol === String(col) && tbody.dataset.sortAsc !== 'true';
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a,b) => {
      let av = val(a.children[col]), bv = val(b.children[col]);
      if (numeric) { av = parseFloat(av)||0; bv = parseFloat(bv)||0; }
      else { av = String(av).toLowerCase(); bv = String(bv).toLowerCase(); }
      if (av === bv) return 0;
      return asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    });
    tbody.dataset.sortCol = col;
    tbody.dataset.sortAsc = asc;
    rows.forEach(r => tbody.appendChild(r));
    tbl.querySelectorAll('th').forEach((th,i) => {
      th.dataset.sorted = i === col ? (asc ? 'asc' : 'desc') : '';
    });
  }
  tbl.querySelectorAll('th[data-col]').forEach(th => {
    th.style.cursor = 'pointer';
    th.onclick = () => sortBy(parseInt(th.dataset.col), th.dataset.numeric === '1');
  });
  tbody.querySelectorAll('tr[data-href]').forEach(r => {
    r.onclick = () => window.location = r.dataset.href;
  });

  const filter = document.getElementById('items-filter');
  const statusFilter = document.getElementById('status-filter');
  function applyFilter() {
    const q = (filter.value || '').toLowerCase();
    const s = statusFilter.value;
    tbody.querySelectorAll('tr').forEach(r => {
      const matchText = !q || r.textContent.toLowerCase().includes(q);
      const matchStatus = !s || r.dataset.status === s;
      r.style.display = (matchText && matchStatus) ? '' : 'none';
    });
  }
  filter.oninput = applyFilter;
  statusFilter.onchange = applyFilter;
})();
""")

        filter_bar = fh.Div(
            fh.Input(id='items-filter', type='search',
                     placeholder='Filter by author/title/year/section…',
                     cls='uk-input uk-form-small flex-1 mr-2'),
            fh.Select(
                fh.Option('all', value=''),
                fh.Option('todo', value='todo'),
                fh.Option('done', value='done'),
                id='status-filter',
                cls='uk-select uk-form-small',
                style='width:auto;',
            ),
            cls='flex items-center mb-3',
        )

        return mui.Container(
            mui.Card(
                mui.CardHeader(
                    mui.H1(f'Annotate: {task.task_name}'),
                    fh.P(
                        'Annotator: ', fh.Strong(annotator),
                        ' · Progress: ', fh.Strong(f'{n_done}/{n_total}'),
                        ' · ',
                        fh.A('Compare with LLM', href='/compare',
                             cls='text-blue-600 hover:underline'),
                        cls='text-sm text-gray-600',
                    ),
                ),
                mui.CardBody(
                    filter_bar,
                    fh.Table(
                        fh.Thead(fh.Tr(
                            fh.Th('#', data_col='0', data_numeric='1',
                                  cls='text-left py-1 px-2'),
                            fh.Th('Year', data_col='1', data_numeric='1',
                                  cls='text-left py-1 px-2'),
                            fh.Th('Author', data_col='2',
                                  cls='text-left py-1 px-2'),
                            fh.Th('Title', data_col='3',
                                  cls='text-left py-1 px-2'),
                            fh.Th('Section', data_col='4',
                                  cls='text-left py-1 px-2'),
                            fh.Th('Status', data_col='5',
                                  cls='text-left py-1 px-2'),
                        )),
                        fh.Tbody(*rows),
                        id='items-table',
                        cls='w-full border-collapse text-sm',
                    ),
                    sort_js,
                ),
            ),
            cls='py-6',
        )

    @rt('/annotate/{idx}')
    def get_annotate(idx: int, model: str = ''):
        if idx < 0 or idx >= len(items):
            return fh.RedirectResponse('/', status_code=303)
        it = items[idx]

        done = _load_annotations(task, annotator)
        existing = done.get(it['key'])
        human_initial = None
        if existing:
            try:
                human_initial = schema.model_validate(existing)
            except Exception:
                human_initial = None

        # Pick which model's annotation to show. Query param wins; else default.
        avail = list(it['llm_results'].keys())
        selected_model = model if model in avail else it['llm_model']
        llm_initial = it['llm_results'].get(selected_model) if selected_model else None

        left = human_form.with_initial_values(human_initial) if human_initial else human_form
        right = llm_form.with_initial_values(llm_initial) if llm_initial else llm_form

        right_label = f'🤖 {selected_model}' if selected_model else '🤖 LLM (no cached result)'
        comparison = ComparisonForm(
            name='anno',
            left_form=left,
            right_form=right,
            left_label='👤 Human (you)',
            right_label=right_label,
            copy_left=False,
            copy_right=True,
        )

        # Metadata pills
        meta = it['metadata']
        meta_pills = fh.Div(
            *[
                mui.Label(f'{k}: {v}', cls='mr-2 mb-2 text-xs')
                for k, v in meta.items() if v
            ],
            cls='flex flex-wrap',
        )

        passage = mui.Card(
            mui.CardBody(
                fh.Pre(
                    it['prompt'],
                    cls='whitespace-pre-wrap font-serif text-sm max-h-96 overflow-y-auto m-0',
                ),
            ),
            cls='mb-4',
        )

        # Field reference — collapsible cheatsheet with full descriptions.
        field_entries = []
        for fname, finfo in schema.model_fields.items():
            desc = (finfo.description or '').strip()
            if not desc:
                continue
            field_entries.append(fh.Div(
                fh.Div(fname, cls='font-semibold text-sm text-gray-800 mb-1'),
                fh.Div(desc, cls='text-xs text-gray-600 whitespace-pre-wrap pl-3 border-l-2 border-gray-200'),
                cls='mb-3',
            ))
        field_ref = fh.Details(
            fh.Summary(
                fh.Span('📖 Field descriptions',
                        cls='font-semibold text-sm cursor-pointer text-blue-700'),
                cls='py-2',
            ),
            fh.Div(*field_entries, cls='p-3 bg-gray-50 rounded max-h-96 overflow-y-auto'),
            cls='mb-4 border rounded',
        )

        nav_items = []
        if idx > 0:
            suffix = f'?model={selected_model}' if selected_model else ''
            nav_items.append(fh.A('◀ Prev', href=f'/annotate/{idx-1}{suffix}',
                                  cls='text-blue-600 hover:underline'))
        nav_items.append(fh.A('☰ List', href='/',
                              cls='text-blue-600 hover:underline'))
        if idx < len(items) - 1:
            suffix = f'?model={selected_model}' if selected_model else ''
            nav_items.append(fh.A('Next ▶', href=f'/annotate/{idx+1}{suffix}',
                                  cls='text-blue-600 hover:underline'))

        if len(avail) > 1:
            model_picker = fh.Form(
                fh.Label('Compare against:', cls='text-sm mr-2'),
                fh.Select(
                    *[fh.Option(m, value=m, selected=(m == selected_model))
                      for m in avail],
                    name='model',
                    onchange='this.form.submit()',
                    cls='uk-select uk-form-small',
                    style='width:auto;display:inline-block;',
                ),
                action=f'/annotate/{idx}',
                method='get',
                cls='ml-auto flex items-center',
            )
            nav_items.append(model_picker)
        elif selected_model:
            nav_items.append(fh.Span(
                f'(only {selected_model} cached)',
                cls='ml-auto text-xs text-gray-500',
            ))

        nav = fh.Div(*nav_items, cls='flex gap-4 mb-4 items-center')

        # Custom form wrapping — bypass form_wrapper so we can set action/method.
        form = fh.Form(
            comparison.render_inputs(),
            fh.Div(
                mui.Button('💾 Save & Next', type='submit',
                           cls=(mui.ButtonT.primary,)),
                mui.Button('💾 Save (stay)', type='submit',
                           formaction=f'/save/{idx}?stay=1',
                           formmethod='post',
                           cls=(mui.ButtonT.secondary, 'ml-2')),
                cls='mt-4 flex items-center',
            ),
            id=f'{comparison.name}-comparison-form',
            action=f'/save/{idx}',
            method='post',
        )

        return mui.Container(
            mui.Card(
                mui.CardHeader(
                    nav,
                    mui.H2(f'Item {idx+1} / {len(items)}'),
                    meta_pills,
                ),
                mui.CardBody(passage, field_ref, form),
            ),
            cls='py-6 max-w-7xl',
        )

    @rt('/save/{idx}')
    async def post_save(idx: int, req, stay: int = 0):
        if idx < 0 or idx >= len(items):
            return fh.RedirectResponse('/', status_code=303)
        it = items[idx]
        try:
            annotation = await human_form.model_validate_request(req)
        except Exception as e:
            return mui.Container(
                mui.Card(
                    mui.CardHeader(mui.H3('Validation error')),
                    mui.CardBody(
                        fh.Pre(str(e), cls='whitespace-pre-wrap text-red-700'),
                        fh.A('← Back', href=f'/annotate/{idx}',
                             cls='text-blue-600 hover:underline'),
                    ),
                ),
                cls='py-6',
            )

        _save_annotation(task, annotator, it['key'], annotation.model_dump())

        if stay:
            return fh.RedirectResponse(f'/annotate/{idx}', status_code=303)
        elif idx < len(items) - 1:
            return fh.RedirectResponse(f'/annotate/{idx+1}', status_code=303)
        return fh.RedirectResponse('/', status_code=303)

    @rt('/compare')
    def get_compare():
        done = _load_annotations(task, annotator)
        if not done:
            return mui.Container(
                mui.Card(mui.CardBody(
                    mui.H2('No human annotations yet'),
                    fh.P(fh.A('Start annotating', href='/',
                              cls='text-blue-600 hover:underline')),
                )),
                cls='py-6',
            )

        field_agreement = {fname: {'agree': 0, 'disagree': 0, 'total': 0}
                           for fname in schema.model_fields}

        seen = set()
        for it in items:
            if it['key'] not in done or it['key'] in seen:
                continue
            seen.add(it['key'])
            human = done[it['key']]
            llm = it['llm_result']
            if llm is None:
                continue
            llm_dict = llm.model_dump() if hasattr(llm, 'model_dump') else llm
            for fname in field_agreement:
                h_val = human.get(fname)
                l_val = llm_dict.get(fname)
                if isinstance(h_val, list):
                    h_val = sorted(str(v) for v in h_val)
                if isinstance(l_val, list):
                    l_val = sorted(str(v) for v in l_val)
                match = str(h_val) == str(l_val)
                field_agreement[fname]['total'] += 1
                field_agreement[fname]['agree' if match else 'disagree'] += 1

        rows = []
        for fname, c in field_agreement.items():
            if c['total'] == 0:
                continue
            pct = 100 * c['agree'] / c['total']
            rows.append(fh.Tr(
                fh.Td(fname, cls='py-1 px-2'),
                fh.Td(str(c['agree']), cls='py-1 px-2 text-center'),
                fh.Td(str(c['disagree']), cls='py-1 px-2 text-center'),
                fh.Td(f'{pct:.0f}%', cls='py-1 px-2 text-right'),
            ))

        n_annotated = sum(1 for i in items if i['key'] in done)

        return mui.Container(
            fh.Div(fh.A('← Back to list', href='/',
                        cls='text-blue-600 hover:underline'), cls='mb-4'),
            mui.Card(
                mui.CardHeader(mui.H1(f'Human vs LLM: {task.task_name}')),
                mui.CardBody(
                    fh.P(f'Annotator: ', fh.Strong(annotator),
                         f' · Annotated: ', fh.Strong(f'{n_annotated}/{len(items)}'),
                         cls='text-sm text-gray-600 mb-4'),
                    mui.H2('Field agreement', cls='text-lg mb-2'),
                    fh.Table(
                        fh.Thead(fh.Tr(
                            fh.Th('Field', cls='text-left py-1 px-2'),
                            fh.Th('Agree', cls='text-center py-1 px-2'),
                            fh.Th('Disagree', cls='text-center py-1 px-2'),
                            fh.Th('%', cls='text-right py-1 px-2'),
                        )),
                        fh.Tbody(*rows),
                        cls='w-full border-collapse',
                    ),
                ),
            ),
            cls='py-6',
        )

    return app


# ── Launch helper ─────────────────────────────────────────────────────────

def run_annotator(task, port=8989, annotator='default', host='127.0.0.1',
                  only_keys: set | None = None):
    """Launch the annotation web app for a task.

    Args:
        only_keys: optional set of (_id, seq_int) tuples to restrict items
            to a manifest. None = all cached items.
    """
    import uvicorn
    app = create_app(task, annotator=annotator, only_keys=only_keys)
    items = _get_items(task, only_keys=only_keys)
    print(f"Annotation app for '{task.task_name}' at http://{host}:{port}")
    print(f"Annotator: {annotator}")
    print(f"Items: {len(items)}"
          + (f" (filtered from manifest of {len(only_keys)} keys)"
             if only_keys else ""))
    print(f"Annotations saved under: {task.human_stash(annotator).root_dir}")
    uvicorn.run(app, host=host, port=port)


def load_manifest_keys(csv_path: str) -> set:
    """Parse a manifest CSV (must have _id and seq columns) → set of
    (_id, seq_int) tuples suitable for passing to run_annotator(only_keys=...).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if '_id' not in df.columns or 'seq' not in df.columns:
        raise ValueError(
            f"Manifest {csv_path!r} must have '_id' and 'seq' columns; "
            f"got {list(df.columns)}"
        )
    return set(zip(df['_id'].astype(str), df['seq'].astype(int)))


# ── CLI entrypoint ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import sys
    from . import tasks

    parser = argparse.ArgumentParser(description='Human annotation app for LLM tasks')
    parser.add_argument('task_name', help='Task name (e.g. classify_passage_form)')
    parser.add_argument('--port', type=int, default=8989)
    parser.add_argument('--annotator', default='default')
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    task_map = {}
    for attr_name in dir(tasks):
        obj = getattr(tasks, attr_name)
        if isinstance(obj, type) and hasattr(obj, 'task_name') and hasattr(obj, 'schema'):
            try:
                t = obj()
                task_map[t.task_name] = obj
            except Exception:
                pass

    if args.task_name not in task_map:
        print(f"Unknown task: {args.task_name}")
        print(f"Available: {', '.join(sorted(task_map.keys()))}")
        sys.exit(1)

    task = task_map[args.task_name]()
    run_annotator(task, port=args.port, annotator=args.annotator, host=args.host)
