"""Generic human annotation web app for any Task.

Generates a form from the task's Pydantic schema so human annotators can
classify the same items the LLM processed. Supports inter-annotator
reliability comparison between human and LLM annotations.

Usage:
    # From Python
    from largeliterarymodels.tasks import PassageTask
    task = PassageTask()
    task.annotate(port=8989)

    # From CLI
    python -m largeliterarymodels.annotate classify_passage
    python -m largeliterarymodels.annotate classify_passage --port 9000 --annotator alice
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import get_origin, get_args

from .llm import STASH_PATH

# ── Schema introspection ─────────────────────────────────────────────────

def _extract_options(description: str) -> list[str]:
    """Extract enum-like options from a field description.

    Looks for patterns like:
    - "One of: val1, val2, val3."
    - "'option1' = description. 'option2' = description."
    """
    # Pattern 1: "One of: val1, val2, val3."
    match = re.search(r'[Oo]ne of:\s*([^.]+)', description)
    if match:
        raw = match.group(1).strip().rstrip('.')
        options = [o.strip().strip("'\"") for o in raw.split(',')]
        return [o for o in options if o and ' ' not in o and len(o) < 40]

    # Pattern 2: 'option' = description (multiple quoted options with = )
    quoted = re.findall(r"'([a-z_]+)'\s*=", description)
    if len(quoted) >= 2:
        return quoted

    return []


def _field_widget(name: str, field_info) -> dict:
    """Convert a Pydantic FieldInfo into a form widget descriptor.

    Returns a dict with keys: name, type, label, description, options,
    default, required.
    """
    annotation = field_info.annotation
    description = field_info.description or ''
    default = field_info.default if field_info.default is not None else ''
    required = field_info.is_required()

    # Unwrap Optional
    origin = get_origin(annotation)
    if origin is type(None):
        annotation = str

    widget = {
        'name': name,
        'label': name.replace('_', ' ').title(),
        'description': description,
        'default': default,
        'required': required,
    }

    # Determine widget type
    if annotation == bool:
        widget['type'] = 'checkbox'
        widget['default'] = bool(default) if default != '' else False
    elif annotation == float:
        widget['type'] = 'number'
        widget['step'] = 0.05
        if 'confidence' in name.lower() or '0.0 to 1.0' in description:
            widget['min'] = 0.0
            widget['max'] = 1.0
    elif annotation == str:
        options = _extract_options(description)
        if options:
            widget['type'] = 'select'
            widget['options'] = options
            if not required or default == '':
                widget['options'] = [''] + widget['options']
        elif 'reasoning' in name or 'signal' in name or 'notes' in name:
            widget['type'] = 'textarea'
        else:
            widget['type'] = 'text'
    elif origin is list:
        widget['type'] = 'text'  # comma-separated
        widget['description'] = description + ' (comma-separated)'
        widget['default'] = ', '.join(default) if isinstance(default, list) else ''
    else:
        widget['type'] = 'text'

    return widget


def _schema_to_widgets(schema) -> list[dict]:
    """Convert a Pydantic model class into a list of form widget descriptors."""
    widgets = []
    for name, field_info in schema.model_fields.items():
        widgets.append(_field_widget(name, field_info))
    return widgets


# ── Data loading ──────────────────────────────────────────────────────────

ANNOTATIONS_DIR = os.path.join(STASH_PATH, '_human_annotations')


def _annotations_path(task_name: str, annotator: str) -> str:
    d = os.path.join(ANNOTATIONS_DIR, task_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{annotator}.jsonl')


def _load_annotations(task_name: str, annotator: str) -> dict:
    """Load existing human annotations as {item_key: annotation_dict}."""
    path = _annotations_path(task_name, annotator)
    annotations = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    key = entry.get('_annotation_key', '')
                    annotations[key] = entry
    return annotations


def _save_annotation(task_name: str, annotator: str, annotation: dict):
    """Append a single annotation to the annotator's JSONL file."""
    path = _annotations_path(task_name, annotator)
    with open(path, 'a') as f:
        f.write(json.dumps(annotation, ensure_ascii=False) + '\n')


def _get_items(task) -> list[dict]:
    """Get annotatable items from a task's cache.

    Returns list of dicts with 'key' (unique ID), 'prompt' (display text),
    'metadata' (from cache), and 'llm_result' (LLM's annotation).
    """
    items = []
    from .llm import _parse_json_response, _validate_parsed
    for cache_key, raw in task.stash.items():
        if not isinstance(raw, str):
            continue
        try:
            parsed = _parse_json_response(raw)
            result = _validate_parsed(parsed, task.schema)
        except Exception:
            continue

        if not isinstance(cache_key, dict):
            continue

        prompt = cache_key.get('prompt', '')
        metadata = cache_key.get('metadata', {}) or {}

        # Build a stable key from metadata or prompt hash
        if metadata.get('_id') and metadata.get('section_id'):
            item_key = f"{metadata['_id']}::{metadata['section_id']}"
        else:
            item_key = str(hash(prompt))[:12]

        items.append({
            'key': item_key,
            'prompt': prompt,
            'metadata': metadata,
            'llm_result': result.model_dump() if not isinstance(result, list) else result[0].model_dump(),
        })

    return items


# ── FastAPI app ───────────────────────────────────────────────────────────

def create_app(task, annotator='default'):
    """Create a FastAPI annotation app for the given task."""
    from fastapi import FastAPI, Request, Form
    from fastapi.responses import HTMLResponse, RedirectResponse
    import uvicorn

    app = FastAPI(title=f"Annotate: {task.task_name}")
    widgets = _schema_to_widgets(task.schema)
    items = _get_items(task)

    # ── HTML templates ────────────────────────────────────────────────

    def _render_widget_html(w, value=None):
        """Render a single form widget as HTML."""
        val = value if value is not None else w['default']
        name = w['name']
        label = w['label']
        desc = w['description']
        # Truncate long descriptions for tooltip
        short_desc = desc[:120] + '...' if len(desc) > 120 else desc
        html = f'<div class="field" title="{desc}">\n'
        html += f'  <label for="{name}">{label}</label>\n'

        if w['type'] == 'select':
            html += f'  <select name="{name}" id="{name}">\n'
            for opt in w.get('options', []):
                sel = ' selected' if str(val) == opt else ''
                display = opt if opt else '(none)'
                html += f'    <option value="{opt}"{sel}>{display}</option>\n'
            html += '  </select>\n'
        elif w['type'] == 'checkbox':
            checked = ' checked' if val else ''
            html += f'  <input type="checkbox" name="{name}" id="{name}"{checked}>\n'
        elif w['type'] == 'number':
            step = w.get('step', 0.05)
            mn = w.get('min', '')
            mx = w.get('max', '')
            min_attr = f' min="{mn}"' if mn != '' else ''
            max_attr = f' max="{mx}"' if mx != '' else ''
            html += f'  <input type="number" name="{name}" id="{name}" value="{val}" step="{step}"{min_attr}{max_attr}>\n'
        elif w['type'] == 'textarea':
            html += f'  <textarea name="{name}" id="{name}" rows="3">{val}</textarea>\n'
        else:
            html += f'  <input type="text" name="{name}" id="{name}" value="{val}">\n'

        html += f'  <small>{short_desc}</small>\n'
        html += '</div>\n'
        return html

    STYLE = """
    <style>
        * { box-sizing: border-box; font-family: -apple-system, system-ui, sans-serif; }
        body { max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
        h1 { font-size: 1.3em; color: #333; }
        h2 { font-size: 1.1em; color: #555; margin-top: 2em; }
        .meta { background: #e9ecef; padding: 8px 12px; border-radius: 4px; font-size: 0.85em; margin-bottom: 12px; }
        .meta span { margin-right: 16px; }
        .passage { background: white; padding: 16px; border: 1px solid #dee2e6; border-radius: 6px;
                   max-height: 400px; overflow-y: auto; line-height: 1.6; margin-bottom: 20px;
                   font-family: Georgia, serif; font-size: 0.95em; white-space: pre-wrap; }
        .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .col { background: white; padding: 16px; border: 1px solid #dee2e6; border-radius: 6px; }
        .col h2 { margin-top: 0; }
        .field { margin-bottom: 12px; }
        .field label { display: block; font-weight: 600; font-size: 0.85em; margin-bottom: 2px; }
        .field select, .field input[type=text], .field textarea, .field input[type=number] {
            width: 100%; padding: 6px 8px; border: 1px solid #ced4da; border-radius: 4px; font-size: 0.9em; }
        .field small { color: #6c757d; font-size: 0.75em; display: block; margin-top: 2px; }
        .field input[type=checkbox] { margin-right: 8px; }
        button { background: #0d6efd; color: white; border: none; padding: 10px 24px;
                border-radius: 4px; cursor: pointer; font-size: 0.95em; margin-top: 8px; }
        button:hover { background: #0b5ed7; }
        .btn-secondary { background: #6c757d; }
        .btn-secondary:hover { background: #5c636a; }
        .nav { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }
        .nav a { text-decoration: none; color: #0d6efd; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; }
        .badge-done { background: #d1e7dd; color: #0f5132; }
        .badge-todo { background: #fff3cd; color: #664d03; }
        .llm-val { color: #6c757d; font-size: 0.8em; font-style: italic; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #dee2e6; font-size: 0.85em; }
        th { background: #f8f9fa; }
        .agree { background: #d1e7dd; }
        .disagree { background: #f8d7da; }
        .item-list { list-style: none; padding: 0; }
        .item-list li { padding: 8px 12px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
        .item-list li:hover { background: #f0f0f0; }
        .item-list a { text-decoration: none; color: #333; flex: 1; }
    </style>
    """

    # ── Routes ────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        done = _load_annotations(task.task_name, annotator)
        n_done = len(done)
        n_total = len(items)
        rows = ''
        for i, item in enumerate(items):
            meta = item['metadata']
            is_done = item['key'] in done
            badge = '<span class="badge badge-done">done</span>' if is_done else '<span class="badge badge-todo">todo</span>'
            label_parts = []
            if meta.get('_id'):
                label_parts.append(meta['_id'].split('/')[-1])
            if meta.get('chapter_title'):
                label_parts.append(meta['chapter_title'][:40])
            elif meta.get('section_id'):
                label_parts.append(meta['section_id'])
            label = ' / '.join(label_parts) or f'Item {i}'
            rows += f'<li><a href="/annotate/{i}">{label}</a> {badge}</li>\n'

        return f"""<!DOCTYPE html><html><head><title>Annotate: {task.task_name}</title>{STYLE}</head><body>
        <h1>Annotate: {task.task_name}</h1>
        <div class="meta">
            <span>Annotator: <strong>{annotator}</strong></span>
            <span>Progress: <strong>{n_done}/{n_total}</strong></span>
            <span><a href="/compare">Compare with LLM</a></span>
        </div>
        <ul class="item-list">{rows}</ul>
        </body></html>"""

    @app.get("/annotate/{idx}", response_class=HTMLResponse)
    async def annotate_item(idx: int):
        if idx < 0 or idx >= len(items):
            return RedirectResponse("/")
        item = items[idx]
        done = _load_annotations(task.task_name, annotator)
        existing = done.get(item['key'])
        llm = item['llm_result']

        # Parse prompt to extract passage text
        prompt = item['prompt']
        meta = item['metadata']

        # Header with metadata
        meta_html = '<div class="meta">'
        for k, v in meta.items():
            if v:
                meta_html += f'<span>{k}: <strong>{v}</strong></span>'
        meta_html += '</div>'

        # Passage text
        passage_html = f'<div class="passage">{prompt}</div>'

        # Form fields — two columns: human form + LLM values
        form_fields = ''
        for w in widgets:
            # Use existing human annotation if available, else empty
            val = existing.get(w['name']) if existing else None
            llm_val = llm.get(w['name'], '')
            if isinstance(llm_val, list):
                llm_val = ', '.join(str(v) for v in llm_val)
            form_fields += '<div style="display:flex;gap:8px;align-items:flex-start;">'
            form_fields += f'<div style="flex:1;">{_render_widget_html(w, val)}</div>'
            form_fields += f'<div class="llm-val" style="flex:0 0 180px;padding-top:20px;" title="LLM: {llm_val}">LLM: {llm_val}</div>'
            form_fields += '</div>'

        # Navigation
        prev_link = f'<a href="/annotate/{idx-1}">&larr; Prev</a>' if idx > 0 else ''
        next_link = f'<a href="/annotate/{idx+1}">Next &rarr;</a>' if idx < len(items)-1 else ''

        return f"""<!DOCTYPE html><html><head><title>Annotate #{idx}</title>{STYLE}</head><body>
        <div class="nav">{prev_link} <a href="/">List</a> {next_link}</div>
        <h1>Item {idx+1}/{len(items)}</h1>
        {meta_html}
        {passage_html}
        <form method="POST" action="/save/{idx}">
            <h2>Your annotation</h2>
            {form_fields}
            <button type="submit">Save & Next</button>
            <button type="submit" formaction="/save/{idx}?stay=1" class="btn-secondary">Save</button>
        </form>
        </body></html>"""

    @app.post("/save/{idx}")
    async def save_item(idx: int, request: Request, stay: int = 0):
        if idx < 0 or idx >= len(items):
            return RedirectResponse("/")
        item = items[idx]
        form = await request.form()

        annotation = {'_annotation_key': item['key'], '_annotator': annotator}
        for w in widgets:
            name = w['name']
            if w['type'] == 'checkbox':
                annotation[name] = name in form
            elif w['type'] == 'number':
                try:
                    annotation[name] = float(form.get(name, 0))
                except (ValueError, TypeError):
                    annotation[name] = 0.0
            else:
                val = form.get(name, '')
                # Handle list fields (comma-separated)
                field_info = task.schema.model_fields.get(name)
                if field_info and get_origin(field_info.annotation) is list:
                    annotation[name] = [v.strip() for v in val.split(',') if v.strip()]
                else:
                    annotation[name] = val

        # Add metadata from the item
        for k, v in item['metadata'].items():
            annotation[f'_meta_{k}'] = v

        _save_annotation(task.task_name, annotator, annotation)

        if stay:
            return RedirectResponse(f"/annotate/{idx}", status_code=303)
        elif idx < len(items) - 1:
            return RedirectResponse(f"/annotate/{idx+1}", status_code=303)
        else:
            return RedirectResponse("/", status_code=303)

    @app.get("/compare", response_class=HTMLResponse)
    async def compare():
        done = _load_annotations(task.task_name, annotator)
        if not done:
            return f"""<!DOCTYPE html><html><head>{STYLE}</head><body>
            <h1>No human annotations yet</h1><p><a href="/">Start annotating</a></p></body></html>"""

        # Build comparison table
        # For each annotated item, compare human vs LLM per field
        field_agreement = {}
        for w in widgets:
            if w['type'] in ('textarea', 'text') and w['name'] not in ('scene_type', 'narration_mode', 'setting', 'social_epistemology', 'allegorical_regime', 'abs_conc_tendency', 'character_intro_method'):
                continue
            field_agreement[w['name']] = {'agree': 0, 'disagree': 0, 'total': 0}

        comparisons = []
        for item in items:
            if item['key'] not in done:
                continue
            human = done[item['key']]
            llm = item['llm_result']
            row = {'key': item['key']}
            for fname in field_agreement:
                h_val = human.get(fname, '')
                l_val = llm.get(fname, '')
                if isinstance(l_val, list):
                    l_val = ', '.join(l_val)
                match = str(h_val).strip() == str(l_val).strip()
                row[fname] = {'human': h_val, 'llm': l_val, 'match': match}
                field_agreement[fname]['total'] += 1
                if match:
                    field_agreement[fname]['agree'] += 1
                else:
                    field_agreement[fname]['disagree'] += 1
            comparisons.append(row)

        # Summary table
        summary = '<h2>Agreement summary</h2><table><tr><th>Field</th><th>Agree</th><th>Disagree</th><th>Agreement %</th></tr>\n'
        for fname, counts in field_agreement.items():
            if counts['total'] == 0:
                continue
            pct = 100 * counts['agree'] / counts['total']
            summary += f'<tr><td>{fname}</td><td>{counts["agree"]}</td><td>{counts["disagree"]}</td><td>{pct:.0f}%</td></tr>\n'
        summary += '</table>'

        # Detail table
        detail = '<h2>Item-level comparison</h2>'
        for comp in comparisons[:50]:
            detail += f'<h3>{comp["key"]}</h3><table><tr><th>Field</th><th>Human</th><th>LLM</th></tr>\n'
            for fname in field_agreement:
                if fname not in comp:
                    continue
                c = comp[fname]
                cls = 'agree' if c['match'] else 'disagree'
                detail += f'<tr class="{cls}"><td>{fname}</td><td>{c["human"]}</td><td>{c["llm"]}</td></tr>\n'
            detail += '</table>'

        return f"""<!DOCTYPE html><html><head><title>Compare</title>{STYLE}</head><body>
        <div class="nav"><a href="/">Back to list</a></div>
        <h1>Human vs LLM: {task.task_name}</h1>
        <div class="meta"><span>Annotator: <strong>{annotator}</strong></span>
        <span>Annotated: <strong>{len(comparisons)}/{len(items)}</strong></span></div>
        {summary}
        {detail}
        </body></html>"""

    return app


# ── Launch helper ─────────────────────────────────────────────────────────

def run_annotator(task, port=8989, annotator='default', host='127.0.0.1'):
    """Launch the annotation web app for a task."""
    import uvicorn
    app = create_app(task, annotator=annotator)
    print(f"Annotation app for '{task.task_name}' at http://{host}:{port}")
    print(f"Annotator: {annotator}")
    print(f"Items: {len(_get_items(task))}")
    print(f"Annotations saved to: {_annotations_path(task.task_name, annotator)}")
    uvicorn.run(app, host=host, port=port)


# ── CLI entrypoint ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Human annotation app for LLM tasks')
    parser.add_argument('task_name', help='Task name (e.g. classify_passage, classify_frye)')
    parser.add_argument('--port', type=int, default=8989)
    parser.add_argument('--annotator', default='default')
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    # Import the task by name
    from . import tasks
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
