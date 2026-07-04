"""Cost estimation for Anthropic API calls.

Usage:
    from largeliterarymodels.costs import estimate, print_estimate, dry_run

    # Exact cost from actual inputs
    dry_run(PlotGenreTask, 'data/social_network_export/')

    # Manual estimate
    estimate('sonnet', input_tokens=15000, output_tokens=200, n_calls=1455,
             cached_tokens=13000)

    # Compare models
    print_estimate(input_tokens=15000, output_tokens=200, n_calls=1455,
                   cached_tokens=13000)
"""

import json
from pathlib import Path

PRICING_FILE = Path(__file__).parent.parent / 'data' / 'anthropic_pricing.json'

_pricing = None


def _load_pricing():
    global _pricing
    if _pricing is None:
        with open(PRICING_FILE) as f:
            _pricing = json.load(f)
    return _pricing


def resolve_model(model: str) -> str:
    p = _load_pricing()
    if model in p['models']:
        return model
    if model in p['aliases']:
        return p['aliases'][model]
    matches = [key for key in p['models'] if model in key]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(
            f"Ambiguous model {model!r} matches {matches}; use the full ID."
        )
    raise ValueError(f"Unknown model: {model}. Available: {list(p['models'].keys())}")


def estimate(
    model: str,
    input_tokens: int,
    output_tokens: int,
    n_calls: int = 1,
    cached_tokens: int = 0,
) -> dict:
    """Estimate cost for a batch of API calls.

    Args:
        model: Model name or alias ('sonnet', 'haiku', 'opus', etc.)
        input_tokens: Total input tokens per call (including cached portion)
        output_tokens: Output tokens per call
        n_calls: Number of calls
        cached_tokens: Portion of input_tokens that hits prompt cache
            (system prompt + examples). First call pays write cost,
            subsequent calls pay hit cost.

    Returns:
        Dict with cost breakdown and total.
    """
    p = _load_pricing()
    model_id = resolve_model(model)
    m = p['models'][model_id]

    uncached_per_call = input_tokens - cached_tokens
    mtok = 1_000_000

    if cached_tokens > 0 and n_calls > 1:
        cache_write_cost = (cached_tokens / mtok) * m['cache_5m_write']
        cache_hit_cost = ((n_calls - 1) * cached_tokens / mtok) * m['cache_hit']
        uncached_input_cost = (n_calls * uncached_per_call / mtok) * m['input']
    elif cached_tokens > 0 and n_calls == 1:
        cache_write_cost = (cached_tokens / mtok) * m['cache_5m_write']
        cache_hit_cost = 0
        uncached_input_cost = (uncached_per_call / mtok) * m['input']
    else:
        cache_write_cost = 0
        cache_hit_cost = 0
        uncached_input_cost = (n_calls * input_tokens / mtok) * m['input']

    output_cost = (n_calls * output_tokens / mtok) * m['output']
    total = cache_write_cost + cache_hit_cost + uncached_input_cost + output_cost

    no_cache_total = (n_calls * input_tokens / mtok) * m['input'] + output_cost
    savings = no_cache_total - total

    return {
        'model': model_id,
        'n_calls': n_calls,
        'input_tokens_per_call': input_tokens,
        'cached_tokens_per_call': cached_tokens,
        'output_tokens_per_call': output_tokens,
        'cache_write': round(cache_write_cost, 4),
        'cache_hits': round(cache_hit_cost, 4),
        'uncached_input': round(uncached_input_cost, 4),
        'output': round(output_cost, 4),
        'total': round(total, 4),
        'without_cache': round(no_cache_total, 4),
        'cache_savings': round(savings, 4),
    }


def print_estimate(
    input_tokens: int,
    output_tokens: int,
    n_calls: int = 1,
    cached_tokens: int = 0,
    models: list[str] | None = None,
):
    """Print cost comparison across models.

    Args:
        input_tokens: Total input tokens per call
        output_tokens: Output tokens per call
        n_calls: Number of calls
        cached_tokens: Cached portion of input tokens
        models: List of models to compare (default: sonnet, haiku, opus)
    """
    if models is None:
        models = ['haiku', 'sonnet', 'opus']

    print(f"Cost estimate: {n_calls:,} calls × {input_tokens:,} input "
          f"({cached_tokens:,} cached) + {output_tokens:,} output tokens\n")
    print(f"{'Model':<25s} {'Total':>8s} {'w/o cache':>10s} {'Savings':>8s}  "
          f"{'Cache W':>8s} {'Cache H':>8s} {'Input':>8s} {'Output':>8s}")
    print("-" * 95)

    for model in models:
        try:
            e = estimate(model, input_tokens, output_tokens, n_calls, cached_tokens)
            print(f"{e['model']:<25s} ${e['total']:>7.2f} ${e['without_cache']:>9.2f} "
                  f"${e['cache_savings']:>7.2f}  "
                  f"${e['cache_write']:>7.4f} ${e['cache_hits']:>7.4f} "
                  f"${e['uncached_input']:>7.4f} ${e['output']:>7.4f}")
        except ValueError as ex:
            print(f"{model:<25s} {str(ex)}")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (Claude's approximate tokenizer)."""
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    return len(enc.encode(text))


def dry_run(
    task_class,
    input_dir: str,
    output_tokens: int = 200,
    models: list[str] | None = None,
    limit: int = 0,
):
    """Measure exact token counts from a task's actual prompts, then estimate cost.

    Args:
        task_class: A Task class with format_input() and system_prompt
            (e.g., PlotGenreTask, SubgenreTask)
        input_dir: Directory of social network JSON exports
        output_tokens: Estimated output tokens per call
        models: Models to compare (default: haiku, sonnet, opus)
        limit: Max files to scan (0 = all)
    """
    import glob
    import json as _json

    files = sorted(glob.glob(f'{input_dir}/*.json'))
    if limit:
        files = files[:limit]

    task = task_class()
    system_prompt = getattr(task, 'system_prompt', '')

    examples_text = ''
    if hasattr(task, 'examples') and task.examples:
        for inp, out in task.examples:
            examples_text += str(inp) + str(out.model_dump_json() if hasattr(out, 'model_dump_json') else out)

    cached_text = system_prompt + examples_text
    cached_tokens = count_tokens(cached_text)

    prompt_tokens = []
    for f in files:
        with open(f) as fh:
            sn = _json.load(fh)
        prompt = task_class.format_input(sn)
        prompt_tokens.append(count_tokens(prompt))

    n = len(prompt_tokens)
    if n == 0:
        print("No files found.")
        return

    import numpy as np
    arr = np.array(prompt_tokens)

    total_input_per_call = int(arr.mean()) + cached_tokens

    print(f"Task: {task_class.name}")
    print(f"Files: {n}")
    print(f"System prompt + examples: {cached_tokens:,} tokens (cached)")
    print(f"User prompt tokens: mean={arr.mean():.0f}, "
          f"median={np.median(arr):.0f}, "
          f"min={arr.min()}, max={arr.max()}, "
          f"std={arr.std():.0f}")
    print(f"Total input per call: ~{total_input_per_call:,} tokens "
          f"({cached_tokens:,} cached + {int(arr.mean()):,} prompt)")
    print(f"Est. output per call: {output_tokens}")
    print()

    print_estimate(
        input_tokens=total_input_per_call,
        output_tokens=output_tokens,
        n_calls=n,
        cached_tokens=cached_tokens,
        models=models,
    )
