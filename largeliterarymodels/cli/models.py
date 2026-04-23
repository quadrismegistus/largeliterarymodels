"""Short model tags → fully-qualified model IDs.

Convention: `<family>-<variant>[-<backend>]`. Default (no suffix) =
LM Studio GGUF (parallel-capable). `-mlx` suffix = LM Studio MLX variant
(faster on Apple silicon but no parallel).
"""


MODEL_TAGS: dict[str, str] = {
    # Gemma (Google, via LM Studio)
    'gemma-e2b':     'lmstudio/gemma-4-e2b-it',
    'gemma-e2b-mlx': 'lmstudio/gemma-4-e2b-it-mlx',
    'gemma-31b':     'lmstudio/gemma-4-31b-it',
    'gemma-31b-mlx': 'lmstudio/gemma-4-31b-it-mlx',

    # Qwen (via LM Studio)
    'qwen-27b':      'lmstudio/qwen3.5-27b',
    'qwen-35b':      'lmstudio/qwen3.5-35b-a3b',   # MoE, ~3B active

    # Llama (via LM Studio)
    'llama-70b':     'lmstudio/meta-llama-3.1-70b-instruct',

    # Anthropic (family name already disambiguates)
    'sonnet':        'claude-sonnet-4-6',
    'opus':          'claude-opus-4-7',
    'haiku':         'claude-haiku-4-5-20251001',

    # Google Gemini
    'gemini-flash':  'gemini-2.5-flash',
    'gemini-pro':    'gemini-2.5-pro',
}


def resolve_model(tag: str) -> str:
    """Return fully-qualified model ID for a short tag, or pass through
    if `tag` looks fully qualified (contains `/` or starts with a provider
    prefix like `claude-`, `gpt-`, `gemini-`)."""
    if tag in MODEL_TAGS:
        return MODEL_TAGS[tag]
    if '/' in tag or tag.startswith(('claude-', 'gpt-', 'gemini-')):
        return tag
    raise SystemExit(
        f"Unknown model tag: {tag!r}. Known: {sorted(MODEL_TAGS)}. "
        f"Or pass a fully-qualified ID like 'lmstudio/...' or 'claude-...'."
    )
