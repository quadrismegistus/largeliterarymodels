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

    # Anthropic (family name already disambiguates; rolling aliases so all
    # three tags behave the same — pin a dated ID explicitly if you need
    # byte-exact reproducibility)
    'sonnet':        'claude-sonnet-4-6',
    'sonnet-5':      'claude-sonnet-5',
    'opus':          'claude-opus-4-7',
    'haiku':         'claude-haiku-4-5',

    # OpenAI (gpt-5 tier; these need max_completion_tokens, which
    # providers._chat_completion negotiates from the API's own error)
    'gpt5':          'openai/gpt-5.4',
    'gpt5-mini':     'openai/gpt-5.4-mini',
    'gpt5-nano':     'openai/gpt-5.4-nano',

    # Google Gemini
    'gemini-flash':  'gemini-2.5-flash',
    'gemini-pro':    'gemini-2.5-pro',

    # DeepSeek (hosted API; requires DEEPSEEK_API_KEY)
    'deepseek-pro':   'deepseek/deepseek-v4-pro',
    'deepseek-flash': 'deepseek/deepseek-v4-flash',
    # Retired name, still served as a server-side alias for v4-flash. Left
    # pointing at the alias on purpose: re-pointing it would silently change
    # the model string in existing cache keys and force a re-run. Prefer the
    # explicit tags above; call_deepseek warns once when the alias is used.
    'deepseek-chat':  'deepseek/deepseek-chat',
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
