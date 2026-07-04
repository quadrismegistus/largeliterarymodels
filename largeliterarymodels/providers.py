"""LLM provider backends: Anthropic, OpenAI, Google GenAI, Claude CLI.

Each provider function takes a standard set of arguments and returns the
response text as a string. No litellm — direct SDK calls only.

Supports multimodal inputs via the `images` parameter: a list of file paths,
bytes, or PIL Image objects.
"""

import base64
import io
import os

# SDK clients are memoized per (provider, credentials, base_url, timeout):
# each client owns an httpx connection pool, and constructing one per call
# churns TCP connections/file descriptors under parallel batch runs.
_CLIENT_CACHE = {}


def _cached_client(cache_key, factory):
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = factory()
        _CLIENT_CACHE[cache_key] = client
    return client


def _get_key(env_var):
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Missing {env_var} in environment")
    return key


# DeepSeek's hosted API models. Bare names like 'deepseek-r1:8b' are local
# checkpoints and must NOT route to the paid API — use an explicit
# 'ollama/'/'lmstudio/' prefix for those.
_DEEPSEEK_API_MODELS = ("deepseek-chat", "deepseek-reasoner")


def route_provider(model):
    """Return the appropriate provider function for a model string."""
    model_lower = model.lower()
    if model_lower.startswith(("local/", "ollama/", "vllm/", "lmstudio/")):
        return call_local
    if model_lower.startswith("claude-cli/"):
        return call_claude_cli
    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        return call_anthropic
    elif model_lower.startswith("deepseek/") or model_lower in _DEEPSEEK_API_MODELS:
        return call_deepseek
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or model_lower.startswith("openai/"):
        return call_openai
    elif "gemini" in model_lower or model_lower.startswith("google/"):
        return call_google
    else:
        raise ValueError(
            f"Cannot determine provider for model '{model}'. "
            f"Model name should contain 'claude', 'gpt', or 'gemini', "
            f"or use a prefix like 'anthropic/', 'openai/', 'google/', 'deepseek/', 'claude-cli/', or 'local/'."
        )


def _strip_prefix(model):
    """Remove provider prefix like 'anthropic/' or 'openai/' from model name."""
    for prefix in ("anthropic/", "openai/", "google/", "deepseek/",
                   "claude-cli/", "local/", "ollama/", "vllm/", "lmstudio/"):
        if model.lower().startswith(prefix):
            return model[len(prefix):]
    return model


def _load_image_bytes(image):
    """Convert an image (path, bytes, or PIL Image) to (bytes, mime_type)."""
    if isinstance(image, str):
        # File path
        with open(image, "rb") as f:
            data = f.read()
        ext = os.path.splitext(image)[1].lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        }
        return data, mime_map.get(ext, "image/png")
    elif isinstance(image, bytes):
        return image, "image/png"
    else:
        # PIL Image
        buf = io.BytesIO()
        fmt = getattr(image, "format", "PNG") or "PNG"
        image.save(buf, format=fmt)
        mime = f"image/{fmt.lower()}"
        return buf.getvalue(), mime


# Model families where the API rejects sampling params (temperature/top_p).
# Substring check against the stripped model id; extend as new families ship.
_NO_TEMPERATURE_MODELS = ("opus-4-7", "opus-4-8", "sonnet-5", "fable", "mythos")


def _supports_temperature(model):
    model_lower = model.lower()
    return not any(tag in model_lower for tag in _NO_TEMPERATURE_MODELS)


def call_anthropic(prompt, model="claude-sonnet-4-6", system_prompt=None,
                   temperature=0.7, max_tokens=4096, images=None,
                   timeout=None, **kwargs):
    """Call Anthropic's Claude API directly."""
    from anthropic import Anthropic

    api_key = _get_key("ANTHROPIC_API_KEY")
    client = _cached_client(
        ("anthropic", api_key, timeout),
        lambda: Anthropic(api_key=api_key) if timeout is None
        else Anthropic(api_key=api_key, timeout=timeout),
    )
    model = _strip_prefix(model)

    # Build content blocks
    if images:
        content = []
        for img in images:
            data, mime = _load_image_bytes(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(data).decode("utf-8"),
                },
            })
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt

    api_kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    # Newer model families reject sampling params entirely; skip when unsupported.
    if temperature is not None and _supports_temperature(model):
        api_kwargs["temperature"] = temperature
    # Mark system (which includes few-shot examples per llm._build_extract_prompt)
    # as cacheable. Task batches reuse the same system across hundreds-to-thousands
    # of calls; the per-call user message is tiny. Caching cuts input cost ~10x on
    # cache hits. Below the model's cache threshold Anthropic silently skips.
    if system_prompt:
        api_kwargs["system"] = [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

    response = client.messages.create(**api_kwargs)
    return response.content[0].text


def call_claude_cli(prompt, model="claude-cli/opus", system_prompt=None,
                    temperature=0.7, max_tokens=4096, images=None, **kwargs):
    """Call Claude via the `claude` CLI tool (`claude -p --bare`).

    Uses the locally installed Claude Code CLI, which authenticates via
    the user's existing subscription. No ANTHROPIC_API_KEY needed.

    Model string after prefix selects the model:
        claude-cli/opus   → --model claude-opus-4-6
        claude-cli/sonnet → --model claude-sonnet-4-6
        claude-cli/haiku  → --model claude-haiku-4-5
        claude-cli/<full> → --model <full>  (pass-through)
    """
    import json
    import shutil
    import subprocess

    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "Claude CLI not found. Install from https://claude.com/claude-code"
        )

    model_name = _strip_prefix(model)
    model_map = {
        "opus": "claude-opus-4-6",
        "sonnet": "claude-sonnet-4-6",
        "haiku": "claude-haiku-4-5",
    }
    model_name = model_map.get(model_name, model_name)

    cmd = [
        claude_bin, "-p", "--bare",
        "--output-format", "json",
        "--model", model_name,
    ]

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"<system>\n{system_prompt}\n</system>\n\n{prompt}"

    result = subprocess.run(
        cmd, input=full_prompt,
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): "
            f"{result.stderr[:500]}"
        )

    try:
        data = json.loads(result.stdout)
        return data.get("result", result.stdout)
    except json.JSONDecodeError:
        return result.stdout


def call_openai(prompt, model="gpt-4o-mini", system_prompt=None,
                temperature=0.7, max_tokens=4096, images=None,
                timeout=None, **kwargs):
    """Call OpenAI's API directly."""
    from openai import OpenAI

    api_key = _get_key("OPENAI_API_KEY")
    client = _cached_client(
        ("openai", api_key, timeout),
        lambda: OpenAI(api_key=api_key) if timeout is None
        else OpenAI(api_key=api_key, timeout=timeout),
    )
    model = _strip_prefix(model)

    # Build content
    if images:
        content = []
        for img in images:
            data, mime = _load_image_bytes(img)
            b64 = base64.b64encode(data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_deepseek(prompt, model="deepseek/deepseek-chat", system_prompt=None,
                  temperature=0.7, max_tokens=4096, images=None,
                  timeout=None, **kwargs):
    """Call DeepSeek's API (OpenAI-compatible, text-only)."""
    from openai import OpenAI

    if images:
        raise ValueError(
            "DeepSeek's chat API is text-only; images are not supported."
        )

    api_key = _get_key("DEEPSEEK_API_KEY")
    client = _cached_client(
        ("deepseek", api_key, timeout),
        lambda: OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        if timeout is None
        else OpenAI(api_key=api_key, base_url="https://api.deepseek.com",
                    timeout=timeout),
    )
    model = _strip_prefix(model)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_google(prompt, model="gemini-3.1-pro-preview", system_prompt=None,
                temperature=0.7, max_tokens=4096, images=None,
                timeout=None, **kwargs):
    """Call Google's GenAI API directly."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment")

    def _make_google_client():
        if timeout is None:
            return genai.Client(api_key=api_key)
        # google-genai takes timeout in milliseconds via http_options.
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    client = _cached_client(("google", api_key, timeout), _make_google_client)
    model = _strip_prefix(model)

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system_prompt:
        config.system_instruction = system_prompt

    # Build contents
    if images:
        parts = []
        for img in images:
            data, mime = _load_image_bytes(img)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        parts.append(types.Part.from_text(text=prompt))
        contents = parts
    else:
        contents = prompt

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return response.text


_LOCAL_BACKEND_DEFAULTS = {
    "ollama":   "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm":     "http://localhost:8000/v1",
    # "local/" has no canonical backend; falls through to LOCAL_BASE_URL
    # or Ollama's port as the final default.
}


def _resolve_local_base_url(model: str) -> str:
    """Pick the base URL for a local/OSS model string.

    Priority:
      1. Per-backend env override (OLLAMA_BASE_URL, LMSTUDIO_BASE_URL, VLLM_BASE_URL)
      2. LOCAL_BASE_URL (legacy global override)
      3. Per-backend default (lmstudio→1234, ollama→11434, vllm→8000)
      4. Ollama default as final fallback

    The routing is prefix-pinned so `lmstudio/...` always hits LM Studio and
    `ollama/...` always hits Ollama, even when both servers are running.
    """
    prefix = None
    model_lower = model.lower()
    for p in _LOCAL_BACKEND_DEFAULTS:
        if model_lower.startswith(p + "/"):
            prefix = p
            break

    if prefix:
        per_backend = os.getenv(f"{prefix.upper()}_BASE_URL")
        if per_backend:
            return per_backend

    global_override = os.getenv("LOCAL_BASE_URL")
    if global_override:
        return global_override

    if prefix and _LOCAL_BACKEND_DEFAULTS.get(prefix):
        return _LOCAL_BACKEND_DEFAULTS[prefix]

    return "http://localhost:11434/v1"


def call_local(prompt, model="llama3.3", system_prompt=None,
               temperature=0.7, max_tokens=4096, images=None,
               timeout=None, **kwargs):
    """Call a local OpenAI-compatible API (Ollama, vLLM, LM Studio, llama.cpp server).

    Routing is prefix-pinned: `lmstudio/<model>` always hits LM Studio (port
    1234), `ollama/<model>` always hits Ollama (11434), `vllm/<model>` always
    hits vLLM (8000). Override any of them with the corresponding
    `<BACKEND>_BASE_URL` env var. `LOCAL_BASE_URL` still works as a global
    override that wins over the per-backend defaults.

    No API key required; the OpenAI SDK needs a non-empty string so we pass 'local'.

    Quality caveat: open-weight models are meaningfully below API-tier Claude
    and GPT for structured extraction with multilingual content, specialist
    literary knowledge, and strict JSON compliance. Treat as a complement
    (validation passes, dev iteration, cost-free experimentation) rather than
    a drop-in replacement for GenreTask / TranslationTask / PassageTask.
    """
    from openai import OpenAI

    base_url = _resolve_local_base_url(model)
    client = _cached_client(
        ("local", base_url, timeout),
        lambda: OpenAI(api_key="local", base_url=base_url) if timeout is None
        else OpenAI(api_key="local", base_url=base_url, timeout=timeout),
    )
    model = _strip_prefix(model)

    if images:
        content = []
        for img in images:
            data, mime = _load_image_bytes(img)
            b64 = base64.b64encode(data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    # Disable thinking mode for qwen3.5+ which defaults to reasoning — otherwise
    # max_tokens gets burned in `reasoning_content` leaving empty `content`. The
    # OpenAI-compat layer forwards this to Qwen's chat template.
    extra_body = {"cache_prompt": True}
    if "qwen" in model.lower():
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg or "econnrefused" in msg:
            raise RuntimeError(
                f"Local inference server at {base_url} is not reachable. "
                f"Is the expected backend running? Override via "
                f"OLLAMA_BASE_URL / LMSTUDIO_BASE_URL / VLLM_BASE_URL / "
                f"LOCAL_BASE_URL env if using a different host/port."
            ) from e
        raise
    return response.choices[0].message.content


def check_api_keys(verbose=False):
    """Check which provider API keys are available in the environment."""
    keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
    }
    available = {k: v for k, v in keys.items() if v}
    if verbose:
        for k, v in keys.items():
            status = "+" if v else "X"
            print(f"  {status} {k}")
    return available


def set_api_keys():
    """Interactively set API keys (safe for Colab — keys stay in memory only).

    Prompts for each provider key. Press Enter to skip. Uses getpass to
    mask input where available (Colab, terminals), falls back to input().
    Keys are set as environment variables for the current process only.
    """
    try:
        from getpass import getpass
        ask = getpass
    except ImportError:
        ask = input

    providers = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("OPENAI_API_KEY", "OpenAI (GPT)"),
        ("GEMINI_API_KEY", "Google (Gemini)"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
    ]
    for env_var, label in providers:
        existing = os.getenv(env_var)
        if existing:
            print(f"  + {label}: already set")
            continue
        val = ask(f"  {label} API key (Enter to skip): ").strip()
        if val:
            os.environ[env_var] = val
            print(f"  + {label}: set")
        else:
            print(f"  - {label}: skipped")
