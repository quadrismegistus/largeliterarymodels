"""LLM provider backends: Anthropic, OpenAI, Google GenAI.

Each provider function takes a standard set of arguments and returns the
response text as a string. No litellm — direct SDK calls only.

Supports multimodal inputs via the `images` parameter: a list of file paths,
bytes, or PIL Image objects.
"""

import base64
import io
import os


def _get_key(env_var):
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Missing {env_var} in environment")
    return key


def route_provider(model):
    """Return the appropriate provider function for a model string."""
    model_lower = model.lower()
    if model_lower.startswith(("local/", "ollama/", "vllm/", "lmstudio/")):
        return call_local
    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        return call_anthropic
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or model_lower.startswith("openai/"):
        return call_openai
    elif "gemini" in model_lower or model_lower.startswith("google/"):
        return call_google
    else:
        raise ValueError(
            f"Cannot determine provider for model '{model}'. "
            f"Model name should contain 'claude', 'gpt', or 'gemini', "
            f"or use a prefix like 'anthropic/', 'openai/', 'google/', or 'local/'."
        )


def _strip_prefix(model):
    """Remove provider prefix like 'anthropic/' or 'openai/' from model name."""
    for prefix in ("anthropic/", "openai/", "google/",
                   "local/", "ollama/", "vllm/", "lmstudio/"):
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


def call_anthropic(prompt, model="claude-sonnet-4-20250514", system_prompt=None,
                   temperature=0.7, max_tokens=4096, images=None, **kwargs):
    """Call Anthropic's Claude API directly."""
    from anthropic import Anthropic

    client = Anthropic(api_key=_get_key("ANTHROPIC_API_KEY"))
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
    # claude-opus-4-7 and later deprecate `temperature`; skip when unsupported.
    if temperature is not None and 'opus-4-7' not in model:
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


def call_openai(prompt, model="gpt-4o-mini", system_prompt=None,
                temperature=0.7, max_tokens=4096, images=None, **kwargs):
    """Call OpenAI's API directly."""
    from openai import OpenAI

    client = OpenAI(api_key=_get_key("OPENAI_API_KEY"))
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


def call_google(prompt, model="gemini-3.1-pro-preview", system_prompt=None,
                temperature=0.7, max_tokens=4096, images=None, **kwargs):
    """Call Google's GenAI API directly."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment")

    client = genai.Client(api_key=api_key)
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
               temperature=0.7, max_tokens=4096, images=None, **kwargs):
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
    client = OpenAI(api_key="local", base_url=base_url)
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
    effective_max = max_tokens
    model_lower = model.lower()
    if "qwen" in model_lower:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    effective_max = max_tokens

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=effective_max,
            extra_body=extra_body or None,
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
