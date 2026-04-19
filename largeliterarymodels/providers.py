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
        temperature=temperature,
        messages=[{"role": "user", "content": content}],
    )
    if system_prompt:
        api_kwargs["system"] = system_prompt

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


def call_google(prompt, model="gemini-2.5-flash", system_prompt=None,
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
    print("--------------------------------")
    print(response.candidates[0].finish_reason)  # SAFETY? MAX_TOKENS? STOP?
    print(response.text)
    print("--------------------------------")
    return response.text


def call_local(prompt, model="llama3.3", system_prompt=None,
               temperature=0.7, max_tokens=4096, images=None, **kwargs):
    """Call a local OpenAI-compatible API (Ollama, vLLM, LM Studio, llama.cpp server).

    Defaults to Ollama at http://localhost:11434/v1. Override by setting
    LOCAL_BASE_URL in the environment. No API key required; the OpenAI SDK
    needs a non-empty string so we pass 'local'.

    Quality caveat: open-weight models are meaningfully below API-tier Claude
    and GPT for structured extraction with multilingual content, specialist
    literary knowledge, and strict JSON compliance. Treat as a complement
    (validation passes, dev iteration, cost-free experimentation) rather than
    a drop-in replacement for GenreTask / TranslationTask / PassageTask.
    """
    from openai import OpenAI

    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg or "econnrefused" in msg:
            raise RuntimeError(
                f"Local inference server at {base_url} is not reachable. "
                f"Is Ollama (or vLLM / LM Studio) running? "
                f"Override via LOCAL_BASE_URL env if using a different host/port."
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
