"""Core LLM class: unified interface for text generation with HashStash caching."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashstash import HashStash
from tqdm import tqdm
from .providers import route_provider, check_api_keys

log = logging.getLogger(__name__)

# Model constants
CLAUDE_OPUS = "claude-opus-4-6"
CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GEMINI_PRO = "gemini-2.5-pro"
GEMINI_FLASH = "gemini-2.5-flash"

DEFAULT_MODEL = CLAUDE_SONNET
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
STASH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stash")


def _call_provider(prompt, model, system_prompt=None, temperature=DEFAULT_TEMPERATURE,
                   max_tokens=DEFAULT_MAX_TOKENS, images=None, **kwargs):
    """Dispatch a prompt to the appropriate provider. Used as the cacheable function."""
    provider_fn = route_provider(model)
    return provider_fn(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        images=images,
        **kwargs,
    )


def _make_key(prompt, model, system_prompt=None, temperature=DEFAULT_TEMPERATURE,
              max_tokens=DEFAULT_MAX_TOKENS, schema_name=None, images=None,
              metadata=None):
    """Build the dict used as a HashStash key.

    Args:
        metadata: Optional dict of user-defined metadata (e.g. page_number,
                  source_file). Stored in the key for retrieval via task.df
                  but does not affect the LLM call.
        images: Optional list of image paths. Paths are included in the key
                for cache differentiation; the actual bytes are not stored.
    """
    key = {
        "prompt": prompt,
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if schema_name:
        key["schema"] = schema_name
    if images:
        # Store paths/identifiers for cache key differentiation
        key["images"] = [
            img if isinstance(img, str) else f"<bytes:{len(img)}>"
            if isinstance(img, bytes) else f"<image:{id(img)}>"
            for img in images
        ]
    if metadata:
        key["metadata"] = metadata
    return key


def _schema_to_json_spec(schema):
    """Convert a Pydantic model (or list[Model]) to a JSON schema description for the prompt."""
    is_list, item_schema = _unwrap_schema(schema)
    json_schema = item_schema.model_json_schema()
    schema_json = json.dumps(json_schema, indent=2)
    if is_list:
        return f"a JSON array of objects, where each object matches this schema:\n{schema_json}"
    else:
        return f"a JSON object matching this schema:\n{schema_json}"


def _unwrap_schema(schema):
    """Unwrap list[Model] into (True, Model) or (False, schema)."""
    origin = getattr(schema, "__origin__", None)
    if origin is list:
        args = schema.__args__
        return True, args[0]
    return False, schema


def _schema_name(schema):
    """Get a stable name for a schema, handling list[Model]."""
    is_list, item_schema = _unwrap_schema(schema)
    name = item_schema.__name__
    return f"list[{name}]" if is_list else name


def _format_examples(examples, schema):
    """Format few-shot examples into prompt text."""
    if not examples:
        return ""
    parts = []
    for i, (input_text, output) in enumerate(examples, 1):
        if hasattr(output, "model_dump_json"):
            output_json = output.model_dump_json(indent=2)
        elif isinstance(output, dict):
            output_json = json.dumps(output, indent=2)
        elif isinstance(output, list):
            output_json = json.dumps(
                [o.model_dump() if hasattr(o, "model_dump") else o for o in output],
                indent=2,
            )
        else:
            output_json = str(output)
        parts.append(f"Example {i} input:\n{input_text}\n\nExample {i} output:\n{output_json}")
    return "\n\n---\n\n".join(parts)


def _build_extract_prompt(prompt, schema, system_prompt=None, examples=None):
    """Build the full system prompt and user prompt for structured extraction."""
    schema_spec = _schema_to_json_spec(schema)

    system_parts = []
    if system_prompt:
        system_parts.append(system_prompt)
    system_parts.append(
        f"You must respond with ONLY valid JSON matching the following specification — "
        f"no markdown fencing, no commentary, no extra text.\n\n"
        f"Respond with {schema_spec}"
    )

    examples_text = _format_examples(examples, schema)
    if examples_text:
        system_parts.append(f"Here are some examples:\n\n{examples_text}")

    full_system = "\n\n".join(system_parts)
    return full_system, prompt


def _parse_json_response(text):
    """Extract JSON from an LLM response, handling markdown fencing and surrounding text.

    Falls back to json_repair for common malformations (e.g. missing opening
    quotes on string values — observed with qwen3.5 on large multi-field schemas).
    """
    text = text.strip()
    # strip markdown fencing
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # try as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find first [ or { and match to last ] or }
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    # Last resort: json_repair for malformed output from local models.
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        # repair_json returns '' when completely unrecoverable.
        if repaired not in ('', None):
            return repaired
    except ImportError:
        pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def _validate_parsed(data, schema):
    """Validate parsed JSON against the Pydantic schema."""
    is_list, item_schema = _unwrap_schema(schema)
    data = _unwrap_envelopes(data, item_schema)
    if is_list:
        if not isinstance(data, list):
            data = [data]
        return [item_schema.model_validate(_unwrap_envelopes(item, item_schema))
                for item in data]
    else:
        return item_schema.model_validate(data)


def _unwrap_envelopes(data, item_schema):
    """Apply all known output-envelope unwraps in sequence."""
    data = _unwrap_schema_envelope(data, item_schema)
    data = _unwrap_per_field_envelope(data)
    return data


def _unwrap_schema_envelope(data, item_schema):
    """Some models (observed: gemma4) echo the JSON-schema structure back as
    an output envelope: {"properties": {...actual fields...}}. Detect and
    unwrap when the inner dict clearly matches the schema better than the
    outer one."""
    if not (isinstance(data, dict) and "properties" in data
            and isinstance(data["properties"], dict)):
        return data
    try:
        expected = set(item_schema.model_fields.keys())
    except Exception:
        return data
    outer_match = len(set(data.keys()) & expected)
    inner_match = len(set(data["properties"].keys()) & expected)
    if inner_match > outer_match and inner_match > 0:
        return data["properties"]
    return data


def _unwrap_per_field_envelope(data):
    """Some models (observed: llama-3.1-70b on large schemas) wrap each field
    value as a JSON-schema field descriptor containing a 'value' key — e.g.
    {"type": "boolean", "value": false} or {"description": "...", "title": "...",
    "type": "boolean", "value": false}. Detect when every value in the dict
    has both 'type' and 'value' keys and unwrap to the bare values."""
    if not isinstance(data, dict) or not data:
        return data
    for v in data.values():
        if not (isinstance(v, dict) and "type" in v and "value" in v):
            return data
    return {k: v["value"] for k, v in data.items()}


class LLM:
    """Unified LLM interface with automatic caching via HashStash.

    Usage:
        llm = LLM("claude-sonnet-4-20250514")
        text = llm.generate("What is the plot of Pamela?")

        # cached: identical calls return instantly
        text2 = llm.generate("What is the plot of Pamela?")

        # structured extraction
        from pydantic import BaseModel, Field
        class Character(BaseModel):
            name: str
            role: str = Field(description="Role in the narrative")
        characters = llm.extract("Describe the characters in Pamela.",
                                  schema=list[Character])

        # with images
        llm = LLM("gemini-2.5-flash")
        text = llm.generate("Describe this page.", images=["page1.png"])
    """

    def __init__(self, model=DEFAULT_MODEL, system_prompt=None, temperature=DEFAULT_TEMPERATURE,
                 max_tokens=DEFAULT_MAX_TOKENS, stash=None):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stash = stash if stash is not None else HashStash(
            STASH_PATH, engine="pairtree", append_mode=True,
        )

    def _resolve(self, system_prompt=None, temperature=None, max_tokens=None):
        """Resolve per-call overrides against instance defaults."""
        return (
            system_prompt if system_prompt is not None else self.system_prompt,
            temperature if temperature is not None else self.temperature,
            max_tokens if max_tokens is not None else self.max_tokens,
        )

    def generate(self, prompt, system_prompt=None, temperature=None,
                 max_tokens=None, images=None, metadata=None, force=False, **kwargs):
        """Generate text from the LLM, with caching.

        Args:
            prompt: The user prompt.
            system_prompt: Override instance system_prompt for this call.
            temperature: Override instance temperature for this call.
            max_tokens: Override instance max_tokens for this call.
            images: List of images (file paths, bytes, or PIL Images).
            metadata: Dict of user-defined metadata to store with the cache entry.
            force: If True, bypass cache and force a new generation.
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: The generated text.
        """
        system_prompt, temperature, max_tokens = self._resolve(
            system_prompt, temperature, max_tokens,
        )
        key = _make_key(prompt, self.model, system_prompt, temperature, max_tokens,
                        images=images, metadata=metadata)

        if not force and key in self.stash:
            return self.stash[key]

        result = _call_provider(
            prompt=prompt,
            model=self.model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            images=images,
            **kwargs,
        )
        self.stash[key] = result
        return result

    def extract(self, prompt, schema, system_prompt=None, examples=None,
                temperature=None, max_tokens=None, images=None, metadata=None,
                force=False, retries=1, **kwargs):
        """Extract structured data from text using a Pydantic schema.

        Args:
            prompt: The input text to extract from.
            schema: A Pydantic BaseModel class, or list[BaseModel] for multiple items.
            system_prompt: Domain-specific instructions prepended to the schema prompt.
            examples: Few-shot examples as list of (input_str, output) tuples.
                      Output can be a BaseModel instance, dict, or list thereof.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            images: List of images (file paths, bytes, or PIL Images).
            metadata: Dict of user-defined metadata to store with the cache entry.
            force: If True, bypass cache.
            retries: Number of retries on malformed JSON (default 1).
            **kwargs: Additional provider-specific arguments.

        Returns:
            A validated Pydantic model instance (or list of instances).
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        full_system, user_prompt = _build_extract_prompt(
            prompt, schema, system_prompt=system_prompt, examples=examples,
        )
        s_name = _schema_name(schema)
        key = _make_key(user_prompt, self.model, full_system, temperature, max_tokens,
                        schema_name=s_name, images=images, metadata=metadata)

        if not force and key in self.stash:
            cached = self.stash[key]
            if isinstance(cached, str):
                return _validate_parsed(_parse_json_response(cached), schema)
            return cached

        last_error = None
        for attempt in range(1 + retries):
            if attempt == 0:
                call_system = full_system
                call_prompt = user_prompt
            else:
                log.warning(
                    "extract retry %d/%d for %s (model=%s): %s",
                    attempt, retries, s_name, self.model, last_error,
                )
                call_system = full_system
                call_prompt = (
                    f"Your previous response was not valid JSON. "
                    f"Return ONLY valid JSON matching the schema, nothing else.\n\n"
                    f"{user_prompt}"
                )

            raw = _call_provider(
                prompt=call_prompt,
                model=self.model,
                system_prompt=call_system,
                temperature=temperature,
                max_tokens=max_tokens,
                images=images,
                **kwargs,
            )

            try:
                parsed = _parse_json_response(raw)
                result = _validate_parsed(parsed, schema)
                self.stash[key] = raw
                return result
            except (ValueError, json.JSONDecodeError, Exception) as e:
                last_error = e
                continue

        print(f"Raw response: {raw}")
        raise ValueError(
            f"Failed to extract valid {s_name} after {1 + retries} attempts. "
            f"Last error: {last_error}"
            f"Raw response: {raw}"
        )

    def map(self, prompts, system_prompt=None, temperature=None,
            max_tokens=None, images_list=None, metadata_list=None,
            num_workers=4, force=False, **kwargs):
        """Generate text for multiple prompts, with caching and parallelism.

        Args:
            prompts: List of prompt strings.
            system_prompt: Override instance system_prompt.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            images_list: List of image lists, one per prompt (or None).
            metadata_list: List of metadata dicts, one per prompt (or None).
            num_workers: Number of parallel threads (default 4).
            force: If True, bypass cache and force new generations.
            **kwargs: Additional provider-specific arguments.

        Returns:
            list[str]: Generated texts in the same order as prompts.
        """
        system_prompt, temperature, max_tokens = self._resolve(
            system_prompt, temperature, max_tokens,
        )

        results = [None] * len(prompts)
        to_compute = []

        for i, prompt in enumerate(prompts):
            images = images_list[i] if images_list else None
            metadata = metadata_list[i] if metadata_list else None
            key = _make_key(prompt, self.model, system_prompt, temperature, max_tokens,
                            images=images, metadata=metadata)
            if not force and key in self.stash:
                results[i] = self.stash[key]
            else:
                to_compute.append((i, prompt, key, images))

        total = len(prompts)
        fresh = len(to_compute)
        cached = total - fresh
        if total >= 10:
            log.info("generate_map: %d/%d cached, %d API calls needed (model=%s)",
                     cached, total, fresh, self.model)

        if not to_compute:
            return results

        def _do_one(item):
            i, prompt, key, images = item
            result = _call_provider(
                prompt=prompt,
                model=self.model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                images=images,
                **kwargs,
            )
            self.stash[key] = result
            return i, result

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_do_one, item): item for item in to_compute}
            for future in tqdm(as_completed(futures), total=len(to_compute), desc=f"Generating ({self.model})"):
                i, result = future.result()
                results[i] = result

        return results

    def extract_map(self, prompts, schema, system_prompt=None, examples=None,
                    temperature=None, max_tokens=None, images_list=None,
                    metadata_list=None, num_workers=4,
                    force=False, retries=1, verbose=False, **kwargs):
        """Extract structured data from multiple prompts, with caching and parallelism.

        Args:
            prompts: List of input texts.
            schema: Pydantic BaseModel class or list[BaseModel].
            system_prompt: Domain-specific instructions.
            examples: Few-shot examples as list of (input_str, output) tuples.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            images_list: List of image lists, one per prompt (or None).
            metadata_list: List of metadata dicts, one per prompt (or None).
            num_workers: Number of parallel threads (default 4).
            force: If True, bypass cache.
            retries: Number of retries on malformed JSON (default 1).
            verbose: If True, print a compact per-call summary as each result
                lands (plays nicely with tqdm via tqdm.write). If a callable,
                use it as a custom formatter — signature
                (i: int, prompt: str, metadata: dict|None, result) -> str.
            **kwargs: Additional provider-specific arguments.

        Returns:
            list: Validated Pydantic model instances (or lists thereof) in prompt order.
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        full_system, _ = _build_extract_prompt(
            "", schema, system_prompt=system_prompt, examples=examples,
        )
        s_name = _schema_name(schema)

        def _default_verbose_line(i, prompt, metadata, result, from_cache=False):
            meta_str = ""
            if isinstance(metadata, dict) and metadata:
                meta_str = " ".join(f"{k}={v}" for k, v in metadata.items() if v not in ("", None))
            try:
                if isinstance(result, list):
                    payload = f"[list x{len(result)}]"
                    if result:
                        payload += " " + ", ".join(
                            f"{k}={v!r}" for k, v in list(result[0].model_dump().items())[:3]
                        )
                else:
                    payload = ", ".join(
                        f"{k}={v!r}" for k, v in list(result.model_dump().items())[:4]
                    )
            except Exception:
                payload = str(result)[:120]
            prompt_head = (prompt or "").splitlines()[0][:60]
            tag = "⊛" if from_cache else "→"
            return f"[{i:>5}] {meta_str}  {tag} {payload}  ({prompt_head!r})"

        def _emit_verbose(i, prompt, metadata, result, from_cache):
            try:
                line = (verbose(i, prompt, metadata, result) if callable(verbose)
                        else _default_verbose_line(i, prompt, metadata, result, from_cache))
                tqdm.write(line)
            except Exception as e:
                tqdm.write(f"[{i}] <verbose formatter error: {e}>")

        results = [None] * len(prompts)
        to_compute = []

        for i, prompt in enumerate(prompts):
            images = images_list[i] if images_list else None
            metadata = metadata_list[i] if metadata_list else None
            key = _make_key(prompt, self.model, full_system, temperature, max_tokens,
                            schema_name=s_name, images=images, metadata=metadata)
            if not force and key in self.stash:
                cached = self.stash[key]
                if isinstance(cached, str):
                    try:
                        results[i] = _validate_parsed(_parse_json_response(cached), schema)
                    except Exception:
                        to_compute.append((i, prompt, key, images))
                        continue
                else:
                    results[i] = cached
                if verbose and results[i] is not None:
                    _emit_verbose(i, prompt, metadata, results[i], from_cache=True)
            else:
                to_compute.append((i, prompt, key, images))

        total = len(prompts)
        fresh = len(to_compute)
        n_cached = total - fresh
        if total >= 10:
            log.info("extract_map: %d/%d cached, %d API calls needed (model=%s)",
                     n_cached, total, fresh, self.model)
            if n_cached == 0 and fresh >= 100:
                try:
                    has_old_entries = next(iter(self.stash.items()), None) is not None
                except Exception:
                    has_old_entries = False
                if has_old_entries:
                    log.warning(
                        "extract_map: 0/%d cached despite existing entries in %s's stash. "
                        "System prompt, examples, schema, temperature, or max_tokens may "
                        "have changed since the last run — previous cache keys are unreachable.",
                        total, s_name,
                    )

        if not to_compute:
            return results

        def _do_one(item):
            i, prompt, key, images = item
            last_error = None
            raw = None
            for attempt in range(1 + retries):
                call_prompt = prompt
                if attempt > 0:
                    log.warning(
                        "extract_map retry %d/%d for prompt %d (model=%s): %s",
                        attempt, retries, i, self.model, last_error,
                    )
                    call_prompt = (
                        f"Your previous response was not valid JSON. "
                        f"Return ONLY valid JSON matching the schema, nothing else.\n\n"
                        f"{prompt}"
                    )
                try:
                    raw = _call_provider(
                        prompt=call_prompt,
                        model=self.model,
                        system_prompt=full_system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        images=images,
                        **kwargs,
                    )
                    parsed = _parse_json_response(raw)
                    result = _validate_parsed(parsed, schema)
                    self.stash[key] = raw
                    return i, result
                except Exception as e:
                    last_error = e
                    continue
            # Exhausted retries — return None so the pool drains cleanly and
            # the pilot continues past individual failures rather than hanging.
            log.error(
                "extract_map giving up on prompt %d after %d attempts (model=%s). "
                "Last error: %s. Raw (truncated): %s",
                i, 1 + retries, self.model, last_error,
                (raw or '')[:400],
            )
            return i, None

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_do_one, item): item for item in to_compute}
            for future in tqdm(as_completed(futures), total=len(to_compute),
                               desc=f"Extracting {s_name} ({self.model})"):
                i, result = future.result()
                results[i] = result
                if verbose:
                    prompt = prompts[i]
                    metadata = metadata_list[i] if metadata_list else None
                    _emit_verbose(i, prompt, metadata, result, from_cache=False)

        return results

    def __repr__(self):
        return f"LLM(model={self.model!r})"
