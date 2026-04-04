"""Core LLM class: unified interface for text generation with HashStash caching."""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashstash import HashStash
from tqdm import tqdm
from .providers import route_provider, check_api_keys

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
                   max_tokens=DEFAULT_MAX_TOKENS, **kwargs):
    """Dispatch a prompt to the appropriate provider. Used as the cacheable function."""
    provider_fn = route_provider(model)
    return provider_fn(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def _make_key(prompt, model, system_prompt=None, temperature=DEFAULT_TEMPERATURE,
              max_tokens=DEFAULT_MAX_TOKENS, schema_name=None):
    """Build the dict used as a HashStash key."""
    key = {
        "prompt": prompt,
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if schema_name:
        key["schema"] = schema_name
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
    """Extract JSON from an LLM response, handling markdown fencing and surrounding text."""
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
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def _validate_parsed(data, schema):
    """Validate parsed JSON against the Pydantic schema."""
    is_list, item_schema = _unwrap_schema(schema)
    if is_list:
        if not isinstance(data, list):
            data = [data]
        return [item_schema.model_validate(item) for item in data]
    else:
        return item_schema.model_validate(data)


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
                 max_tokens=None, force=False, **kwargs):
        """Generate text from the LLM, with caching.

        Args:
            prompt: The user prompt.
            system_prompt: Override instance system_prompt for this call.
            temperature: Override instance temperature for this call.
            max_tokens: Override instance max_tokens for this call.
            force: If True, bypass cache and force a new generation.
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: The generated text.
        """
        system_prompt, temperature, max_tokens = self._resolve(
            system_prompt, temperature, max_tokens,
        )
        key = _make_key(prompt, self.model, system_prompt, temperature, max_tokens)

        if not force and key in self.stash:
            return self.stash[key]

        result = _call_provider(
            prompt=prompt,
            model=self.model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.stash[key] = result
        return result

    def extract(self, prompt, schema, system_prompt=None, examples=None,
                temperature=None, max_tokens=None, force=False, retries=1, **kwargs):
        """Extract structured data from text using a Pydantic schema.

        Args:
            prompt: The input text to extract from.
            schema: A Pydantic BaseModel class, or list[BaseModel] for multiple items.
            system_prompt: Domain-specific instructions prepended to the schema prompt.
            examples: Few-shot examples as list of (input_str, output) tuples.
                      Output can be a BaseModel instance, dict, or list thereof.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
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
                        schema_name=s_name)

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

        raise ValueError(
            f"Failed to extract valid {s_name} after {1 + retries} attempts. "
            f"Last error: {last_error}"
        )

    def map(self, prompts, system_prompt=None, temperature=None,
            max_tokens=None, num_workers=4, force=False, **kwargs):
        """Generate text for multiple prompts, with caching and parallelism.

        Args:
            prompts: List of prompt strings.
            system_prompt: Override instance system_prompt.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
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
            key = _make_key(prompt, self.model, system_prompt, temperature, max_tokens)
            if not force and key in self.stash:
                results[i] = self.stash[key]
            else:
                to_compute.append((i, prompt, key))

        if not to_compute:
            return results

        def _do_one(item):
            i, prompt, key = item
            result = _call_provider(
                prompt=prompt,
                model=self.model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
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
                    temperature=None, max_tokens=None, num_workers=4,
                    force=False, retries=1, **kwargs):
        """Extract structured data from multiple prompts, with caching and parallelism.

        Args:
            prompts: List of input texts.
            schema: Pydantic BaseModel class or list[BaseModel].
            system_prompt: Domain-specific instructions.
            examples: Few-shot examples as list of (input_str, output) tuples.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            num_workers: Number of parallel threads (default 4).
            force: If True, bypass cache.
            retries: Number of retries on malformed JSON (default 1).
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

        results = [None] * len(prompts)
        to_compute = []

        for i, prompt in enumerate(prompts):
            key = _make_key(prompt, self.model, full_system, temperature, max_tokens,
                            schema_name=s_name)
            if not force and key in self.stash:
                cached = self.stash[key]
                if isinstance(cached, str):
                    try:
                        results[i] = _validate_parsed(_parse_json_response(cached), schema)
                    except Exception:
                        to_compute.append((i, prompt, key))
                else:
                    results[i] = cached
            else:
                to_compute.append((i, prompt, key))

        if not to_compute:
            return results

        def _do_one(item):
            i, prompt, key = item
            last_error = None
            for attempt in range(1 + retries):
                call_prompt = prompt
                if attempt > 0:
                    call_prompt = (
                        f"Your previous response was not valid JSON. "
                        f"Return ONLY valid JSON matching the schema, nothing else.\n\n"
                        f"{prompt}"
                    )
                raw = _call_provider(
                    prompt=call_prompt,
                    model=self.model,
                    system_prompt=full_system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                try:
                    parsed = _parse_json_response(raw)
                    result = _validate_parsed(parsed, schema)
                    self.stash[key] = raw
                    return i, result
                except Exception as e:
                    last_error = e
                    continue
            raise ValueError(
                f"Failed to extract valid {s_name} for prompt {i} after {1 + retries} attempts. "
                f"Last error: {last_error}"
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_do_one, item): item for item in to_compute}
            for future in tqdm(as_completed(futures), total=len(to_compute),
                               desc=f"Extracting {s_name} ({self.model})"):
                i, result = future.result()
                results[i] = result

        return results

    def __repr__(self):
        return f"LLM(model={self.model!r})"
