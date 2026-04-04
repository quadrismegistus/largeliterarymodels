"""Task class: reusable structured extraction tasks with their own cache."""

import json
import os
import pandas as pd
from hashstash import HashStash
from .llm import (
    LLM, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, STASH_PATH,
    _parse_json_response, _validate_parsed, _unwrap_schema,
)


class Task:
    """A reusable structured extraction task.

    Bundles together a Pydantic schema, system prompt, few-shot examples,
    and retry config. Each task gets its own HashStash subdirectory.

    Subclass to define a task:

        class BechdelTask(Task):
            name = "bechdel"
            schema = BechdelResult
            system_prompt = "You are a literary critic assessing the Bechdel test..."
            examples = [
                ("INT. HOUSE...", BechdelResult(...)),
            ]

    Then use:

        task = BechdelTask()
        result = task.run(scene_text)
        results = task.map(scenes)
    """

    name = None  # defaults to class name if not set
    schema = None
    system_prompt = None
    examples = []
    retries = 1
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._stash = None

    @property
    def task_name(self):
        return self.name or self.__class__.__name__

    @property
    def stash(self):
        if self._stash is None:
            stash_dir = os.path.join(STASH_PATH, self.task_name)
            self._stash = HashStash(stash_dir, engine="pairtree", append_mode=True)
        return self._stash

    def _get_llm(self, model=None):
        """Get an LLM instance using this task's stash."""
        return LLM(
            model=model or DEFAULT_MODEL,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stash=self.stash,
        )

    def run(self, prompt, model=None, system_prompt=None, examples=None,
            force=False, **kwargs):
        """Extract structured data from a single input.

        Args:
            prompt: The input text to extract from.
            model: Override the default model.
            system_prompt: Override the task's system_prompt.
            examples: Override the task's few-shot examples.
            force: Bypass cache.
            **kwargs: Additional arguments passed to LLM.extract().

        Returns:
            Validated Pydantic model instance (or list thereof).
        """
        if self.schema is None:
            raise ValueError(f"Task '{self.name}' has no schema defined.")
        llm = self._get_llm(model)
        return llm.extract(
            prompt=prompt,
            schema=self.schema,
            system_prompt=system_prompt or self.system_prompt,
            examples=examples if examples is not None else self.examples,
            retries=self.retries,
            force=force,
            **kwargs,
        )

    def map(self, prompts, model=None, system_prompt=None, examples=None,
            num_workers=4, force=False, **kwargs):
        """Extract structured data from multiple inputs, with parallelism.

        Args:
            prompts: List of input texts.
            model: Override the default model.
            system_prompt: Override the task's system_prompt.
            examples: Override the task's few-shot examples.
            num_workers: Number of parallel threads.
            force: Bypass cache.
            **kwargs: Additional arguments passed to LLM.extract_map().

        Returns:
            list: Validated Pydantic model instances in prompt order.
        """
        if self.schema is None:
            raise ValueError(f"Task '{self.name}' has no schema defined.")
        llm = self._get_llm(model)
        return llm.extract_map(
            prompts=prompts,
            schema=self.schema,
            system_prompt=system_prompt or self.system_prompt,
            examples=examples if examples is not None else self.examples,
            num_workers=num_workers,
            retries=self.retries,
            force=force,
            **kwargs,
        )

    @property
    def results(self):
        """Iterate over all cached (key_dict, parsed_result) pairs.

        Yields:
            tuple: (key_dict, validated pydantic object or list thereof)
        """
        for key, raw in self.stash.items():
            if not isinstance(raw, str):
                continue
            try:
                parsed = _parse_json_response(raw)
                result = _validate_parsed(parsed, self.schema)
                yield key, result
            except Exception:
                continue

    @property
    def df(self):
        """Build a DataFrame from all cached results.

        For list[Model] schemas, each item becomes its own row.
        Key metadata (model, prompt snippet, temperature) are included as columns.
        """
        rows = []
        is_list, item_schema = _unwrap_schema(self.schema)
        for key, result in self.results:
            meta = {}
            if isinstance(key, dict):
                meta["model"] = key.get("model", "")
                meta["temperature"] = key.get("temperature", "")
                prompt = key.get("prompt", "")
                meta["prompt"] = prompt[:200] if isinstance(prompt, str) else str(prompt)[:200]

            items = result if is_list else [result]
            for item in items:
                row = {**meta, **item.model_dump()}
                rows.append(row)

        return pd.DataFrame(rows)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.task_name!r}, schema={_schema_repr(self.schema)})"


def _schema_repr(schema):
    if schema is None:
        return "None"
    origin = getattr(schema, "__origin__", None)
    if origin is list:
        return f"list[{schema.__args__[0].__name__}]"
    return schema.__name__
