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

        # with images and metadata
        result = task.run("Extract entries from this page.",
                          images=["page1.png"],
                          metadata={"page": 1, "source": "mish_biblio.pdf"})
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
        self._human_stashes = {}

    @property
    def task_name(self):
        return self.name or self.__class__.__name__

    @property
    def stash(self):
        if self._stash is None:
            stash_dir = os.path.join(STASH_PATH, self.task_name)
            self._stash = HashStash(stash_dir, engine="pairtree", append_mode=True)
        return self._stash

    def human_stash(self, annotator: str = 'default'):
        """JSONL-backed stash for human annotations by this annotator.

        Uses hashstash flat mode: each write appends a plain-JSON line with
        dict fields inlined at top level (greppable, jq-queryable). Append-
        only on disk (full edit history preserved); reads return the latest
        value per key.

        Usage:
            stash = task.human_stash('ryan')
            stash[item_key] = {'field1': True, ...}   # append edit
            stash[item_key]                            # latest dict for key
            stash.items()                              # {key: latest_value}
            stash.df                                   # all history as DataFrame

        Files live under data/stash/_human_annotations/<task>/<annotator>/
        jsonl.hashstash.raw/data.jsonl.
        """
        if annotator not in self._human_stashes:
            root = os.path.join(
                STASH_PATH, '_human_annotations', self.task_name, annotator,
            )
            # hashstash >= 0.4 defaults jsonl engine to flat/raw/no-b64 —
            # no flags needed.
            self._human_stashes[annotator] = HashStash(
                root_dir=root, engine='jsonl',
            )
        return self._human_stashes[annotator]

    def _get_llm(self, model=None):
        """Get an LLM instance using this task's stash."""
        return LLM(
            model=model or getattr(self, 'model', None) or DEFAULT_MODEL,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stash=self.stash,
        )

    def run(self, prompt, model=None, system_prompt=None, examples=None,
            images=None, metadata=None, force=False, **kwargs):
        """Extract structured data from a single input.

        Args:
            prompt: The input text to extract from.
            model: Override the default model.
            system_prompt: Override the task's system_prompt.
            examples: Override the task's few-shot examples.
            images: List of images (file paths, bytes, or PIL Images).
            metadata: Dict of user-defined metadata (e.g. page_number).
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
            images=images,
            metadata=metadata,
            retries=self.retries,
            force=force,
            **kwargs,
        )

    def map(self, prompts, model=None, system_prompt=None, examples=None,
            images_list=None, metadata_list=None,
            num_workers=4, force=False, verbose=False, **kwargs):
        """Extract structured data from multiple inputs, with parallelism.

        Args:
            prompts: List of input texts.
            model: Override the default model.
            system_prompt: Override the task's system_prompt.
            examples: Override the task's few-shot examples.
            images_list: List of image lists, one per prompt (or None).
            metadata_list: List of metadata dicts, one per prompt (or None).
            num_workers: Number of parallel threads.
            force: Bypass cache.
            verbose: If True, print a compact per-call summary as each
                result lands. If a callable, use it as a custom formatter
                (see LLM.extract_map for the signature).
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
            images_list=images_list,
            metadata_list=metadata_list,
            num_workers=num_workers,
            retries=self.retries,
            force=force,
            verbose=verbose,
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
        Key metadata (model, prompt snippet, temperature) and user-defined
        metadata are included as columns.
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
                # Include user-defined metadata
                user_meta = key.get("metadata")
                if isinstance(user_meta, dict):
                    for k, v in user_meta.items():
                        meta[f"meta_{k}"] = v

            items = result if is_list else [result]
            for item in items:
                row = {**meta, **item.model_dump()}
                rows.append(row)

        return pd.DataFrame(rows)

    def annotate(self, port=8989, annotator='default', host='127.0.0.1'):
        """Launch a web app for human annotation of this task's cached items.

        The app generates form fields from the Pydantic schema, shows the
        LLM's annotation alongside for comparison, and saves human annotations
        to a JSONL file per annotator. A /compare page shows inter-annotator
        agreement statistics.

        Args:
            port: Port to serve on.
            annotator: Annotator ID (each gets their own JSONL file).
            host: Host to bind to.
        """
        from .annotate import run_annotator
        run_annotator(self, port=port, annotator=annotator, host=host)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.task_name!r}, schema={_schema_repr(self.schema)})"


class SequentialTask(Task):
    """Base class for tasks that process a text chunk-by-chunk with
    feedforward state (summaries, character registers, etc.).

    Unlike Task.run() which handles a single prompt, SequentialTask.run()
    processes a full text by splitting it into chunks of passages, maintaining
    rolling state across chunks, and aggregating the results.

    Subclass and implement:
        - build_state(): return initial state dict
        - format_context(state): format state for the prompt
        - parse_response(raw): parse LLM output into structured result
        - update_state(state, result, chunk_idx, start, end): update state from result
        - aggregate(all_results, state): combine chunk results into final output
    """

    chunk_size = 10
    max_tokens = 8192

    def build_state(self):
        """Initialize the rolling state. Override in subclasses."""
        return {}

    def format_context(self, state):
        """Format the rolling state as a prompt prefix. Override in subclasses."""
        raise NotImplementedError

    def format_passages(self, passages_df, start_idx):
        """Format a chunk of passages for the prompt."""
        parts = []
        for i, (_, row) in enumerate(passages_df.iterrows()):
            pnum = start_idx + i
            parts.append(f"--- P{pnum:03d} ({row['n_words']} words) ---")
            parts.append(row['text'])
            parts.append("")
        return '\n'.join(parts)

    def parse_response(self, raw):
        """Parse raw LLM output into a structured result dict. Override in subclasses."""
        import re
        json_text = raw.strip()
        if json_text.startswith('```'):
            json_text = re.sub(r'^```(?:json)?\s*', '', json_text)
            json_text = re.sub(r'\s*```\s*$', '', json_text)
        return json.loads(json_text)

    def update_state(self, state, result, chunk_idx, start, end):
        """Update rolling state from the chunk's output. Override in subclasses."""
        raise NotImplementedError

    def aggregate(self, all_results, state):
        """Combine all chunk results into a final output dict. Override in subclasses."""
        raise NotImplementedError

    @staticmethod
    def _load_passages(source, passage_size=500):
        """Load passages from a text_id, file path, or list of strings.

        Returns:
            tuple: (pd.DataFrame with 'text' and 'n_words' columns, source_label)
        """
        if isinstance(source, list):
            rows = [{'text': t, 'n_words': len(t.split()), 'seq': i}
                    for i, t in enumerate(source)]
            return pd.DataFrame(rows), 'list'

        if isinstance(source, str) and (source.endswith('.txt') or '/' in source
                                         and not source.startswith('_')):
            import os
            if os.path.isfile(source):
                with open(source) as f:
                    full_text = f.read()
                words = full_text.split()
                passages = []
                for i in range(0, len(words), passage_size):
                    chunk = ' '.join(words[i:i + passage_size])
                    passages.append({
                        'text': chunk, 'n_words': len(words[i:i + passage_size]),
                        'seq': len(passages),
                    })
                return pd.DataFrame(passages), os.path.basename(source)

        import lltk
        pdf = lltk.db.get_passages([source])
        pdf = pdf.sort_values('seq').reset_index(drop=True)
        return pdf, source

    def run(self, source, model=None, chunk_size=None, limit_chunks=0,
            force=False, verbose=True, save=None):
        """Process a full text chunk-by-chunk with feedforward state.

        Args:
            source: One of:
                - lltk text ID (e.g. '_chadwyck/.../haywood.13')
                - path to a .txt file (auto-chunked into ~500-word passages)
                - list of passage strings
            model: Override the default model.
            chunk_size: Override the default chunk size.
            limit_chunks: Stop after N chunks (0=all).
            force: Bypass cache.
            verbose: Print progress to stderr.
            save: Path to save JSON output (or True for auto-naming).

        Returns:
            dict: Aggregated results from all chunks.
        """
        import sys
        import time

        chunk_size = chunk_size or self.chunk_size
        model = model or getattr(self, 'model', None) or DEFAULT_MODEL

        pdf, source_label = self._load_passages(source)
        n_chunks = (len(pdf) + chunk_size - 1) // chunk_size
        if limit_chunks:
            n_chunks = min(n_chunks, limit_chunks)

        if verbose:
            print(f"Model: {model}", file=sys.stderr)
            print(f"Text: {source_label}", file=sys.stderr)
            print(f"Passages: {len(pdf)}, Chunk size: {chunk_size}, "
                  f"Chunks: {n_chunks}", file=sys.stderr)

        llm = self._get_llm(model)
        state = self.build_state()
        all_results = []

        t0 = time.time()
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(pdf))
            chunk_df = pdf.iloc[start:end]

            context = self.format_context(state)
            passages_text = self.format_passages(chunk_df, start)
            prompt = context + "\n\n" + f"PASSAGES:\n{passages_text}"

            cache_key = {
                'task': self.task_name, 'text_id': source_label,
                'chunk': chunk_idx, 'model': model,
                'chunk_size': chunk_size,
            }

            try:
                raw = llm.generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    cache_key=cache_key,
                    force=force,
                )
            except Exception as e:
                if verbose:
                    print(f"  [Chunk {chunk_idx:02d}] FAILED: {e!s:.100s}",
                          file=sys.stderr)
                all_results.append(None)
                continue

            try:
                result = self.parse_response(raw)
            except (json.JSONDecodeError, Exception) as e:
                if verbose:
                    print(f"  [Chunk {chunk_idx:02d}] PARSE FAILED: {e!s:.80s}",
                          file=sys.stderr)
                all_results.append(None)
                continue

            state = self.update_state(state, result, chunk_idx, start, end)
            all_results.append(result)

            if verbose:
                elapsed = time.time() - t0
                self.log_chunk(chunk_idx, start, end, elapsed, state, result)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nDone: {n_chunks} chunks in {elapsed:.0f}s "
                  f"({elapsed/max(1,n_chunks):.1f}s/chunk)", file=sys.stderr)

        output = self.aggregate(all_results, state)
        output['metadata'] = {
            'source': source_label,
            'model': model,
            'n_passages': len(pdf),
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'elapsed_seconds': elapsed,
        }

        if save:
            self._save_result(output, save, source_label, model)

        return output

    def _save_result(self, output, save, source_label, model):
        """Save aggregated result to JSON."""
        if save is True:
            slug = source_label.replace('/', '_').replace(' ', '_').strip('_')
            model_slug = model.split('/')[-1].replace('.', '').replace(' ', '_')
            save = os.path.join(
                STASH_PATH, '..', f'{self.task_name}_{slug}_{model_slug}.json',
            )
            save = os.path.normpath(save)
        os.makedirs(os.path.dirname(save), exist_ok=True)
        with open(save, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        import sys
        print(f"Saved to {save}", file=sys.stderr)

    def log_chunk(self, chunk_idx, start, end, elapsed, state, result):
        """Print per-chunk progress. Override for custom logging."""
        import sys
        print(f"  [Chunk {chunk_idx:02d}] P{start:03d}-P{end-1:03d}  "
              f"{elapsed:6.1f}s", file=sys.stderr)


def _schema_repr(schema):
    if schema is None:
        return "None"
    origin = getattr(schema, "__origin__", None)
    if origin is list:
        return f"list[{schema.__args__[0].__name__}]"
    return schema.__name__
