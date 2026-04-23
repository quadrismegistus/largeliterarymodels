# CLAUDE.md — largeliterarymodels

## What this is

Python package for structured data extraction from literary texts using LLMs. Pydantic schemas define what to extract; provider backends (Anthropic, OpenAI, Google, local via LM Studio/vLLM) do the inference. Results are cached via HashStash and stored in ClickHouse.

Package: `largeliterarymodels`. CLI: `litmod`. Never abbreviate as "lltm".

## Architecture

```
Task (task.py)          -- base class: schema + prompt + examples + cache
  ├── tasks/            -- concrete tasks (GenreTask, PassageContentTask, etc.)
  ├── llm.py            -- LLM call wrapper, JSON parsing, retry
  ├── providers.py      -- direct SDK calls (no litellm): route_provider() dispatches
  └── integrations/
       ├── lltk.py      -- lltk corpus helpers (format_passage, etc.)
       └── llmtasks.py  -- ClickHouse read/write (llmtasks.passage_annotations)
```

- `analysis/` — schema-aware CH readers + cross-task discrimination analysis
- `cli/` — `litmod ls|show|smoke|run|annotate` subcommands
- `annotate.py` — FastAPI human-annotation web app (auto-generates forms from Pydantic)

## Task system

Subclass `Task`, set `name`, `schema` (Pydantic model), `system_prompt`, `examples`. Then:
```python
task = MyTask(model="lmstudio/qwen3.5-35b-a3b")
result = task.run(text)           # single
results = task.map(texts)         # batch with caching
```

Tasks are lazy-imported via `tasks/__init__.py`. The task catalog:
GenreTask, FryeTask, PassageTask, PassageContentTask, PassageFormTask,
CharacterTask, CharacterIntroTask, TranslationTask, BibliographyTask.

## Provider routing

`providers.py:route_provider()` dispatches on model string prefix:
- `lmstudio/`, `local/`, `vllm/`, `ollama/` → OpenAI-compat local endpoint
- `claude*`, `anthropic/` → Anthropic SDK (with prompt caching)
- `gpt*`, `o1*`, `o3*`, `openai/` → OpenAI SDK
- `gemini*`, `google/` → Google GenAI SDK

## ClickHouse store

`llmtasks.passage_annotations` — long-form table (one row per passage+field). Accessed via `lltk.db.client`, lives in `llmtasks` database (separate from `lltk.*`). Key concepts:
- 4-column source split: `source_family / source_agent / task / task_version`
- `config_sha256` — byte-exact hash of prompt+schema+model for reproducibility
- `passage_annotations_latest` — argMax view for deduped reads

## Cross-repo contracts

Three coordinating repos (with peer Claude Code sessions):

| Repo | Owns | Reads from |
|------|------|------------|
| **largeliterarymodels** | task schemas, CH write, `analysis/` cross-task | lltk corpus data |
| **lltk** | corpus tables, `lltk.analysis.stats` (Fisher, FDR) | — |
| **abstraction** | Ch5 notebooks, figures | largeliterarymodels.analysis + lltk.analysis.stats |

## Model tags

Short tags in `cli/models.py::MODEL_TAGS`. Convention: `<family>-<variant>[-<backend>]`.
Default = LM Studio GGUF. `-mlx` suffix = MLX variant. Anthropic tags (`sonnet`, `opus`, `haiku`) stand alone.

## Running

```bash
pip install -e ".[lltk,analysis,annotate]"   # dev install with all extras
litmod ls                                      # list tasks
litmod smoke PassageContentTask --model sonnet # quick single-passage test
litmod run PassageContentTask --input data/manifest.csv --model lmstudio/qwen3.5-35b-a3b
```

## Scripts

`scripts/` has pilot runners and analysis scripts. Naming: `pilot_<task>.py`, `smoke_<task>.py`, `analyze_<analysis>.py`.

## Tests

```bash
pytest tests/
```
