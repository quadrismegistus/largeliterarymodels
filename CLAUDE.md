# CLAUDE.md — largeliterarymodels

## What this is

Python package for structured data extraction from literary texts using LLMs. Pydantic schemas define what to extract; provider backends (Anthropic, OpenAI, Google, local via LM Studio/vLLM) do the inference. Results are cached via HashStash.

Package: `largeliterarymodels`. CLI: `litmod`. Never abbreviate as "lltm".

## Architecture

```
Task (task.py)          -- base class: schema + prompt + examples + cache
SequentialTask          -- chunks + rolling state (social network, passage annotation)
  ├── tasks/            -- concrete tasks (lazy-loaded via tasks/__init__.py)
  ├── llm.py            -- LLM call wrapper, JSON parsing, retry
  ├── providers.py      -- direct SDK calls (no litellm): route_provider() dispatches
  ├── analysis/         -- cross-task analysis: Fisher tests, ensembles, social networks
  ├── cli/              -- litmod ls|show|smoke|run|annotate|cloud
  └── annotate.py       -- FastHTML human-annotation web app (auto-generates forms from Pydantic)
```

## Relationship to lltk

**largeliterarymodels is a pure extraction library. lltk imports it, not the reverse.**

| | largeliterarymodels | lltk |
|---|---|---|
| Role | Pure extraction (str → structured data) | Corpus management + orchestration |
| Knows about | Schemas, LLMs, providers, caching | Corpora, passages, metadata, ClickHouse |
| Input | `str` or `list[str]` | Text IDs, database queries |
| Output | Pydantic models / dicts | Annotations stored in task paths + CH |

The core task system has zero lltk imports. `integrations/llmtasks.py` (ClickHouse adapter) is legacy, being migrated to lltk.

## Task system

Subclass `Task`, set `name`, `schema` (Pydantic model), `system_prompt`, `examples`. Each task can set a per-task default `model` so callers don't need to know the best model.

```python
task = MyTask(model="lmstudio/qwen3.5-35b-a3b")
result = task.run(text)           # single
results = task.map(texts)         # batch with caching
```

Task catalog:
- **Base tasks**: GenreTask, GenreTaskLite, FryeTask, CharacterTask, CharacterIntroTask, TranslationTask, BibliographyTask, OCRCleanTask
- **Sequential tasks**: PassageContentTask, PassageFormTask, SocialNetworkTask

## Provider routing

`providers.py:route_provider()` dispatches on model string prefix:
- `lmstudio/`, `local/`, `vllm/`, `ollama/` → OpenAI-compat local endpoint
- `claude*`, `anthropic/` → Anthropic SDK (with prompt caching)
- `gpt*`, `o1*`, `o3*`, `openai/` → OpenAI SDK
- `gemini*`, `google/` → Google GenAI SDK
- `claude-cli/` → shells out to `claude -p --bare` (subscription auth, small tasks only)

## Model tags

Short tags in `cli/models.py::MODEL_TAGS`. Convention: `<family>-<variant>[-<backend>]`.
Default = LM Studio GGUF. `-mlx` suffix = MLX variant. Anthropic tags (`sonnet`, `opus`, `haiku`) stand alone.

## Running

Always activate the local venv before running Python. Prefix every bash python/pip/litmod command with `source .venv/bin/activate &&` or use `.venv/bin/python` directly.

```bash
source .venv/bin/activate
pip install -e ".[dev]"
litmod ls                                      # list tasks
litmod smoke GenreTask --model sonnet          # quick single-passage test
litmod run GenreTask --input data/manifest.csv --model sonnet
litmod cloud launch                            # rent A100 on Vast.ai
litmod cloud status                            # check batch progress
```

## Scripts

`scripts/` has pilot runners and analysis scripts. Naming: `pilot_<task>.py`, `smoke_<task>.py`, `analyze_<analysis>.py`.

Key scripts:
- `batch_social_network.py` — batch runner for Colab/HPC/cloud
- `analyze_social_networks.py` — network statistics across parsed texts
- `hpc/export_passages.py` — export passages from CH for remote processing

## Tests

```bash
source .venv/bin/activate && pytest tests/
```
