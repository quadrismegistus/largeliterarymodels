# Large Literary Models

[![PyPI version](https://badge.fury.io/py/largeliterarymodels.svg)](https://pypi.org/project/largeliterarymodels/)
[![Tests](https://github.com/quadrismegistus/largeliterarymodels/actions/workflows/test.yml/badge.svg)](https://github.com/quadrismegistus/largeliterarymodels/actions/workflows/test.yml)

A Python toolkit for using Large Language Models (LLMs) to produce structured, annotated data from unstructured texts. Built for digital humanities research.

**What it does:** You give it messy text (OCR scans, bibliographies, novel excerpts, archival documents) and a description of the structured data you want back (characters, citations, sentiments, relationships). It sends the text to an LLM, parses the response into clean structured data, caches everything so you never pay for the same query twice, and hands you back validated Python objects you can export to CSV or a pandas DataFrame.

**Supported LLM providers:** Claude (Anthropic), GPT (OpenAI), Gemini (Google), and any OpenAI-compatible local server (vLLM, LM Studio, Ollama, llama.cpp).

## Table of Contents

- [Installation](#installation)
- [Setup: API Keys](#setup-api-keys)
- [Quick Start](#quick-start)
- [Structured Extraction](#structured-extraction)
- [Defining a Task](#defining-a-task)
- [Working with Multiple Prompts](#working-with-multiple-prompts)
- [Sequential Tasks](#sequential-tasks)
- [Caching](#caching)
- [Using local models (vLLM, LM Studio, Ollama)](#using-local-models-vllm-lm-studio-ollama)
- [CLI: litmod](#cli-litmod)
- [Running at Scale](#running-at-scale)
- [Relationship to LLTK](#relationship-to-lltk)
- [Example: Bibliography Extraction](#example-bibliography-extraction)
- [Starting Your Own Project](#starting-your-own-project)
- [Available Tasks](#available-tasks)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

You need **Python 3.10 or later**.

### Install from PyPI

```bash
pip install largeliterarymodels
```

This installs `largeliterarymodels` and all its dependencies: the Anthropic, OpenAI, and Google AI client libraries, [pydantic](https://docs.pydantic.dev/) for structured data extraction, and [HashStash](https://github.com/quadrismegistus/hashstash) for caching.

### Install from source (for development)

```bash
git clone https://github.com/quadrismegistus/largeliterarymodels.git
cd largeliterarymodels
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Setup: API Keys

To use an LLM, you need an API key from at least one provider. You only need one, but having multiple lets you compare models.

| Provider | Get a key at | Environment variable |
|----------|-------------|---------------------|
| Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com/) | `ANTHROPIC_API_KEY` |
| OpenAI (GPT) | [platform.openai.com](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Google (Gemini) | [aistudio.google.com](https://aistudio.google.com/app/apikey) | `GEMINI_API_KEY` |

Set them in your shell:

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export OPENAI_API_KEY="sk-your-key-here"
export GEMINI_API_KEY="your-key-here"
```

For local models (vLLM, LM Studio, Ollama), no API key is needed.

## Quick Start

### Basic text generation

```python
from largeliterarymodels import LLM

llm = LLM()  # defaults to Claude Sonnet
response = llm.generate("What is the plot of Pamela by Samuel Richardson?")
print(response)
```

The response is automatically cached. Running the exact same call again returns instantly without using API credits.

### Structured extraction

This is the core feature. Define a **schema** using pydantic, and the LLM fills it in:

```python
from pydantic import BaseModel, Field
from largeliterarymodels import LLM

class Sentiment(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="confidence score from 0.0 to 1.0")
    explanation: str = Field(description="one-sentence explanation")

llm = LLM()
result = llm.extract(
    "It was the best of times, it was the worst of times.",
    schema=Sentiment,
)

print(result.sentiment)     # "neutral"
print(result.confidence)    # 0.75
print(result.explanation)   # "The passage juxtaposes extremes..."
```

The `result` is a validated Python object with dot-notation access, not a string or raw JSON.

### Extracting a list of items

Use `list[YourModel]` when you expect multiple results:

```python
class Character(BaseModel):
    name: str
    role: str = Field(description="role in the narrative")
    gender: str

characters = llm.extract(
    "Who are the main characters in Pride and Prejudice?",
    schema=list[Character],
)
for c in characters:
    print(f"{c.name} ({c.gender}): {c.role}")
```

## Defining a Task

A **Task** bundles together everything needed for a specific extraction job: the schema, system prompt, examples, and configuration.

```python
from pydantic import BaseModel, Field
from largeliterarymodels import Task

class Character(BaseModel):
    name: str
    gender: str
    role: str = Field(description="role in the narrative")
    prominence: int = Field(description="1-10 prominence score")

class CharacterTask(Task):
    schema = list[Character]
    system_prompt = "You are a literary scholar. Extract all named characters."
    examples = [
        (
            "Mr. Darcy danced with Elizabeth at the ball.",
            [
                Character(name="Mr. Darcy", gender="Male", role="love interest", prominence=9),
                Character(name="Elizabeth", gender="Female", role="protagonist", prominence=10),
            ],
        ),
    ]
    retries = 2
```

```python
task = CharacterTask()
characters = task.run(chapter_text)
characters_gpt = task.run(chapter_text, model="gpt-4o-mini")
```

## Working with Multiple Prompts

```python
task = CharacterTask()
results = task.map(
    [chapter_1_text, chapter_2_text, chapter_3_text],
    model="claude-sonnet-4-6",
)
```

`map()` runs requests in parallel (4 threads by default), shows a progress bar, and caches results. Re-running the same batch skips already-cached prompts.

## Sequential Tasks

For long texts that need to be processed in chunks with rolling context (e.g., extracting a social network from a 300-page novel), use `SequentialTask`:

```python
from largeliterarymodels.tasks import SocialNetworkTask

task = SocialNetworkTask(model="vllm/qwen3.6-27b")

# Pass a list of passage strings
result = task.run(passages, cache_key="my_text_id")

# Or pass a .txt file path (auto-chunked)
result = task.run("novel.txt", cache_key="novel")
```

`SequentialTask.run()` processes passages in chunks, feeding forward a rolling state (e.g., the character roster so far) to maintain consistency across chunks. Results are cached per-chunk, so interrupted runs resume where they left off.

**Key parameters:**
- `source`: list of passage strings, or path to a .txt file
- `cache_key`: stable identifier for caching (e.g., a text ID)
- `save`: path to save the aggregated JSON result, or `False` to skip

## Caching

All LLM calls are automatically cached using [HashStash](https://github.com/quadrismegistus/hashstash). The cache key includes the prompt, model, system prompt, temperature, and schema. Same inputs = instant cached result.

Cache is stored in `data/stash/` inside the repository. To force a fresh generation:

```python
response = llm.generate("What is the plot of Pamela?", force=True)
```

Provider-side prompt caching (Anthropic, OpenAI) is enabled automatically for repeat calls within a batch, cutting input costs ~10x on long system prompts.

## Using local models (vLLM, LM Studio, Ollama)

Any OpenAI-compatible local inference server works:

```python
from largeliterarymodels import LLM
llm = LLM(model="lmstudio/qwen3.5-35b-a3b")
llm = LLM(model="vllm/qwen3.6-27b")
llm = LLM(model="ollama/mistral")
```

By default, local models connect to `http://localhost:11434/v1` (Ollama's default). Override with:

```bash
export LOCAL_BASE_URL="http://localhost:8000/v1"   # vLLM
export LOCAL_BASE_URL="http://localhost:1234/v1"   # LM Studio
```

## CLI: litmod

The package includes a CLI tool:

```bash
litmod ls                                      # list available tasks
litmod show GenreTask                          # show task schema + fixtures
litmod smoke GenreTask --model sonnet          # test on built-in fixtures
litmod run GenreTask --input data/manifest.csv --model sonnet
litmod annotate GenreTask --port 8989          # human annotation web app
```

### Cloud GPU management (Vast.ai)

For running sequential tasks at scale on rented GPUs:

```bash
litmod cloud launch              # find + rent cheapest A100 80GB (~$0.85/hr)
litmod cloud setup               # install vLLM + largeliterarymodels over SSH
litmod cloud upload passages_c19 # rsync passage files to instance
litmod cloud run passages_c19    # start vLLM + batch in tmux (survives disconnects)
litmod cloud status              # check progress, running cost, tail log
litmod cloud download            # rsync results back locally
litmod cloud stop                # destroy instance (stops all billing)
litmod cloud ssh                 # interactive shell access
```

State is persisted in `.vastai.json` so everything is resumable across disconnects. Requires a [Vast.ai](https://vast.ai) account and API key (`pip install vastai && vastai set api-key YOUR_KEY`).

## Running at Scale

For processing hundreds or thousands of texts, the workflow splits across two environments:

### On GPU (Colab, Vast.ai, or HPC)

1. **Export passages** locally (where your database is):
   ```bash
   python scripts/hpc/export_passages.py --subcollection Nineteenth-Century_Fiction --out data/passages_c19
   ```

2. **Upload and run** on the GPU:
   ```bash
   # Vast.ai
   litmod cloud upload passages_c19
   litmod cloud run passages_c19

   # Or Colab: upload JSONL files, then:
   python scripts/batch_social_network.py --text-dir passages/ --output-dir results/ --model vllm-qwen36 --workers 4
   ```

3. **Download results** and ingest locally:
   ```bash
   litmod cloud download
   lltk ingest-tasks social_network data/cloud_results/passages_c19/
   ```

The batch script handles resume-from-failure (skips texts with existing output), parallel workers, and sharding across multiple processes.

### Passage export format

Exported files are JSONL with an `_id` metadata header:

```json
{"_id": "_chadwyck/Nineteenth-Century_Fiction/ncf0101.01", "_n_passages": 298}
{"seq": 0, "text": "CHAPTER I. In which the reader...", "n_words": 487}
{"seq": 1, "text": "The morning was bright and clear...", "n_words": 512}
```

The `_id` in the header is authoritative for result placement -- filenames are slugified and not used for identity.

## Relationship to LLTK

This package is designed to work with [LLTK](https://github.com/quadrismegistus/lltk) (Literary Language Toolkit) but does not depend on it. The division of labor:

| | largeliterarymodels | lltk |
|---|---|---|
| **Role** | Pure extraction library | Corpus management + orchestration |
| **Knows about** | Schemas, LLMs, providers, caching | Corpora, passages, metadata, ClickHouse |
| **Input** | `str` or `list[str]` | Text IDs, database queries |
| **Output** | Pydantic models / dicts | Annotations, task paths, scalar features |

**lltk imports largeliterarymodels** (not the reverse). This means largeliterarymodels works anywhere -- laptops, Colab, HPC, cloud GPUs -- without needing a database connection.

When used together, lltk orchestrates the pipeline:

```python
import lltk

# lltk resolves passages and calls largeliterarymodels tasks
lltk.annotate.run_task('genre', ids=['_estc/T068056'], model='gemini-2.5-flash')
lltk.annotate.run_task('social_network', ids=['_chadwyck/.../haywood.02'], model='vllm/qwen3.6-27b')

# Results stored in lltk's annotation system
# Full JSON blobs go to lltk.task_path()
# Scalar features go to lltk.annotations
```

Install together: `pip install largeliterarymodels lltk-dh`

## Example: Bibliography Extraction

```python
from largeliterarymodels.tasks import BibliographyTask, chunk_bibliography

task = BibliographyTask()

with open("data/bibliography.html") as f:
    raw_html = f.read()

chunks = chunk_bibliography(raw_html, max_entries=20)
all_entries = task.map(chunks)

flat = [entry for chunk_entries in all_entries for entry in chunk_entries]
df = pd.DataFrame([e.model_dump() for e in flat])
df.to_csv("data/bibliography.csv", index=False)
```

## Starting Your Own Project

Create a separate repository that depends on this package:

```python
from pydantic import BaseModel, Field
from largeliterarymodels import Task

class MyEntry(BaseModel):
    # your custom fields
    ...

class MyTask(Task):
    schema = list[MyEntry]
    system_prompt = "Your domain-specific instructions..."
    examples = [...]
```

```bash
pip install largeliterarymodels
```

## Available Tasks

| Task | Type | Input | Output |
|------|------|-------|--------|
| `GenreTask` | Base | Title/author metadata | Genre, subgenre, translation status, confidence |
| `GenreTaskLite` | Base | Title/author metadata | Constrained genre tags (form + mode) |
| `FryeTask` | Base | Text passages | Frye mode, mythos, referential mode |
| `PassageContentTask` | Sequential | Passage list | 43 binary content flags per passage |
| `PassageFormTask` | Sequential | Passage list | Formal/stylistic features per passage |
| `SocialNetworkTask` | Sequential | Passage list | Characters, relations, events, dialogue, summaries |
| `CharacterTask` | Base | BookNLP character roster | Merged/cleaned character list |
| `CharacterIntroTask` | Base | Character first-mention passages | Introduction mode, social class |
| `TranslationTask` | Base | Word in context | Historical translation + connotations |
| `BibliographyTask` | Base | OCR bibliography pages | Structured bibliography entries |

**Base tasks** process a single prompt and return a Pydantic model. **Sequential tasks** process a list of passages in chunks with rolling state and return an aggregated dict.

## Project Structure

```
largeliterarymodels/
    __init__.py              # Exports: LLM, Task, model constants
    llm.py                   # Core LLM class: generate, extract, map
    task.py                  # Task + SequentialTask base classes
    providers.py             # Direct API calls to Anthropic, OpenAI, Google, local
    tasks/                   # Built-in task definitions (lazy-loaded)
    analysis/                # Cross-task analysis: Fisher tests, ensembles, social networks
    cli/                     # litmod CLI: ls, show, smoke, run, annotate, cloud
    integrations/            # ClickHouse adapter (being migrated to lltk)
    annotate.py              # FastAPI human-annotation web app
scripts/
    batch_social_network.py  # Batch runner for SocialNetworkTask (Colab/HPC/cloud)
    analyze_social_networks.py  # Network statistics across parsed texts
    hpc/                     # Passage export, SLURM scripts, Colab notebooks
    cloud/                   # Vast.ai standalone entry point
tests/                       # Test suite (pytest)
```

## License

MIT
