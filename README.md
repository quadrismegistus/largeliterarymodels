# Large Literary Models

[![PyPI version](https://badge.fury.io/py/largeliterarymodels.svg)](https://pypi.org/project/largeliterarymodels/)
[![Tests](https://github.com/quadrismegistus/largeliterarymodels/actions/workflows/test.yml/badge.svg)](https://github.com/quadrismegistus/largeliterarymodels/actions/workflows/test.yml)

A Python toolkit for using Large Language Models (LLMs) to produce structured, annotated data from unstructured texts. Built for digital humanities research.

**What it does:** You give it messy text (OCR scans, bibliographies, novel excerpts, archival documents) and a description of the structured data you want back (characters, citations, sentiments, relationships). It sends the text to an LLM, parses the response into clean structured data, caches everything so you never pay for the same query twice, and hands you back validated Python objects you can export to CSV or a pandas DataFrame.

**Supported LLM providers:** Claude (Anthropic), GPT (OpenAI), Gemini (Google). No account with all three is needed -- any one will work.

## Table of Contents

- [Installation](#installation)
- [Setup: API Keys](#setup-api-keys)
- [Quick Start](#quick-start)
- [Structured Extraction](#structured-extraction)
- [Defining a Task](#defining-a-task)
- [Working with Multiple Prompts](#working-with-multiple-prompts)
- [Caching](#caching)
- [Example: Bibliography Extraction](#example-bibliography-extraction)
- [Starting Your Own Project](#starting-your-own-project)
- [Using with LLTK](#using-with-lltk)
- [Model Constants](#model-constants)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

You need **Python 3.10 or later**. To check your version, open a terminal and run:

```bash
python --version
```

If you see something like `Python 3.10.6` or higher, you're good. If not, install a newer Python from [python.org](https://www.python.org/downloads/) or via [pyenv](https://github.com/pyenv/pyenv).

### Install from PyPI

The simplest way to install:

```bash
pip install largeliterarymodels
```

This installs `largeliterarymodels` and all its dependencies: the Anthropic, OpenAI, and Google AI client libraries, [pydantic](https://docs.pydantic.dev/) for structured data extraction, and [HashStash](https://github.com/quadrismegistus/hashstash) for caching.

### Install with LLTK (for literary corpus analysis)

To use the built-in literary analysis tasks (genre classification, character networks, Frye mode analysis) with [LLTK](https://github.com/quadrismegistus/lltk) corpora:

```bash
pip install "largeliterarymodels[lltk]"
```

This adds [lltk-dh](https://pypi.org/project/lltk-dh/), which provides 50+ literary corpora, cross-corpus matching, and DuckDB-backed metadata. See [Using with LLTK](#using-with-lltk) below.

### Install from source (for development)

If you want to modify the library itself:

```bash
git clone https://github.com/quadrismegistus/largeliterarymodels.git
cd largeliterarymodels
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Setup: API Keys

To use an LLM, you need an API key from at least one provider. You only need one, but having multiple lets you compare models.

| Provider | Get a key at | Environment variable |
|----------|-------------|---------------------|
| Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com/) | `ANTHROPIC_API_KEY` |
| OpenAI (GPT) | [platform.openai.com](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Google (Gemini) | [aistudio.google.com](https://aistudio.google.com/app/apikey) | `GEMINI_API_KEY` |

Once you have a key, set it in your terminal before running any code:

```bash
# Pick whichever provider(s) you have:
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export OPENAI_API_KEY="sk-your-key-here"
export GEMINI_API_KEY="your-key-here"
```

To avoid typing these every time, add the `export` lines to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) or create a `.env` file in your project directory.

To check which keys are set:

```python
from largeliterarymodels import check_api_keys
check_api_keys(verbose=True)
```

```
  + ANTHROPIC_API_KEY
  X OPENAI_API_KEY
  + GEMINI_API_KEY
```

## Quick Start

### Basic text generation

```python
from largeliterarymodels import LLM

# Create an LLM instance (defaults to Claude Sonnet)
llm = LLM()

# Or specify a model:
from largeliterarymodels import CLAUDE_OPUS, GPT_4O_MINI, GEMINI_FLASH
llm = LLM(GPT_4O_MINI)
llm = LLM(GEMINI_FLASH)

# Generate text
response = llm.generate("What is the plot of Pamela by Samuel Richardson?")
print(response)
```

The response is automatically cached. If you run the exact same call again, it returns instantly without using any API credits.

### Changing default parameters

```python
from largeliterarymodels import LLM, CLAUDE_SONNET

# Lower temperature = more deterministic output
llm = LLM(CLAUDE_SONNET, temperature=0.2)

# Set a system prompt that applies to all calls
llm = LLM(system_prompt="You are an expert in 18th-century English literature.")
response = llm.generate("Who is Pamela?")

# Override per-call
response = llm.generate(
    "Who is Pamela?",
    system_prompt="You are a children's librarian. Explain simply.",
    temperature=0.9,
)
```

## Structured Extraction

This is the core feature. Instead of getting back free-form text, you define a **schema** -- a description of the exact fields you want -- and the LLM fills them in.

You define schemas using [pydantic](https://docs.pydantic.dev/), which is a way of describing data structures in Python.

### A simple example

```python
from pydantic import BaseModel, Field
from largeliterarymodels import LLM

# Define what you want back
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

The `result` is a validated Python object -- not a string, not raw JSON. You can access its fields with dot notation.

### Extracting a list of items

Use `list[YourModel]` when you expect multiple results:

```python
class Character(BaseModel):
    name: str
    role: str = Field(description="role in the narrative")
    gender: str

llm = LLM()
characters = llm.extract(
    "Who are the main characters in Pride and Prejudice?",
    schema=list[Character],
    system_prompt="You are a literary scholar.",
)

for c in characters:
    print(f"{c.name} ({c.gender}): {c.role}")
```

```
Elizabeth Bennet (Female): Protagonist; witty and independent young woman
Mr. Darcy (Male): Male lead; proud wealthy gentleman
Jane Bennet (Female): Elizabeth's gentle elder sister
...
```

### Adding context with system prompts

The `system_prompt` tells the LLM *how* to approach the task -- what expertise to assume, what conventions to follow:

```python
result = llm.extract(
    scene_text,
    schema=BechdelResult,
    system_prompt="You are a film critic. Assess whether this scene passes the Bechdel test.",
)
```

### Few-shot examples

Few-shot examples show the LLM exactly what you expect. Each example is a pair: `(input_text, expected_output)`. The output can be a pydantic object or a plain dictionary.

```python
examples = [
    # Example 1: show the LLM what good output looks like
    (
        "[INT. HOUSE]\nEMILY: What do you think about Michael?\nEMMA: He seems risky.",
        Sentiment(sentiment="negative", confidence=0.7, explanation="Apprehension about Michael."),
    ),
    # Example 2: a contrasting case
    (
        "The sun shone brightly on the meadow.",
        Sentiment(sentiment="positive", confidence=0.85, explanation="Bright, pleasant imagery."),
    ),
]

result = llm.extract(
    "The room was dark and cold.",
    schema=Sentiment,
    examples=examples,
)
```

Few-shot examples dramatically improve accuracy, especially for domain-specific tasks. Even one or two examples help.

### Error handling

If the LLM returns malformed JSON, `extract()` will automatically retry (once by default). You can control this:

```python
result = llm.extract(prompt, schema=MySchema, retries=3)  # up to 3 retries
```

## Defining a Task

A **Task** bundles together everything needed for a specific extraction job: the schema, system prompt, examples, and configuration. This means you define your task once, then reuse it across many inputs and models.

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
    system_prompt = "You are a literary scholar. Extract all named characters from the text."
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

### Using a Task

```python
task = CharacterTask()

# Extract from one text
characters = task.run(chapter_text)
for c in characters:
    print(f"{c.name}: {c.role} ({c.prominence}/10)")

# Try a different model
characters_gpt = task.run(chapter_text, model="gpt-4o-mini")

# Override system prompt for one call
characters = task.run(chapter_text, system_prompt="Focus only on female characters.")
```

### Task caching and results

Each task gets its own separate cache directory (at `data/stash/<TaskClassName>/`). This keeps results organized and means you can clear one task's cache without affecting others.

You can access all cached results as a DataFrame at any time:

```python
task = CharacterTask()
task.map(chapter_texts)    # populate the cache

# Get all results as a DataFrame
df = task.df
print(df.head())
```

The DataFrame includes metadata columns (`model`, `temperature`, `prompt`) alongside all schema fields. For `list[Model]` schemas, each item in the list becomes its own row.

## Working with Multiple Prompts

### Batch generation

```python
llm = LLM()
responses = llm.map(
    ["Summarize Chapter 1.", "Summarize Chapter 2.", "Summarize Chapter 3."],
    system_prompt="Summarize in one paragraph.",
)
# responses is a list of strings, in the same order as the prompts
```

### Batch extraction

```python
task = CharacterTask()
results = task.map(
    [chapter_1_text, chapter_2_text, chapter_3_text],
    model="claude-sonnet-4-6",
)
# results is a list of list[Character], one per chapter
```

Both `map()` methods run requests in parallel (4 threads by default), show a progress bar, and cache results. Re-running the same batch skips already-cached prompts:

```python
# This will only compute the new chapters, not re-do 1-3
results = task.map(
    [chapter_1_text, chapter_2_text, chapter_3_text, chapter_4_text],
)
```

### Exporting to CSV

Since extraction results are pydantic objects, converting to a pandas DataFrame is straightforward:

```python
import pandas as pd

# If results is a list of lists (from task.map), flatten first
flat = [entry for chunk in results for entry in chunk]

df = pd.DataFrame([entry.model_dump() for entry in flat])
df.to_csv("characters.csv", index=False)
print(df.head())
```

Or use the task's built-in DataFrame (see [Task caching and results](#task-caching-and-results) above).

## Caching

All LLM calls are automatically cached using [HashStash](https://github.com/quadrismegistus/hashstash). The cache key is the combination of:

- `prompt` (the input text)
- `model` (which LLM you used)
- `system_prompt`
- `temperature`
- `max_tokens`
- `schema` name (for `extract()` calls)

This means:
- **Same prompt + same model = cached** (instant, free)
- **Same prompt + different model = separate cache entry** (lets you compare models)
- **Same prompt + different system_prompt = separate cache entry**

Cache is stored in `data/stash/` inside the repository. To force a fresh generation:

```python
response = llm.generate("What is the plot of Pamela?", force=True)
```

## Example: Bibliography Extraction

The library ships with a ready-made task for parsing messy OCR bibliography entries into structured data. This is a real-world example of the kind of work `largeliterarymodels` is designed for.

```python
from largeliterarymodels.tasks import BibliographyTask, chunk_bibliography
import pandas as pd

# Load the task
task = BibliographyTask()

# Load and chunk your HTML file
with open("data/bibliography.html") as f:
    raw_html = f.read()

# Split into chunks (by year heading, max 20 entries each)
chunks = chunk_bibliography(raw_html, max_entries=20)

# Extract from all chunks (parallel, cached)
all_entries = task.map(chunks)

# Flatten and export
flat = [entry for chunk_entries in all_entries for entry in chunk_entries]
df = pd.DataFrame([e.model_dump() for e in flat])
df.to_csv("data/bibliography.csv", index=False)

# Or use the built-in DataFrame
df = task.df
```

The `BibliographyEntry` schema extracts fields including: author, title, subtitle, year, edition, bibliographic ID, translation status, translator, printer, publisher, bookseller, and notes. See `largeliterarymodels/tasks/extract_bibliography.py` for the full schema and few-shot examples.

### Comparing models

```python
from largeliterarymodels import CLAUDE_SONNET, GPT_4O_MINI, GEMINI_FLASH

for model in [CLAUDE_SONNET, GPT_4O_MINI, GEMINI_FLASH]:
    entries = task.run(chunks[0], model=model)
    print(f"{model}: {len(entries)} entries extracted")
```

## Starting Your Own Project

`largeliterarymodels` is a general-purpose toolkit. For your specific research project, we recommend creating a **separate repository** that depends on it:

```
my-bibliography-project/
    task.py                 # your Task subclass with custom schema/examples
    data/
        source.html         # your input data
        output.csv          # your results
    notebooks/
        extract.ipynb       # your working notebook
```

Your `task.py` defines only what's specific to your project:

```python
from pydantic import BaseModel, Field
from largeliterarymodels import Task

class MyEntry(BaseModel):
    # your custom fields here
    ...

class MyBibliographyTask(Task):
    schema = list[MyEntry]
    system_prompt = "Your domain-specific instructions..."
    examples = [...]
```

Install `largeliterarymodels` in your project's environment:

```bash
pip install largeliterarymodels
```

This way your project-specific decisions (field names, few-shot examples, OCR quirks) live in their own tracked repository, separate from the general-purpose toolkit.

## Using with LLTK

The library includes tasks designed for literary analysis with [LLTK](https://github.com/quadrismegistus/lltk) corpora. Install with `pip install "largeliterarymodels[lltk]"`.

### Genre classification

Classify texts by genre from title/author metadata:

```python
from largeliterarymodels.tasks import GenreTask, format_text_for_classification

task = GenreTask()
prompt = format_text_for_classification(title="Pamela", author_norm="richardson", year=1740)
result = task.run(prompt)
print(result.genre, result.genre_raw, result.confidence)
# Fiction Novel, Epistolary fiction 1.0
```

### Character resolution (BookNLP cleanup)

BookNLP's NER is noisy on early modern texts. This task merges fragmented character clusters and filters noise:

```python
import lltk
from largeliterarymodels.tasks import CharacterTask, format_character_roster

t = lltk.load('chadwyck').text('Eighteenth-Century_Fiction/fieldinh.06')  # Tom Jones
t.booknlp.parse()  # run BookNLP first

task = CharacterTask()
prompt = format_character_roster(t, max_chars=30)
results = task.run(prompt)  # returns list[CharacterResolution]
for r in results:
    if r.type == 'character':
        print(f"{r.name}: {r.ids}")
# Tom Jones: ['C822', 'C625', 'C491']
# Sophia Western: ['C821', 'C888', 'C4113']
```

Or use the LLTK wrapper directly:

```python
t.booknlp.resolve_characters()   # runs CharacterTask, saves JSON
t.booknlp.plot_network()         # co-mention network visualization
```

### Available tasks

| Task | Input | Output |
|------|-------|--------|
| `GenreTask` | Title/author metadata | Genre, subgenre, translation status |
| `FryeTask` | Text passages (opening/middle/closing) | Frye mode, mythos, referential mode |
| `PassageTask` | ~1K-word passages | Scene type, narration mode, allegorical regime |
| `CharacterTask` | BookNLP character roster | Merged/cleaned character list |
| `CharacterIntroTask` | Character first-mention passages | Introduction mode, social class, interiority |
| `BibliographyTask` | OCR bibliography pages | Structured bibliography entries |

## Model Constants

For convenience, common model names are available as constants:

```python
from largeliterarymodels import (
    CLAUDE_OPUS,    # claude-opus-4-6
    CLAUDE_SONNET,  # claude-sonnet-4-6
    CLAUDE_HAIKU,   # claude-haiku-4-5-20251001
    GPT_4O,         # gpt-4o
    GPT_4O_MINI,    # gpt-4o-mini
    GEMINI_PRO,     # gemini-2.5-pro
    GEMINI_FLASH,   # gemini-2.5-flash
)

llm = LLM(CLAUDE_OPUS)
```

## Project Structure

```
largeliterarymodels/
    __init__.py              # Exports: LLM, Task, model constants, check_api_keys
    llm.py                   # Core LLM class: generate, extract, map, extract_map
    task.py                  # Task class: reusable extraction task definition
    providers.py             # Direct API calls to Anthropic, OpenAI, Google
    utils.py                 # Utility functions
    tasks/
        extract_bibliography.py  # Built-in bibliography extraction task
tests/                       # Test suite (run with: pytest)
```

## License

MIT
