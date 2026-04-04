# Large Literary Models

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
- [Project Structure](#project-structure)

## Installation

### Prerequisites

You need **Python 3.10 or later**. To check your version, open a terminal and run:

```bash
python --version
```

If you see something like `Python 3.10.6` or higher, you're good. If not, install a newer Python from [python.org](https://www.python.org/downloads/) or via [pyenv](https://github.com/pyenv/pyenv).

### Step 1: Clone the repository

```bash
git clone https://github.com/quadrismegistus/largeliterarymodels.git
cd largeliterarymodels
```

### Step 2: Create a virtual environment

A virtual environment keeps this project's dependencies separate from your other Python projects. Run:

```bash
python -m venv .venv
```

Then activate it:

```bash
# On Mac/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal prompt. You'll need to activate this environment each time you open a new terminal to work on this project.

### Step 3: Install the package

```bash
pip install -e "."
```

This installs `largeliterarymodels` and all its dependencies (the Anthropic, OpenAI, and Google AI client libraries, plus [HashStash](https://github.com/quadrismegistus/hashstash) for caching).

To also install [pydantic](https://docs.pydantic.dev/) for structured data extraction (recommended):

```bash
pip install -e ".[pydantic]"
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

To avoid typing these every time, add the `export` lines to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) or create a `.env` file in the project directory.

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
llm = LLM("gpt-4o-mini")
llm = LLM("gemini-2.5-flash")

# Generate text
response = llm.generate("What is the plot of Pamela by Samuel Richardson?")
print(response)
```

The response is automatically cached. If you run the exact same call again, it returns instantly without using any API credits.

### Changing default parameters

```python
# Lower temperature = more deterministic output
llm = LLM("claude-sonnet-4-20250514", temperature=0.2)

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

You define schemas using [pydantic](https://docs.pydantic.dev/), which is a way of describing data structures in Python. Install it with `pip install pydantic` if you haven't already.

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

### Task caching

Each task gets its own separate cache directory (at `data/stash/<TaskClassName>/`). This keeps results organized and means you can clear one task's cache without affecting others.

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
    model="claude-sonnet-4-20250514",
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
from largeliterarymodels.tasks import BibliographyTask
import re, pandas as pd

# Load the task
task = BibliographyTask()

# Load and chunk your HTML file
with open("data/bibliography.html") as f:
    raw_html = f.read()

# Split on year headings -- each chunk is one year's entries
chunks = re.split(r'(?=<h2[^>]*>)', raw_html)
chunks = [c.strip() for c in chunks if c.strip()]

# Extract from all chunks (parallel, cached)
all_entries = task.map(chunks)

# Flatten and export
flat = [entry for chunk_entries in all_entries for entry in chunk_entries]
df = pd.DataFrame([e.model_dump() for e in flat])
df = df.sort_values(["year", "author"]).reset_index(drop=True)
df.to_csv("data/bibliography.csv", index=False)

print(f"{len(df)} entries extracted")
print(df[["year", "author", "title", "printer", "publisher"]].head())
```

The `BibliographyEntry` schema extracts 14 fields from each entry: author, title, subtitle, year, edition, bibliographic ID, translation status, translator, printer, publisher, bookseller, and notes. See `largeliterarymodels/tasks/extract_bibliography.py` for the full schema and few-shot examples.

### Comparing models

```python
for model in ["claude-sonnet-4-20250514", "gpt-4o-mini", "gemini-2.5-flash"]:
    entries = task.run(chunks[0], model=model)
    print(f"{model}: {len(entries)} entries extracted")
```

## Project Structure

```
largeliterarymodels/
    __init__.py              # Exports: LLM, Task, check_api_keys, available_models
    llm.py                   # Core LLM class: generate, extract, map, extract_map
    task.py                  # Task class: reusable extraction task definition
    providers.py             # Direct API calls to Anthropic, OpenAI, Google (no litellm)
    utils.py                 # Utility functions
    tasks/
        extract_bibliography.py  # Built-in bibliography extraction task
data/
    stash/                   # Cached LLM responses (auto-generated)
notebooks/
    extract_bibliography.ipynb   # Example notebook
tests/                       # Test suite (run with: pytest)
```

## License

MIT
