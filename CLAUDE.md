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
  │                        reliability.py: categorical agreement AND rank agreement —
  │                        pick by data type, they answer different questions
  ├── cli/              -- litmod ls|show|render|smoke|run|annotate|cloud
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

errors = {}                       # which items failed, and why
results = task.map(texts, metadata_list=metas, errors=errors)
print(task.usage.summary_line())  # tokens + prompt-cache hit rate
```

The stash is append-mode: a `force=True` rerun appends a version rather than
overwriting. `task.results`/`task.df` show the latest per key;
`task.results_history` yields every version. Consequence for methods claims —
without `force=True` a repeat run is a cache hit, so identical output across
two runs shows caching, not determinism. `temperature=0` is not evidence of a
stable annotation until it has been forced and the versions compared.

### Administering a task outside the API path

A scheme only one code path can run is committed, not frozen. `Task` can
serialise itself so a second coder — another provider, a subagent, a human on
paper — gets the *same* instrument rather than a hand transcription:

```python
task.instrument_text()               # byte-identical to the API system prompt
task.render_instrument(item=text)    # + delimited item + contract reminder
task.instrument_sha256()             # freezes instrument + item wrapper
task.administration_record()         # + model/temperature/pydantic version
diag = {}
task.parse_and_validate(reply, diagnosis=diag)   # same enforcement
task.retry_prompt(item, diag.get("partial_field"))  # same reprompt, targeted
```

`litmod render <Task> [--item TEXT | --item-file F | --fixture]` does the same
from the shell; its default output is byte-exact (`--digest` appends the
provenance footer). These delegate to `llm._build_extract_prompt` — never
re-render the parts, or the rendered instrument drifts from the administered
one. The digest deliberately omits model/temperature/max_tokens (same scheme,
different administration); `administration_record()` is the receipt that
carries both halves. Byte-identity holds for extract-path tasks only:
`SequentialTask` builds prompts from rolling state and its instrument methods
refuse rather than fabricate, and the `claude-cli/` transport flattens the
system prompt into the user turn.

Task catalog:
- **Base tasks**: GenreTask, GenreTaskLite, FryeTask, CharacterTask, CharacterIntroTask, TranslationTask, BibliographyTask, OCRCleanTask
- **Sequential tasks**: PassageContentTask, PassageFormTask, SocialNetworkTask

## Provider drift — the recurring bug class

Every provider bug this package has shipped has the same shape: **the code is
correct about the model generation it was written against, and wrong about the
current one.** A unit test cannot catch it; the mismatch is between our
constants and a live API. Three landed at once in 2026-07:

| Provider | Symptom | Now |
|---|---|---|
| deepseek | `deepseek-chat` silently served the cheap tier — annotations published off it | legacy aliases warn once naming what they resolve to |
| openai | `max_tokens` 400s across the whole gpt-5 tier | `_chat_completion` reads the replacement param out of the error and memoizes it |
| anthropic | `content[0].text` raises on any thinking model (Sonnet 5, Opus 5, Fable 5) | `_first_text` walks for the first text block |
| deepseek | v4 reasons by default on *both* tiers; thinking mode accepts `temperature` and ignores it silently, so runs read as temperature-pinned when they were not | `call_deepseek` sends `extra_body={"thinking": {"type": "disabled"}}` by default (for **every** deepseek id — deciding from the name is this same bug class), records the drop in `dropped_params` when thinking is on, and a response-side audit catches a disable that didn't take |

A useful generalisation from the fourth: our source comment had asserted that
`deepseek-reasoner` resolving to flash "silently gets you a *non-reasoning*
model". It was a plausible inference from the tier names and it was backwards.
Do not infer a model's behaviour from its name — probe it. (Probed while
fixing it: sonnet-5/opus-5 reject `temperature` even with thinking explicitly
disabled — "`temperature` is deprecated for this model" — so that constant is
a family fact, not a thinking artifact.)

**Run `litmod doctor` after touching provider code or model constants.** It
probes each provider's cheap tier, current frontier tier, and the package's
own per-provider default with a two-field schema. FAIL is a broken parse;
**WARN is a valid parse whose usage receipt shows what a PASS would bury** —
dropped params, reasoning billed as output, or the server naming a different
model than requested. The hard-failure class (max_tokens rename, content[0])
FAILs; the silent class (alias downgrades, accepted-and-ignored params)
WARNs, because it cannot break a parse by construction. Exit 1 on any FAIL
*and* when nothing was probed at all — a doctor with no API keys is not a
clean bill. Providers also log the server-resolved model id once per run;
record *that* (or `per_item_usage[i]['response_model']`) as the model of
record.

## Cost controls worth knowing

Measured 2026-07-30, `claude-sonnet-5`, PassageFormTask instrument (6,863 tokens):

- **Extended thinking is off by default on the extract path.** Thinking is
  on-by-default on Sonnet 5 / Opus 5 / Fable 5 and bills as *output*. Identical
  call: **1,124 output tokens with thinking on vs 340 with it disabled — 3.3x**,
  for a deliberation we parse as JSON and discard. `providers.thinking_default`
  disables it where permitted; Fable/Mythos reject an explicit disable and warn.
  **DeepSeek v4 is the same story with a sting in it.** Thinking is on by
  default at effort `high` on *both* flash and pro. A/B on the two-field
  doctor probe, 2 reps each, measured 2026-08-04:

  | model | thinking | latency | output | reasoning |
  |---|---|---|---|---|
  | v4-flash | off | 1.1–1.4s | 15 | 0 |
  | v4-flash | on | 4.0–4.9s | 326–383 | 306–363 (94%) |
  | v4-pro | off | 1.7–2.0s | 19 | 0 |
  | v4-pro | on | 4.3–5.5s | 279–354 | 259–334 (93%) |

  So ~20x the output tokens and ~3x the wall-clock, for text we discard.
  `deepseek_thinking_default` now disables it. Note what the table does to
  tier selection: flash's reputation as the fast tier and pro's as the terse
  one are both artifacts of thinking — with it off the two are within 0.6s and
  4 tokens of each other, and the choice is price against quality alone.
  The sting is that thinking mode also *accepts and ignores*
  `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` — no error,
  per DeepSeek's own docs — so the `_chat_completion` repair loop, which can
  only see a 400, never recorded the drop. Any DeepSeek annotation taken in
  thinking mode is uncontrolled sampling. `task.usage.summary_line()` now
  reports `reasoning=N (X% of output)` so this is a receipt rather than a
  suspicion, and `task.usage.no_reasoning_observed()` is the publication
  gate: multi-signal (token counts, DeepSeek's `reasoning_content` body,
  Anthropic thinking *blocks*, Google thought tokens), so it correctly FAILS
  a Fable run — Anthropic prices no reasoning split, and a token-only gate
  passed the one family whose thinking cannot be disabled. Stated limit: a
  local model that thinks inline in its answer text leaves no structured
  signal and reads as clean. **Gemini reasons by default too** (found by the
  doctor's first WARN sweep, 2026-08-04: 363 thought tokens against 14
  answer tokens on the two-field probe) and its `thoughts_token_count` is
  *not* included in `candidates_token_count` — usage now sums them so
  output_tokens means billed-output everywhere. No thinking-off default is
  wired for Google yet (2.5-pro cannot disable; flash could via
  `thinking_budget=0`) — `litmod doctor` WARNs on every Google tier until
  someone does, which is the honest state.
- **The cache key carries the thinking state and the *effective*
  temperature.** Thinking-on and thinking-off output must never share a key
  (a non-forced rerun would hand back one as the other, silently), and a key
  asserting `temperature: 0.0` on a model that rejected it plants a false
  methods claim in the durable artifact. Keys record `thinking` only when a
  thinking parameter is actually sent, and temperature as None where it is
  known-rejected (Sonnet 5/Opus 5/Fable) or known-ignored (DeepSeek thinking
  mode) — so pre-thinking-era caches stay byte-stable and reachable, while
  sonnet-5/opus-5/deepseek extract caches from before this change are
  deliberately orphaned: their provenance was compromised anyway. Custom
  `cache_key=` dicts gain `model`/`schema` via setdefault (keys that already
  carry `model`, like SequentialTask's chunk keys, are untouched).
- **The prompt cache is real and measurable, not assumed.** An 8-item batch at
  `num_workers=3` recorded one 6,863-token write and 48,041 tokens of reads —
  87% hit rate. Read it via `task.usage.summary_line()` / `.report()`.
- **Batches warm the cache on one item before fanning out** (`warm_cache=True`).
  A cache entry isn't readable until its write lands, so N parallel first calls
  otherwise each pay the ~1.25x write premium.
- **Anthropic silently declines to cache below a per-model token floor** —
  Haiku 4.5 needs 4,096, Sonnet/Opus 4.8 1,024, Opus 5 512, and it is *not*
  monotonic across generations. No error, `cache_creation_input_tokens: 0`,
  every call at full price. Never estimate this from character count: measured
  density ranges 2.7–3.9 chars/token across real instruments. Use
  `count_tokens`, or let `UsageTracker.cache_warning()` catch it from real usage.
  **This inverts the usual economics of prompt length.** A cached prefix bills
  at ~0.1×, so on a 4,096-token floor any prompt over ~410 tokens is *cheaper*
  padded past the floor than left short of it — and a shorter, better-written
  instrument can cost several times more per call than a longer one. Measured:
  a 2,737-token instrument uncached costs 5.8× a 4,726-token instrument cached,
  and padding the short one to the floor would make it 6.7× cheaper. Extra
  few-shot examples pay for themselves twice here.
- **`fail_fast`** aborts a batch whose failures are *systematic*. The unit is
  the **item outcome** — both predecessors failed, in opposite directions. A
  count trigger killed a real ~1,500-item run whose failures were a sparse
  recoverable tail (author-reported; the exact figures did not reconstruct at
  review, but the shape of the failure doesn't depend on them). Its rate
  replacement measured *attempts*, which double-counts retried items: the
  nominal 20% threshold aborted healthy runs at a measured 5–12% per-item
  failure rate, and its total-failure condition could not fire at all at
  `num_workers>1`. Now: the first `floor` completed items all failing
  identically, or one signature exceeding 20% of completed items after 30 —
  and an item that retries into success is a success, so a transient 429
  burst no longer reads as systematic. `fail_fast=N` is a floor, `{}` means
  defaults, junk raises `TypeError` (both used to silently mean "5" or
  "off"). Aborts raise `BatchAborted` (completed work is in the stash — a
  rerun resumes), and `errors` records only *attempted* failures, so
  `len(errors)` is a failure count, not the batch size.
- **List-typed schema fields carry a small reliability tax.** Across all three
  providers a model occasionally returns the bare value of a list field
  instead of the whole object. `_diagnose_partial_response` identifies the
  field and the retry names it, instead of complaining about JSON that parsed
  fine — a generic reprompt asks the model to guess what it did wrong when the
  caller already knows.
- **Dropped parameters are recorded, not just logged.** `temperature` is
  rejected by Sonnet 5 / Opus 5 / Fable and was being silently dropped — so
  "administered at temperature 0" read as true with nothing in the output to
  contradict it. It now warns loudly, appears in
  `task.usage.summary_line()` as `DROPPED PARAMS: temperature xN`, and is
  queryable via `report()['dropped_params']` so a methods note can be written
  from a receipt. `LITMOD_STRICT_PARAMS=1` escalates it to
  `DroppedParameterError`. Not the default: every Task sets a temperature, so
  erroring by default would stop the newest Anthropic models running at all.
- **`per_item_usage={}`** on `map`/`imap` gives per-index token counts
  (retries accumulate). Output tokens per item are a free difficulty signal —
  no need to re-tokenise the text.

## Provider routing

`providers.py:route_provider()` dispatches on model string prefix:
- `lmstudio/`, `local/`, `vllm/`, `ollama/` → OpenAI-compat local endpoint
- `claude*`, `anthropic/` → Anthropic SDK (with prompt caching)
- `gpt*`, `o1*`, `o3*`, `openai/` → OpenAI SDK
- `gemini*`, `google/` → Google GenAI SDK
- `claude-cli/` → **DISABLED.** Anthropic does not permit using the Claude Code
  CLI as a programmatic backend; raises unless `LITMOD_ALLOW_CLAUDE_CLI=1`.
  MajorGenreTask still carries `claude-cli/sonnet` as its default model string
  (left in place because `model` is part of the cache key), so it needs an
  explicit `model=` override.

## Model tags

Short tags in `cli/models.py::MODEL_TAGS`. Convention: `<family>-<variant>[-<backend>]`.
Default = LM Studio GGUF. `-mlx` suffix = MLX variant. Anthropic tags (`sonnet`, `opus`, `haiku`) stand alone.

Hosted APIs alias retired names onto current checkpoints without erroring, so
the id you ask for is not always the id you get. Providers log the resolved id
once per run at WARNING when it differs — record *that* as the model of record.
Live case: DeepSeek's `deepseek-chat` **and** `deepseek-reasoner` both resolve
to `deepseek-v4-flash`; pin `deepseek/deepseek-v4-pro` or `-flash` explicitly.

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
