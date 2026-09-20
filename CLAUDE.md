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

Two stash facts that cost money when learned the hard way (lacan seat's field reports):

- **Values are RAW model text, not parsed objects** — deliberately, so a schema or validator change re-parses without re-paying (the 497 pre-validator classify_emotion annotations are the standing example). Raw is model-dependent: haiku fences its JSON in triple-backticks (48/48 measured), sonnet-5 almost never (1/211), deepseek/gpt-4o-mini never. Never read `stash[key]` directly for analysis — `task.results`/`task.df`/`results_history` parse and validate, and `task.parse_and_validate(text)` is the single-item audit path with the same enforcement the run applied.
- **Metadata is part of the key.** It rides there so `task.df` can surface it (the raw value has no home for it), which makes it administration identity: a resumed run must pass byte-identical `metadata_list` or every item silently re-keys and re-pays at full price while `cache_hit_rate` honestly reports 0%. Same principle as "transport is not part of an administration's identity", pointing the other way — and unchangeable now without orphaning every metadata-bearing key.
- **Schema identity rides in the key as the administered BYTES, except under custom cache_keys.** On the extract path the key's `system_prompt` is the built instrument embedding the schema's full JSON spec — a field *or description* change re-keys (money cost, honest receipts: a "cosmetic" questionnaire edit is a questionnaire edit). The key's `schema` field is a name-only discriminator, so old-shape rows under old keys still surface through enumeration (`task.results` walks every administration) — group by `sha256(key["system_prompt"])` against `task.instrument_sha256()` to census administrations; no field-set archaeology needed. **Custom `cache_key` dicts (SequentialTask chunk keys) carry schema by name only** — a questionnaire change does NOT re-key, and a warm read serves answers to a question that no longer exists (malign seat's mirror of the metadata hazard: theirs costs correctness and announces nothing). Sanctioned mechanisms: `SequentialTask.prompt_version` (enters the chunk key; revisions of one construct) or rename the class (a different construct). Backstop with a stated limit: warm hits re-validate against the current schema and recompute on failure, but pydantic ignores extra fields, so a pure field *removal* leaves old rows validating — the one drift validation cannot see.

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

## Survey — typed questions, for System One models (branch `survey-jev`)

**Not merged yet.** A sibling of `Task`, not a mode of it. A Task sends a system prompt built from a Pydantic schema and parses text back; a `Survey` sends state plus a dict of typed questions and gets typed answers back, each carrying a probability.

```python
class Abstraction(Survey):
    questions = {"abstract": Noul(instructions=...),          # -> P(true)
                 "mode":     Choice(instructions=..., criteria={...}),
                 "degree":   Score(instructions=..., criteria=[...])}  # -> position
answers = Abstraction().map(items)
answers[0].p("abstract")     # 0.91, never a bool
```

**Why not a mode of Task.** Compiling a schema to questions throws away the reason to be there: `bool` → Noul → threshold at 0.5 → `bool` pays for a probability and stores a boolean. And the compile fails on the cases that gain most — a `list[Model]` field became tractable only because each item became a question KEY, which needs the item list at call time and is not in the schema. The reverse direction is total, so `as_task()` goes questions → Task and there is no route back.

**Measured, 2026-09-17, 583 passages against a Sonnet reference.** Thresholding each answer at 0.5 and averaging booleans gave a half-century contrast of −0.111, CI [−0.276, +0.050]; averaging the probability gave −0.129, CI [−0.179, −0.081]. Same items, CI 3× tighter, and a discontinuity the booleans could not resolve. The quantization was the whole loss — hence no `bool` property and no implicit threshold anywhere. Best per-field cut ranged 0.24–0.59 while AUC stayed above 0.86: the ordering is good and the absolute placement is not.

**jev accepts BPE fragments as complete words, and deepseek does not.** A truncated token sitting in a frame reads to the model as the completed word: `blaz`, `kimon`, `knick`, `legg`, `trou`, `underp`, `shir` were all called GARMENT.

What generalises is the **direction**: across 479 candidates every disagreement ran one way — jev placing in the worn set what deepseek would not — with **zero reverse cases** (lacan seat, malignment garment run, 2026-09-18). That asymmetry is the finding, and it is why a two-coder veto works here at all rather than being two opinions.

**Count the disagreement, not one of its labels.** The same jev failure arrives under more than one of the other coder's categories: on that run 58 words went to `NOT_A_WORD` and a further 89 to `OTHER_NOUN`, one failure wearing two labels, 147 of 479. Tallying a single category undercounted it by more than half. Treat this as a lint rather than a note about one entry: **anywhere a two-coder gate reports its own disagreement rate, check whether the denominator is admission or a category.** The same undercount was found independently in the producer's own source comment, beside the code computing the veto.

What does NOT generalise is the rate. Those counts are facts about a candidate list drawn from one slot's moving vocabulary, whose fragment base rate is a property of that list; another corpus has a different base rate and therefore a different error rate. Same rule this file applies to model generations, pointed at item sets: do not carry a magnitude across the boundary it was measured inside. The 0.75–0.91 inter-coder scale agreement on accepted words is list-specific in the same way.

**What the veto COSTS is not measured.** A veto discards every word the two coders disagree about, not only the fragments — on a corpus with a low fragment base rate that is a selection on inter-coder agreement, and the surviving set is biased toward easy items. Whether that bias is material is untested and could not be tested here: this list contains no stratum of clean, gradable words the two coders merely disagree about, so there was nothing to measure the cost against. The list that makes the veto obviously worth it is the same list that makes its cost unmeasurable. Carry this as an open question, not as a cheap price.

Pricing it is its own experiment, not an appendix to a run: a corpus with a deliberately low fragment rate and a stratum of well-formed but genuinely ambiguous items, scored veto-in against veto-out on a criterion external to **both coders and to the item selection** — external to the selection too, or the ambiguous stratum is chosen by the same judgement that later scores it.

The mitigation does generalise. A gate distinguishing words from fragments needs `"judge the string EXACTLY as given; do not complete or expand it"` in its `focus`, and a two-coder veto — or, for fragment-vs-word specifically, a plain string check in code — is more reliable than either model alone.

**The API accepts no decoder parameters at all** — no temperature, top_p, penalties or seed — so `_SAMPLING_SUPPORT["typesafe"]` is `_REJECTS` for every one and `effective_temperature` returns None. Consequence for methods: a Survey *cannot* be temperature-pinned, so replicates (`sample=i`) are the only handle on the model's own variance, and identical output across two runs demonstrates caching. `results_history` reads versions back after `force=True`.

**No cache tier.** Every input token bills fresh at $0.042/MTok, output free (console, 2026-09-17 — not a published rate card). `cached: null` in the pricing table is what makes that correct; the cache-floor padding economics of the Anthropic path are actively wrong here.

**An unanswered question is a retryable defect, not a result.** A response omitting questions used to parse, stash, and report success — and because the stash is the artifact, the gap was then served as complete on every later read, free, forever, undiscoverable by rerunning because rerunning hit the cache. Now it raises `IncompleteAnswers`, is retried, and the partial body is not stashed; the error row names the missing ids. Whole-call retry, not subset re-asking: a subset would be answer-equivalent (questions are scored independently) but produces a second administration, and a derived column computed across two of them — a gate from one, scores from the other — is a quantity no response ever had.

**Both arms of a crossing must see one item.** `state_text()` is the single renderer, so `key["state"]` is byte-identical to `as_task_prompts()` output and a cross-arm check is string equality over stash keys — a route that does not depend on either arm reporting it saw the right thing.

**Instrument identity** is the canonical question set as administered bytes, in the key, exactly as `system_prompt` is on the extract path. Order is administered: reordering Score levels renumbers every answer. The question types are ours rather than the SDK's so an SDK release cannot re-key a corpus. Per-item question sets are supported; there is then no static instrument and `instrument_sha256()` refuses rather than fabricating, as `SequentialTask` does. `administration_record()` carries `package_version`, because the question digest cannot see a change to gating, retry or parsing.

**Not stable:** unmerged, no CLI verb, no batch transport, >68 questions per call untested, concurrency unexercised (every test runs `num_workers=1`). `tests/fixtures/typesafe_system_one_response.json` is a recorded live response and the only thing in the suite that would notice the API changing its envelope — but it is a recording, not a contract: re-record it when the parser is touched, and expect it to be stale rather than trusting it to be current.

## Provider drift — the recurring bug class

Every provider bug this package has shipped has the same shape: **the code is
correct about the model generation it was written against, and wrong about the
current one.** A unit test cannot catch it; the mismatch is between our
constants and a live API. Three landed at once in 2026-07:

| Provider | Symptom | Now |
|---|---|---|
| deepseek | `deepseek-chat` silently served the cheap tier — annotations published off it | legacy aliases warn once naming what they resolve to |
| openai | `max_tokens` 400s across the whole gpt-5 tier | `_chat_completion` reads the replacement param out of the error and memoizes it |
| anthropic | `content[0].text` raises on any thinking model (Sonnet 5, Opus 5, Fable 5) | `_response_text` concatenates every text block (interleaved thinking splits the answer across several) |
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

## Decoder parameters — top_p and its neighbours

`top_p` was not a parameter of this package until 2026-08, and the absence was not neutrality. `**kwargs` reached `call_deepseek` and were forwarded verbatim; `call_anthropic`, `call_openai`, `call_google` and `call_local` each declared `**kwargs` and silently discarded them. **A caller who "pinned" `top_p` across four vendors pinned exactly one**, with no error, no warning, and byte-identical cache keys either way — a per-vendor covariate manufactured by our own passthrough asymmetry (lacan seat, after 6,600 passages across four vendors ran unpinned on a surprisal axis that nucleus truncation moves).

```python
task = MyTask(sampling={"top_p": 0.9})          # or LLM(..., sampling={...})
llm.extract(text, Schema, sampling={"top_p": 0.9})
```

One door (`sampling=`), not five keyword arguments — and a bare `top_p=` kwarg now **raises** on every route, because that spelling either vanishes or is sent without entering the key.

**The key records what governed the call and omits the field when nothing was sent.** A value present in a key means that value governed; absent means unpinned — the vendor chose, and which value it chose that day is a fact the key cannot carry and must not pretend to. Written flat, one field per parameter (`top_p: 0.9`, not a nested blob), so "which fields were pinned" is readable off the key itself, which is the claim a methods note has to make. Every pre-2026-08 key is byte-stable, and adding a sixth parameter later orphans nothing.

**Pinning on an unmeasured family raises `SamplingUnmeasured`.** Neither guess is safe: assume it governs and the key carries a pin that may never have applied (the false claim `effective_temperature` exists to prevent); assume it does not and two differently-sampled runs share one cell (the poisoning `thinking_fingerprint` exists to prevent). So the third state is a refusal. Resolving a family means adding a constant to `providers._SAMPLING_SUPPORT` with its measurement date and recipe — reviewable and versioned, deliberately **not** an env flag, because key identity must not depend on ambient configuration that nothing records.

**Measured 2026-08-21** (`litmod probe-sampling --param top_p --rejection-scan`, six calls per model, lacan seat):

| model | result |
|---|---|
| `claude-opus-5` | only 0.99 accepted; everything else, control included, refused **"`top_p` is deprecated for this model."** |
| `claude-sonnet-5` | same |
| `claude-haiku-4-5` | refuses the **parameter** — all six values, one error shape |
| `deepseek-v4-flash`, `gpt-5.4-nano`, `gemini-3.5-flash-lite` | SETTABLE — every value accepted, control included |

**Three models in one vendor's current lineup, three regimes.** A family claim from one member would have been right about Sonnet by luck and wrong about Haiku — which is why `_SAMPLING_FAMILIES` keys on `(route, family-substring)` and `claude-opus-4-8` still raises.

**A correction worth keeping, because the evidence never changed — only the reading.** The Opus pattern was first recorded as "accepts iff equals the undisclosed default", making 0.99 Anthropic's unpublished `top_p` default, and `_SAMPLING_DEFAULTS` briefly held it. Then someone read the 400 text instead of truncating it: `top_p` is *deprecated* there, not validated against a default. A surviving legacy value, a no-op and a default produce identical acceptance evidence — and a parameter that does nothing has no reason to reject anything, so the no-op reading is if anything likelier. **What governed `top_p` on those runs is undisclosed, and 0.99 licenses no claim.** The state is `_DEPRECATED`: never sent, reported in `dropped_params` with the reason, kept out of the key. An accepted-and-ignored value in a cache key is the exact false pin the schema exists to prevent.

Two things changed in the tool as a result. The verdict is now `ONLY_VALUE_ACCEPTED` rather than `DEFAULT_FOUND` — named for what was observed, since the old name was true about the API and false about the world — and it prints **no constant**, deferring to the refusal text. And refusal text is printed **verbatim, never truncated**: the sentence that overturned the verdict sat past a 120-character cut, on screen the whole time. Pinned by test.

**`_SAMPLING_DEFAULTS` is empty**, and empty means *unmeasured* — not 1.0. No vendor default for any decoder parameter has been established by measurement. It never enters a cache key even when populated (a vendor default can change under us, and every old key would then assert a number nobody re-measured); it rides in `task.administration_record()` beside `sampling_unavailable`, which says what a model would not have let a caller pin anyway.

**SETTABLE is not `_ACCEPTS`.** Accepting the absurd control proves the parameter is settable and says nothing about whether it *governs* — DeepSeek's thinking mode accepts all four and ignores them. Those three get no constant; a future corpus can pin them explicitly, but an already-collected one cannot recover what it ran at.

**`temperature` is a denylist, and the asymmetry bites.** A family in `_NO_TEMPERATURE_MODELS` is *measured* to reject; a family absent is only "not known to reject", which is neither "accepts" nor "applies". `litmod probe-sampling --param temperature` now measures it — the one question the support test answers with power to spare, since 0.0 against 1.0 is an enormous effect where `top_p` 0.95 against 1.0 is a sliver. `administration_record()` carries `temperature` (requested) beside `temperature_effective` (what governed), and they differ exactly where a claim would be false. Note the split *within* a vendor: `haiku-4-5` honours a temperature that `sonnet-5` and `opus-5` discard, so "the Anthropic arm was temperature-pinned" can be true of one model and false of the next in the same corpus.

**Per-family, not per-route.** `_SAMPLING_FAMILIES` and `_SAMPLING_DEFAULTS` key on `(route, family-substring)` and are consulted before the route-wide table, because a measurement on one model generation is a fact about that generation — generalising it is this package's recurring bug class, and `claude-sonnet-5` still raises while `claude-opus-5` does not.

`_SAMPLING_DEFAULTS` never enters a cache key (a vendor default can change under us, and every old key would then assert a number nobody re-measured). It rides in `task.administration_record()` as `vendor_sampling_defaults`, so a methods note can state what an *unpinned* field actually ran at. **Empty there means unmeasured, not 1.0.**

Otherwise the table has DeepSeek (`top_p`, `frequency_penalty`, `presence_penalty`, from their docs) and claude-cli (rejects everything, like temperature). **OpenAI, Google, local and the non-Opus Anthropic families all raise until someone probes them.** The vendor-doc ground is worse than "undocumented": DeepSeek documents a default of 1; OpenAI publishes no prose default, only example payloads echoing 1.0, possibly contingent on `reasoning_effort`; Google states the default *varies by model* and is retrievable only via an authenticated `getModel`, while the 3.x guide deprecates the sampling params outright; Anthropic publishes no `top_p` default for any model, and the Opus line rejects non-default values — a fixed number that is both unchangeable and undisclosed. "top_p was 1.0" is a claim the documentation supports for one vendor of four.

**How to resolve a cell — a 200 is not evidence.** DeepSeek's thinking mode accepts all four and ignores them (handled: they land in `dropped_params` instead of the key). Measure **support size**, not distributional distance: truncation's signature is that outputs become *impossible*, not merely rarer. On a flat-continuation prompt ("Pick a random integer between 1 and 100; reply with the number only"), count *distinct* outputs at the pinned value against n at 1.0 — types, not densities, because a distance test at n=100 has almost no power against a default of 0.95 and its null reads as "the parameter did nothing", a false no-op in the one direction that is unrecoverable. Hold temperature fixed and above zero in both arms, and never run it on a schema prompt: JSON-constrained output is low-entropy by construction and shows a ceiling effect as a no-op.

**A route that truncates unasked cannot be partially pinned.** `providers._SAMPLING_COMPANIONS` names parameters a stack applies whether or not you set them; pinning anything on such a route requires pinning the companion explicitly. Currently just `local` → `top_k`. The measurement that motivates it is adjacent rather than ours, and the difference matters: transformers 5.4.0 applies an effective `top_k=50` when the field is *absent* while `GenerationConfig().top_k`, the checkpoint's own config and the merged `model.generation_config.top_k` **all report None** (lacan seat, SmolLM2-360M-Instruct, 2026-08-21, CPU and MPS alike: 200 draws reached max rank 47 unset against 7090 at `top_k=0`). That is the HF `generate()` path, which we do not use — `local/` speaks HTTP to llama.cpp, vLLM, LM Studio or Ollama, each with its own defaults (llama.cpp has historically applied `top_k=40` and `top_p=0.95` unasked). The number is not ours; the class is, and four backends behind one prefix is worse. **"The config says None" is not evidence a parameter is unset** — which is precisely the inference this package exists to refuse. Cost of not having the rule: with an unrequested `top_k=50` in force, nucleus sizes of 436/223/53 at `top_p` 0.95/0.9/0.7 all collapse to the same top-50 set, so a four-arm sweep varies nothing and returns a flat line reading as "truncation has no effect".

```bash
litmod probe-sampling --model deepseek/deepseek-v4-pro --param top_p --n 20
```

Runs an UNSET arm plus one arm per value, counts distinct outputs, and prints the constant to paste into `_SAMPLING_SUPPORT`. It writes no keys and touches no stash — that is what makes its `allow_unmeasured=True` safe, and that flag has exactly this one caller. It also reports whether the **instrument** was validated, which is the part that keeps a null honest: a no-move at `top_p=0.95` is uninformative alone, because 0.95 removes a sliver of tail mass that n draws mostly miss. The low arms are the positive control — if 0.7 collapses support, the test has demonstrated sensitivity and the 0.95 null becomes a *bound* on the effect; if no arm moves at all, nothing was measured and the tool says so rather than reporting "ignored". Known limit: for a hosted model the sharper statistic (rate of draws outside a reference nucleus, a 5%-vs-0% test) needs the model's own distribution, so it works locally and not through an API.

**`--rejection-scan` maps which values a family will accept, in under ten calls.** Cheap enough to run before spending n on any restrictive family. It carries a **negative control** (an in-range value that is certainly nobody's default), because the interesting premise — that an API accepts a value *iff* it equals the default, making acceptance an equality test against an unpublished number — is itself testable and fails silently in two directions. Four outcomes:

- `ONLY_VALUE_ACCEPTED` — one value accepted, control refused. **Consistent with accepts-iff-default AND with a surviving legacy value or a no-op under a deprecated parameter, and the acceptance pattern cannot tell them apart.** So it prints no constant and defers to the refusal text, which usually can. This verdict was originally called `DEFAULT_FOUND` and that name shipped a false reading of a true observation; see the correction below.
- `NOT_SETTABLE` — everything refused with one error shape, control included → `_REJECTS`.
- `SETTABLE` — the control was accepted too, so the equality premise is false; fall back to the support test. Without this branch a permissive family returns the first candidate as a "default", and 1.0 leads the list.
- `INCONCLUSIVE` — refusals differ in shape; read them.

**Refusal text is printed verbatim and never truncated**, because the sentence that decides between those readings lives in the message, and clipping it at 120 characters is how a withdrawn constant got written in the first place.

**Replicates: `sample=i`.** n samples of one prompt were n identical calls and therefore one cache cell — the second through nth collapsed into cache hits and the run reported one observation as n. `sample=` is a replicate index that keys each one separately. Its own field rather than a metadata entry, so it cannot collide with real metadata, and omitted when None. Not named `n=`: OpenAI's API already has an `n` meaning one call returning n completions, which is a different thing.

**`calls == 0` is now two distinguishable states.** Everything cached, or everything failed — the usage sink only fires on success, so the tracker could not tell them apart and a missing `ANTHROPIC_API_KEY` read as a 100% cache hit rate. `UsageTracker` counts `failed_calls` and `failure_types` (attempts, not items) at the single dispatch point in `_call_provider`, and `summary_line()` says which state a zero-call run is in.

**`usage_log` works off `LLM`, not just `Task`.** `LLM(usage_log="run_name")` writes the same per-batch JSONL record `Task` writes, from `llm.append_usage_log` — one writer, because a Task-only sink is why a production run driven straight off `LLM.map` kept no durable record of a single token. In-process `llm.usage.report()` was always live; that is what makes the gap easy to miss.

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
  *not* included in `candidates_token_count` — usage sums them so
  output_tokens means billed-output everywhere. `google_thinking_setting`
  now sends the off-equivalent by default — and **the two Gemini
  generations take DIFFERENT parameters**: 2.5 wants `thinking_budget=0`;
  3.x wants `thinking_level` ("minimal"/"low"/"medium"/"high"), where the
  budget survives only as a deprecated back-compat field whose ZERO is a
  generic 400 (probed live on 3.6-flash — the error prose drifted too, so
  the loud-failure wrap matches on what WE sent, not on Google's message).
  Per Google's docs 3.x cannot fully disable thinking; "minimal" is the
  documented off-equivalent, measured at zero reported thoughts where the
  API default thought 370. Also measured: `thinking_budget` is a TARGET,
  not a cap — budget 128 produced 190 thought tokens on a real
  instrument. Field cost of the gap this closes: a 14,520-call
  gemini-3.6-flash run billed ~966 output tokens/call (~96% thoughts,
  £147); the same call now runs at ~15. gemini-2.5-pro and gemini-3.1-pro
  cannot express "off" at all (3.1-pro rejects even level "minimal"; its
  floor is "low", ~4x cheaper than its "high" default, available as an
  explicit opt-in) — they keep the Fable arrangement: nothing sent,
  warned once, doctor WARNs, keys byte-stable. 3.x default keys say
  `level:minimal`, never "disabled" — no key claims an off state the
  model cannot deliver. OpenAI's gpt-5.4 tier is reasoning-capable but
  measured ZERO reasoning tokens on all three doctor probes — nothing to
  cut, so nothing was changed on no evidence. Gemini's implicit prompt
  caching has a silent floor of its own (4,096 tokens on 3.x): a
  3,906-token instrument ran 14,520 times at full input price ~130
  tokens under it, so `cache_warning` now covers Gemini's zero-reads
  shape alongside Anthropic's.
- **The cache key carries the thinking state and the *effective*
  temperature.** Thinking-on and thinking-off output must never share a key
  (a non-forced rerun would hand back one as the other, silently), and a key
  asserting `temperature: 0.0` on a model that rejected it plants a false
  methods claim in the durable artifact. Keys record `thinking` only when a
  thinking parameter is actually sent, and temperature as None where it is
  known-rejected (Sonnet 5/Opus 5/Fable, Opus 4.7/4.8, claude-cli) or
  known-ignored (DeepSeek in anything but explicit disabled). Pre-schema
  entries split two ways, and only one stays orphaned:
  **behaviour-differed** entries (sonnet-5/opus-5/deepseek extract, produced
  with thinking on and uncontrolled sampling) are deliberately unreachable —
  serving them under a thinking-off key is the poisoning the schema
  prevents; **inert-field** entries (Opus 4.7/4.8, Fable, claude-cli — the
  old key recorded a temperature that never applied, so the annotation is
  byte-identical to a fresh call's) are served by a read-only legacy
  fallback and copied forward under the new key. The predicate is exact:
  a legacy read is offered iff this call sends no thinking parameter,
  which is what old code always did. Deliberately NOT a flag — key
  identity must not depend on ambient configuration that nothing records.
  Custom `cache_key=` dicts gain `model`/`schema` via setdefault (keys that
  already carry `model`, like SequentialTask's chunk keys, are untouched;
  bare custom keys get no legacy fallback — a pre-schema custom key could
  hold any model's annotation, and that ambiguity is the defect).
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

## Batch transport

`task.map(prompts, batch=True)` / `litmod run --batch` submits via the
provider's batch API — **50% pricing on Anthropic, OpenAI and Google;
DeepSeek has none (verified) and local endpoints have nothing to discount:
both raise.** Blocks until the batch completes (usually well under the 24h
window); `LLM.extract_batch(..., wait=False)` returns a `BatchHandle` that
reconstructs from the ledger in a fresh process.

What makes it safe to use on registered work:

- **Identical stash keys.** Batch results write under exactly the keys the
  streaming path computes — transport is not part of an administration's
  identity — so warm reads serve batch results, a half-warm stash batches
  only its cold half, and force/cache-cold discipline is unchanged.
- **Ledger before money** (`data/batch_ledger/`, a sibling-root pairtree
  HashStash — settled with the hashstash seat from multi-process test
  receipts: sibling because `parent.clear()` destroys nested sub-stashes
  and a billing record must not share a lifetime with a cache; pairtree
  because atomic-rename-per-version has no torn-line failure class;
  `append_mode` versioning IS the submitting→open→closed audit trail via
  `_Ledger().history(cid)`). Resolution is **per item**: a rerun over any
  overlap — subsets, after partial progress — resumes the open batch; a
  process that died inside the submission call leaves the one ambiguous
  state, which is first **reconciled against the provider itself**:
  submissions are tagged provider-visibly (OpenAI batch metadata, Google
  `display_name`, Anthropic by deterministic custom_id content-match on
  ended batches), so `reconcile(sub)` learns definitively whether the
  death fell before the API accepted (no batch — abandon and resubmit)
  or after (a live billable batch with no recorded id — attach and
  resume; blind reclaim here is the double-bill). Only an inconclusive
  lookup still raises `AmbiguousBatchState`, carrying the candidates,
  with working operator resolutions (`reconcile(sub)` retry /
  `_Ledger().attach(sub, batch_id)` / `.abandon(sub)`), and `force=True`
  cannot bypass it. Neither crash window shows up under contention
  testing — only under a kill in a specific millisecond (hashstash
  seat's analysis) — so both are pinned by forged-death tests and were
  drilled live against real batches on all three providers. The
  read-decide-submit section runs under hashstash's cross-process
  `key_lock` (an unlocked race measured 8/8 double-submissions). Stated
  limits: the lock is a local flock — one machine per task's batches —
  and no fsync means an OS crash can lose page-cached records.
- **Probe-first** (default): one sync call before submission catches
  rejected params where a batch has no repair loop, warms the OpenAI
  param-rename memo, and is `fail_fast`'s only meaningful moment — after
  submission the spend is committed.
- **Receipts intact**: per-item usage with reasoning evidence and
  `response_model`, tagged `transport: "batch"`, into `usage_log` and
  `costs.price_report(batch=True)`.
- **Effective discount ~48-49%**, not 50: failed/unparseable items fall
  back to the sync path at list price (with the full retry machinery,
  partial-field reprompts included), and the probe runs sync. Stated in
  the completion log. Anthropic batches default `cache_ttl="1h"`.
- No images on the batch path (payload size) — the concurrent path takes
  those.

## Raw-response sidecar

`Task(raw_log=True)` / `LLM(raw_log=True)` keeps the provider's serialized reply, whole, in `data/raw_responses` — a sibling-root append-mode pairtree stash keyed by the SAME dict as the annotation, so raw ↔ annotation joins are a lookup and retries/forced reruns append versions. The receipts only carry fields the normalizers knew to extract when written, and every provider-drift bug was diagnosed from a raw response that had already been discarded — the sidecar is insurance against the unanticipated field (thinking blocks, safety ratings, fingerprints). Envelopes carry `transport` (`sync`/`sync-probe`/`sync-fallback`/`batch`) agreeing with the usage rows. Off (the default) is genuinely off: no sink is constructed, providers skip serialization, and the annotation stash is byte-identical either way — write-path isolation is pinned by test, not stated (an additive feature that shares a write path is additive only until it is not). A sidecar write failure logs and never fails the run it documents — but it is COUNTED, because never-fail alone makes the sidecar's completeness unverifiable (a missing receipt is indistinguishable from a call never made): `rawlog.receipt()` is this process's `{recorded, failed, errors}` — and `failed` counts *sidecar* faults only, by construction: the sink fires exclusively with a body in hand, so transport failures can't inflate it (converse bonus: it fires *before* parsing, so failed extractions still leave their bodies). `rawlog.audit(keys)` is the durable post-hoc check — present/missing/degraded/corrupt over a *scoped* key set — but a run fires many times (resumption is the normal case), so receipts are retained per firing (`flush_receipt()` at the imap/map/collect boundaries — all in `finally` — and immediately on any counted failure; `receipts()` reads them back, fork-safe: a child re-mints its firing and counts from zero), and **`certify(keys)` is the one-call composed claim** — from a Task, `task.certify_raw()` sources the denominator from the annotation stash, making the correct call the easy one. `complete=True` requires a *usable* body for every key (not missing, not degraded — empty/unserialisable — not corrupt; one corrupt entry reports per-key rather than denying the other 9,999) and is retention-independent (presence proves it). Two semantics the review wave forced: **presence is key-level by default** — a forced rerun whose bodies were all dropped still reads present on the previous run's bodies; pass `since=<run start>` for the administration-level claim — and **absence is explained by per-key ATTRIBUTION, never by arithmetic on global counts**: a missing key counts as `known_drops` only if some firing receipt recorded *that key* as dropped, so an unrelated run's old outage can never absolve this run's missing bodies and `unaccounted` — the alarm — cannot disarm as the store ages (`drops_all_firings` survives as context, never arithmetic). Treat unaccounted as dropped, never clean. On a batch-fallback item `get(key)` returns the retry's body; `history(key)` shows the failed batch body beneath it (error results are captured too — the provider's error object is the richest drift evidence the transport produces). Not covered: `claude-cli/` (flattened transport). Same fsync caveat as the ledger: an OS crash (not process crash) can lose a page-cached body counted as recorded. For recent batch runs there is also a zero-code fallback: providers retain full raw results server-side (per their docs as of 2026-08 — Anthropic 29 days, Google 6 weeks, OpenAI until file deletion; verify before relying) and the ledger holds every batch id.

## Costing

`costs.py` prices workloads against the packaged multi-provider table
(`largeliterarymodels/model_pricing.json` — Anthropic/OpenAI/DeepSeek/Google,
fetched with source URLs; every estimate prints the fetch date, because a
pricing table is a constants file and constants rot). Validated against a
real invoice: predicted $1.8511, billed $1.86 — pinned as a test.

```python
from largeliterarymodels import costs
results = task.map(prompts)
costs.print_report(task.model, task.usage.report())   # this run, priced
costs.price("gemini-3.6-flash", fresh=74e6, output=14e6)  # counterfactual
```

`litmod price --fresh N --cached N --output N [--model M | --provider P]
[--batch] [--prefix-tokens N] [--on DATE]` does it from the shell.

What the numbers know that a naive rate-card lookup does not:
- **`cached: null` means NO cache tier** — those tokens bill at full input
  rate, the largest driver on high-cache workloads.
- **Prospective estimates are cache-floor-aware**: pass `prefix_tokens` (the
  old `estimate()` API passes it automatically) and a prefix below the
  model's floor re-bills at full rate with the padding economics stated,
  instead of promising a discount the provider silently declines. Floors
  come from `providers.cache_minimum_tokens` — one source, not two.
- **Reasoning floors follow measured behaviour**, not the vendor's product
  tier: fable is `reasoning: false` as a product and unavoidable-thinking
  as a fact; the FLOOR warning follows the fact.
- **Dated rows**: sonnet-5's introductory pricing ends 2026-08-31 (a 44%
  jump); `on=` selects the row and estimates within 30 days of a boundary
  warn.
- **Batch discounts are per-provider facts**: anthropic/openai/google 0.5,
  deepseek has NO batch API — requesting batch there prices at list, loudly.

The stash stores only response text, so token counts die with the process —
a past run cannot be priced from the stash. `Task(usage_log=True)` appends
one JSONL record per `map()`/`imap()` batch (batch totals + per-item rows
incl. `response_model`, with failed items marked so spend on a
non-annotation never reads as an annotation's price; single `run()` calls
are not logged) to `data/usage_logs/<task>.jsonl`;
`costs.price_report(rec["model"], rec["batch"])` prices it later. Additive:
no stash value or key is touched, and a receipt-write failure logs an
error rather than failing the run it receipts. Caveat for fan-out jobs:
appends from multiple *processes* to one file are not locked.

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

## Data root

ALL persistent state — stash, batch ledger, raw sidecars, usage logs, human annotations — derives from one root, resolved by `llm._data_dir()`: `LITMOD_DATA_DIR` if set; else a *non-empty* package-relative `data/` **outside site-packages** (the clone/editable workflow — an existing repo's history stays put); else `~/.largeliterarymodels/data`. Data inside site-packages is never used, even when present: the pre-fix derivation was `__file__`-relative, so a plain pip install silently pointed the stash *and the ledger* into the venv, where the run reported success, `certify()` said complete, and the next `--force-reinstall` deleted the lot (lacan seat's field report). A path pip owns is not storage. Non-editable installs that want a project's existing data must set `LITMOD_DATA_DIR` to that project's `data/` dir.

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
