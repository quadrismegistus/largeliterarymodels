"""Passage-level narratological and interpretive annotation — PassageFormTask.

Complement to PassageContentTask (classify_passage_content.py). Where content
asks WHAT happens/is-present, Form asks HOW it is narrated and what ABSTRACT
features structure the passage. Designed for frontier models (Sonnet, Opus)
since several fields require interpretive judgment that local models handle
unreliably.

Fields draw on:
- **Genette** (Narrative Discourse): story_time_span (duration), narrative_frequency
  (singulative/iterative), narration_modes, interiority_focus
- **Bakhtin**: voices_heterogeneous (heteroglossia), setting+distance_traveled
  (chronotope — shared with PassageContentTask.setting)
- **Lukács** (Narrate or Describe): narrate_vs_describe
- **Jameson** (Antinomies of Realism): mood (récit vs affect)
- **Barthes**: has_reality_effect (l'effet de réel)
- **Abstract Realism (Ch5)**: concrete_bespeaks_abstract, abstractions_as_agents,
  characters_known_by_reputation, characters_known_by_physical_appearance,
  uses_nominalization, physicality_level, distance_traveled

Designed to be run alongside PassageContentTask on the same passage. Both share
`format_passage` from classify_passage.py.
"""

from typing import Literal
from pydantic import BaseModel, Field
from largeliterarymodels.task import Task
from largeliterarymodels.tasks.classify_passage import format_passage  # re-exported


# ── Vocabularies ──────────────────────────────────────────────────────────

NarrationMode = Literal[
    "dramatized_scene",
    "narrative_summary",
    "epistolary",
    "embedded_letter",
    "first_person",
    "authorial_commentary",
]

InteriorityFocus = Literal[
    "character_thoughts",
    "free_indirect_or_monologue",
]

EmotionalRegister = Literal[
    "extreme",
    "restrained_or_ironic",
]

# Log-scale short codes; analysis layer converts to canonical time/distance.
StoryTimeSpan = Literal[
    "5m",    # moment / brief exchange
    "1h",    # within an hour
    "6h",    # half a day
    "1d",    # single day
    "3d",    # a few days
    "1w",    # ~week
    "1mo",   # weeks to a month
    "3mo",   # several months
    "1y",    # ~year
    "years", # multi-year span
]

DistanceTraveled = Literal[
    "0",        # stationary — single room, dialogue scene
    "10m",      # within a room / gestures
    "100m",     # around a house / garden
    "1km",      # within a town / estate grounds
    "10km",     # day's journey / between towns
    "100km",    # cross-regional
    "1000km+",  # international, transatlantic, cross-continental
]

PhysicalityLevel = Literal[
    "absent",      # no bodily/sensory detail
    "incidental",  # brief physical detail in passing
    "sustained",   # sustained focus on bodies, sensation, physical detail
]

NarrativeFrequency = Literal[
    "singulative",  # "he went to bed at nine" — one event told once
    "iterative",    # "every Sunday they dined at..." — habitual/recurring
    "mixed",        # combines both registers
]

NarrateVsDescribe = Literal[
    "narrate",   # event-embedded meaning; what happens matters for plot
    "describe",  # reified spectacle; world stands as object for contemplation
    "balanced",  # both registers in equal measure
]

MoodAxis = Literal[
    "event_driven",        # story-time advances via events (Jameson's récit)
    "affective_stillness", # atmosphere/mood dominates; little event (Jameson's affect)
    "mixed",
]


class PassageFormAnnotation(BaseModel):
    """Narratological + interpretive annotation for a fiction passage."""

    # ── Meta-filter ───────────────────────────────────────────────────────
    is_prose_fiction: bool = Field(
        default=True,
        description="True for narrative prose fiction (novels, tales, romances, short "
        "stories, prose narrative). FALSE for: poetry (verse, lyric, song), drama "
        "(playscript with speaker names + stage directions), sermon, essay, treatise, "
        "letter *without* surrounding narrative frame, cookbook/manual, scholarly "
        "apparatus (preface, index, table of contents), or non-narrative material. "
        "Tagged Fiction at the text level does NOT guarantee a passage is prose fiction — "
        "embedded poems, plays, and paratexts are common. When in doubt (e.g. allegorical "
        "prose, ornate oratory within a novel), prefer TRUE.",
    )

    # ── Narration modes (multi-label) ─────────────────────────────────────
    narration_modes: list[NarrationMode] = Field(
        default_factory=list,
        description=(
            "Modes of narration present in the passage. Multi-label — include EVERY "
            "mode that appears, even briefly. A typical passage uses 2-4 modes. "
            "\n\n"
            "dramatized_scene = real-time action and dialogue ('showing'); characters speak "
            "and move moment-to-moment. "
            "\n\n"
            "narrative_summary = ANY stretch where the narrator reports events at a distance "
            "rather than staging them moment-by-moment. INCLUDE whenever ANY non-dialogue "
            "narration is present alongside other modes, even in one or two sentences — "
            "bridging sentences between dialogue ('he placed himself in the corner and "
            "waited'), compressed recaps ('in this manner they passed the time till the "
            "clock struck five'), scene transitions, embedded retrospective narration of "
            "off-stage events. Do NOT reserve this for passages that are purely summary; a "
            "passage dominated by dialogue with even brief narratorial bridges should still "
            "include narrative_summary. Under-reporting this mode is the common failure. "
            "\n\n"
            "epistolary = the passage IS a letter (salutation, signature, letter-form narration). "
            "\n\n"
            "embedded_letter = passage QUOTES or TRANSCRIBES a letter within a non-letter frame. "
            "\n\n"
            "first_person = narrated in first person (testimonial, picaresque 'I', memoir form). "
            "\n\n"
            "authorial_commentary = narrator steps outside the story to address the reader, "
            "moralize, or make general statements (Fielding's digressions, Eliot's reflections, "
            "Thackeray's asides). Also: a character-narrator offering didactic/homiletic "
            "commentary addressed to the reader ('Such a grace, without thinking of it, every "
            "one should strive to have'). Distinct from interior reflection that stays "
            "within the fictional frame."
        ),
    )

    interiority_focus: list[InteriorityFocus] = Field(
        default_factory=list,
        description=(
            "Access to character mental states. "
            "character_thoughts = any access to a character's internal mental states (by any technique). "
            "free_indirect_or_monologue = specifically free indirect discourse (narrator voice "
            "blended with character) or first-person interior monologue."
        ),
    )

    emotional_register: list[EmotionalRegister] = Field(
        default_factory=list,
        description=(
            "Affective register of the PASSAGE (narrative framing), not merely of the characters. "
            "extreme = the passage presents terror, ecstasy, madness, despair, rage with "
            "narrative intensity (gothic/sensationalist register). "
            "restrained_or_ironic = decorous, detached, ironic, or satirical framing "
            "(polite/neoclassical register). "
            "A passage where a character feels extreme emotion but the narrator frames it "
            "ironically should be restrained_or_ironic, not extreme."
        ),
    )

    # ── Genette: diegetic scope ───────────────────────────────────────────
    story_time_span: StoryTimeSpan = Field(
        default="1h",
        description=(
            "Approximate elapsed story-time covered by the DOMINANT NARRATION of the passage. "
            "Pick the closest bucket: 5m (a moment / brief exchange), 1h (within an hour), "
            "6h (half a day), 1d (a single day), 3d (a few days), 1w (~one week), "
            "1mo (weeks to a month), 3mo (several months), 1y (~year), years (multi-year span)."
            "\n\n"
            "DOMINANT NARRATION rule: identify which narrative layer occupies the majority of "
            "the passage, and measure that layer's story-time. "
            "\n\n"
            "Case A — frame dominates (dialogue/present-action fills most of the passage, with "
            "only brief references to past events): measure the frame. Example: a parlour "
            "conversation in which one character mentions 'last year I was in Ostend' is 1h "
            "(the conversation), not 1y — because the frame dominates and the embedding is "
            "incidental. "
            "\n\n"
            "Case B — embedded retrospection dominates (the passage opens with a short "
            "setup and then gives over to a character's extended first-person recounting, or "
            "to a narrator's multi-paragraph flashback): measure the embedded content. "
            "Example: two sentences of 'Brother, let me tell you what happened…' followed by "
            "a page of picaresque travel covering weeks and hundreds of miles → measure the "
            "picaresque content. Frame-tale, epistolary, and amatory/romance fiction "
            "frequently fit this pattern. "
            "\n\n"
            "When genuinely split ~50/50, measure whichever layer spans the longer "
            "story-time (rewards narrative reach). `has_retrospective_embedding=True` fires "
            "whenever substantial embedding is present, regardless of which layer 'wins' here."
        ),
    )

    distance_traveled: DistanceTraveled = Field(
        default="0",
        description=(
            "Approximate spatial range covered by characters in the DOMINANT NARRATION. "
            "0 = stationary (single room, dialogue scene, no movement). "
            "10m = within a room (gestures, pacing). "
            "100m = across a house or garden. "
            "1km = within a town or estate grounds. "
            "10km = day's journey / between nearby towns. "
            "100km = cross-regional travel. "
            "1000km+ = international, transatlantic, cross-continental."
            "\n\n"
            "DOMINANT NARRATION rule (same as story_time_span): measure distance in whichever "
            "narrative layer fills most of the passage. "
            "\n\n"
            "Case A — frame dominates: a parlour conversation in which someone mentions a "
            "transatlantic trip is 0 (the parlour), not 1000km+. The embedding is incidental. "
            "\n\n"
            "Case B — embedded retrospection dominates: a brief setup followed by a page of "
            "a character recounting travel from St. Omers to Amiens to Paris → measure that "
            "embedded travel (100km). The embedded narration IS the passage's narrative "
            "substance."
        ),
    )

    has_retrospective_embedding: bool = Field(
        default=False,
        description=(
            "Does the passage contain embedded retrospective narration — a character or "
            "narrator recounting past events within the frame? Typical markers: a character "
            "says 'I was once in Ostend when…' and launches into a multi-sentence recap; a "
            "narrator pauses the present action to summarize earlier history ('Twenty years "
            "before, he had…'); an embedded letter or diary entry describing past events. "
            "\n\n"
            "True if the retrospective embedding is substantial (more than a passing phrase). "
            "False if past-tense references are only incidental ('as he had promised') or if "
            "the whole passage is simply a past-tense narration with no nested retrospective "
            "layer. This field captures 'time depth' independently of story_time_span and "
            "distance_traveled, which measure only the frame."
        ),
    )

    # ── Genette: frequency ────────────────────────────────────────────────
    narrative_frequency: NarrativeFrequency = Field(
        default="singulative",
        description=(
            "Genette's narrative frequency. "
            "singulative = the passage narrates events that happen ONCE ('he went to bed at nine'). "
            "iterative = the passage narrates HABITUAL or recurring events "
            "('every Sunday they dined at...', 'each morning she would...'). "
            "mixed = both registers present."
        ),
    )

    # ── Lukács: narrate vs describe ───────────────────────────────────────
    narrate_vs_describe: NarrateVsDescribe = Field(
        default="balanced",
        description=(
            "Lukács's 'Narrate or Describe?' axis. "
            "narrate = events drive meaning; description is instrumental to action "
            "(Scott, Tolstoy's race in Anna Karenina). "
            "describe = world presented as reified spectacle for contemplation, "
            "detached from plot significance (Flaubert, Zola's race in Nana). "
            "balanced = neither register dominates."
        ),
    )

    # ── Jameson: récit vs affect ──────────────────────────────────────────
    mood: MoodAxis = Field(
        default="event_driven",
        description=(
            "Jameson's récit/affect distinction. "
            "event_driven = story-time advances; something happens (classical récit). "
            "affective_stillness = mood/atmosphere dominates with little event; "
            "the passage dwells in feeling or sensation (late James, Woolf, Flaubert). "
            "mixed = combines both."
        ),
    )

    # ── Physicality (Ch5 / embodiment) ────────────────────────────────────
    physicality_level: PhysicalityLevel = Field(
        default="incidental",
        description=(
            "How much the passage dwells on bodies, sensation, and concrete physical detail. "
            "absent = abstract, conceptual, minimal bodily/sensory content. "
            "incidental = brief physical detail in passing. "
            "sustained = sustained focus on bodies, physical sensation, embodied action "
            "(gothic terror scenes, Dickensian description, naturalist detail)."
        ),
    )

    # ── Bakhtin: heteroglossia ────────────────────────────────────────────
    voices_heterogeneous: bool = Field(
        default=False,
        description=(
            "Bakhtinian heteroglossia: does the passage mix multiple socially-marked "
            "voices or registers (different class idioms, professional jargons, dialects, "
            "ideological languages)? Monological passages (single narrator voice, even with "
            "dialogue) are False. Polyphonic passages (Dickens, Eliot, Scott) are True."
        ),
    )

    # ── Barthes: reality-effect ───────────────────────────────────────────
    has_reality_effect: bool = Field(
        default=False,
        description=(
            "Barthes's l'effet de réel ('The Reality Effect', 1968). A detail is a "
            "reality-effect ONLY if it is narratively UNMOTIVATED — no plot function, no "
            "symbolic weight, no characterological significance — and its only purpose is "
            "to signify 'here is the real world'. Barthes's canonical example is Flaubert's "
            "barometer on Madame Aubain's piano: it does nothing, means nothing, and that "
            "IS the point. "
            "\n\n"
            "A detail is NOT a reality-effect if it: "
            "(a) enables the plot — a dark corner where a character hides, a lock to be "
            "picked, a coin given as a bribe, a window through which something is seen; "
            "(b) carries symbolic or allegorical meaning — a storm foreshadowing disaster, "
            "a ruin signifying decay, a garden as moral landscape; "
            "(c) characterizes a person — a threadbare coat that reveals poverty, a room's "
            "disorder that mirrors its occupant. "
            "\n\n"
            "Reality-effect requires a detail that is GRATUITOUS in plot and meaning terms "
            "and present ONLY because 'the world contains such things'. Canonical in "
            "Flaubert, Balzac, mature realism (roughly 1830+); rare in romance and early-18C "
            "fiction where almost every detail earns its keep narratively. "
            "\n\n"
            "When in doubt, default FALSE. Over-attribution (calling any concrete detail a "
            "reality-effect) is the common failure mode. Distinct from "
            "characters_known_by_physical_appearance, which concerns persons."
        ),
    )

    # ── Abstract Realism (Ch5) axes ───────────────────────────────────────
    concrete_bespeaks_abstract: bool = Field(
        default=False,
        description=(
            "Does the passage bind concrete particulars to abstract meanings in a sustained way? "
            "Examples: C17 romance where concrete landscape is legible as allegorical/moral terrain; "
            "C19 novel where concrete commodities/objects bespeak social/abstract forces. "
            "C18 novel characteristically LACKS this binding (concrete stays merely concrete). "
            "Central Ch5 axis: if this drops in C18 and rises in C17 and C19, confirms the thesis."
        ),
    )

    abstractions_as_agents: bool = Field(
        default=False,
        description=(
            "Do abstract nouns or forces act as grammatical AGENTS in the passage? "
            "Examples: 'Extravagance required this sacrifice,' 'the world made him such,' "
            "'Fortune frowned upon them.' Austenian signature. Distinct from "
            "allegorical_personification (which names Virtue/Despair as characters); "
            "here the abstraction simply acts via ordinary-noun grammar."
        ),
    )

    characters_known_by_reputation: bool = Field(
        default=False,
        description=(
            "Are characters identified/known primarily through social reputation, rumor, "
            "or community knowledge — rather than physical or psychological interiority? "
            "Williams's 'knowable community'; characteristic of Austen, early realist novel."
        ),
    )

    characters_known_by_physical_appearance: bool = Field(
        default=False,
        description=(
            "Are characters identified/known primarily through detailed PHYSICAL description "
            "(faces, bodies, gestures, clothing as physical fact)? Reification signature — "
            "Dickens, Brontë, Balzac. Distinct from has_reality_effect (which concerns world-objects)."
        ),
    )

    uses_nominalization: bool = Field(
        default=False,
        description=(
            "Does the passage rely on abstract capitalized or singularly-marked nouns where "
            "concrete verbs/descriptions would do? Haywoodian signature: "
            "'those Insinuations,' 'her Confusion,' 'the Ardour of his Passion.' "
            "Abstract noun phrases doing the work that concrete scene-making might."
        ),
    )

    # ── Free-text ─────────────────────────────────────────────────────────
    notes: str = Field(
        default="",
        description="Optional: one sentence on anything surprising, ambiguous, or driving a "
        "non-obvious judgment. Empty if nothing notable.",
    )

    confidence: float = Field(
        default=0.5,
        description="Overall confidence 0.0 to 1.0. Lower for ambiguous or fragmentary passages.",
    )


SYSTEM_PROMPT = """You are annotating passages from English-language fiction (1500-2020) for a study of narrative form and the history of abstract/concrete language in fiction.

You will receive a single passage (~500-1500 words) with metadata (title, author, year). Classify it along narratological axes drawn from Genette (duration, frequency, narration mode), Bakhtin (heteroglossia, chronotope), Lukács (narrate vs describe), Jameson (récit vs affect), Barthes (reality-effect), and a set of axes specific to the "Abstract Realism" project (concrete_bespeaks_abstract, abstractions_as_agents, ways-of-knowing-characters, nominalization).

The `is_prose_fiction` field is a meta-filter: most passages ARE narrative prose fiction, but texts tagged Fiction can contain embedded poetry, dramatic scenes, paratexts (indexes, prefaces), letters without narrative frame, or sermons. Mark `is_prose_fiction=False` for these; mark True for ordinary narrative prose. When `is_prose_fiction=False`, the other fields may be empty/default — the narratological axes don't apply cleanly to non-narrative material.

## Classification principles

- **Interpretive judgment is expected**: these are not surface features. You must read the passage carefully for narrative form, register, and the relationship between concrete detail and abstract meaning.
- **In-text only**: base classifications on the passage text, NOT on outside knowledge about the author. Even when author-signal is strong (Austen → knowable-community), judge from what the PASSAGE shows.
- **Empty lists and default scalars are acceptable** when features are genuinely absent or weakly present.
- **Ambiguity → lower confidence**: for unclear cases, lower confidence rather than force a call.
- **Key distinctions to keep straight**:
  - `allegorical_personification` (in PassageContentTask) = named abstractions as characters (Virtue, Christian). `abstractions_as_agents` (here) = ordinary-noun abstractions grammatically acting ('Extravagance required...').
  - `characters_known_by_physical_appearance` = persons described physically. `has_reality_effect` = world-objects, and only when NARRATIVELY UNMOTIVATED (no plot/symbolic/characterological function). Plot-functional concrete details (a dark corner used for hiding, a bribe, a letter) are NOT reality-effects.
  - `emotional_register` = NARRATIVE framing. A passage where characters feel intensely but the narrator ironizes is `restrained_or_ironic`, not `extreme`.
  - `narrate_vs_describe` (Lukács) vs `concrete_bespeaks_abstract` (Ch5): Lukács is about whether world is action-embedded or reified spectacle; Ch5 is about whether concrete particulars CARRY abstract meaning. A passage can be `describe` + `concrete_bespeaks_abstract=True` (C19 reification that still bespeaks abstract forces).

- **Calibration of commonly-misjudged fields**:
  - `narration_modes`: over-include rather than under-include. If the passage has any bridging narration between dialogue or any compressed-time summary, add `narrative_summary` alongside `dramatized_scene`. A passage with only dialogue and no narratorial voice is rare.
  - `has_reality_effect`: UNDER-include rather than over-include. Default False. Only True if you can identify a specific detail that is genuinely gratuitous in plot, symbolism, and characterization. Pre-1830 passages will almost always be False.
  - `story_time_span` and `distance_traveled`: measure the DOMINANT NARRATION of the passage. If frame dialogue/present-action fills most of the passage with only brief references to the past, measure the frame (short / stationary). If embedded retrospection dominates (a brief setup followed by a character's extended recounting, as in frame-tale, epistolary, or amatory fiction), measure the embedded content (which may be long / far-travelled). `has_retrospective_embedding=True` fires whenever substantial embedding is present, regardless.

notes: empty unless you have something specific (a surprising call, a phrase that drove the judgment, an ambiguity)."""


class PassageFormTask(Task):
    """Classify a fiction passage along narratological and interpretive axes.

    Designed for frontier models (Sonnet, Opus). Several fields require interpretive
    judgment that local models handle unreliably. Pair with PassageContentTask
    (run on cheaper local models for surface content).

    Usage:
        from largeliterarymodels.tasks import PassageFormTask, format_passage
        task = PassageFormTask()
        result = task.run(format_passage("passage...", title=..., author=..., year=...))
        # result is a PassageFormAnnotation instance
    """
    name = "classify_passage_form"
    schema = PassageFormAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = []  # Add 1-2 few-shot examples after smoke-test if judgment axes need anchoring.
    retries = 2
    temperature = 0.2
    max_tokens = 2048
