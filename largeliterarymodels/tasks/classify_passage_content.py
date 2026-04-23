"""Passage-level content annotation — surface/content features.

V2 (current, recommended): multi-label list fields + solo bools. 6 list fields
(scene_content, setting, character_classes, character_genders, fantastical_elements,
threats) + 3 solo bools (introduces_new_character, has_dialogue, has_child_character)
+ passage_summary/notes/confidence. Restructures V1's 39 separate bools to reduce
output tokens and sharpen local-model reliability.

V1 (PassageContentTaskV1) retained for V1-vs-V2 agreement audits and historical
reproducibility. V1 pilot (1110 passages, qwen3.5-35b-a3b) is logged in
llmtasks.passage_annotations with task_version=1.

Companion: PassageFormTask (classify_passage_form.py) — narratological and
interpretive features (narration mode, interiority, Genette duration/frequency,
Bakhtin heteroglossia, Lukács narrate/describe, Jameson récit/affect, Ch5 axes).
Run both per passage or either independently.
"""

from typing import Literal
from pydantic import BaseModel, Field
from largeliterarymodels.task import Task
from largeliterarymodels.tasks.classify_passage import format_passage  # re-exported


# ══════════════════════════════════════════════════════════════════════════
# V2 (current)
# ══════════════════════════════════════════════════════════════════════════

SceneContent = Literal[
    "battle_or_violence",
    "courtship",
    "travel_or_journey",
    "domestic_routine",
    "moral_reflection",
    "sexual_encounter",
    "religious_content",
    "deception_or_intrigue",
    "legal_or_financial",
    "illness_or_death",
    "shipwreck_or_storm",
    "abduction_or_captivity",
    "ball_or_social_gathering",
    "wedding_or_marriage_ceremony",
    "reading_or_writing_activity",
    "feast_or_meal",
]

SettingKind = Literal[
    "domestic_interior",
    "grand_estate",
    "wilderness",
    "urban_street",
    "confined_space",
    "rural_outdoor",
    "maritime",
]

CharacterClass = Literal[
    "noble_aristocratic",
    "gentry_or_middling",
    "laboring_or_servant",
    "clergy",
    "criminal_or_underclass",
    "hermit_or_recluse",
]

CharacterGender = Literal["male", "female"]

FantasticalElement = Literal[
    "supernatural",
    "ghost_or_haunting",
    "prophecy_or_omen",
    "monster_or_creature",
    "dream_or_vision",
    "allegorical_personification",
]

ThreatKind = Literal[
    "physical",
    "supernatural",
    "social_or_reputational",
    "psychological_or_emotional",
    "sexual_assault",
]


class PassageContentAnnotation(BaseModel):
    """V2 surface-content annotation for a fiction passage."""

    scene_content: list[SceneContent] = Field(
        default_factory=list,
        description=(
            "Zero or more scene types DEPICTED in the passage (not merely alluded to). "
            "battle_or_violence = depicted combat/duel/beating/execution/attack; exclude "
            "recounted or merely threatened violence (use threats=physical for those). "
            "courtship = romantic encounter, flirtation, declarations, marriage negotiation. "
            "travel_or_journey = characters in transit between places (roads, ships, carriages). "
            "domestic_routine = SUSTAINED household activity filling narrative time (meals as "
            "routine, servants at work, management of the home); exclude passing mentions. "
            "moral_reflection = meditation on virtue/duty/principle or generalization about conduct. "
            "sexual_encounter = erotic or amatory scene (distinct from courtship; also distinct "
            "from threatened assault — see threats=sexual_assault). "
            "religious_content = prayer, sermon, scripture, spiritual crisis, devotional reflection. "
            "deception_or_intrigue = plotting, disguise, eavesdropping, lying, secret schemes. "
            "legal_or_financial = trials, contracts, debts, inheritance, monetary transactions. "
            "illness_or_death = sickness, sickroom, deathbed, mourning, funeral. "
            "shipwreck_or_storm = maritime disaster, wreck, being lost at sea. "
            "abduction_or_captivity = kidnapping or ongoing state of being held against will. "
            "ball_or_social_gathering = balls, assemblies, masquerades, formal social events. "
            "wedding_or_marriage_ceremony = the wedding itself (distinct from courtship). "
            "reading_or_writing_activity = characters reading books/letters, composing, studying. "
            "feast_or_meal = ritualized eating as event (distinct from background domestic_routine)."
        ),
    )

    setting: list[SettingKind] = Field(
        default_factory=list,
        description=(
            "Zero or more settings present. Multiple allowed if passage spans them. "
            "domestic_interior = ordinary home (sitting room, bedchamber, kitchen, parlor). "
            "grand_estate = court, palace, castle, noble country house as site of power/society. "
            "wilderness = forest, mountains, desert, undomesticated nature. "
            "urban_street = city streets, squares, markets, urban crowds. "
            "confined_space = prison, dungeon, locked room, tomb — an entrapping space. "
            "rural_outdoor = gardens, parks, estate grounds, country roads (NOT wilderness). "
            "maritime = ship deck/cabin, port, naval setting, at sea."
        ),
    )

    character_classes: list[CharacterClass] = Field(
        default_factory=list,
        description=(
            "Social classes of characters present. "
            "noble_aristocratic = REQUIRES explicit title or clear indication of nobility "
            "(duke, earl, count, lord, lady, prince, royal). Do NOT mark for wealthy "
            "characters without title. "
            "gentry_or_middling = landed gentry, merchants, professionals, educated middle class. "
            "laboring_or_servant = servants, laborers, rural poor, working class. "
            "clergy = ministers, monks, nuns, priests, religious officials. "
            "criminal_or_underclass = thieves, highwaymen, pirates, beggars, prostitutes, vagrants. "
            "hermit_or_recluse = solitary religious or philosophical figure living apart."
        ),
    )

    character_genders: list[CharacterGender] = Field(
        default_factory=list,
        description="Genders of named or described characters appearing in the passage.",
    )

    fantastical_elements: list[FantasticalElement] = Field(
        default_factory=list,
        description=(
            "Non-naturalistic elements present within the fiction. "
            "supernatural = within-fiction supernatural AGENTS or events (magic, divine intervention "
            "acting in the plot, occult forces). EXCLUDE ordinary religion (prayer, sermon, "
            "scripture reading) — those are scene_content=religious_content. "
            "ghost_or_haunting = ghosts, spirits of the dead, haunted places. "
            "prophecy_or_omen = prophecy, oracle, portent foretelling future events. "
            "monster_or_creature = dragons, giants, demons, non-human creatures presented as real. "
            "dream_or_vision = dream, vision, or trance conveying narrative content. "
            "allegorical_personification = named abstractions (Virtue, Despair, Christian) as agents."
        ),
    )

    threats: list[ThreatKind] = Field(
        default_factory=list,
        description=(
            "Dangers facing characters, whether consummated or merely looming. "
            "physical = bodily harm, death, imprisonment, physical violation — depicted OR implied. "
            "supernatural = curse, demonic threat, ghostly peril, prophetic doom. "
            "social_or_reputational = loss of reputation/standing, scandal, rejection. "
            "psychological_or_emotional = threat to sanity, identity, emotional stability. "
            "sexual_assault = threatened or attempted sexual violence, coerced sexual encounter "
            "(mark even if not consummated in the passage)."
        ),
    )

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

    introduces_new_character: bool = Field(
        default=False,
        description="Passage introduces a character for the first time with name and/or "
        "first significant description.",
    )

    has_dialogue: bool = Field(
        default=False,
        description="Passage contains direct speech (quoted dialogue between characters).",
    )

    has_child_character: bool = Field(
        default=False,
        description="A child or minor (pre-adolescent or adolescent) is present as a character.",
    )

    passage_summary: str = Field(
        default="",
        description="One sentence, 25 words maximum. Who does what, where. "
        "Include character names if present.",
    )

    notes: str = Field(
        default="",
        description="Optional: one sentence on anything surprising, ambiguous, or driving a "
        "non-obvious classification. Empty if nothing notable.",
    )

    confidence: float = Field(
        default=0.5,
        description="Overall confidence 0.0 to 1.0. Lower for ambiguous or fragmentary passages.",
    )


SYSTEM_PROMPT = """You are annotating passages from English-language fiction (1500-2020) for a computational study of fiction subgenres (romance, domestic novel, gothic, picaresque, amatory, scandal chronicle, spiritual biography, etc.).

You will receive a single passage (~500-1500 words) with metadata (title, author, year). Classify it along 6 multi-label list fields (scene_content, setting, character_classes, character_genders, fantastical_elements, threats) plus 4 solo boolean fields (is_prose_fiction, introduces_new_character, has_dialogue, has_child_character). All capture SURFACE CONTENT — what happens, where, who, what threats, what fantastical elements.

The is_prose_fiction field is a meta-filter: most passages ARE narrative prose fiction, but texts tagged Fiction can contain embedded poetry, dramatic scenes, paratexts (indexes, prefaces), letters without narrative frame, or sermons. Mark is_prose_fiction=False for these; mark True for ordinary narrative prose.

## Classification principles

- **Multi-label**: each list field may contain zero, one, or many values. A passage can have `courtship` + `deception_or_intrigue` in scene_content AND `grand_estate` + `rural_outdoor` in setting simultaneously.
- **In-text only**: base classifications on the passage text, NOT on outside knowledge about the author or work. Do not infer features absent from the text.
- **Present means depicted or played out**, not merely mentioned in passing as backstory.
- **Empty list is a valid answer**: omit values entirely rather than fabricate evidence. Empty is better than wrong.
- **Ambiguity → lower confidence**: for fragmentary or unclear passages, lower the confidence score rather than making aggressive calls.

passage_summary: one sentence, 25 words max, naming characters where possible. notes: empty unless you have something specific to add (surprising inclusion, ambiguous call, driving phrase)."""


class PassageContentTask(Task):
    """V2: Classify a fiction passage along surface content features (multi-label).

    Designed for local-model reliability at bulk scale. All features are surface-
    detectable without recognition or world knowledge — locals like qwen/gemma
    perform well here.

    Usage:
        from largeliterarymodels.tasks import PassageContentTask, format_passage
        task = PassageContentTask()
        result = task.run(format_passage("passage...", title=..., author=..., year=...))
        # result is a PassageContentAnnotation instance
    """
    name = "classify_passage_content"
    schema = PassageContentAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = []  # Add 1-2 few-shot examples later if smoke-test quality requires.
    retries = 2
    temperature = 0.2
    max_tokens = 2048


# ══════════════════════════════════════════════════════════════════════════
# V1 (legacy) — preserved for V1-vs-V2 agreement audits
# ══════════════════════════════════════════════════════════════════════════

class PassageContentAnnotationV1(BaseModel):
    """V1 schema: 39 independent boolean features. Superseded by V2 list-based schema.

    Retained for historical reproducibility and V1-vs-V2 comparison. Data from the
    V1 pilot (1110 passages, qwen3.5-35b-a3b, 2026-04-20) is stored in
    llmtasks.passage_annotations under task_version=1.
    """

    # ── Scene content (what happens) ──────────────────────────────────────
    has_battle_or_violence: bool = Field(default=False, description="Combat, duels, beatings, executions, torture, physical attack.")
    has_courtship: bool = Field(default=False, description="Romantic encounter, flirtation, declarations of love, marriage negotiation.")
    has_travel_or_journey: bool = Field(default=False, description="Characters in transit between places.")
    has_domestic_routine: bool = Field(default=False, description="Meals, household management, daily domestic life, servants at work.")
    has_moral_reflection: bool = Field(default=False, description="Meditation on virtue, duty, principle, or generalization about conduct.")
    has_sexual_encounter: bool = Field(default=False, description="Erotic content, seduction, sexual violence, amatory scenes.")
    has_religious_content: bool = Field(default=False, description="Prayer, sermon, spiritual crisis, scripture, devotional reflection.")
    has_deception_or_intrigue: bool = Field(default=False, description="Plotting, disguise, eavesdropping, lying, secret schemes.")
    has_legal_or_financial: bool = Field(default=False, description="Trials, contracts, debts, inheritance, monetary transactions driving action.")

    # ── Setting (where) ───────────────────────────────────────────────────
    setting_domestic_interior: bool = Field(default=False, description="Ordinary home: sitting room, bedchamber, kitchen, parlor.")
    setting_grand_estate: bool = Field(default=False, description="Court, palace, castle, noble country house.")
    setting_wilderness: bool = Field(default=False, description="Forest, mountains, sea, desert, undomesticated nature.")
    setting_urban_street: bool = Field(default=False, description="City streets, squares, markets, urban crowds.")
    setting_confined_space: bool = Field(default=False, description="Prison, dungeon, locked room, tomb.")
    setting_rural_outdoor: bool = Field(default=False, description="Gardens, parks, estate grounds, country roads.")

    # ── Character demographics ────────────────────────────────────────────
    characters_noble_aristocratic: bool = Field(default=False, description="Nobility, royalty, or titled aristocracy.")
    characters_gentry_or_middling: bool = Field(default=False, description="Landed gentry, merchants, professionals, middle class.")
    characters_laboring_or_servant: bool = Field(default=False, description="Servants, laborers, rural poor, working class.")
    characters_clergy: bool = Field(default=False, description="Clergy, ministers, monks, nuns.")
    characters_criminal_or_underclass: bool = Field(default=False, description="Thieves, highwaymen, pirates, beggars, prostitutes.")
    has_male_characters: bool = Field(default=False, description="Named or described male characters appear.")
    has_female_characters: bool = Field(default=False, description="Named or described female characters appear.")
    introduces_new_character: bool = Field(default=False, description="Passage introduces a character for the first time.")

    # ── Fantastical / supernatural ────────────────────────────────────────
    has_supernatural: bool = Field(default=False, description="Divine, occult, miraculous, or non-naturalistic events or agents.")
    has_ghost_or_haunting: bool = Field(default=False, description="Ghosts, spirits of the dead, haunted places.")
    has_prophecy_or_omen: bool = Field(default=False, description="Prophecy, oracle, dream-omen, portent.")
    has_monster_or_creature: bool = Field(default=False, description="Dragons, giants, demons, monsters, non-human creatures.")
    has_dream_or_vision: bool = Field(default=False, description="Dream, vision, or trance conveying narrative content.")
    has_allegorical_personification: bool = Field(default=False, description="Named abstractions (Virtue, Despair) acting as agents.")

    # ── Threat / danger ───────────────────────────────────────────────────
    threat_physical: bool = Field(default=False, description="Bodily harm, death, imprisonment, physical violation.")
    threat_supernatural: bool = Field(default=False, description="Curse, demonic, ghostly, prophetic doom.")
    threat_social_or_reputational: bool = Field(default=False, description="Loss of reputation, standing, scandal, rejection.")

    # ── Emotional register ────────────────────────────────────────────────
    emotion_extreme: bool = Field(default=False, description="Terror, ecstasy, madness, despair, rage.")
    emotion_restrained_or_ironic: bool = Field(default=False, description="Decorous, detached, ironic, or satirical register.")

    # ── Interiority and narrative form ────────────────────────────────────
    has_character_thoughts: bool = Field(default=False, description="Access to a character's internal mental states.")
    has_free_indirect_or_monologue: bool = Field(default=False, description="Free indirect discourse or interior monologue.")
    is_dramatized_scene: bool = Field(default=False, description="Real-time action and dialogue.")
    is_narrative_summary: bool = Field(default=False, description="Compressed time, reported at a distance.")
    is_epistolary: bool = Field(default=False, description="Passage is written as a letter.")
    contains_embedded_letter: bool = Field(default=False, description="Passage quotes or transcribes a letter within a non-letter frame.")

    # ── Free-text ─────────────────────────────────────────────────────────
    passage_summary: str = Field(default="", description="One sentence, 25 words maximum.")
    notes: str = Field(default="", description="Optional: surprising or ambiguous observations.")
    confidence: float = Field(default=0.5, description="Overall confidence 0.0 to 1.0.")


SYSTEM_PROMPT_V1 = """You are annotating passages from English-language fiction (1500-2020) for a computational study of fiction subgenres (romance, domestic novel, gothic, picaresque, spiritual biography, amatory, scandal chronicle, etc.).

You will receive a single passage of approximately 500-1500 words with metadata (title, author, year). Classify the passage along 39 independent True/False features covering scene content, setting, character demographics, fantastical elements, threat, emotional register, and narrative form.

## Classification principles

- Each feature is INDEPENDENT. A passage can be True on many features simultaneously (e.g. has_courtship + has_deception_or_intrigue + setting_grand_estate + characters_noble_aristocratic).
- Base classifications on what is IN THE PASSAGE, not on what you know about the author or work. Do not guess at features absent from the text.
- Prefer True when a feature is clearly present, even if brief. Prefer False when the passage lacks clear evidence.
- "Present" means: named, depicted, or played out in the passage. Not: merely mentioned in passing as backstory.
- When character class is unclear, mark the most specific class for which you have textual evidence; leave others False.
- For ambiguous or fragmentary passages, lower the confidence score rather than making aggressive True calls.

The passage_summary must be one sentence, 25 words or fewer, naming characters where possible. The notes field should be empty unless you have something specific to add (a surprising True, a phrase that drove a judgment call)."""


class PassageContentTaskV1(Task):
    """V1 (legacy): 39-bool passage-content classification. Superseded by PassageContentTask (V2).

    Retained for V1-vs-V2 agreement audits and reproducibility of the 2026-04-20 pilot.
    """
    name = "classify_passage_content"  # same name as V2; task_version=1 distinguishes in CH
    schema = PassageContentAnnotationV1
    system_prompt = SYSTEM_PROMPT_V1
    examples = []
    retries = 2
    temperature = 0.2
    max_tokens = 4096
