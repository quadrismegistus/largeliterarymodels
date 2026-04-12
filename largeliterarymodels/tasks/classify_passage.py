"""Passage-level narratological annotation: classify ~1K-word passages from fiction
by scene type, narration mode, setting, social epistemology, allegorical regime,
and abstract/concrete semantic relations.

Designed for the "Abstraction: A Literary History" project. Annotates the formal
and narratological features that Ch 5 ("Abstract Realism") argues drive the
historical arc of concrete → abstract → concrete language across four centuries
of fiction (1600–2000). Each passage receives a single structured classification
covering multiple independent axes.

Unlike Frye classification (which reads opening/middle/closing of a whole text),
this task classifies individual passages — typically 500–1500 words sampled from
within a text, often stratified by concreteness score.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from largeliterarymodels.task import Task


# ── Vocabularies ──────────────────────────────────────────────────────────

SCENE_TYPE_VOCAB = [
    'battle_combat',        # duels, sieges, physical confrontation between combatants
    'courtship',            # romantic encounter, flirtation, declarations of love
    'travel_movement',      # journeys, arrivals, departures, navigation of roads/seas
    'domestic_routine',     # meals, household management, daily domestic life
    'moral_reflection',     # narratorial or character meditation on virtue, duty, principle
    'conversation',         # extended dialogue or social exchange (not reducible to another type)
    'description_persons',  # introduction or description of a character's appearance/character
    'description_place',    # description of setting, landscape, architecture, rooms
    'sexual_encounter',     # erotic or amatory scenes, seduction, sexual violence
    'legal_financial',      # trials, contracts, debts, inheritance, property disputes
    'religious_devotional', # prayer, sermons, spiritual crisis, religious instruction
    'letter_writing',       # composition or reading of letters within the narrative
    'festivity_social',     # balls, dinners, assemblies, masquerades, public gatherings
    'violence_punishment',  # non-combat violence: beatings, executions, torture, imprisonment
    'deception_intrigue',   # plots, disguises, eavesdropping, secret plans
    'education_instruction',# lessons, didactic address, mentor/pupil scenes
    'other',
]

NARRATION_MODE_VOCAB = [
    'dramatized_scene',     # real-time action and dialogue, "showing"
    'narrative_summary',    # compressed time, "telling", narrator reports events at a distance
    'free_indirect',        # character's thoughts in narrator's voice, blended perspective
    'interior_monologue',   # direct access to character's thoughts (first-person stream)
    'authorial_commentary', # narrator steps outside story to address reader or generalize
    'reported_speech',      # dialogue or speech rendered indirectly ("she told him that...")
    'epistolary',           # letter-form narration within the passage
    'mixed',                # passage mixes modes without a clear dominant
]

SETTING_VOCAB = [
    'domestic_interior',    # homes, sitting rooms, bedchambers, kitchens
    'public_interior',      # inns, taverns, theatres, churches, courtrooms, shops
    'urban_street',         # city streets, squares, markets, crowds
    'rural_road',           # country roads, paths between towns, coaching routes
    'garden_grounds',       # gardens, parks, estate grounds, pastoral outdoors
    'wilderness',           # forests, mountains, seas, undomesticated nature
    'battlefield',          # sites of combat, military camps
    'court_palace',         # royal or aristocratic court, grand houses as power centers
    'ship_voyage',          # at sea, aboard ships
    'abstract_unspecified', # no physical setting specified; action in a purely social/mental space
    'other',
]

SOCIAL_EPISTEMOLOGY_VOCAB = [
    'reputation',           # characters known through shared social judgment, rumor, opinion
    'physical_inspection',  # characters known through concrete bodily/material appearance
    'intimate_knowledge',   # characters known through personal history, direct experience
    'self_report',          # character narrates or reveals themselves (letters, confession)
    'allegorical_emblem',   # character known through symbolic/emblematic attributes
    'anonymous_encounter',  # characters encounter each other as strangers, no prior knowledge
    'not_applicable',       # passage does not involve interpersonal knowing
]

ALLEGORICAL_REGIME_VOCAB = [
    'allegory_of_symbol',       # concrete emblems encode abstract ideas top-down (rock of virtue, flower of fidelity, lion = kingship). Thing-to-idea by convention or sacred analogy.
    'social_grounding',         # abstractions sustained by shared social judgment, not emblems. Virtue, reputation, taste are real because collectively acknowledged. The "knowable community."
    'allegory_of_commodity',    # concrete appearances decoded bottom-up for social meaning (nostrils denoting choler, dirty rooms = irresponsibility). Thing-to-idea by physiognomic/material reading.
    'nominalization',           # abstract nouns reify action into public property ("those Insinuations," "her Confusion"). Verbs become nouns; private experience becomes impersonal discourse.
    'mixed',                    # passage combines multiple regimes
    'none',                     # no significant abstract/concrete mediation visible
]

ABS_CONC_TENDENCY_VOCAB = [
    'predominantly_abstract',   # passage dominated by abstract nouns, social/moral/psychological vocabulary
    'predominantly_concrete',   # passage dominated by concrete nouns, physical/sensory vocabulary
    'balanced',                 # roughly equal presence of both registers
    'abstract_punctuated',      # abstract dominant, but concrete words appear at key moments (Austen's storms/sprains)
    'concrete_punctuated',      # concrete dominant, but abstract words appear at key moments (Cusk's "anti-description")
]


# ── Schema ────────────────────────────────────────────────────────────────

class PassageAnnotation(BaseModel):
    scene_type: str = Field(
        description="Primary scene type / what the passage is about. "
        f"One of: {', '.join(SCENE_TYPE_VOCAB)}. "
        "Choose the single best label. If a courtship scene includes moral reflection, "
        "choose 'courtship' if the romantic interaction drives the passage, or "
        "'moral_reflection' if the meditation dominates."
    )
    scene_type_secondary: str = Field(
        default="",
        description="Optional secondary scene type, if the passage clearly blends two. "
        f"One of: {', '.join(SCENE_TYPE_VOCAB)}, or empty string if not applicable."
    )
    narration_mode: str = Field(
        description="Dominant narration mode in this passage. "
        f"One of: {', '.join(NARRATION_MODE_VOCAB)}. "
        "'dramatized_scene' = real-time showing with dialogue/action. "
        "'narrative_summary' = compressed telling. "
        "'free_indirect' = character thought in narrator's voice. "
        "'authorial_commentary' = narrator generalizes or addresses reader."
    )
    narration_mode_secondary: str = Field(
        default="",
        description="Secondary narration mode if the passage mixes modes. "
        f"One of: {', '.join(NARRATION_MODE_VOCAB)}, or empty string."
    )
    setting: str = Field(
        description="Physical setting of the passage. "
        f"One of: {', '.join(SETTING_VOCAB)}. "
        "'abstract_unspecified' if the passage operates in a purely social or mental space "
        "with no physical setting described."
    )
    character_introduction: bool = Field(
        default=False,
        description="True if the passage introduces a character for the first time "
        "(or presents their first significant description). False otherwise."
    )
    character_intro_method: str = Field(
        default="",
        description="If character_introduction is True: how is the character introduced? "
        "'physical_concrete' = through bodily appearance, clothes, posture, physiognomy. "
        "'abstract_social' = through abstract qualities (youth, beauty, elegance, reputation). "
        "'behavioral' = through dramatized actions and speech. "
        "'mixed' = combination. "
        "Empty string if character_introduction is False."
    )
    social_epistemology: str = Field(
        description="How do characters know or assess each other in this passage? "
        f"One of: {', '.join(SOCIAL_EPISTEMOLOGY_VOCAB)}. "
        "'reputation' = shared social judgment (the knowable community). "
        "'physical_inspection' = reading bodies and commodities for meaning. "
        "'intimate_knowledge' = personal history and direct experience. "
        "'allegorical_emblem' = symbolic attributes (flower of virtue, battle as proof of honor). "
        "'anonymous_encounter' = strangers meeting without prior knowledge."
    )
    allegorical_regime: str = Field(
        description="How do abstract and concrete language relate in this passage? "
        f"One of: {', '.join(ALLEGORICAL_REGIME_VOCAB)}. "
        "'allegory_of_symbol' = concrete objects encode abstract ideas by convention "
        "(Bunyan's Slough of Despondency, heraldic emblems, flowers of fidelity). "
        "'social_grounding' = abstractions are real because collectively acknowledged, "
        "not because grounded in things (Austen's knowable community). "
        "'allegory_of_commodity' = concrete appearances are decoded for social meaning "
        "(Dickens' physiognomy, Bronte's 'nostrils denoting choler'). "
        "'nominalization' = verbs become abstract nouns, private action becomes "
        "impersonal public property (Haywood's 'those Insinuations'). "
        "'none' = no significant abstract/concrete mediation."
    )
    abs_conc_tendency: str = Field(
        description="Overall abstract/concrete balance of the passage's diction. "
        f"One of: {', '.join(ABS_CONC_TENDENCY_VOCAB)}. "
        "Consider the density of abstract nouns (virtue, duty, elegance, confusion) "
        "vs. concrete nouns (teeth, rain, door, sword, dress). "
        "'abstract_punctuated' = mostly abstract with notable concrete moments. "
        "'concrete_punctuated' = mostly concrete with notable abstract moments."
    )
    key_abstractions: list[str] = Field(
        default=[],
        description="Up to 5 notable abstract nouns or nominalizations in the passage. "
        "Words like: virtue, reputation, elegance, confusion, insinuation, propriety, "
        "obligation, sentiment, extravagance. Choose words that are characteristic "
        "of the passage's semantic register, not just any abstract word."
    )
    key_concretions: list[str] = Field(
        default=[],
        description="Up to 5 notable concrete nouns in the passage. "
        "Words like: sword, door, rain, teeth, dress, carriage, mud, blood. "
        "Choose words that are characteristic of the passage's physical world."
    )
    passage_summary: str = Field(
        description="1-2 sentence summary of what happens in this passage. "
        "Focus on the action, situation, or argument — who does what, where, why. "
        "Be specific enough to locate the passage in the plot (e.g. 'Pamela hides "
        "in the closet as Mr. B attempts to assault her; Mrs. Jervis intervenes' "
        "not 'a tense scene between characters'). Include character names."
    )
    confidence: float = Field(
        description="Overall confidence in the annotation, 0.0 to 1.0. "
        "Lower if the passage is ambiguous, fragmentary, or difficult to classify."
    )
    reasoning: str = Field(
        description="1-3 sentences explaining the most important or non-obvious "
        "classification decisions. Focus on what makes this passage distinctive "
        "rather than restating the labels. Quote a key phrase if relevant."
    )


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a literary critic annotating passages from English-language fiction (1500–2020) for a computational study of abstract and concrete language in the history of the novel.

You will receive a single passage of approximately 500–1500 words, along with metadata (title, author, year) when available. Classify the passage along multiple independent axes.

## Theoretical framework

This annotation supports research into what Ryan Heuser calls "abstract realism": the paradox that the eighteenth-century novel — the genre critics associate with "concrete particularity" (Watt) and the "reality effect" (Barthes) — is in fact linguistically more abstract than the romances and picaresques that preceded it. Concrete language falls across the C17-C18, then rises dramatically in the C19.

The key question is WHY. The annotations you provide will help decompose this historical arc into its formal, narratological, and social components:

### Scene type
What is the passage about? Battle scenes, travel, and picaresque survival are inherently concrete. Moral reflection and social assessment are inherently abstract. The disappearance of certain scene types (combat, inter-social spaces) from the C18 novel partly explains its abstract turn. We need to test this computationally.

### Narration mode
"Show, don't tell" (Lubbock/James) is the modernist rewriting of formal realism. But in C18 fiction, "telling" is not failed "showing" — it is a distinct realist mode that uses abstract language to summarize, generalize, and assess. Narrative summary and authorial commentary use more abstract language than dramatized scenes. We need to measure whether the abstract turn is a shift in narration mode.

### Setting
The novel "goes indoors" (Armstrong): the romance's roads, battlefields, and enchanted forests are replaced by domestic interiors and recognizable social spaces. Later, C19 urbanization brings back "inter-social spaces" (streets, crowds) that require concrete language to navigate. Setting type should correlate with concreteness.

### Social epistemology
How characters know each other reflects the social formation:
- In romance: through allegorical emblems and deeds of arms (concrete proves abstract)
- In C18 novel: through reputation and shared social judgment (abstract floats free — the "knowable community" per Raymond Williams)
- In C19 novel: through physical inspection of bodies, clothes, commodities (concrete decoded for abstract — reification per Lukács)

### Allegorical regime
The core theoretical distinction in Ch 5:
- **Allegory of the symbol** (C16-C17 romance): Bunyan's "Slough of Despondency," Sidney's "flower of magnanimity." Concrete objects encode abstract meaning by sacred or conventional analogy. Top-down: the abstract idea pre-exists and is deposited into a thing.
- **Social grounding** (C18 novel): Austen's "youth, beauty, elegance." Abstractions are real because collectively acknowledged — grounded in shared social judgment, not in things. The "knowable community" underwrites abstraction.
- **Allegory of the commodity** (C19 novel): Dickens' "nostrils denoting choler," Brontë's "decisive nose." Concrete appearances are decoded bottom-up for social meaning. Things speak — but they lie (Marx). Reification.
- **Nominalization** (pervasive in C18): Haywood's "those Insinuations" — verbs reified into abstract nouns, private action transformed into impersonal public property. A distinctive mechanism of abstract realism.

### Abstract/concrete tendency
Beyond computational word scores, your qualitative assessment of the passage's dominant semantic register — and especially whether abstract and concrete language punctuate each other at key moments (Austen's storms arriving to destabilize abstract social analysis; Cusk's concrete description reluctantly interrupting preferred abstraction).

## Classification principles

- Each axis is INDEPENDENT. A passage can be a dramatized scene (narration) set in a domestic interior (setting) where characters know each other by reputation (epistemology) and the narrator nominalizes their actions (allegorical regime).
- Base classifications on the TEXT, not on what you know about the author. Austen passages can be concrete; Dickens passages can be abstract.
- When in doubt between two labels, choose the one that better captures what is DISTINCTIVE about this passage — what makes it different from a generic passage of fiction.
- The key_abstractions and key_concretions fields should capture the passage's characteristic vocabulary, not just any abstract or concrete word. "Virtue" in an Austen passage is more characteristic than "time."
"""


# ── Few-shot examples ─────────────────────────────────────────────────────

EXAMPLES = [
    # 1. Austen — abstract character introduction via social judgment
    (
        "Title: Sense and Sensibility\nAuthor: Austen\nYear: 1811\n\n"
        "PASSAGE:\n"
        "His person and air were equal to what her fancy had ever drawn for the hero "
        "of a favourite story; and in his carrying her into the house with so little "
        "previous formality, there was a rapidity of thought which particularly "
        "recommended him to her. Every circumstance belonging to him was interesting. "
        "His name was good, his residence was in their favourite village, and she soon "
        "found out that of all manly dresses a shooting-jacket was the most becoming. "
        "His manners had all the elegance which his own could not have wanted. His "
        "understanding was beyond all doubt acute, his taste delicate. Elinor saw "
        "nothing to censure in him. She was only disturbed by a tendency to say too "
        "much of what he thought on every occasion, without attention to persons or "
        "circumstances. In hastily forming and giving his opinion of other people, in "
        "sacrificing general politeness to the enjoyment of undivided attention where "
        "his heart was engaged, and in slighting too easily the forms of worldly "
        "propriety, he displayed a want of caution which Elinor could not approve.",
        PassageAnnotation(
            scene_type="description_persons",
            scene_type_secondary="moral_reflection",
            narration_mode="narrative_summary",
            narration_mode_secondary="free_indirect",
            setting="domestic_interior",
            character_introduction=True,
            character_intro_method="abstract_social",
            social_epistemology="reputation",
            allegorical_regime="social_grounding",
            abs_conc_tendency="predominantly_abstract",
            key_abstractions=["elegance", "taste", "propriety", "caution", "politeness"],
            key_concretions=["shooting-jacket", "house"],
            passage_summary="Willoughby is introduced after carrying Marianne into the house; "
            "Marianne is immediately charmed by his manner while Elinor judges his character "
            "more cautiously, noting his want of propriety.",
            confidence=0.95,
            reasoning="Willoughby is introduced almost entirely through abstract social "
            "qualities — 'elegance,' 'understanding,' 'taste' — with the 'shooting-jacket' "
            "as the single concrete detail, immediately absorbed into abstract assessment "
            "('the most becoming'). Elinor's judgment operates through the shared social "
            "vocabulary of the knowable community: 'propriety,' 'caution,' 'politeness.'",
        ),
    ),
    # 2. Lazarillo de Tormes — picaresque concreteness, inter-social space
    (
        "Title: Lazarillo de Tormes\nAuthor: Anonymous\nYear: 1554\n\n"
        "PASSAGE:\n"
        "In this affliction I found myself one day, when my crafty master had gone "
        "out of the house (I say crafty, for there were in him so many signs of cunning "
        "and deceit, though he also used it for a good purpose, as shall be told). "
        "Being alone, I began to reason thus with myself: 'This chest is very old, "
        "large, and broken in places, though with small holes. There may be mice about, "
        "and they may gnaw through to the bread within.' I brought the bread out, and "
        "finding it untouched, said, 'This trunk is not mouse-proof.' I began to pick "
        "at the bread and crumble the crust, scattering the crumbs on some cheap "
        "cloths that were in the chest. Then I took one loaf and left the other, and "
        "from that I took some little as a beggar might, and so ate it. When my master "
        "came home and opened the chest, he saw the damage and no doubt believed it "
        "was caused by mice, for I had so perfectly counterfeited their work.",
        PassageAnnotation(
            scene_type="deception_intrigue",
            scene_type_secondary="domestic_routine",
            narration_mode="dramatized_scene",
            narration_mode_secondary="interior_monologue",
            setting="domestic_interior",
            character_introduction=False,
            character_intro_method="",
            social_epistemology="intimate_knowledge",
            allegorical_regime="none",
            abs_conc_tendency="predominantly_concrete",
            key_abstractions=["cunning", "deceit"],
            key_concretions=["chest", "bread", "mice", "crumbs", "cloths"],
            passage_summary="Lazaro, left alone and starving, devises a scheme to steal bread "
            "from his master's locked chest by picking at the loaf and scattering crumbs to "
            "simulate mouse damage.",
            confidence=0.9,
            reasoning="Classic picaresque concreteness: Lazaro's hunger drives a minute, "
            "step-by-step account of physical objects — chest, bread, crumbs, cloths — "
            "in a scheme for survival. The two abstract words ('cunning,' 'deceit') are "
            "attributed to the master in passing; everything else is thingly, tactile, "
            "immediate.",
        ),
    ),
    # 3. Dickens — allegory of the commodity, physiognomic reading
    (
        "Title: Bleak House\nAuthor: Dickens\nYear: 1853\n\n"
        "PASSAGE:\n"
        "He was a small man with a head sunk sideways between his shoulders, and "
        "with such a quantity of what he called his 'property' heaped around him that "
        "he seemed a mere bundle of old rags, bones, and odds-and-ends. His shop was "
        "blinded by cobwebs, heaped with bottles, bones, scraps of old leather, "
        "horsehair, rags, and waste. 'Everything seemed to be bought, and nothing to "
        "be sold there.' A cat, with a tail like a worn-out hearth-broom, "
        "sat upon a shelf over the doorway. Under the shelf were several bundles of "
        "old papers, among them yellow parchments with red and black ink upon them. "
        "The old man's breath issued from him like smoke from a damp chimney. He was "
        "short, cadaverous, and withered; with his head sunk sideways between his "
        "shoulders, and the breath issuing in visible smoke from his mouth, as if he "
        "were on fire within. His throat, chin, and eyebrows were so frosted with "
        "white hairs, and so gnarled with veins and puckered skin, that he looked "
        "from his breast upward, like some old root in a fall of snow.",
        PassageAnnotation(
            scene_type="description_persons",
            scene_type_secondary="description_place",
            narration_mode="dramatized_scene",
            narration_mode_secondary="authorial_commentary",
            setting="public_interior",
            character_introduction=True,
            character_intro_method="physical_concrete",
            social_epistemology="physical_inspection",
            allegorical_regime="allegory_of_commodity",
            abs_conc_tendency="predominantly_concrete",
            key_abstractions=["property"],
            key_concretions=["bones", "rags", "cobwebs", "bottles", "parchments"],
            passage_summary="Krook the rag-and-bone dealer is introduced through a detailed "
            "description of his body and shop, both composed of the same decaying materials — "
            "rags, bones, cobwebs, old papers.",
            confidence=0.95,
            reasoning="Krook is introduced through pure materiality — his body and shop "
            "are indistinguishable, both heaps of 'bones, rags, odds-and-ends.' The passage "
            "decodes his moral character from physical appearance: 'cadaverous and withered,' "
            "'like some old root.' His one abstraction, 'property,' is ironic — he calls "
            "his junk 'property,' encoding the commodity-fetish directly.",
        ),
    ),
    # 4. Haywood — nominalization, abstract realism, sexual circumlocution
    (
        "Title: The Masqueraders\nAuthor: Haywood\nYear: 1725\n\n"
        "PASSAGE:\n"
        "Dorimenus, who was a Person of a great deal of Wit and Address, took care "
        "to improve those Moments which Fortune so favourably had thrown in his Way; "
        "he press'd her with all those soft undoing Insinuations, those melting "
        "Tendernesses, those seducing Arts which Desire never fails to instruct, and "
        "which are still most prevailing in the Mouths of those who are most "
        "passionately affected. In fine, he drew from her in Conversation so much of "
        "her Affairs, as to know she was a Widow, young, and to Appearance, not "
        "disinclined to receive the Addresses of a Lover: two Chairs were immediately "
        "call'd, and they went together to her Lodgings, which were handsome and "
        "well-furnish'd. The God of tender Inspirations with pleasure beheld the "
        "Sacrifice was made him, and blest the amorous Pair with doubled Vigour, and "
        "uncommon Rapture.",
        PassageAnnotation(
            scene_type="sexual_encounter",
            scene_type_secondary="courtship",
            narration_mode="narrative_summary",
            narration_mode_secondary="authorial_commentary",
            setting="domestic_interior",
            character_introduction=False,
            character_intro_method="",
            social_epistemology="anonymous_encounter",
            allegorical_regime="nominalization",
            abs_conc_tendency="predominantly_abstract",
            key_abstractions=["Insinuations", "Tendernesses", "Desire", "Vigour", "Rapture"],
            key_concretions=["Chairs", "Lodgings"],
            passage_summary="Dorimenus seduces a young widow he has just met at a masquerade; "
            "the encounter and sexual consummation are narrated entirely through abstract "
            "nominalizations and mythological circumlocution.",
            confidence=0.9,
            reasoning="A paradigm of abstract realism via nominalization. Dorimenus' seduction "
            "is never shown — only 'those Insinuations,' 'those melting Tendernesses,' "
            "reifying his private actions into impersonal public property ('those'). The sexual "
            "encounter itself is circumlocuted through mythological abstraction ('the God of "
            "tender Inspirations'). The two concrete details — 'Chairs' and 'Lodgings' — "
            "dispatch physical movement in a single sentence.",
        ),
    ),
    # 5. Bunyan — allegory of the symbol
    (
        "Title: The Pilgrim's Progress\nAuthor: Bunyan\nYear: 1678\n\n"
        "PASSAGE:\n"
        "Now I saw in my Dream, that just as they had ended this talk, they drew "
        "near to a very miry Slough that was in the midst of the Plain, and they "
        "being heedless, did both fall suddenly into the bog. The name of the Slough "
        "was Despond. Here therefore they wallowed for a time, being grievously "
        "bedaubed with the dirt. And Christian, because of the burden that was on his "
        "back, began to sink in the mire. Then said Pliable, Ah, Neighbour Christian, "
        "where are you now? Truly, said Christian, I do not know. At which Pliable "
        "began to be offended; and angrily said to his Fellow, Is this the happiness "
        "you have told me all this while of? If we have such ill speed at our first "
        "setting out, what may we expect betwixt this and our Journey's end? May I "
        "get out again with my life, you shall possess the brave Country alone for me. "
        "And with that he gave a desperate struggle or two, and got out of the Mire, "
        "on that side of the Slough which was next to his own House: so away he went, "
        "and Christian saw him no more.",
        PassageAnnotation(
            scene_type="travel_movement",
            scene_type_secondary="moral_reflection",
            narration_mode="dramatized_scene",
            narration_mode_secondary="",
            setting="wilderness",
            character_introduction=False,
            character_intro_method="",
            social_epistemology="allegorical_emblem",
            allegorical_regime="allegory_of_symbol",
            abs_conc_tendency="concrete_punctuated",
            key_abstractions=["Despond", "happiness", "burden"],
            key_concretions=["Slough", "mire", "dirt", "bog"],
            passage_summary="Christian and Pliable fall into the Slough of Despond; Pliable, "
            "disgusted by the difficulty, struggles out and abandons Christian, who sinks "
            "under his burden.",
            confidence=0.95,
            reasoning="The Slough of Despond is the paradigmatic allegory of the symbol: "
            "a concrete place (miry bog, dirt, sinking) that directly encodes an abstract "
            "idea (despair) by name. Characters are likewise emblematic — 'Pliable' is his "
            "nature. The passage is predominantly concrete in texture (wallowing, sinking, "
            "struggling) but every concrete detail exists to embody its abstract meaning.",
        ),
    ),
]


# ── Task class ────────────────────────────────────────────────────────────

class PassageTask(Task):
    """Classify a ~1K-word fiction passage along narratological and semantic axes.

    Designed for bulk annotation of passages sampled from fiction corpora (1500–2020).
    Each passage receives a single structured classification covering scene type,
    narration mode, setting, social epistemology, allegorical regime, and
    abstract/concrete tendency.

    Usage:
        task = PassageTask()
        result = task.run(format_passage("passage text...", title="Pamela", author="Richardson", year=1740))
        # result is a PassageAnnotation instance

        # Bulk:
        prompts = [format_passage(p, **meta) for p, meta in passages]
        results = task.map(prompts, num_workers=4)
    """
    name = "classify_passage"
    schema = PassageAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.2
    max_tokens = 8192


# ── Helper functions ──────────────────────────────────────────────────────

# Chapters shorter than this are merged with the next; longer than this are
# split into windows. These thresholds keep prompt size consistent for the LLM.
MIN_CHAPTER_WORDS = 200
MAX_CHAPTER_WORDS = 3000


def format_passage(passage_text, title=None, author=None, year=None,
                   passage_index=None, chapter_title=None,
                   concreteness_score=None, _id=None, section_id=None):
    """Format a passage and its metadata into a prompt string + metadata dict.

    Args:
        passage_text: The passage text (~500–3000 words).
        title: Title of the source text.
        author: Author name.
        year: Publication year.
        passage_index: Optional index of the passage within the text (e.g. "3/12").
        chapter_title: Optional chapter/section heading from the source text.
        concreteness_score: Optional pre-computed concreteness score (z-score).
        _id: LLTK text _id (e.g. "_chadwyck/richards.04"). Stored in metadata.
        section_id: Section/chapter ID within the text (e.g. "S0005"). Stored in metadata.

    Returns:
        tuple[str, dict]: (prompt_string, metadata_dict). The metadata dict
        contains _id, section_id, passage_index, chapter_title, year, and
        n_words — pass it to task.run(prompt, metadata=meta) so it appears
        in the cache and in task.df as meta_* columns.
    """
    parts = []

    # Metadata header (visible to LLM)
    header = []
    if title:
        header.append(f"Title: {title}")
    if author:
        name = author.split(',')[0].strip() if ',' in author else author
        header.append(f"Author: {name}")
    if year:
        header.append(f"Year: {year}")
    if chapter_title:
        header.append(f"Chapter: {chapter_title}")
    if passage_index is not None:
        header.append(f"Passage: {passage_index}")
    if concreteness_score is not None:
        header.append(f"Concreteness score (z): {concreteness_score:.3f}")
    if header:
        parts.append('\n'.join(header))

    parts.append(f"PASSAGE:\n{passage_text}")

    prompt = '\n\n'.join(parts)

    # Metadata dict (stored in cache, appears as meta_* columns in task.df)
    # Cast year to plain int — pandas Int64 dtype is not JSON-serializable.
    meta = {
        '_id': _id or '',
        'section_id': section_id or '',
        'passage_index': passage_index or '',
        'chapter_title': chapter_title or '',
        'year': int(year) if year is not None else None,
        'n_words': len(passage_text.split()),
    }

    return prompt, meta


def format_chapters(text_obj, max_chapters=None, min_words=MIN_CHAPTER_WORDS,
                    max_words=MAX_CHAPTER_WORDS):
    """Generate prompts from a text's chapter/section markup (LLTK text objects).

    Preferred method for chadwyck and other XML-marked corpora. Falls back to
    windowed passages if the text has no chapters or too few meaningful sections.

    Chapters shorter than min_words are merged with the next chapter.
    Chapters longer than max_words are split into ~max_words windows.

    Args:
        text_obj: An lltk text object with .chapters access.
        max_chapters: Maximum number of chapters to return (None = all).
        min_words: Minimum words per chapter; shorter ones are merged forward.
        max_words: Maximum words per chapter; longer ones are split into windows.

    Returns:
        list[tuple[str, dict]]: List of (prompt, metadata) tuples.
        Returns empty list if the text has no usable chapter structure
        (caller should fall back to format_passages_from_text).
    """
    title = getattr(text_obj, 'title', '') or ''
    author = getattr(text_obj, 'author', '') or ''
    year = getattr(text_obj, 'year', None)
    _id = getattr(text_obj, '_id', '') or ''
    if not _id:
        corpus_id = getattr(getattr(text_obj, '_corpus', None), 'id', '') or ''
        text_id = getattr(text_obj, 'id', '') or ''
        if corpus_id and text_id:
            _id = f"_{corpus_id}/{text_id}"

    # Get chapters from LLTK
    try:
        sections = list(text_obj.chapters.texts())
    except Exception:
        return []

    if len(sections) < 3:
        # Too few sections (e.g. Man in the Moone: title page + preface + body)
        # — fall back to windowed
        return []

    # Extract (section_id, chapter_title, text) tuples, skipping paratextual
    raw_chapters = []
    for s in sections:
        txt = (s.txt or '').strip()
        if not txt:
            continue
        ch_title = getattr(s, 'title', '') or s.id
        section_id = getattr(s, 'id', '')
        # Skip obvious paratextual sections
        ch_lower = ch_title.lower()
        if any(skip in ch_lower for skip in
               ['title page', 'table of contents', 'contents', 'errata',
                'advertisement', 'subscribers', 'index']):
            continue
        raw_chapters.append((section_id, ch_title, txt))

    if len(raw_chapters) < 3:
        return []

    # Merge short chapters forward
    merged = []
    buf_sid, buf_title, buf_text = raw_chapters[0]
    for section_id, ch_title, ch_text in raw_chapters[1:]:
        buf_words = len(buf_text.split())
        if buf_words < min_words:
            # Merge: combine title and text, keep first section_id
            buf_title = f"{buf_title} / {ch_title}"
            buf_text = buf_text + '\n\n' + ch_text
        else:
            merged.append((buf_sid, buf_title, buf_text))
            buf_sid, buf_title, buf_text = section_id, ch_title, ch_text
    merged.append((buf_sid, buf_title, buf_text))  # flush last

    # Split long chapters into windows
    final = []
    for section_id, ch_title, ch_text in merged:
        words = ch_text.split()
        if len(words) <= max_words:
            final.append((section_id, ch_title, ch_text))
        else:
            # Split into ~max_words windows
            n_windows = (len(words) + max_words - 1) // max_words
            window_size = len(words) // n_windows
            for w in range(n_windows):
                start = w * window_size
                end = start + window_size if w < n_windows - 1 else len(words)
                part_label = f"{ch_title} (part {w+1}/{n_windows})"
                part_sid = f"{section_id}.{w+1}" if n_windows > 1 else section_id
                final.append((part_sid, part_label, ' '.join(words[start:end])))

    # Optionally limit
    if max_chapters and len(final) > max_chapters:
        # Evenly sample, always including first and last
        if max_chapters <= 2:
            indices = [0, len(final) - 1][:max_chapters]
        else:
            indices = [0]
            inner = max_chapters - 2
            for i in range(inner):
                idx = int((i + 1) * (len(final) - 1) / (inner + 1))
                indices.append(idx)
            indices.append(len(final) - 1)
        final = [final[i] for i in indices]

    # Format prompts with metadata
    total = len(final)
    results = []
    for i, (section_id, ch_title, ch_text) in enumerate(final):
        results.append(format_passage(
            ch_text, title=title, author=author, year=year,
            passage_index=f"{i+1}/{total}", chapter_title=ch_title,
            _id=_id, section_id=section_id,
        ))
    return results


def format_passages_from_text(text_obj=None, txt=None, title=None, author=None,
                              year=None, n_words=1000, n_passages=5,
                              strategy='stratified', use_chapters=True,
                              max_chapter_words=MAX_CHAPTER_WORDS, _id=None):
    """Generate passage prompts from a text, preferring chapters when available.

    If use_chapters=True and the text object has chapter markup (e.g. chadwyck
    XML), returns one prompt per chapter (merged/split to reasonable sizes).
    Otherwise falls back to n-word windows sampled from the raw text.

    Args:
        text_obj: An lltk text object.
        txt: Raw text string (alternative to text_obj).
        title: Title string.
        author: Author string.
        year: Publication year.
        n_words: Words per passage for windowed fallback (~1000).
        n_passages: Number of passages for windowed fallback.
        strategy: Windowed sampling strategy:
            'even' — evenly spaced across the text
            'stratified' — opening + closing + evenly spaced middle passages
            'endpoints' — first and last passages only
        use_chapters: If True, try chapter-based segmentation first.
        max_chapter_words: Max words per chapter before splitting.
        _id: LLTK text _id. Inferred from text_obj if not provided.

    Returns:
        list[tuple[str, dict]]: List of (prompt, metadata) tuples.
        Metadata contains _id, section_id, passage_index, chapter_title,
        year, n_words. Pass to task.run(prompt, metadata=meta) or unzip
        for task.map(prompts, metadata_list=metas).
    """
    # Try chapters first
    if use_chapters and text_obj is not None:
        results = format_chapters(text_obj, max_words=max_chapter_words)
        if results:
            return results

    # Fall back to windowed passages
    if text_obj is not None:
        txt = txt or text_obj.txt
        title = title or getattr(text_obj, 'title', '') or ''
        author = author or getattr(text_obj, 'author', '') or ''
        year = year or getattr(text_obj, 'year', None)
        if not _id:
            _id = getattr(text_obj, '_id', '') or ''
            if not _id:
                corpus_id = getattr(getattr(text_obj, '_corpus', None), 'id', '') or ''
                text_id = getattr(text_obj, 'id', '') or ''
                if corpus_id and text_id:
                    _id = f"_{corpus_id}/{text_id}"

    if not txt:
        return []

    words = txt.split()
    total = len(words)

    if total < n_words:
        return [format_passage(' '.join(words), title=title, author=author,
                               year=year, passage_index="1/1",
                               _id=_id, section_id="W0001")]

    # Determine passage start positions
    if strategy == 'endpoints':
        starts = [0, max(0, total - n_words)]
        n_passages = 2
    elif strategy == 'stratified':
        if n_passages <= 2:
            starts = [0, max(0, total - n_words)]
        else:
            starts = [0]
            inner = n_passages - 2
            for i in range(inner):
                pos = int((i + 1) * (total - n_words) / (inner + 1))
                starts.append(pos)
            starts.append(max(0, total - n_words))
    else:  # 'even'
        if n_passages == 1:
            starts = [total // 2 - n_words // 2]
        else:
            starts = [int(i * (total - n_words) / (n_passages - 1))
                      for i in range(n_passages)]

    results = []
    for i, start in enumerate(starts):
        passage = ' '.join(words[start:start + n_words])
        section_id = f"W{i+1:04d}"
        results.append(format_passage(
            passage, title=title, author=author, year=year,
            passage_index=f"{i+1}/{len(starts)}",
            _id=_id, section_id=section_id,
        ))

    return results
