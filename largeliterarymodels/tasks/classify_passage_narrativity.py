"""Passage narrativity classification: what is this prose doing?

Per-passage task: classifies discourse mode, focalization, narrator presence,
and dialogue density. Implements Genette's voice/mood/duration framework for
computational analysis of narrative technique across periods.

Designed to complement PassageSettingTask (where) with narratological analysis
(how). Together they give the full chronotope: where, when, how told, who sees.

Usage:
    from largeliterarymodels.tasks import PassageNarrativityTask

    task = PassageNarrativityTask()
    result = task.run(passage_text)
"""

from typing import Literal
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


class PassageNarrativityAnnotation(BaseModel):
    discourse_mode: Literal[
        'narration', 'scene', 'description', 'commentary',
        'interior_monologue', 'free_indirect_discourse', 'summary',
    ] = Field(
        description=(
            "The PRIMARY discourse mode of the passage — what the prose is DOING. "
            "narration: telling events as they happen, in sequence, at moderate pace. "
            "'He walked to the door and opened it. The rain had stopped.' "
            "scene: real-time dialogue and action, mimetic showing. "
            "'\"I cannot stay,\" she said. He reached for her hand.' "
            "description: setting or character portrayed, time paused. "
            "'The room was small and dark, with heavy curtains drawn against the light.' "
            "commentary: narrator's moral, philosophical, or interpretive reflection. "
            "No events occur; the narrator steps back to reflect or judge. "
            "'It is a truth universally acknowledged...' "
            "interior_monologue: a character's unmediated thought stream. "
            "May use first person even in third-person narration. "
            "'Why had she come? What was the use of it all?' "
            "free_indirect_discourse: narrator and character voice merge — "
            "third person but coloured by the character's perspective, idiom, emotion. "
            "'She would not think of him. She had more important things to attend to.' "
            "summary: compressed narration covering a long timespan quickly. "
            "'Over the next three years he traveled widely, returning each autumn.'"
        )
    )
    discourse_mode_secondary: Literal[
        'narration', 'scene', 'description', 'commentary',
        'interior_monologue', 'free_indirect_discourse', 'summary',
        'none',
    ] = Field(
        default='none',
        description=(
            "If the passage mixes modes, the secondary mode. 'none' if the passage "
            "is predominantly one mode. Most passages mix narration with something else."
        )
    )
    focalization: Literal[
        'zero', 'internal', 'external', 'variable',
    ] = Field(
        description=(
            "Genette's focalization — WHO SEES / whose perspective organizes the passage. "
            "zero: omniscient narrator who can see into anyone's mind, knows the future, "
            "judges freely. The narrator knows MORE than any character. "
            "'He did not know that she had already decided to leave.' "
            "internal: the passage is filtered through ONE character's perception — "
            "we see what they see, feel what they feel, know only what they know. "
            "'The room seemed darker than before. Was that a sound at the door?' "
            "external: the narrator sees from OUTSIDE, reporting only visible behavior — "
            "no access to anyone's thoughts. Like a camera. "
            "'He sat down. She left the room.' "
            "variable: the passage shifts between characters' perspectives."
        )
    )
    narrator_presence: Literal[
        'intrusive', 'effaced', 'character_voice',
    ] = Field(
        description=(
            "How visible/audible is the narrator in this passage? "
            "intrusive: the narrator comments, addresses the reader, makes judgments, "
            "generalizes, or draws attention to the act of narrating. "
            "'The reader will perhaps wonder...' / 'It must be confessed that...' / "
            "'Our heroine, as we shall call her...' (Fielding, Eliot, Thackeray.) "
            "effaced: the narrator is invisible — prose presents events, speech, "
            "and perception without overt narratorial commentary or judgment. "
            "The narrator never says 'I' or addresses 'you'. (Flaubert, James, Hemingway.) "
            "character_voice: the narrator IS a character — first-person narration, "
            "epistolary 'I', diary, or confession. The voice belongs to someone in the story. "
            "(Pamela, Moll Flanders, Jane Eyre, Huckleberry Finn.)"
        )
    )
    dialogue_density: Literal[
        'none', 'sparse', 'mixed', 'dialogue_heavy',
    ] = Field(
        description=(
            "Roughly what fraction of the passage is quoted speech (dialogue)? "
            "none: no quoted speech at all. "
            "sparse: a few lines of dialogue embedded in narration (<25%). "
            "mixed: dialogue and narration roughly balanced (25-75%). "
            "dialogue_heavy: the passage is mostly quoted speech (>75%)."
        )
    )


SYSTEM_PROMPT = """\
You are classifying the NARRATIVE TECHNIQUE of a passage from English prose fiction.

You will receive a ~500-1500 word passage. No title, author, or date is provided — \
classify based solely on what is in the text.

Determine:

1. DISCOURSE MODE (primary) — What is the prose primarily DOING?
   - narration: telling events in sequence at moderate pace
   - scene: real-time dialogue and action, mimetic showing
   - description: setting/character portrayed, time paused
   - commentary: narrator reflects, judges, generalizes — no events
   - interior_monologue: character's unmediated thought stream
   - free_indirect_discourse: third person coloured by character's voice/perspective — \
the narrator's grammar but the character's idiom, emotion, attitude
   - summary: compressed narration covering long timespan quickly

2. DISCOURSE MODE (secondary) — If the passage mixes modes, the secondary one. \
'none' if predominantly one mode.

3. FOCALIZATION — Whose perspective organizes the passage?
   - zero: omniscient, narrator knows more than characters
   - internal: filtered through one character's perception
   - external: outside view only, no access to thoughts
   - variable: shifts between characters' perspectives

4. NARRATOR PRESENCE — How visible is the narrator?
   - intrusive: comments, addresses reader, judges, generalizes
   - effaced: invisible, prose just presents
   - character_voice: first-person, the narrator is a character

5. DIALOGUE DENSITY — How much quoted speech?
   - none / sparse (<25%) / mixed (25-75%) / dialogue_heavy (>75%)

IMPORTANT: Distinguish carefully between:
- commentary (NARRATOR reflects) vs interior_monologue (CHARACTER thinks)
- free_indirect_discourse (third person but character's voice) vs \
interior_monologue (direct thought, often first person or questions)
- narration (events at normal pace) vs summary (compressed events over long time)"""


EXAMPLES = [
    (
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife. However "
        "little known the feelings or views of such a man may be on his "
        "first entering a neighbourhood, this truth is so well fixed in "
        "the minds of the surrounding families, that he is considered the "
        "rightful property of some one or other of their daughters.",
        PassageNarrativityAnnotation(
            discourse_mode='commentary',
            discourse_mode_secondary='none',
            focalization='zero',
            narrator_presence='intrusive',
            dialogue_density='none',
        ),
    ),
    (
        "\"I cannot bear it,\" she cried, turning away. \"You ask too much "
        "of me.\" He stood motionless by the window. \"I ask nothing,\" he "
        "said quietly, \"that you have not already promised.\" She looked "
        "at him then, and something in his expression — was it resignation? "
        "contempt? — made her falter. What right had he to look so calm?",
        PassageNarrativityAnnotation(
            discourse_mode='scene',
            discourse_mode_secondary='free_indirect_discourse',
            focalization='internal',
            narrator_presence='effaced',
            dialogue_density='mixed',
        ),
    ),
    (
        "Over the next three years he traveled widely, visiting France, "
        "Italy, and the Low Countries, returning each autumn to his estate "
        "in Suffolk. His health, never strong, declined steadily; the cough "
        "that had troubled him since the winter of '08 grew worse with each "
        "journey. By the spring of 1812 he was confined to his rooms.",
        PassageNarrativityAnnotation(
            discourse_mode='summary',
            discourse_mode_secondary='description',
            focalization='zero',
            narrator_presence='effaced',
            dialogue_density='none',
        ),
    ),
]


DEFAULT_MODEL = 'lmstudio/qwen/qwen3.6-35b-a3b'


class PassageNarrativityTask(Task):
    name = "classify_passage_narrativity"
    model = DEFAULT_MODEL
    schema = PassageNarrativityAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.2

    @staticmethod
    def format_input(passage_text: str) -> str:
        """Return passage text only — no title/author/year metadata."""
        return passage_text.strip()
