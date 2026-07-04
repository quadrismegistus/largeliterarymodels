"""Emotion classification task: identify emotions in a passage using Plutchik's Wheel of Emotions."""

from pydantic import BaseModel, Field
from largeliterarymodels.task import Task

EMOTION_WHEEL = {
    "Happy": {
        "secondary": [
            "Playful", "Content", "Interested", "Proud",
            "Accepted", "Powerful", "Peaceful", "Optimistic",
        ],
        "tertiary": [
            "Aroused", "Cheeky", "Free", "Joyful", "Curious", "Inquisitive",
            "Successful", "Confident", "Respected", "Fulfilled", "Important",
            "Courageous", "Provocative", "Loving", "Hopeful", "Sensitive",
            "Intimate", "Energetic", "Liberated", "Ecstatic", "Amused",
        ],
    },
    "Surprise": {
        "secondary": ["Startled", "Confused", "Amazed", "Excited"],
        "tertiary": [
            "Shocked", "Dismayed", "Disillusioned", "Perplexed", "Astonished",
        ],
    },
    "Fear": {
        "secondary": [
            "Scared", "Anxious", "Insecure", "Submissive", "Rejected", "Humiliated",
        ],
        "tertiary": [
            "Frightened", "Terrified", "Overwhelmed", "Worried",
            "Inadequate", "Inferior", "Worthless", "Insignificant",
            "Alienated", "Ridiculed", "Disrespected", "Embarrassed", "Devastated",
        ],
    },
    "Anger": {
        "secondary": [
            "Hurt", "Threatened", "Hateful", "Mad",
            "Aggressive", "Frustrated", "Distant", "Critical",
        ],
        "tertiary": [
            "Jealous", "Resentful", "Violated", "Furious", "Enraged",
            "Provoked", "Hostile", "Infuriated", "Irritated",
            "Withdrawn", "Suspicious", "Sarcastic", "Skeptical",
        ],
    },
    "Disgust": {
        "secondary": ["Disapproving", "Disappointed", "Awful", "Avoidance"],
        "tertiary": [
            "Judgmental", "Loathing", "Repugnant", "Revolted",
            "Detestable", "Aversion", "Hesitant",
        ],
    },
    "Sad": {
        "secondary": [
            "Lonely", "Guilty", "Depressed", "Bored", "Abandoned", "Despair",
        ],
        "tertiary": [
            "Isolated", "Victimized", "Powerless", "Vulnerable",
            "Empty", "Ashamed", "Remorseful", "Ignored",
            "Apathetic", "Indifferent",
        ],
    },
}

ALL_EMOTIONS = set()
for _primary, _tiers in EMOTION_WHEEL.items():
    ALL_EMOTIONS.add(_primary)
    ALL_EMOTIONS.update(_tiers["secondary"])
    ALL_EMOTIONS.update(_tiers["tertiary"])
ALL_EMOTIONS = frozenset(ALL_EMOTIONS)


def _format_wheel_for_prompt():
    lines = []
    for primary, tiers in EMOTION_WHEEL.items():
        secondary = ", ".join(tiers["secondary"])
        tertiary = ", ".join(tiers["tertiary"])
        lines.append(f"- **{primary}**: {secondary}")
        lines.append(f"  Specific: {tertiary}")
    return "\n".join(lines)


class EmotionInstance(BaseModel):
    emotion: str = Field(
        description="Name of the emotion, drawn from the Plutchik Wheel vocabulary. "
        "Prefer the most specific (tertiary) term that fits. "
        "Use a secondary or primary term only when the emotion is clearly present "
        "but no tertiary term captures it precisely."
    )
    quote: str = Field(
        description="A short quotation from the passage (verbatim) that indicates "
        "or evokes this emotion. Keep under ~30 words."
    )


class EmotionAnnotation(BaseModel):
    emotions: list[EmotionInstance] = Field(
        default_factory=list,
        description="Emotions represented, implied, or invoked in the passage. "
        "Include both character emotions (felt by characters in the text) and "
        "reader emotions (evoked in the reader by tone, imagery, or situation). "
        "List each distinct emotion once, with its strongest supporting quote.",
    )
    dominant_emotion: str = Field(
        default="",
        description="The single most prominent emotion in the passage overall. "
        "Must be one of the emotions listed above.",
    )
    emotional_valence: float = Field(
        default=0.0,
        description="Overall emotional valence of the passage from -1.0 (entirely "
        "negative: sad, fearful, angry, disgusted) to +1.0 (entirely positive: "
        "happy, surprised-positive). 0.0 for neutral or balanced.",
    )
    confidence: float = Field(
        default=0.5,
        description="Overall confidence 0.0 to 1.0. Lower for ambiguous or "
        "emotionally flat passages.",
    )


SYSTEM_PROMPT = f"""You are annotating passages from literary texts for emotional content using Plutchik's Wheel of Emotions.

You will receive a passage with optional metadata (title, author, year). Identify which emotions from the vocabulary below are represented, implied, or invoked in the text.

## Emotion vocabulary (Plutchik's Wheel)

{_format_wheel_for_prompt()}

**Only use emotions from this vocabulary.** Prefer the most specific (tertiary) term when possible. Use a broader (secondary or primary) term only when no tertiary term fits precisely.

## Classification principles

- **Represented**: emotions explicitly named or described in the text ("she felt a pang of jealousy").
- **Implied**: emotions strongly suggested by action, dialogue, or situation without being named ("he slammed the door and refused to speak" → Frustrated, Hostile).
- **Invoked**: emotions the passage is designed to evoke in the reader through tone, imagery, or dramatic irony (a child's death scene invokes Sad/Despair even if no character's grief is described).
- **Quote verbatim**: copy a short phrase directly from the passage — do not paraphrase.
- **One entry per emotion**: if the same emotion appears multiple times, pick the strongest instance.
- **Empty is valid**: if the passage is emotionally flat or purely expository, return an empty list.
- **Dominant emotion**: choose the single emotion that best characterizes the passage's overall emotional register.
- **Valence**: rate the overall emotional direction. Most literary passages skew mixed or negative; do not default to positive."""


class EmotionTask(Task):
    name = "classify_emotion"
    schema = EmotionAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = []
    retries = 2
    temperature = 0.2
    max_tokens = 2048
