"""Emotion classification task: identify emotions in a passage using the
Feelings Wheel (Willcox).

Taxonomy note: the vocabulary below is the Willcox/Junto "Feelings Wheel"
(6 primaries — Happy, Surprise, Fear, Anger, Disgust, Sad — with synonym
rings), NOT Plutchik's Wheel of Emotions (8 primaries organized by intensity,
including trust and anticipation). The two are often conflated; cite the
Feelings Wheel in any write-up of results from this task.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from largeliterarymodels.task import Task
from ._emotion_common import CLASSIFICATION_PRINCIPLES_CORE

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

# Case-insensitive lookup so 'joyful' normalizes to 'Joyful' instead of
# failing validation; anything outside the vocabulary raises and feeds the
# task's retry loop.
_CANONICAL = {e.lower(): e for e in ALL_EMOTIONS}


def normalize_emotion(value: str) -> str:
    """Return the canonical vocabulary spelling, or raise ValueError."""
    canon = _CANONICAL.get(value.strip().lower())
    if canon is None:
        raise ValueError(
            f"emotion {value!r} is not in the Feelings Wheel vocabulary"
        )
    return canon


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
        description="Name of the emotion, drawn from the Feelings Wheel vocabulary. "
        "Prefer the most specific (tertiary) term that fits. "
        "Use a secondary or primary term only when the emotion is clearly present "
        "but no tertiary term captures it precisely."
    )
    quote: str = Field(
        description="A short quotation from the passage (verbatim) that indicates "
        "or evokes this emotion. Keep under ~30 words."
    )

    @field_validator("emotion")
    @classmethod
    def _emotion_in_vocabulary(cls, v):
        return normalize_emotion(v)


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
        default=0.0, ge=-1.0, le=1.0,
        description="Overall emotional valence of the passage from -1.0 (entirely "
        "negative: sad, fearful, angry, disgusted) to +1.0 (entirely positive: "
        "happy, surprised-positive). 0.0 for neutral or balanced.",
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Overall confidence 0.0 to 1.0. Lower for ambiguous or "
        "emotionally flat passages.",
    )

    @model_validator(mode="after")
    def _dominant_consistent(self):
        if self.dominant_emotion:
            self.dominant_emotion = normalize_emotion(self.dominant_emotion)
            listed = {e.emotion for e in self.emotions}
            if self.dominant_emotion not in listed:
                raise ValueError(
                    f"dominant_emotion {self.dominant_emotion!r} must appear "
                    f"in `emotions` (listed: {sorted(listed)})"
                )
        return self


SYSTEM_PROMPT = f"""You are annotating passages from literary texts for emotional content using the Feelings Wheel (Willcox).

You will receive a passage with optional metadata (title, author, year). Identify which emotions from the vocabulary below are represented, implied, or invoked in the text.

## Emotion vocabulary (Feelings Wheel)

{_format_wheel_for_prompt()}

**Only use emotions from this vocabulary.** Prefer the most specific (tertiary) term when possible. Use a broader (secondary or primary) term only when no tertiary term fits precisely.

## Classification principles

{CLASSIFICATION_PRINCIPLES_CORE}
- **Dominant emotion**: choose the single emotion that best characterizes the passage's overall emotional register; it must also appear in the emotions list.
- **Valence**: rate the overall emotional direction. Most literary passages skew mixed or negative; do not default to positive."""


class EmotionTask(Task):
    name = "classify_emotion"
    schema = EmotionAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = [
        (
            "PASSAGE: The wind rose in the night, and Ellen sat alone by the "
            "cold hearth long after the candle had guttered out. He would not "
            "come back; she had known it since morning, though she had not "
            "dared to say the word even to herself.",
            EmotionAnnotation(
                emotions=[
                    EmotionInstance(
                        emotion="Despair",
                        quote="He would not come back; she had known it since morning",
                    ),
                    EmotionInstance(
                        emotion="Lonely",
                        quote="Ellen sat alone by the cold hearth long after the candle had guttered out",
                    ),
                    EmotionInstance(
                        emotion="Worried",
                        quote="she had not dared to say the word even to herself",
                    ),
                ],
                dominant_emotion="Despair",
                emotional_valence=-0.8,
                confidence=0.85,
            ),
        ),
        (
            "PASSAGE: When the letter came at last, Tom read it twice on the "
            "doorstep, laughing aloud at nothing, and ran the whole mile home "
            "to tell his mother — though a small cold doubt, even then, "
            "tugged at the edge of his joy.",
            EmotionAnnotation(
                emotions=[
                    EmotionInstance(
                        emotion="Joyful",
                        quote="laughing aloud at nothing",
                    ),
                    EmotionInstance(
                        emotion="Excited",
                        quote="ran the whole mile home to tell his mother",
                    ),
                    EmotionInstance(
                        emotion="Worried",
                        quote="a small cold doubt, even then, tugged at the edge of his joy",
                    ),
                ],
                dominant_emotion="Joyful",
                emotional_valence=0.6,
                confidence=0.8,
            ),
        ),
        (
            "PASSAGE: The parish of Elmswell contains four thousand acres of "
            "arable land, the greater part enclosed in 1782. The soil is a "
            "heavy clay, and the chief crops are wheat and beans.",
            EmotionAnnotation(
                emotions=[],
                dominant_emotion="",
                emotional_valence=0.0,
                confidence=0.9,
            ),
        ),
    ]
    retries = 2
    temperature = 0.2
    max_tokens = 2048
