"""Emotion classification using Chinese-native taxonomy (七情 + TCM + classical literary vocabulary).

NOT a translation of Western emotion wheels. The primary categories derive
from the Confucian 七情 (Liji) and TCM 七情, with secondary/tertiary terms
drawn from classical Chinese literary emotion vocabulary. Categories that have
no primary equivalent in the Feelings Wheel — EmotionTask's Western taxonomy —
(欲 desire, 思 pensiveness, 愛 love-as-primary) are preserved; categories
where the mapping is partial (惡 moral-revulsion vs. the Feelings Wheel's
physical Disgust) are annotated.

Designed for cross-taxonomy comparison with EmotionTask (Feelings Wheel).
Run both tasks on both English and Chinese texts; divergences in coverage
are findings.

Script note: the vocabulary is traditional-Chinese throughout. Model outputs
in simplified script are normalized to the canonical traditional form before
validation (see normalize_emotion_zh).
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from largeliterarymodels.task import Task
from ._emotion_common import CLASSIFICATION_PRINCIPLES_CORE


EMOTION_WHEEL_ZH = {
    "喜": {
        "pinyin": "xǐ",
        "gloss": "joy, delight",
        "secondary": [
            ("樂", "lè", "pleasure, enjoyment, contentment"),
            ("歡", "huān", "elation, jubilation, merriment"),
            ("悅", "yuè", "delight, satisfaction, being pleased"),
            ("慰", "wèi", "comfort, consolation, relief"),
        ],
        "tertiary": [
            ("快", "kuài", "gratification, gladness"),
            ("暢", "chàng", "exhilaration, uninhibited joy"),
            ("幸", "xìng", "good fortune, felicity"),
            ("得意", "déyì", "triumphant satisfaction, self-satisfaction"),
            ("逸", "yì", "carefree ease, transcendent joy"),
        ],
    },
    "怒": {
        "pinyin": "nù",
        "gloss": "anger, wrath, fury",
        "secondary": [
            ("憤", "fèn", "indignation, righteous anger"),
            ("怨", "yuàn", "resentment, grievance, complaint"),
            ("恨", "hèn", "rancor, lasting bitterness; in classical usage also: regret"),
            ("憎", "zēng", "loathing, detestation"),
        ],
        "tertiary": [
            ("忿", "fèn", "grudge, acute resentment"),
            ("嫉", "jí", "jealousy, envious resentment"),
            ("躁", "zào", "agitation, restless anger"),
            ("惱", "nǎo", "vexation, irritation, annoyance"),
            ("嗔", "chēn", "wrath, quick temper (Buddhist)"),
        ],
    },
    "哀": {
        "pinyin": "āi",
        "gloss": "grief, sorrow, lamentation",
        "secondary": [
            ("悲", "bēi", "pathos, compassionate sorrow, tragic feeling"),
            ("愁", "chóu", "melancholy, brooding sorrow"),
            ("傷", "shāng", "heartbreak, wounded feelings"),
            ("憂", "yōu", "anxious sorrow, troubled worry"),
        ],
        "tertiary": [
            ("慟", "tòng", "anguished wailing grief"),
            ("惋", "wǎn", "regretful pity, sighing over loss"),
            ("淒", "qī", "desolation, bleakness, cold sorrow"),
            ("寂", "jì", "loneliness, solitary emptiness"),
            ("惆悵", "chóuchàng", "wistful melancholy, vague lingering sadness"),
        ],
    },
    "恐": {
        "pinyin": "kǒng",
        "gloss": "fear, dread, terror",
        "secondary": [
            ("懼", "jù", "dread, deep apprehension"),
            ("畏", "wèi", "awe, reverential fear, respect mixed with fear"),
            ("怯", "qiè", "timidity, cowardice, shrinking back"),
            ("惶", "huáng", "panic, frantic alarm"),
        ],
        "tertiary": [
            ("慄", "lì", "trembling with fear, shuddering"),
            ("悚", "sǒng", "horrified shudder, hair-raising fright"),
            ("怖", "bù", "terror, dread (Buddhist: 恐怖)"),
        ],
    },
    "愛": {
        "pinyin": "ài",
        "gloss": "love, affection, attachment",
        "note": "Primary emotion in Chinese taxonomy; no primary equivalent "
                "in the Feelings Wheel (love appears there only as a tertiary "
                "'Loving' under Happy)",
        "secondary": [
            ("慕", "mù", "longing, admiration, yearning for the absent"),
            ("戀", "liàn", "romantic attachment, lovesickness"),
            ("憐", "lián", "tender pity, compassionate love"),
            ("敬", "jìng", "respect, reverence, esteem"),
        ],
        "tertiary": [
            ("親", "qīn", "closeness, intimacy, familial warmth"),
            ("恩", "ēn", "gratitude, grace, felt indebtedness"),
            ("念", "niàn", "missing someone, tender remembrance"),
            ("羨", "xiàn", "admiring envy, longing to emulate"),
            ("感", "gǎn", "being moved, stirred, touched (感動)"),
        ],
    },
    "惡": {
        "pinyin": "wù",
        "gloss": "revulsion, moral disgust, aversion",
        "note": "Partial overlap with the Feelings Wheel's Disgust, but 惡 is "
                "primarily moral, not physical",
        "secondary": [
            ("恥", "chǐ", "shame, moral self-reproach (Mencius: 羞惡之心)"),
            ("厭", "yàn", "disgust, weariness, surfeit"),
            ("鄙", "bǐ", "contempt, looking down on"),
            ("嫌", "xián", "dislike, distaste, fastidious aversion"),
        ],
        "tertiary": [
            ("蔑", "miè", "scorn, utter disregard"),
            ("羞", "xiū", "embarrassment, bashful shame"),
            ("愧", "kuì", "guilt, ashamed conscience"),
            ("忌", "jì", "taboo-dread, jealous wariness"),
            ("棄", "qì", "rejection, abandonment, casting away"),
        ],
    },
    "欲": {
        "pinyin": "yù",
        "gloss": "desire, longing, craving",
        "note": "Primary emotion in Chinese taxonomy; no Feelings Wheel equivalent",
        "secondary": [
            ("貪", "tān", "greed, covetousness, insatiability"),
            ("望", "wàng", "hope, aspiration, expectation"),
            ("迷", "mí", "fascination, enchantment, obsession"),
            ("渴", "kě", "thirst, urgent craving"),
        ],
        "tertiary": [
            ("癡", "chī", "infatuation, delusion, obsessive attachment (Buddhist)"),
            ("企", "qǐ", "yearning anticipation, standing on tiptoe"),
            ("狂", "kuáng", "wild abandon, ecstatic frenzy"),
        ],
    },
    "思": {
        "pinyin": "sī",
        "gloss": "pensiveness, contemplation, rumination",
        "note": "From TCM 七情; no Feelings Wheel equivalent. Emotion-as-"
                "cognition: the ache of thinking itself",
        "secondary": [
            ("慮", "lǜ", "deliberation, anxious calculation"),
            ("懷", "huái", "cherishing in memory, nostalgia"),
            ("惘", "wǎng", "bewilderment, being at a loss"),
            ("悶", "mèn", "pent-up brooding, stifled restlessness"),
        ],
        "tertiary": [
            ("憶", "yì", "recollection, vivid remembering"),
            ("疑", "yí", "doubt, suspicion, uncertainty"),
            ("困", "kùn", "perplexity, being stuck, mental exhaustion"),
        ],
    },
    "驚": {
        "pinyin": "jīng",
        "gloss": "fright, alarm, startlement",
        "note": "From TCM 七情; overlaps with the Feelings Wheel's Surprise "
                "but weighted toward alarm",
        "secondary": [
            ("駭", "hài", "shock, being violently startled"),
            ("愕", "è", "astonishment, being stunned speechless"),
            ("異", "yì", "wonder, marveling at the strange"),
        ],
        "tertiary": [
            ("詫", "chà", "taken aback, surprised disapproval"),
            ("奇", "qí", "curiosity, wonder at the marvelous"),
            ("怔", "zhēng", "dazed, stunned into stillness"),
        ],
    },
}


# Flat sets for validation
ALL_EMOTIONS_ZH = set()
for _primary, _data in EMOTION_WHEEL_ZH.items():
    ALL_EMOTIONS_ZH.add(_primary)
    ALL_EMOTIONS_ZH.update(char for char, _, _ in _data["secondary"])
    ALL_EMOTIONS_ZH.update(char for char, _, _ in _data["tertiary"])
ALL_EMOTIONS_ZH = frozenset(ALL_EMOTIONS_ZH)

# Character → (pinyin, gloss) lookup
EMOTION_GLOSSARY_ZH = {}
for _primary, _data in EMOTION_WHEEL_ZH.items():
    EMOTION_GLOSSARY_ZH[_primary] = (_data["pinyin"], _data["gloss"])
    for _char, _pin, _gl in _data["secondary"]:
        EMOTION_GLOSSARY_ZH[_char] = (_pin, _gl)
    for _char, _pin, _gl in _data["tertiary"]:
        EMOTION_GLOSSARY_ZH[_char] = (_pin, _gl)

# Backwards-compatible alias (pre-rename export name).
EMOTION_GLOSSARY = EMOTION_GLOSSARY_ZH

# Where the two taxonomies partially overlap and where they diverge.
# Maps each 七情 primary to the Feelings Wheel primaries (EmotionTask's
# EMOTION_WHEEL) it partially corresponds to; [] = no primary equivalent.
FEELINGS_WHEEL_PARTIAL_EQUIVALENCES = {
    "喜": ["Happy"],
    "怒": ["Anger"],
    "哀": ["Sad"],
    "恐": ["Fear"],
    "愛": [],
    "惡": ["Disgust"],
    "欲": [],
    "思": [],
    "驚": ["Surprise"],
}

# Backwards-compatible alias: earlier versions mislabeled the Western
# reference taxonomy as Plutchik's; the vocabulary it maps to is the
# Feelings Wheel (Willcox). Prefer FEELINGS_WHEEL_PARTIAL_EQUIVALENCES.
PLUTCHIK_PARTIAL_EQUIVALENCES = FEELINGS_WHEEL_PARTIAL_EQUIVALENCES

# Simplified (or common variant) → canonical traditional spelling for every
# vocabulary character that differs. Applied before membership validation so
# simplified-corpus runs (e.g. May Fourth fiction pipelines) don't fail
# lookup, and so traditional-corpus outputs never mix scripts.
SIMPLIFIED_TO_CANONICAL_ZH = {
    "爱": "愛", "恶": "惡", "惊": "驚",
    "乐": "樂", "欢": "歡", "悦": "悅", "畅": "暢",
    "愤": "憤", "恼": "惱", "恸": "慟", "凄": "淒",
    "惧": "懼", "栗": "慄", "恋": "戀", "怜": "憐",
    "亲": "親", "羡": "羨", "耻": "恥", "厌": "厭",
    "弃": "棄", "贪": "貪", "痴": "癡", "虑": "慮",
    "怀": "懷", "闷": "悶", "忆": "憶", "骇": "駭",
    "异": "異", "诧": "詫", "伤": "傷", "忧": "憂",
    "惆怅": "惆悵",
}


def normalize_emotion_zh(value: str) -> str:
    """Normalize to the canonical traditional spelling, or raise ValueError."""
    v = value.strip()
    v = SIMPLIFIED_TO_CANONICAL_ZH.get(v, v)
    if v not in ALL_EMOTIONS_ZH:
        per_char = "".join(SIMPLIFIED_TO_CANONICAL_ZH.get(ch, ch) for ch in v)
        if per_char in ALL_EMOTIONS_ZH:
            return per_char
        raise ValueError(
            f"emotion {value!r} is not in the 七情 vocabulary"
        )
    return v


def _format_wheel_for_prompt():
    lines = []
    for primary, data in EMOTION_WHEEL_ZH.items():
        pinyin, gloss = data["pinyin"], data["gloss"]
        note = data.get("note", "")
        header = f"**{primary}** ({pinyin}) — {gloss}"
        if note:
            header += f"  [{note}]"
        lines.append(f"### {header}")
        sec = " · ".join(
            f"{ch} ({py}) {gl}" for ch, py, gl in data["secondary"]
        )
        lines.append(f"Secondary: {sec}")
        ter = " · ".join(
            f"{ch} ({py}) {gl}" for ch, py, gl in data["tertiary"]
        )
        lines.append(f"Specific: {ter}")
        lines.append("")
    return "\n".join(lines)


class EmotionInstanceZh(BaseModel):
    emotion: str = Field(
        description="The Chinese character(s) for the emotion, drawn from the "
        "vocabulary below. Prefer the most specific (tertiary) term that fits. "
        "Use a secondary or primary term only when no tertiary term captures it."
    )
    emotion_pinyin: str = Field(
        description="Pinyin romanization of the emotion character(s)."
    )
    emotion_gloss: str = Field(
        description="Brief English gloss (2-5 words). This is an approximate "
        "translation — note in the gloss if the English is imprecise."
    )
    quote: str = Field(
        description="A short quotation from the passage (verbatim, in the "
        "passage's original language) that indicates or evokes this emotion. "
        "Keep under ~30 words."
    )

    @field_validator("emotion")
    @classmethod
    def _emotion_in_vocabulary(cls, v):
        return normalize_emotion_zh(v)


class EmotionAnnotationZh(BaseModel):
    emotions: list[EmotionInstanceZh] = Field(
        default_factory=list,
        description="Emotions represented, implied, or invoked in the passage. "
        "Include both character emotions (felt within the text) and reader "
        "emotions (evoked by tone, imagery, or situation). "
        "List each distinct emotion once with its strongest supporting quote.",
    )
    dominant_emotion: str = Field(
        default="",
        description="The single most prominent emotion (Chinese character). "
        "Must be one of the emotions listed above.",
    )
    dominant_emotion_pinyin: str = Field(
        default="",
        description="Pinyin of the dominant emotion.",
    )
    emotional_valence: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Overall emotional valence from -1.0 (negative: 哀, 恐, 怒, 惡) "
        "to +1.0 (positive: 喜, 愛). 0.0 for neutral, mixed, or dominated by "
        "cognitively-toned emotions (思, 驚).",
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Overall confidence 0.0 to 1.0. Lower for ambiguous, "
        "emotionally flat, or culturally opaque passages.",
    )

    @model_validator(mode="after")
    def _dominant_consistent(self):
        if self.dominant_emotion:
            self.dominant_emotion = normalize_emotion_zh(self.dominant_emotion)
            listed = {e.emotion for e in self.emotions}
            if self.dominant_emotion not in listed:
                raise ValueError(
                    f"dominant_emotion {self.dominant_emotion!r} must appear "
                    f"in `emotions` (listed: {sorted(listed)})"
                )
            if not self.dominant_emotion_pinyin:
                entry = EMOTION_GLOSSARY_ZH.get(self.dominant_emotion)
                if entry:
                    self.dominant_emotion_pinyin = entry[0]
        return self


SYSTEM_PROMPT = f"""You are annotating literary passages for emotional content using a Chinese-native emotion taxonomy derived from the Confucian 七情 (Liji), TCM 七情, and classical Chinese literary vocabulary.

This is NOT a translation of Western emotion models. The categories below reflect Chinese philosophical and literary traditions. Some have no Western equivalent (欲 desire-as-primary, 思 pensiveness-as-emotion, 愛 love-as-primary). Where overlap with Western categories exists, it is partial — 惡 (moral revulsion) is not the same as English "disgust" (physical revulsion), and 畏 (awe/reverential fear) has no single English equivalent.

The passage may be in English or Chinese. Regardless of the passage language:
- Always use the Chinese character(s) from the vocabulary below as the emotion label, in the traditional spelling shown
- Quote verbatim from the passage in its original language
- Provide pinyin and an approximate English gloss for each emotion

## Emotion vocabulary

{_format_wheel_for_prompt()}

## Classification principles

{CLASSIFICATION_PRINCIPLES_CORE}
- **Dominant emotion**: the single Chinese character(s) that best characterize the passage's emotional register; it must also appear in the emotions list.
- **Translation gaps are expected**: when annotating English text, some passages will resist Chinese emotion categories. If an English emotion has no good Chinese equivalent, use the closest term and note the imprecision in the gloss. If nothing fits, omit it — gaps are data, not errors."""


class EmotionTaskZh(Task):
    name = "classify_emotion_zh"
    schema = EmotionAnnotationZh
    system_prompt = SYSTEM_PROMPT
    examples = [
        (
            "PASSAGE: 夜深了，她獨自坐在窗前，想起母親臨終的話，眼淚無聲地流下來。"
            "遠處傳來爆竹聲，是別人家的團圓。",
            EmotionAnnotationZh(
                emotions=[
                    EmotionInstanceZh(
                        emotion="傷",
                        emotion_pinyin="shāng",
                        emotion_gloss="heartbreak, wounded feelings",
                        quote="眼淚無聲地流下來",
                    ),
                    EmotionInstanceZh(
                        emotion="念",
                        emotion_pinyin="niàn",
                        emotion_gloss="tender remembrance",
                        quote="想起母親臨終的話",
                    ),
                    EmotionInstanceZh(
                        emotion="寂",
                        emotion_pinyin="jì",
                        emotion_gloss="loneliness, solitary emptiness",
                        quote="她獨自坐在窗前",
                    ),
                ],
                dominant_emotion="傷",
                dominant_emotion_pinyin="shāng",
                emotional_valence=-0.8,
                confidence=0.85,
            ),
        ),
        (
            "PASSAGE: He read the letter a third time, still not believing "
            "his luck, and burst out laughing alone in the empty hall.",
            EmotionAnnotationZh(
                emotions=[
                    EmotionInstanceZh(
                        emotion="歡",
                        emotion_pinyin="huān",
                        emotion_gloss="elation, jubilation",
                        quote="burst out laughing alone in the empty hall",
                    ),
                    EmotionInstanceZh(
                        emotion="幸",
                        emotion_pinyin="xìng",
                        emotion_gloss="good fortune, felicity",
                        quote="still not believing his luck",
                    ),
                ],
                dominant_emotion="歡",
                dominant_emotion_pinyin="huān",
                emotional_valence=0.8,
                confidence=0.8,
            ),
        ),
        (
            "PASSAGE: 本縣土地平曠，物產以稻米為主，歲輸漕糧三千石。",
            EmotionAnnotationZh(
                emotions=[],
                dominant_emotion="",
                dominant_emotion_pinyin="",
                emotional_valence=0.0,
                confidence=0.9,
            ),
        ),
    ]
    retries = 2
    temperature = 0.2
    max_tokens = 2048
