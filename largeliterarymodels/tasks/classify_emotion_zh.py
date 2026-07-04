"""Emotion classification using Chinese-native taxonomy (七情 + TCM + classical literary vocabulary).

NOT a translation of Plutchik. The primary categories derive from the Confucian 七情
(Liji) and TCM 七情, with secondary/tertiary terms drawn from classical Chinese
literary emotion vocabulary. Categories that have no Plutchik equivalent (欲 desire,
思 pensiveness, 愛 love-as-primary) are preserved; categories where the mapping is
partial (惡 moral-revulsion vs. Plutchik's physical-disgust) are annotated.

Designed for cross-taxonomy comparison with EmotionTask (Plutchik). Run both tasks
on both English and Chinese texts; divergences in coverage are findings.
"""

from pydantic import BaseModel, Field
from largeliterarymodels.task import Task


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
        "note": "Primary emotion in Chinese taxonomy; no Plutchik primary equivalent",
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
            ("羡", "xiàn", "admiring envy, longing to emulate"),
            ("感", "gǎn", "being moved, stirred, touched (感動)"),
        ],
    },
    "惡": {
        "pinyin": "wù",
        "gloss": "revulsion, moral disgust, aversion",
        "note": "Partial Plutchik overlap with Disgust, but 惡 is primarily moral, not physical",
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
        "note": "Primary emotion in Chinese taxonomy; no Plutchik equivalent",
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
        "note": "From TCM 七情; no Plutchik equivalent. Emotion-as-cognition: the "
                "ache of thinking itself",
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
    "惊": {
        "pinyin": "jīng",
        "gloss": "fright, alarm, startlement",
        "note": "From TCM 七情; overlaps with Plutchik Surprise but weighted toward alarm",
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
EMOTION_GLOSSARY = {}
for _primary, _data in EMOTION_WHEEL_ZH.items():
    EMOTION_GLOSSARY[_primary] = (_data["pinyin"], _data["gloss"])
    for _char, _pin, _gl in _data["secondary"]:
        EMOTION_GLOSSARY[_char] = (_pin, _gl)
    for _char, _pin, _gl in _data["tertiary"]:
        EMOTION_GLOSSARY[_char] = (_pin, _gl)

# Where the two taxonomies partially overlap and where they diverge
PLUTCHIK_PARTIAL_EQUIVALENCES = {
    "喜": ["Happy"],
    "怒": ["Anger"],
    "哀": ["Sad"],
    "恐": ["Fear"],
    "愛": [],
    "惡": ["Disgust"],
    "欲": [],
    "思": [],
    "惊": ["Surprise"],
}


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
        default=0.0,
        description="Overall emotional valence from -1.0 (negative: 哀, 恐, 怒, 惡) "
        "to +1.0 (positive: 喜, 愛). 0.0 for neutral, mixed, or dominated by "
        "cognitively-toned emotions (思, 惊).",
    )
    confidence: float = Field(
        default=0.5,
        description="Overall confidence 0.0 to 1.0. Lower for ambiguous, "
        "emotionally flat, or culturally opaque passages.",
    )


SYSTEM_PROMPT = f"""You are annotating literary passages for emotional content using a Chinese-native emotion taxonomy derived from the Confucian 七情 (Liji), TCM 七情, and classical Chinese literary vocabulary.

This is NOT a translation of Western emotion models. The categories below reflect Chinese philosophical and literary traditions. Some have no Western equivalent (欲 desire-as-primary, 思 pensiveness-as-emotion, 愛 love-as-primary). Where overlap with Western categories exists, it is partial — 惡 (moral revulsion) is not the same as English "disgust" (physical revulsion), and 畏 (awe/reverential fear) has no single English equivalent.

The passage may be in English or Chinese. Regardless of the passage language:
- Always use the Chinese character(s) from the vocabulary below as the emotion label
- Quote verbatim from the passage in its original language
- Provide pinyin and an approximate English gloss for each emotion

## Emotion vocabulary

{_format_wheel_for_prompt()}

## Classification principles

- **Represented**: emotions explicitly named or described ("她心中充滿了愁緒" / "she was consumed with melancholy").
- **Implied**: emotions strongly suggested by action, dialogue, or situation without being named.
- **Invoked**: emotions the passage evokes in the reader through tone, imagery, or dramatic irony.
- **Quote verbatim**: copy a short phrase directly from the passage in its original language — do not translate or paraphrase.
- **One entry per emotion**: if the same emotion appears multiple times, pick the strongest instance.
- **Empty is valid**: if the passage is emotionally flat or purely expository, return an empty list.
- **Dominant emotion**: the single Chinese character(s) that best characterize the passage's emotional register.
- **Translation gaps are expected**: when annotating English text, some passages will resist Chinese emotion categories. If an English emotion has no good Chinese equivalent, use the closest term and note the imprecision in the gloss. If nothing fits, omit it — gaps are data, not errors."""


class EmotionTaskZh(Task):
    name = "classify_emotion_zh"
    schema = EmotionAnnotationZh
    system_prompt = SYSTEM_PROMPT
    examples = []
    retries = 2
    temperature = 0.2
    max_tokens = 2048
