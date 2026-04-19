"""Cross-lingual word translation task: English word + POS → top-3 German
and top-3 French translations, for building multilingual psycholinguistic
anchor tables from Brysbaert-style lexicons.

Designed for the abstraction project's replication of abstract-concrete
scoring in German and French fiction. Brysbaert's ratings are attached
downstream to the chosen translations; the LLM never sees the rating
(avoids circularity).

Schema is flat (de_1/de_2/de_3/fr_1/fr_2/fr_3) for DataFrame-friendly
downstream use. Top-k preserves candidate spread as an implicit ambiguity
signal — no separate ambiguity flag needed.
"""

from pydantic import BaseModel, Field
from largeliterarymodels.task import Task


POS_VOCAB = [
    'Noun', 'Verb', 'Adjective', 'Adverb',
    'Pronoun', 'Preposition', 'Conjunction', 'Interjection',
    'Determiner', 'Numeral',
    'Other',
]


class WordTranslation(BaseModel):
    de_1: str = Field(
        description="Primary (most likely) German translation. "
        "Infinitive lowercase for verbs (e.g. 'laufen'). "
        "Bare noun no article, capitalised as in German orthography "
        "(e.g. 'Tisch'). Adjective bare positive form (e.g. 'schnell'). "
        "Canonical form for separable-prefix verbs (e.g. 'aufstehen' not "
        "'stehen auf')."
    )
    de_2: str = Field(
        default="",
        description="Second-ranked German translation, or empty if no "
        "distinct second candidate exists. Same formatting rules as de_1."
    )
    de_3: str = Field(
        default="",
        description="Third-ranked German translation, or empty. "
        "Only include if it's a genuinely distinct sense or register — "
        "do not pad with near-synonyms or regional variants."
    )
    fr_1: str = Field(
        description="Primary (most likely) French translation. "
        "Lowercase infinitive for verbs (e.g. 'courir'). "
        "Bare noun, no article, masculine singular headword if both "
        "genders exist (e.g. 'étudiant' not 'étudiante'). "
        "Adjective as masculine singular positive form."
    )
    fr_2: str = Field(
        default="",
        description="Second-ranked French translation, or empty."
    )
    fr_3: str = Field(
        default="",
        description="Third-ranked French translation, or empty. "
        "Only include if genuinely distinct sense — do not pad."
    )
    sense_note: str = Field(
        default="",
        description="One brief sentence explaining the sense chosen when "
        "the English word is polysemous or POS-ambiguous. Leave empty for "
        "unambiguous words. Useful for debugging and for flagging entries "
        "that deserve human review."
    )


SYSTEM_PROMPT = """You are a multilingual lexicographer translating English words into German and French for a psycholinguistic anchor table.

Your input: one English word plus its part-of-speech tag (Noun, Verb, Adjective, etc.). Your output: top-3 German translations and top-3 French translations, ranked by likelihood of being the translation a native speaker would choose for the given sense.

## Critical rules

1. **Use the POS tag to disambiguate homographs.** "will" as Verb = modal ('werden' / 'vouloir'); "will" as Noun = testament/volition ('Wille' / 'volonté'). The POS is authoritative — do not second-guess it.

2. **Formatting is strict and canonical:**
   - German verbs: lowercase infinitive ('laufen', 'aufstehen' — canonical separable form, not 'stehen auf').
   - German nouns: standard German capitalisation ('Tisch', 'Freiheit'), no article.
   - German adjectives: bare positive ('schnell', 'frei'), no inflection.
   - French verbs: lowercase infinitive ('courir', 'aimer').
   - French nouns: bare noun without article. If both genders exist (étudiant/étudiante), use the masculine singular headword.
   - French adjectives: masculine singular positive.

3. **Do not pad the top-K.** If only one or two genuine translations exist, leave de_2/de_3 or fr_2/fr_3 empty. Padding with near-synonyms, regional variants, or archaic forms corrupts the downstream signal.

4. **Rank by primary-sense likelihood, not by completeness.** de_1 should be what a native speaker says first. de_2/de_3 cover distinct senses or clear alternates, not exhaustive synonym lists.

5. **sense_note is for polysemy and hard cases.** Leave empty for unambiguous words. Use it when: the word has multiple genuine senses; the POS resolves a homograph; the translation required a judgement call (e.g. modal 'will' interpretation, polysemous 'light').

## Additional guidance

- For abstract vocabulary: prefer the most common lexeme over technical register.
- For phrasal verbs that Brysbaert lists as single words, translate the core lexeme — note the phrasal constraint in sense_note if it matters.
- If the English word genuinely has no direct translation in one language, give the closest single-word equivalent and note the mismatch in sense_note.
"""


EXAMPLES = [
    # Common verb, multiple distinct senses → non-empty top-3
    (
        "English word: run\nPOS: Verb",
        WordTranslation(
            de_1="laufen",
            de_2="rennen",
            de_3="betreiben",
            fr_1="courir",
            fr_2="diriger",
            fr_3="fonctionner",
            sense_note="Primary sense is physical motion; de_3/fr_2-3 cover the "
                       "'operate/manage' sense (run a company, run a program).",
        ),
    ),
    # Common noun, low ambiguity
    (
        "English word: table\nPOS: Noun",
        WordTranslation(
            de_1="Tisch",
            de_2="Tabelle",
            de_3="",
            fr_1="table",
            fr_2="tableau",
            fr_3="",
            sense_note="de_1/fr_1 are the furniture; de_2/fr_2 cover the "
                       "'data table' sense. No clear third sense.",
        ),
    ),
    # POS-disambiguated homograph
    (
        "English word: will\nPOS: Noun",
        WordTranslation(
            de_1="Wille",
            de_2="Testament",
            de_3="",
            fr_1="volonté",
            fr_2="testament",
            fr_3="",
            sense_note="Noun POS excludes the modal-verb reading. Primary is "
                       "volitional 'Wille/volonté'; secondary is legal testament.",
        ),
    ),
    # Polysemous adjective
    (
        "English word: light\nPOS: Adjective",
        WordTranslation(
            de_1="leicht",
            de_2="hell",
            de_3="",
            fr_1="léger",
            fr_2="clair",
            fr_3="",
            sense_note="Two distinct senses: 'not heavy' (leicht/léger) and "
                       "'bright' (hell/clair). Both common; neither dominates.",
        ),
    ),
    # Abstract noun — typical Brysbaert anchor
    (
        "English word: freedom\nPOS: Noun",
        WordTranslation(
            de_1="Freiheit",
            de_2="",
            de_3="",
            fr_1="liberté",
            fr_2="",
            fr_3="",
            sense_note="",
        ),
    ),
    # Polysemous verb — senses split differently across DE and FR
    (
        "English word: carry\nPOS: Verb",
        WordTranslation(
            de_1="tragen",
            de_2="befördern",
            de_3="übertragen",
            fr_1="porter",
            fr_2="transporter",
            fr_3="véhiculer",
            sense_note="Three distinct senses: physical carrying (tragen/porter), "
                       "transport or conveyance of people/goods (befördern/transporter), "
                       "and metaphorical conveyance of meaning or signals (übertragen/"
                       "véhiculer). Ranking reflects the physical-object sense as primary.",
        ),
    ),
    # Adjective — tests French masculine-singular headword convention
    (
        "English word: pretty\nPOS: Adjective",
        WordTranslation(
            de_1="hübsch",
            de_2="schön",
            de_3="",
            fr_1="joli",
            fr_2="beau",
            fr_3="",
            sense_note="German 'hübsch' is the closer register match to English "
                       "'pretty' (lighter than 'schön' = beautiful). French 'joli' "
                       "similarly matches register; 'beau' covers the stronger "
                       "'beautiful' sense. Both French entries given as masculine "
                       "singular headwords per convention.",
        ),
    ),
    # False-friend trap: English 'gift' vs German 'Gift' (=poison)
    (
        "English word: gift\nPOS: Noun",
        WordTranslation(
            de_1="Geschenk",
            de_2="Präsent",
            de_3="Gabe",
            fr_1="cadeau",
            fr_2="présent",
            fr_3="don",
            sense_note="CRITICAL false friend: German 'Gift' means POISON, never "
                       "'gift'. Correct translations are Geschenk (common), Präsent "
                       "(formal), Gabe (formal/biblical). French is straightforward: "
                       "cadeau primary, présent formal, don for formal/charitable giving.",
        ),
    ),
]


class TranslationTask(Task):
    """Translate an English word into top-3 German and French equivalents.

    Usage:
        task = TranslationTask()
        prompt = format_word_for_translation("run", "Verb")
        result = task.run(prompt, metadata={'word': 'run', 'pos': 'Verb'})

        # bulk
        prompts = [format_word_for_translation(w, p) for w, p in words_pos]
        metadata_list = [{'word': w, 'pos': p} for w, p in words_pos]
        task.map(prompts, metadata_list=metadata_list,
                 model='claude-haiku-4-5-20251001', num_workers=16)
        df = task.df
    """
    name = "translate_word"
    schema = WordTranslation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.3
    max_tokens = 1024


def format_word_for_translation(word: str, pos: str) -> str:
    """Format a single Brysbaert-style (word, pos) entry as a prompt."""
    return f"English word: {word}\nPOS: {pos}"
