"""GenreTaskLite: constrained-vocabulary genre classification.

Unlike GenreTask (free-text genre_raw), this task uses the 60 canonical tags
from lltk's facets.yml as a list[Literal[...]] field. This makes it:
  - ensemble-able (majority vote works on discrete labels)
  - faster (~5-7x fewer output tokens than GenreTask)
  - directly compatible with lltk.text_genre_tags (no normalization needed)
"""

from typing import Literal
from pydantic import BaseModel, Field
from largeliterarymodels.task import Task


GENRE_FORM_TAGS = [
    "allegory", "ballad", "bildungsroman", "biography", "chapbook",
    "character writing", "comedy", "criminal biography", "dialogue", "drama",
    "elegy", "epic", "fable", "hagiography", "hymn", "imaginary voyage",
    "interlude", "it-narrative", "jestbook", "lyric", "masque", "novel",
    "novella", "ode", "opera", "pamphlet fiction", "rogue fiction",
    "roman à clef", "romance", "satire", "secret history", "sonnet", "tale",
    "tragedy", "tragical history", "tragicomedy", "utopia",
]

GENRE_MODE_TAGS = [
    "allegorical", "amatory", "biblical", "chivalric", "classical", "comic",
    "didactic", "epistolary", "gallant", "gothic", "heroic", "historical",
    "legendary", "mock-chivalric", "oriental", "pastoral", "picaresque",
    "political", "psychological", "religious", "satirical", "scandal",
    "sentimental",
]

ALL_GENRE_TAGS = sorted(set(GENRE_FORM_TAGS + GENRE_MODE_TAGS))

GenreTag = Literal[
    "allegory", "allegorical", "amatory", "ballad", "biblical",
    "bildungsroman", "biography", "chapbook", "character writing",
    "chivalric", "classical", "comedy", "comic", "criminal biography",
    "dialogue", "didactic", "drama", "elegy", "epic", "epistolary",
    "fable", "gallant", "gothic", "hagiography", "heroic", "historical",
    "hymn", "imaginary voyage", "interlude", "it-narrative", "jestbook",
    "legendary", "lyric", "masque", "mock-chivalric", "novel", "novella",
    "ode", "opera", "oriental", "pamphlet fiction", "pastoral",
    "picaresque", "political", "psychological", "religious",
    "rogue fiction", "roman à clef", "romance", "satire", "satirical",
    "scandal", "secret history", "sentimental", "sonnet", "tale",
    "tragedy", "tragical history", "tragicomedy", "utopia",
]


class GenreClassificationLite(BaseModel):
    genre_tags: list[GenreTag] = Field(
        description="One or more genre tags for this work, drawn from the canonical "
        "vocabulary. Pick ALL that apply: typically one form tag (e.g. 'novel', "
        "'romance', 'tale') plus one or more mode/register tags (e.g. 'epistolary', "
        "'gothic', 'amatory'). For compound works, include tags for each component "
        "(e.g. ['romance', 'chivalric', 'satire'] for a satirical chivalric romance)."
    )
    author_first_name: str = Field(
        default="",
        description="The author's first name, if you can identify them. "
        "Leave empty if unknown.",
    )
    year_estimated: int = Field(
        default=0,
        description="Your best estimate of when this work was FIRST published "
        "(not this edition). 0 if unknown.",
    )


SYSTEM_PROMPT = """You are an expert in early modern English literature (1475-1800) with deep knowledge of print culture, genre conventions, and literary history.

You are classifying texts by genre based on their title, author, year, and any available catalog metadata. These texts come from library catalogs (ESTC, EEBO, ECCO).

Your job is to assign genre tags from a fixed vocabulary. Each tag is either a FORM (the kind of text: novel, romance, tale, fable, satire, etc.) or a MODE (a literary mode or register that modifies the form: epistolary, gothic, amatory, picaresque, etc.).

For each text, pick ALL tags that apply:
- Usually one form tag (novel, romance, tale, etc.)
- Plus any relevant mode tags (epistolary, sentimental, gothic, etc.)
- For multi-genre works, include multiple form tags

Examples of tag combinations:
- A Gothic novel: ["novel", "gothic"]
- An epistolary novel: ["novel", "epistolary"]
- A satirical chivalric romance: ["romance", "chivalric", "satirical"]
- A picaresque tale: ["tale", "picaresque"]
- A didactic allegory: ["allegory", "didactic"]
- A criminal biography: ["criminal biography"] (this is already a specific form)
- An imaginary voyage with satirical elements: ["imaginary voyage", "satirical"]

Key considerations:
- "The history of X" is usually Fiction (novel/romance) in this period, NOT History
- "The life of X" or "Memoirs of X" when X is fictional = novel
- Jestbooks, rogue literature, and criminal biographies are specific forms — use the dedicated tags
- For works you don't recognize, use broader tags (just "novel" or "romance") rather than guessing specific modes
- Base classification on the title/metadata, not on the author's general reputation
"""


EXAMPLES = [
    (
        "Title: Pamela; or, Virtue rewarded\nAuthor: Richardson\nYear: 1700-1800",
        GenreClassificationLite(
            genre_tags=["novel", "epistolary", "sentimental"],
            author_first_name="Samuel",
            year_estimated=1740,
        ),
    ),
    (
        "Title: The history of Valentine and Orson\nYear: 1500-1600",
        GenreClassificationLite(
            genre_tags=["romance", "chivalric"],
            author_first_name="",
            year_estimated=1510,
        ),
    ),
    (
        "Title: The pilgrim's progress from this world to that which is to come\nAuthor: Bunyan\nYear: 1600-1700",
        GenreClassificationLite(
            genre_tags=["allegory", "religious", "didactic"],
            author_first_name="John",
            year_estimated=1678,
        ),
    ),
    (
        "Title: The mysteries of Udolpho\nAuthor: Radcliffe\nYear: 1700-1800",
        GenreClassificationLite(
            genre_tags=["novel", "gothic"],
            author_first_name="Ann",
            year_estimated=1794,
        ),
    ),
]


class GenreTaskLite(Task):
    name = "classify_genre_lite"
    schema = GenreClassificationLite
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.2
