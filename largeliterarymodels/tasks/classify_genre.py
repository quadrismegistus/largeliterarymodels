"""Genre classification task: classify early modern English texts by genre using title/author/metadata."""

from pydantic import BaseModel, Field
from typing import Optional
from lltk.tools.vocabs import GENRE_VOCAB
from largeliterarymodels.task import Task

GENRE_RAW_EXAMPLES = [
    'Novel', 'Romance', 'Romance, chivalric', 'Romance, heroic',
    'Romance, pastoral', 'Romance, allegorical', 'Picaresque', 'Rogue fiction', 'Novel, epistolary',
    'Novel, sentimental', 'Fiction, satire', 'Fable', 'Allegory',
    'Imaginary voyage', 'Utopia', 'Novel, It Narrative', 'Tale', 'Novella',
    'Jestbook', 'Tale, oriental', 'Tale, fairy', 'Historical fiction',
    'Novel, anti-Jacobin', 'Novel, Jacobin', 'Novel, Gothic',
    'Chapbook', 'Criminal biography', 'Tragical history',
    'Novel, Silver Fork'
]
GENRE_RAW_EXAMPLES_STR = "; ".join([f"{x}" for x in GENRE_RAW_EXAMPLES])


class GenreClassification(BaseModel):
    genre: str = Field(
        description=f"Broad genre category. One of: {', '.join(sorted(GENRE_VOCAB))}. "
        "Use 'Fiction' for prose fiction (novels, romances, tales, etc.). "
        "Use 'Poetry' for verse. Use 'Drama' for plays. "
        "Use 'Nonfiction' if it doesn't fit any specific nonfiction category."
    )
    genre_raw: str = Field(
        default="",
        description="More specific subgenre for fiction. Examples: "
        f"{GENRE_RAW_EXAMPLES_STR}. "
        "Leave empty if genre is not Fiction or if you can't determine the subgenre. "
        "For non-fiction misclassified as fiction, leave empty and set genre correctly. "
        "Format: [major subgenre][comma][qualifiers...]. "
        "Feel free to invent your own but prefer existing ones. "
        "Lean toward using period-appropriate designators: e.g. 'Tragical history' not 'Cautionary tale'. " \
        "Try to provide a subcategory if at all possible. So: 'Romance, chivalric' or 'Romance, pastoral' instead of just 'Romance'. "
    )
    author_first_name: str = Field(
        default="",
        description="The author's first name, if you can identify them from the metadata. "
        "This serves as verification that you recognize the author. Leave empty if unknown."
    )
    is_translated: bool = Field(
        default=False,
        description="True if the work is a translation into English. "
        "Look for clues: 'Englished by', 'rendred into English', 'translated by a person of quality', etc."
    )
    translated_from: str = Field(
        default="",
        description="The language this text was translated from, if stated."
    )
    year_estimated: int = Field(
        default=0,
        description="Your best estimate of when this work was FIRST published (not this edition). 0 if unknown."
    )
    confidence: float = Field(
        description="Your confidence in the genre classification, 0.0 to 1.0. "
        "High (>0.8) if you recognize the work or the title is unambiguous. "
        "Medium (0.5-0.8) if making an informed guess from title/author conventions. "
        "Low (<0.5) if the title is too generic or ambiguous to classify."
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why you chose this genre."
    )
    notes: str = Field(
        default="",
        description="Everything you know about this text. 1-5 sentences."
    )


SYSTEM_PROMPT = """You are an expert in early modern English literature (1475-1800) with deep knowledge of print culture, genre conventions, and literary history.

You are classifying texts by genre based on their title, author, year, and any available catalog metadata. These texts come from library catalogs (ESTC, EEBO, ECCO) and some have been pre-classified as "Fiction" but that classification may be wrong.

Draw on your knowledge of the text only. Do NOT simply guess based on the title. If you are not familiar with the play, report low confidence.

Your job:
1. Determine the broad genre (Fiction, Poetry, Drama, Sermon, Treatise, etc.)
2. For fiction, determine the specific subgenre (Novel, Romance, Picaresque, etc.)
3. Flag translations
4. Rate your confidence

Key considerations:
- "The history of X" is usually Fiction (novel/romance) in this period, NOT History
- "The life of X" or "Memoirs of X" when X is fictional = Fiction
- "The life of X" when X is a real person = Biography
- Jestbooks, rogue literature, and criminal biographies are borderline — classify as Fiction
- Chapbooks (short popular fiction) are Fiction
- Verse romances and verse tales are Poetry, not Fiction
- Satirical pamphlets are Essay or Satire, not Fiction, unless they have a sustained fictional narrative
- "Letters" or "Epistles" between real people = Letters; fictional letters = Epistolary fiction
- Works in languages other than English appearing in English catalogs may be translations or may have been miscataloged
"""


EXAMPLES = [
    # Novel — clear case
    (
        "Title: Pamela; or, Virtue rewarded\nAuthor: Richardson\nYear: 1700-1800",
        GenreClassification(
            genre="Fiction",
            genre_raw="Novel, Epistolary fiction",
            is_translated=False,
            author_first_name="Samuel",
            year_estimated=1740,
            confidence=1.0,
            reasoning="Richardson's Pamela is a canonical epistolary novel.",
            notes="Pamela; or, Virtue Rewarded (1740) by Samuel Richardson is one of the earliest English novels. Written as a series of letters, it tells the story of a young servant girl whose master attempts to seduce her. Widely influential, it sparked both imitations and parodies, including Fielding's Shamela.",
        ),
    ),
    # Romance, chivalric — medieval tradition
    (
        "Title: The history of Valentine and Orson\nYear: 1500-1600",
        GenreClassification(
            genre="Fiction",
            genre_raw="Romance, chivalric",
            is_translated=True,
            author_first_name="",
            year_estimated=1510,
            confidence=0.95,
            reasoning="Valentine and Orson is a medieval chivalric romance, originally French, reprinted as a chapbook throughout the 18th century.",
            notes="Valentine and Orson is a French chivalric romance about twin brothers separated at birth — one raised at court, the other by a bear in the forest. First printed in French c.1489, it was translated into English by Henry Watson c.1510 and remained one of the most popular chapbook romances for centuries.",
        ),
    ),
    # Not fiction — satirical pamphlet misclassified
    (
        "Title: Reasons for abolishing ceremony\nAuthor: Swift\nYear: 1700-1800\nSubject: English fiction",
        GenreClassification(
            genre="Essay",
            genre_raw="",
            is_translated=False,
            author_first_name="Jonathan",
            year_estimated=1708,
            confidence=0.9,
            reasoning="Despite the catalog subject heading, this is a satirical essay by Swift, not fiction.",
            notes="An Argument Against Abolishing Christianity (1708) by Jonathan Swift is a satirical essay written in the persona of a nominal Christian defending the established church. It is not fiction despite sometimes appearing under 'English fiction' in library catalogs.",
        ),
    ),
    # Allegory
    (
        "Title: The pilgrim's progress from this world to that which is to come\nAuthor: Bunyan\nYear: 1600-1700",
        GenreClassification(
            genre="Fiction",
            genre_raw="Allegory",
            is_translated=False,
            author_first_name="John",
            year_estimated=1678,
            confidence=1.0,
            reasoning="Bunyan's Pilgrim's Progress is a religious allegory with a sustained fictional narrative.",
            notes="The Pilgrim's Progress (1678) by John Bunyan is a Christian allegory following the character Christian on his journey from the City of Destruction to the Celestial City. One of the most widely read works in English, it was written while Bunyan was imprisoned for unlicensed preaching.",
        ),
    ),
    # Translation — foreign author
    (
        "Title: The adventures of Telemachus\nAuthor: Fénelon\nYear: 1600-1700",
        GenreClassification(
            genre="Fiction",
            genre_raw="Romance, didactic",
            is_translated=True,
            author_first_name="François",
            year_estimated=1699,
            confidence=0.95,
            reasoning="Fénelon's Télémaque is a French didactic romance, widely translated into English.",
            notes="Les Aventures de Télémaque (1699) by François de Salignac de la Mothe-Fénelon was written for the education of the Duke of Burgundy. A prose sequel to Homer's Odyssey following Telemachus's search for his father, it was read as a veiled critique of Louis XIV's reign. One of the most translated and reprinted works of the 18th century.",
        ),
    ),
    # Not fiction — scholarly work
    (
        "Title: Inscriptiones citie\nAuthor: Swinton\nYear: 1700-1800\nSubject: English fiction",
        GenreClassification(
            genre="Academic",
            genre_raw="",
            is_translated=False,
            author_first_name="John",
            year_estimated=1750,
            confidence=0.85,
            reasoning="Despite the catalog subject, this appears to be a scholarly work on inscriptions, not fiction. Swinton was an orientalist and antiquary.",
            notes="John Swinton (1703–1777) was an English orientalist and antiquary at Christ Church, Oxford. His works on ancient inscriptions are scholarly, not fictional, despite occasional miscataloging under 'English fiction' in ESTC.",
        ),
    ),
    # Gothic novel
    (
        "Title: The mysteries of Udolpho\nAuthor: Radcliffe\nYear: 1700-1800",
        GenreClassification(
            genre="Fiction",
            genre_raw="Gothic",
            is_translated=False,
            author_first_name="Ann",
            year_estimated=1794,
            confidence=1.0,
            reasoning="Radcliffe's Mysteries of Udolpho is the archetypal Gothic novel.",
            notes="The Mysteries of Udolpho (1794) by Ann Radcliffe is the quintessential Gothic novel. It follows Emily St. Aubert through dark castles and apparent supernatural terrors, all ultimately given rational explanations. Radcliffe's 'explained supernatural' technique was enormously influential and famously parodied in Austen's Northanger Abbey.",
        ),
    ),
    # # Criminal biography — borderline
    # (
    #     "Title: The lives of noted highwaymen viz\nYear: 1700-1800",
    #     GenreClassification(
    #         genre="Fiction",
    #         genre_raw="Criminal biography",
    #         is_translated=False,
    #         author_first_name="",
    #         year_estimated=0,
    #         confidence=0.6,
    #         reasoning="Collections of highwayman lives in this period are heavily fictionalized. Borderline between fiction and biography, but the sensationalized format suggests popular fiction.",
    #     ),
    # ),
]


class GenreTask(Task):
    name = "classify_genre"
    schema = GenreClassification
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.2


def format_text_for_classification(title, author=None, author_norm=None,
                                    year=None, year_range_width=100,
                                    subject_topic=None, form=None, **extra):
    """Format a text's metadata into a prompt string for the GenreTask.

    Uses author_norm (last name only) and a year range instead of exact year,
    so the LLM's author_first_name and year_estimated responses serve as
    verification that it actually recognizes the work.
    """
    parts = [f"Title: {title}"]
    # Send last name only (for verification via author_first_name)
    name = author_norm or author
    if name:
        # If we got full author, extract last name
        if author and not author_norm:
            name = name.split(',')[0].strip()
        parts.append(f"Author: {name.title()}")
    # Send century range (for verification via year_estimated)
    if year:
        y = int(year)
        century_start = (y // 100) * 100
        parts.append(f"Year: {century_start}-{century_start + year_range_width}")
    if subject_topic:
        parts.append(f"Subject: {subject_topic}")
    if form:
        parts.append(f"Form: {form}")
    for k, v in extra.items():
        if v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts)

def format_lltk_text_for_classification(text, **kwargs):
    meta = text.meta
    title = meta.get('title','').split('(')[0].strip()
    title_sub = meta.get('title_sub','')
    has_punc = title_sub and not title_sub[0].isalpha()
    title = title + (' '+title_sub if not has_punc else title_sub)
    author = meta.get('author','')
    year = int(meta.get('year',0))
    return format_text_for_classification(
        title.strip(),
        author=author,
        year=year,
    )