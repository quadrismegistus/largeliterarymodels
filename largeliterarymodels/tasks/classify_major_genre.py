"""Major genre classification from title and metadata.

Lightweight task: given a title (and optionally author/year), classify
whether the text is fiction or non-fiction, identify the major genre,
and extract first name of author if available.

Designed for claude-cli provider (free on subscription, ~2-4s/call).

Usage:
    from largeliterarymodels.tasks import MajorGenreTask

    task = MajorGenreTask(model="claude-cli/sonnet")
    result = task.run("The History of Tom Jones, a Foundling (1749)")
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


GENRE_VOCAB = [
    'fiction', 'poetry', 'drama', 'periodical', 'essay', 'treatise',
    'letters', 'sermon', 'biography', 'history', 'criticism',
    'legal', 'speech', 'reference', 'almanac',
]


class MajorGenreAnnotation(BaseModel):
    is_fiction: bool = Field(
        description="Is this prose fiction (novel, romance, tale, novella)?",
    )
    major_genre: Literal[
        'fiction', 'poetry', 'drama', 'periodical', 'essay', 'treatise',
        'letters', 'sermon', 'biography', 'history', 'criticism',
        'legal', 'speech', 'reference', 'almanac',
    ] = Field(
        description="The primary genre category.",
    )
    author_first_name: Optional[str] = Field(
        default=None,
        description=(
            "Author's first/given name, identified from your knowledge of "
            "literary history. Only the surname is provided — use the title "
            "and surname together to identify the author. Null if you cannot "
            "confidently identify them."
        ),
    )
    year: Optional[int] = Field(
        default=None,
        description=(
            "Year of first publication, from your knowledge. "
            "Null if unknown or uncertain."
        ),
    )


SYSTEM_PROMPT = """\
You are a literary historian identifying and classifying English texts.

Given a title and optionally the author's SURNAME ONLY, determine:

1. IS THIS FICTION? True if it is prose fiction (novel, romance, tale, novella, \
short story collection). False for everything else.

2. MAJOR GENRE: Use "fiction" for prose fiction. For non-fiction, choose: \
poetry, drama, periodical, essay, treatise, letters, sermon, biography, \
history, criticism, legal, speech, reference, almanac.

3. AUTHOR'S FIRST NAME: You are given only the surname. Use your knowledge of \
literary history to identify the author's first/given name. For example, if told \
"Pamela" by "Richardson" → "Samuel". If told "Tom Jones" by "Fielding" → "Henry". \
Return null only if you genuinely cannot identify who this author is.

4. YEAR: The year of first publication, from your knowledge. Return null if \
genuinely uncertain.

Many pre-1800 titles are long and descriptive. Use the full title for genre clues."""


EXAMPLES = [
    (
        "The History of Tom Jones, a Foundling by Fielding",
        MajorGenreAnnotation(
            is_fiction=True,
            major_genre='fiction',
            author_first_name='Henry',
            year=1749,
        ),
    ),
    (
        "A vindication of the rights of woman: with strictures on political "
        "and moral subjects by Wollstonecraft",
        MajorGenreAnnotation(
            is_fiction=False,
            major_genre='treatise',
            author_first_name='Mary',
            year=1792,
        ),
    ),
    (
        "The interesting narrative of the life of Olaudah Equiano, or Gustavus "
        "Vassa, the African by Equiano",
        MajorGenreAnnotation(
            is_fiction=False,
            major_genre='biography',
            author_first_name='Olaudah',
            year=1789,
        ),
    ),
]


DEFAULT_MODEL = 'claude-cli/sonnet'


class MajorGenreTask(Task):
    name = "classify_major_genre"
    model = DEFAULT_MODEL
    schema = MajorGenreAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.1
