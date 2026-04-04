"""Bibliography extraction task: parse messy OCR/HTML bibliography entries into structured data."""

from pydantic import BaseModel, Field
from largeliterarymodels.task import Task


class BibliographyEntry(BaseModel):
    author: str = Field(default="", description="Author name (e.g. 'Greene, Robert' from 'GREENE, ROBERT')")
    title: str = Field(description="Main title only, before any subtitle (e.g. 'Panthalia' from 'Panthalia: or The Royal romance')")
    title_sub: str = Field(default="", description="Subtitle if any, including leading punctuation (e.g. ': or The Royal Romance')")
    year: int = Field(description="Year of publication")
    edition: str = Field(default="", description="Edition statement if given (e.g. 'Third edition', 'Sixth impression')")
    id_biblio: str = Field(default="", description="Bibliographic ID if stated (e.g. 'STC 11502', 'Wing U132')")
    is_translated: bool = Field(default=False, description="True if the work is a translation — look for 'Englished by', 'translated by', 'rendred into English', etc.")
    translated_from: str = Field(default="", description="Source language if stated (e.g. 'Latin', 'French')")
    translator: str = Field(default="", description="Translator name if stated")
    printer: str = Field(default="", description="Printer, from 'Printed by X' or 'X [printer]' (e.g. 'T. Creede')")
    publisher: str = Field(default="", description="Publisher, from 'for X' (e.g. 'N. Ling')")
    bookseller: str = Field(default="", description="Bookseller, from 'sold by X' (e.g. 'E. White')")
    notes_biblio: str = Field(default="", description="Other bibliographic notes from the entry (holding libraries, earlier editions, provenance)")
    notes: str = Field(default="", description="Any additional observations about the entry")
    uncertain_author: bool = Field(default=False, description="True if you are uncertain about whether the uppercased text is an author name or the main title")


SYSTEM_PROMPT = """You are an expert bibliographer specializing in early modern English print culture (1500-1800).

You are parsing entries from a messy OCR/HTML bibliography of English prose fiction. Each entry typically contains:
- An author name (often in CAPS)
- A title (sometimes with subtitle after colon or comma)
- Imprint information: printer ("Printed by X" or just "X"), publisher ("for Y"), bookseller ("sold by Z")
- A parenthetical note with bibliographic IDs (STC, Wing), edition info, and holding library references
- There may be OCR errors in the text. Please clean these up as best you can.
- The HTML may be slightly malformed. Use your best judgment. The better way to separate entries is by a new author name rather than the HTML formatting.

Key conventions:
- "T. Creede" after a title = printer. "for N. Ling" = publisher. "sold by E. White" = bookseller.
- Edition may appear as ordinal ("Third edition") or as impression ("sixth impression").
- Translation clues: "Englished by", "translated by", "rendred into English", or the phrase "intituled in Latin" in the title.
- The year usually comes from a heading above a group of entries, not from within each entry.
- Bibliographic IDs like "STC 12254" or "Wing U132" appear in parenthetical notes.
- References to libraries (Folger, Huntington, Bodleian, etc.) and to other scholars (Esdaile) go in notes_biblio.
- OCR artifacts are common: "1G08" = "1608", "i6o2" = "1602", etc. Clean these up.
- Very important: in some cases what appears to be an author name (uppercased) is in fact the main title, with the subsequent text the subtitle; in this case, the author is unknown, and you should set author to an empty string. Please use your judgment to decide if this is the case. Hint: if the uppercased text is repeated in the subsequent text/title, the uppercased text is probably the main title. If you are uncertain, set uncertain_author to True.

"""


EXAMPLES = [
    # Example 1: simple entry with printer and publisher
    (
        """<h2 style="font-weight:normal;"><span class="font5">1600</span></h2>
<p><span class="font5">GREENE, ROBERT. Greenes Never too late. J. Roberts for N. Ling. (STC 12254. Second edition; first edition in 1590.)</span></p>""",
        BibliographyEntry(
            author="Greene, Robert",
            title="Greenes Never too late",
            title_sub="",
            year=1600,
            edition="Second edition",
            id_biblio="STC 12254",
            is_translated=False,
            translated_from="",
            translator="",
            printer="J. Roberts",
            publisher="N. Ling",
            bookseller="",
            notes_biblio="First edition in 1590.",
            notes="",
            uncertain_author=False,
        ),
    ),
    # Example 2: translated work with subtitle
    (
        """<h2 style="font-weight:normal;"><span class="font5">1601</span></h2>
<p><span class="font5">BIDPAI. The morall philosophic of Doni: drawne out of the ancient writers. Englished by Sir Thomas North. S. Stafford. (STC 3054. Second edition; first edition in 1570.)</span></p>""",
        BibliographyEntry(
            author="Bidpai",
            title="The morall philosophic of Doni",
            title_sub=": drawne out of the ancient writers. Englished by Sir Thomas North.",
            year=1601,
            edition="Second edition",
            id_biblio="STC 3054",
            is_translated=True,
            translated_from="",
            translator="Sir Thomas North",
            printer="S. Stafford",
            publisher="",
            bookseller="",
            notes_biblio="First edition in 1570.",
            notes="",
            uncertain_author=True,
        ),
    ),
    # Example 3: printer + bookseller, translation, complex notes
    (
        """<h2 style="font-weight:normal;"><span class="font5">1601</span></h2>
<p><span class="font5">HUON OF BORDEAUX. The ancient, honorable, famous, and delight-ful historic of Huon of Bourdeaux. Being the third time imprinted. T. Purfoot, sould by E. White. (STC 13999. Third edition; first edition ca.1534, second edition lost. Translated by Lord Berners.)</span></p>""",
        BibliographyEntry(
            author="",
            title="Huon of Bordeaux",
            title_sub=". The ancient, honorable, famous, and delight-ful historic of Huon of Bourdeaux. Being the third time imprinted.",
            year=1601,
            edition="Third edition",
            id_biblio="STC 13999",
            is_translated=True,
            translated_from="",
            translator="Lord Berners",
            printer="T. Purfoot",
            publisher="",
            bookseller="E. White",
            notes_biblio="First edition ca.1534, second edition lost.",
            notes="",
            uncertain_author=False,
        ),
    ),
]


def chunk_bibliography(html, max_entries=20):
    """Split bibliography HTML into chunks, respecting entry boundaries.

    Splits on year headings first (<h2>), then splits large year-sections
    into smaller chunks of at most max_entries paragraph-level entries.
    Each sub-chunk retains the year heading so the LLM knows the date.

    Args:
        html: Raw HTML string of the bibliography.
        max_entries: Maximum number of <p> entries per chunk (default 20).

    Returns:
        list[str]: Chunks of HTML, each small enough for one LLM call.
    """
    import re

    # Split on year headings, keeping the heading with its entries
    year_chunks = re.split(r'(?=<h2[^>]*>)', html)
    year_chunks = [c.strip() for c in year_chunks if c.strip()]

    result = []
    for chunk in year_chunks:
        # Extract the heading
        heading_match = re.match(r'(<h2.*?</h2>)', chunk, re.DOTALL)
        if not heading_match:
            result.append(chunk)
            continue

        heading = heading_match.group(1)
        body = chunk[heading_match.end():].strip()

        # Split body into individual entry blocks (on <p> tags)
        entries = re.split(r'(?=<p[ >])', body)
        entries = [e.strip() for e in entries if e.strip()]

        if len(entries) <= max_entries:
            result.append(chunk)
        else:
            # Split into sub-chunks, each prefixed with the year heading
            for i in range(0, len(entries), max_entries):
                batch = entries[i:i + max_entries]
                result.append(heading + "\n" + "\n".join(batch))

    return result


class BibliographyTask(Task):
    schema = list[BibliographyEntry]
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    max_tokens = 16384
