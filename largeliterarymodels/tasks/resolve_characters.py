"""Character resolution task: clean up BookNLP character clusters using LLM."""

from pydantic import BaseModel, Field
from typing import Optional
from largeliterarymodels.task import Task


class CharacterResolution(BaseModel):
    name: str = Field(
        description="Canonical name for this character or entity. "
        "Use the most common proper name form (e.g. 'Jason' not 'jason'). "
        "For unnamed characters, use their most specific common noun (e.g. 'the queen'). "
        "For noise clusters, use the misidentified word (e.g. 'Certes')."
    )
    ids: list[str] = Field(
        description="List of BookNLP cluster IDs that refer to this same entity. "
        "Merge fragmented clusters here (e.g. ['C085', 'C1603', 'C3821'] if all are Jason)."
    )
    type: str = Field(
        description="One of: character, collective, place, abstraction, noise. "
        "'character' = individual person (real or fictional). "
        "'collective' = group (e.g. 'the ladies', 'his men', 'the gods'). "
        "'place' = geographic entity misidentified as person. "
        "'abstraction' = abstract concept (e.g. 'Fortune', 'Love'). "
        "'noise' = discourse marker, sentence-initial word, or parsing artifact "
        "(e.g. 'Certes', 'Thenne', 'Incontinent')."
    )
    gender: str = Field(
        default="",
        description="male, female, mixed, or unknown. "
        "For collectives, use 'mixed' unless clearly gendered (e.g. 'the ladies' = female)."
    )
    notes: str = Field(
        default="",
        description="Brief note on identification, especially for uncertain merges "
        "or characters you recognize from literary history."
    )


SYSTEM_PROMPT = """You are an expert in early modern English literature (1475-1800) with deep knowledge of literary characters, naming conventions, and the quirks of early modern spelling and typography.

You are resolving character clusters extracted by BookNLP from early modern texts. BookNLP's NER and coreference resolution is noisy on these texts because:
- Early modern spelling varies (e.g. "jason"/"Jason"/"Iason")
- Sentence-initial capitalized words get misidentified as proper nouns (e.g. "Certes" = certainly, "Thenne" = then, "Incontinent" = immediately)
- One character often gets split across multiple clusters (e.g. "Medea" in one cluster, "the noble lady Medea" in another)
- Generic noun phrases like "the king" or "his son" may refer to the same character throughout, or to different characters in different episodes

Your job is to produce a clean character list by:
1. Identifying noise clusters (discourse markers, places, abstractions)
2. Merging clusters that refer to the same character
3. Giving each real character a canonical name

When merging, use the evidence available: if two clusters share proper names, or a common noun cluster ("the noble knight") clearly refers to a named cluster ("Jason") based on the verb profiles and gender, merge them. When uncertain, keep clusters separate rather than guessing.

For well-known literary works, use your knowledge of the characters to inform merges. But mark your confidence in the notes field."""


EXAMPLES = [
    # Example with a clear merge + noise detection
    (
        """Text: "The Life and Strange Surprizing Adventures of Robinson Crusoe" (1719) by Defoe. Genre: Fiction.

CLUSTERS:
C001 (850x) proper=['Crusoe', 'Robinson'] common=[] gender=he/him/his does=['found', 'made', 'saw', 'went', 'took', 'came'] done_to=['saved', 'brought', 'told', 'carried']
C045 (320x) proper=[] common=['my man Friday', 'Friday'] gender=he/him/his does=['came', 'ran', 'said', 'brought', 'killed'] done_to=['taught', 'saved', 'called', 'bid']
C102 (200x) proper=['Friday'] common=[] gender=he/him/his does=['learned', 'followed', 'spoke', 'understood'] done_to=['instructed', 'named', 'sent']
C203 (150x) proper=['Certes'] common=[] gender=he/him/his does=['made', 'found'] done_to=[]
C088 (80x) proper=[] common=['the savages', 'the Savages'] gender=they/them/their does=['came', 'landed', 'brought', 'killed'] done_to=['defeated', 'fled', 'surprised']
C099 (40x) proper=['Providence'] common=[] gender=he/him/his does=['directed', 'sent', 'preserved'] done_to=['thanked', 'praised']""",
        [
            CharacterResolution(
                name="Robinson Crusoe",
                ids=["C001"],
                type="character",
                gender="male",
                notes="Protagonist and narrator.",
            ),
            CharacterResolution(
                name="Friday",
                ids=["C045", "C102"],
                type="character",
                gender="male",
                notes="C045 and C102 both refer to Friday — split between common noun and proper name mentions.",
            ),
            CharacterResolution(
                name="Certes",
                ids=["C203"],
                type="noise",
                gender="",
                notes="'Certes' = 'certainly', a discourse marker misidentified as a proper noun.",
            ),
            CharacterResolution(
                name="the savages",
                ids=["C088"],
                type="collective",
                gender="mixed",
                notes="Generic collective, likely refers to multiple groups across different episodes.",
            ),
            CharacterResolution(
                name="Providence",
                ids=["C099"],
                type="abstraction",
                gender="",
                notes="Divine providence personified in Crusoe's narration, not a character.",
            ),
        ],
    ),
]


class CharacterTask(Task):
    name = "resolve_characters"
    schema = list[CharacterResolution]
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.2
    max_tokens = 16384


def format_character_roster(text, min_count=10, max_chars=50):
    """Build a character roster prompt from a BookNLP-parsed lltk text.

    Args:
        text: An lltk text object with .booknlp already parsed.
        min_count: Minimum mention count to include a cluster.
        max_chars: Maximum number of clusters to include.

    Returns:
        str: Formatted prompt ready for CharacterTask.run().
    """
    import json, os

    bnlp = text.booknlp
    book_path = bnlp.paths['chardata']
    if not os.path.exists(book_path):
        raise FileNotFoundError(f"BookNLP not yet parsed for {text.addr}. Run text.booknlp.parse() first.")

    with open(book_path) as f:
        book = json.load(f)

    chars = [c for c in book['characters'] if c['count'] >= min_count and c['id'] != 0]
    chars.sort(key=lambda c: -c['count'])
    chars = chars[:max_chars]

    lines = []
    for c in chars:
        cid = f"C{c['id']:03d}"
        proper = [m['n'] for m in c['mentions']['proper'][:5]]
        common = [m['n'] for m in c['mentions']['common'][:5]]
        g = c.get('g')
        gender = g.get('argmax', '?') if g else '?'
        agent = [a['w'] for a in c['agent'][:6]]
        patient = [a['w'] for a in c['patient'][:6]]
        lines.append(
            f"{cid} ({c['count']}x) proper={proper} common={common} "
            f"gender={gender} does={agent} done_to={patient}"
        )

    meta = text.meta if hasattr(text, 'meta') else {}
    title = getattr(text, 'title', '') or meta.get('title', '')
    author = getattr(text, 'author', '') or meta.get('author', '')
    year = getattr(text, 'year', '') or meta.get('year', '')
    genre = meta.get('genre', '') if isinstance(meta, dict) else getattr(text, 'genre', '')

    header = f'Text: "{title}" ({year})'
    if author:
        header += f" by {author}"
    if genre:
        header += f". Genre: {genre}"
    header += "."

    prompt = f"""{header}

CLUSTERS:
"""
    prompt += "\n".join(lines)
    return prompt
