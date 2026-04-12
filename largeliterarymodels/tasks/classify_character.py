"""Character introduction and description annotation: classify how named characters
are introduced and described in fiction, using BookNLP coreference to locate
first-mention passages.

Designed for the "Abstraction: A Literary History" project. Tests Ch 5's claim
that C18 novels introduce characters through abstract social/moral vocabulary
("youth, beauty, elegance") whereas romances use concrete emblematic description
and C19 novels use concrete physiognomic/material description.

Requires BookNLP output (text.entities, text.tokens, text.book) and
LLM-resolved characters (characters_resolved.json) in the text's booknlp dir.
"""

from pydantic import BaseModel, Field
from typing import Optional
from largeliterarymodels.task import Task
import pandas as pd
import json
import os


# ── Vocabularies ──────────────────────────────────────────────────────────

INTRO_MODE_VOCAB = [
    'abstract_social',       # character introduced through social qualities: reputation, elegance, taste, rank, wit, fortune
    'abstract_moral',        # character introduced through moral qualities: virtue, honour, goodness, piety, innocence
    'physical_appearance',   # character introduced through body, face, clothes, posture, physiognomy
    'physical_action',       # character introduced through what they do: arriving, fighting, fleeing
    'material_commodity',    # character introduced through possessions, property, dwelling, dress as social sign
    'emblematic_symbolic',   # character introduced through allegorical or symbolic attributes (Bunyan, romance)
    'behavioral',            # character introduced through speech, manner, habits observed in scene
    'relational',            # character introduced primarily through relationships (X's daughter, Y's rival)
    'mixed',                 # no single dominant mode
]

DESCRIPTOR_REGISTER_VOCAB = [
    'predominantly_abstract',   # descriptors are mostly abstract nouns/adjectives (beauty, wit, elegance, virtue)
    'predominantly_concrete',   # descriptors are mostly concrete/physical (tall, dark, scarred, richly dressed)
    'balanced',                 # roughly equal mix
    'abstract_punctuated',      # abstract dominant but with key concrete details
    'concrete_punctuated',      # concrete dominant but with key abstract evaluations
]

SOCIAL_LEGIBILITY_VOCAB = [
    'transparent',        # character's nature immediately legible from description (what you see = what they are)
    'opaque',             # character's nature withheld or ambiguous; description underdetermines character
    'ironic',             # description and actual character diverge (handsome villain, plain heroine)
    'performative',       # character actively constructs their social appearance (disguise, self-fashioning)
    'not_applicable',     # introduction too brief or indirect to assess
]


# ── Schema ────────────────────────────────────────────────────────────────

SOCIAL_CLASS_VOCAB = [
    'royalty_aristocracy',   # kings, queens, lords, dukes, counts
    'gentry',                # landed gentlemen, esquires, minor nobility
    'professional',          # clergy, lawyers, physicians, officers, merchants
    'middling',              # tradespeople, shopkeepers, clerks, farmers
    'servant_laborer',       # domestic servants, workers, apprentices, soldiers (rank and file)
    'vagrant_outcast',       # beggars, criminals, wanderers, exiles, orphans
    'unclear',               # social position not specified or ambiguous
]

CHARACTER_ROLE_VOCAB = [
    'protagonist',           # central character, narrative focus
    'love_interest',         # primary romantic object
    'antagonist',            # opposes protagonist's aims
    'authority_figure',      # parent, master, magistrate, employer — wields power over others
    'confidant',             # friend, advisor, correspondent — receives confidences
    'minor',                 # secondary or incidental character
]

INTERIORITY_VOCAB = [
    'interior_access',       # passage gives direct access to character's thoughts or feelings
    'external_only',         # character described only from outside (appearance, actions, reputation)
    'implied_interiority',   # inner life implied through behavior or free indirect discourse
]


class CharacterIntroAnnotation(BaseModel):
    character_name: str = Field(
        description="Name of the character being introduced, as given in the passages."
    )
    intro_passage_number: int = Field(
        description="Which passage number (1-indexed) contains the character's TRUE "
        "introduction — the first passage where the narrator or another character "
        "presents this character to the reader with descriptive or evaluative language. "
        "A passing mention ('my friend X arrived') is NOT an introduction. "
        "An introduction is where we first learn WHO this character IS."
    )
    character_gender: str = Field(
        description="Gender of the character as presented in the passages. "
        "One of: 'male', 'female', 'ambiguous'."
    )
    social_class: str = Field(
        description="Social class or rank of the character as presented. "
        f"One of: {', '.join(SOCIAL_CLASS_VOCAB)}. "
        "Base this on what the passages tell you, not on external knowledge."
    )
    character_role: str = Field(
        description="Narrative role of the character. "
        f"One of: {', '.join(CHARACTER_ROLE_VOCAB)}. "
        "Judge from context — who is this character to the story?"
    )
    interiority: str = Field(
        description="Does the introduction give access to the character's inner life? "
        f"One of: {', '.join(INTERIORITY_VOCAB)}. "
        "'interior_access' = we learn what they think or feel directly. "
        "'external_only' = described only from outside. "
        "'implied_interiority' = inner life suggested through behavior or indirect discourse."
    )
    intro_mode: str = Field(
        description="Primary mode of character introduction (in the intro passage you selected). "
        f"One of: {', '.join(INTRO_MODE_VOCAB)}. "
        "'abstract_social' = introduced through social qualities like rank, elegance, wit, fortune, "
        "reputation — qualities that exist in collective social judgment. "
        "'abstract_moral' = introduced through moral qualities like virtue, honour, piety — "
        "qualities posited as intrinsic rather than socially mediated. "
        "'physical_appearance' = introduced through body, face, clothes, complexion, stature. "
        "'physical_action' = introduced doing something concrete (arriving, fighting, working). "
        "'material_commodity' = introduced through possessions, property, furnishings as social signs. "
        "'emblematic_symbolic' = introduced through allegorical attributes (romance emblems, "
        "personification names, heraldic description). "
        "'behavioral' = introduced through observed speech and manner in a scene. "
        "'relational' = introduced primarily as someone's relation (daughter, servant, rival)."
    )
    intro_mode_secondary: str = Field(
        default="",
        description="Secondary introduction mode if the intro passage blends two. "
        f"One of: {', '.join(INTRO_MODE_VOCAB)}, or empty string."
    )
    descriptor_register: str = Field(
        description="Abstract/concrete register of the descriptive vocabulary in the intro passage. "
        f"One of: {', '.join(DESCRIPTOR_REGISTER_VOCAB)}. "
        "Consider whether the character is described with abstract nouns (beauty, elegance, wit, "
        "virtue, honour, fortune) or concrete nouns (hair, eyes, dress, sword, horse, house). "
        "'abstract_punctuated' if mostly abstract with one or two telling concrete details."
    )
    social_legibility: str = Field(
        description="Is the character's nature immediately legible from the introduction? "
        f"One of: {', '.join(SOCIAL_LEGIBILITY_VOCAB)}. "
        "'transparent' = description makes character's nature clear (the 'knowable community'). "
        "'opaque' = character remains mysterious or ambiguous after introduction. "
        "'ironic' = appearance and reality diverge (narrator signals this). "
        "'performative' = character is actively managing their social appearance."
    )
    key_descriptors: list[str] = Field(
        default=[],
        description="Up to 8 key words or short phrases used to describe the character "
        "in the intro passage. Include the most characteristic descriptors, whether "
        "abstract (elegant, virtuous, spirited) or concrete (tall, dark-haired, richly "
        "dressed). Quote directly from text."
    )
    is_narrator_assessment: bool = Field(
        description="True if the introduction comes from the narrator's evaluative voice "
        "(telling), False if the character is revealed through dramatized action/speech "
        "(showing) or through another character's perspective."
    )
    describing_voice: str = Field(
        description="Who produces the description? "
        "'narrator' = third-person narrator or first-person narrator describing others. "
        "'self' = character describing themselves (epistolary, diary, confession). "
        "'other_character' = another character describes them (in dialogue or letter). "
        "'collective' = described by general social opinion ('everyone agreed she was...')."
    )
    passage_summary: str = Field(
        description="1-2 sentence summary of the introduction passage you selected. "
        "Focus on how the character enters the narrative and what we learn about them."
    )
    confidence: float = Field(
        description="Overall confidence in the annotation, 0.0 to 1.0."
    )
    reasoning: str = Field(
        description="1-3 sentences explaining your choice of intro passage and the "
        "classification. Quote a key phrase from the intro passage."
    )


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a literary critic annotating character introductions in English-language fiction (1500–2020) for a computational study of abstract and concrete language in the history of the novel.

You will receive 3-5 passages from a novel, each ~250 words, showing the first several appearances of a named character. The character's name is given. Your tasks:

1. **Identify which passage is the TRUE INTRODUCTION** — the first passage where the narrator or another character *presents* this character to the reader with descriptive or evaluative language. A passing mention ("my friend X arrived") is NOT an introduction. The introduction is where we first learn WHO this character IS — their qualities, appearance, social position, or moral nature.

2. **Classify that introduction passage** using the schema below.

No title, author, or date is provided — classify based solely on the passage text.

## Theoretical framework

This annotation supports research into how character description changes across literary history:

### Romance (C16-C17): Emblematic/concrete
Characters in romance are introduced through concrete emblems and physical prowess: shields, armor, beauty "like a flower," strength proven in combat. Abstract qualities (honour, virtue) must be proven through concrete deeds. Description is allegorical: things encode ideas.

### 18C novel: Abstract/social
Characters in the C18 novel are introduced through ABSTRACT social and moral vocabulary. Austen's Willoughby: "His manners had all the elegance... his taste delicate... a want of caution which Elinor could not approve." Virtually no physical description — the character exists as a bundle of social abstractions. This is "abstract realism": characters are real because socially legible, not because physically described.

### 19C novel: Concrete/commodity
Characters in the C19 novel are introduced through CONCRETE physical description that encodes social meaning: Dickens' characters readable through bodies, clothes, dirt. Brontë's "nostrils denoting choler." Things speak again — but now as commodities to be decoded, not as symbols to be recognized.

## Classification principles

- Base your classification on what is ACTUALLY in the passage, not on what you know about the author's general style.
- The key question is: does the passage tell us what the character IS (abstract) or what they LOOK LIKE (concrete)?
- Austen can be concrete; Dickens can be abstract. Classify the passage, not the author.
"""


# ── Few-shot examples ─────────────────────────────────────────────────────

EXAMPLES = [
    # 1. Austen — abstract social introduction (multi-passage)
    (
        "Character to classify: Willoughby\n\n"
        "PASSAGE 1:\n"
        "A gentleman carrying a gun, with two pointers playing round him, was passing "
        "up the hill and within a few yards of Marianne, when her accident happened. "
        "He put down his gun and ran to her assistance. She had raised herself from the "
        "ground, but her foot had been twisted in the fall, and she was scarcely able to "
        "stand. The gentleman offered his services, and perceiving that her modesty "
        "declined what her situation rendered necessary, took her up in his arms without "
        "farther delay, and carried her down the hill.\n\n"
        "PASSAGE 2:\n"
        "His person and air were equal to what her fancy had ever drawn for the hero "
        "of a favourite story; and in his carrying her into the house with so little "
        "previous formality, there was a rapidity of thought which particularly "
        "recommended him to her. Every circumstance belonging to him was interesting. "
        "His name was good, his residence was in their favourite village, and she soon "
        "found out that of all manly dresses a shooting-jacket was the most becoming. "
        "His manners had all the elegance which his own could not have wanted. His "
        "understanding was beyond all doubt acute, his taste delicate. "
        "In hastily forming and giving his opinion of other people, in "
        "sacrificing general politeness to the enjoyment of undivided attention where "
        "his heart was engaged, and in slighting too easily the forms of worldly "
        "propriety, he displayed a want of caution which Elinor could not approve.\n\n"
        "PASSAGE 3:\n"
        "Willoughby called at the cottage early the next morning to make his personal "
        "inquiries. He was received by Mrs. Dashwood with more than politeness — with a "
        "kindness which Sir John's account of him and her own gratitude prompted.",
        CharacterIntroAnnotation(
            character_name="Willoughby",
            intro_passage_number=2,
            character_gender="male",
            social_class="gentry",
            character_role="love_interest",
            interiority="external_only",
            intro_mode="abstract_social",
            intro_mode_secondary="",
            descriptor_register="predominantly_abstract",
            social_legibility="transparent",
            key_descriptors=["elegance", "understanding", "taste", "rapidity of thought",
                             "want of caution", "politeness", "propriety", "shooting-jacket"],
            is_narrator_assessment=True,
            describing_voice="narrator",
            passage_summary="Willoughby is introduced after carrying Marianne into the house; "
            "his character is assessed entirely through abstract social qualities (elegance, "
            "taste, propriety) with the shooting-jacket as the sole concrete detail.",
            confidence=0.95,
            reasoning="Passage 1 is a passing encounter — 'a gentleman carrying a gun' is "
            "anonymous action, not an introduction. Passage 2 is the true introduction: "
            "Willoughby becomes a bundle of social abstractions — 'elegance,' 'understanding,' "
            "'taste,' 'propriety.' The one concrete detail ('shooting-jacket') is immediately "
            "absorbed into abstract evaluation ('the most becoming').",
        ),
    ),
    # 2. Dickens — concrete physiognomic introduction
    (
        "Character to classify: Krook\n\n"
        "PASSAGE 1:\n"
        "He was a small man with a head sunk sideways between his shoulders, and "
        "with such a quantity of what he called his 'property' heaped around him that "
        "he seemed a mere bundle of old rags, bones, and odds-and-ends. His shop was "
        "blinded by cobwebs, heaped with bottles, bones, scraps of old leather, "
        "horsehair, rags, and waste. A cat, with a tail like a worn-out hearth-broom, "
        "sat upon a shelf over the doorway. The old man's breath issued from him like "
        "smoke from a damp chimney. He was short, cadaverous, and withered; with his "
        "head sunk sideways between his shoulders, and the breath issuing in visible "
        "smoke from his mouth, as if he were on fire within. His throat, chin, and "
        "eyebrows were so frosted with white hairs, and so gnarled with veins and "
        "puckered skin, that he looked from his breast upward, like some old root "
        "in a fall of snow.",
        CharacterIntroAnnotation(
            character_name="Krook",
            intro_passage_number=1,
            character_gender="male",
            social_class="middling",
            character_role="minor",
            interiority="external_only",
            intro_mode="material_commodity",
            intro_mode_secondary="physical_appearance",
            descriptor_register="predominantly_concrete",
            social_legibility="transparent",
            key_descriptors=["small", "head sunk sideways", "bundle of old rags",
                             "cadaverous", "withered", "like some old root",
                             "frosted with white hairs", "property"],
            is_narrator_assessment=True,
            describing_voice="narrator",
            passage_summary="Krook is introduced through a detailed inventory of his body and "
            "shop, both composed of the same decaying materials — rags, bones, cobwebs. "
            "His physical person encodes his social meaning.",
            confidence=0.95,
            reasoning="Allegory of the commodity: Krook's body and possessions are "
            "indistinguishable, both 'heaps of bones, rags, odds-and-ends.' His moral character "
            "is readable from his physical appearance — 'cadaverous,' 'like some old root.' "
            "The sole abstraction, 'property,' is ironic.",
        ),
    ),
    # 3. Behn — abstract moral + relational
    (
        "Character to classify: Atlante and Charlot\n\n"
        "PASSAGE 1:\n"
        "This old Count had two only Daughters, of exceeding Beauty, who gave the "
        "Generous Father ten thousand Torments, as often as he beheld them, when he "
        "consider'd their Extream Beauty, their fine Wit, their Innocence, Modesty, "
        "and above all, their Birth; and that he had not the Fortune to marry them "
        "according to their Quality; and below it he had rather see 'em laid in "
        "their silent Graves, than consent to it. Often when he was alone, he has been "
        "heard to cry out, Oh cruel Fortune! that would give me Quality and Beauty, "
        "without the common Blessing of the World, Fortune.",
        CharacterIntroAnnotation(
            character_name="Atlante and Charlot",
            intro_passage_number=1,
            character_gender="female",
            social_class="royalty_aristocracy",
            character_role="protagonist",
            interiority="external_only",
            intro_mode="abstract_moral",
            intro_mode_secondary="relational",
            descriptor_register="predominantly_abstract",
            social_legibility="transparent",
            key_descriptors=["exceeding Beauty", "fine Wit", "Innocence", "Modesty",
                             "Birth", "Quality", "Fortune"],
            is_narrator_assessment=True,
            describing_voice="narrator",
            passage_summary="The Count's two daughters are introduced through a catalogue of "
            "abstract qualities (Beauty, Wit, Innocence, Modesty, Birth, Quality) set against "
            "the father's anguish over their lack of Fortune.",
            confidence=0.9,
            reasoning="Pure abstract moral introduction: the daughters are a list of capitalized "
            "abstractions. No physical detail — 'Beauty' is abstract quality, not described "
            "appearance. The passage's drama is entirely social: Quality without Fortune.",
        ),
    ),
]


# ── Task class ────────────────────────────────────────────────────────────

class CharacterIntroTask(Task):
    """Classify how a named character is introduced in fiction.

    Uses BookNLP coreference to locate the first-mention passage and
    provides the character's full-text descriptor profile as context.

    Usage:
        task = CharacterIntroTask()
        prompt, meta = format_character_intro(booknlp_dir, char_info, text_meta)
        result = task.run(prompt, metadata=meta)
    """
    name = "classify_character_intro"
    schema = CharacterIntroAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.2
    max_tokens = 8192


# ── Helper functions ──────────────────────────────────────────────────────

GENERIC_PREFIXES = ('the ', 'a ', 'an ', 'his ', 'her ', 'my ', 'our ', 'any ',
                    'some ', 'this ', 'that ', 'unnamed', 'unknown', 'generic')


def is_named_character(char_info):
    """Check if a resolved character is a named individual (not generic/collective)."""
    if char_info.get('type') not in ('character',):
        return False
    name = char_info.get('name', '').strip()
    if not name or not name[0].isupper():
        return False
    if any(name.lower().startswith(p) for p in GENERIC_PREFIXES):
        return False
    return True


def get_mention_passages(booknlp_dir, char_info, n_passages=5, window=125):
    """Find the first N mention passages of a character.

    Strategy: find proper noun mentions matching the character's name,
    sorted by position. Extract a ~250-word window around each.
    Non-overlapping: if two mentions fall within the same window, only
    the first is kept.

    Args:
        booknlp_dir: Path to booknlp output directory.
        char_info: Resolved character dict with 'ids', 'name', etc.
        n_passages: Maximum number of passages to return.
        window: Words each side of mention to extract (~250 words total).

    Returns:
        list of (passage_text, token_idx, mention_text) tuples, or empty list.
    """
    try:
        ents = pd.read_csv(os.path.join(booknlp_dir, 'text.entities'), sep='\t',
                           on_bad_lines='skip', engine='python')
        tokens = pd.read_csv(os.path.join(booknlp_dir, 'text.tokens'), sep='\t',
                             on_bad_lines='skip', engine='python')
    except Exception:
        return []

    cluster_ids = set()
    for cid_str in char_info.get('ids', []):
        try:
            cluster_ids.add(int(cid_str[1:]))
        except (ValueError, IndexError):
            pass

    if not cluster_ids:
        return []

    char_ents = ents[ents['COREF'].isin(cluster_ids)]
    if len(char_ents) == 0:
        return []

    # Build name tokens for matching
    char_name = char_info.get('name', '')
    name_tokens = set(char_name.lower().split())
    name_tokens -= {'sir', 'lady', 'lord', 'count', 'countess', 'mr', 'mrs',
                    'miss', 'dr', 'captain', 'colonel', 'duke', 'prince',
                    'princess', 'king', 'queen', 'monsieur', 'madame', 'de'}

    # Find matching proper noun mentions, sorted by position
    candidates = []

    if name_tokens:
        prop_ents = char_ents[char_ents['prop'] == 'PROP'].copy()
        if len(prop_ents):
            def matches_name(mention_text):
                return any(nt in str(mention_text).lower() for nt in name_tokens)
            name_matches = prop_ents[prop_ents['text'].apply(matches_name)]
            candidates = name_matches.sort_values('start_token')

    # Fallback: all proper mentions
    if len(candidates) == 0:
        prop_ents = char_ents[char_ents['prop'] == 'PROP'].sort_values('start_token')
        candidates = prop_ents

    # Fallback: any mentions
    if len(candidates) == 0:
        candidates = char_ents.sort_values('start_token')

    if len(candidates) == 0:
        return []

    # Select non-overlapping passages
    results = []
    last_end = -1
    for _, row in candidates.iterrows():
        tok = int(row['start_token'])
        if tok < last_end + window:  # skip if overlaps previous window
            continue
        passage = _extract_passage(tokens, tok, window)
        results.append((passage, tok, str(row['text'])))
        last_end = tok + window
        if len(results) >= n_passages:
            break

    return results


def _extract_passage(tokens, center_tok, window):
    """Extract passage text centered on a token position."""
    start = max(0, center_tok - window)
    end = min(len(tokens), center_tok + window)
    ptoks = tokens[(tokens['token_ID_within_document'] >= start) &
                   (tokens['token_ID_within_document'] <= end)]
    return ' '.join(ptoks['word'].astype(str).tolist())


def get_booknlp_descriptors(booknlp_dir, char_info, top_n=15):
    """Get BookNLP modifier/agent/patient/poss words for a character.

    Returns dict with 'mod', 'agent', 'patient', 'poss' keys,
    each a list of (word, count) tuples.
    """
    with open(os.path.join(booknlp_dir, 'text.book')) as f:
        raw_chars = json.load(f)['characters']

    cluster_ids = set()
    for cid_str in char_info.get('ids', []):
        try:
            cluster_ids.add(int(cid_str[1:]))
        except (ValueError, IndexError):
            pass

    result = {'mod': [], 'agent': [], 'patient': [], 'poss': []}
    for rc in raw_chars:
        if rc['id'] not in cluster_ids:
            continue
        for role in ['mod', 'agent', 'patient', 'poss']:
            for item in rc.get(role, []):
                result[role].append(item['w'])

    # Count and sort
    from collections import Counter
    for role in result:
        counts = Counter(result[role]).most_common(top_n)
        result[role] = counts

    return result


def format_character_intro(booknlp_dir, char_info, title=None, author=None,
                           year=None, _id=None, n_passages=5, window=125):
    """Format character mention passages as a multi-passage prompt.

    Sends the first N non-overlapping passages where the character is
    mentioned by name. The LLM identifies which is the true introduction
    and classifies it. No title/author/year shown to LLM.

    Args:
        booknlp_dir: Path to booknlp output directory.
        char_info: Resolved character dict.
        title, author, year: Text metadata (stored in meta, not shown to LLM).
        _id: LLTK text _id.
        n_passages: Number of mention passages to include (default 5).
        window: Words each side of mention center (default 125 = ~250 words/passage).

    Returns:
        tuple: (prompt_string, metadata_dict) or (None, None) if no passages found.
    """
    passages = get_mention_passages(
        booknlp_dir, char_info, n_passages=n_passages, window=window)

    if not passages:
        return None, None

    # Build prompt with numbered passages
    parts = [f"Character to classify: {char_info['name']}"]
    total_words = 0
    for i, (passage_text, tok, mention) in enumerate(passages, 1):
        parts.append(f"\nPASSAGE {i}:\n{passage_text}")
        total_words += len(passage_text.split())

    prompt = '\n'.join(parts)

    first_tok = passages[0][1]
    meta = {
        '_id': _id or '',
        'title': title or '',
        'author': author or '',
        'char_name': char_info['name'],
        'char_gender': char_info.get('gender', ''),
        'char_type': char_info.get('type', ''),
        'first_mention_token': first_tok,
        'n_passages_shown': len(passages),
        'mention_tokens': [tok for _, tok, _ in passages],
        'year': int(year) if year is not None else None,
        'n_words': total_words,
    }

    return prompt, meta


def format_all_character_intros(booknlp_dir, title=None, author=None,
                                year=None, _id=None, n_passages=5, window=125):
    """Generate prompts for all named characters in a text.

    Returns:
        list[tuple[str, dict]]: List of (prompt, metadata) tuples.
    """
    resolved_path = os.path.join(booknlp_dir, 'characters_resolved.json')
    if not os.path.exists(resolved_path):
        return []

    with open(resolved_path) as f:
        data = json.load(f)
    chars = data.get('characters', []) if isinstance(data, dict) else data

    results = []
    for c in chars:
        if not is_named_character(c):
            continue
        try:
            prompt, meta = format_character_intro(
                booknlp_dir, c, title=title, author=author,
                year=year, _id=_id, n_passages=n_passages, window=window)
            if prompt is not None:
                results.append((prompt, meta))
        except Exception:
            continue

    return results
