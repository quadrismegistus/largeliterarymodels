"""Frye mode/mythos classification: classify narrative texts using Northrop Frye's
categories from Anatomy of Criticism (1957), based on textual evidence.

Unlike genre classification (which uses metadata only), this task reads actual
text passages — opening, middle, and closing — to assess narrative mode and mythos.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from largeliterarymodels.task import Task


# ── Frye's five modes (First Essay: Historical Criticism) ──────────────
# Defined by the hero's power relative to other people and the environment.

MODE_VOCAB = ['myth', 'romance', 'high_mimetic', 'low_mimetic', 'ironic']

MODE_DESCRIPTIONS = {
    'myth': 'Hero is divine or supernatural, superior in kind to other people and nature. Gods, creation, metamorphosis.',
    'romance': 'Hero is superior in degree to other people and nature. Marvellous exploits, enchanted settings, quests, monsters.',
    'high_mimetic': 'Hero is superior in degree to other people but not nature. Leaders, kings, tragic heroes. Epic and tragedy.',
    'low_mimetic': 'Hero is one of us — no special authority or power. Ordinary social world. Realist fiction, comedy of manners.',
    'ironic': 'Hero is inferior to us in power or intelligence. We look down on scenes of bondage, frustration, absurdity.',
}

# ── Frye's four mythoi (Third Essay: Archetypal Criticism) ─────────────
# The four narrative archetypes, corresponding to seasons.

MYTHOS_VOCAB = ['comedy', 'romance', 'tragedy', 'irony']

MYTHOS_DESCRIPTIONS = {
    'comedy': 'Spring. Movement from confusion to order, from bondage to freedom. Society reconstituted, often via marriage. The blocking characters are overcome or absorbed.',
    'romance': 'Summer. The quest: agon (conflict), pathos (death-struggle), anagnorisis (recognition/triumph). Good vs evil, with the hero triumphant.',
    'tragedy': 'Autumn. The hero falls from prosperity. Hamartia, isolation, catastrophe. The hero is expelled from the social order.',
    'irony': 'Winter. The mythical patterns of tragedy and comedy are present but attenuated, mocked, or inverted. Anti-romance. Pharmakos (scapegoat) figures.',
}

# ── Narrative voice/perspective (complements mode) ─────────────────────

NARRATION_VOCAB = [
    'first_person', 'third_person_omniscient', 'third_person_limited',
    'epistolary', 'frame_narrative', 'dialogue', 'second_person', 'mixed',
]

# ── Schema ─────────────────────────────────────────────────────────────

class FryeClassification(BaseModel):
    mode: str = Field(
        description="Frye's narrative mode based on the hero's power relative to others and environment. "
        f"One of: {', '.join(MODE_VOCAB)}. "
        "myth = divine hero; romance = marvellous hero; high_mimetic = leader/tragic hero; "
        "low_mimetic = ordinary person; ironic = hero inferior to reader."
    )
    mode_confidence: float = Field(
        description="Confidence in mode classification, 0.0 to 1.0."
    )
    mythos: str = Field(
        description="Frye's narrative archetype / overall plot shape. "
        f"One of: {', '.join(MYTHOS_VOCAB)}. "
        "comedy = movement toward integration and social harmony; "
        "romance = quest with triumph; "
        "tragedy = fall from prosperity, isolation; "
        "irony = patterns inverted or attenuated, anti-romance."
    )
    mythos_confidence: float = Field(
        description="Confidence in mythos classification, 0.0 to 1.0. "
        "Lower if only opening/middle available and ending unclear."
    )
    narration: str = Field(
        description=f"Narrative voice. One of: {', '.join(NARRATION_VOCAB)}."
    )
    mode_signals: str = Field(
        description="Brief evidence for the mode classification: what specific language, "
        "imagery, or narrative posture in the passages indicates this mode? 1-3 sentences. "
        "Quote specific phrases where possible."
    )
    mythos_signals: str = Field(
        description="Brief evidence for the mythos classification: what plot trajectory, "
        "character dynamics, or narrative movement is visible? 1-3 sentences."
    )
    referential_mode: str = Field(
        description="The text's relationship to real-world reference (Gallagher/Paige). "
        "One of: 'nobody' (explicitly fictional characters — the novel proper), "
        "'somebody' (claims to be about real people — secret histories, scandal narratives, memoirs of real persons), "
        "'pseudo_referential' (invented story claiming to be true — found manuscripts, 'authentic memoirs' of fictional people, travel narratives to nonexistent places), "
        "'ambiguous' (genuinely unclear whether characters are real or invented)."
    )
    referential_signals: str = Field(
        description="Evidence for the referential mode. What textual markers indicate "
        "somebody/nobody/pseudo-referential status? E.g. claims of authenticity, "
        "generic character names, editorial apparatus, disclaimers. 1-2 sentences."
    )
    modal_shifts: str = Field(
        default="",
        description="If the text shifts modes between passages (e.g. ironic frame narrating "
        "a romance plot, or low mimetic opening that escalates to high mimetic), describe "
        "the shift. Empty if the mode is consistent."
    )
    displacement: str = Field(
        default="",
        description="Frye's concept: how far the narrative is 'displaced' from pure myth "
        "toward realism. Note any mythic patterns visible beneath realistic surface "
        "(e.g. a realistic novel with a death-and-rebirth pattern, or a comic plot "
        "that recapitulates the seasonal cycle). Empty if not applicable."
    )
    notes: str = Field(
        default="",
        description="Any other observations about the text's literary mode, style, "
        "or relationship to genre conventions. 1-3 sentences."
    )


SYSTEM_PROMPT = """You are a literary critic with expertise in Northrop Frye's modal theory from Anatomy of Criticism (1957). You are classifying narrative texts by their fictional mode and mythos based on textual evidence.

You will receive three passages from a text:
- OPENING: the first ~1000 words
- MIDDLE: ~1000 words from the middle of the text
- CLOSING: the final ~1000 words

Use all three to assess the text's mode and mythos. The opening often establishes the modal contract with the reader. The middle reveals the narrative's characteristic texture. The closing reveals the plot shape (comedy, tragedy, etc.).

## Frye's Five Modes (by hero's power)

1. **Myth**: Hero is a god or supernatural being, superior IN KIND to people and nature. Divine action, metamorphosis, creation. Rare in post-medieval literature except in deliberate mythopoeic works.

2. **Romance**: Hero is superior IN DEGREE to other people and the environment. Marvellous courage, enchanted worlds, quests, monsters, magical helpers. Laws of nature partly suspended. The world of fairy tale, chivalric romance, adventure.

3. **High mimetic**: Hero is superior in degree to other people but NOT to nature. Authority, dignity, passion on a grand scale. The world of epic and tragedy — kings, generals, leaders whose fall has public consequence.

4. **Low mimetic**: Hero is one of us — no special power or authority. Realistic social world, domestic concerns, comedy of manners. Most 18th-19th century fiction. The reader's response is "he is one of us."

5. **Ironic**: Hero is inferior to us in power or intelligence. We look down on scenes of bondage, frustration, absurdity, victimization. The hero may be a pharmakos (scapegoat). Includes unreliable narrators who reveal more than they understand.

## Frye's Four Mythoi (by plot shape)

- **Comedy**: Movement from a society controlled by habit, ritual bondage, or blocking characters toward a new, freer society. Often ends in marriage, feast, or reconciliation. The obstacles are overcome; the social order is renewed.

- **Romance** (as mythos): The quest narrative. Three stages: agon (conflict/adventure), pathos (crucial struggle, often a death-battle), anagnorisis (recognition of the hero, exaltation). Good triumphs over evil.

- **Tragedy**: Fall from prosperity to catastrophe. The hero's hamartia (not a moral flaw but a structural feature that makes the fall inevitable). Isolation from society. The social order survives but is diminished.

- **Irony/Satire**: The mythos of winter. Tragic or comic patterns are present but deflated, mocked, or unresolved. The pharmakos is punished without clear guilt. Anti-quest, anti-romance. Parody, absurdity, cyclical futility.

## Key principles

- Mode and mythos are INDEPENDENT axes. You can have ironic comedy (Fielding), romantic tragedy (Webster), low mimetic romance (adventure novel), etc.
- Most post-1600 fiction is displaced — mythic patterns operate beneath a realistic surface. Look for displacement: the realistic novel that recapitulates a quest or a seasonal cycle.
- A text can shift modes: an ironic frame narrator telling a romance story, or a low mimetic novel that rises to high mimetic in its climax.
- Early modern fiction often mixes modes freely. Don't force a single label if the text genuinely straddles categories.
- Base your classification on TEXTUAL EVIDENCE in the passages, not on what you know about the work from literary history. A well-known "novel" might read as romance in its actual texture.

## Referential Mode (Gallagher/Paige)

The text's relationship to real-world reference — a key axis in the history of fiction:

- **Nobody**: Explicitly fictional characters. The reader enters a fictional compact knowing these people don't exist. Generic names, novelistic conventions (free indirect discourse, interiority), no claim to documentary truth. The "novel proper" — Pamela, Tom Jones, Evelina.

- **Somebody**: Claims to be about real, identifiable people. Secret histories, scandal chronicles, biographical narratives about named historical figures. The text's interest depends on the reader believing (or pretending) the persons are real.

- **Pseudo-referential**: Invented stories claiming to be true. Found manuscripts, "authentic memoirs" of people who don't exist, travel narratives to nonexistent places, discovered correspondence. The fictional status is obvious to the modern reader but the text maintains a pretense of documentary truth. Defoe's Robinson Crusoe, Swift's Gulliver. Before ~1740, most prose fiction operates in this mode.

- **Ambiguous**: Genuinely unclear whether the characters/events are real or invented. Texts that hover between memoir and fiction, or that mix real and fictional persons without clear demarcation.

Look for: editorial prefaces claiming truth, "found among the papers of," generic vs. specific names, claims to have "changed names to protect," addresses to the reader about authenticity.
"""

EXAMPLES = [
    # Low mimetic comedy — Austen
    (
        "Title: Pride and Prejudice\nAuthor: Austen\n\n"
        "OPENING:\nIt is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters...\n\n"
        "MIDDLE:\nElizabeth's spirits soon rising to playfulness again, she wanted Mr. Darcy to account for his having ever fallen in love with her...\n\n"
        "CLOSING:\n...and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them.",
        FryeClassification(
            mode="low_mimetic",
            mode_confidence=0.95,
            mythos="comedy",
            mythos_confidence=0.95,
            narration="third_person_omniscient",
            mode_signals="The opening's ironic universalizing ('a truth universally acknowledged') establishes the hero as operating within ordinary social constraints — fortune, marriage, neighbourhood opinion. No special authority or power.",
            mythos_signals="Classic comic plot: blocking characters (pride, prejudice, Lady Catherine) overcome, society reconstituted through marriage. Movement from misunderstanding to recognition and union.",
            referential_mode="nobody",
            referential_signals="Explicitly fictional characters with generic-but-plausible names (Bennet, Darcy, Bingley). No claim to documentary truth, no editorial preface asserting authenticity. Pure novelistic convention.",
            modal_shifts="",
            displacement="The marriage plot displaces the comic mythos of spring — renewal through union. Elizabeth's 'rising spirits' echo the comic rhythm of liberation.",
            notes="The opening sentence's irony creates a slight tilt toward the ironic mode, but the narrative voice is ultimately sympathetic rather than superior to its characters.",
        ),
    ),
    # Romance — Spenser/Sidney type
    (
        "Title: The Countess of Pembroke's Arcadia\nAuthor: Sidney\n\n"
        "OPENING:\nIt was in the time that the earth begins to put on her new apparel against the approach of her lover, and that the sun running a most even course becomes an indifferent arbiter between the night and the day, when the hopeless shepherd Strephon was come to the sands which lie against the Island of Cithera...\n\n"
        "MIDDLE:\nBut Musidorus, who by the sentence of the Oracle was to remain with Pamela, found his mind divided between hope and fear...\n\n"
        "CLOSING:\n...the solemnities of their marriages with the deserved praises of all the assembly.",
        FryeClassification(
            mode="romance",
            mode_confidence=0.9,
            mythos="romance",
            mythos_confidence=0.85,
            narration="third_person_omniscient",
            mode_signals="The opening's pastoral personification ('earth puts on her new apparel') and mythological geography ('Island of Cithera') signal a world where nature responds to human feeling. Heroes face oracles and enchantments.",
            mythos_signals="Quest structure visible: heroes (Musidorus, Strephon) separated from beloveds, undergo trials, achieve recognition and union. Marriage ending = comic resolution of romantic quest.",
            referential_mode="nobody",
            referential_signals="Fictional characters with literary-pastoral names (Strephon, Musidorus, Pamela). No pretense of historical truth — the Arcadian setting is explicitly a literary construct.",
            modal_shifts="",
            displacement="Relatively undisplaced romance — the pastoral setting, oracle, and mythological references operate openly rather than beneath a realistic surface.",
            notes="Sidney's Arcadia is a crucial text for English prose romance, blending Greek romance plot structures with pastoral and chivalric elements.",
        ),
    ),
    # Ironic mode, ironic mythos — Sterne
    (
        "Title: The Life and Opinions of Tristram Shandy\nAuthor: Sterne\n\n"
        "OPENING:\nI wish either my father or my mother, or indeed both of them, as they were in duty both equally bound to it, had minded what they were about when they begot me; had they duly consider'd how much depended upon what they were then doing...\n\n"
        "MIDDLE:\nMy mother, you must know, —— but I have fifty things more necessary to let you know first...\n\n"
        "CLOSING:\nL—d! said my mother, what is all this story about? —— A COCK and a BULL, said Yorick —— And one of the best of its kind, I ever heard.",
        FryeClassification(
            mode="ironic",
            mode_confidence=0.95,
            mythos="irony",
            mythos_confidence=0.9,
            narration="first_person",
            mode_signals="The narrator is inferior to his own narrative — he cannot even get born, cannot tell his story in order, is defeated by digression. 'I wish they had minded what they were about' — the hero as victim of circumstance from before birth.",
            mythos_signals="Anti-narrative: the plot never arrives, the ending declares itself a 'cock and bull story.' Comic patterns are present but perpetually deferred. The seasonal cycle stalls in winter.",
            referential_mode="pseudo_referential",
            referential_signals="The title claims to be a 'Life and Opinions' — the memoir/autobiography form — but Tristram is transparently fictional. The pseudo-referential frame (editor, dedications, date-stamped chapters) is part of the ironic apparatus.",
            modal_shifts="",
            displacement="Deliberately anti-displaced: Sterne exposes the mythic machinery of narrative (birth, quest, death) by making it malfunction. The displacement IS the subject.",
            notes="Tristram Shandy is Frye's ironic mode taken to its logical extreme — the hero who cannot even narrate himself into existence.",
        ),
    ),
]


class FryeTask(Task):
    name = "classify_frye"
    schema = FryeClassification
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.3


def format_text_for_frye(text_obj=None, txt=None, title=None, author=None,
                          n_opening=2000, n_middle=1000, n_closing=2000):
    """Format a text into opening/middle/closing passages for Frye classification.

    Args:
        text_obj: An lltk text object (used to get txt, title, author).
        txt: Raw text string (alternative to text_obj).
        title: Title string (used if text_obj not provided).
        author: Author string (used if text_obj not provided).
        n_opening: Number of words for the opening passage (~3000 default).
        n_middle: Number of words for the middle passage (~1000 default).
        n_closing: Number of words for the closing passage (~1000 default).

    Returns:
        str: Formatted prompt with OPENING, MIDDLE, CLOSING sections.
    """
    if text_obj is not None:
        txt = txt or text_obj.txt
        title = title or getattr(text_obj, 'title', '') or ''
        author = author or getattr(text_obj, 'author', '') or ''

    if not txt:
        return None

    words = txt.split()
    total = len(words)

    if total < n_opening + n_middle:
        # Short text: just use the whole thing
        opening = ' '.join(words)
        middle = ''
        closing = ''
    else:
        opening = ' '.join(words[:n_opening])
        mid_start = max(n_opening, total // 2 - n_middle // 2)
        middle = ' '.join(words[mid_start:mid_start + n_middle])
        closing = ' '.join(words[-n_closing:])

    parts = []
    if title or author:
        header = []
        if title:
            header.append(f"Title: {title}")
        if author:
            # Send last name only for verification
            name = author.split(',')[0].strip() if ',' in author else author
            header.append(f"Author: {name}")
        parts.append('\n'.join(header))

    parts.append(f"OPENING:\n{opening}")
    if middle:
        parts.append(f"MIDDLE:\n{middle}")
    if closing:
        parts.append(f"CLOSING:\n{closing}")

    return '\n\n'.join(parts)
