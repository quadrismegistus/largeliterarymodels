"""Plot and genre classification from social network summaries.

Second-pass task: takes the summary chain from a SocialNetworkTask result
(~2-3K tokens) and classifies the text's plot structure, genre, and narrative
mode using a synthesis of Frye's mythoi, Pavel's narrative traditions, and
standard literary-historical genre vocabulary.

Designed for cheap bulk runs on local models. Input is already extracted —
no need for frontier models or full-text access.

Usage:
    from largeliterarymodels.tasks import PlotGenreTask

    task = PlotGenreTask(model="lmstudio/qwen/qwen3.6-35b-a3b")
    result = task.run(summary_text)
"""

from typing import Literal
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


# === Schema ===

class PlotGenreAnnotation(BaseModel):
    # --- Frye ---
    mythos: Literal[
        'comedy', 'romance', 'tragedy', 'irony_satire'
    ] = Field(
        description=(
            "Frye's four mythoi — the fundamental plot archetype. "
            "comedy: society blocks desire (often young lovers), obstacles are "
            "overcome, ending in integration, festivity, or marriage. "
            "The movement is from confusion/constraint toward freedom/union. "
            "romance: a quest or adventure; the hero faces trials and triumphs "
            "through courage, virtue, or marvellous aid. Good vs. evil is clear. "
            "tragedy: a hero of stature falls from prosperity through error or "
            "circumstance; ends in catastrophe, death, or irreversible loss. "
            "irony_satire: no triumph, no clear villain; characters are trapped "
            "by society, self-deception, or absurdity. Exposes folly without resolution."
        )
    )
    frye_mode: Literal[
        'myth', 'romance', 'high_mimetic', 'low_mimetic', 'ironic'
    ] = Field(
        description=(
            "The protagonist's power relative to other characters and their world. "
            "myth: protagonist is a god or has divine powers (rare in novels). "
            "romance: protagonist is human but superior to others AND to natural law — "
            "enchantments work, impossible escapes succeed, fate intervenes. "
            "Characters in Greek romances, chivalric tales, fairy tales. "
            "high_mimetic: protagonist is human, superior to others in authority or "
            "stature (a king, general, noble leader) but NOT exempt from natural law. "
            "Characters in epic, classical tragedy, heroic drama. "
            "low_mimetic: protagonist is an ordinary person, equal to the reader. "
            "Most C18-C19 novels: Pamela, Tom Jones, Emma, Pip. "
            "ironic: protagonist is inferior to the reader in power or self-knowledge. "
            "We watch them fail to understand what we can see. "
            "Gulliver, Underground Man, Kafka's K."
        )
    )

    # --- Pavel ---
    pavel_tradition: Literal[
        'idealist', 'anti_idealist', 'novella', 'synthesis'
    ] = Field(
        description=(
            "Pavel's four narrative traditions. "
            "idealist: the protagonist embodies a moral ideal and is set apart from "
            "a fallen or corrupt world. The plot tests their virtue against adversity. "
            "Greek romance, chivalric tales, Richardson's Pamela/Clarissa, gothic. "
            "anti_idealist: the protagonist is flawed, roguish, or ordinary, and "
            "struggles against social norms they cannot or will not meet. "
            "Picaresque, Fielding, Stendhal, ironic realism. "
            "novella: a single striking event or crisis reveals an unexpected chasm "
            "between an individual and their milieu. Short, concentrated, unified action. "
            "synthesis: combines idealist and anti-idealist — the protagonist has "
            "genuine virtue but is also fallible, and the world is neither purely "
            "corrupt nor purely good. Austen, Eliot, Tolstoy."
        )
    )
    moral_source: Literal[
        'transcendent', 'immanent', 'contextual', 'aesthetic', 'absent'
    ] = Field(
        description=(
            "Where moral authority originates in the text's world. "
            "transcendent: moral ideals exist outside/above the human domain — "
            "divine law, fate, cosmic order, allegory (premodern romance, Bunyan). "
            "immanent: moral beauty radiates from within the individual heart — "
            "the 'beautiful soul' whose inner virtue is self-evident "
            "(Richardson, Rousseau, sentimental novel). "
            "contextual: morality is rooted in specific social and historical "
            "circumstances — what is right depends on where and when "
            "(Austen, Balzac, Eliot, realist novel). "
            "aesthetic: moral concerns are secondary to artistic freedom, "
            "subjective experience, or formal experiment (Wilde, Joyce, Woolf). "
            "absent: no coherent moral framework; nihilistic, absurdist, or "
            "deliberately amoral (Sade, Beckett, some naturalism)."
        )
    )

    # --- Plot structure ---
    plot_structure: Literal[
        'episodic', 'unified', 'bildungsroman', 'circular', 'frame_tale'
    ] = Field(
        description=(
            "episodic: loosely connected adventures/incidents across time and space; "
            "removing or reordering episodes would not break the story "
            "(picaresque, Don Quixote, idealist romance). "
            "unified: a single dramatic arc with cause-and-effect chain; "
            "every event is necessary to the outcome (Pamela, Clarissa, Emma). "
            "bildungsroman: the protagonist's growth, education, or moral development "
            "is the structuring principle (Tom Jones, Evelina, David Copperfield). "
            "circular: the protagonist returns to the starting point after a journey, "
            "changed or unchanged (Odyssey pattern, Moll Flanders). "
            "frame_tale: embedded narratives within a framing structure "
            "(Decameron, Canterbury Tales, Frankenstein)."
        )
    )
    ending_type: Literal[
        'marriage', 'death', 'reform', 'return', 'open', 'ambiguous'
    ] = Field(
        description=(
            "How the main plot resolves. "
            "marriage: union or integration resolves the central conflict. "
            "death: the protagonist or a key character dies, ending the action. "
            "reform: moral transformation — a character repents or changes. "
            "return: protagonist comes back home or to their origin. "
            "open: no decisive resolution; life continues. "
            "ambiguous: deliberately unclear, ironic, or multi-layered ending."
        )
    )


# === Prompt ===

SYSTEM_PROMPT = """\
You are an expert literary historian classifying narrative fiction by plot, genre, and mode.

You will receive a SUMMARY of a novel or narrative text (extracted from a social network \
analysis). The summary describes the plot, characters, and key events in sequence.

Classify the text along these axes:

1. FRYE'S MYTHOS — the fundamental plot archetype:
   - comedy: obstacles to desire/union overcome → integration, festivity, marriage
   - romance: quest or adventure → hero triumphs through virtue or marvellous aid
   - tragedy: hero of stature falls → catastrophe, death, irreversible loss
   - irony_satire: characters trapped by society/self-deception → no resolution, folly exposed

2. FRYE'S MODE — the protagonist's power relative to others and their world:
   - myth: divine or supernatural protagonist (rare in novels)
   - romance: human but EXEMPT from natural law — enchantments, impossible rescues, \
fate intervenes (Greek romance, chivalric tales, fairy tales)
   - high_mimetic: human leader SUBJECT TO natural law — kings, generals, nobles \
with authority but no supernatural protection (epic, heroic drama, classical tragedy)
   - low_mimetic: ordinary person, equal to the reader (most C18-C19 novels)
   - ironic: protagonist inferior to reader in power or self-knowledge

3. PAVEL'S TRADITION:
   - idealist: protagonist embodies moral ideal, set apart from a fallen world \
(Greek romance, chivalric, Richardson, gothic)
   - anti_idealist: flawed protagonist vs. social norms (picaresque, Fielding, Stendhal)
   - novella: single striking event reveals gap between individual and milieu
   - synthesis: protagonist has genuine virtue but is also fallible; world is mixed \
(Austen, Eliot, Tolstoy)

4. MORAL SOURCE — where moral authority originates:
   - transcendent: outside/above human domain (divine law, fate, allegory)
   - immanent: radiates from the individual heart (C18 'beautiful soul', Richardson)
   - contextual: rooted in social/historical circumstances (C19 realism)
   - aesthetic: moral concerns secondary to art/freedom (modernism)
   - absent: no coherent moral framework (nihilistic, absurdist)

5. PLOT STRUCTURE: episodic / unified / bildungsroman / circular / frame_tale

6. ENDING TYPE: marriage / death / reform / return / open / ambiguous

Base your classification on the PLOT and EVENTS described in the summary."""


EXAMPLES = [
    (
        "Summary: Pamela Andrews, a young servant girl, resists the sexual advances of "
        "her master Mr. B after his mother dies. He imprisons her, intercepts her letters, "
        "and attempts seduction and assault. Pamela's virtue never wavers. Eventually "
        "Mr. B reads her journal, is moved by her goodness, and proposes marriage. "
        "She accepts, wins over hostile relatives, and rises to respected gentlewoman.",
        PlotGenreAnnotation(
            mythos='comedy',
            frye_mode='low_mimetic',
            pavel_tradition='idealist',
            moral_source='immanent',
            plot_structure='unified',
            ending_type='marriage',
        ),
    ),
    (
        "Summary: Moll Flanders recounts her life from birth in Newgate prison through "
        "five marriages, numerous love affairs, widowhood, incest discovered, poverty, "
        "twelve years as a thief, eventual arrest and transportation to Virginia, "
        "where she prospers as a plantation owner with her Lancashire husband. "
        "In old age she returns to England penitent and wealthy.",
        PlotGenreAnnotation(
            mythos='irony_satire',
            frye_mode='low_mimetic',
            pavel_tradition='anti_idealist',
            moral_source='contextual',
            plot_structure='episodic',
            ending_type='return',
        ),
    ),
]


DEFAULT_MODEL = 'lmstudio/qwen/qwen3.6-35b-a3b'


class PlotGenreTask(Task):
    name = "classify_plot_genre"
    model = DEFAULT_MODEL
    schema = PlotGenreAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.3

    @staticmethod
    def format_input(social_network_json: dict) -> str:
        """Extract summary text from a social network result dict."""
        summaries = social_network_json.get('summaries', [])
        if isinstance(summaries, list):
            if summaries and isinstance(summaries[0], dict):
                text = '\n\n'.join(s.get('summary', s.get('text', ''))
                                   for s in summaries)
            else:
                text = '\n\n'.join(str(s) for s in summaries)
        else:
            text = str(summaries)
        return f"Summary:\n{text}"
