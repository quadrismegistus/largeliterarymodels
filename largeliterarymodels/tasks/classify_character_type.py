"""Character archetype and class classification from social network data.

Second-pass task: takes the character roster + summaries from a SocialNetworkTask
result and classifies each character's archetype(s) and social class. Combines
Frye's structural types, Pavel's historical types, and period-specific archetypes.

Input per text: character records (name, descriptions, intro_text, class, gender,
notes) plus the summary chain for plot context. Output: one annotation per
character with archetype(s) and refined class.

Usage:
    from largeliterarymodels.tasks import CharacterTypeTask

    task = CharacterTypeTask(model="claude-sonnet-4-6")
    result = task.run(CharacterTypeTask.format_input(sn_json))
"""

from typing import Literal
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


ARCHETYPE_TAGS = [
    # Frye structural types (from Anatomy of Criticism)
    'alazon',
    'eiron',
    'bomolochoi',
    'pharmakos',

    # Pavel historical types (from Lives of the Novel)
    'beautiful_soul',
    'exceptional_being',
    'persecuted_innocent',
    'picaro',
    'fallible_everyman',
    'fallible_virtuous',
    'innocent_child',

    # Period-specific
    'rake',
    'ingenue',
    'virtuous_maiden',
    'reformed_sinner',
    'trickster',
    'servant_confidante',
    'tyrant',
    'mentor',
    'suitor',
    'rival',
    'coquette',
    'wanderer',
]

CLASS_TAGS = [
    'royalty',
    'titled_nobility',
    'untitled_gentry',
    'clergy',
    'professional',
    'merchant',
    'artisan',
    'yeoman_farmer',
    'servant',
    'laborer',
    'soldier',
    'sailor',
    'criminal',
    'beggar',
    'slave',
    'foreign_noble',
    'commoner',
    'religious',
    'apprentice',
    'clerk',
    'student',
    'teacher',
    'supernatural',
    'unknown',
]


class CharacterAnnotation(BaseModel):
    name: str = Field(description="Character name as given in the social network data.")
    archetypes: list[Literal[
        'alazon', 'eiron', 'bomolochoi', 'pharmakos',
        'beautiful_soul', 'exceptional_being', 'persecuted_innocent',
        'picaro', 'fallible_everyman', 'fallible_virtuous', 'innocent_child',
        'rake', 'ingenue', 'virtuous_maiden', 'reformed_sinner',
        'trickster', 'servant_confidante', 'tyrant', 'mentor',
        'suitor', 'rival', 'coquette', 'wanderer',
    ]] = Field(
        default_factory=list,
        description="Character archetype(s). Select 1-3 that best fit.",
    )
    social_class: Literal[
        'royalty', 'titled_nobility', 'untitled_gentry', 'clergy',
        'professional', 'merchant', 'artisan', 'yeoman_farmer',
        'servant', 'laborer', 'soldier', 'sailor',
        'criminal', 'beggar', 'slave', 'foreign_noble', 'commoner',
        'religious', 'apprentice', 'clerk', 'student', 'teacher',
        'supernatural', 'unknown',
    ] = Field(
        default='unknown',
        description="Social class/station. Use the most specific applicable label.",
    )


class CharacterTypeAnnotation(BaseModel):
    characters: list[CharacterAnnotation] = Field(
        description="One annotation per character in the input roster.",
    )


SYSTEM_PROMPT = """\
You are a literary historian classifying characters from English prose fiction \
by archetype and social class.

You will receive:
1. A CHARACTER ROSTER with each character's name, gender, existing class label, \
descriptions, and introduction text.
2. A PLOT SUMMARY showing what happens to these characters.

For each character, determine:

## ARCHETYPES (select 1-3)

STRUCTURAL TYPES (Frye):
- alazon: the impostor/pretender — blocking figure who pretends to more than they \
are. Heavy fathers, braggart soldiers, pedants, hypocrites. They obstruct the \
plot's resolution and must be exposed or defeated.
- eiron: the self-deprecator — appears less than they are but ultimately prevails. \
Tricky servants, witty heroines, scheming helpers. They outwit the alazon.
- bomolochoi: the buffoon — provides comic relief, entertainment, social lubrication. \
Parasites, fools, clowns, comic sidekicks.
- pharmakos: the scapegoat — an innocent or relatively innocent figure who is \
expelled, sacrificed, or made to suffer for others. Central to tragedy and pathos.

HISTORICAL TYPES (Pavel):
- beautiful_soul: C18 concept — moral perfection radiates from within the individual. \
Their inner virtue is self-evident and inspires others. (Pamela, Clarissa, Julie.)
- exceptional_being: pre-C18 — morally perfect hero set apart from the world by \
birth, destiny, or divine favor. Not ordinary. (Greek romance heroes, chivalric knights.)
- persecuted_innocent: virtue tested by adversity — imprisonment, assault, \
separation, slander — but never broken. (Pamela, Griselda, many romance heroines.)
- picaro: low-born, roguish, lives by wit and trickery, drifts through society \
serving different masters. No moral growth. (Lazarillo, Moll Flanders, Jack.)
- fallible_everyman: ordinary person, not evil but imperfect, makes mistakes \
from weakness rather than malice. (Tom Jones, most C18-C19 protagonists.)
- fallible_virtuous: genuinely virtuous but also makes real errors they must \
recognize and correct. The synthesis type. (Elizabeth Bennet, Dorothea Brooke.)
- innocent_child: moral purity located in childhood or youth. (Oliver Twist, \
David Copperfield, Fanny Price as child.)

PERIOD-SPECIFIC:
- rake: aristocratic seducer/libertine, sexually predatory, often charming. \
(Lovelace, Willmore, Mr. B before reform.)
- ingenue: young, innocent, inexperienced woman entering society. \
(Evelina, Fanny Burney's heroines.)
- virtuous_maiden: specifically defined by sexual virtue under threat. \
Distinct from ingenue (who is naive) and beautiful soul (who has moral authority).
- reformed_sinner: character who undergoes moral transformation from vice to \
virtue. (Mr. B in Pamela, Moll Flanders at the end.)
- trickster: deceives others for gain or amusement, not necessarily low-born \
(unlike picaro). Shape-shifter, disguise artist. (Fantomina, many Behn characters.)
- servant_confidante: loyal servant or friend who aids the protagonist, \
knows their secrets, facilitates the plot. (Pamela's Mrs. Jervis, waiting-women.)
- tyrant: exercises illegitimate power over others — fathers, husbands, lords, \
jailors. Often the antagonist in idealist fiction.
- mentor: wise advisor, teacher, or guardian who guides the protagonist. \
(Allworthy in Tom Jones, Mr. Villars in Evelina.)
- suitor: character whose primary plot function is courting or pursuing marriage.
- rival: competitor for the protagonist's love interest or position.
- coquette: flirtatious, manipulative in romantic contexts, plays with desire. \
Distinct from rake (female-coded) and trickster (not necessarily romantic).
- wanderer: defined by movement and displacement — travelers, exiles, \
castaways, pilgrims. (Crusoe, Gulliver, pilgrimage protagonists.)

## SOCIAL CLASS

Classify each character's social station using the most SPECIFIC label available. \
Use evidence from descriptions and intro_text — look for titles (Duke, Earl, Lord, \
Sir, Lady), occupations (merchant, lawyer, clerk), and social markers.

- royalty: kings, queens, princes, princesses
- titled_nobility: duke, duchess, earl, countess, baron, baroness, marquis, \
viscount, lord, lady (with actual peerage, not courtesy)
- untitled_gentry: esquire, gentleman, gentlewoman, landed families without titles
- clergy: priests, ministers, bishops, monks, nuns, abbots
- professional: lawyers, doctors, commissioned officers, scholars, tutors, \
writers, artists, actors, musicians, government officials, magistrates, administrators
- merchant: traders, shopkeepers, bankers, shipowners
- artisan: skilled tradespeople, craftsmen, innkeepers
- yeoman_farmer: independent small landholders, farmers
- servant: domestic servants, maids, valets, stewards, governesses
- laborer: unskilled workers, porters, sailors' wives
- soldier: enlisted military (not officers — those are professional)
- sailor: seamen, pirates, naval ratings
- criminal: thieves, prostitutes, highwaymen (by primary occupation)
- beggar: destitute, homeless, wandering poor
- slave: enslaved persons
- foreign_noble: non-English royalty or nobility
- commoner: ordinary person in a setting where the primary class distinction is \
royal/noble vs common (classical, biblical, fairy-tale settings where finer \
English social distinctions don't apply)
- religious: monks, nuns, friars, abbesses, hermits in religious orders \
(distinct from clergy who hold church office)
- apprentice: bound to a master for training in a trade
- clerk: office workers, scribes, bookkeepers, counting-house employees
- student: university students, schoolboys, pupils
- teacher: tutors, governesses, schoolmasters, professors
- supernatural: gods, angels, demons, allegorical personifications, spirits, \
mythological beings, prophets with divine powers
- unknown: class cannot be determined from the evidence

## IMPORTANT

- Only annotate characters in the input roster. Match by name exactly.
- If a character is too minor to classify (mentioned once, no description), \
use archetypes=[] and social_class='unknown'.
- A character can have multiple archetypes (e.g. rake + alazon, or \
persecuted_innocent + beautiful_soul).
- For social class, use the character's PRIMARY station, not where they end up \
(Pamela is 'servant' not 'untitled_gentry')."""


EXAMPLES = [
    (
        "CHARACTERS:\n"
        "- Pamela Andrews (female, class: servant). Descriptions: 'young, beautiful "
        "servant girl'; 'her virtue is unwavering'; 'writes letters documenting her "
        "trials'. Intro: 'Pamela, a young waiting-maid to Lady B, is left without "
        "protection when her mistress dies.'\n"
        "- Mr. B (male, class: gentleman). Descriptions: 'wealthy landowner'; "
        "'attempts seduction and imprisonment'; 'eventually moved by Pamela's virtue'. "
        "Intro: 'Mr. B, the young master of the estate, begins pursuing Pamela.'\n"
        "- Mrs. Jervis (female, class: servant). Descriptions: 'housekeeper'; "
        "'sympathetic to Pamela'; 'tries to protect her'. Intro: 'Mrs. Jervis, the "
        "kind housekeeper, becomes Pamela's ally.'\n\n"
        "SUMMARY:\nPamela resists Mr. B's advances. He imprisons her. She never wavers. "
        "He reads her journal, is moved, proposes. She accepts.",
        CharacterTypeAnnotation(characters=[
            CharacterAnnotation(
                name='Pamela Andrews',
                archetypes=['persecuted_innocent', 'beautiful_soul'],
                social_class='servant',
            ),
            CharacterAnnotation(
                name='Mr. B',
                archetypes=['rake', 'reformed_sinner', 'alazon'],
                social_class='untitled_gentry',
            ),
            CharacterAnnotation(
                name='Mrs. Jervis',
                archetypes=['servant_confidante'],
                social_class='servant',
            ),
        ]),
    ),
]


DEFAULT_MODEL = 'lmstudio/qwen/qwen3.6-35b-a3b'


class CharacterTypeTask(Task):
    name = "classify_character_type"
    model = DEFAULT_MODEL
    schema = CharacterTypeAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.3

    @staticmethod
    def format_input(social_network_json: dict) -> str:
        """Format character roster + summaries from a social network result."""
        chars = social_network_json.get('characters', [])
        lines = ["CHARACTERS:"]
        for c in chars:
            name = c.get('name', '?')
            gender = c.get('gender', '?')
            cls = c.get('class', '?')
            descs = c.get('descriptions', [])
            intro = c.get('intro_text', '')
            notes = c.get('notes', '')

            desc_str = '; '.join(descs[:5]) if descs else 'none'
            parts = [f"- {name} ({gender}, class: {cls}). Descriptions: '{desc_str}'"]
            if intro:
                parts.append(f"Intro: '{intro[:200]}'")
            if notes:
                parts.append(f"Notes: '{notes[:100]}'")
            lines.append(' '.join(parts))

        summaries = social_network_json.get('summaries', [])
        if summaries:
            lines.append("\nSUMMARY:")
            if isinstance(summaries[0], dict):
                lines.extend(s.get('text', s.get('summary', ''))
                             for s in summaries)
            else:
                lines.extend(str(s) for s in summaries)

        return '\n'.join(lines)
