"""Subgenre classification and fiction verification from social network summaries.

Second-pass task: takes the summary chain from a SocialNetworkTask result and
classifies fine-grained subgenre (beyond the coarse form/mode tags in GenreTaskLite)
plus verifies whether the text is actually fiction.

The subgenre vocabulary draws on the Deep Research genre taxonomy compiled from
McKeon, Doody, Davis, Hunter, Salzman, Richetti, Ballaster, and others. Terms
are chosen for detectability from plot summaries and relevance to pre-1800 English
prose fiction corpora (ECCO, EEBO, Chadwyck, EarlyPrint).

Usage:
    from largeliterarymodels.tasks import SubgenreTask

    task = SubgenreTask(model="gemini-2.5-flash")
    result = task.run(summary_text)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


NONFICTION_TYPES = [
    'sermon', 'treatise', 'history', 'biography', 'periodical',
    'essay', 'letters', 'speech', 'legal', 'reference', 'criticism',
    'poetry', 'drama', 'almanac',
]

SUBGENRE_TAGS = [
    # Fine-grained (gaps in GenreTaskLite vocabulary)
    'domestic_fiction',
    'conduct_fiction',
    'pseudo_memoir',
    'travel_narrative',
    'philosophical_tale',
    'frame_tale',
    'anatomy',
    'menippean_satire',
    'spiritual_autobiography',
    'scandalous_memoir',
    'fairy_tale',
    'robinsonade',
    'captivity_narrative',
    'jacobin_novel',
    'national_tale',
    'providence_tale',
    'anti_romance',
    'parody',
    'heroic_romance',
    'beast_fable',
    'exemplum',
    'fabliau',
    'newgate_narrative',
    'children_fiction',
    # Duplicated from GenreTaskLite for completeness
    'picaresque',
    'rogue_fiction',
    'oriental_tale',
    'roman_a_clef',
    'secret_history',
    'it_narrative',
    'imaginary_voyage',
    'amatory_fiction',
    'gothic',
    'epistolary',
    'sentimental',
    'allegory',
    'pastoral',
    'historical_fiction',
]


class SubgenreAnnotation(BaseModel):
    is_fiction: bool = Field(
        description=(
            "Is this text prose fiction (novel, romance, tale, novella)? "
            "False for sermons, treatises, histories, biographies, periodical essays, "
            "poetry collections, drama, legal documents, or other non-fiction that "
            "may have been miscatalogued as fiction in the corpus."
        )
    )
    non_fiction_type: Optional[Literal[
        'sermon', 'treatise', 'history', 'biography', 'periodical',
        'essay', 'letters', 'speech', 'legal', 'reference', 'criticism',
        'poetry', 'drama', 'almanac',
    ]] = Field(
        default=None,
        description=(
            "If is_fiction is False, what type of non-fiction is this? "
            "Null if is_fiction is True."
        ),
    )
    subgenres: list[Literal[
        'domestic_fiction',
        'conduct_fiction',
        'pseudo_memoir',
        'travel_narrative',
        'philosophical_tale',
        'frame_tale',
        'anatomy',
        'menippean_satire',
        'spiritual_autobiography',
        'scandalous_memoir',
        'fairy_tale',
        'robinsonade',
        'captivity_narrative',
        'jacobin_novel',
        'national_tale',
        'providence_tale',
        'anti_romance',
        'parody',
        'heroic_romance',
        'beast_fable',
        'exemplum',
        'fabliau',
        'newgate_narrative',
        'children_fiction',
        'picaresque',
        'rogue_fiction',
        'oriental_tale',
        'roman_a_clef',
        'secret_history',
        'it_narrative',
        'imaginary_voyage',
        'amatory_fiction',
        'gothic',
        'epistolary',
        'sentimental',
        'allegory',
        'pastoral',
        'historical_fiction',
    ]] = Field(
        default_factory=list,
        description=(
            "Fine-grained subgenre labels. Select all that apply (typically 0-3). "
            "Empty list is fine if none fit or if is_fiction is False."
        ),
    )


SYSTEM_PROMPT = """\
You are an expert in the history of English prose fiction, classifying texts by \
fine-grained subgenre from plot summaries.

You will receive a SUMMARY of a text extracted from a social network analysis. \
First determine whether the text is actually fiction. Then classify its subgenre(s).

## Step 1: Is this fiction?

Many corpora contain misclassified texts. If the summary describes a sermon, \
treatise, historical chronicle, biography of a real person, periodical essay, \
poetry collection, or dramatic work — mark is_fiction=False and identify the type. \
If it IS prose fiction (novel, romance, tale, novella), mark is_fiction=True.

## Step 2: Subgenre classification

Select all subgenres that apply from the list below. Most texts will match 0-3. \
Only assign a label if you see clear evidence in the summary.

DOMESTIC FICTION — The plot centers on household life, family relationships, \
courtship, and marriage within a recognizable social milieu. The drama comes from \
social interactions, manners, and domestic crises, not from adventure or travel. \
Distinguished from "didactic" by focus on realistic social texture rather than \
explicit moral instruction. (Burney's Evelina, Austen, much C18 fiction.)

CONDUCT FICTION — The narrative is explicitly framed as moral instruction, \
typically for young women. The protagonist's choices are presented as models \
to follow or avoid. Often features a preface or narrator declaring didactic intent. \
Distinguished from domestic fiction by the overtly pedagogical frame. \
(Richardson's Pamela as conduct model, Edgeworth, Hannah More.)

PSEUDO-MEMOIR / FICTIONAL AUTOBIOGRAPHY — The text presents itself as the \
first-person life story of a fictional character, imitating the conventions of \
autobiography or memoir. Look for: "I was born...", retrospective narration, \
the character recounting their own adventures. (Defoe's Robinson Crusoe, \
Moll Flanders, Roxana; many C18 novels use this frame.)

TRAVEL NARRATIVE — The plot is structured around a journey to real or plausible \
foreign places, with emphasis on what the traveler sees, encounters, and learns. \
Distinguished from imaginary voyage (which visits impossible places) and from \
picaresque (where travel is incidental to roguish episodes). \
(Defoe's Captain Singleton, Smollett's travels, fictional Grand Tour narratives.)

PHILOSOPHICAL TALE / CONTE PHILOSOPHIQUE — A narrative subordinated to \
demonstrating a philosophical or moral thesis. Characters and events are \
schematic, illustrating ideas rather than developing psychologically. Often \
features a wise mentor, a naive questioner, or a journey through exemplary \
situations. (Johnson's Rasselas, Voltaire's Candide, Goldsmith's Citizen.)

FRAME TALE — Multiple stories embedded within a framing narrative. Look for: \
a group of characters who take turns telling stories, or a narrator who \
interrupts the main plot with inset tales. (Decameron pattern, \
Canterbury Tales, Marguerite de Navarre, some C17 collections.)

ANATOMY — Frye's fourth prose form: a systematic, encyclopedic dissection of \
a subject, mixing fictional narrative with digressions, catalogues, learned \
quotation, and satirical commentary. The plot is loose or absent; the pleasure \
is intellectual abundance. (Burton's Anatomy of Melancholy, Rabelais, \
Sterne's Tristram Shandy, Swift's Tale of a Tub.)

MENIPPEAN SATIRE — Satire that mixes genres, registers, voices, and tones \
to attack false learning, pedantry, or fanaticism. Often features fantastic \
settings, talking animals, or absurd premises. Distinguished from regular \
satire by the formal heterogeneity and intellectual target. \
(Lucian, Swift's Gulliver, Sterne, Pope's Dunciad.)

SPIRITUAL AUTOBIOGRAPHY / CONVERSION NARRATIVE — A first-person account of \
the narrator's journey from sin to grace, structured around temptation, \
despair, and divine deliverance. Look for: intense religious experience, \
providential interpretation of events, Pauline conversion pattern. \
(Bunyan's Grace Abounding, Puritan testimonies, feeds into Defoe.)

SCANDALOUS MEMOIR — A narrative (first- or third-person) of a notorious \
woman's sexual adventures, seductions, and social transgressions, blending \
criminal biography with amatory fiction and secret history. Distinguished \
from amatory fiction by focus on transgression and scandal rather than \
desire and sentiment. (The London Jilt, Defoe's Roxana, Cleland's Fanny Hill, \
Manley's Rivella.)

FAIRY TALE / CONTE DE FÉES — A narrative featuring magic, enchantment, \
transformation, fairies, or supernatural helpers in a non-religious context. \
Look for: wishes granted, enchanted objects, animal transformation, \
curses broken by love. (Perrault, d'Aulnoy, Beauty and the Beast.)

ROBINSONADE — A narrative centered on shipwreck, isolation, and survival, \
where the protagonist must build a life from nothing in a remote place. \
Look for: castaway on island/wilderness, solo survival, construction \
of shelter/economy, eventual rescue or escape. (Defoe's Robinson Crusoe \
and its many imitators.)

CAPTIVITY NARRATIVE — A first-person account of capture and imprisonment \
by a foreign or hostile group (Native Americans, Barbary pirates, Catholics, \
Turks), emphasizing suffering, faith, and eventual release. Look for: \
abduction, forced march, alien culture, ransom or escape. \
(Rowlandson, Barbary narratives, feeds into adventure fiction.)

JACOBIN NOVEL — A 1790s novel explicitly advancing or attacking French \
Revolutionary ideas: political justice, rights of women, critique of \
aristocracy, or (anti-Jacobin) defense of tradition. Look for: political \
argument woven into plot, characters who embody ideological positions, \
themes of tyranny vs. liberty. (Godwin's Caleb Williams, Wollstonecraft's \
Maria, Bage's Hermsprong; anti-Jacobin: Hamilton, West.)

NATIONAL TALE — A narrative that foregrounds the customs, landscape, language, \
and character of a specific nation or region, often with ethnographic intent. \
Look for: Irish/Scottish/Welsh setting, local dialects, descriptions of \
national customs, cross-cultural romance. (Edgeworth's Castle Rackrent, \
proto-Scott, Lady Morgan's Wild Irish Girl.)

PROVIDENCE TALE — A short narrative of miraculous deliverance, divine \
punishment, or supernatural prodigy, interpreted as evidence of God's \
active intervention. Look for: storms, monsters, miraculous escapes, \
divine warnings, moral explicitly drawn. (Cheap-print wonder books, \
embedded episodes in spiritual autobiography.)

ANTI-ROMANCE — A narrative that explicitly mocks, deflates, or critiques \
the conventions of chivalric or heroic romance. Look for: a protagonist \
who reads too many romances, quixotic delusions, bathetic contrast \
between romantic expectations and mundane reality. (Cervantes' influence, \
Lennox's Female Quixote, Fielding's Joseph Andrews.)

PARODY — A narrative that imitates and satirizes a specific earlier work \
or genre convention. Look for: recognizable characters or situations from \
the target, exaggeration of the original's features for comic effect. \
(Fielding's Shamela parodying Pamela, anti-Pamelas, Spiritual Quixote.)

HEROIC ROMANCE — A long, high-style prose romance set in classical or \
pseudo-historical antiquity, focused on the loves and martial exploits \
of princely heroes and heroines, with interlaced subplots, embedded \
histories, and elaborate speeches. Look for: royal/noble characters in \
ancient settings, wars and sieges intertwined with love plots, separated \
lovers reunited after long trials, multiple interlocking storylines. \
Distinct from chivalric romance (which features knights and quests) and \
from the modern novel. (Scudéry's Grand Cyrus, La Calprenède's \
Cassandre, Barclay's Argenis, Boyle's Parthenissa.)

BEAST FABLE — A narrative in which animals speak, act, and reason like \
humans, typically to satirize human society or teach moral lessons. \
Look for: named animal characters with human social roles, political \
or moral allegory through animal behavior. (Reynard the Fox, Aesop \
adaptations, Caxton's beast fables.)

EXEMPLUM — A short illustrative tale designed to enforce a moral or \
doctrinal point, typically found embedded in sermons or moral treatises \
but also standing alone. Look for: extreme brevity, a single moral \
lesson explicitly stated, stock characters who illustrate virtue or vice. \
(Gesta Romanorum tales, sermon illustrations.)

FABLIAU — A short, bawdy, comic narrative of trickery, adultery, or \
lower-class intrigue. Look for: sexual deception, cuckolded husbands, \
clever servants or wives, farcical situations, frank treatment of \
bodily functions. Ancestor of the novella. (Boccaccio's comic tales, \
Chaucer's Miller's Tale in prose adaptations.)

NEWGATE NARRATIVE — A prose account of the life, crimes, and execution \
of a criminal, often structured as a cautionary biography. Look for: \
named real or realistic criminals, detailed accounts of crimes and \
punishments, gallows speeches, moral framing. Distinguished from \
rogue fiction by focus on punishment rather than adventure. \
(Ordinary of Newgate's accounts, Newgate Calendar, criminal lives.)

CHILDREN'S FICTION — A prose narrative written for child readers, \
featuring young protagonists, simple moral lessons, and age-appropriate \
content. Look for: child protagonists, school or nursery settings, \
explicit moral instruction aimed at children, fairy-tale elements \
in a domestic frame. (Goody Two-Shoes, Sarah Fielding's Governess, \
Edgeworth's Parent's Assistant.)

PICARESQUE — A first-person, episodic narrative of a low-born rogue \
who serves a succession of masters, satirizing each social rank. \
Look for: born poor, drifts through society, lives by wits and \
trickery, no moral growth. (Lazarillo, Moll Flanders, Roderick Random.)

ROGUE FICTION — Prose narrative centered on criminal trickery, \
cony-catching, and underworld life. Overlaps with picaresque but \
may be third-person and more focused on exposing criminal methods \
than satirizing society. (Greene's cony-catching pamphlets, \
The English Rogue, criminal lives.)

ORIENTAL TALE — A narrative set in the Islamic East, India, or China, \
deploying exotic décor, frame-tale structure, and often philosophical \
or satirical purpose. Look for: Eastern settings, sultans/viziers, \
magical elements, didactic or satirical intent. (Arabian Nights, \
Rasselas, Vathek, Citizen of the World.)

ROMAN À CLEF — A fiction in which real contemporary persons are \
represented under fictional names. Look for: thinly veiled portraits \
of known political or social figures, scandal, political intrigue \
mapped onto fictional characters. (Barclay's Argenis, Manley's \
New Atalantis, Scudéry's Grand Cyrus.)

SECRET HISTORY — Narrative purporting to expose hidden court, sexual, \
or political history behind the public record. Look for: "behind \
the curtain" revelations, aristocratic scandal, political conspiracy \
presented as insider knowledge. (Manley's New Atalantis, Haywood's \
Memoirs of a Certain Island.)

IT-NARRATIVE — A fiction narrated by a non-human object (coin, \
banknote, animal, coach) passed from owner to owner, each episode \
exposing a different social milieu. Look for: object as narrator, \
circulation through society, satirical observation. (Chrysal, \
Pompey the Little, Adventures of a Guinea.)

IMAGINARY VOYAGE — A first-person narrative of travel to impossible \
or fantastical locations (Moon, floating islands, underground worlds), \
often with utopian or satirical intent. Distinguished from travel \
narrative by the impossibility of the destination. (Godwin's Man \
in the Moone, Gulliver's Travels, Cyrano.)

AMATORY FICTION — Short prose fiction centered on heterosexual \
seduction, female desire, and aristocratic intrigue. Look for: \
seduction plots, female protagonists caught between desire and \
virtue, passionate language, often scandalous. (Behn's Love-Letters, \
Haywood's Love in Excess, Manley.)

GOTHIC — A fiction deploying ruined castles, secret passages, \
supernatural or quasi-supernatural agency, persecuted heroines, \
and an atmosphere of terror. Look for: medieval/Catholic settings, \
mysterious manuscripts, ghosts or explained-supernatural, tyrannical \
villains, underground passages. (Walpole's Otranto, Radcliffe's \
Udolpho, Lewis's Monk.)

EPISTOLARY — A fiction told through letters, journals, or documents \
exchanged between characters. Look for: "I write to you...", \
multiple correspondents, real-time narration through letters, \
editorial framing. (Richardson's Pamela/Clarissa, Burney's Evelina, \
Smollett's Humphry Clinker.)

SENTIMENTAL — A fiction prioritizing fine feeling, sympathetic \
identification, and benevolence. Look for: tearful scenes, characters \
valued for emotional sensitivity, appeals to reader's sympathy, \
moral beauty of humble beings. (Sterne's Sentimental Journey, \
Mackenzie's Man of Feeling, Goldsmith's Vicar.)

ALLEGORY — A narrative in which characters, places, and events \
systematically stand for abstract concepts, doctrines, or historical \
referents. Look for: characters named after virtues/vices, a journey \
representing spiritual or moral progress, transparent symbolic mapping. \
(Bunyan's Pilgrim's Progress, The Holy War, The Isle of Man.)

PASTORAL — A narrative set in an idealized rural landscape, featuring \
shepherds, nymphs, or courtiers in pastoral disguise, with themes of \
love, retreat from court life, and the contrast between city and country. \
Look for: Arcadian settings, shepherdesses, poetic interludes, idyllic \
natural descriptions. (Sidney's Arcadia, Lodge's Rosalynde, d'Urfé's \
L'Astrée.)

HISTORICAL FICTION — A narrative set in a recognizable historical period, \
incorporating real historical events, figures, or settings as a backdrop \
for fictional characters and plots. Look for: named wars, battles, or \
political events; real historical figures as characters; period-specific \
social conditions. (Defoe's Memoirs of a Cavalier, Journal of the Plague \
Year, Scott's Waverley.)

## Notes

- Many texts combine subgenres: a robinsonade can also be a spiritual \
autobiography (Robinson Crusoe), a domestic fiction can also be conduct \
fiction (Pamela).
- If the summary is too brief to determine subgenre, return an empty list.
- The subgenre list is for FICTION ONLY. If is_fiction=False, return empty subgenres."""


EXAMPLES = [
    (
        "Summary: Robinson Crusoe, shipwrecked on a desert island, survives alone "
        "for years. He builds shelter, grows crops, tames goats, and keeps a journal. "
        "He rescues a native he names Friday. After 28 years he is rescued by a passing "
        "ship. Throughout he reflects on God's providence and his earlier sinful life.",
        SubgenreAnnotation(
            is_fiction=True,
            non_fiction_type=None,
            subgenres=['robinsonade', 'pseudo_memoir', 'spiritual_autobiography'],
        ),
    ),
    (
        "Summary: The narrator, a golden guinea coin, passes from hand to hand through "
        "London society. Each owner — a merchant, a prostitute, a politician, a beggar — "
        "reveals their vices and follies. The coin observes and comments on the moral "
        "corruption it witnesses at every social level.",
        SubgenreAnnotation(
            is_fiction=True,
            non_fiction_type=None,
            subgenres=['menippean_satire'],
        ),
    ),
    (
        "Summary: A series of sermons on the duties of Christian charity, delivered "
        "at St. Paul's Cathedral. Each sermon takes a biblical text and applies it "
        "to contemporary London life, exhorting listeners to generosity.",
        SubgenreAnnotation(
            is_fiction=False,
            non_fiction_type='sermon',
            subgenres=[],
        ),
    ),
]


DEFAULT_MODEL = 'lmstudio/qwen/qwen3.6-35b-a3b'


class SubgenreTask(Task):
    name = "classify_subgenre"
    model = DEFAULT_MODEL
    schema = SubgenreAnnotation
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
