"""Modern subgenre classification from social network summaries (post-1800).

Second-pass task for C19-C20 fiction: takes the summary chain from a
SocialNetworkTask result and classifies fine-grained subgenre plus
fiction verification. Vocabulary drawn from post-1800 genre scholarship
(Watt, Armstrong, D.A. Miller, Moretti, Jameson, etc.).

Companion to SubgenreTask (pre-1800). Same format, different tag vocabulary.

Usage:
    from largeliterarymodels.tasks import ModernSubgenreTask

    task = ModernSubgenreTask(model="claude-sonnet-4-6")
    result = task.run(ModernSubgenreTask.format_input(sn_json))
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
    # C19 realist tradition
    'novel_of_manners',
    'condition_of_england',
    'sensation_novel',
    'naturalism',
    'regional_novel',
    'silver_fork_novel',
    'new_woman_fiction',
    'newgate_novel',

    # C19 popular forms
    'detective_fiction',
    'sea_story',
    'adventure',
    'imperial_romance',
    'scientific_romance',

    # C19-C20 continuing forms
    'bildungsroman',
    'historical_fiction',
    'gothic',
    'epistolary',
    'domestic_fiction',
    'sentimental',
    'picaresque',
    'satire',
    'allegory',

    # C20 literary movements
    'modernist',
    'stream_of_consciousness',
    'postmodern',
    'metafiction',
    'magical_realism',
    'autofiction',
    'minimalist',
    'maximalist',
    'existentialist',

    # C20 specific forms
    'southern_gothic',
    'campus_novel',
    'proletarian_novel',
    'spy_fiction',
    'hardboiled',
    'noir',
    'dystopian',
    'psychological_thriller',
    'domestic_noir',

    # Speculative
    'science_fiction',
    'utopian',
    'weird_fiction',
    'climate_fiction',

    # Additional
    'romance_novel',
    'postcolonial_novel',
    'war_novel',
    'horror',
    'western',
    'thriller',
]


class ModernSubgenreAnnotation(BaseModel):
    is_fiction: bool = Field(
        description=(
            "Is this text prose fiction (novel, romance, tale, novella)? "
            "False for non-fiction miscatalogued in the corpus."
        )
    )
    non_fiction_type: Optional[Literal[
        'sermon', 'treatise', 'history', 'biography', 'periodical',
        'essay', 'letters', 'speech', 'legal', 'reference', 'criticism',
        'poetry', 'drama', 'almanac',
    ]] = Field(
        default=None,
        description="If is_fiction is False, what type of non-fiction? Null if fiction.",
    )
    subgenres: list[Literal[
        'novel_of_manners',
        'condition_of_england',
        'sensation_novel',
        'naturalism',
        'regional_novel',
        'silver_fork_novel',
        'new_woman_fiction',
        'newgate_novel',
        'detective_fiction',
        'sea_story',
        'adventure',
        'imperial_romance',
        'scientific_romance',
        'bildungsroman',
        'historical_fiction',
        'gothic',
        'epistolary',
        'domestic_fiction',
        'sentimental',
        'picaresque',
        'satire',
        'allegory',
        'modernist',
        'stream_of_consciousness',
        'postmodern',
        'metafiction',
        'magical_realism',
        'autofiction',
        'minimalist',
        'maximalist',
        'existentialist',
        'southern_gothic',
        'campus_novel',
        'proletarian_novel',
        'spy_fiction',
        'hardboiled',
        'noir',
        'dystopian',
        'psychological_thriller',
        'domestic_noir',
        'science_fiction',
        'utopian',
        'weird_fiction',
        'climate_fiction',
        'romance_novel',
        'postcolonial_novel',
        'war_novel',
        'horror',
        'western',
        'thriller',
    ]] = Field(
        default_factory=list,
        description="Fine-grained subgenre labels. Select all that apply (typically 1-4).",
    )


SYSTEM_PROMPT = """\
You are an expert in the history of English prose fiction from 1800 to the present, \
classifying texts by fine-grained subgenre from plot summaries.

You will receive a SUMMARY of a text extracted from a social network analysis. \
First determine whether the text is actually fiction. Then classify its subgenre(s).

## Step 1: Is this fiction?

If the summary describes non-fiction (biography, history, essay collection, etc.), \
mark is_fiction=False and identify the type.

## Step 2: Subgenre classification

Select all subgenres that apply (typically 1-4). Only assign a label if you see \
clear evidence in the summary.

### C19 REALIST TRADITION

NOVEL OF MANNERS — Plot driven by social customs, courtship, class distinctions, \
and moral judgments within a recognizable stratified community. The drama is in \
violations of decorum, match-making, and reputation. Distinguished from domestic \
fiction by emphasis on social performance rather than family life. \
(Austen's successors, Trollope, James, Wharton.)

CONDITION OF ENGLAND / SOCIAL-PROBLEM NOVEL — Directly engages with industrial \
labor, class antagonism, urban poverty, or political reform. Look for: factory \
workers, strikes, Parliamentary debate, explicit contrast between rich and poor. \
(Gaskell's Mary Barton/North and South, Dickens's Hard Times, Kingsley, Disraeli.)

SENSATION NOVEL — Combines domestic realism's surface with gothic/criminal plots: \
bigamy, fraud, secret identities, falsified death, asylum incarceration, \
inheritance schemes. Look for: respectable surfaces hiding dark secrets, \
identity revelation as climax. (Collins's Woman in White/Moonstone, Braddon's \
Lady Audley, Mrs. Henry Wood's East Lynne.)

NATURALISM — Deterministic, clinical depiction of characters trapped by heredity, \
environment, economics, or addiction. Look for: inevitable decline, squalid \
conditions described unflinchingly, characters who cannot escape their fate. \
(Gissing, Moore, Hardy at his darkest, Crane, Norris, Dreiser, Zola-influenced.)

REGIONAL NOVEL — Foregrounds dialect, folkways, landscape, and local custom of a \
specific region. Look for: detailed place description, regional speech patterns, \
community as character. (Hardy's Wessex, Jewett, Harte, Eggleston, later Faulkner.)

SILVER-FORK NOVEL — Depicts fashionable aristocratic London life with copious \
detail of balls, clubs, dress, and etiquette. Look for: Almack's, dandies, \
high-society intrigue, name-dropping of London venues. (Hook, Gore, Bulwer-Lytton, \
Disraeli's early novels. 1825-1845.)

NEW WOMAN FICTION — Features independent, sexually frank, professionally ambitious \
women protagonists challenging Victorian gender norms. Look for: female education, \
refusal of marriage, sexual autonomy, feminist arguments. (Grand, Schreiner, \
Egerton, Allen. 1880s-1890s.)

NEWGATE NOVEL — Draws characters from criminal history, glamorizing highwaymen \
or thieves. Look for: named historical criminals, romanticized crime, explicit \
Newgate Calendar source material. (Bulwer-Lytton's Paul Clifford/Eugene Aram, \
Ainsworth's Rookwood/Jack Sheppard. 1830s-1840s.)

### C19 POPULAR FORMS

DETECTIVE FICTION — Plot structured around the investigation and solution of a \
crime by a detective figure. Look for: a mystery to solve, clue-gathering, \
logical deduction, revelation of the criminal. (Poe's Dupin, Collins, \
Conan Doyle, Golden Age whodunits, hardboiled.)

SEA STORY — Plot centered on maritime life: voyages, naval discipline, storms, \
battles at sea, mutiny. Look for: ships, captains, sailors, ocean settings. \
(Marryat, Cooper's sea novels, Melville, Conrad's maritime fiction, O'Brian.)

ADVENTURE — Plot driven by physical danger, exotic settings, and action rather \
than psychological development. Look for: quests, escapes, fights, journeys \
through dangerous territory. (Stevenson, Haggard, Kipling, Buchan.)

IMPERIAL ROMANCE — Adventure set at the edges of the British Empire, featuring \
encounters with colonized peoples, treasure hunts, or military expeditions in \
Africa, India, or the Pacific. (Haggard's King Solomon's Mines, Kipling's Kim, \
Buchan, Edgar Wallace.)

SCIENTIFIC ROMANCE — Proto-science fiction using speculative science or technology \
as plot driver: time travel, alien invasion, invisibility, evolution. \
(Wells's Time Machine/War of the Worlds/Invisible Man, Verne in translation.)

### CONTINUING FORMS (C19-C20)

BILDUNGSROMAN — Protagonist's growth, education, or moral development structures \
the plot from youth to maturity. (Dickens's David Copperfield/Great Expectations, \
Brontë's Jane Eyre, Joyce's Portrait, Woolf's The Waves.)

HISTORICAL FICTION — Set in a recognizable past with real events or figures as \
backdrop. (Scott's successors, Eliot's Romola, Tolstoy-influenced, Hilary Mantel.)

GOTHIC — Atmosphere of dread, supernatural or quasi-supernatural elements, \
confined spaces, secrets, persecution. Continues from C18 through C19 \
(Brontës, Stoker, du Maurier) to modern horror.

EPISTOLARY — Told through letters, diaries, or documents. (Continues from C18; \
Stoker's Dracula, Walker's The Color Purple, modern variants.)

DOMESTIC FICTION — Centers on household life, family relationships, marriage. \
The private sphere as primary arena. (Much Victorian fiction, mid-C20 women's \
fiction, domestic noir's literary ancestor.)

SENTIMENTAL — Prioritizes emotional sensitivity, tearful scenes, sympathetic \
identification. (Continues from C18 into Victorian melodrama and weepy.)

PICARESQUE — Episodic, low-born protagonist drifting through society. \
(Dickens's early novels, Twain, Bellow's Augie March.)

SATIRE — Primary mode is social critique through irony, exaggeration, or \
ridicule. (Thackeray, Waugh, Huxley, Amis.)

ALLEGORY — Systematic symbolic meaning beneath the surface narrative. \
(Bunyan's heirs, Orwell's Animal Farm, Golding's Lord of the Flies.)

### C20 LITERARY MOVEMENTS

MODERNIST — Formal experimentation: fragmented chronology, stream of \
consciousness, mythic scaffolding, difficulty, self-conscious style. \
(Joyce, Woolf, Faulkner, Ford, Richardson. 1910s-1940s.)

STREAM OF CONSCIOUSNESS — Specifically foregrounds unmediated interior thought-flow. \
Look for: unpunctuated runs, associative logic, sensory impressions unfiltered \
by narrator. (Woolf, Joyce, Faulkner, Richardson.)

POSTMODERN — Metafiction, paranoid plotting, ontological play, ironic \
self-awareness, encyclopedic scope, genre-mixing. (Pynchon, DeLillo, \
Barth, Coover, Nabokov, Rushdie.)

METAFICTION — The novel foregrounds its own fictionality: characters aware \
they're in a book, narrative commenting on narrative. (Barth, Fowles, \
Calvino, Spark, B.S. Johnson.)

MAGICAL REALISM — The marvelous treated as ordinary within otherwise realistic \
fiction. Look for: impossible events narrated without surprise. \
(Morrison, Rushdie, Carter, Erdrich, Okri.)

AUTOFICTION — First-person fiction drawing openly on the author's life without \
claiming strict autobiography. (Lerner, Cusk, Heti, Knausgård.)

MINIMALIST — Spare diction, working-class subjects, emotional reticence, \
white space. (Carver, Beattie, Wolff, Ford.)

MAXIMALIST — Long, encyclopedic, language-saturated, digressive. \
(Pynchon, Gaddis, Wallace, Vollmann.)

EXISTENTIALIST — Characters confront meaninglessness, absurdity, radical freedom. \
Plot structured around existential crisis rather than social conflict. \
(Wright's The Outsider, Bellow, Percy, Murdoch.)

### C20 SPECIFIC FORMS

SOUTHERN GOTHIC — Southern US setting with grotesque characters, racial haunting, \
decayed aristocracy, and gothic atmosphere. (Faulkner, O'Connor, McCullers, \
Capote, Welty, McCarthy.)

CAMPUS NOVEL — Set in a university; academics as characters, intellectual \
life as subject. (Amis's Lucky Jim, Lodge, Tartt's Secret History.)

PROLETARIAN NOVEL — Working-class characters, labor struggle, class consciousness. \
(Steinbeck, Sillitoe, Tressell, Greenwood.)

SPY FICTION — Espionage, intelligence services, Cold War or geopolitical intrigue. \
(Le Carré, Deighton, Fleming, Ambler.)

HARDBOILED — Cynical detective, urban corruption, vernacular prose, violent realism. \
(Hammett, Chandler, Ross Macdonald.)

NOIR — First-person criminal/loser perspective, existential bleakness, no redemption. \
(Cain, Thompson, Highsmith, Woolrich.)

DYSTOPIAN — An oppressive future or alternate society, systematic control, \
resistance or submission. (Huxley, Orwell, Atwood, Ishiguro.)

PSYCHOLOGICAL THRILLER — Suspense driven by unreliable perception, obsession, \
or mental instability rather than physical danger. (Highsmith, Rendell, \
Harris, modern unreliable-narrator thrillers.)

DOMESTIC NOIR — Psychological thriller set in homes and marriages, female POV, \
secrets within intimate relationships. (Flynn's Gone Girl, Hawkins, Moriarty.)

### SPECULATIVE

SCIENCE FICTION — Extrapolation from science or technology structures the world \
and plot. (Wells onwards through Golden Age, New Wave, cyberpunk, contemporary.)

UTOPIAN — Depicts an ideal or improved society. (Bellamy, Morris, Le Guin, \
Piercy, Robinson.)

WEIRD FICTION — Cosmic dread, atmosphere over plot, reality itself as unstable. \
(Lovecraft, Aickman, VanderMeer, Miéville.)

CLIMATE FICTION — Climate change as central concern: ecological catastrophe, \
adaptation, loss. (Ballard, Robinson, Bacigalupi, Kingsolver.)

### ADDITIONAL

ROMANCE NOVEL — Plot centered on a love story with an emotionally satisfying \
ending (HEA/HFN). The romantic relationship IS the main plot, not a subplot. \
Look for: meet-cute, obstacles to union, declaration, resolution into partnership. \
Distinguished from novels that contain romance by the structural centrality of \
the love plot. (Heyer, Harlequin/Mills & Boon, contemporary romance.)

POSTCOLONIAL NOVEL — Engages with the legacy of colonialism: imperial power \
relations, cultural displacement, hybrid identity, the experience of colonized \
or formerly colonized peoples. Look for: settings in former colonies, tensions \
between indigenous and imperial cultures, migration, identity crisis rooted in \
colonial history. (Achebe, Naipaul, Rushdie, Kincaid, Coetzee, Adichie.)

WAR NOVEL — Combat, military life, or the direct experience of war as central \
subject. Look for: battles, soldiers, front lines, trauma of combat, military \
hierarchy. Distinguished from historical fiction by the centrality of warfare \
itself. (Remarque, Hemingway's Farewell to Arms, Heller's Catch-22, \
O'Brien's The Things They Carried, Barker's Regeneration.)

HORROR — Fiction designed to evoke fear, dread, or revulsion. Supernatural or \
natural threats; monsters, serial killers, hauntings, body horror. Distinguished \
from gothic (which emphasizes atmosphere and setting) by the centrality of \
fear as the intended reader response. (King, Straub, Barker, Shirley Jackson, \
Koontz.)

WESTERN — Set in the American frontier (or analogous frontier), featuring \
cowboys, outlaws, lawmen, Native Americans, open landscape. Look for: \
horses, gunfights, frontier justice, wilderness survival, ranches. \
(Grey, L'Amour, McMurtry's Lonesome Dove, McCarthy's Blood Meridian.)

THRILLER — Plot driven by suspense, danger, and high stakes with rapid pacing. \
Distinguished from detective fiction (which centers investigation) and from \
horror (which centers fear) by the emphasis on pursuit, countdown, or \
conspiracy. (Ludlum, Forsyth, Clancy, le Carré's action-oriented work.)

## Notes

- Many texts combine subgenres: a sensation novel can also be detective fiction \
(The Moonstone), a modernist novel can be a bildungsroman (Portrait of the Artist).
- If the summary is too brief to determine subgenre, return an empty list.
- Return empty subgenres if is_fiction=False."""


EXAMPLES = [
    (
        "Summary: In a northern English industrial town, Margaret Hale adjusts to "
        "life after moving from the rural south. She witnesses a cotton workers' "
        "strike, is caught between the manufacturer Thornton and the dying union "
        "organizer Higgins, and gradually comes to understand both industrial "
        "capitalism and the dignity of labor. She eventually marries Thornton "
        "after inheriting money that saves his failing mill.",
        ModernSubgenreAnnotation(
            is_fiction=True,
            non_fiction_type=None,
            subgenres=['condition_of_england', 'domestic_fiction', 'bildungsroman'],
        ),
    ),
    (
        "Summary: Lady Audley, beautiful wife of Sir Michael, is discovered to "
        "have faked her death, abandoned her child, committed bigamy, and pushed "
        "her first husband down a well. Her nephew Robert Audley investigates and "
        "exposes her. She is confined to a Belgian asylum.",
        ModernSubgenreAnnotation(
            is_fiction=True,
            non_fiction_type=None,
            subgenres=['sensation_novel', 'detective_fiction'],
        ),
    ),
    (
        "Summary: Stephen Dedalus grows from infancy through school, sexual "
        "awakening, religious crisis, and university to reject family, church, "
        "and nation in favor of art. The prose style evolves with his "
        "consciousness from childlike to aesthetically complex.",
        ModernSubgenreAnnotation(
            is_fiction=True,
            non_fiction_type=None,
            subgenres=['modernist', 'bildungsroman', 'stream_of_consciousness', 'autofiction'],
        ),
    ),
]


DEFAULT_MODEL = 'claude-sonnet-4-6'


class ModernSubgenreTask(Task):
    name = "classify_subgenre_modern"
    model = DEFAULT_MODEL
    schema = ModernSubgenreAnnotation
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
