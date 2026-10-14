"""Proppian plot-function annotation for short narrative texts.

Annotates a whole short story (~1,500 words fits comfortably) with Vladimir
Propp's plot functions, in order, each anchored to an exact quoted span.

Source of the scheme: Propp, *Morphology of the Folktale*, 2nd ed., trans.
Laurence Scott, rev. Louis A. Wagner (Austin: University of Texas Press,
1968). Function definitions here are condensed from Propp's own wording,
verified against the printed page rather than against secondary summaries —
which garble the notation often enough to matter. Two live examples: a
peer-reviewed summary table reverses J and I (branding is J, victory is I,
and J *precedes* I), and a widely-circulated retyping collapses Greek α
(initial situation) into Latin a (lack), which are different functions.

SCOPE. Propp derived these from Russian wondertales (Afanasyev), and scoped
his claims there. Applied to literary short fiction the honest expectation
is partial or empty results, and `is_tale_structured` exists so that an
empty result is a finding rather than a failure. A Chekhov story returning
no functions is the instrument working.

Design notes:

  * This is an EXTRACT-path task, not a SequentialTask — the whole story
    goes in one call. That buys byte-identical instrument serialisation
    (`task.instrument_text()`), prompt caching, and the batch transport,
    none of which SequentialTask gets.

  * Only the 34 top-level symbols are offered (Propp's 31 functions, plus
    α, plus `a`, plus his residual sign X). Propp enumerates ~150 numbered
    sub-variants; a 187-way discrimination per span is much harder than a
    34-way one, and the variants are better resolved by a second pass that
    already knows the function. The full variant table lives beside this
    module as reference data.

  * `move` matters more than it looks. Propp's functions are ordered WITHIN
    a move, not across the tale, and a tale can contain several. Without a
    move index a two-move story looks like a violation of Propp's central
    ordering claim — villainy after a wedding — when it is perfectly
    canonical. That would measure the schema's blindness and read as
    evidence against Propp.

  * `event_order` exists because the model reads the order of the TELLING
    while Propp's sequence is a claim about the order of EVENTS. For a
    plain wondertale these coincide; for literary short fiction that opens
    in medias res they do not, and the disagreement is itself informative.

  * Spans must be verbatim so they can be checked. `locate_spans()` is the
    audit: it returns character offsets and flags any span the model
    paraphrased, turning a hallucinated quote into a counted failure rather
    than a silent one. It is a helper rather than a pydantic validator
    because a validator cannot see the source text.

Usage:
    from largeliterarymodels.tasks import ProppTask

    task = ProppTask(model="claude-sonnet-5")
    result = task.run(story_text)
    for f in result.functions:
        print(f.move, f.function, repr(f.span[:60]))

    from largeliterarymodels.tasks import locate_spans
    located = locate_spans(story_text, result)   # offsets + verbatim flags
"""

from typing import Literal
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task
from largeliterarymodels.tasks._propp_examples import EXAMPLES


# ── Propp's functions: the 34 top-level symbols ────────────────────────
# Ordered as Propp orders them. `stage` follows Propp's OWN sectioning
# (Appendix I, "Materials for a Tabulation of the Tale", pp. 119-127), not
# the six-stage scheme common in secondary literature — which differs in
# two substantive ways. There is no separate "transference" stage: G sits
# with the struggle, and the donor bloc is D-E-F only. And struggle and
# return are not separate stages: H J I K ↓ Pr Rs are one continuous
# block. Propp also treats H-I and M-N as ALTERNATIVE central developments
# rather than sequential ones, so a vocabulary ordering struggle before
# recognition encodes a sequence he explicitly calls a fork.

PROPP_FUNCTIONS = (
    {
        'name': 'initial_situation',
        'symbol': 'α',
        'title': 'Initial situation',
        'stage': 'initial_situation',
        'description': (
            "The tale's opening tableau: members of a family are enumerated, or "
            'the future hero is introduced by name or status, often together '
            'with a description of unusual prosperity that will set off the '
            'coming misfortune. Propp states explicitly that this is NOT one of '
            'the 31 functions but a preparatory morphological element. Annotate '
            'the span before the first function, excluding narratorial framing '
            'that stands outside story time.'
        ),
    },
    {
        'name': 'absentation',
        'symbol': 'β',
        'title': 'Absentation',
        'stage': 'preparatory_section',
        'description': (
            'One of the members of a family absents himself from home. The '
            'absence creates the unguarded moment the villain will exploit; it '
            "is the absence itself, not the journey's content, that is the "
            'function.'
        ),
    },
    {
        'name': 'interdiction',
        'symbol': 'γ',
        'title': 'Interdiction',
        'stage': 'preparatory_section',
        'description': (
            'An interdiction is addressed to the hero. The prohibition may be '
            'softened into a request or a piece of advice, or strengthened by '
            'physically confining the children; it need not be connected to an '
            'absentation.'
        ),
    },
    {
        'name': 'violation',
        'symbol': 'δ',
        'title': 'Violation',
        'stage': 'preparatory_section',
        'description': (
            'The interdiction is violated. Functions II and III form a paired '
            'element and the forms of violation correspond one-to-one to the '
            'forms of interdiction ; the second half of the pair can occur '
            'without the first, the prohibition being left unstated. It is at '
            'this point that the villain enters the tale.'
        ),
    },
    {
        'name': 'reconnaissance',
        'symbol': 'ε',
        'title': 'Reconnaissance',
        'stage': 'preparatory_section',
        'description': (
            'The villain makes an attempt at reconnaissance — an effort to '
            'obtain information, usually the whereabouts of children or of '
            'precious objects.'
        ),
    },
    {
        'name': 'delivery',
        'symbol': 'ζ',
        'title': 'Delivery',
        'stage': 'preparatory_section',
        'description': (
            'The villain receives information about his victim. Paired with '
            'reconnaissance and often realised as dialogue; the second half of '
            'the pair can exist without the first, in which case the '
            'information is given away by a careless act.'
        ),
    },
    {
        'name': 'trickery',
        'symbol': 'η',
        'title': 'Trickery',
        'stage': 'preparatory_section',
        'description': (
            'The villain attempts to deceive his victim in order to take '
            'possession of him or of his belongings. The villain first assumes '
            'a disguise (a dragon turns into a golden goat, a witch into a '
            'sweet old lady), then the function proper follows.'
        ),
    },
    {
        'name': 'complicity',
        'symbol': 'θ',
        'title': 'Complicity',
        'stage': 'preparatory_section',
        'description': (
            'The victim submits to deception and thereby unwittingly helps his '
            'enemy. Note the asymmetry Propp draws: interdictions are always '
            'broken and deceitful proposals are always accepted.'
        ),
    },
    {
        'name': 'villainy',
        'symbol': 'A',
        'title': 'Villainy',
        'stage': 'complication',
        'description': (
            'The villain causes harm or injury to a member of a family. Propp '
            'calls this the most important function: it creates the actual '
            'movement of the tale, and the complication begins here. A and a '
            '(Lack) are ALTERNATIVES occupying the same slot (VIII / VIIIa) — '
            "'elements A or a are required for each tale' — neither is a "
            'subtype of the other. The villain often commits two or three '
            'harmful acts at once.'
        ),
    },
    {
        'name': 'lack',
        'symbol': 'a',
        'title': 'Lack',
        'stage': 'complication',
        'description': (
            'One member of a family either lacks something or desires to have '
            "something (function VIIIa). Propp's ALTERNATIVE to Villainy in the "
            "same slot, not a subtype of it: 'lack can be considered the "
            "morphological equivalent of seizure' — in villainy a lack is "
            'created from without, in lack it is realised from within, and '
            'either provokes the same quest. Lowercase Latin a; do not confuse '
            'with the Greek α of the initial situation.'
        ),
    },
    {
        'name': 'mediation',
        'symbol': 'B',
        'title': 'Mediation, the connective incident',
        'stage': 'complication',
        'description': (
            'Misfortune or lack is made known; the hero is approached with a '
            'request or command, or he is allowed to go, or he is dispatched. '
            'This is the function that brings the hero into the tale and causes '
            'his departure. Distinguishes the SEEKER hero (dispatched, or asked '
            'to go) from the VICTIMIZED hero (banished, released, or carried '
            'off).'
        ),
    },
    {
        'name': 'beginning_counteraction',
        'symbol': 'C',
        'title': 'Beginning counteraction',
        'stage': 'complication',
        'description': (
            "The seeker agrees to or decides upon counteraction: 'Permit us to "
            "go in search of your princess.' Sometimes unspoken, but a "
            'volitional decision precedes the search. Characteristic ONLY of '
            'seeker-heroes; banished, vanquished, bewitched and substituted '
            'heroes show no such volition and the element is simply absent.'
        ),
    },
    {
        'name': 'departure',
        'symbol': '↑',
        'title': 'Departure',
        'stage': 'complication',
        'description': (
            'The hero leaves home. Distinct from the temporary absence of β: ↑ '
            'designates the route along which the narrative is developed, '
            'whether the hero is a seeker (departure with search as its goal) '
            'or a victim (a journey without search, on which adventures await). '
            'Sometimes intensified into flight; sometimes absent, the whole '
            'action occurring in one place. The elements A B C ↑ constitute the '
            'complication.'
        ),
    },
    {
        'name': 'donor_first_function',
        'symbol': 'D',
        'title': 'First function of the donor',
        'stage': 'donors',
        'description': (
            'The hero is tested, interrogated, attacked etc., which prepares '
            'the way for his receiving either a magical agent or a helper. A '
            'new character, the donor or provider, enters, usually encountered '
            'by accident. Distinguish a fight here from function H by its '
            'result: if the hero gains an agent for further searching it is D, '
            'if he gains the object of his quest it is H.'
        ),
    },
    {
        'name': 'hero_reaction',
        'symbol': 'E',
        'title': "The hero's reaction",
        'stage': 'donors',
        'description': (
            'The hero reacts to the actions of the future donor. In the '
            'majority of instances the reaction is either positive or '
            'negative,.'
        ),
    },
    {
        'name': 'receipt_of_magical_agent',
        'symbol': 'F',
        'title': 'Provision or receipt of a magical agent',
        'stage': 'donors',
        'description': (
            'The hero acquires the use of a magical agent. The agents may be '
            'animals, objects out of which helpers appear, objects with magical '
            'properties, or capacities directly given (e.g. transformation into '
            'animals). Propp groups the transmission forms into two types: '
            'seizure (linked to an attempt to destroy the hero, a request for '
            'apportionment, or a proposed exchange, i.e. unfriendly or deceived '
            'donors), and all other forms (friendly donors).'
        ),
    },
    {
        'name': 'spatial_transference',
        'symbol': 'G',
        'title': 'Spatial transference between two kingdoms, guidance',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            'The hero is transferred, delivered, or led to the whereabouts of '
            "an object of search. The object generally lies in 'another' "
            'kingdom, far away horizontally or very high up or deep down. Where '
            'the hero simply walks there — G being a natural continuation of ↑ '
            '— Propp does not single the function out.'
        ),
    },
    {
        'name': 'struggle',
        'symbol': 'H',
        'title': 'Struggle',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            'The hero and the villain join in direct combat. To be '
            'distinguished from a fight with a hostile donor (see D) by the '
            'result: if the hero wins the very object of his quest, this is H. '
            'H-I (struggle-victory) is one of the two alternative developments '
            'a move can take, the other being M-N (difficult task-solution).'
        ),
    },
    {
        'name': 'branding',
        'symbol': 'J',
        'title': 'Branding, marking',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            "The hero is branded. Note the ordering: in Propp's sequence "
            'branding (XVII, J) comes BETWEEN struggle (XVI, H) and victory '
            '(XVIII, I). Corresponds to recognition (Q) later in the tale.'
        ),
    },
    {
        'name': 'victory',
        'symbol': 'I',
        'title': 'Victory',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            'The villain is defeated (function XVIII). Paired with struggle. '
            'Note the counter-intuitive symbol assignment: victory is I and '
            "branding is J, and I follows J in Propp's order."
        ),
    },
    {
        'name': 'liquidation_of_lack',
        'symbol': 'K',
        'title': 'Liquidation of the initial misfortune or lack',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            'The initial misfortune or lack is liquidated (function XIX). This '
            'function together with villainy (A) constitutes a pair, and the '
            'narrative reaches its peak here.'
        ),
    },
    {
        'name': 'return',
        'symbol': '↓',
        'title': 'Return',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            'The hero returns. Generally accomplished by the same forms as the '
            'outward arrival, and sometimes having the nature of flight. Propp '
            'attaches no further function after a return because returning '
            'already implies a surmounting of space.'
        ),
    },
    {
        'name': 'pursuit',
        'symbol': 'Pr',
        'title': 'Pursuit, chase',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            'The hero is pursued.'
        ),
    },
    {
        'name': 'rescue',
        'symbol': 'Rs',
        'title': 'Rescue from pursuit',
        'stage': 'helper_to_end_of_first_move',
        'description': (
            'The hero is rescued from pursuit. Paired with Pr; a great many '
            'tales end on this note, though a second move may follow.'
        ),
    },
    {
        'name': 'unrecognized_arrival',
        'symbol': 'o',
        'title': 'Unrecognized arrival',
        'stage': 'second_move',
        'description': (
            'The hero, unrecognized, arrives home or in another country. Propp '
            'distinguishes two classes but assigns them no separate '
            'designations: (1) arrival home, where the hero stays with an '
            'artisan — goldsmith, tailor, shoemaker — as an apprentice; (2) '
            "arrival at some king's court, where he serves as cook or groom. "
            'Simple arrival sometimes needs designating too. Lowercase Latin o.'
        ),
    },
    {
        'name': 'unfounded_claims',
        'symbol': 'L',
        'title': 'Unfounded claims',
        'stage': 'second_move',
        'description': (
            'A false hero presents unfounded claims. If the hero has arrived '
            'home the claims come from his brothers, who pose as the capturers '
            'of the prize; if he is serving in another kingdom they come from a '
            'general, a water-carrier or another, the general posing as the '
            'conqueror of the dragon. Propp says these two forms may be '
            'considered special classes but gives them no designations.'
        ),
    },
    {
        'name': 'difficult_task',
        'symbol': 'M',
        'title': 'Difficult task',
        'stage': 'second_move',
        'description': (
            "A difficult task is proposed to the hero — 'one of the tale's "
            "favorite elements'. Propp deliberately assigns NO subtype "
            'designations here, saying the tasks are so varied that each would '
            'need its own and no exact distribution will be made; he only '
            'groups them approximately: ordeal by food and drink; ordeal by '
            'fire; riddle guessing; ordeal of choice; hide and seek; tests of '
            'strength, adroitness and fortitude; tests of endurance; tasks of '
            'supply and manufacture; and others. M-N is the alternative to H-I '
            "as a move's central development."
        ),
    },
    {
        'name': 'solution',
        'symbol': 'N',
        'title': 'Solution',
        'stage': 'second_move',
        'description': (
            'The task is resolved. The forms of solution correspond exactly to '
            'the forms of task.'
        ),
    },
    {
        'name': 'recognition',
        'symbol': 'Q',
        'title': 'Recognition',
        'stage': 'second_move',
        'description': (
            'The hero is recognized: by a mark or brand (a wound, a star '
            'marking), by a thing given to him (a ring, a towel) — in which '
            'case recognition corresponds to branding, J — by his '
            'accomplishment of a difficult task (almost always preceded by an '
            'unrecognized arrival), or simply after a long separation, parents '
            'and children or brothers and sisters recognizing one another.'
        ),
    },
    {
        'name': 'exposure',
        'symbol': 'Ex',
        'title': 'Exposure',
        'stage': 'second_move',
        'description': (
            'The false hero or villain is exposed. Usually connected with '
            'recognition. Sometimes the result of an uncompleted task (the '
            "false hero cannot lift the dragon's heads); most often presented "
            'as a story told from the beginning, with the villain among the '
            'listeners giving himself away by expressions of disapproval; '
            'sometimes a song is sung that tells what happened.'
        ),
    },
    {
        'name': 'transfiguration',
        'symbol': 'T',
        'title': 'Transfiguration',
        'stage': 'second_move',
        'description': (
            'The hero is given a new appearance.'
        ),
    },
    {
        'name': 'punishment',
        'symbol': 'U',
        'title': 'Punishment',
        'stage': 'second_move',
        'description': (
            "The villain is punished: shot, banished, tied to a horse's tail, "
            'or he commits suicide. Usually only the villain of the second move '
            'and the false hero are punished; the first villain is punished '
            'only where battle and pursuit are absent, since otherwise he is '
            'killed in battle or perishes in the chase.'
        ),
    },
    {
        'name': 'wedding',
        'symbol': 'W',
        'title': 'Wedding',
        'stage': 'second_move',
        'description': (
            'The hero is married and ascends the throne. The tale draws to a '
            "close here. Propp's forms distinguish marriage, accession to the "
            'throne, marriage without accession, and mere monetary compensation '
            "in place of the princess's hand."
        ),
    },
    {
        'name': 'unclear',
        'symbol': 'X',
        'title': 'Unclear elements',
        'stage': 'unassigned',
        'description': (
            "Propp's residual sign, not one of the 31 functions. A few actions "
            'of tale heroes conform to none of the functions; they are either '
            'forms that cannot be understood without comparative material, or '
            'forms transferred from other genres (anecdotes, legends). Such '
            'cases are rare.'
        ),
    },
)

FUNCTION_VOCAB = tuple(f['name'] for f in PROPP_FUNCTIONS)
SYMBOL_BY_NAME = {f['name']: f['symbol'] for f in PROPP_FUNCTIONS}
STAGE_BY_NAME = {f['name']: f['stage'] for f in PROPP_FUNCTIONS}

#: Propp's own sectioning, in order. See the note above PROPP_FUNCTIONS for
#: why this is not the six-stage vocabulary of the secondary literature.
STORY_STAGES = (
    'initial_situation',
    'preparatory_section',
    'complication',
    'donors',
    'helper_to_end_of_first_move',
    'second_move',
    'unassigned',
)


def _function_glossary():
    """The function list as it appears in the instrument.

    Built once, at import, and embedded in the system prompt — so it is
    part of the administration's identity and any edit to it re-keys the
    stash. That is correct: a change to a definition is a change to the
    questionnaire, not a cosmetic edit.
    """
    lines = []
    for f in PROPP_FUNCTIONS:
        lines.append(
            f"- {f['name']} ({f['symbol']}, {f['title']}; stage: "
            f"{f['stage']}): {f['description']}"
        )
    return "\n".join(lines)


FUNCTION_GLOSSARY = _function_glossary()


# ── Schema ─────────────────────────────────────────────────────────────

class ProppFunctionInstance(BaseModel):
    function: Literal[FUNCTION_VOCAB] = Field(  # type: ignore[valid-type]
        description=(
            "Which Proppian function this span instantiates. Judge by the "
            "role the action plays in the plot, NOT by its surface content: "
            "the same act is a different function depending on where it "
            "stands, and different acts are the same function. A journey is "
            "not automatically 'departure'; a wedding is not automatically "
            "'wedding'."
        )
    )
    span: str = Field(
        description=(
            "The EXACT text from the story that instantiates this function, "
            "quoted verbatim and copied character for character. Do not "
            "paraphrase, summarise, correct spelling, or normalise "
            "punctuation — the span is checked against the source and a "
            "paraphrase is recorded as a failure. Keep it to the shortest "
            "stretch that carries the function, usually one or two sentences."
        )
    )
    move: int = Field(
        default=1,
        description=(
            "Which move this function belongs to, numbered from 1. A move is "
            "one complete development from a villainy or a lack through to "
            "its resolution. A tale may contain several: a second villainy "
            "after the first move has resolved opens move 2, and moves can "
            "also be embedded inside one another. Functions are ordered "
            "WITHIN a move, so a villainy in move 2 legitimately appears "
            "after a wedding in move 1. Omit only for a single-move tale, "
            "where it defaults to 1: a whole annotation used to be discarded "
            "because one item lacked this integer, which is a worse outcome "
            "than assuming the commonest case."
        )
    )
    event_order: int = Field(
        description=(
            "Position of this function in the order events actually HAPPENED "
            "in the story world, numbered from 1 across the whole text. For a "
            "story told straight through this simply matches the order you "
            "list them in. Where the story is told out of sequence — opening "
            "in the middle, or a character recounting earlier events — this "
            "field records the underlying chronology while the list order "
            "records the telling."
        )
    )
    note: str = Field(
        description=(
            "One sentence on why this span is this function rather than a "
            "neighbouring one, where the judgement was not obvious. Empty "
            "string when it was obvious. Say so here if the span is a poor "
            "fit that you assigned anyway."
        )
    )


class ProppAnnotation(BaseModel):
    is_tale_structured: bool = Field(
        description=(
            "True if this text is organised as a tale in Propp's sense — a "
            "villainy or lack that sets a plot in motion and is worked "
            "through to a resolution. False for texts that are not: "
            "character sketches, mood pieces, essays, anecdotes without a "
            "complication, most modern literary short fiction. Answer this "
            "on the text in front of you, not on what it resembles. A false "
            "answer here is an ordinary and expected outcome, and `functions` "
            "should then usually be empty or nearly so — do NOT manufacture "
            "functions to fill the list."
        )
    )
    n_moves: int = Field(
        description=(
            "How many distinct moves the text contains. 0 if it is not tale-"
            "structured, 1 for the ordinary single-move tale."
        )
    )
    functions: list[ProppFunctionInstance] = Field(
        description=(
            "The functions found, listed in the order they appear in the "
            "TEXT (the order of the telling). May be empty. The same function "
            "may appear more than once — Propp's tales repeat functions, and "
            "the donor sequence is classically trebled — so list each "
            "occurrence separately with its own span."
        )
    )
    overall_note: str = Field(
        description=(
            "One to three sentences on the text's structure as a whole: the "
            "shape of the move or moves, anything that resisted the scheme, "
            "and why. If the text is not tale-structured, say what it is "
            "instead."
        )
    )


# ── Instrument ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are annotating a short narrative text with Vladimir Propp's plot
functions, as set out in Morphology of the Folktale (1928).

THE CENTRAL DIFFICULTY, and the thing this annotation is for: a Proppian
function is defined by the ROLE an action plays in the plot, not by what
the action is. Propp's own point is that the same act is a different
function depending on where it stands, and that different acts can be the
same function. A hero riding away from home is 'departure' only if it opens
the route the narrative then follows; if he is merely stepping out and will
be back before anything happens, it is 'absentation'. A fight is 'struggle'
if the hero wins the object of his quest by it, and 'donor_first_function'
if he wins the means of continuing his search. Do not classify on
vocabulary. Ask what the action does to the plot.

THE FUNCTIONS

{FUNCTION_GLOSSARY}

RULES

1. Work through the text once, in reading order, and list every function
   you find. A function may occur more than once — Propp's tales repeat
   them, and the donor sequence is classically trebled. List each
   occurrence separately.

2. Quote spans VERBATIM. Copy the exact characters from the text, including
   its spelling and punctuation. The spans are checked against the source
   and a paraphrase is recorded as an error. Keep each span to the shortest
   stretch that carries the function.

3. Number the moves. A move is one complete development from a villainy or
   a lack through to its resolution. Most tales have one. A fresh villainy
   after the first has been resolved opens move 2 — the classic case is the
   hero's brothers stealing the prize on the way home — and moves can be
   embedded rather than merely consecutive. Functions are ordered WITHIN a
   move, so a villainy belonging to move 2 rightly appears after move 1's
   wedding.

4. Two alternations to keep straight. Villainy (A) and lack (a) are
   alternatives filling one slot, not one a kind of the other: a tale is
   set going by one or the other. And struggle/victory (H-I) and difficult
   task/solution (M-N) are alternative central developments — most tales
   take one road or the other, and a few take neither.

5. Watch the order of branding and victory. In Propp's sequence the hero is
   branded (J) BETWEEN the struggle (H) and the victory (I). The letters
   are counter-intuitive and are easy to transpose.

6. AN EMPTY OR NEARLY EMPTY ANSWER IS A LEGITIMATE ANSWER. Propp derived
   this scheme from Russian wondertales and limited his claims to them.
   Much short fiction is not built this way, and forcing it into the scheme
   produces worse than nothing. If the text has no villainy or lack driving
   it to a resolution, set is_tale_structured to false and return few or no
   functions. Do not pad the list.

7. Where a span is a poor fit that you assigned anyway, say so in its note.
   A recorded doubt is more useful than a confident label.

"""


class ProppTask(Task):
    """Annotate a short story with Propp's plot functions, in order.

    Sized for short texts — a 1,500-word story is about 2,000 tokens and
    goes in a single call, which is what lets the model see the whole
    sequence and judge move structure. Not suitable for novels: the unit
    Propp analyses is a tale, and a novel-length text would need the
    SequentialTask machinery and would lose the whole-sequence view that
    makes move numbering possible.

    Model note. The default rejects a caller-supplied `temperature` (see
    providers._NO_TEMPERATURE_MODELS), so runs on it are NOT
    temperature-pinned and `task.usage.report()['dropped_params']` will say
    so. That is a deliberate trade of sampling control for the reasoning
    this task needs; if a pinned temperature matters more than capability
    for your run, pass a model that accepts one and record which you used.
    """

    name = "ProppTask"
    schema = ProppAnnotation
    system_prompt = SYSTEM_PROMPT
    # Propp's own analyses of three tales, plus one negative case. See
    # _propp_examples: the function sequences are his, only the spans were
    # inferred, and they are chosen against a measured defect (zero-shot
    # precision 0.47 against his schemes, from over-annotating roughly 2x).
    examples = EXAMPLES
    model = "claude-sonnet-5"
    temperature = 0.0
    # A dense tale can yield 25-30 functions, each carrying a quoted span
    # and a note; 4096 truncates those and the JSON fails to parse.
    max_tokens = 8192


# ── Input preparation ──────────────────────────────────────────────────

import re as _re

_FOOTNOTE = _re.compile(r'\[\d{1,4}\]')
# Gutenberg-style emphasis: _puds_ -> puds. Bounded and single-line so a
# stray underscore cannot swallow a paragraph.
_ITALIC = _re.compile(r'_([^_\n]{1,80})_')
# A full-width rule near the top of a file is how the packaged tale texts
# separate a provenance header from the tale. The rule ALONE is not enough
# to identify one: a story can open with a scene divider, and treating that
# as a header silently deletes the opening (measured — "Once there was a
# king.\n\n--------\n\nHe had three sons." lost its first sentence). So a
# header must also LOOK like one: a `Key: value` line above the rule.
_RULE = _re.compile(r'^[=_-]{20,}\s*$', _re.M)
_HEADER_FIELD = _re.compile(r'^[A-Z][A-Za-z /()\'.-]{2,40}:\s*\S', _re.M)


def prepare_tale_text(text, strip_footnotes=True, strip_emphasis=True,
                 strip_header=True, join_wrapped=True):
    """Normalise a tale text before annotating it.

    Two transformations, both of which measurably change results rather
    than merely tidying:

    HARD LINE-WRAPPING. The 19th-century translations in data/propp are
    wrapped at ~54 characters. Any quoted span crossing a line break then
    fails an exact match, so the verbatim rate measures the typesetting
    instead of the model — on a first pass it read 9%, and 85% after
    unwrapping, with no change in model behaviour. Paragraph breaks are
    preserved; mid-sentence breaks are joined.

    INLINE FOOTNOTE MARKERS. The same editions carry editorial apparatus
    inside the sentence ("put forth his might,[283] and said"). 14 of the
    24 packaged tale texts contain them. A model quoting the sentence
    naturally omits the marker, and the span then fails verification for a
    reason that has nothing to do with the annotation. Set
    strip_footnotes=False to keep them.

    PROVENANCE HEADERS. Every packaged tale text opens with a header —
    Afanasyev number, titles, translator, source URL — closed by a
    full-width rule. It also carries a one-line content note ("the boy who
    understands birds and rises to greatness"), so feeding the file
    verbatim hands the annotator the tale's title AND a summary of its plot
    before it reads a word. That contaminates every judgement the task
    makes, and `is_tale_structured` most of all. Everything up to and
    including a rule in the first 20 lines is dropped. Set
    strip_header=False to keep it.

    EMPHASIS MARKUP. Gutenberg-style underscores ("thirty _puds_ of hemp",
    9 of 24 files) fail the same way, and the failure is worse because it
    reads as a fabricated quote: the span looks invented when the model
    merely dropped a typographic marker. Set strip_emphasis=False to keep
    it.

    NONE OF THIS IS A NO-OP ON CLEAN TEXT. Every option alters text that
    merely resembles the apparatus it targets: `foo_bar_baz` loses its
    underscores, a literal "[3]" disappears, and — the one that matters —
    text using SINGLE newlines between paragraphs is joined into one
    paragraph by join_wrapped. On material that carries no editorial
    apparatus (LLM output, clean modern prose) pass
    strip_footnotes=False, strip_emphasis=False, strip_header=False, and
    set join_wrapped only if the source really is hard-wrapped.

    Deliberately NOT part of the Task: this changes the item text, so it
    changes the cache key, and a silent normalisation inside run() would
    make two callers with the same file disagree about what they
    administered. Call it explicitly and the key records what was sent.
    """
    text = text.replace("\r\n", "\n")
    if strip_header:
        head = "\n".join(text.split("\n")[:20])
        m = _RULE.search(head)
        # Both conditions: a rule, AND a `Key: value` line above it. A rule
        # on its own is a scene divider far more often than a header.
        if m and _HEADER_FIELD.search(head[:m.start()]):
            text = text[m.end():]
    if strip_footnotes:
        text = _FOOTNOTE.sub("", text)
    if strip_emphasis:
        text = _ITALIC.sub(r'\1', text)
    if join_wrapped:
        text = _re.sub(r'\n{2,}', '\x00', text)
        text = _re.sub(r'[ \t]*\n[ \t]*', ' ', text)
        text = text.replace('\x00', '\n\n')
    text = _re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── Span verification ──────────────────────────────────────────────────

def _normalise_ws(text):
    """(normalised_text, index_map) collapsing whitespace runs to one space.

    index_map[i] is the offset in the ORIGINAL text of normalised char i, so
    a match found in normalised space can be reported against real offsets.
    """
    out, index_map = [], []
    prev_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_ws:
                continue
            out.append(" ")
            index_map.append(i)
            prev_ws = True
        else:
            out.append(ch)
            index_map.append(i)
            prev_ws = False
    return "".join(out), index_map


def locate_spans(text, annotation):
    """Locate each annotated span in `text`; the audit for quoted evidence.

    A span the model paraphrased is indistinguishable from one it quoted,
    unless someone checks — and an unchecked quote is exactly the kind of
    evidence that reads as verifiable and is not. This returns one record
    per function with character offsets and how the match was obtained:

        method='exact'       found character-for-character
        method='whitespace'  found after collapsing whitespace runs; offsets
                             are real, the quote is not literally verbatim
        method='not_found'   NOT IN THE TEXT — treat as fabricated

    Matching walks forward through the text, so a span that occurs twice is
    attributed to the occurrence after the previous function's — which is
    what the annotation's own ordering claims.

    Returns list[dict] with keys: index, function, symbol, move, span,
    start, end, method, verbatim.
    """
    norm, index_map = _normalise_ws(text)
    cursor = 0
    norm_cursor = 0
    out = []
    for i, fn in enumerate(annotation.functions):
        span = fn.span
        start = text.find(span, cursor)
        if start == -1:
            start = text.find(span)          # out of order, but present
        if start != -1:
            end = start + len(span)
            cursor = end
            norm_cursor = 0
            method, verbatim = "exact", True
        else:
            span_norm, _ = _normalise_ws(span.strip())
            j = norm.find(span_norm, norm_cursor)
            if j == -1:
                j = norm.find(span_norm)
            if j != -1 and span_norm:
                start = index_map[j]
                end = index_map[min(j + len(span_norm) - 1, len(index_map) - 1)] + 1
                cursor = end
                norm_cursor = j + len(span_norm)
                method, verbatim = "whitespace", False
            else:
                start = end = None
                method, verbatim = "not_found", False
        out.append({
            "index": i,
            "function": fn.function,
            "symbol": SYMBOL_BY_NAME.get(fn.function, ""),
            "move": fn.move,
            "span": span,
            "start": start,
            "end": end,
            "method": method,
            "verbatim": verbatim,
        })
    return out


def consensus(text, annotations, min_support=None):
    """Majority-vote annotation across replicate runs of the same text.

    A single run of this task is not reproducible: measured on eight tales
    at temperature 0, deepseek-v4-flash returned an identical function
    sequence in 5 of 24 replicate pairs and self-agreed at 0.82 by LCS;
    sonnet-5 managed 0 of 24. So a single pass carries real measurement
    error, and any per-tale claim drawn from one is drawn from one draw of
    a distribution.

    Two annotations are treated as reporting the SAME function when they
    agree on the label and their spans overlap in the text. Overlap rather
    than exact match, because replicates routinely pick different sentence
    boundaries for the same event — requiring identical spans would count
    agreement as disagreement.

    Returns items ordered by position, each with the number of runs
    supporting it, so a caller can report coverage honestly rather than
    silently keeping the majority. `min_support` defaults to a strict
    majority of the runs supplied.
    """
    n = len(annotations)
    if n == 0:
        return []
    if min_support is None:
        min_support = n // 2 + 1
    placed = []
    for run_i, ann in enumerate(annotations):
        for item in locate_spans(text, ann):
            if item["start"] is None:
                continue          # fabricated spans cannot vote
            placed.append((run_i, item))
    clusters = []
    for run_i, item in sorted(placed, key=lambda x: x[1]["start"]):
        for c in clusters:
            if (c["function"] == item["function"]
                    and item["start"] < c["end"] and c["start"] < item["end"]
                    and run_i not in c["runs"]):
                c["runs"].add(run_i)
                c["start"] = min(c["start"], item["start"])
                c["end"] = max(c["end"], item["end"])
                c["moves"].append(item["move"])
                break
        else:
            clusters.append({"function": item["function"],
                             "symbol": item["symbol"],
                             "start": item["start"], "end": item["end"],
                             "runs": {run_i}, "moves": [item["move"]]})
    out = []
    for c in sorted(clusters, key=lambda c: c["start"]):
        support = len(c["runs"])
        if support < min_support:
            continue
        out.append({"function": c["function"], "symbol": c["symbol"],
                    "span": text[c["start"]:c["end"]],
                    "start": c["start"], "end": c["end"],
                    "move": max(set(c["moves"]), key=c["moves"].count),
                    "support": support, "of": n})
    return out


def span_report(located):
    """Summary of a locate_spans() result, for a run-level receipt.

    `fabricated` is the number that matters: spans the model reported as
    quotations and that are not in the text.
    """
    total = len(located)
    exact = sum(1 for r in located if r["method"] == "exact")
    ws = sum(1 for r in located if r["method"] == "whitespace")
    missing = [r["index"] for r in located if r["method"] == "not_found"]
    return {
        "total": total,
        "exact": exact,
        "whitespace_only": ws,
        "fabricated": len(missing),
        "fabricated_indices": missing,
        "verbatim_rate": (exact / total) if total else 0.0,
    }
