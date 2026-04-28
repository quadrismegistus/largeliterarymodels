"""Passage setting classification: where does this passage take place?

Per-passage task: classifies the physical setting(s), setting specificity,
and time specificity of a ~500-word fiction passage. Designed to join with
passage abstractness scores for Ch5's concretization thesis.

~70 place tags organized by functional category, plus ordinal specificity
scales for both place and time.

Usage:
    from largeliterarymodels.tasks import PassageSettingTask
    from largeliterarymodels.tasks.classify_passage import format_passage

    task = PassageSettingTask(model="claude-sonnet-4-6")
    prompt = format_passage(text, title=title, author=author, year=year)
    result = task.run(prompt)
"""

from typing import Literal
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


SETTING_TAGS = [
    # Domestic interior
    'drawing_room', 'parlor', 'bedchamber', 'kitchen', 'library_study',
    'dining_room', 'nursery', 'servants_quarters', 'garret_attic', 'cellar',
    'hallway_staircase', 'sickroom', 'smoking_room', 'morning_room',
    'apartment', 'bathroom',

    # Grand / institutional
    'palace', 'castle', 'great_hall', 'ballroom', 'country_house',
    'abbey_monastery', 'church_cathedral', 'chapel', 'temple', 'courtroom',
    'parliament', 'university', 'school', 'hospital', 'almshouse_workhouse',
    'guildhall_town_hall', 'museum_gallery', 'office',

    # Urban
    'city_street', 'square_plaza', 'alley', 'marketplace', 'shop',
    'tavern_inn', 'coffeehouse', 'cafe', 'theater', 'brothel', 'gambling_den',
    'factory_mill', 'counting_house_bank', 'gentlemens_club', 'salon',
    'slum', 'sewer_underground', 'train_station', 'docks_wharf',
    'hotel', 'town',

    # Rural
    'village', 'cottage', 'farmhouse', 'field_meadow', 'forest',
    'garden_park', 'heath_moor', 'mountain', 'hill', 'river_lake',
    'seaside_beach', 'orchard_vineyard', 'barn_stable', 'parsonage_vicarage',
    'churchyard', 'marsh_fen', 'cliff', 'hermitage',

    # Outdoor enclosed
    'courtyard', 'tent',

    # Confined
    'prison', 'dungeon', 'asylum', 'convent', 'locked_room',
    'workhouse', 'tower', 'cave', 'mine',

    # Maritime
    'ship_deck', 'ship_cabin', 'port_harbor', 'open_sea', 'island',
    'lighthouse', 'shipwreck', 'riverboat',

    # Transit / interstitial
    'road_highway', 'coach_carriage', 'train_compartment', 'inn_lodging',
    'bridge', 'gateway_threshold', 'border_frontier', 'camp',
    'waiting_room', 'corridor_passage', 'porch_veranda',

    # Exotic / other
    'battlefield', 'desert', 'jungle', 'ruin', 'graveyard',
    'fairground_circus', 'plantation', 'frontier_outpost',

    # Liminal / otherworldly
    'dream_vision', 'enchanted_otherworld', 'underworld',
    'wilderness_biblical', 'paradise_garden',
]


class PassageSettingAnnotation(BaseModel):
    settings_other: list[str] = Field(
        default_factory=list,
        description=(
            "Settings NOT covered by the main list. Free-text, use lowercase_with_underscores. "
            "Only use this if none of the 104 standard tags fit. Empty list if standard tags suffice."
        ),
    )
    settings: list[Literal[
        'drawing_room', 'parlor', 'bedchamber', 'kitchen', 'library_study',
        'dining_room', 'nursery', 'servants_quarters', 'garret_attic', 'cellar',
        'hallway_staircase', 'sickroom', 'smoking_room', 'morning_room',
        'apartment', 'bathroom',
        'palace', 'castle', 'great_hall', 'ballroom', 'country_house',
        'abbey_monastery', 'church_cathedral', 'chapel', 'temple', 'courtroom',
        'parliament', 'university', 'school', 'hospital', 'almshouse_workhouse',
        'guildhall_town_hall', 'museum_gallery', 'office',
        'city_street', 'square_plaza', 'alley', 'marketplace', 'shop',
        'tavern_inn', 'coffeehouse', 'cafe', 'theater', 'brothel', 'gambling_den',
        'factory_mill', 'counting_house_bank', 'gentlemens_club', 'salon',
        'slum', 'sewer_underground', 'train_station', 'docks_wharf',
        'hotel', 'town',
        'village', 'cottage', 'farmhouse', 'field_meadow', 'forest',
        'garden_park', 'heath_moor', 'mountain', 'hill', 'river_lake',
        'seaside_beach', 'orchard_vineyard', 'barn_stable', 'parsonage_vicarage',
        'churchyard', 'marsh_fen', 'cliff', 'hermitage',
        'courtyard', 'tent',
        'prison', 'dungeon', 'asylum', 'convent', 'locked_room',
        'workhouse', 'tower', 'cave', 'mine',
        'ship_deck', 'ship_cabin', 'port_harbor', 'open_sea', 'island',
        'lighthouse', 'shipwreck', 'riverboat',
        'road_highway', 'coach_carriage', 'train_compartment', 'inn_lodging',
        'bridge', 'gateway_threshold', 'border_frontier', 'camp',
        'waiting_room', 'corridor_passage', 'porch_veranda',
        'battlefield', 'desert', 'jungle', 'ruin', 'graveyard',
        'fairground_circus', 'plantation', 'frontier_outpost',
        'dream_vision', 'enchanted_otherworld', 'underworld',
        'wilderness_biblical', 'paradise_garden',
    ]] = Field(
        default_factory=list,
        description="All physical settings present in the passage. Typically 1-3.",
    )
    setting_specificity: Literal[
        'generic', 'typed', 'named', 'fully_described'
    ] = Field(
        description=(
            "How specifically is the setting rendered? "
            "generic: 'a room', 'a street', 'the country' — no distinguishing detail. "
            "typed: 'a drawing-room', 'a London street', 'the English countryside' — "
            "category specified but not individuated. "
            "named: 'the drawing-room at Pemberley', 'Bond Street', 'Yorkshire' — "
            "a particular place is identified by name. "
            "fully_described: named AND given sensory/spatial detail beyond the name — "
            "furniture, light, dimensions, sounds, smells."
        )
    )
    time_specificity: Literal[
        'generic', 'typed', 'named', 'fully_described'
    ] = Field(
        description=(
            "How specifically is time marked in the passage? "
            "generic: 'one day', 'in the evening', 'later' — no particular time. "
            "typed: 'a Sunday morning', 'winter', 'after dinner' — period specified. "
            "named: 'March 1798', 'Tuesday', 'the day after Michaelmas' — "
            "a specific date or occasion. "
            "fully_described: 'Tuesday the 12th of March, at three in the afternoon' — "
            "precise temporal coordinates."
        )
    )
    narrative_frequency: Literal[
        'singulative', 'iterative', 'mixed'
    ] = Field(
        description=(
            "Genette's narrative frequency — how events relate to time. "
            "singulative: the passage narrates events that happen ONCE — "
            "'he walked to the door', 'she said'. Specific scenes, specific actions. "
            "iterative: the passage narrates HABITUAL or recurring events — "
            "'he would often walk', 'every Sunday they dined', 'it was her custom to'. "
            "The imperfect tense, 'would' + verb, 'used to', and 'every/always/never' "
            "are strong signals. "
            "mixed: the passage shifts between singulative and iterative."
        )
    )
    space_traversed: Literal[
        'none', 'room', 'building', 'grounds', 'neighborhood',
        'city', 'region', 'country', 'international',
    ] = Field(
        description=(
            "How much PHYSICAL SPACE does the action of this passage traverse? "
            "This is NOT the setting — it is the RANGE OF MOVEMENT within the passage. "
            "A drawing-room conversation where nobody moves = 'room'. "
            "A drawing-room scene where someone arrives from across town = 'city'. "
            "none: no movement — a gesture, a glance, interior thought, static scene. "
            "room: movement across a single room — standing, crossing to the window, etc. "
            "building: moving between rooms or floors. "
            "grounds: house and its immediate surroundings (garden, yard, drive). "
            "neighborhood: a few streets, a village, a parish. "
            "city: across a city or large town. "
            "region: between towns, across a county or district. "
            "country: cross-country journey. "
            "international: across borders, overseas voyage."
        )
    )
    time_elapsed: Literal[
        'moment', 'minutes', 'hours', 'day',
        'days', 'weeks', 'months', 'years', 'lifetime',
    ] = Field(
        description=(
            "How much STORY TIME passes during this passage? "
            "moment: seconds — a gesture, a single line of dialogue, a glance. "
            "minutes: a short conversation, a brief encounter. "
            "hours: an evening, a dinner party, a visit, a morning's work. "
            "day: sunrise to sunset, or a single full day. "
            "days: a few days compressed into the passage. "
            "weeks: weeks summarized. "
            "months: months compressed. "
            "years: years pass in a paragraph or page. "
            "lifetime: birth-to-death or a large portion of a life."
        )
    )


SYSTEM_PROMPT = """\
You are classifying the SETTING of a passage from an English prose fiction text.

You will receive a ~500-1500 word passage of prose fiction. No title, author, or \
date is provided — classify based solely on what is in the text.

Determine:

1. SETTINGS — Where does this passage take place? Select all physical locations \
present. A passage may move between settings (1-3 typical). Pick the most specific \
tag that fits — use 'drawing_room' not just 'country_house' if the scene is \
clearly in a drawing room.

If NO PLACE is inferrable from the passage (pure dialogue, interior monologue, \
abstract reflection with no spatial markers), return an EMPTY settings list.

If a setting is present but not covered by any of the standard tags, put it in \
settings_other as a free-text label (lowercase_with_underscores).

DOMESTIC INTERIOR: drawing_room (formal reception), parlor (less formal sitting), \
bedchamber, kitchen, library_study, dining_room, nursery, servants_quarters, \
garret_attic, cellar, hallway_staircase, sickroom, smoking_room, morning_room.

GRAND / INSTITUTIONAL: palace, castle, great_hall, ballroom, country_house \
(the house as a whole, not a specific room), abbey_monastery, church_cathedral, \
chapel, courtroom, parliament, university, hospital, almshouse_workhouse, \
guildhall_town_hall, museum_gallery.

URBAN: city_street, square_plaza, alley, marketplace, shop, tavern_inn, \
coffeehouse, theater, brothel, gambling_den, factory_mill, counting_house_bank, \
gentlemens_club, salon, slum, sewer_underground, train_station, docks_wharf.

RURAL: village, cottage, farmhouse, field_meadow, forest, garden_park, \
heath_moor, mountain, river_lake, seaside_beach, orchard_vineyard, barn_stable, \
parsonage_vicarage, churchyard, marsh_fen, cliff.

CONFINED: prison, dungeon, asylum, convent, locked_room, workhouse, tower, \
cave, mine.

MARITIME: ship_deck, ship_cabin, port_harbor, open_sea, island, lighthouse, \
shipwreck, riverboat.

TRANSIT / INTERSTITIAL: road_highway, coach_carriage, train_compartment, \
inn_lodging, bridge, gateway_threshold, border_frontier, camp, waiting_room, \
corridor_passage, porch_veranda.

EXOTIC / OTHER: battlefield, desert, jungle, ruin, graveyard, \
fairground_circus, plantation, frontier_outpost.

LIMINAL / OTHERWORLDLY: dream_vision, enchanted_otherworld, underworld, \
wilderness_biblical, paradise_garden.

2. SETTING SPECIFICITY — How concretely is the place rendered?
   - generic: "a room", "a street" — bare category, no detail
   - typed: "a drawing-room", "a London street" — type specified but interchangeable
   - named: "the drawing-room at Pemberley", "Bond Street" — a particular place
   - fully_described: named + sensory/spatial detail (furniture, light, sounds)

3. TIME SPECIFICITY — How precisely is time marked?
   - generic: "one day", "later", "in the evening"
   - typed: "a Sunday morning", "winter", "after dinner"
   - named: "March 1798", "the day after Michaelmas"
   - fully_described: "Tuesday the 12th of March, at three in the afternoon"

4. NARRATIVE FREQUENCY — How does the passage relate events to time?
   - singulative: events happen ONCE — "he walked to the door", "she said"
   - iterative: HABITUAL/recurring — "he would often", "every Sunday they", \
"it was her custom to", "they always"
   - mixed: the passage shifts between singulative and iterative

5. SPACE TRAVERSED — How much physical space does the ACTION traverse? \
This is NOT the setting — it is the range of movement within the passage. \
A drawing-room conversation where nobody moves = "room". \
A drawing-room scene where someone arrives from across town = "city".
   none / room / building / grounds / neighborhood / city / region / country / international

6. TIME ELAPSED — How much story time passes during this passage?
   moment / minutes / hours / day / days / weeks / months / years / lifetime

Base classification on WHAT IS IN THE PASSAGE, not what you know about the novel."""


EXAMPLES = [
    (
        "I took the liberty to go up to my late lady's dressing-room, "
        "and there sat me down and wept. Mrs. Jervis came up to me and "
        "said, 'Why weeping so, Pamela?' I said, 'Oh dear Mrs. Jervis, "
        "how can I help it? My dear lady is dead, and I have lost my "
        "best friend.' She said, 'You must not give way so.'",
        PassageSettingAnnotation(
            settings=['bedchamber'],
            setting_specificity='typed',
            time_specificity='generic',
            narrative_frequency='singulative',
            space_traversed='room',
            time_elapsed='minutes',
        ),
    ),
    (
        "The drawing-room at the Grange was the scene of a Christmas "
        "party. Old Mr. Brooke stood near the fire-place, his thin grey "
        "hair combed carefully across his forehead, one hand in the pocket "
        "of his drab waistcoat. The candelabra threw a warm light across "
        "the Turkey carpet and the heavy curtains drawn against the December "
        "dark. Mrs. Cadwallader occupied the sofa nearest the pianoforte.",
        PassageSettingAnnotation(
            settings=['drawing_room', 'country_house'],
            setting_specificity='fully_described',
            time_specificity='named',
            narrative_frequency='singulative',
            space_traversed='room',
            time_elapsed='hours',
        ),
    ),
]


DEFAULT_MODEL = 'lmstudio/qwen/qwen3.6-35b-a3b'


class PassageSettingTask(Task):
    name = "classify_passage_setting"
    model = DEFAULT_MODEL
    schema = PassageSettingAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.2

    @staticmethod
    def format_input(passage_text: str) -> str:
        """Return passage text only — no title/author/year metadata."""
        return passage_text.strip()
