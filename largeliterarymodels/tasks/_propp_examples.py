"""Few-shot examples for ProppTask, built from Propp's OWN analyses.

Three annotated tales plus one negative case. The function SEQUENCES are
Propp's, transcribed from his printed schemes — not the annotator's
judgement. Only the spans were inferred, and every one is verified to occur
verbatim in the text it is attached to; there is a test asserting that,
because an example whose quote does not appear in its own text teaches the
model to fabricate.

Chosen against a measured defect. Zero-shot the task scored recall 0.80
against Propp but precision 0.47 — it found most of what he found and
roughly twice as much besides, annotating 17-26 functions where he records
8-14. So every example here is SPARSE, and that is the lesson: Propp
records a skeleton, not an inventory. #148 is nine functions over 683
words; #108 is nine over 1,264 and declines to name B, C, F, K or W at all;
#95 runs two moves with neither H-I nor M-N, the direct counter to the
observed habit of reaching for a difficult task whenever a second move
appears.

Texts are public-domain 19th-century translations passed through
prepare_tale_text (hard wrapping joined, footnote markers and emphasis
underscores removed). Lean Lisa is Grimm via Ashliman: a fairy-tale surface
with no Proppian architecture, so the null case has to be decided on
structure rather than register.

Emitted as exact fixed-width slices rather than word-wrapped: wrapping
collapses whitespace, which silently alters the text and breaks every span
crossing a wrap boundary.
"""

EXAMPLES = (
    # Afanasyev #148 — Nikita the Tanner
    # Propp's scheme: (single). A1 B2^4 C ↑ H1 I1 K4 ↓ W_*
    (
        (
            'Afanasyev no. (1936-40/1957 numbering): 148 Russian title: Никита Ко'
            'жемяка English title as printed: Nikíta the Tanner Translator / year'
            ': Leonard A. Magnus, 1916 — Russian Folk-Tales (London: Kegan Paul, '
            '1916) — translated directly from Afanasyev Source: https://www.guten'
            'berg.org/ebooks/62509 Match confidence: CERTAIN — Nikita Kozhemyaka,'
            ' the dragon, and the furrow ploughed to the sea. ==================='
            '=====================================================\n\n NIKÍTA THE T'
            'ANNER\n\nOne day, somewhere near Kíev, a dragon appeared, who demanded'
            ' heavy tribute from the people. He demanded every time to eat a fair'
            ' maiden: and at last the turn came to the Tsarévna, the princess. Bu'
            't the dragon would not eat her, she was too beautiful. He dragged he'
            'r into his den and made her his wife. When he flew out on business, '
            'he used to pile logs of wood in front of the den to prevent the Tsar'
            'évna escaping. But the Tsarévna had a little dog that had followed h'
            'er all the way from home. When she wrote a letter to her father and '
            'mother she used to tie it to the neck of her little dog, which would'
            ' run all the way home and bring an answer back. One day her parents '
            'wrote to her: “Try to discover any one who is stronger than the drag'
            'on.” The Tsarévna got every day on more intimate terms with her drag'
            'on in order to discover who was stronger. At last he owned that Nikí'
            'ta, the tanner at Kíev, was the stronger. So the Tsarévna at once wr'
            'ote to her father: “Look for Nikíta, the tanner at Kíev, and send hi'
            'm on to me to deliver me from my imprisonment.”\n\nSo the Tsar looked '
            'for Nikíta, and went to him himself to beg him to release the land f'
            'rom the cruelty of the dragon and redeem the princess.\n\nJust then Ni'
            'kíta was tanning skins. He was just enfolding twelve hides in his ha'
            'nds. But when he saw the Tsar come to see him, his hands so trembled'
            ' for fear that he rent the twelve hides. But, however much the Tsar '
            'and the Tsarítsa asked him, he would not set out against the dragon.'
            ' Then the Tsar assembled five thousand children, who were to mollify'
            ' the tanner with their bitter tears. The little ones came to Nikíta '
            'and begged him to go and fight the dragon. And when he saw them weep'
            ', Nikíta the tanner himself almost felt the tears flowing. He took t'
            'hirty puds of hemp, tarred it, and swathed himself in it in order th'
            'at the dragon might find him a hard morsel, and then set out. But th'
            'e dragon locked himself up in his den and would not come to view.\n\n“'
            'Come with me into the open field, otherwise I will shatter your den '
            'to pieces!” said the tanner, and began clattering at the doors.\n\nThe'
            'n the dragon, seeing his doom approach, came out into the open. Nikí'
            'ta the tanner fought the grisly worm some time, maybe long, maybe sh'
            'ort, and at last got him under.\n\nThen the dragon besought Nikíta the'
            ' tanner: “Do not beat me to death. Stronger than us two there is not'
            'hing in the white world. Let us divide the earth. You may live on th'
            'e one half and I on the other.”\n\n“Very well!” said Nikíta, “only we '
            'must delimit frontiers.”\n\nSo the tanner took the plough, which weigh'
            'ed three hundred puds, and harnessed to it the dragon, and drew the '
            'harrow all the way from Kíev to the Caspian Sea.\n\n“Now we have divid'
            'ed the entire earth,” said the dragon.\n\n“Yes, we have divided the ea'
            'rth, but not the sea; we must also divide the sea, otherwise you wou'
            'ld say I was taking your share of the water.” So they then set out i'
            'nto the middle of the sea, and there Nikíta slew the dragon and drow'
            'ned him.\n\nThe trench may still be seen: it is two fathoms deep. They'
            ' plough all round it; but never touch the bottom: those who do not k'
            'now whence came this trench call it a battlement.\n\nWhen Nikíta had d'
            'one this feat, he demanded no reward for it, but went home and went '
            'on tanning.'
        ),
        {
            'is_tale_structured': True,
            'n_moves': 1,
            'functions': [
                {'function': 'villainy', 'span': 'a dragon appeared, who demanded heavy tribute from the people. He demanded every time to eat a fair maiden: and at last the turn came to the Tsarévna, the princess. But the dragon would not eat her, she was too beautiful. He dragged her into his den and made her his wife.',
                 'move': 1, 'event_order': 1, 'note': 'The dragon abducts the princess, causing harm to the royal family.'},
                {'function': 'mediation', 'span': 'So the Tsar looked for Nikíta, and went to him himself to beg him to release the land from the cruelty of the dragon and redeem the princess.',
                 'move': 1, 'event_order': 2, 'note': "The Tsar makes the misfortune known and requests Nikita's help."},
                {'function': 'beginning_counteraction', 'span': 'And when he saw them weep, Nikíta the tanner himself almost felt the tears flowing. He took thirty puds of hemp, tarred it, and swathed himself in it in order that the dragon might find him a hard morsel, and then set out.',
                 'move': 1, 'event_order': 3, 'note': 'Nikita decides to act and prepares for the counteraction.'},
                {'function': 'departure', 'span': 'and then set out.',
                 'move': 1, 'event_order': 4, 'note': 'Nikita leaves home to confront the dragon.'},
                {'function': 'struggle', 'span': 'Nikíta the tanner fought the grisly worm some time, maybe long, maybe short, and at last got him under.',
                 'move': 1, 'event_order': 5, 'note': 'Direct combat between Nikita and the dragon.'},
                {'function': 'victory', 'span': 'and at last got him under.',
                 'move': 1, 'event_order': 6, 'note': 'Nikita defeats the dragon in the struggle.'},
                {'function': 'liquidation_of_lack', 'span': 'there Nikíta slew the dragon and drowned him.',
                 'move': 1, 'event_order': 7, 'note': 'The dragon is killed, liquidating the initial misfortune.'},
                {'function': 'return', 'span': 'he demanded no reward for it, but went home',
                 'move': 1, 'event_order': 8, 'note': 'Nikita returns home after his feat.'},
                {'function': 'wedding', 'span': 'went on tanning.',
                 'move': 1, 'event_order': 9, 'note': 'No wedding occurs; Nikita resumes his former life, so this function is absent or negated.'},
            ],
            'overall_note': "Propp's scheme for this tale is (single). A1 B2^4 C ↑ H1 I1 K4 ↓ W_* — 9 functions across 1 move(s).",
        },
    ),
    # Afanasyev #108 — Ivashko and the Witch
    # Propp's scheme: (single). A1 ↑ D8 E8 ↓ { Pr7 Rs10 / Pr7 Rs1 }
    (
        (
            'Afanasyev no. (1936-40/1957 numbering): 108 Russian title: Ивашко и '
            'ведьма English title as printed: The Witch Translator / year: W. R. '
            'S. Ralston, 1873 — Russian Folk-Tales (London, 1873; repr. as Russia'
            'n Fairy Tales) — translated directly from Afanasyev Source: https://'
            'www.gutenberg.org/ebooks/22373 Match confidence: CERTAIN — Ivashko, '
            "the fishing boat, the witch's counterfeit voice, the escape by geese"
            ". Afanasyev prints variants 108-111; this is the group's first form."
            " Also in Magnus as 'Iváshko and the Wise Woman'. ==================="
            '=====================================================\n\nTHE WITCH.\n\nT'
            'here once lived an old couple who had one son called Ivashko; no one'
            ' can tell how fond they were of him!\n\nWell, one day, Ivashko said to'
            ' his father and mother:\n\n"I\'ll go out fishing if you\'ll let me."\n\n"W'
            "hat are you thinking about! you're still very small; suppose you get"
            ' drowned, what good will there be in that?"\n\n"No, no, I shan\'t get d'
            'rowned. I\'ll catch you some fish; do let me go!"\n\nSo his mother put '
            'a white shirt on him, tied a red girdle round him, and let him go. O'
            'ut in a boat he sat and said:\n\nThen the canoe floated on farther and'
            ' farther, and Ivashko began to fish. When some little time had passe'
            'd by, the old woman hobbled down to the river side and called to her'
            ' son:\n\nAnd Ivashko said:\n\nThe boat floated to the shore: the woman t'
            'ook the fish, gave her boy food and drink, changed his shirt for him'
            ' and his girdle, and sent him back to his fishing. Again he sat in h'
            'is boat and said:\n\nThen the canoe floated on farther and farther, an'
            'd Ivashko began to fish. After a little time had passed by, the old '
            'man also hobbled down to the bank and called to his son:\n\nAnd Ivashk'
            'o replied:\n\nThe canoe floated to the shore. The old man took the fis'
            'h, gave his boy food and drink, changed his shirt for him and his gi'
            'rdle, and sent him back to his fishing.\n\nNow a certain witch had hea'
            "rd what Ivashko's parents had cried aloud to him, and she longed to "
            'get hold of the boy. So she went down to the bank and cried with a h'
            "oarse voice:\n\nIvashko perceived that the voice was not his mother's,"
            ' but was that of a witch, and he sang:\n\nThe witch saw that she must '
            'call Ivashko with just such a voice as his mother had.\n\nSo she haste'
            'ned to a smith and said to him:\n\n"Smith, smith! make me just such a '
            "thin little voice as Ivashko's mother has: if you don't, I'll eat yo"
            'u." So the smith forged her a little voice just like Ivashko\'s mothe'
            "r's. Then the witch went down by night to the shore and sang:\n\nIvash"
            'ko came, and she took the fish, and seized the boy and carried him h'
            'ome with her. When she arrived she said to her daughter Alenka, "Hea'
            't the stove as hot as you can, and bake Ivashko well, while I go and'
            ' collect my friends for the feast." So Alenka heated the stove hot, '
            'ever so hot, and said to Ivashko,\n\n"Come here and sit on this shovel'
            '!"\n\n"I\'m still very young and foolish," answered Ivashko: "I haven\'t'
            ' yet quite got my wits about me. Please teach me how one ought to si'
            't on a shovel."\n\n"Very good," said Alenka; "it won\'t take long to te'
            'ach you."\n\nBut the moment she sat down on the shovel, Ivashko instan'
            'tly pitched her into the oven, slammed to the iron plate in front of'
            ' it, ran out of the hut, shut the door, and hurriedly climbed up eve'
            'r so high an oak-tree [which stood close by].\n\nPresently the witch a'
            'rrived with her guests and knocked at the door of the hut. But nobod'
            'y opened it for her.\n\n"Ah! that cursed Alenka!" she cried. "No doubt'
            ' she\'s gone off somewhere to amuse herself." Then she slipped in thr'
            'ough the window, opened the door, and let in her guests. They all sa'
            "t down to table, and the witch opened the oven, took out Alenka's ba"
            'ked body, and served it up. They all ate their fill and drank their '
            'fill, and then they went out into the courtyard and began rolling ab'
            'out on the grass.\n\n"I turn about, I roll about, having fed on Ivashk'
            'o\'s flesh," cried the witch. "I turn about, I roll about, having fed'
            ' on Ivashko\'s flesh."\n\nBut Ivashko called out to her from the top of'
            ' the oak:\n\n"Turn about, roll about, having fed on Alenka\'s flesh!"\n\n'
            '"Did I hear something?" said the witch. "No it was only the noise of'
            ' the leaves." Again the witch began:\n\n"I turn about, I roll about, h'
            'aving fed on Ivashko\'s flesh!"\n\nAnd Ivashko repeated:\n\n"Turn about, '
            'roll about, having fed on Alenka\'s flesh!"\n\nThen the witch looked up'
            ' and saw Ivashko, and immediately rushed at the oak on which Ivashko'
            ' was seated, and began to gnaw away at it. And she gnawed, and gnawe'
            'd, and gnawed, until at last she smashed two front teeth. Then she r'
            'an to a forge, and when she reached it she cried, "Smith, smith! mak'
            'e me some iron teeth; if you don\'t I\'ll eat you!"\n\nSo the smith forg'
            'ed her two iron teeth.\n\nThe witch returned and began gnawing the oak'
            ' again.\n\nShe gnawed, and gnawed, and was just on the point of gnawin'
            'g it through, when Ivashko jumped out of it into another tree which '
            'stood beside it. The oak that the witch had gnawed through fell down'
            ' to the ground; but then she saw that Ivashko was sitting up in anot'
            'her tree, so she gnashed her teeth with spite and set to work afresh'
            ', to gnaw that tree also. She gnawed, and gnawed, and gnawed--broke '
            'two lower teeth, and ran off to the forge.\n\n"Smith, smith!" she crie'
            'd when she got there, "make me some iron teeth; if you don\'t I\'ll ea'
            't you!"\n\nThe smith forged two more iron teeth for her. She went back'
            " again, and once more began to gnaw the oak.\n\nIvashko didn't know wh"
            'at he was to do now. He looked out, and saw that swans and geese wer'
            'e flying by, so he called to them imploringly:\n\n"Let those in the ce'
            'ntre carry you," said the birds.\n\nIvashko waited; a second flock fle'
            'w past, and he again cried imploringly:\n\n"Let those in the rear carr'
            'y you!" said the birds.\n\nAgain Ivashko waited. A third flock came fl'
            'ying up, and he cried:\n\nAnd those swans and geese took hold of him a'
            'nd carried him back, flew up to the cottage, and dropped him in the '
            'upper room.\n\nEarly the next morning his mother set to work to bake p'
            'ancakes, baked them, and all of a sudden fell to thinking about her '
            'boy. "Where is my Ivashko?" she cried; "would that I could see him, '
            'were it only in a dream!"\n\nThen his father said, "I dreamed that swa'
            'ns and geese had brought our Ivashko home on their wings."\n\nAnd when'
            ' she had finished baking the pancakes, she said, "Now, then, old man'
            ", let's divide the cakes: there's for you, father! there's for me! T"
            'here\'s for you, father! there\'s for me."\n\n"And none for me?" called '
            'out Ivashko.\n\n"There\'s for you, father!" went on the old woman, "the'
            're\'s for me."\n\n"And none for me!" [repeated the boy.]\n\n"Why, old man'
            '," said the wife, "go and see whatever that is up there."\n\nThe fathe'
            'r climbed into the upper room and there he found Ivashko. The old pe'
            'ople were delighted, and asked their boy about everything that had h'
            'appened. And after that he and they lived on happily together.'
        ),
        {
            'is_tale_structured': True,
            'n_moves': 1,
            'functions': [
                {'function': 'villainy', 'span': 'Then the witch went down by night to the shore and sang:\n\nIvashko came, and she took the fish, and seized the boy and carried him home with her.',
                 'move': 1, 'event_order': 1, 'note': 'The witch seizes and carries off Ivashko.'},
                {'function': 'departure', 'span': 'ran out of the hut, shut the door, and hurriedly climbed up ever so high an oak-tree',
                 'move': 1, 'event_order': 2, 'note': "Ivashko leaves the witch's hut and climbs the oak."},
                {'function': 'donor_first_function', 'span': 'Then the witch looked up and saw Ivashko, and immediately rushed at the oak on which Ivashko was seated, and began to gnaw away at it.',
                 'move': 1, 'event_order': 3, 'note': 'The witch attacks the tree, testing Ivashko.'},
                {'function': 'hero_reaction', 'span': 'He looked out, and saw that swans and geese were flying by, so he called to them imploringly',
                 'move': 1, 'event_order': 4, 'note': 'Ivashko reacts by calling for help.'},
                {'function': 'return', 'span': 'And those swans and geese took hold of him and carried him back, flew up to the cottage, and dropped him in the upper room.',
                 'move': 1, 'event_order': 5, 'note': 'The geese carry Ivashko home.'},
                {'function': 'pursuit', 'span': 'She went back again, and once more began to gnaw the oak.',
                 'move': 1, 'event_order': 6, 'note': 'The witch resumes her pursuit by gnawing the tree.'},
                {'function': 'rescue', 'span': 'And those swans and geese took hold of him and carried him back, flew up to the cottage, and dropped him in the upper room.',
                 'move': 1, 'event_order': 7, 'note': 'The geese rescue Ivashko from the witch.'},
                {'function': 'pursuit', 'span': 'She went back again, and once more began to gnaw the oak.',
                 'move': 1, 'event_order': 8, 'note': "The witch's renewed gnawing is the second pursuit."},
                {'function': 'rescue', 'span': 'And those swans and geese took hold of him and carried him back, flew up to the cottage, and dropped him in the upper room.',
                 'move': 1, 'event_order': 9, 'note': 'The geese again rescue Ivashko from the witch.'},
            ],
            'overall_note': "Propp's scheme for this tale is (single). A1 ↑ D8 E8 ↓ { Pr7 Rs10 / Pr7 Rs1 } — 9 functions across 1 move(s).",
        },
    ),
    # Afanasyev #95 — Frost (Morozko)
    # Propp's scheme: I. A9 B5 D1 E1 f1 ↓ || II. [a6] B2^5 C ↑ D1 E_1 F_ ↓
    (
        (
            'Afanasyev no. (1936-40/1957 numbering): 95 Russian title: Морозко En'
            'glish title as printed: Frost Translator / year: W. R. S. Ralston, 1'
            '873 — Russian Folk-Tales (London, 1873; repr. as Russian Fairy Tales'
            ') — translated directly from Afanasyev Source: https://www.gutenberg'
            '.org/ebooks/22373 Match confidence: PROBABLE — Morozko. Afanasyev pr'
            "ints two variants (95-96); Ralston's stepdaughter is named Marfa. Va"
            'riant identity within the 95/96 pair not established. =============='
            '==========================================================\n\nFROST.\n\n'
            'There was once an old man who had a wife and three daughters. The wi'
            'fe had no love for the eldest of the three, who was her stepdaughter'
            ', but was always scolding her. Moreover, she used to make her get up'
            ' ever so early in the morning, and gave her all the work of the hous'
            'e to do. Before daybreak the girl would feed the cattle and give the'
            'm to drink, fetch wood and water indoors, light the fire in the stov'
            'e, give the room a wash, mend the dresses, and set everything in ord'
            'er. Even then her stepmother was never satisfied, but would grumble '
            'away at Marfa, exclaiming:--\n\n"What a lazybones! what a slut! Why he'
            "re's a brush not in its place, and there's something put wrong, and "
            'she\'s left the muck inside the house!"\n\nThe girl held her peace, and'
            ' wept; she tried in every way to accommodate herself to her stepmoth'
            'er, and to be of service to her stepsisters. But they, taking patter'
            'n by their mother, were always insulting Marfa, quarrelling with her'
            ', and making her cry: that was even a pleasure to them! As for them,'
            ' they lay in bed late, washed themselves in water got ready for them'
            ", dried themselves with a clean towel, and didn't sit down to work t"
            'ill after dinner.\n\nWell, our girls grew and grew, until they grew up'
            ' and were old enough to be married. The old man felt sorry for his e'
            'ldest daughter, whom he loved because she was industrious and obedie'
            'nt, never was obstinate, always did as she was bid, and never uttere'
            "d a word of contradiction. But he didn't know how he was to help her"
            ' in her trouble. He was feeble, his wife was a scold, and her daught'
            'ers were as obstinate as they were indolent.\n\nWell, the old folks se'
            't to work to consider--the husband how he could get his daughters se'
            'ttled, the wife how she could get rid of the eldest one. One day she'
            ' says to him:--\n\n"I say, old man! let\'s get Marfa married."\n\n"Gladly'
            '," says he, slinking off (to the sleeping-place) above the stove. Bu'
            't his wife called after him:--\n\n"Get up early to-morrow, old man, ha'
            'rness the mare to the sledge, and drive away with Marfa. And, Marfa,'
            ' get your things together in a basket, and put on a clean shift; you'
            '\'re going away to-morrow on a visit."\n\nPoor Marfa was delighted to h'
            'ear of such a piece of good luck as being invited on a visit, and sh'
            'e slept comfortably all night. Early next morning she got up, washed'
            ' herself, prayed to God, got all her things together, packed them aw'
            'ay in proper order, dressed herself (in her best things), and looked'
            ' something like a lass!--a bride fit for any place whatsoever!\n\nNow '
            'it was winter time, and out of doors was a rattling frost. Early in '
            'the morning, between daybreak and sunrise, the old man harnessed the'
            ' mare to the sledge, and led it up to the steps. Then he went indoor'
            's, sat down on the window-sill, and said:--\n\n"Now then! I\'ve got eve'
            'rything ready."\n\n"Sit down to table and swallow your victuals!" repl'
            'ied the old woman.\n\nThe old man sat down to table, and made his daug'
            'hter sit by his side. On the table stood a pannier; he took out a lo'
            'af, and cut bread for himself and his daughter. Meantime his wife se'
            'rved up a dish of old cabbage soup, and said:--\n\n"There, my pigeon, '
            "eat and be off; I've looked at you quite enough! Drive Marfa to her "
            'bridegroom, old man. And look here, old greybeard! drive straight al'
            'ong the road at first, and then turn off from the road to the right,'
            ' you know, into the forest--right up to the big pine that stands on '
            'the hill, and there hand Marfa over to Morozko (Frost)."\n\nThe old ma'
            'n opened his eyes wide, also his mouth, and stopped eating, and the '
            'girl began lamenting.\n\n"Now then, what are you hanging your chaps an'
            'd squealing about?" said her stepmother. "Surely your bridegroom is '
            "a beauty, and he's that rich! Why, just see what a lot of things bel"
            'ong to him, the firs, the pine-tops, and the birches, all in their r'
            'obes of down--ways and means that any one might envy; and he himself'
            ' a bogatir!"\n\nThe old man silently placed the things on the sledge, '
            'made his daughter put on a warm pelisse, and set off on the journey.'
            ' After a time, he reached the forest, turned off from the road; and '
            'drove across the frozen snow. When he got into the depths of the for'
            'est, he stopped, made his daughter get out, laid her basket under th'
            'e tall pine, and said:--\n\n"Sit here, and await the bridegroom. And m'
            'ind you receive him as pleasantly as you can."\n\nThen he turned his h'
            'orse round and drove off homewards.\n\nThe girl sat and shivered. The '
            'cold had pierced her through. She would fain have cried aloud, but s'
            'he had not strength enough; only her teeth chattered. Suddenly she h'
            'eard a sound. Not far off, Frost was cracking away on a fir. From fi'
            'r to fir was he leaping, and snapping his fingers. Presently he appe'
            'ared on that very pine under which the maiden was sitting and from a'
            'bove her head he cried:--\n\n"Art thou warm, maiden?"\n\n"Warm, warm am '
            'I, dear Father Frost," she replied.\n\nFrost began to descend lower, a'
            'll the more cracking and snapping his fingers. To the maiden said Fr'
            'ost:--\n\n"Art thou warm, maiden? Art thou warm, fair one?"\n\nThe girl '
            'could scarcely draw her breath, but still she replied:\n\n"Warm am I, '
            'Frost dear: warm am I, father dear!"\n\nFrost began cracking more than'
            ' ever, and more loudly did he snap his fingers, and to the maiden he'
            ' said:--\n\n"Art thou warm, maiden? Art thou warm, pretty one? Art tho'
            'u warm, my darling?"\n\nThe girl was by this time numb with cold, and '
            'she could scarcely make herself heard as she replied:--\n\n"Oh! quite '
            'warm, Frost dearest!"\n\nThen Frost took pity on the girl, wrapped her'
            ' up in furs, and warmed her with blankets.\n\nNext morning the old wom'
            'an said to her husband:--\n\n"Drive out, old greybeard, and wake the y'
            'oung couple!"\n\nThe old man harnessed his horse and drove off. When h'
            'e came to where his daughter was, he found she was alive and had got'
            ' a good pelisse, a costly bridal veil, and a pannier with rich gifts'
            '. He stowed everything away on the sledge without saying a word, too'
            'k his seat on it with his daughter, and drove back. They reached hom'
            "e, and the daughter fell at her stepmother's feet. The old woman was"
            ' thunderstruck when she saw the girl alive, and the new pelisse and '
            'the basket of linen.\n\n"Ah, you wretch!" she cries. "But you shan\'t t'
            'rick me!"\n\nWell, a little later the old woman says to her husband:--'
            '\n\n"Take my daughters, too, to their bridegroom. The presents he\'s ma'
            'de are nothing to what he\'ll give them."\n\nWell, early next morning t'
            'he old woman gave her girls their breakfast, dressed them as befitte'
            'd brides, and sent them off on their journey. In the same way as bef'
            'ore the old man left the girls under the pine.\n\nThere the girls sat,'
            ' and kept laughing and saying:\n\n"Whatever is mother thinking of! All'
            ' of a sudden to marry both of us off! As if there were no lads in ou'
            'r village, forsooth! Some rubbishy fellow may come, and goodness kno'
            'ws who he may be!"\n\nThe girls were wrapped up in pelisses, but for a'
            'll that they felt the cold.\n\n"I say, Prascovia! the frost\'s skinning'
            " me alive. Well, if our bridegroom doesn't come quick, we shall be f"
            'rozen to death here!"\n\n"Don\'t go talking nonsense, Mashka; as if sui'
            "tors generally turned up in the forenoon. Why it's hardly dinner-tim"
            'e yet!"\n\n"But I say, Prascovia! if only one comes, which of us will '
            'he take?"\n\n"Not you, you stupid goose!"\n\n"Then it will be you, I sup'
            'pose!"\n\n"Of course it will be me!"\n\n"You, indeed! there now, have do'
            'ne talking stuff and treating people like fools!"\n\nMeanwhile, Frost '
            "had numbed the girl's hands, so our damsels folded them under their "
            'dress, and then went on quarrelling as before.\n\n"What, you fright! y'
            "ou sleepy-face! you abominable shrew! why, you don't know so much as"
            " how to begin weaving: and as to going on with it, you haven't an id"
            'ea!"\n\n"Aha, boaster! and what is it you know? Why, nothing at all ex'
            "cept to go out to merry-makings and lick your lips there. We'll soon"
            ' see which he\'ll take first!"\n\nWhile the girls went on scolding like'
            ' that, they began to freeze in downright earnest. Suddenly they both'
            ' cried out at once:\n\n"Whyever is he so long coming. Do you know, you'
            '\'ve turned quite blue!"\n\nNow, a good way off, Frost had begun cracki'
            'ng, snapping his fingers, and leaping from fir to fir. To the girls '
            'it sounded as if some one was coming.\n\n"Listen, Prascovia! He\'s comi'
            'ng at last, and with bells, too!"\n\n"Get along with you! I won\'t list'
            'en; my skin is peeling with cold."\n\n"And yet you\'re still expecting '
            'to get married!"\n\nThen they began blowing on their fingers.\n\nNearer '
            'and nearer came Frost. At length he appeared on the pine, above the '
            'heads of the girls, and said to them:\n\n"Are ye warm, maidens? Are ye'
            ' warm, pretty ones? Are ye warm, my darlings?"\n\n"Oh, Frost, it\'s awf'
            "ully cold! we're utterly perished! We're expecting a bridegroom, but"
            ' the confounded fellow has disappeared."\n\nFrost slid lower down the '
            'tree, cracked away more, snapped his fingers oftener than before.\n\n"'
            'Are ye warm, maidens? Are ye warm, pretty ones?"\n\n"Get along with yo'
            "u! Are you blind that you can't see our hands and feet are quite dea"
            'd?"\n\nStill lower descended Frost, still more put forth his might, an'
            'd said:\n\n"Are ye warm, maidens?"\n\n"Into the bottomless pit with you!'
            ' Out of sight, accursed one!" cried the girls--and became lifeless f'
            'orms.\n\nNext morning the old woman said to her husband:\n\n"Old man, go'
            ' and get the sledge harnessed; put an armful of hay in it, and take '
            'some sheep-skin wraps. I daresay the girls are half-dead with cold. '
            "There's a terrible frost outside! And, mind you, old greybeard, do i"
            't quickly!"\n\nBefore the old man could manage to get a bite he was ou'
            't of doors and on his way. When he came to where his daughters were,'
            ' he found them dead. So he lifted the girls on to the sledge, wrappe'
            'd a blanket round them, and covered them up with a bark mat. The old'
            ' woman saw him from afar, ran out to meet him, and called out ever s'
            'o loud:\n\n"Where are the girls?"\n\n"In the sledge."\n\nThe old woman lif'
            'ted the mat, undid the blanket, and found the girls both dead.\n\nThen'
            ', like a thunderstorm, she broke out against her husband, abusing hi'
            'm saying:\n\n"What have you done, you old wretch? You have destroyed m'
            'y daughters, the children of my own flesh and blood, my never-enough'
            '-to-be-gazed-on seedlings, my beautiful berries! I will thrash you w'
            'ith the tongs; I will give it you with the stove-rake."\n\n"That\'s eno'
            'ugh, you old goose! You flattered yourself you were going to get ric'
            'hes, but your daughters were too stiff-necked. How was I to blame? i'
            't was you yourself would have it."\n\nThe old woman was in a rage at f'
            'irst, and used bad language; but afterwards she made it up with her '
            'stepdaughter, and they all lived together peaceably, and thrived, an'
            'd bore no malice. A neighbor made an offer of marriage, the wedding '
            'was celebrated, and Marfa is now living happily. The old man frighte'
            "ns his grandchildren with (stories about) Frost, and doesn't let the"
            'm have their own way.'
        ),
        {
            'is_tale_structured': True,
            'n_moves': 2,
            'functions': [
                {'function': 'villainy', 'span': 'drive straight along the road at first, and then turn off from the road to the right, you know, into the forest--right up to the big pine that stands on the hill, and there hand Marfa over to Morozko (Frost).',
                 'move': 1, 'event_order': 1, 'note': 'The stepmother commands that Marfa be abandoned to Frost.'},
                {'function': 'mediation', 'span': 'The old man silently placed the things on the sledge, made his daughter put on a warm pelisse, and set off on the journey.',
                 'move': 1, 'event_order': 2, 'note': 'The father is dispatched to carry out the abandonment.'},
                {'function': 'donor_first_function', 'span': 'Art thou warm, maiden?',
                 'move': 1, 'event_order': 3, 'note': 'Frost tests the girl by questioning her.'},
                {'function': 'hero_reaction', 'span': 'Warm, warm am I, dear Father Frost',
                 'move': 1, 'event_order': 4, 'note': 'The girl responds politely and patiently.'},
                {'function': 'receipt_of_magical_agent', 'span': 'Then Frost took pity on the girl, wrapped her up in furs, and warmed her with blankets.',
                 'move': 1, 'event_order': 5, 'note': 'Frost rewards her with warmth and gifts.'},
                {'function': 'return', 'span': 'He stowed everything away on the sledge without saying a word, took his seat on it with his daughter, and drove back.',
                 'move': 1, 'event_order': 6, 'note': 'The father brings Marfa home.'},
                {'function': 'lack', 'span': "Take my daughters, too, to their bridegroom. The presents he's made are nothing to what he'll give them.",
                 'move': 2, 'event_order': 7, 'note': 'The stepmother desires rich gifts for her own daughters.'},
                {'function': 'mediation', 'span': 'Well, early next morning the old woman gave her girls their breakfast, dressed them as befitted brides, and sent them off on their journey.',
                 'move': 2, 'event_order': 8, 'note': 'The stepmother dispatches her daughters.'},
                {'function': 'beginning_counteraction', 'span': 'In the same way as before the old man left the girls under the pine.',
                 'move': 2, 'event_order': 9, 'note': 'The father agrees to carry out the counteraction.'},
                {'function': 'departure', 'span': 'In the same way as before the old man left the girls under the pine.',
                 'move': 2, 'event_order': 10, 'note': 'The girls depart from home to the forest.'},
                {'function': 'donor_first_function', 'span': 'Are ye warm, maidens? Are ye warm, pretty ones? Are ye warm, my darlings?',
                 'move': 2, 'event_order': 11, 'note': 'Frost tests the girls by questioning them.'},
                {'function': 'hero_reaction', 'span': "Oh, Frost, it's awfully cold! we're utterly perished! We're expecting a bridegroom, but the confounded fellow has disappeared.",
                 'move': 2, 'event_order': 12, 'note': 'The girls respond rudely and impatiently.'},
                {'function': 'receipt_of_magical_agent', 'span': 'Into the bottomless pit with you! Out of sight, accursed one!',
                 'move': 2, 'event_order': 13, 'note': 'The girls receive death as the result of their reaction.'},
                {'function': 'return', 'span': 'When he came to where his daughters were, he found them dead. So he lifted the girls on to the sledge, wrapped a blanket round them, and covered them up with a bark mat.',
                 'move': 2, 'event_order': 14, 'note': 'The father brings the dead girls home.'},
            ],
            'overall_note': "Propp's scheme for this tale is I. A9 B5 D1 E1 f1 ↓ || II. [a6] B2^5 C ↑ D1 E_1 F_ ↓ — 14 functions across 2 move(s).",
        },
    ),
    # Ashliman/Grimm, ATU 1430 — a jest, not a wondertale.
    (
        (
            'Lean Lisa was not at all like Lazy Heinz and Fat Trina, who would no'
            't allow anything to disturb their rest. She burned herself out from '
            'morning until evening and loaded so much work on her husband, Lanky '
            'Lenz, that it was harder for him than for a donkey loaded with three'
            ' sacks. But it was all for naught. They had nothing, and they got no'
            'thing. One evening she was lying in bed, too tired to move a muscle '
            'but still unable to fall asleep, when she poked her husband in the s'
            "ide with her elbow and said, 'Lenz, listen to what I just thought of"
            '. If I were to find a florin, and you were to give me another one, t'
            "hen I'd borrow yet another one, and you'd give me still another one,"
            " and then I would take the four florins and buy a young cow.' The ma"
            "n agreed. 'I don't know,' he said, 'where I'm to get that florin I'm"
            ' supposed to give you, but after you have the money to buy a cow, it'
            " will be a good thing.' Then he added, 'I'm looking forward to the t"
            'ime after the cow calves, so I can have some good refreshing milk to'
            " drink.' 'The milk is not for you,' said the woman. 'We will let the"
            ' calf suck, so it will grow large and fat, and we can sell it for a '
            "good price.' 'Of course,' said the man, 'but it won't hurt anything "
            "if we take a little milk.' 'Who taught you about cows?' said the wom"
            "an. 'I won't allow it, whether it will hurt anything or not. You can"
            " stand on your head, but you won't get a single drop of milk. Lanky "
            'Lenz, just because you are always hungry, you think that you can dev'
            "our everything that my hard work brings in.' 'Woman,' said the man, "
            "'be quiet, or I'll plant one on the side of your face.' 'What!' she "
            "cried. 'Are you threatening me! You glutton! You good-for-nothing! Y"
            "ou lazybones!' She was reaching for his hair, but Lanky Lenz raised "
            'himself up, took hold of both her skinny arms with one hand, then pu'
            'shed her head into the pillow with the other one. He held her there '
            'and let her scold until she fell asleep from exhaustion. The next mo'
            'rning when she woke up, I do not know whether she continued to quarr'
            'el, or whether she went out to look for the florin that she wanted t'
            'o find.'
        ),
        {
            'is_tale_structured': False,
            'n_moves': 0,
            'functions': [],
            'overall_note': "A jest built on a daydream that collapses. Nothing is lacking that a quest could supply and no villain acts; the quarrel resolves nothing and restores nothing. Grimm's register makes it look like a wondertale, but there is no move here, so no function is annotated.",
        },
    ),
)
