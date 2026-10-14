"""ProppTask: the vocabulary, the instrument, and the span audit.

The span audit is the part that most needs pinning. A quoted span reads as
verifiable evidence whether or not anyone verified it, so the failure mode
is silent by construction: a paraphrased quote and a real one look identical
in the output.
"""

import pytest
from pydantic import ValidationError

from largeliterarymodels.costs import count_tokens
from largeliterarymodels.tasks.classify_propp import (
    FUNCTION_VOCAB, PROPP_FUNCTIONS, STORY_STAGES, ProppAnnotation,
    ProppFunctionInstance, ProppTask, locate_spans, span_report)


def _fn(function, span, move=1, order=1, note=""):
    return ProppFunctionInstance(function=function, span=span, move=move,
                                 event_order=order, note=note)


def _ann(*functions, tale=True, moves=1):
    return ProppAnnotation(is_tale_structured=tale, n_moves=moves,
                           overall_note="", functions=list(functions))


class TestVocabulary:
    def test_the_thirty_one_functions_are_all_present(self):
        """Propp's 31, plus α (not a function, his preparatory element),
        plus a (Lack, slot VIIIa), plus X (his residual sign)."""
        assert len(FUNCTION_VOCAB) == 34
        assert len(set(FUNCTION_VOCAB)) == 34

    def test_villainy_and_lack_are_both_offered(self):
        """They are ALTERNATIVES in one slot, not parent and child. A tale
        driven by a lack would otherwise be forced to report villainy."""
        assert 'villainy' in FUNCTION_VOCAB
        assert 'lack' in FUNCTION_VOCAB

    def test_branding_precedes_victory(self):
        """J = branding, I = victory, and J comes first. Counter-intuitive
        enough that a peer-reviewed summary table has them reversed — while
        its own worked formula contradicts its table."""
        names = list(FUNCTION_VOCAB)
        by_symbol = {f['symbol']: f['name'] for f in PROPP_FUNCTIONS}
        assert by_symbol['J'] == 'branding'
        assert by_symbol['I'] == 'victory'
        assert names.index('branding') < names.index('victory')

    def test_alpha_and_lowercase_a_are_distinct(self):
        """A widely-circulated retyping collapses Greek α (initial
        situation) into Latin a (lack). They are different functions."""
        by_symbol = {f['symbol']: f['name'] for f in PROPP_FUNCTIONS}
        assert by_symbol['α'] == 'initial_situation'
        assert by_symbol['a'] == 'lack'

    def test_stages_follow_propps_own_sectioning(self):
        """Not the six-stage scheme of the secondary literature: Propp's
        donor bloc is D-E-F only (G sits with the struggle), and struggle
        and return are one continuous block, not two stages."""
        stages = {f['stage'] for f in PROPP_FUNCTIONS}
        assert stages <= set(STORY_STAGES)
        by_name = {f['name']: f['stage'] for f in PROPP_FUNCTIONS}
        assert by_name['spatial_transference'] == 'helper_to_end_of_first_move'
        assert by_name['receipt_of_magical_agent'] == 'donors'
        assert by_name['struggle'] == by_name['return'] == \
            'helper_to_end_of_first_move'


class TestInstrument:
    def test_every_function_is_defined_in_the_prompt(self):
        """The schema carries only the enum NAMES. Without the glossary in
        the system prompt the model gets 34 bare labels and classifies on
        vocabulary — the one thing Propp's scheme is designed to defeat."""
        instrument = ProppTask().instrument_text()
        for f in PROPP_FUNCTIONS:
            assert f['name'] in instrument
            assert f['description'][:40] in instrument

    def test_the_instrument_clears_every_cache_floor(self):
        """Measured, not estimated from character count — measured density
        ranges 2.7-3.9 chars/token and a char-based estimate of this
        instrument was 26% high. Below a model's floor Anthropic silently
        declines to cache and every call bills at full input price, so a
        SHORTER instrument here would cost several times more on Haiku.
        The margin over the 4,096 floor is thin: trim with this in view."""
        n = count_tokens(ProppTask().instrument_text())
        assert n >= 4096, (
            f"instrument is {n} tokens, below Haiku 4.5's 4,096-token cache "
            f"floor — it would bill uncached on every call")

    def test_the_empty_answer_is_licensed_in_the_prompt(self):
        """Propp scoped the scheme to the wondertale. Without an explicit
        licence to return nothing, a model strains to find functions in
        material that has none, and over-firing on the null case would
        quietly wreck the dataset."""
        instrument = ProppTask().instrument_text()
        assert "EMPTY OR NEARLY EMPTY ANSWER IS A LEGITIMATE ANSWER" in instrument


class TestSpanAudit:
    TEXT = ("Once there was an old man who had three sons.\n\n"
            "The father went away to market.   Before he left he said: "
            "'Do not open the cellar door.'\n\n"
            "The youngest son opened the cellar door.")

    def test_an_exact_quote_is_located(self):
        r = locate_spans(self.TEXT, _ann(
            _fn('absentation', 'The father went away to market.')))[0]
        assert r['method'] == 'exact' and r['verbatim'] is True
        assert self.TEXT[r['start']:r['end']] == 'The father went away to market.'

    def test_a_fabricated_quote_is_caught(self):
        """The whole point. A span the model invented is indistinguishable
        from one it copied, unless something checks."""
        r = locate_spans(self.TEXT, _ann(
            _fn('villainy', 'a serpent swooped down and stole the girl')))[0]
        assert r['method'] == 'not_found'
        assert r['verbatim'] is False and r['start'] is None

    def test_a_whitespace_paraphrase_is_located_but_not_verbatim(self):
        """Located, so the offsets are usable; flagged, because the model
        did not in fact copy the characters it claimed to."""
        r = locate_spans("He   went\n away to  market.", _ann(
            _fn('absentation', 'He went away to market.')))[0]
        assert r['method'] == 'whitespace'
        assert r['verbatim'] is False
        assert r['start'] == 0 and r['end'] == 27

    def test_a_repeated_span_advances_through_the_text(self):
        """Propp's tales repeat functions — the donor sequence is
        classically trebled — so the same span can legitimately appear
        three times, and each occurrence must get its own offsets rather
        than three copies of the first."""
        text = "He asked. He asked. He asked."
        located = locate_spans(text, _ann(
            _fn('donor_first_function', 'He asked.', order=1),
            _fn('donor_first_function', 'He asked.', order=2),
            _fn('donor_first_function', 'He asked.', order=3)))
        assert [r['start'] for r in located] == [0, 10, 20]
        assert all(r['method'] == 'exact' for r in located)

    def test_the_report_counts_fabrication(self):
        located = locate_spans(self.TEXT, _ann(
            _fn('absentation', 'The father went away to market.'),
            _fn('villainy', 'a serpent carried off his sister')))
        rep = span_report(located)
        assert rep['total'] == 2 and rep['exact'] == 1
        assert rep['fabricated'] == 1 and rep['fabricated_indices'] == [1]
        assert rep['verbatim_rate'] == 0.5

    def test_an_empty_annotation_reports_cleanly(self):
        """A non-tale returns nothing, and nothing must not divide by zero."""
        rep = span_report(locate_spans(self.TEXT, _ann(tale=False, moves=0)))
        assert rep == {'total': 0, 'exact': 0, 'whitespace_only': 0,
                       'fabricated': 0, 'fabricated_indices': [],
                       'verbatim_rate': 0.0}


class TestSchema:
    def test_an_unknown_function_is_rejected(self):
        """The enum is what stops the label set drifting into a long tail
        of near-synonyms across runs."""
        with pytest.raises(ValidationError):
            _fn('hero_gets_a_horse', 'x')

    def test_a_non_tale_can_report_no_functions(self):
        ann = _ann(tale=False, moves=0)
        assert ann.functions == [] and ann.n_moves == 0

    def test_moves_are_carried_per_function(self):
        """Functions are ordered WITHIN a move. Without the index a
        two-move tale reads as a violation of Propp's ordering claim —
        villainy after a wedding — when it is perfectly canonical."""
        ann = _ann(_fn('wedding', 'they married', move=1, order=1),
                   _fn('villainy', 'the brothers stole it', move=2, order=2),
                   moves=2)
        assert [f.move for f in ann.functions] == [1, 2]


class TestFewShotExamples:
    """The examples carry Propp's own function sequences. Two things must
    hold or they teach the wrong lesson."""

    def test_every_example_span_is_verbatim_in_its_own_text(self):
        """An example whose quote does not appear in its own text teaches
        the model to fabricate quotes — and the span audit would then be
        measuring a habit the instrument installed."""
        from largeliterarymodels.tasks._propp_examples import EXAMPLES
        for text, ann in EXAMPLES:
            for f in ann['functions']:
                assert f['span'] in text, (
                    f"{f['function']} span not found verbatim in its own "
                    f"example text: {f['span'][:60]!r}")

    def test_every_example_function_is_in_the_vocabulary(self):
        from largeliterarymodels.tasks._propp_examples import EXAMPLES
        for _text, ann in EXAMPLES:
            for f in ann['functions']:
                assert f['function'] in FUNCTION_VOCAB

    def test_the_examples_are_sparse(self):
        """The defect they exist to correct is over-annotation: zero-shot
        the task returned 17-26 functions where Propp records 8-14. An
        example denser than Propp would reinforce exactly that."""
        from largeliterarymodels.tasks._propp_examples import EXAMPLES
        for text, ann in EXAMPLES:
            per_100w = 100 * len(ann['functions']) / len(text.split())
            assert per_100w < 1.5, (
                f"example annotates {per_100w:.2f} functions per 100 words; "
                f"Propp's own density across these tales is under 1.5")

    def test_a_negative_example_is_present(self):
        """Without one the model strains to find functions in material that
        has none, and over-firing on the null case is the failure that
        would quietly wreck a corpus-scale run."""
        from largeliterarymodels.tasks._propp_examples import EXAMPLES
        nulls = [a for _t, a in EXAMPLES if not a['is_tale_structured']]
        assert len(nulls) >= 1
        assert all(n['functions'] == [] and n['n_moves'] == 0 for n in nulls)

    def test_multi_move_is_demonstrated(self):
        """Move numbering is the thing a model will not produce unprompted."""
        from largeliterarymodels.tasks._propp_examples import EXAMPLES
        assert any(a['n_moves'] >= 2 for _t, a in EXAMPLES)


class TestPreparation:
    """prepare_tale_text removes three kinds of editorial apparatus. Each one was
    found by a failure, not by inspection."""

    def test_a_provenance_header_is_removed(self):
        """The packaged tale texts open with a header that names the tale
        AND summarises its plot ("the boy who understands birds and rises
        to greatness"). Feeding a file verbatim hands the annotator the
        answer before it reads a word — it contaminates every judgement,
        and `is_tale_structured` most of all."""
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        raw = ("Afanasyev no.: 247\nEnglish title: The Language of the Birds\n"
               "Match confidence: CERTAIN — the boy who understands birds "
               "and rises to greatness.\n"
               "====================================================\n\n"
               "In a certain city there was a merchant.")
        out = prepare_tale_text(raw)
        assert out == "In a certain city there was a merchant."

    def test_a_rule_deep_in_the_text_is_not_a_header(self):
        """Only the first 20 lines are searched: a rule used as a scene
        break must not decapitate the tale."""
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        body = "\n".join(f"line {i}" for i in range(30))
        out = prepare_tale_text(body + "\n" + "=" * 40 + "\nafter")
        assert out.startswith("line 0")
        assert "after" in out

    def test_footnote_markers_and_emphasis_go(self):
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        assert prepare_tale_text("thirty _puds_ of hemp,[283] and off") == \
            "thirty puds of hemp, and off"

    def test_hard_wrapping_is_joined_but_paragraphs_kept(self):
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        assert prepare_tale_text("one\ntwo\n\nthree\nfour") == "one two\n\nthree four"


class TestMoveDefault:
    def test_a_missing_move_defaults_to_one(self):
        """Observed live: the model omits `move` on an item now and then,
        and a required field discarded the whole tale's annotation over one
        integer. Most tales are single-move, so 1 is the honest default."""
        f = ProppFunctionInstance(function='villainy', span='x',
                                  event_order=1, note='')
        assert f.move == 1

    def test_an_explicit_move_still_wins(self):
        f = ProppFunctionInstance(function='villainy', span='x', move=2,
                                  event_order=1, note='')
        assert f.move == 2


class TestPrepareTextIsNotDestructive:
    """prepare_text targets editorial apparatus, and everything it targets
    has an innocent lookalike. These pin the lookalikes."""

    def test_a_scene_divider_is_not_a_header(self):
        """A rule alone is a scene divider far more often than a header;
        treating it as one silently deleted a story's opening."""
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        t = "Once there was a king.\n\n--------------------\n\nHe had three sons."
        assert "Once there was a king." in prepare_tale_text(t)

    def test_a_real_header_is_still_removed(self):
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        t = ("Afanasyev no.: 247\nMatch confidence: CERTAIN — the boy who "
             "understands birds.\n" + "=" * 40 + "\n\nIn a certain city.")
        assert prepare_tale_text(t) == "In a certain city."

    def test_single_newline_paragraphs_can_be_preserved(self):
        """LLM output routinely separates paragraphs with one newline;
        joining them is a real alteration, not tidying."""
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        t = "She left at dawn.\nThe road was long."
        assert prepare_tale_text(t) == "She left at dawn. The road was long."
        assert prepare_tale_text(t, join_wrapped=False) == t

    def test_clean_text_is_untouched_with_strippers_off(self):
        from largeliterarymodels.tasks.classify_propp import prepare_tale_text
        for t in ("The variable was named foo_bar_baz in the log.",
                  "He counted them [3] times.",
                  "It was _her_ decision.",
                  "Once there was a king.\n\n-------------------\n\nHe had sons."):
            assert prepare_tale_text(t, strip_footnotes=False, strip_emphasis=False,
                                strip_header=False, join_wrapped=False) == t


class TestRegistryHasNoDuplicates:
    def test_no_task_name_is_registered_twice(self):
        """A dict literal silently keeps the LAST duplicate key, so a
        collision is invisible at runtime and the loser's import resolves to
        someone else's function. Found live: `prepare_text` was registered
        by both classify_propp and score_alignment_asymmetry, and the
        documented import silently returned the wrong one — it accepts a
        single positional argument, so it did not even raise."""
        import ast
        import collections
        import largeliterarymodels.tasks as T
        src = open(T.__file__).read()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "_LAZY_IMPORTS"
                            for t in node.targets)):
                keys = [k.value for k in node.value.keys]
                dupes = [k for k, n in collections.Counter(keys).items() if n > 1]
                assert not dupes, f"duplicate registrations: {dupes}"
                return
        raise AssertionError("_LAZY_IMPORTS not found")

    def test_both_prepare_functions_remain_reachable(self):
        from largeliterarymodels.tasks import prepare_tale_text, prepare_text
        assert prepare_tale_text.__module__.endswith("classify_propp")
        assert prepare_text.__module__.endswith("score_alignment_asymmetry")
