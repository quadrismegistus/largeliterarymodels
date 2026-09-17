"""The question types, their validation, and the canonical form the key hashes."""
import json

import pytest

from largeliterarymodels.questions import (
    Choice, Noul, QuestionError, Score, canonical, questions_sha256, to_wire,
)


class TestWireForm:
    def test_noul_minimal(self):
        assert to_wire({"q": Noul(instructions="Is it abstract?")}) == {
            "q": {"type": "noul", "instructions": "Is it abstract?"}}

    def test_noul_omits_criteria_when_absent(self):
        # Absent, not null: a `criteria: null` in the payload is a different
        # administration from no criteria at all, and would re-key every
        # question written before criteria existed.
        assert "criteria" not in to_wire({"q": Noul("x")})["q"]

    def test_structured_instructions_pass_through(self):
        ins = {"question": "Does it bind?", "inspect": "passage",
               "focus": "Judge the binding, not the detail."}
        assert to_wire({"q": Noul(instructions=ins)})["q"]["instructions"] == ins

    def test_score_levels_keep_their_order(self):
        s = Score("How abstract?", ["concrete", "mixed", "abstract"])
        assert to_wire({"q": s})["q"]["criteria"] == [
            "concrete", "mixed", "abstract"]

    def test_raw_dict_passes_through(self):
        raw = {"type": "noul", "instructions": "x", "weight": 2}
        assert to_wire({"q": raw})["q"] == raw


class TestValidation:
    def test_empty_instructions(self):
        for bad in (None, "", {}, []):
            with pytest.raises(QuestionError, match="empty"):
                Noul(instructions=bad)

    def test_unserialisable_instructions(self):
        with pytest.raises(QuestionError, match="JSON"):
            Noul(instructions={"f": object()})

    def test_noul_criteria_is_true_false_only(self):
        with pytest.raises(QuestionError, match="Choice"):
            Noul("x", criteria={"true": "a", "false": "b", "maybe": "c"})

    def test_choice_needs_two_options(self):
        with pytest.raises(QuestionError, match="at least 2"):
            Choice("x", {"only": None})

    def test_choice_rejects_a_list(self):
        with pytest.raises(QuestionError, match="Score"):
            Choice("x", ["a", "b"])

    def test_score_rejects_a_mapping(self):
        with pytest.raises(QuestionError, match="Choice"):
            Score("x", {"low": "a", "high": "b"})

    @pytest.mark.parametrize("n", [0, 1, 11, 20])
    def test_score_level_count_bounds(self, n):
        with pytest.raises(QuestionError, match="levels"):
            Score("x", [f"l{i}" for i in range(n)])

    @pytest.mark.parametrize("n", [2, 3, 7, 10])
    def test_score_level_count_allowed(self, n):
        assert Score("x", [f"l{i}" for i in range(n)]).n_levels == n

    def test_empty_question_set(self):
        with pytest.raises(QuestionError, match="at least one"):
            to_wire({})

    def test_bad_question_id(self):
        for bad in ("", 3, None):
            with pytest.raises(QuestionError, match="question id"):
                to_wire({bad: Noul("x")})

    def test_wrong_type_named_in_the_error(self):
        with pytest.raises(QuestionError, match="str; expected"):
            to_wire({"q": "just a string"})


class TestCanonicalIdentity:
    Q = {"a": Noul("Is it abstract?"),
         "b": Score("How much?", ["low", "high"])}

    def test_round_trips_to_the_wire_form(self):
        # canonical() IS what is sent. If these ever diverge, the digest
        # describes something other than the administration.
        assert json.loads(canonical(self.Q)) == to_wire(self.Q)

    def test_stable_across_calls(self):
        assert questions_sha256(self.Q) == questions_sha256(self.Q)

    def test_question_order_is_part_of_the_instrument(self):
        flipped = {"b": self.Q["b"], "a": self.Q["a"]}
        assert questions_sha256(flipped) != questions_sha256(self.Q)

    def test_score_level_order_changes_the_digest(self):
        # Reordering levels renumbers every position, so every answer means
        # something else. This must never hash alike.
        a = {"q": Score("x", ["low", "high"])}
        b = {"q": Score("x", ["high", "low"])}
        assert questions_sha256(a) != questions_sha256(b)

    def test_choice_option_order_changes_the_digest(self):
        a = {"q": Choice("x", {"p": None, "q": None})}
        b = {"q": Choice("x", {"q": None, "p": None})}
        assert questions_sha256(a) != questions_sha256(b)

    def test_adding_criteria_changes_the_digest(self):
        bare = {"q": Noul("x")}
        crit = {"q": Noul("x", criteria={"true": "yes", "false": "no"})}
        assert questions_sha256(bare) != questions_sha256(crit)

    def test_wording_change_changes_the_digest(self):
        # The questionnaire-edit rule: a description change is a different
        # instrument, not a cosmetic edit.
        a = {"q": Noul("Is it abstract?")}
        b = {"q": Noul("Is it abstract? ")}
        assert questions_sha256(a) != questions_sha256(b)

    def test_digest_does_not_depend_on_the_sdk_being_installed(self):
        # The whole reason these types exist rather than the SDK's: an SDK
        # upgrade must not be able to re-key a corpus.
        import largeliterarymodels.questions as m
        assert "typesafe_sdk" not in json.dumps(
            [getattr(m, n).__module__ for n in ("Noul", "Choice", "Score")])

    def test_non_ascii_is_not_escaped(self):
        q = {"q": Noul("Does it name 七情?")}
        assert "七情" in canonical(q)
