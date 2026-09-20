"""The wire shape, pinned against a REAL response.

Every other fixture in this suite is synthetic, which means it was written to
match the parser — it can only fail if the parser contradicts itself, never if
the API changes. This one was recorded from a live call to jev-latest on
2026-09-20 and is the only thing in the suite that would notice TypeSafe
changing their envelope.

It is a recording, not a contract: it cannot tell us the API is still like
this today, only that the parser still handles what the API did then. The
class of bug it catches is ours drifting from a fixed reality, which is the
half a synthetic fixture cannot reach. Re-record it when the parser is next
touched, and expect it to be stale rather than trusting it to be current.

The practical argument for recording it at all, from the seat that ran the
first real corpus on this path: the cases that actually broke things were
accidents of a real vocabulary, not cases anyone designed — `bathro` passing a
gate while `legg` failed it. Synthetic cases keep being easier than the data.
"""
import json
import pathlib

import pytest

from largeliterarymodels import answers as A
from largeliterarymodels.questions import canonical, to_wire

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures"
     / "typesafe_system_one_response.json").read_text())
BODY = FIXTURE["response_body"]
ASKED = FIXTURE["request_questions"]


class TestRealResponseParses:
    def test_all_three_primitives_came_back(self):
        kinds = {a["type"] for a in BODY["answers"].values()}
        assert kinds == {"noul", "choice", "score"}

    def test_parses_with_nothing_unknown_and_nothing_missing(self):
        a = A.parse(BODY, asked=ASKED)
        assert a.unknown == (), "an unmodelled answer kind in a real response"
        assert a.missing == (), "the API answered everything it was asked"

    def test_envelope_keys(self):
        assert set(BODY) == {"answers", "model", "usage"}
        assert set(BODY["usage"]) == {"input_tokens", "output_tokens"}, \
            "no cache or reasoning fields — the absences the pricing row rests on"

    def test_the_resolved_model_is_not_the_alias_requested(self):
        # jev-latest resolves server-side; the id of record is this one.
        assert BODY["model"].startswith("jev-")
        assert BODY["model"] != "jev-latest"


class TestAnswerFieldsRealResponsesCarry:
    A = A.parse(BODY, asked=ASKED)

    def test_noul_is_a_bare_probability(self):
        n = self.A["plain_noul"]
        assert 0.0 <= n.p <= 1.0
        assert not hasattr(n, "confidence"), "Nouls carry no confidence"

    def test_choice_carries_a_distribution_over_the_options_sent(self):
        c = self.A["choice_q"]
        assert set(c.probabilities) == set(ASKED["choice_q"]["criteria"])
        assert c.choice in c.probabilities
        assert c.confidence is not None

    def test_score_carries_legend_and_per_level_probabilities(self):
        s = self.A["score_q"]
        n = len(ASKED["score_q"]["criteria"])
        assert set(s.legend) == {str(i) for i in range(n)}
        assert set(s.probabilities) == {str(i) for i in range(n)}
        assert 0.0 <= s.score <= n - 1
        assert s.normalized is not None


class TestStructuredFormsSurviveTheRoundTrip:
    """Structured instructions and criteria are the forms a hand-written
    fixture is least likely to get right, and the ones the API is most likely
    to change. These went out on the recorded call and came back answered."""

    def test_structured_instructions_were_sent_as_objects(self):
        ins = ASKED["structured_noul"]["instructions"]
        assert set(ins) == {"question", "inspect", "focus"}

    def test_structured_criteria_were_sent_as_objects(self):
        cr = ASKED["structured_noul"]["criteria"]
        assert "what" in cr["true"] and "examples" in cr["true"]
        assert "not_for" in cr["false"]

    def test_mixed_criteria_shapes_in_one_choice(self):
        # string, object and null options together — the API took all three.
        cr = ASKED["choice_q"]["criteria"]
        assert isinstance(cr["narrative"], str)
        assert isinstance(cr["didactic"], dict)
        assert cr["descriptive"] is None

    def test_a_score_level_may_be_an_object(self):
        assert isinstance(ASKED["score_q"]["criteria"][0], dict)
        assert isinstance(ASKED["score_q"]["criteria"][1], str)

    def test_every_structured_question_was_answered(self):
        a = A.parse(BODY, asked=ASKED)
        assert a["structured_noul"].p is not None


class TestOurCanonicalFormIsWhatWentOnTheWire:
    """The load-bearing claim of questions.py: canonical() IS the request."""

    def test_recorded_request_is_our_wire_form(self):
        assert json.loads(canonical(ASKED)) == ASKED

    def test_wire_form_is_idempotent_over_raw_dicts(self):
        assert to_wire(ASKED) == ASKED
