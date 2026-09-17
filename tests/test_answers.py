"""Parsing a System One body, and the thresholding that must stay explicit."""
import pytest

from largeliterarymodels import answers as A

BODY = {
    "model": "jev-1.13.0",
    "usage": {"input_tokens": 326, "output_tokens": 33},
    "answers": {
        "abstract": {"type": "noul", "noul": 0.85},
        "mode": {"type": "choice", "choice": "didactic", "confidence": 0.51,
                 "probabilities": {"narrative": 0.0, "descriptive": 0.32,
                                   "didactic": 0.68}},
        "degree": {"type": "score", "score": 1.98, "confidence": 0.97,
                   "legend": {"0": "concrete", "1": "mixed", "2": "abstract"},
                   "probabilities": {"0": 0.0, "1": 0.02, "2": 0.98}},
    },
}


class TestParse:
    def test_types(self):
        a = A.parse(BODY)
        assert isinstance(a["abstract"], A.NoulAnswer)
        assert isinstance(a["mode"], A.ChoiceAnswer)
        assert isinstance(a["degree"], A.ScoreAnswer)

    def test_carries_provenance(self):
        a = A.parse(BODY)
        assert a.model == "jev-1.13.0"
        assert a.usage["input_tokens"] == 326

    def test_is_a_mapping(self):
        a = A.parse(BODY)
        assert set(a) == {"abstract", "mode", "degree"} and len(a) == 3

    def test_missing_answer_names_itself(self):
        a = A.parse(BODY)
        with pytest.raises(A.AnswerError, match="no answer for 'nope'"):
            a["nope"]

    def test_no_answers_field_is_called_drift(self):
        with pytest.raises(A.AnswerError, match="provider drift"):
            A.parse({"model": "x", "usage": {}})

    def test_non_mapping_body(self):
        with pytest.raises(A.AnswerError, match="not a\\s+mapping"):
            A.parse("[]")

    def test_malformed_answer_reports_the_raw(self):
        bad = {"answers": {"q": {"type": "noul"}}}  # no `noul` field
        with pytest.raises(A.AnswerError, match="does not carry its"):
            A.parse(bad)


class TestUnknownKinds:
    BODY = {"answers": {"q": {"type": "quantile", "q50": 3, "q90": 7}}}

    def test_kept_not_skipped(self):
        # The SDK skips these. Skipping is how a provider's new field goes
        # unnoticed until the run that needed it is already paid for.
        a = A.parse(self.BODY)
        assert isinstance(a["q"], A.UnknownAnswer)
        assert a["q"].body == {"type": "quantile", "q50": 3, "q90": 7}

    def test_reported(self):
        assert A.parse(self.BODY).unknown == ("q",)

    def test_known_kinds_report_nothing(self):
        assert A.parse(BODY).unknown == ()


class TestMissing:
    def test_empty_when_everything_answered(self):
        assert A.parse(BODY, asked=["abstract", "mode", "degree"]).missing == ()

    def test_names_what_was_asked_and_not_answered(self):
        a = A.parse(BODY, asked=["abstract", "ghost"])
        assert a.missing == ("ghost",) and a.unanswered


class TestNoThresholdByDefault:
    def test_there_is_no_bool_accessor(self):
        n = A.parse(BODY)["abstract"]
        assert not hasattr(n, "bool")
        assert not hasattr(n, "__bool__") or isinstance(n, A.NoulAnswer)

    def test_at_requires_an_explicit_cut(self):
        n = A.parse(BODY)["abstract"]
        with pytest.raises(TypeError):
            n.at()
        assert n.at(0.5) is True
        assert n.at(0.9) is False

    def test_value_is_the_probability_not_a_label(self):
        assert A.parse(BODY).value("abstract") == 0.85

    def test_p_refuses_a_non_noul(self):
        a = A.parse(BODY)
        with pytest.raises(A.AnswerError, match="not a Noul"):
            a.p("degree")


class TestChoice:
    def test_option_mass(self):
        c = A.parse(BODY)["mode"]
        assert c.p("didactic") == 0.68
        assert c.p("absent-option") == 0.0

    def test_margin_is_not_confidence(self):
        c = A.parse(BODY)["mode"]
        assert c.margin() == pytest.approx(0.36)
        assert c.confidence == 0.51

    def test_margin_needs_two_options(self):
        one = {"answers": {"q": {"type": "choice", "choice": "a",
                                 "probabilities": {"a": 1.0}}}}
        assert A.parse(one)["q"].margin() is None


class TestScore:
    def test_position_and_normalization(self):
        s = A.parse(BODY)["degree"]
        assert s.score == 1.98
        assert s.n_levels == 3
        assert s.normalized == pytest.approx(0.99)

    def test_normalization_makes_scales_comparable(self):
        # A 4-level and a 3-level question at the top both read 1.0; this is
        # what let a Score be compared against a Noul in SD units.
        four = {"answers": {"q": {"type": "score", "score": 3.0,
                                  "legend": {"0": "a", "1": "b", "2": "c",
                                             "3": "d"}}}}
        assert A.parse(four)["q"].normalized == 1.0

    def test_level_rounds_but_is_not_the_default(self):
        s = A.parse(BODY)["degree"]
        assert s.level() == 2
        assert s.value == 1.98, "the continuous position stays the value"

    def test_degenerate_level_count(self):
        bare = {"answers": {"q": {"type": "score", "score": 0.0}}}
        assert A.parse(bare)["q"].normalized is None
