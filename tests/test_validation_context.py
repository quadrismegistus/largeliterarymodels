"""Per-item validation context, and the reprompt that stops lying.

A `model_validator(mode="after")` that raises was ALREADY a rejection hook —
both retry loops catch it and re-ask — and that was under-advertised. Three
things stopped it being usable for an item-dependent rule (malign-logits
seat, 2026-09-20): the validator could not see the item, the reprompt told
the model its JSON was malformed when it was not, and at temperature 0 the
repeat short-circuit fired before the new information could be used.
"""
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from largeliterarymodels.llm import (
    LLM, _retry_complaint, _retry_prompt, _validate_parsed,
)


class Groups(BaseModel):
    """The shape that motivated this: which column a word was shown in is a
    property of the ITEM, so no schema-only rule can check it."""
    words_b: list[str]

    @model_validator(mode="after")
    def _only_group_b(self, info):
        shown_in_a = (info.context or {}).get("group_a", ())
        bad = [w for w in self.words_b if w in shown_in_a]
        if bad:
            raise ValueError(
                f"the word {bad[0]} appears in words_b but was shown in GROUP A")
        return self


class FakeStash(dict):
    def __contains__(self, k):
        return dict.__contains__(self, _f(k))

    def __setitem__(self, k, v):
        dict.__setitem__(self, _f(k), v)

    def __getitem__(self, k):
        return dict.__getitem__(self, _f(k))


def _f(k):
    import json
    return json.dumps(k, sort_keys=True, default=str)


def _llm():
    return LLM("claude-sonnet-4-6", temperature=0.0, max_tokens=64,
               stash=FakeStash())


class TestContextReachesTheValidator:
    def test_without_context_the_rule_cannot_fire(self):
        # Not a bug — it is the gap. The validator has nothing to check.
        assert _validate_parsed({"words_b": ["strangle"]}, Groups).words_b

    def test_with_context_it_fires(self):
        with pytest.raises(ValidationError, match="shown in GROUP A"):
            _validate_parsed({"words_b": ["strangle"]}, Groups,
                             {"group_a": ["strangle"]})

    def test_context_flows_through_list_schemas_too(self):
        with pytest.raises(ValidationError):
            _validate_parsed([{"words_b": ["strangle"]}], list[Groups],
                             {"group_a": ["strangle"]})

    def test_it_is_not_in_the_cache_key(self):
        # It governs whether an answer is ACCEPTED, not what was
        # administered, and the item it derives from is already the prompt.
        from largeliterarymodels.llm import _make_key
        assert "validation_context" not in _make_key("p", "m")


class TestReprompt:
    def test_the_old_complaint_is_kept_for_unparseable_output(self):
        assert "not valid JSON" in _retry_complaint()

    def test_a_validation_failure_is_not_called_a_format_failure(self):
        # The false assertion: it parsed. Told its format was wrong, a model
        # reformats instead of fixing the defect.
        c = _retry_complaint(validation_error="strangle is in the wrong group")
        assert "not valid JSON" not in c
        assert "do not reformat" in c
        assert "strangle is in the wrong group" in c

    def test_partial_field_still_wins(self):
        c = _retry_complaint(partial_field="relations",
                             validation_error="something else")
        assert "`relations`" in c

    def test_a_long_message_is_capped_and_says_so(self):
        c = _retry_complaint(validation_error="x" * 5000)
        assert "truncated" in c and len(c) < 3000

    def test_the_prompt_survives_the_complaint(self):
        out = _retry_prompt("ORIGINAL", validation_error="bad")
        assert out.endswith("ORIGINAL")


class TestRetryLoopUsesIt:
    GOOD = '{"words_b": ["kiss"]}'
    BAD = '{"words_b": ["strangle"]}'

    @patch("largeliterarymodels.llm._call_provider")
    def test_a_semantic_failure_is_retried_with_a_named_defect(self, call):
        call.side_effect = [self.BAD, self.GOOD]
        out = _llm().extract("p", Groups, retries=2,
                             validation_context={"group_a": ["strangle"]})
        assert out.words_b == ["kiss"]
        second = call.call_args_list[1].kwargs.get("prompt") or \
            call.call_args_list[1].args[0]
        assert "shown in GROUP A" in second, \
            "the reprompt must name the defect, not complain about JSON"
        assert "not valid JSON" not in second

    @patch("largeliterarymodels.llm._call_provider")
    def test_the_informative_reprompt_always_gets_one_shot(self, call):
        """The answer to "does the repeat test make this inert at
        temperature 0": no, and not because the test changed — because the
        complaint did.

        Attempt 1 sends the plain prompt. Attempt 2 sends the reprompt, which
        now NAMES the defect, so the model is told what is wrong before
        anything short-circuits. Attempt 3 would send a byte-identical
        request, which at temperature 0 cannot produce a different answer, so
        stopping there is right and is the saving the short-circuit exists
        for. Before the reprompt carried the validation message, attempt 2
        spent itself asserting the JSON was malformed and the model never
        heard the real complaint — the inertness was in the MESSAGE, not in
        the loop."""
        call.side_effect = [self.BAD, self.BAD, self.GOOD]
        with pytest.raises(Exception):
            _llm().extract("p", Groups, retries=3,
                           validation_context={"group_a": ["strangle"]})
        assert call.call_count == 2
        second = (call.call_args_list[1].kwargs.get("prompt")
                  or call.call_args_list[1].args[0])
        assert "shown in GROUP A" in second, \
            "the one informative attempt must actually carry the defect"

    @patch("largeliterarymodels.llm._call_provider")
    def test_a_recoverable_semantic_failure_recovers_on_that_shot(self, call):
        call.side_effect = [self.BAD, self.GOOD]
        out = _llm().extract("p", Groups, retries=3,
                             validation_context={"group_a": ["strangle"]})
        assert out.words_b == ["kiss"] and call.call_count == 2

    @patch("largeliterarymodels.llm._call_provider")
    def test_an_unchanging_complaint_still_short_circuits(self, call):
        # The saving the short-circuit exists for is kept: once the same
        # complaint has been sent and produced the same error, stop.
        call.side_effect = [self.BAD] * 8
        with pytest.raises(Exception):
            _llm().extract("p", Groups, retries=6,
                           validation_context={"group_a": ["strangle"]})
        assert call.call_count == 2, "one plain attempt, one informative, stop"

    @patch("largeliterarymodels.llm._call_provider")
    def test_map_forwards_the_per_item_list(self, call):
        # extract_map accepted the argument and dropped it — the passthrough
        # asymmetry this package has a section about.
        call.side_effect = [self.BAD, self.GOOD, self.GOOD]
        errs = {}
        out = _llm().extract_map(
            ["a", "b"], Groups, num_workers=1, retries=0, errors=errs,
            validation_context_list=[{"group_a": ["strangle"]}, {}])
        assert out[0] is None, "item 0's rule must have been applied"
        assert out[1] is not None

    @patch("largeliterarymodels.llm._call_provider")
    def test_a_warm_hit_is_revalidated_under_the_items_rule(self, call):
        # A cached answer the item's own rule rejects must not be served.
        call.side_effect = [self.BAD, self.GOOD]
        llm = _llm()
        llm.extract("p", Groups, retries=0)          # cached with no rule
        out = llm.extract("p", Groups, retries=1,
                          validation_context={"group_a": ["strangle"]})
        assert out.words_b == ["kiss"], "the warm row failed the rule and recomputed"


class Setting(BaseModel):
    """The oh-james shape: a list schema whose items are checked against the
    source text, so one bad item rejects an otherwise good 40-item answer."""
    place: str
    quote: str

    @model_validator(mode="after")
    def _quote_in_source(self, info):
        source = (info.context or {}).get("source", "")
        if self.quote not in source:
            raise ValueError(f"the quote {self.quote!r} is not in the source")
        return self


class TestListSchemaRetry:
    """A stateless retry plus an item-level complaint closing on "return the
    corrected object" was answered with the fixed item alone, which validated
    and replaced the whole list (oh-james seat, 2026-10: 40 settings -> 1 on
    two of three films, deepseek-v4-pro, no error)."""

    SOURCE = {"source": "under the parachutes; at the bank; in the vault"}
    FULL_BAD = ('[{"place": "a", "quote": "under the parachutes"},'
                ' {"place": "b", "quote": "NOT THERE"},'
                ' {"place": "c", "quote": "in the vault"},'
                ' {"place": "d", "quote": "at the bank"}]')
    ONLY_FIXED = '[{"place": "b", "quote": "at the bank"}]'

    def test_the_rejection_names_the_failing_items_index(self):
        import json
        with pytest.raises(ValidationError) as ei:
            _validate_parsed(json.loads(self.FULL_BAD), list[Setting],
                             self.SOURCE)
        assert [e["loc"][0] for e in ei.value.errors()] == [1]

    def test_every_failing_item_is_reported_not_just_the_first(self):
        with pytest.raises(ValidationError) as ei:
            _validate_parsed([{"place": "a", "quote": "x"},
                              {"place": "b", "quote": "y"}],
                             list[Setting], self.SOURCE)
        assert [e["loc"][0] for e in ei.value.errors()] == [0, 1]

    def test_the_list_complaint_asks_for_the_complete_array(self):
        c = _retry_complaint(validation_error="1\n  bad quote", is_list=True)
        assert "COMPLETE array" in c and "every item" in c
        assert "corrected object" not in c
        assert "do not reformat" in c

    def test_the_object_complaint_is_unchanged(self):
        c = _retry_complaint(validation_error="bad")
        assert c.endswith("return the corrected object.")
        assert "COMPLETE" not in c

    @patch("largeliterarymodels.llm._call_provider")
    def test_extract_sends_the_list_complaint(self, call):
        call.side_effect = [self.FULL_BAD, self.ONLY_FIXED]
        _llm().extract("p", list[Setting], retries=1,
                       validation_context=self.SOURCE)
        second = call.call_args_list[1].kwargs["prompt"]
        assert "COMPLETE array" in second
        assert "list[Setting]\n1\n" in second, "the index reaches the model"

    @patch("largeliterarymodels.llm._call_provider")
    def test_imap_sends_the_list_complaint(self, call):
        call.side_effect = [self.FULL_BAD, self.ONLY_FIXED]
        _llm().extract_map(["p"], list[Setting], num_workers=1, retries=1,
                           validation_context_list=[self.SOURCE])
        assert "COMPLETE array" in call.call_args_list[1].kwargs["prompt"]

    @patch("largeliterarymodels.llm._call_provider")
    def test_a_collapsed_retry_is_announced(self, call, caplog):
        # Accepted, not rejected: dropping the bad item can be a legitimate
        # fix. But 4 -> 1 must leave a trace beyond a row count.
        call.side_effect = [self.FULL_BAD, self.ONLY_FIXED]
        out = _llm().extract("p", list[Setting], retries=1,
                             validation_context=self.SOURCE)
        assert len(out) == 1
        assert "returned 1 item(s) where the rejected response had 4" \
            in caplog.text

    @patch("largeliterarymodels.llm._call_provider")
    def test_a_collapsed_imap_retry_is_announced(self, call, caplog):
        call.side_effect = [self.FULL_BAD, self.ONLY_FIXED]
        _llm().extract_map(["p"], list[Setting], num_workers=1, retries=1,
                           validation_context_list=[self.SOURCE])
        assert "rejected response had 4" in caplog.text

    @patch("largeliterarymodels.llm._call_provider")
    def test_a_complete_retry_is_not_announced(self, call, caplog):
        call.side_effect = [self.FULL_BAD, self.FULL_BAD.replace(
            "NOT THERE", "at the bank")]
        out = _llm().extract("p", list[Setting], retries=1,
                             validation_context=self.SOURCE)
        assert len(out) == 4 and "rejected response had" not in caplog.text

    def test_the_hand_path_says_the_same_thing(self):
        from largeliterarymodels.task import Task

        class T(Task):
            schema = list[Setting]
            system_prompt = "Extract settings."
        T.name = "list_retry_parity"
        assert T().retry_prompt("ITEM", None, "bad") == \
            _retry_prompt("ITEM", None, "bad", is_list=True)
        assert "COMPLETE array" in T().retry_prompt("ITEM", None, "bad")
