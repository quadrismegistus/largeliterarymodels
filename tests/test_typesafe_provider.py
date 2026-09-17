"""The TypeSafe (System One / Jev) route.

What these pin is mostly ABSENCES, because the absences are what a later
change would quietly fill in with a plausible wrong value: no text, no
decoder parameters, no cache tier, no reasoning split. Each of those has a
specific way of going wrong, and each gets a test rather than a comment.
"""
import json

import pytest

from largeliterarymodels import costs
from largeliterarymodels import providers as P


class TestRouting:
    @pytest.mark.parametrize("model", [
        "jev-latest", "jev-preview", "jev-1.13.0", "JEV-LATEST",
        "typesafe/jev-latest", "typesafe/jev-preview",
    ])
    def test_routes_here(self, model):
        assert P.route_provider(model) is P.call_typesafe

    @pytest.mark.parametrize("model,expected", [
        ("claude-sonnet-5", "call_anthropic"),
        ("gpt-5.4-mini", "call_openai"),
        ("gemini-3.6-flash", "call_google"),
        ("deepseek/deepseek-v4-pro", "call_deepseek"),
        ("lmstudio/qwen3.5-35b-a3b", "call_local"),
        ("claude-cli/sonnet", "call_claude_cli"),
    ])
    def test_other_routes_untouched(self, model, expected):
        assert P.route_provider(model).__name__ == expected

    def test_provider_route_name_matches_the_sampling_table_key(self):
        # provider_route derives the key from the dispatcher, so a rename of
        # call_typesafe that missed _SAMPLING_SUPPORT would surface here
        # rather than as a silent _UNMEASURED raise at call time.
        assert P.provider_route("jev-latest") == "typesafe"
        assert "typesafe" in P._SAMPLING_SUPPORT

    def test_prefix_is_stripped_before_it_reaches_the_api(self):
        assert P._strip_prefix("typesafe/jev-preview") == "jev-preview"
        assert P._strip_prefix_typesafe("typesafe/jev-preview") == "jev-preview"
        assert P._strip_prefix_typesafe("jev-preview") == "jev-preview"

    def test_local_prefix_still_wins(self):
        # A local server can serve a checkpoint called jev; the prefix is the
        # caller's explicit statement of transport and must outrank a family
        # substring, exactly as it does for claude and gpt.
        assert P.route_provider("lmstudio/jev-7b") is P.call_local


class TestNotATextModel:
    def test_text_path_raises_rather_than_rendering(self):
        with pytest.raises(P.NotATextModel) as e:
            P.call_typesafe("anything", model="jev-latest")
        msg = str(e.value)
        assert "never returns text" in msg
        assert "Survey" in msg, "the error must name the thing to use instead"

    def test_it_is_a_typeerror(self):
        # So `except TypeError` in caller code catches it; this is a wrong-kind
        # -of-thing error, not a transport failure to be retried.
        assert issubclass(P.NotATextModel, TypeError)


class TestSamplingAndTemperature:
    @pytest.mark.parametrize("param", P.SAMPLING_PARAMS)
    def test_every_decoder_parameter_rejects(self, param):
        assert P._sampling_state("typesafe", param) is P._REJECTS

    def test_pinning_drops_rather_than_raising_or_keying(self):
        sampling, dropped = P.effective_sampling("jev-latest", {"top_p": 0.9})
        assert sampling == {}, "nothing may enter the key"
        assert dropped == ("top_p",)

    def test_rejection_reason_does_not_claim_a_transport_ate_it(self):
        assert (P._dropped_reason("typesafe", "top_p", "jev-latest")
                == "the System One API takes no decoder parameters at all")

    def test_temperature_never_enters_a_key(self):
        assert P.effective_temperature("jev-latest", 0.0) is None
        assert P.effective_temperature("typesafe/jev-preview", 1.0) is None

    def test_bare_kwarg_still_guarded(self):
        with pytest.raises(ValueError, match="sampling="):
            P.guard_bare_sampling("typesafe", "jev-latest", {"top_p": 0.9})


class TestUsage:
    BODY = {"model": "jev-1.13.0",
            "usage": {"input_tokens": 11480, "output_tokens": 1206},
            "answers": {}}

    def test_tokens_normalised(self):
        u = P._usage_typesafe(self.BODY)
        assert u["input_tokens"] == 11480
        assert u["output_tokens"] == 1206
        assert u["response_model"] == "jev-1.13.0"

    def test_no_cache_tier_is_zero_not_invented(self):
        u = P._usage_typesafe(self.BODY)
        assert u["cache_read_tokens"] == 0
        assert u["cache_write_tokens"] == 0
        assert P.cache_minimum_tokens("jev-latest") is None

    def test_reasoning_is_unreported_not_reported_zero(self):
        # The distinction UsageTracker.no_reasoning_observed() rests on: a
        # provider that does not report the field is not the same as one
        # reporting zero, and collapsing them is how a token-only gate passed
        # a family whose thinking cannot be disabled.
        u = P._usage_typesafe(self.BODY)
        assert u["reasoning_reported"] is False
        assert u["reasoning_observed"] is False
        assert u["reasoning_tokens"] == 0

    def test_empty_body_does_not_explode(self):
        for body in (None, {}, {"usage": None}):
            u = P._usage_typesafe(body)
            assert u["input_tokens"] == 0 and u["response_model"] is None


class TestPricing:
    def test_row_exists_with_a_source(self):
        p = json.load(open("largeliterarymodels/model_pricing.json"))
        assert "jev-latest" in p["typesafe"]
        assert p["sources"]["typesafe"]["url"]
        assert p["sources"]["typesafe"]["fetched"]

    def test_cached_is_null_so_cache_tokens_bill_at_full_rate(self):
        # null, not 0.0. A zero would price cached tokens as free; null is the
        # flag costs.py reads to bill them as fresh, which is what actually
        # happens on an API with no cache tier.
        p = json.load(open("largeliterarymodels/model_pricing.json"))
        assert p["typesafe"]["jev-latest"]["cached"] is None

    def test_output_is_free_and_that_is_a_measured_row_not_a_default(self):
        p = json.load(open("largeliterarymodels/model_pricing.json"))
        assert p["typesafe"]["jev-latest"]["output"] == 0.0

    @pytest.mark.parametrize("model", ["jev-latest", "typesafe/jev-latest",
                                       "jev-preview"])
    def test_resolves(self, model):
        prov, name, _ = costs.resolve(model)
        assert prov == "typesafe"

    def test_reproduces_two_measured_runs(self):
        # Receipts from 2026-09-17: the PassageFormTask calibration run over
        # 583 passages, and task_charge over 275 cells. Both were paid; if a
        # future edit to the row or to the cached-is-null handling changes
        # these, it changed a bill that is already known.
        assert costs.price("jev-latest", fresh=1_222_457,
                           output=113_685)["usd"] == pytest.approx(0.0513, abs=5e-5)
        assert costs.price("jev-latest", fresh=4_364_476,
                           output=454_922)["usd"] == pytest.approx(0.1833, abs=5e-5)

    def test_no_batch_discount_claimed(self):
        # No batch API observed. A default 0.5 here would promise a discount
        # that silently is not applied, which is the deepseek failure the
        # batch module raises on.
        p = json.load(open("largeliterarymodels/model_pricing.json"))
        assert p["typesafe"]["jev-latest"]["batch_discount"] is None
