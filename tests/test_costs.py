"""Tests for the provider-agnostic costing module.

The anchor is a real invoice: Registration P's gpt-4o-mini arm, predicted
$1.8511 from measured tokens, billed $1.86. Every abstraction here must
keep reproducing that number.
"""

import pytest

from largeliterarymodels import costs


class TestKnownAnswer:
    def test_the_invoice_gate(self):
        """517,547 fresh + 18,389,760 cached + 657,056 output on
        gpt-4o-mini must price at $1.8511 — validated against a real
        $1.86 bill (0.05% off)."""
        est = costs.price("gpt-4o-mini", fresh=517_547, cached=18_389_760,
                          output=657_056)
        assert abs(est["usd"] - 1.8511) < 0.0005, est["usd"]

    def test_the_147_pound_run(self):
        """74M in + 14M out on gemini-3.6-flash: $216 list — the number
        malign-logits computed by hand from the vendor page."""
        est = costs.price("gemini-3.6-flash", fresh=74_000_000,
                          output=14_000_000)
        assert est["usd"] == pytest.approx(216.0)


class TestCachedNullSemantics:
    def test_no_cache_tier_bills_full_input_rate(self):
        """cached: null means NO cache tier — not zero, not a discount.
        The warning must name the volume, because this is the largest
        cost driver on high-cache workloads and the easiest to miss."""
        p = costs._load()
        null_cache = [(prov, name) for prov in costs._PROVIDERS
                      for name, r in p.get(prov, {}).items()
                      if r["cached"] is None]
        assert null_cache, "table no longer has a null-cache model to pin"
        prov, name = null_cache[0]
        rate = p[prov][name]["input"]
        est = costs.price(name, cached=1_000_000)
        assert est["usd"] == pytest.approx(rate, rel=1e-6)
        assert any("NO cache tier" in w for w in est["warnings"])


class TestBatchDiscounts:
    def test_anthropic_halves(self):
        full = costs.price("claude-sonnet-4-6", fresh=1_000_000)
        batch = costs.price("claude-sonnet-4-6", fresh=1_000_000, batch=True)
        assert batch["usd"] == pytest.approx(full["usd"] / 2)

    def test_deepseek_has_no_batch_and_says_so(self):
        """Verified absent from their pricing page, not assumed — a batch
        'discount' that does not exist must price at list and warn."""
        full = costs.price("deepseek-v4-pro", fresh=1_000_000)
        batch = costs.price("deepseek-v4-pro", fresh=1_000_000, batch=True)
        assert batch["usd"] == full["usd"]
        assert any("no batch" in w.lower() for w in batch["warnings"])

    def test_google_halves(self):
        full = costs.price("gemini-3.6-flash", fresh=1_000_000)
        batch = costs.price("gemini-3.6-flash", fresh=1_000_000, batch=True)
        assert batch["usd"] == pytest.approx(full["usd"] / 2)


class TestDatedPricing:
    """Sonnet 5's introductory rate ends 2026-08-31; pricing an August
    run at September rates (or vice versa) is a silent 44% error."""

    def test_intro_rate_before_the_boundary(self):
        est = costs.price("claude-sonnet-5", fresh=1_000_000, on="2026-08-15")
        assert est["usd"] == pytest.approx(2.0)

    def test_post_rate_after_the_boundary(self):
        est = costs.price("claude-sonnet-5", fresh=1_000_000, on="2026-09-15")
        assert est["usd"] == pytest.approx(3.0)
        assert est["model"].endswith("post-2026-09-01")

    def test_expiry_warned_inside_thirty_days(self):
        est = costs.price("claude-sonnet-5", fresh=1_000_000, on="2026-08-15")
        assert any("2026-09-01" in w for w in est["warnings"])

    def test_no_expiry_noise_after_the_boundary(self):
        est = costs.price("claude-sonnet-5", fresh=1_000_000, on="2026-10-15")
        assert not any("changes on" in w for w in est["warnings"])


class TestReasoningFloor:
    def test_derived_from_measured_behaviour_not_the_table(self):
        """The table marks claude-fable-5 reasoning=false (it is not a
        reasoning-tier PRODUCT); measured behaviour says its thinking
        cannot be disabled and bills as output. The floor warning must
        follow the measurement."""
        p = costs._load()
        assert p["anthropic"]["claude-fable-5"]["reasoning"] is False
        assert costs.thinking_unavoidable("claude-fable-5")
        est = costs.price("claude-fable-5", fresh=1000, output=1000)
        assert any("FLOOR" in w for w in est["warnings"])

    def test_gemini_pros_are_floors_and_flashes_are_not(self):
        assert costs.thinking_unavoidable("gemini-2.5-pro")
        assert costs.thinking_unavoidable("gemini-3.1-pro-preview")
        assert not costs.thinking_unavoidable("gemini-3.6-flash")
        assert not costs.thinking_unavoidable("claude-sonnet-5")


class TestResolution:
    def test_aliases_and_prefixes(self):
        assert costs.resolve("sonnet")[1].startswith("claude-sonnet")
        assert costs.resolve("anthropic/claude-sonnet-4-6")[1] == \
            "claude-sonnet-4-6"
        assert costs.resolve("deepseek/deepseek-v4-pro")[0] == "deepseek"

    def test_dated_snapshot_falls_back_to_table_prefix(self):
        """Providers serve back dated ids (response_model); those must
        price as their family, longest prefix winning."""
        assert costs.resolve("claude-sonnet-4-6-20260219")[1] == \
            "claude-sonnet-4-6"

    def test_unknown_raises_with_the_fetch_date(self):
        with pytest.raises(ValueError, match="model_pricing.json"):
            costs.resolve("claude-nonexistent-99")

    def test_local_is_zero_with_a_note(self):
        est = costs.price("lmstudio/qwen3.5-35b-a3b", fresh=10_000_000,
                          output=1_000_000)
        assert est["usd"] == 0.0
        assert any("electricity" in w for w in est["warnings"])


class TestPriceReport:
    """The UsageTracker integration: report() keys map straight in, and
    the run's receipts (reasoning share, dropped params, served models)
    arrive beside the dollar figure."""

    def _report(self, **over):
        base = {"calls": 12, "input_tokens": 5_000,
                "cache_read_tokens": 48_041, "cache_write_tokens": 6_863,
                "output_tokens": 4_000, "reasoning_tokens": 0,
                "dropped_params": {}, "response_models": {"m": 12}}
        base.update(over)
        return base

    def test_maps_tracker_keys(self):
        est = costs.price_report("claude-sonnet-4-6", self._report())
        assert set(est["lines"]) == {"fresh_input", "cached_reads",
                                     "cache_writes_5m", "output"}

    def test_reasoning_share_warned(self):
        est = costs.price_report(
            "deepseek-v4-pro",
            self._report(reasoning_tokens=3_600, output_tokens=4_000))
        assert any("90%" in w for w in est["warnings"])

    def test_dropped_params_ride_along(self):
        est = costs.price_report(
            "claude-sonnet-5",
            self._report(dropped_params={"temperature": 12}))
        assert any("temperature" in w for w in est["warnings"])

    def test_split_serving_warned(self):
        est = costs.price_report(
            "deepseek-v4-pro",
            self._report(response_models={"a": 6, "b": 6}))
        assert any("2 model ids" in w for w in est["warnings"])


class TestBackCompat:
    """The previous module's surface, kept working: cli dry_run and
    batch_summary_task import these."""

    def test_estimate_shape(self):
        e = costs.estimate("sonnet", input_tokens=15_000, output_tokens=200,
                           n_calls=100, cached_tokens=13_000)
        for k in ("model", "total", "without_cache", "cache_savings",
                  "cache_write", "cache_hits", "uncached_input", "output"):
            assert k in e
        assert e["total"] < e["without_cache"], \
            "caching must save money on a 100-call batch"

    def test_estimate_arithmetic_anchored(self):
        """One call, no cache: n * (in*rate_in + out*rate_out)."""
        e = costs.estimate("claude-sonnet-4-6", input_tokens=1_000_000,
                           output_tokens=0, n_calls=1)
        p = costs._load()
        assert e["total"] == pytest.approx(
            p["anthropic"]["claude-sonnet-4-6"]["input"])

    def test_print_estimate_runs(self, capsys):
        costs.print_estimate(10_000, 200, n_calls=10, cached_tokens=8_000)
        out = capsys.readouterr().out
        assert "prices fetched" in out


class TestCompare:
    def test_sorted_and_no_dated_variants(self):
        rows = costs.compare(fresh=1_000_000, output=100_000)
        usds = [r["usd"] for r in rows]
        assert usds == sorted(usds)
        assert not any("-post-" in r["model"] for r in rows)
        assert len(rows) > 50, "expected the whole table"


class TestCacheAwarePricing:
    """Floors come from providers.cache_minimum_tokens — one source, the
    same constants the call path warns from. Prospective estimates that
    assume caching must know when the assumption is false: below the
    floor, providers decline SILENTLY and every call bills fresh."""

    def test_below_floor_rebills_at_full_rate(self):
        """Haiku 4.5's floor is 4,096. A 3,000-token prefix will not
        cache: the 'cached' reads must price as fresh input, and the
        warning must state the padding economics."""
        p = costs._load()
        rate = p["anthropic"]["claude-haiku-4-5"]["input"]
        est = costs.price("claude-haiku-4-5", cached=10_000_000,
                          prefix_tokens=3_000)
        assert est["usd"] == pytest.approx(10 * rate)
        assert any("BELOW" in w and "PADDING" in w for w in est["warnings"])

    def test_above_floor_prices_the_discount(self):
        p = costs._load()
        cached_rate = p["anthropic"]["claude-haiku-4-5"]["cached"]
        est = costs.price("claude-haiku-4-5", cached=10_000_000,
                          prefix_tokens=5_000)
        assert est["usd"] == pytest.approx(10 * cached_rate)
        assert not any("BELOW" in w for w in est["warnings"])

    def test_the_gemini_field_case(self):
        """The exact shape from the field report: a 3,906-token instrument
        on gemini-3.6-flash, ~130 tokens under the 4,096 floor. A
        cache-aware estimate refuses to promise the discount."""
        est = costs.price("gemini-3.6-flash", cached=57_000_000,
                          prefix_tokens=3_906)
        assert any("4,096" in w for w in est["warnings"])
        p = costs._load()
        assert est["usd"] == pytest.approx(
            57 * p["google"]["gemini-3.6-flash"]["input"])

    def test_unknown_floor_is_flagged_not_assumed(self):
        """OpenAI floors are unmeasured here: the estimate keeps the
        discount but says the assumption out loud."""
        est = costs.price("gpt-4o-mini", cached=1_000_000,
                          prefix_tokens=3_000)
        assert any("no measured cache floor" in w for w in est["warnings"])

    def test_old_api_gets_floor_awareness_for_free(self):
        """estimate()'s cached_tokens IS the prefix — a dry_run against a
        short instrument on Haiku now warns instead of promising."""
        e = costs.estimate("claude-haiku-4-5", input_tokens=4_000,
                           output_tokens=200, n_calls=100,
                           cached_tokens=3_000)
        assert any("BELOW" in w for w in e["warnings"])
        assert e["cache_savings"] == pytest.approx(0.0), \
            "no savings can be claimed below the floor"

    def test_retrospective_pricing_needs_no_floor(self):
        """price_report prices MEASURED reads; if the run cached, it
        cached, whatever the floor table thinks."""
        est = costs.price_report("claude-haiku-4-5", {
            "input_tokens": 1000, "cache_read_tokens": 50_000,
            "cache_write_tokens": 5_000, "output_tokens": 2_000,
        })
        assert not any("BELOW" in w for w in est["warnings"])


class TestReviewFindings:
    """Each test pins a defect from the Opus review of this branch."""

    def test_floor_gate_survives_an_alias(self):
        """S1 — the flagship feature was inert on every alias: the floor
        lookup got the raw string, cache_floor('haiku') is None, and the
        exact names print_estimate/dry_run default to under-estimated 2x
        in the cheaper-than-reality direction."""
        by_alias = costs.price("haiku", cached=10_000_000,
                               prefix_tokens=3_000)
        by_name = costs.price("claude-haiku-4-5", cached=10_000_000,
                              prefix_tokens=3_000)
        assert by_alias["usd"] == by_name["usd"]
        assert any("BELOW" in w for w in by_alias["warnings"])

    def test_pricing_aliases_agree_with_model_tags(self):
        """S2 — `litmod price --model sonnet` must price the model
        `litmod run --model sonnet` actually bills. The adopted table said
        sonnet->claude-sonnet-5; MODEL_TAGS says claude-sonnet-4-6 — a 33%
        under-estimate on the package's own default model."""
        from largeliterarymodels.cli.models import MODEL_TAGS
        p = costs._load()
        for tag, target in MODEL_TAGS.items():
            if tag in p["aliases"] and tag != "_note":
                priced = costs.resolve(tag)[1]
                ran = costs.resolve(target)[1]
                assert priced == ran, (tag, priced, ran)

    def test_unknown_variant_does_not_silently_prefix_match(self):
        """S5 — variant suffixes are different PRODUCTS: gpt-5.6 priced as
        gpt-5, a 'mini' at its parent's rate, a flash-lite at 5x. Snapshot
        suffixes are the same product's checkpoints and must still work."""
        for wrong in ("gpt-5.6", "gpt-5.5-mini-x", "gemini-3.6-flash-lite",
                      "o3-deep-research-9"):
            with pytest.raises(ValueError):
                costs.resolve(wrong)
        assert costs.resolve("claude-sonnet-4-6-20260219")[1] == \
            "claude-sonnet-4-6"
        assert costs.resolve("gemini-2.5-flash-002")[1] == "gemini-2.5-flash"

    def test_batch_lines_sum_to_usd(self):
        """S7 — components at list price under a discounted headline:
        $18.00 of lines under a $9.00 total."""
        est = costs.price("claude-sonnet-4-6", fresh=5_000_000,
                          output=1_000_000, batch=True)
        assert sum(est["lines"].values()) == pytest.approx(est["usd"],
                                                           abs=1e-4)

    def test_no_contradictory_floor_advisory_on_null_cache(self):
        """S11 — 'has NO cache tier' and 'assumes the prefix caches' are
        not both sayable about one model in one breath."""
        p = costs._load()
        null_model = next(name for prov in costs._PROVIDERS
                          for name, r in p.get(prov, {}).items()
                          if r["cached"] is None)
        est = costs.price(null_model, cached=1_000_000, prefix_tokens=100)
        assert any("NO cache tier" in w for w in est["warnings"])
        assert not any("no measured cache floor" in w
                       for w in est["warnings"])

    def test_claude_cli_models_price(self):
        """S12 — MajorGenreTask's default model string must not make
        print_report raise."""
        assert costs.resolve("claude-cli/sonnet")[1].startswith(
            "claude-sonnet")

    def test_compare_applies_the_floor_gate(self):
        """S6 — table mode is where non-monotonic floors reorder the
        ranking; the flag was accepted and silently discarded there."""
        rows = costs.compare(cached=10_000_000, prefix_tokens=2_000)
        by_name = {r["model"]: r for r in rows}
        assert any("BELOW" in w
                   for w in by_name["claude-haiku-4-5"]["warnings"])
        assert not any("BELOW" in w
                       for w in by_name["claude-opus-5"]["warnings"]), \
            "opus-5's floor is 512; a 2,000-token prefix caches there"

    def test_price_run_shim_agrees_with_costs(self, capsys):
        """S8 — the standalone reimplementation had already diverged
        (no dated rows, no floor gate, table-flag reasoning). Now it IS
        the module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "price_run", "scripts/price_run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.selftest() == 0


class TestProvidersCoupling:
    def test_lazy_imports_exist(self):
        """costs.py imports five names from providers lazily — a rename
        there would otherwise break pricing at call time, in the field,
        instead of here."""
        from largeliterarymodels.providers import (  # noqa: F401
            cache_minimum_tokens, _THINKING_CANNOT_DISABLE,
            _GOOGLE_THINKING_CANNOT_DISABLE, _family_match, _strip_prefix)
