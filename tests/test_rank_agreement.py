"""Rank agreement: Kendall's W and pairwise rank correlation.

These exist because the categorical agreement functions in the same module
answer a different question and do not say so. Applied to rankings,
pairwise_agreement scores rank 1 against rank 2 as exactly the same
disagreement as rank 1 against rank 13 — it returns a plausible number for a
question nobody asked, which is worse than an error.

Most tests below name the specific wrong number they exist to prevent. Where a
value is pinned it was computed by hand first and only then checked against the
code, so the test is not a transcription of whatever the implementation
happened to return.
"""

import itertools
import math
import random
import statistics

import pandas as pd
import pytest
from pydantic import BaseModel
from typing import Literal

from largeliterarymodels.analysis.reliability import (
    SUMMARY_COLUMNS,
    _as_tie_groups,
    _midranks,
    _restrict_and_rerank,
    kendall_w,
    pairwise_agreement,
    pairwise_rank_correlation,
    rank_agreement_summary,
)


class TestMidranks:
    def test_tie_group_gets_the_average_position(self):
        assert _midranks(_as_tie_groups(["a", ["b", "c"], "d"])) == {
            "a": 1.0, "b": 2.5, "c": 2.5, "d": 4.0}

    def test_mapping_form_with_equal_ranks_is_a_tie(self):
        assert _midranks(_as_tie_groups({"a": 1, "b": 2, "c": 2, "d": 3})) == {
            "a": 1.0, "b": 2.5, "c": 2.5, "d": 4.0}

    def test_both_input_forms_agree(self):
        a = _midranks(_as_tie_groups(["x", ["y", "z"], "q"]))
        b = _midranks(_as_tie_groups({"x": 1, "y": 2, "z": 2, "q": 3}))
        assert a == b


class TestRestrictAndRerank:
    def test_empty_and_single_survivor_do_not_need_special_casing(self):
        """N-15. Both are reachable from a real intersection and both are the
        kind of edge that gets 'fixed' into a crash by a later refactor."""
        groups = _as_tie_groups(["a", ["b", "c"], "d"])
        assert _restrict_and_rerank(groups, set()) == ({}, [])
        assert _restrict_and_rerank(groups, {"b"}) == ({"b": 1.0}, [["b"]])
        # A survivor from the middle of a tie group is rank 1, not 2.5.
        assert _restrict_and_rerank(groups, {"c"}) == ({"c": 1.0}, [["c"]])


class TestKendallW:
    def test_perfect_agreement_is_exactly_one(self):
        r = {f"c{i}": ["a", "b", "c", "d", "e"] for i in range(3)}
        assert kendall_w(r)["w"] == 1.0

    def test_reversed_pair_is_exactly_zero(self):
        r = {"a": ["a", "b", "c", "d", "e"], "b": ["e", "d", "c", "b", "a"]}
        assert kendall_w(r)["w"] == 0.0

    def test_w_matches_the_hand_computed_value_with_ties(self):
        """N-2. r1 = a < (b=c) < d, r2 = (a=b) < c < d.

        Midranks: r1 a=1, b=c=2.5, d=4; r2 a=b=1.5, c=3, d=4.
        Rank sums 2.5, 4.0, 5.5, 8.0 about a mean of m(n+1)/2 = 5, so
        S = 6.25 + 1 + 0.25 + 9 = 16.5. Two tie groups of 2 give a correction
        of 2*(2^3-2) = 12, so the denominator is 4*(4^3-4) - 2*12 = 216 and
        W = 12*16.5/216 = 198/216. Dropping the tie correction gives 198/240 =
        0.825 — the number this test exists to reject.
        """
        r = {"r1": ["a", ["b", "c"], "d"], "r2": [["a", "b"], "c", "d"]}
        res = kendall_w(r)
        assert res["w"] == pytest.approx(198 / 216)
        assert res["w"] != pytest.approx(0.825)
        assert res["ties_present"] is True

    def test_w_matches_the_hand_computed_value_without_ties(self):
        """N-3. Three coders, no ties: abcd / abdc / bacd.

        Rank sums a=4, b=5, c=10, d=11 about a mean of 3*5/2 = 7.5, so
        S = 12.25 + 6.25 + 6.25 + 12.25 = 37 and W = 12*37/(9*60) = 444/540.
        """
        r = {"r1": ["a", "b", "c", "d"],
             "r2": ["a", "b", "d", "c"],
             "r3": ["b", "a", "c", "d"]}
        res = kendall_w(r)
        assert res["w"] == pytest.approx(444 / 540)
        assert res["ties_present"] is False

    def test_ties_do_not_break_perfect_agreement(self):
        # Without the tie correction in the denominator, identical rankings
        # containing a tie score below 1.
        r = {"a": ["x", ["y", "z"], "q"], "b": ["x", ["y", "z"], "q"]}
        res = kendall_w(r, min_items=4)
        assert res["w"] == pytest.approx(1.0)
        assert res["ties_present"] is True

    def test_tie_correction_uses_the_restricted_groups(self):
        """N-1. The intersection drops z, breaking a's (q, z) tie.

        A tie that no longer exists in the ranking being scored must not be
        corrected for: correcting it shrinks the denominator from 240 to 228
        and returns W = 1.0526 for two coders who agree perfectly. This is the
        one tie-correction mutant the rest of the suite does not kill, because
        every other tied case has the tie surviving the intersection.
        """
        r = {"a": ["p", ["q", "z"], "s", "t"], "b": ["p", "q", "s", "t"]}
        res = kendall_w(r, min_items=4)
        assert res["w"] == 1.0
        assert res["ties_present"] is False

    def test_ranks_are_recomputed_after_intersecting(self):
        """A coder's 2nd, 4th, 6th and 8th picks are ranks 1-4 over the
        survivors.

        Filtering without re-ranking inflates S and reports disagreement that
        is an artefact of the other coder's coverage: b's global positions
        2, 4, 6, 8 against a's 1, 2, 3, 4 give S = 70 and W = 3.5.
        """
        r = {"a": ["p", "q", "r", "s"],
             "b": ["z1", "p", "z2", "q", "z3", "r", "z4", "s"]}
        assert kendall_w(r)["w"] == pytest.approx(1.0)


class TestMalformedRankings:
    """Every case here used to return a number instead of raising."""

    def test_a_repeated_item_raises_instead_of_pushing_w_above_one(self):
        """N-4 / FIND-1. _midranks overwrote on the repeat, so a's rank vector
        summed to less than n(n+1)/2 and W came back as 1.1."""
        r = {"a": ["p", "q", "p", "r", "s"], "b": ["p", "q", "r", "s", "t"]}
        with pytest.raises(ValueError, match="'a'.*'p'.*more than once"):
            kendall_w(r, min_items=2)

    def test_repeats_are_caught_inside_a_tie_group_too(self):
        with pytest.raises(ValueError, match="more than once"):
            kendall_w({"a": ["p", ["q", "q"], "r"], "b": ["p", "q", "r"]},
                      min_items=2)

    def test_dropped_per_coder_is_this_coder_s_items_minus_the_intersection(self):
        """FIND-1's second half. `dropped_per_coder` counted tokens, so a coder
        that repeated an item was reported as having dropped one.

        Honest note on what this test can and cannot catch: with duplicates now
        rejected outright, tokens and distinct items coincide on every input
        that reaches this line, so the token-counting mutant is unreachable
        rather than killed. What is pinned here is the definition — per-coder,
        not per-corpus. `len(union) - n` would report a=1 for a coder that
        dropped nothing.
        """
        r = {"a": ["p", "q", "r", "s"], "b": ["p", "q", "r", "s", "t"]}
        assert kendall_w(r)["dropped_per_coder"] == {"a": 0, "b": 1}

    def test_nan_rank_raises_the_same_way_under_either_insertion_order(self):
        """N-6 / FIND-2. NaN != NaN, so it formed its own tie group whose
        sorted position depended on which key was inserted first: W came back
        as 1.0 one way round and 0.75 the other."""
        first = {"a": {"p": 1, "q": float("nan"), "r": 3},
                 "b": {"p": 1, "q": 2, "r": 3}}
        second = {"a": {"q": float("nan"), "p": 1, "r": 3},
                  "b": {"p": 1, "q": 2, "r": 3}}
        messages = []
        for r in (first, second):
            with pytest.raises(ValueError) as excinfo:
                kendall_w(r, min_items=2)
            messages.append(str(excinfo.value))
        assert messages[0] == messages[1]
        assert "'a'" in messages[0] and "'q'" in messages[0]
        assert "NaN" in messages[0]

    def test_none_rank_raises_a_value_error_naming_the_coder(self):
        """FIND-2. Previously a bare TypeError out of float()."""
        with pytest.raises(ValueError, match="'a'.*'q'.*None"):
            kendall_w({"a": {"p": 1, "q": None, "r": 3},
                       "b": {"p": 1, "q": 2, "r": 3}}, min_items=2)

    def test_a_bare_string_is_rejected_rather_than_ranked_by_character(self):
        """FIND-10. "abcd" iterated into four single-character items and
        returned W = 1.0 against another string."""
        with pytest.raises(ValueError, match="bare string"):
            kendall_w({"a": "abcd", "b": "abcd"})

    def test_non_numeric_rank_names_the_coder(self):
        """FIND-10. Previously `could not convert string to float: 'high'`,
        with nothing saying whose ranking it came from."""
        with pytest.raises(ValueError, match="'a'.*'p'.*'high'"):
            kendall_w({"a": {"p": "high", "q": 2, "r": 3, "s": 4},
                       "b": {"p": 1, "q": 2, "r": 3, "s": 4}})


class TestZeroVarianceCoder:
    def test_a_coder_who_ties_everything_is_named_in_the_notes(self):
        """N-9 / FIND-4. b contributes no ordering at all, yet still counts
        toward m and pulls W to exactly 0.5 — a number that reads as
        'moderate agreement' rather than 'one of your two coders abstained'."""
        r = {"a": ["p", "q", "r", "s"], "b": [["p", "q", "r", "s"]]}
        res = kendall_w(r)
        assert res["w"] == pytest.approx(0.5)
        assert "zero-variance" in res["note"]
        assert "'b'" in res["note"]

    def test_undefined_pairs_are_excluded_from_mean_spearman_not_scored_zero(self):
        """FIND-3's second half: the identity reported mean_spearman = 0.0
        here, which asserts independence where the correlation is undefined."""
        r = {"a": ["p", "q", "r", "s"], "b": [["p", "q", "r", "s"]]}
        res = kendall_w(r)
        assert res["mean_spearman"] is None
        assert "undefined pair" in res["note"]


class TestMeanSpearman:
    def test_mean_spearman_is_computed_not_inferred_from_the_identity(self):
        """N-7 / FIND-3. r1 ties its first five items, r2 ties nothing.

        W = 0.8, so the identity W = (1 + (m-1)r̄)/m gives r̄ = 0.6. The actual
        Spearman on the midranks [3,3,3,3,3,6] vs [1,2,3,4,5,6] is
        7.5/sqrt(7.5*17.5) = 0.6547. The identity assumes an identical tie
        structure across coders and there isn't one.
        """
        r = {"r1": [["a", "b", "c", "d", "e"], "f"],
             "r2": ["a", "b", "c", "d", "e", "f"]}
        res = kendall_w(r)
        assert res["w"] == pytest.approx(0.8)
        assert res["mean_spearman"] == pytest.approx(
            7.5 / math.sqrt(7.5 * 17.5))
        assert res["mean_spearman"] == pytest.approx(0.6546536707, abs=1e-9)
        assert res["mean_spearman"] != pytest.approx(0.6, abs=1e-3)

    def test_agrees_with_pairwise_when_every_coder_ranked_everything(self):
        """The two are computed over different item sets by design; when the
        sets coincide they must still land on the same number."""
        r = {"j1": {"A": 1, "B": 6, "C": 3, "D": 2, "E": 5, "F": 4},
             "j2": {"A": 2, "B": 5, "C": 4, "D": 1, "E": 6, "F": 3},
             "j3": {"A": 6, "B": 2, "C": 5, "D": 4, "E": 1, "F": 3}}
        res = kendall_w(r)
        rho = statistics.mean(
            pairwise_rank_correlation(r, method="spearman")["coefficient"])
        assert res["mean_spearman"] == pytest.approx(rho, abs=1e-9)

    def test_differs_from_pairwise_when_coverage_differs_and_says_so(self):
        """N-8 / FIND-3. Three coders over 12 items; c ranked only w0-w5.

        Global intersection = {w0..w5}, on which a = 1..6, b = 6..1 and
        c = 2,4,1,6,3,5. Pairwise there: rho(a,b) = -1, rho(a,c) = 0.4857,
        rho(b,c) = -0.4857, so mean_spearman = -1/3.

        pairwise_rank_correlation uses per-pair intersections, so (a,b) is
        scored over all 12 items: sum d^2 = 70 gives rho = 1 - 35/143 =
        108/143, and the mean over the three computable pairs is 36/143.

        Two defensible numbers a factor of five apart with opposite signs. The
        docstrings have to say which item set each one is on.
        """
        POOL = [f"w{i}" for i in range(12)]
        r = {"a": POOL,
             "b": POOL[:6][::-1] + POOL[6:],
             "c": ["w2", "w0", "w4", "w1", "w5", "w3"]}

        res = kendall_w(r, pool=12)
        assert res["n_items"] == 6
        assert res["coverage"] == pytest.approx(6 / 12)
        assert res["mean_spearman"] == pytest.approx(-1 / 3)

        df = pairwise_rank_correlation(r)
        assert df["computable"].all()
        assert df.set_index(["coder_a", "coder_b"]).loc[("a", "b"), "coefficient"] \
            == pytest.approx(108 / 143)
        assert df.loc[df["computable"], "coefficient"].mean() == \
            pytest.approx(36 / 143)

        assert res["mean_spearman"] != pytest.approx(36 / 143, abs=1e-3)
        assert "GLOBAL intersection" in kendall_w.__doc__
        assert "PER-PAIR" in kendall_w.__doc__
        assert "mean_spearman" in pairwise_rank_correlation.__doc__
        assert "different item sets" in pairwise_rank_correlation.__doc__


class TestCoverageReporting:
    """The number the caller asked to have returned: W over 4 of 15 items is
    not comparable to W over 15, however high it is."""

    POOL = [f"w{i}" for i in range(15)]

    def test_reports_the_n_it_actually_used(self):
        r = {"a": self.POOL, "b": self.POOL[::-1],
             "c": ["w0", "w1", "w2", "w3"]}
        res = kendall_w(r, pool=15)
        assert res["n_items"] == 4
        assert res["coverage"] == pytest.approx(4 / 15)
        assert res["dropped_per_coder"] == {"a": 11, "b": 11, "c": 0}

    def test_coverage_is_reported_even_when_no_pool_is_declared(self):
        """FIND-5. Coverage used to be None unless the caller opted in by
        passing pool, so the one guard against quoting a restricted W was off
        by default. Fall back to intersection / union."""
        r = {"a": self.POOL, "b": self.POOL[::-1],
             "c": ["w0", "w1", "w2", "w3"]}
        assert kendall_w(r)["coverage"] == pytest.approx(4 / 15)

    def test_pool_smaller_than_the_item_set_is_rejected(self):
        """N-11 / FIND-5. pool=2 over 4 items reported coverage 2.0."""
        r = {"a": ["p", "q", "r", "s"], "b": ["p", "q", "r", "s"]}
        with pytest.raises(ValueError, match="smaller than the 4 distinct"):
            kendall_w(r, pool=2)

    def test_pool_zero_is_rejected_rather_than_treated_as_absent(self):
        """N-11 / FIND-5. pool=0 fell through `if pool` to coverage=None."""
        r = {"a": ["p", "q", "r", "s"], "b": ["p", "q", "r", "s"]}
        with pytest.raises(ValueError, match="positive integer"):
            kendall_w(r, pool=0)
        with pytest.raises(ValueError, match="positive integer"):
            kendall_w(r, pool=-3)

    def test_refuses_to_return_a_quotable_number_below_min_items(self):
        r = {"a": self.POOL, "b": self.POOL, "c": ["w0", "w1", "w2"]}
        res = kendall_w(r, pool=15)
        assert res["w"] is None
        assert res["n_items"] == 3
        assert "below min_items" in res["note"]

    def test_every_failure_path_explains_itself(self):
        assert "at least 2 coders" in kendall_w({"a": ["x"]})["note"]
        assert "common to all" in kendall_w(
            {"a": ["x", "y"], "b": ["p", "q"]})["note"]
        # every item tied for every coder: no variation to measure
        allties = {"a": [["x", "y", "z", "q"]], "b": [["x", "y", "z", "q"]]}
        assert "no rank variation" in kendall_w(allties, min_items=2)["note"]

    def test_notes_accumulate_instead_of_overwriting_each_other(self):
        """FIND-9. `note` was a single slot, so the chi-square note clobbered
        whatever diagnosis had been written before it."""
        r = {"a": ["p", "q", "r", "s"], "b": [["p", "q", "r", "s"]]}
        res = kendall_w(r)
        assert len(res["notes"]) >= 3
        assert res["note"] == "; ".join(res["notes"])
        assert any("zero-variance" in note for note in res["notes"])
        assert any("chi-square" in note for note in res["notes"])

    def test_note_is_none_when_there_is_nothing_to_say(self):
        r = {f"c{i}": self.POOL for i in range(3)}
        res = kendall_w(r)
        assert res["notes"] == []
        assert res["note"] is None

    def test_small_n_p_values_are_flagged_as_approximate(self):
        res = kendall_w({"a": ["a", "b", "c", "d"], "b": ["a", "b", "c", "d"]})
        assert res["p_approximate"] is True
        assert "chi-square approximation" in res["note"]
        big = {f"c{i}": self.POOL for i in range(3)}
        assert kendall_w(big)["p_approximate"] is False

    def test_two_coders_are_flagged_at_any_n_and_df_is_n_minus_one(self):
        """N-14 / FIND-8. p_approximate keyed on n alone, so a 15-item
        two-coder run advertised an exact p. With m = 2, W is a rescaled
        Spearman and m(n-1)W is not close to chi-square at any n."""
        two = kendall_w({"a": self.POOL, "b": self.POOL})
        assert two["n_items"] == 15
        assert two["p_approximate"] is True
        assert "2 coders" in two["note"]
        assert two["df"] == two["n_items"] - 1 == 14
        assert two["chi2"] == pytest.approx(2 * 14 * two["w"])

        three = kendall_w({f"c{i}": self.POOL for i in range(3)})
        assert three["p_approximate"] is False
        assert three["df"] == 14


class TestPairwiseRankCorrelation:
    def test_each_pair_uses_its_own_intersection(self):
        # One low-coverage coder must not shrink n for the other pair.
        pool = [f"w{i}" for i in range(10)]
        r = {"a": pool, "b": pool, "c": pool[:3]}
        df = pairwise_rank_correlation(r).set_index(["coder_a", "coder_b"])
        assert df.loc[("a", "b"), "n"] == 10
        assert df.loc[("a", "c"), "n"] == 3

    def test_kendall_and_spearman_return_different_coefficients(self):
        """T-a. Identical rankings score 1.0 under either method, so the old
        version of this test could not tell tau-b from a silent fall-through
        to Spearman. Swapping all four adjacent pairs of 8 items separates
        them: sum d^2 = 8 gives rho = 1 - 48/504 = 0.9048, while 4 discordant
        pairs out of 28 give tau-b = 20/28 = 0.7143."""
        pool = [f"w{i}" for i in range(8)]
        swapped = ["w1", "w0", "w3", "w2", "w5", "w4", "w7", "w6"]
        r = {"a": pool, "b": swapped}
        rho = pairwise_rank_correlation(r, method="spearman").loc[0, "coefficient"]
        tau = pairwise_rank_correlation(r, method="kendall").loc[0, "coefficient"]
        assert rho == pytest.approx(1 - 48 / 504)
        assert rho == pytest.approx(0.9047619, abs=1e-6)
        assert tau == pytest.approx(20 / 28)
        assert tau == pytest.approx(0.7142857, abs=1e-6)
        assert rho != pytest.approx(tau, abs=1e-3)

    def test_kendall_is_tau_b_not_tau_c_on_tied_data(self):
        """N-13. x = a < (b=c) < d against y = a < b < c < d. Five concordant
        pairs, none discordant, one tied pair in x: tau-b = 5/sqrt(5*6) =
        0.9129. tau-c on the same data is 2*5/(16*2/3) = 0.9375."""
        r = {"x": ["a", ["b", "c"], "d"], "y": ["a", "b", "c", "d"]}
        tau = pairwise_rank_correlation(r, method="kendall").loc[0, "coefficient"]
        assert tau == pytest.approx(5 / math.sqrt(30))
        assert tau == pytest.approx(0.9128709, abs=1e-6)
        assert tau != pytest.approx(0.9375, abs=1e-4)

    def test_under_covered_pairs_are_flagged_not_just_left_as_nan(self):
        """N-12 / FIND-7. A NaN coefficient is silently skipped by pandas'
        .mean(), so a mean over three pairs and a mean over one look
        identical. `computable` makes the filtering deliberate."""
        pool = [f"w{i}" for i in range(10)]
        swapped = list(pool)
        swapped[0], swapped[9] = swapped[9], swapped[0]
        r = {"a": pool, "b": swapped, "c": ["w0", "w1"]}
        df = pairwise_rank_correlation(r).set_index(["coder_a", "coder_b"])

        assert bool(df.loc[("a", "c"), "computable"]) is False
        assert df.loc[("a", "c"), "n"] == 2
        assert pd.isna(df.loc[("a", "c"), "coefficient"])
        assert bool(df.loc[("a", "b"), "computable"]) is True

        computable = df[df["computable"]]
        assert len(computable) == 1
        assert computable["coefficient"].mean() == \
            pytest.approx(df.loc[("a", "b"), "coefficient"])
        assert computable["coefficient"].mean() < 1.0

    def test_min_items_threshold_is_a_parameter(self):
        """FIND-7. The n >= 3 cut was hardcoded, and a Spearman over 3 untied
        items can only take four values."""
        pool = [f"w{i}" for i in range(10)]
        r = {"a": pool, "b": pool, "c": pool[:3]}
        loose = pairwise_rank_correlation(r).set_index(["coder_a", "coder_b"])
        assert bool(loose.loc[("a", "c"), "computable"]) is True

        strict = pairwise_rank_correlation(r, min_items=5).set_index(
            ["coder_a", "coder_b"])
        assert bool(strict.loc[("a", "c"), "computable"]) is False
        assert strict.loc[("a", "c"), "n"] == 3
        assert "1/6" in pairwise_rank_correlation.__doc__

    def test_a_zero_variance_coder_is_not_computable(self):
        r = {"a": ["p", "q", "r", "s"], "b": [["p", "q", "r", "s"]]}
        df = pairwise_rank_correlation(r)
        assert bool(df.loc[0, "computable"]) is False
        assert df.loc[0, "n"] == 4

    def test_rejects_an_unknown_method(self):
        with pytest.raises(ValueError, match="spearman"):
            pairwise_rank_correlation({"a": ["x"], "b": ["x"]}, method="pearson")


class TestRandomizedTiedRankings:
    def test_w_stays_in_range_and_ranks_stay_normalised(self):
        """N-5. 200 seeded random tied rankings. Two invariants: W in [0, 1],
        and every coder's restricted midranks summing to n(n+1)/2 — the
        property a duplicated or overwritten item violates, which is how
        FIND-1's W = 1.1 got out. Seeded, so a failure is reproducible."""
        rng = random.Random(20260804)
        checked = 0
        for _ in range(200):
            n_pool = rng.randint(4, 9)
            pool = [f"i{k}" for k in range(n_pool)]
            rankings = {}
            for c in range(rng.randint(2, 4)):
                items = [x for x in pool if rng.random() > 0.15]
                if len(items) < 2:
                    items = list(pool)
                rng.shuffle(items)
                groups, idx = [], 0
                while idx < len(items):
                    size = rng.choice([1, 1, 1, 2, 3])
                    group = items[idx:idx + size]
                    groups.append(group if len(group) > 1 else group[0])
                    idx += size
                rankings[f"c{c}"] = groups

            res = kendall_w(rankings, min_items=2)
            if res["w"] is None:
                continue
            checked += 1
            assert -1e-12 <= res["w"] <= 1 + 1e-12, (rankings, res["w"])

            common = set(res["items"])
            n = len(common)
            expected = n * (n + 1) / 2
            for coder, ranking in rankings.items():
                ranks, _ = _restrict_and_rerank(
                    _as_tie_groups(ranking, coder=coder), common)
                assert len(ranks) == n
                assert sum(ranks.values()) == pytest.approx(expected), (
                    coder, ranking, ranks)
        assert checked > 100, f"only {checked} trials produced a W"


class TestCategoricalStatisticsAreTheWrongTool:
    """The reason this module section exists, asserted rather than described.

    The categorical side is exercised through the real `pairwise_agreement`,
    not a local reimplementation of it: the claim is about the function this
    package ships, and a hand-rolled exact-match loop can agree with the point
    being made while the shipped function does something else.
    """

    class _Position(BaseModel):
        position: Literal[tuple(str(i) for i in range(13))]  # type: ignore[valid-type]

    @staticmethod
    def _frames(coders: dict) -> dict:
        frames = {}
        for coder, order in coders.items():
            rows = [{"_id": word, "scheme": "rank", "seq": 0,
                     "position": str(i)} for i, word in enumerate(order)]
            frames[coder] = pd.DataFrame(rows).set_index(
                ["_id", "scheme", "seq"])
        return frames

    def test_high_concordance_reads_as_low_categorical_agreement(self):
        # Three coders agreeing strongly on ORDER but rarely on exact position.
        coders = {
            "flash": ["kill", "strike", "scream", "fight", "shout", "cry",
                      "plead", "weep", "ask", "run", "walk", "go", "wait"],
            "gpt": ["kill", "fight", "strike", "scream", "cry", "shout",
                    "weep", "plead", "run", "ask", "walk", "wait", "go"],
            "sonnet": ["kill", "strike", "fight", "scream", "shout", "weep",
                       "cry", "plead", "ask", "run", "go", "walk", "wait"],
        }
        w = kendall_w(coders, pool=13)["w"]

        agreement = pairwise_agreement(self._frames(coders), self._Position)
        pair_cols = [f"{a}={b}" for a, b in itertools.combinations(
            sorted(coders), 2)]
        assert set(pair_cols) <= set(agreement.columns)
        exact = statistics.mean(agreement.loc["position", c] for c in pair_cols)

        assert w > 0.95, "the coders plainly concord"
        assert exact < 0.35, "exact-match agreement says they barely do"
        # Same data, opposite conclusions: this is why kappa on ranks inverts
        # rather than merely blurs.
        assert w - exact > 0.6


class TestRankAgreementSummary:
    def test_uncomputable_items_keep_their_row(self):
        pool = [f"w{i}" for i in range(15)]
        summary = rank_agreement_summary(
            {"good": {"a": pool, "b": pool},
             "sparse": {"a": pool, "b": pool, "c": ["w0", "w1", "w2"]}},
            pools={"good": 15, "sparse": 15},
        )
        # A silently shorter table is how a coverage problem becomes invisible.
        assert list(summary.index) == ["good", "sparse"]
        assert summary.loc["good", "w"] == pytest.approx(1.0)
        assert summary.loc["sparse", "w"] != summary.loc["sparse", "w"]  # NaN
        assert "below min_items" in summary.loc["sparse", "note"]
        assert summary.loc["sparse", "coverage"] == pytest.approx(3 / 15)

    def test_empty_input_returns_an_empty_frame_with_the_documented_columns(self):
        """N-10 / FIND-6. `.set_index('item_id')` raised KeyError on an empty
        frame, so a corpus that happened to yield no rankable items crashed
        the caller instead of returning nothing."""
        summary = rank_agreement_summary({})
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
        assert list(summary.columns) == list(SUMMARY_COLUMNS)
        assert summary.index.name == "item_id"
        # Concatenating it with a populated summary must not add columns.
        pool = [f"w{i}" for i in range(15)]
        populated = rank_agreement_summary({"good": {"a": pool, "b": pool}})
        assert list(pd.concat([summary, populated]).columns) == \
            list(SUMMARY_COLUMNS)
