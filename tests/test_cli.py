"""Tests for the litmod CLI: arg parsing, dispatch, model tags, cloud parser."""

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from largeliterarymodels.cli import cloud as cloud_mod
from largeliterarymodels.cli.main import build_parser
from largeliterarymodels.cli.models import MODEL_TAGS, resolve_model
from largeliterarymodels.cli.output import compare_print, header_for, pretty_print


class TestModelTags:
    def test_known_tag(self):
        assert resolve_model("sonnet").startswith("claude-sonnet")

    def test_passthrough_full_id(self):
        assert resolve_model("lmstudio/foo") == "lmstudio/foo"
        assert resolve_model("claude-opus-4-7") == "claude-opus-4-7"

    def test_unknown_tag_exits(self):
        with pytest.raises(SystemExit):
            resolve_model("nonsense-tag")

    def test_all_tags_resolve_to_routable_models(self):
        from largeliterarymodels.providers import route_provider
        for full_id in MODEL_TAGS.values():
            route_provider(full_id)  # raises ValueError if unroutable


class TestMainParser:
    def test_run_flags(self):
        p = build_parser()
        a = p.parse_args(["run", "SomeTask", "--input", "m.csv",
                          "--model", "sonnet", "--no-shuffle"])
        assert a.no_shuffle is True
        assert a.shuffle_seed == 42

    def test_batch_choices_come_from_shared_map(self):
        p = build_parser()
        a = p.parse_args(["batch", "plot_genre", "-i", "dir"])
        assert a.task == "plot_genre"
        with pytest.raises(SystemExit):
            p.parse_args(["batch", "not_a_task", "-i", "dir"])

    def test_smoke_requires_model(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["smoke", "SomeTask"])


class TestCloudParser:
    """--yes must work in every position; ssh must pass flags through."""

    def _parse(self, argv):
        captured = {}

        def cap(args):
            captured["args"] = args

        cmd = next(a for a in argv if not a.startswith("-"))
        with patch.object(cloud_mod, f"cmd_{cmd}", cap):
            cloud_mod.main(argv)
        return captured["args"]

    def test_yes_after_subcommand(self):
        assert self._parse(["stop", "--yes"]).yes is True
        assert self._parse(["stop", "-y"]).yes is True

    def test_yes_before_subcommand(self):
        assert self._parse(["--yes", "stop"]).yes is True

    def test_yes_defaults_false(self):
        assert self._parse(["stop"]).yes is False

    def test_ssh_remainder_captures_flags(self):
        args = self._parse(["ssh", "ls", "-la", "/workspace"])
        assert args.ssh_command == ["ls", "-la", "/workspace"]

    def test_run_task_choices(self):
        args = self._parse(["run", "mydir", "--task", "passage_narrativity"])
        assert args.task == "passage_narrativity"

    def test_summary_task_map_shared_with_batch(self):
        from largeliterarymodels.cli.main import SUMMARY_TASK_MAP
        assert SUMMARY_TASK_MAP is cloud_mod.SUMMARY_TASK_MAP


class TestCloudStateHandling:
    def test_corrupt_state_tolerated(self, tmp_path, capsys):
        bad = tmp_path / ".vastai.json"
        bad.write_text("{not json")
        with patch.object(cloud_mod, "STATE_FILE", bad):
            assert cloud_mod.load_state() == {}
        assert "corrupt" in capsys.readouterr().err

    def test_save_state_atomic_write(self, tmp_path):
        target = tmp_path / ".vastai.json"
        with patch.object(cloud_mod, "STATE_FILE", target):
            cloud_mod.save_state({"instance_id": "123"})
            assert json.loads(target.read_text()) == {"instance_id": "123"}
            assert not (tmp_path / ".vastai.json.tmp").exists()

    def test_session_name_sanitized(self):
        assert cloud_mod._session_name("my dir/x.y") == "batch_my_dir_x_y"

    def test_coerce_price(self):
        assert cloud_mod._coerce_price("1.5") == 1.5
        assert cloud_mod._coerce_price("?") is None
        assert cloud_mod._coerce_price(None) is None


class _Result(BaseModel):
    tags: list[str] = []
    nested: list[dict] = []
    score: float = 0.0
    notes: str = ""


class TestOutput:
    def test_pretty_print_smoke(self, capsys):
        pretty_print(_Result(tags=["a"], score=1.5), "hdr")
        out = capsys.readouterr().out
        assert "hdr" in out and "tags" in out

    def test_compare_print_marks_disagreement(self, capsys):
        r1 = _Result(tags=["a", "b"], score=1.0)
        r2 = _Result(tags=["b", "a"], score=2.0)  # same set, different score
        compare_print({"m1": r1, "m2": r2}, "hdr")
        lines = capsys.readouterr().out.splitlines()
        tags_line = next(ln for ln in lines if "tags" in ln)
        score_line = next(ln for ln in lines if "score" in ln)
        assert not tags_line.startswith("*"), "order-insensitive list equality"
        assert score_line.startswith("*")

    def test_compare_print_unhashable_fields(self, capsys):
        # regression: list-of-dict fields used to crash frozenset()
        r1 = _Result(nested=[{"k": 1}])
        r2 = _Result(nested=[{"k": 2}])
        compare_print({"m1": r1, "m2": r2}, "hdr")
        assert "nested" in capsys.readouterr().out

    def test_header_for(self):
        assert "x" in header_for({"_id": "x", "seq": 3})
        assert header_for({}) == "{}"


class TestRenderShell:
    """The render shell previously had no tests, through which an inverted
    default (provenance footer ON) and an unreachable-task gap shipped."""

    def _parse(self, argv):
        return build_parser().parse_args(argv)

    def test_default_output_is_byte_exact(self, capsys):
        """`litmod render X` piped to a second coder must be the instrument,
        byte for byte. The provenance footer used to default ON, so the
        natural invocation shipped a non-byte-exact instrument to exactly
        the audience byte-exactness was built for."""
        from largeliterarymodels.cli.main import cmd_render
        from largeliterarymodels.tasks import GenreTaskLite
        args = self._parse(["render", "GenreTaskLite"])
        assert cmd_render(args) == 0
        out = capsys.readouterr().out
        assert out.rstrip("\n") == GenreTaskLite().instrument_text()

    def test_digest_is_opt_in(self, capsys):
        from largeliterarymodels.cli.main import cmd_render
        args = self._parse(["render", "GenreTaskLite", "--digest"])
        cmd_render(args)
        assert "instrument_sha256" in capsys.readouterr().out

    def test_unregistered_task_falls_back_to_the_package(self, capsys):
        """The CLI registry maps a curated dozen tasks; the reproducibility
        guarantee is claimed for the whole package. ContradictionResponseTask
        is real, importable, and was unreachable from the shell."""
        from largeliterarymodels.cli.main import cmd_render
        args = self._parse(["render", "ContradictionResponseTask",
                            "--item", "x"])
        assert cmd_render(args) == 0
        assert "ITEM TO ANNOTATE" in capsys.readouterr().out

    def test_unknown_task_still_errors(self):
        from largeliterarymodels.cli.main import cmd_render
        args = self._parse(["render", "NoSuchTask"])
        with pytest.raises(SystemExit):
            cmd_render(args)

    def test_fixture_without_adapter_says_why(self):
        from largeliterarymodels.cli.main import cmd_render
        args = self._parse(["render", "ContradictionResponseTask",
                            "--fixture"])
        with pytest.raises(SystemExit, match="no registered adapter"):
            cmd_render(args)


class TestDoctorShell:
    """cmd_doctor's logic is pure once _probe is stubbed; it stayed untested
    while being the designated answer to the whole provider-drift class —
    'I ran doctor and it passed' is only worth what these pin."""

    def _args(self, **kw):
        import argparse
        defaults = dict(provider=None, cheap_only=False, include_local=False,
                        timeout=5)
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def _run(self, args, probe, keys):
        import contextlib
        import io
        import largeliterarymodels.cli.doctor as D
        out = io.StringIO()
        with patch("largeliterarymodels.providers.check_api_keys",
                   return_value=keys):
            with patch.object(D, "_probe", side_effect=probe):
                with contextlib.redirect_stdout(out):
                    rc = D.cmd_doctor(args)
        return rc, out.getvalue()

    def test_unknown_provider_exits(self):
        import largeliterarymodels.cli.doctor as D
        with pytest.raises(SystemExit, match="unknown provider"):
            D.cmd_doctor(self._args(provider="anthropic,nonsense"))

    def test_nothing_probed_is_not_a_clean_bill(self):
        """Every provider skipped for want of a key exits 1. A doctor that
        exits 0 having probed nothing goes green forever in a CI whose key
        env var was renamed."""
        rc, out = self._run(self._args(), lambda m, t: ("PASS", "d", 0.1), {})
        assert rc == 1
        assert "probed NOTHING" in out

    def test_failure_sets_the_exit_code(self):
        rc, _ = self._run(self._args(provider="openai", cheap_only=True),
                          lambda m, t: ("FAIL", "boom", 0.1),
                          {"OPENAI_API_KEY": "x"})
        assert rc == 1

    def test_warn_is_surfaced_but_not_a_failure(self):
        rc, out = self._run(self._args(provider="deepseek"),
                            lambda m, t: ("WARN", "dropped params", 0.1),
                            {"DEEPSEEK_API_KEY": "x"})
        assert rc == 0
        assert "WARNED:" in out

    def test_summary_goes_to_stdout(self):
        """`litmod doctor > report.txt` used to capture an empty file —
        everything printed to stderr, including the verdict."""
        rc, out = self._run(self._args(provider="openai", cheap_only=True),
                            lambda m, t: ("PASS", "d", 0.1),
                            {"OPENAI_API_KEY": "x"})
        assert "passed 1" in out

    def test_cheap_only_probes_one_tier(self):
        probed = []

        def probe(m, t):
            probed.append(m)
            return "PASS", "d", 0.1
        self._run(self._args(provider="openai", cheap_only=True), probe,
                  {"OPENAI_API_KEY": "x"})
        assert probed == ["openai/gpt-5.4-nano"]

    def test_default_tier_covers_the_packages_own_default(self):
        """A retired per-provider default is exactly the drift doctor exists
        to catch, and it was the one tier the matrix never probed."""
        probed = []

        def probe(m, t):
            probed.append(m)
            return "PASS", "d", 0.1
        self._run(self._args(provider="openai"), probe,
                  {"OPENAI_API_KEY": "x"})
        assert "gpt-5.4-mini" in probed, probed

    def test_matrix_models_all_route(self):
        """A typo'd matrix id makes doctor FAIL for a reason that is not a
        provider bug — the worst outcome for a tool whose job is telling
        you the provider broke."""
        from largeliterarymodels.cli.doctor import (PROVIDER_MATRIX,
                                                    _default_model)
        from largeliterarymodels.providers import route_provider
        for name, spec in PROVIDER_MATRIX.items():
            for tier in ("cheap", "frontier"):
                if spec.get(tier):
                    assert route_provider(spec[tier])
            default = _default_model(spec.get("call"))
            if default:
                assert route_provider(default)


class TestRegistryIntegrity:
    def test_every_registered_task_imports(self):
        """The analysis registry carried an entry pointing at an untracked
        module: 397 tests passed over a registry broken for anyone who
        checked the branch out. Three lines make that impossible to repeat."""
        from largeliterarymodels.analysis.registry import TASK_REGISTRY
        import importlib
        for key, dotted in TASK_REGISTRY.items():
            module_path, cls_name = dotted.split(":")
            module = importlib.import_module(module_path)
            assert getattr(module, cls_name, None) is not None, (key, dotted)


class TestByteIdentityAcrossTheCatalog:
    def test_every_extract_task_renders_what_it_administers(self):
        """The byte-identity guarantee is claimed per-package, pinned on one
        toy task. Run it across the whole registry: any task overriding run
        or fabricating an instrument fails loudly here."""
        from hashstash import HashStash
        from tests.test_task_catalog import TASK_CLASSES
        from largeliterarymodels.task import SequentialTask

        checked = 0
        for cls in TASK_CLASSES:
            if cls.schema is None or issubclass(cls, SequentialTask):
                continue
            task = cls()
            task._stash = HashStash(engine="memory").clear()
            captured = {}

            def grab(**kw):
                captured["system_prompt"] = kw["system_prompt"]
                raise RuntimeError("stop before any parsing")

            with patch("largeliterarymodels.llm._call_provider",
                       side_effect=grab):
                try:
                    task.run("probe item", model="claude-sonnet-4-6")
                except Exception:
                    pass
            assert captured["system_prompt"] == task.instrument_text(), \
                cls.__name__
            checked += 1
        assert checked >= 10, f"only {checked} tasks checked — registry moved?"


class TestPriceShell:
    def _run(self, argv, capsys):
        from largeliterarymodels.cli.main import build_parser
        args = build_parser().parse_args(argv)
        rc = args.func(args)
        return rc, capsys.readouterr().out

    def test_one_model_prints_the_invoice_anchor(self, capsys):
        rc, out = self._run(["price", "--fresh", "517547", "--cached",
                             "18389760", "--output", "657056",
                             "--model", "gpt-4o-mini"], capsys)
        assert rc == 0 and "$1.8511" in out
        assert "prices fetched" in out

    def test_table_mode_ranks_and_flags_floors(self, capsys):
        rc, out = self._run(["price", "--fresh", "1000000",
                             "--output", "100000"], capsys)
        assert rc == 0
        assert "*floor" in out
        assert "gemini-3.6-flash" in out and "deepseek-v4-pro" in out

    def test_prefix_gate_reaches_the_shell(self, capsys):
        rc, out = self._run(["price", "--cached", "10000000",
                             "--prefix-tokens", "3000",
                             "--model", "claude-haiku-4-5"], capsys)
        assert "BELOW" in out and "4,096" in out

    def test_batch_notes_deepseeks_absence(self, capsys):
        rc, out = self._run(["price", "--fresh", "1000000", "--batch"],
                            capsys)
        assert "deepseek does not" in out


class TestPriceShellReviewFindings:
    def _run(self, argv, capsys):
        from largeliterarymodels.cli.main import build_parser
        args = build_parser().parse_args(argv)
        rc = args.func(args)
        out = capsys.readouterr()
        return rc, out.out, out.err

    def test_unknown_model_exits_cleanly(self, capsys):
        """S9 — a typo got a traceback instead of the module's genuinely
        good error message and a nonzero rc."""
        rc, out, err = self._run(["price", "--fresh", "1000",
                                  "--model", "claude-nonexistent-99"],
                                 capsys)
        assert rc == 1
        assert "model_pricing.json" in err

    def test_negative_times_refused(self, capsys):
        rc, out, err = self._run(["price", "--fresh", "1000",
                                  "--times", "-3"], capsys)
        assert rc == 2

    def test_prefix_tokens_reaches_table_mode(self, capsys):
        """S6 — accepted-and-ignored is worse than absent."""
        rc, out, err = self._run(["price", "--cached", "10000000",
                                  "--prefix-tokens", "2000"], capsys)
        assert rc == 0 and "!no-cache" in out
