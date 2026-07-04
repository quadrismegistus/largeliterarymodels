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
