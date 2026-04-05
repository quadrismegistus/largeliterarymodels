"""Tests for provider routing and helper functions (no API calls)."""

import os
import pytest
from unittest.mock import patch
from largeliterarymodels.providers import (
    route_provider, call_anthropic, call_openai, call_google,
    check_api_keys, _get_key, _strip_prefix, _load_image_bytes,
)


class TestRouteProvider:
    def test_claude_models(self):
        assert route_provider("claude-sonnet-4-20250514") is call_anthropic
        assert route_provider("claude-haiku-4-5-20251001") is call_anthropic
        assert route_provider("anthropic/claude-3-opus") is call_anthropic

    def test_openai_models(self):
        assert route_provider("gpt-4o") is call_openai
        assert route_provider("gpt-4o-mini") is call_openai
        assert route_provider("o1-preview") is call_openai
        assert route_provider("o3-mini") is call_openai
        assert route_provider("openai/gpt-4") is call_openai

    def test_google_models(self):
        assert route_provider("gemini-2.5-flash") is call_google
        assert route_provider("gemini-2.5-pro") is call_google
        assert route_provider("google/gemini-pro") is call_google

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot determine provider"):
            route_provider("llama-3.1-70b")

    def test_case_insensitive(self):
        assert route_provider("Claude-Sonnet-4") is call_anthropic
        assert route_provider("GPT-4o") is call_openai
        assert route_provider("Gemini-Pro") is call_google


class TestStripPrefix:
    def test_anthropic_prefix(self):
        assert _strip_prefix("anthropic/claude-3-opus") == "claude-3-opus"

    def test_openai_prefix(self):
        assert _strip_prefix("openai/gpt-4o") == "gpt-4o"

    def test_google_prefix(self):
        assert _strip_prefix("google/gemini-pro") == "gemini-pro"

    def test_no_prefix(self):
        assert _strip_prefix("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


class TestLoadImageBytes:
    def test_from_bytes(self):
        data, mime = _load_image_bytes(b"\x89PNG\r\n")
        assert data == b"\x89PNG\r\n"
        assert mime == "image/png"

    def test_from_path(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff")
        data, mime = _load_image_bytes(str(img_path))
        assert data == b"\xff\xd8\xff"
        assert mime == "image/jpeg"


class TestGetKey:
    def test_existing_key(self):
        with patch.dict(os.environ, {"TEST_KEY_123": "secret"}):
            assert _get_key("TEST_KEY_123") == "secret"

    def test_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="Missing TEST_KEY_123"):
                _get_key("TEST_KEY_123")


class TestCheckApiKeys:
    def test_returns_available_keys(self):
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-xxx",
            "OPENAI_API_KEY": "",
            "GEMINI_API_KEY": "aig-xxx",
        }
        with patch.dict(os.environ, env, clear=True):
            keys = check_api_keys()
            assert "ANTHROPIC_API_KEY" in keys
            assert "GEMINI_API_KEY" in keys
            assert "OPENAI_API_KEY" not in keys

    def test_empty_env(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = check_api_keys()
            assert keys == {}
