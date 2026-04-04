"""Tests for utility functions."""

import os
import pytest
from unittest.mock import patch
from largeliterarymodels.utils import available_models


class TestAvailableModels:
    def test_all_keys_present(self):
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-xxx",
            "OPENAI_API_KEY": "sk-xxx",
            "GEMINI_API_KEY": "aig-xxx",
        }
        with patch.dict(os.environ, env, clear=True):
            models = available_models()
            assert any("claude" in m for m in models)
            assert any("gpt" in m for m in models)
            assert any("gemini" in m for m in models)

    def test_only_anthropic(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant-xxx"}
        with patch.dict(os.environ, env, clear=True):
            models = available_models()
            assert all("claude" in m for m in models)
            assert not any("gpt" in m for m in models)

    def test_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            models = available_models()
            assert models == []
