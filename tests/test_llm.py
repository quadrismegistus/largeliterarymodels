"""Tests for LLM class and extraction helpers (mocked API calls)."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field
from hashstash import HashStash
from largeliterarymodels.llm import (
    LLM, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS,
    _make_key, _schema_to_json_spec, _unwrap_schema, _schema_name,
    _format_examples, _build_extract_prompt, _parse_json_response,
    _validate_parsed,
)


# --- Test schemas ---

class SimpleResult(BaseModel):
    answer: str
    confidence: float = Field(description="0.0 to 1.0")


class Character(BaseModel):
    name: str
    role: str


# --- Module-level helpers ---

class TestMakeKey:
    def test_basic_key(self):
        key = _make_key("hello", "gpt-4o")
        assert key == {
            "prompt": "hello",
            "model": "gpt-4o",
            "system_prompt": None,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

    def test_key_with_schema(self):
        key = _make_key("hello", "gpt-4o", schema_name="SimpleResult")
        assert key["schema"] == "SimpleResult"

    def test_key_without_schema_has_no_schema_field(self):
        key = _make_key("hello", "gpt-4o")
        assert "schema" not in key

    def test_different_params_different_keys(self):
        k1 = _make_key("hello", "gpt-4o", temperature=0.5)
        k2 = _make_key("hello", "gpt-4o", temperature=0.9)
        assert k1 != k2


class TestUnwrapSchema:
    def test_plain_model(self):
        is_list, model = _unwrap_schema(SimpleResult)
        assert is_list is False
        assert model is SimpleResult

    def test_list_model(self):
        is_list, model = _unwrap_schema(list[Character])
        assert is_list is True
        assert model is Character


class TestSchemaName:
    def test_plain(self):
        assert _schema_name(SimpleResult) == "SimpleResult"

    def test_list(self):
        assert _schema_name(list[Character]) == "list[Character]"


class TestSchemaToJsonSpec:
    def test_plain_model(self):
        spec = _schema_to_json_spec(SimpleResult)
        assert "a JSON object matching this schema" in spec
        assert "answer" in spec
        assert "confidence" in spec

    def test_list_model(self):
        spec = _schema_to_json_spec(list[Character])
        assert "a JSON array of objects" in spec
        assert "name" in spec


class TestFormatExamples:
    def test_empty_examples(self):
        assert _format_examples([], SimpleResult) == ""

    def test_pydantic_instance(self):
        examples = [("input text", SimpleResult(answer="yes", confidence=0.9))]
        result = _format_examples(examples, SimpleResult)
        assert "Example 1 input:" in result
        assert "input text" in result
        assert "yes" in result

    def test_dict_output(self):
        examples = [("input text", {"answer": "yes", "confidence": 0.9})]
        result = _format_examples(examples, SimpleResult)
        assert "yes" in result

    def test_list_output(self):
        examples = [("input text", [Character(name="Alice", role="hero")])]
        result = _format_examples(examples, list[Character])
        assert "Alice" in result

    def test_multiple_examples(self):
        examples = [
            ("first", SimpleResult(answer="a", confidence=0.5)),
            ("second", SimpleResult(answer="b", confidence=0.8)),
        ]
        result = _format_examples(examples, SimpleResult)
        assert "Example 1" in result
        assert "Example 2" in result
        assert "---" in result


class TestBuildExtractPrompt:
    def test_basic(self):
        system, user = _build_extract_prompt("analyze this", SimpleResult)
        assert user == "analyze this"
        assert "valid JSON" in system
        assert "answer" in system

    def test_with_system_prompt(self):
        system, user = _build_extract_prompt(
            "analyze this", SimpleResult, system_prompt="You are a scholar."
        )
        assert "You are a scholar." in system
        assert "valid JSON" in system

    def test_with_examples(self):
        examples = [("input", SimpleResult(answer="yes", confidence=0.9))]
        system, user = _build_extract_prompt(
            "analyze this", SimpleResult, examples=examples
        )
        assert "Example 1" in system


class TestParseJsonResponse:
    def test_plain_json(self):
        assert _parse_json_response('{"a": 1}') == {"a": 1}

    def test_json_array(self):
        assert _parse_json_response('[{"a": 1}]') == [{"a": 1}]

    def test_markdown_fenced(self):
        text = '```json\n{"a": 1}\n```'
        assert _parse_json_response(text) == {"a": 1}

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"a": 1}\n```'
        assert _parse_json_response(text) == {"a": 1}

    def test_surrounding_text(self):
        text = 'Here is the result:\n{"a": 1}\nHope that helps!'
        assert _parse_json_response(text) == {"a": 1}

    def test_surrounding_text_array(self):
        text = 'Here: [{"a": 1}] done.'
        assert _parse_json_response(text) == [{"a": 1}]

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            _parse_json_response("no json here at all")


class TestValidateParsed:
    def test_single_model(self):
        result = _validate_parsed({"answer": "yes", "confidence": 0.9}, SimpleResult)
        assert isinstance(result, SimpleResult)
        assert result.answer == "yes"

    def test_list_model(self):
        data = [{"name": "Alice", "role": "hero"}, {"name": "Bob", "role": "villain"}]
        result = _validate_parsed(data, list[Character])
        assert len(result) == 2
        assert result[0].name == "Alice"

    def test_list_model_wraps_single_dict(self):
        result = _validate_parsed({"name": "Alice", "role": "hero"}, list[Character])
        assert len(result) == 1

    def test_validation_error(self):
        with pytest.raises(Exception):
            _validate_parsed({"wrong_field": "x"}, SimpleResult)


# --- LLM class ---

class TestLLMInit:
    def test_defaults(self):
        llm = LLM()
        assert llm.model == DEFAULT_MODEL
        assert llm.temperature == DEFAULT_TEMPERATURE
        assert llm.max_tokens == DEFAULT_MAX_TOKENS
        assert llm.system_prompt is None

    def test_custom_params(self):
        llm = LLM(model="gpt-4o", temperature=0.2, system_prompt="be terse")
        assert llm.model == "gpt-4o"
        assert llm.temperature == 0.2
        assert llm.system_prompt == "be terse"

    def test_custom_stash(self):
        stash = HashStash(engine="memory")
        llm = LLM(stash=stash)
        assert llm.stash is stash

    def test_repr(self):
        llm = LLM("gpt-4o-mini")
        assert repr(llm) == "LLM(model='gpt-4o-mini')"


class TestLLMGenerate:
    @patch("largeliterarymodels.llm._call_provider")
    def test_basic_generate(self, mock_call):
        mock_call.return_value = "hello world"
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        result = llm.generate("say hello")
        assert result == "hello world"
        mock_call.assert_called_once()

    @patch("largeliterarymodels.llm._call_provider")
    def test_caching(self, mock_call):
        mock_call.return_value = "hello world"
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        r1 = llm.generate("say hello")
        r2 = llm.generate("say hello")
        assert r1 == r2
        assert mock_call.call_count == 1

    @patch("largeliterarymodels.llm._call_provider")
    def test_force_bypasses_cache(self, mock_call):
        mock_call.return_value = "hello world"
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        llm.generate("say hello")
        llm.generate("say hello", force=True)
        assert mock_call.call_count == 2

    @patch("largeliterarymodels.llm._call_provider")
    def test_system_prompt_override(self, mock_call):
        mock_call.return_value = "result"
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", system_prompt="default", stash=stash)
        llm.generate("hello", system_prompt="override")
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["system_prompt"] == "override"

    @patch("largeliterarymodels.llm._call_provider")
    def test_instance_system_prompt_used_when_no_override(self, mock_call):
        mock_call.return_value = "result"
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", system_prompt="default", stash=stash)
        llm.generate("hello")
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["system_prompt"] == "default"


class TestLLMExtract:
    @patch("largeliterarymodels.llm._call_provider")
    def test_basic_extract(self, mock_call):
        mock_call.return_value = '{"answer": "42", "confidence": 0.95}'
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        result = llm.extract("what is the answer?", schema=SimpleResult)
        assert isinstance(result, SimpleResult)
        assert result.answer == "42"
        assert result.confidence == 0.95

    @patch("largeliterarymodels.llm._call_provider")
    def test_extract_list(self, mock_call):
        mock_call.return_value = '[{"name": "Alice", "role": "hero"}, {"name": "Bob", "role": "villain"}]'
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        result = llm.extract("list characters", schema=list[Character])
        assert len(result) == 2
        assert result[0].name == "Alice"

    @patch("largeliterarymodels.llm._call_provider")
    def test_extract_with_markdown_fencing(self, mock_call):
        mock_call.return_value = '```json\n{"answer": "yes", "confidence": 0.8}\n```'
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        result = llm.extract("question", schema=SimpleResult)
        assert result.answer == "yes"

    @patch("largeliterarymodels.llm._call_provider")
    def test_extract_retry_on_bad_json(self, mock_call):
        mock_call.side_effect = [
            "not valid json at all",
            '{"answer": "yes", "confidence": 0.8}',
        ]
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        result = llm.extract("question", schema=SimpleResult, retries=1)
        assert result.answer == "yes"
        assert mock_call.call_count == 2

    @patch("largeliterarymodels.llm._call_provider")
    def test_extract_fails_after_retries_exhausted(self, mock_call):
        mock_call.return_value = "garbage"
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        with pytest.raises(ValueError, match="Failed to extract"):
            llm.extract("question", schema=SimpleResult, retries=1)

    @patch("largeliterarymodels.llm._call_provider")
    def test_extract_caching(self, mock_call):
        mock_call.return_value = '{"answer": "42", "confidence": 0.95}'
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        r1 = llm.extract("question", schema=SimpleResult)
        r2 = llm.extract("question", schema=SimpleResult)
        assert r1.answer == r2.answer
        assert mock_call.call_count == 1

    @patch("largeliterarymodels.llm._call_provider")
    def test_extract_with_examples(self, mock_call):
        mock_call.return_value = '{"answer": "no", "confidence": 0.7}'
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        examples = [("example input", SimpleResult(answer="yes", confidence=0.9))]
        result = llm.extract("question", schema=SimpleResult, examples=examples)
        # verify examples made it into the system prompt
        call_kwargs = mock_call.call_args[1]
        assert "Example 1" in call_kwargs["system_prompt"]


class TestLLMMap:
    @patch("largeliterarymodels.llm._call_provider")
    def test_basic_map(self, mock_call):
        mock_call.side_effect = ["one", "two", "three"]
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        results = llm.map(["a", "b", "c"], num_workers=1)
        assert results == ["one", "two", "three"]

    @patch("largeliterarymodels.llm._call_provider")
    def test_map_caching(self, mock_call):
        mock_call.side_effect = ["one", "two"]
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        llm.map(["a", "b"], num_workers=1)
        results = llm.map(["a", "b"], num_workers=1)
        assert results == ["one", "two"]
        assert mock_call.call_count == 2  # only called for first run

    @patch("largeliterarymodels.llm._call_provider")
    def test_map_partial_cache(self, mock_call):
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        mock_call.return_value = "one"
        llm.generate("a")  # cache "a"
        mock_call.return_value = "two"
        results = llm.map(["a", "b"], num_workers=1)
        assert results == ["one", "two"]


class TestLLMExtractMap:
    @patch("largeliterarymodels.llm._call_provider")
    def test_basic_extract_map(self, mock_call):
        mock_call.side_effect = [
            '{"answer": "one", "confidence": 0.9}',
            '{"answer": "two", "confidence": 0.8}',
        ]
        stash = HashStash(engine="memory").clear()
        llm = LLM(model="gpt-4o-mini", stash=stash)
        results = llm.extract_map(
            ["q1", "q2"], schema=SimpleResult, num_workers=1,
        )
        assert len(results) == 2
        assert results[0].answer == "one"
        assert results[1].answer == "two"
