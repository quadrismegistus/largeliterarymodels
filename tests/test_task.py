"""Tests for Task class and BibliographyTask (mocked API calls)."""

import pytest
from unittest.mock import patch
from pydantic import BaseModel, Field
from hashstash import HashStash
from largeliterarymodels.task import Task, _schema_repr
from largeliterarymodels.llm import STASH_PATH


class Sentiment(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float


class TestTaskInit:
    def test_defaults(self):
        task = Task()
        assert task.name is None
        assert task.task_name == "Task"
        assert task.schema is None
        assert task.examples == []

    def test_kwargs_override(self):
        task = Task(name="custom", retries=3)
        assert task.task_name == "custom"
        assert task.retries == 3

    def test_repr(self):
        task = Task()
        assert "Task" in repr(task)


class TestTaskStash:
    def test_stash_path_includes_name(self):
        task = Task(name="test_task")
        stash = task.stash
        assert "test_task" in str(stash)

    def test_stash_is_lazy(self):
        task = Task(name="lazy_test")
        assert task._stash is None
        _ = task.stash
        assert task._stash is not None

    def test_stash_is_cached(self):
        task = Task(name="cached_test")
        s1 = task.stash
        s2 = task.stash
        assert s1 is s2


class TestTaskSubclass:
    def test_subclass_with_schema(self):
        class SentimentTask(Task):
            name = "sentiment"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        assert task.name == "sentiment"
        assert task.schema is Sentiment
        assert "sentiment" in str(task.stash)

    def test_subclass_repr(self):
        class SentimentTask(Task):
            name = "sentiment"
            schema = Sentiment
        assert "Sentiment" in repr(SentimentTask())

    def test_list_schema_repr(self):
        class MultiTask(Task):
            name = "multi"
            schema = list[Sentiment]
        assert "list[Sentiment]" in repr(MultiTask())


class TestTaskRun:
    @patch("largeliterarymodels.llm._call_provider")
    def test_run_returns_validated_model(self, mock_call):
        mock_call.return_value = '{"sentiment": "positive", "confidence": 0.95}'

        class SentimentTask(Task):
            name = "sentiment_test_run"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        result = task.run("I love this!")
        assert isinstance(result, Sentiment)
        assert result.sentiment == "positive"

    def test_run_raises_without_schema(self):
        task = Task(name="no_schema")
        with pytest.raises(ValueError, match="no schema defined"):
            task.run("hello")

    @patch("largeliterarymodels.llm._call_provider")
    def test_run_with_model_override(self, mock_call):
        mock_call.return_value = '{"sentiment": "negative", "confidence": 0.8}'

        class SentimentTask(Task):
            name = "sentiment_test_model"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        task.run("I hate this!", model="gpt-4o-mini")
        assert mock_call.call_args[1]["model"] == "gpt-4o-mini"

    @patch("largeliterarymodels.llm._call_provider")
    def test_run_with_examples(self, mock_call):
        mock_call.return_value = '{"sentiment": "neutral", "confidence": 0.5}'

        class SentimentTask(Task):
            name = "sentiment_test_examples"
            schema = Sentiment
            system_prompt = "Assess sentiment."
            examples = [
                ("Great!", Sentiment(sentiment="positive", confidence=0.9)),
            ]

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        task.run("It's okay.")
        call_kwargs = mock_call.call_args[1]
        assert "Example 1" in call_kwargs["system_prompt"]


class TestTaskMap:
    @patch("largeliterarymodels.llm._call_provider")
    def test_map_returns_list(self, mock_call):
        mock_call.side_effect = [
            '{"sentiment": "positive", "confidence": 0.9}',
            '{"sentiment": "negative", "confidence": 0.8}',
        ]

        class SentimentTask(Task):
            name = "sentiment_test_map"
            schema = Sentiment
            system_prompt = "Assess sentiment."

        task = SentimentTask()
        task._stash = HashStash(engine="memory").clear()
        results = task.map(["I love it", "I hate it"], num_workers=1)
        assert len(results) == 2
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"

    def test_map_raises_without_schema(self):
        task = Task(name="no_schema")
        with pytest.raises(ValueError, match="no schema defined"):
            task.map(["hello"])


class TestSchemaRepr:
    def test_none(self):
        assert _schema_repr(None) == "None"

    def test_plain(self):
        assert _schema_repr(Sentiment) == "Sentiment"

    def test_list(self):
        assert _schema_repr(list[Sentiment]) == "list[Sentiment]"


# --- BibliographyTask ---

class TestBibliographyTask:
    def test_import(self):
        from largeliterarymodels.tasks import BibliographyTask, BibliographyEntry
        task = BibliographyTask()
        assert task.task_name == "BibliographyTask"
        assert task.retries == 2
        assert task.max_tokens == 8192

    def test_schema_is_list(self):
        from largeliterarymodels.tasks import BibliographyTask
        task = BibliographyTask()
        origin = getattr(task.schema, "__origin__", None)
        assert origin is list

    def test_has_examples(self):
        from largeliterarymodels.tasks import BibliographyTask
        task = BibliographyTask()
        assert len(task.examples) == 3
        for input_text, output in task.examples:
            assert isinstance(input_text, str)
            assert hasattr(output, "model_dump")

    def test_bibliography_entry_fields(self):
        from largeliterarymodels.tasks import BibliographyEntry
        entry = BibliographyEntry(
            author="GREENE, ROBERT",
            title="Greenes Never too late",
            year=1600,
        )
        assert entry.author == "GREENE, ROBERT"
        assert entry.is_translated is False
        assert entry.printer == ""

    def test_bibliography_entry_all_fields(self):
        from largeliterarymodels.tasks import BibliographyEntry
        entry = BibliographyEntry(
            author="BIDPAI",
            title="The morall philosophic of Doni",
            title_sub=": drawne out of the ancient writers",
            year=1601,
            edition="Second edition",
            id_biblio="STC 3054",
            is_translated=True,
            translated_from="",
            translator="Sir Thomas North",
            printer="S. Stafford",
            publisher="",
            bookseller="",
            notes_biblio="First edition in 1570.",
            notes="",
        )
        assert entry.is_translated is True
        assert entry.translator == "Sir Thomas North"

    @patch("largeliterarymodels.llm._call_provider")
    def test_bibliography_task_run(self, mock_call):
        from largeliterarymodels.tasks import BibliographyTask
        mock_call.return_value = '''[{
            "author": "DEKKER, THOMAS",
            "title": "The wonderfull yeare",
            "title_sub": "",
            "year": 1603,
            "edition": "",
            "id_biblio": "STC 6534",
            "is_translated": false,
            "translated_from": "",
            "translator": "",
            "printer": "T. Creede",
            "publisher": "",
            "bookseller": "",
            "notes_biblio": "First of three editions.",
            "notes": ""
        }]'''
        task = BibliographyTask()
        task._stash = HashStash(engine="memory").clear()
        entries = task.run("test chunk")
        assert len(entries) == 1
        assert entries[0].author == "DEKKER, THOMAS"
        assert entries[0].printer == "T. Creede"
