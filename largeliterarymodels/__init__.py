"""LLMs for digital humanities."""

from .llm import (
    LLM, DEFAULT_MODEL,
    CLAUDE_OPUS, CLAUDE_SONNET, CLAUDE_HAIKU,
    GPT_4O, GPT_4O_MINI,
    GEMINI_PRO, GEMINI_FLASH,
)
from .task import Task
from .providers import check_api_keys
from .utils import available_models
