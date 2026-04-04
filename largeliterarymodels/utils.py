"""Utility functions for largeliterarymodels."""

from .providers import check_api_keys


def available_models(verbose=False):
    """Return a list of model suggestions based on which API keys are set."""
    keys = check_api_keys(verbose=verbose)
    models = []
    if "ANTHROPIC_API_KEY" in keys:
        models.extend(["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"])
    if "OPENAI_API_KEY" in keys:
        models.extend(["gpt-4o-mini", "gpt-4o"])
    if "GEMINI_API_KEY" in keys:
        models.extend(["gemini-2.5-flash", "gemini-2.5-pro"])
    return models
