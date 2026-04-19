"""Task catalog. Imports are lazy (PEP 562) so that tasks with optional
dependencies (e.g. classify_genre → lltk) don't force those deps on users
who only need lltk-free tasks like TranslationTask.

Usage is unchanged:

    from largeliterarymodels.tasks import TranslationTask, GenreTask
"""

import importlib

_LAZY_IMPORTS = {
    # classify_genre depends on lltk (GENRE_VOCAB is sourced from lltk.tools.vocabs)
    'GenreTask': ('.classify_genre', 'GenreTask'),
    'GenreClassification': ('.classify_genre', 'GenreClassification'),
    'format_text_for_classification': ('.classify_genre', 'format_text_for_classification'),

    # Remaining tasks are lltk-free but lazy-loaded for consistency
    'BibliographyTask': ('.extract_bibliography', 'BibliographyTask'),
    'BibliographyEntry': ('.extract_bibliography', 'BibliographyEntry'),
    'chunk_bibliography': ('.extract_bibliography', 'chunk_bibliography'),

    'FryeTask': ('.classify_frye', 'FryeTask'),
    'FryeClassification': ('.classify_frye', 'FryeClassification'),
    'format_text_for_frye': ('.classify_frye', 'format_text_for_frye'),

    'PassageTask': ('.classify_passage', 'PassageTask'),
    'PassageAnnotation': ('.classify_passage', 'PassageAnnotation'),
    'format_passage': ('.classify_passage', 'format_passage'),
    'format_chapters': ('.classify_passage', 'format_chapters'),
    'format_passages_from_text': ('.classify_passage', 'format_passages_from_text'),

    'CharacterTask': ('.resolve_characters', 'CharacterTask'),
    'CharacterResolution': ('.resolve_characters', 'CharacterResolution'),
    'format_character_roster': ('.resolve_characters', 'format_character_roster'),

    'CharacterIntroTask': ('.classify_character', 'CharacterIntroTask'),
    'CharacterIntroAnnotation': ('.classify_character', 'CharacterIntroAnnotation'),
    'format_character_intro': ('.classify_character', 'format_character_intro'),
    'format_all_character_intros': ('.classify_character', 'format_all_character_intros'),

    'TranslationTask': ('.translate_word', 'TranslationTask'),
    'WordTranslation': ('.translate_word', 'WordTranslation'),
    'format_word_for_translation': ('.translate_word', 'format_word_for_translation'),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys())
