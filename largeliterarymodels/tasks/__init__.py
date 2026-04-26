"""Task catalog with lazy imports (PEP 562).

Usage:
    from largeliterarymodels.tasks import TranslationTask, GenreTask
"""

import importlib

_LAZY_IMPORTS = {
    'GenreTask': ('.classify_genre', 'GenreTask'),
    'GenreClassification': ('.classify_genre', 'GenreClassification'),
    'format_text_for_classification': ('.classify_genre', 'format_text_for_classification'),

    'GenreTaskLite': ('.classify_genre_lite', 'GenreTaskLite'),
    'GenreClassificationLite': ('.classify_genre_lite', 'GenreClassificationLite'),
    'GENRE_FORM_TAGS': ('.classify_genre_lite', 'GENRE_FORM_TAGS'),
    'GENRE_MODE_TAGS': ('.classify_genre_lite', 'GENRE_MODE_TAGS'),
    'ALL_GENRE_TAGS': ('.classify_genre_lite', 'ALL_GENRE_TAGS'),

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

    'PassageContentTask': ('.classify_passage_content', 'PassageContentTask'),
    'PassageContentAnnotation': ('.classify_passage_content', 'PassageContentAnnotation'),
    'PassageContentTaskV1': ('.classify_passage_content', 'PassageContentTaskV1'),
    'PassageContentAnnotationV1': ('.classify_passage_content', 'PassageContentAnnotationV1'),

    'PassageFormTask': ('.classify_passage_form', 'PassageFormTask'),
    'PassageFormAnnotation': ('.classify_passage_form', 'PassageFormAnnotation'),

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

    'SocialNetworkTask': ('.extract_social_network', 'SocialNetworkTask'),

    'OCRCleanTask': ('.clean_ocr', 'OCRCleanTask'),
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
