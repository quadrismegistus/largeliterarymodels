"""Word-family adapter: tasks keyed on a single (word, POS) entry.

Serves TranslationTask. Fixtures are built in.
"""


FIXTURES = [
    {'_id': 'fixture/sensible', 'word': 'sensible', 'pos': 'Adjective'},
    {'_id': 'fixture/awful', 'word': 'awful', 'pos': 'Adjective'},
    {'_id': 'fixture/condescension', 'word': 'condescension', 'pos': 'Noun'},
]


class WordAdapter:
    family = 'word'

    def fixtures(self) -> list[dict]:
        return [dict(r) for r in FIXTURES]

    def build_prompt(self, record: dict) -> tuple[str, dict]:
        from largeliterarymodels.tasks import format_word_for_translation
        prompt = format_word_for_translation(record['word'], record['pos'])
        meta = {k: v for k, v in record.items()}
        return prompt, meta

    def load_input(self, source: str) -> list[dict]:
        """Load records from a CSV manifest with `word` and `pos` columns."""
        import pandas as pd
        df = pd.read_csv(source)
        missing = {'word', 'pos'} - set(df.columns)
        if missing:
            raise SystemExit(
                f"Input manifest {source!r} must have `word` and `pos` "
                f"columns. Found: {list(df.columns)}"
            )
        return [r.to_dict() for _, r in df.iterrows()]
