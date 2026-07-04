"""Work-family adapter: tasks classified from title/author/year metadata.

Serves GenreTask, GenreTaskLite, MajorGenreTask — anything whose prompt is
built from bibliographic metadata rather than passage text. Fixtures are
built in (no ClickHouse needed), so `litmod smoke GenreTask --model sonnet`
works out of the box.
"""


FIXTURE_META = [
    {
        '_id': 'fixture/pamela',
        'title': 'Pamela: or, Virtue Rewarded',
        'author': 'Richardson, Samuel',
        'year': 1740,
    },
    {
        '_id': 'fixture/otranto',
        'title': 'The Castle of Otranto, A Story. Translated by William '
                 'Marshal, Gent. From the Original Italian of Onuphrio '
                 'Muralto',
        'author': 'Walpole, Horace',
        'year': 1764,
    },
    {
        '_id': 'fixture/female_quixote',
        'title': 'The Female Quixote; or, The Adventures of Arabella',
        'author': 'Lennox, Charlotte',
        'year': 1752,
    },
]


class WorkAdapter:
    family = 'work'

    def fixtures(self) -> list[dict]:
        return [dict(r) for r in FIXTURE_META]

    def build_prompt(self, record: dict) -> tuple[str, dict]:
        from largeliterarymodels.tasks import format_text_for_classification
        prompt = format_text_for_classification(
            title=record['title'],
            author=record.get('author'),
            author_norm=record.get('author_norm'),
            year=record.get('year'),
            subject_topic=record.get('subject_topic'),
            form=record.get('form'),
        )
        meta = {k: v for k, v in record.items() if k != 'text'}
        return prompt, meta

    def load_input(self, source: str) -> list[dict]:
        """Load work records from a CSV manifest with a `title` column
        (optional: author, author_norm, year, subject_topic, form, _id)."""
        import pandas as pd
        df = pd.read_csv(source)
        if 'title' not in df.columns:
            raise SystemExit(
                f"Input manifest {source!r} must have a `title` column. "
                f"Found: {list(df.columns)}"
            )
        for col in ('author', 'author_norm', 'subject_topic', 'form'):
            if col in df.columns:
                df[col] = df[col].fillna('')
        records = []
        for _, r in df.iterrows():
            rec = r.to_dict()
            if 'year' in rec and pd.isna(rec.get('year')):
                rec['year'] = None
            records.append(rec)
        return records
