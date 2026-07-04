"""Text-family adapter: tasks that annotate a raw passage of text.

Serves EmotionTask, EmotionTaskZh — anything whose prompt is simply the
passage (plus optional metadata header). Fixtures are built in, so smoke
tests need no ClickHouse.
"""


FIXTURES = [
    {
        '_id': 'fixture/dorrit_marseilles',
        'title': 'Little Dorrit',
        'author': 'Dickens, Charles',
        'year': 1857,
        'text': (
            "Thirty years ago, Marseilles lay burning in the sun, one day. "
            "A blazing sun upon a fierce August day was no greater rarity in "
            "southern France then, than at any other time, before or since. "
            "Everything in Marseilles, and about Marseilles, had stared at "
            "the fervid sky, and been stared at in return, until a staring "
            "habit had become universal there."
        ),
    },
    {
        '_id': 'fixture/wuthering_return',
        'title': 'Wuthering Heights',
        'author': 'Brontë, Emily',
        'year': 1847,
        'text': (
            "My fingers closed on the fingers of a little, ice-cold hand! "
            "The intense horror of nightmare came over me: I tried to draw "
            "back my arm, but the hand clung to it, and a most melancholy "
            "voice sobbed, 'Let me in — let me in!' 'Who are you?' I asked, "
            "struggling, meanwhile, to disengage myself."
        ),
    },
    {
        '_id': 'fixture/gazetteer',
        'title': 'A Topographical Account of the Hundred of Bosmere',
        'author': 'Anon.',
        'year': 1798,
        'text': (
            "The parish contains two thousand one hundred acres, of which "
            "three parts in four are arable. The soil is in general a wet "
            "loam upon clay; the tithes were commuted in the last "
            "enclosure, and the glebe amounts to forty-two acres."
        ),
    },
]


class TextAdapter:
    family = 'text'

    def fixtures(self) -> list[dict]:
        return [dict(r) for r in FIXTURES]

    def build_prompt(self, record: dict) -> tuple[str, dict]:
        header_bits = []
        if record.get('title'):
            header_bits.append(f"title: {record['title']}")
        if record.get('author'):
            header_bits.append(f"author: {record['author']}")
        if record.get('year'):
            header_bits.append(f"year: {record['year']}")
        header = f"({'; '.join(header_bits)})\n" if header_bits else ""
        prompt = f"{header}PASSAGE: {record['text']}"
        meta = {k: v for k, v in record.items() if k != 'text'}
        return prompt, meta

    def load_input(self, source: str) -> list[dict]:
        """Load records from a CSV manifest with a `text` column
        (optional: title, author, year, _id)."""
        import pandas as pd
        df = pd.read_csv(source)
        if 'text' not in df.columns:
            raise SystemExit(
                f"Input manifest {source!r} must have a `text` column. "
                f"Found: {list(df.columns)}"
            )
        records = []
        for _, r in df.iterrows():
            rec = r.to_dict()
            if not isinstance(rec.get('text'), str) or not rec['text'].strip():
                continue
            records.append(rec)
        return records
