"""Passage-family adapter.

Serves PassageContentTask, PassageContentTaskV1, PassageFormTask, PassageTask —
any task whose prompt is built from (passage text, title, author, year).
"""


FIXTURE_META = [
    {
        '_id': '_chadwyck/Early_English_Prose_Fiction/ee01010.02',
        'seq': 25,
        'title': 'Alcander and Philocrates',
        'author': 'Anon.',
        'year': 1696,
    },
    {
        '_id': '_chadwyck/Eighteenth-Century_Fiction/amory.01',
        'seq': 25,
        'title': 'John Buncle',
        'author': 'Amory, Thomas',
        'year': 1756,
    },
    {
        '_id': '_chadwyck/Nineteenth-Century_Fiction/ncf0302.01',
        'seq': 25,
        'title': 'Hermsprong; or, Man As He Is Not',
        'author': 'Bage, Robert',
        'year': 1796,
    },
]


class PassageAdapter:
    family = 'passage'

    def fixtures(self) -> list[dict]:
        import lltk
        ids = [f['_id'] for f in FIXTURE_META]
        df = lltk.db.get_passages(ids)
        lookup = {(r['_id'], int(r['seq'])): r['text'] for _, r in df.iterrows()}
        out = []
        for f in FIXTURE_META:
            key = (f['_id'], f['seq'])
            if key not in lookup:
                raise SystemExit(
                    f"Missing passage {key} — re-run `lltk db-passages`?"
                )
            out.append({**f, 'text': lookup[key]})
        return out

    def build_prompt(self, record: dict) -> tuple[str, dict]:
        from largeliterarymodels.tasks import format_passage
        return format_passage(
            record['text'],
            title=record['title'],
            author=record['author'],
            year=record['year'],
            _id=record['_id'],
            section_id=f"p500:{record['seq']}",
        )

    def load_input(self, source: str) -> list[dict]:
        """Load passage records from a CSV manifest.

        The manifest must have columns `_id` and `seq`. Optional columns
        (`title`, `author`, `year`, `tag_label`, `n_words`, ...) are
        preserved on each record and written through to the output CSV.
        Passage text is fetched via `lltk.db.get_passages` and attached
        as `text` on each record.
        """
        import lltk
        import pandas as pd

        df = pd.read_csv(source)
        for col in ('title', 'author', 'tag_label'):
            if col in df.columns:
                df[col] = df[col].fillna('')

        if '_id' not in df.columns or 'seq' not in df.columns:
            raise SystemExit(
                f"Input manifest {source!r} must have `_id` and `seq` columns. "
                f"Found: {list(df.columns)}"
            )

        ids = list(dict.fromkeys(df['_id'].tolist()))
        wanted = set(zip(df['_id'], df['seq'].astype(int)))
        pdf = lltk.db.get_passages(ids)
        pdf = pdf[pdf.apply(
            lambda r: (r['_id'], int(r['seq'])) in wanted, axis=1)]

        text_lookup = {(r['_id'], int(r['seq'])): r['text']
                       for _, r in pdf.iterrows()}
        nwords_lookup = {(r['_id'], int(r['seq'])): int(r['n_words'])
                         for _, r in pdf.iterrows()}

        records = []
        missing = 0
        for _, r in df.iterrows():
            key = (r['_id'], int(r['seq']))
            if key not in text_lookup:
                missing += 1
                continue
            rec = r.to_dict()
            rec['seq'] = int(rec['seq'])
            rec['text'] = text_lookup[key]
            if 'n_words' not in rec or pd.isna(rec.get('n_words')):
                rec['n_words'] = nwords_lookup.get(key, 0)
            if 'year' in rec and pd.isna(rec.get('year')):
                rec['year'] = None
            records.append(rec)
        if missing:
            print(f"[load_input] {missing} manifest rows had no passage text "
                  f"(skipped)", flush=True)
        return records
