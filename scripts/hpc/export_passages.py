"""Export passages from ClickHouse to JSONL files for HPC use.

Run locally (where ClickHouse is available) to produce a directory of
JSONL files that can be rsynced to HPC and processed with:

    python scripts/batch_social_network.py --text-dir texts/ --output-dir output/ --model vllm-qwen36 --workers 4

Usage:
    # Export all Early English Prose Fiction
    python scripts/hpc/export_passages.py --subcollection Early_English_Prose_Fiction --out texts/

    # Export all pre-1800 English fiction with passages
    python scripts/hpc/export_passages.py --query "genre='Fiction' AND year<1800 AND lang='en'" --out texts/

    # Export specific text IDs from a file (one per line)
    python scripts/hpc/export_passages.py --id-file manifest.txt --out texts/
"""

import argparse
import json
import os
import sys

import lltk


def export_text(text_id, out_dir):
    """Export one text's passages to a JSONL file.

    Each line: {"seq": N, "text": "...", "n_words": N}
    Filename: text_id with slashes replaced by underscores.
    """
    pdf = lltk.db.get_passages([text_id])
    if pdf is None or pdf.empty:
        return None
    pdf = pdf.sort_values('seq').reset_index(drop=True)
    slug = text_id.replace('/', '_').replace(' ', '_').strip('_')
    path = os.path.join(out_dir, f'{slug}.jsonl')
    with open(path, 'w') as f:
        for _, row in pdf.iterrows():
            f.write(json.dumps({
                'seq': int(row['seq']),
                'text': row['text'],
                'n_words': int(row.get('n_words', len(row['text'].split()))),
            }, ensure_ascii=False) + '\n')
    return path, len(pdf)


def main():
    parser = argparse.ArgumentParser(description='Export passages to JSONL for HPC')
    parser.add_argument('--subcollection', type=str,
                        help='Chadwyck subcollection name')
    parser.add_argument('--query', type=str,
                        help='WHERE clause for lltk.texts (e.g. "year<1800 AND lang=\'en\'")')
    parser.add_argument('--id-file', type=str,
                        help='File with one text ID per line')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for JSONL files')
    parser.add_argument('--min-passages', type=int, default=5,
                        help='Skip texts with fewer passages')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.id_file:
        with open(args.id_file) as f:
            text_ids = [line.strip() for line in f if line.strip()]
    elif args.subcollection:
        df = lltk.db.query(f"""
            SELECT t._id, count(p._id) as n_passages
            FROM (SELECT * FROM texts FINAL) AS t
            JOIN passages AS p ON t._id = p._id
            WHERE t._id LIKE '_chadwyck/{args.subcollection}/%'
            GROUP BY t._id
            HAVING n_passages >= {args.min_passages}
            ORDER BY t._id
        """)
        text_ids = df['_id'].tolist()
    elif args.query:
        df = lltk.db.query(f"""
            SELECT t._id, count(p._id) as n_passages
            FROM (SELECT * FROM texts FINAL) AS t
            JOIN passages AS p ON t._id = p._id
            WHERE {args.query}
            GROUP BY t._id
            HAVING n_passages >= {args.min_passages}
            ORDER BY t._id
        """)
        text_ids = df['_id'].tolist()
    else:
        parser.error('Provide --subcollection, --query, or --id-file')

    print(f"Exporting {len(text_ids)} texts to {args.out}/", file=sys.stderr)

    exported = 0
    for i, text_id in enumerate(text_ids):
        result = export_text(text_id, args.out)
        if result:
            path, n_psg = result
            exported += 1
            if (i + 1) % 50 == 0 or i == len(text_ids) - 1:
                print(f"  [{i+1}/{len(text_ids)}] {exported} exported",
                      file=sys.stderr)
        else:
            print(f"  SKIP {text_id} (no passages)", file=sys.stderr)

    print(f"\nDone: {exported}/{len(text_ids)} texts exported to {args.out}/",
          file=sys.stderr)


if __name__ == '__main__':
    main()
