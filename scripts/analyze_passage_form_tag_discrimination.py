"""Fisher-exact discrimination tests for PassageFormTask annotations.

Reads from llmtasks.passage_annotations via largeliterarymodels.analysis,
runs passage_groups + fisher_tests + BH-FDR. Outputs tidy CSV + per-kind
console summary.

Usage:
    python scripts/analyze_passage_form_tag_discrimination.py
    python scripts/analyze_passage_form_tag_discrimination.py \\
        --source-agent claude-sonnet-4-6 --alpha 0.001 --top 20
"""

import argparse
import sys

from largeliterarymodels.analysis import (
    bh_fdr, fisher_tests, joint_feature_matrix, passage_groups,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-version', type=int, default=1)
    ap.add_argument('--source-agent', default='claude-sonnet-4-6')
    ap.add_argument('--min-tag-passages', type=int, default=30)
    ap.add_argument('--min-feature-passages', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=0.001)
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--out', default=(
        '/Users/rj416/github/largeliterarymodels/data/'
        'analyze_passage_form_discrimination.csv'))
    ap.add_argument('--no-prose-filter', action='store_true')
    ap.add_argument('--no-pairs', action='store_true')
    ap.add_argument('--no-halfcent', action='store_true')
    ap.add_argument('--no-halfcent-tag', action='store_true')
    ap.add_argument('--no-feature-pairs', action='store_true')
    args = ap.parse_args()

    feats = joint_feature_matrix(
        tasks=['passage-form'],
        task_versions={'passage-form': args.task_version},
        source_agents={'passage-form': args.source_agent},
        is_prose_fiction=not args.no_prose_filter,
    )
    print(f"Loaded {len(feats)} passages × {feats.shape[1]} features",
          file=sys.stderr)

    groups, kind = passage_groups(
        feats.index,
        include_pairs=not args.no_pairs,
        include_halfcent=not args.no_halfcent,
        include_halfcent_tag=not args.no_halfcent_tag,
        min_group_n=args.min_tag_passages,
    )
    print(f"{groups.shape[1]} groups", file=sys.stderr)

    results = fisher_tests(
        feats, groups,
        min_group_n=args.min_tag_passages,
        min_feature_n=args.min_feature_passages,
        include_feature_pairs=not args.no_feature_pairs,
    )
    results['q_value'] = bh_fdr(results['p_value'])
    results['group_kind'] = results['group'].map(
        lambda g: kind.get(g, 'feature'))
    print(f"{len(results)} tests, sig at q<{args.alpha}: "
          f"{(results['q_value'] < args.alpha).sum()}", file=sys.stderr)

    results.to_csv(args.out, index=False)
    print(f"Wrote {args.out}", file=sys.stderr)

    for label, gk in [('SINGLE TAGS', 'single'), ('TAG PAIRS', 'pair'),
                       ('HALF-CENTURY', 'halfcent'),
                       ('HALFCENT × TAG', 'halfcent_tag'),
                       ('FEATURE × FEATURE', 'feature')]:
        sub = (results[(results['group_kind'] == gk)
                       & (results['q_value'] < args.alpha)]
               .sort_values('p_value').head(args.top))
        if len(sub):
            print(f"\n=== Top {args.top} {label} (q<{args.alpha}) ===")
            print(sub[['group', 'feature', 'rate_in_group', 'rate_not_group',
                       'odds_ratio', 'p_value', 'q_value']].to_string(
                           index=False))


if __name__ == '__main__':
    main()
