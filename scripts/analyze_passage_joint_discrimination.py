"""Fisher-exact discrimination tests on the JOINT Content + Form feature space.

Merges PassageContentTask and PassageFormTask annotations from CH on
(_id, scheme, seq), yields ~90 features per passage. Best for tests of Ch5
theses that require both content (scene_content, character classes) and
form (concrete_bespeaks_abstract, narrate_vs_describe).

Usage:
    python scripts/analyze_passage_joint_discrimination.py
    python scripts/analyze_passage_joint_discrimination.py \\
        --content-version 2 --form-version 1 --alpha 0.001 --top 20

Defaults align with the V1 manifest (Content V2 qwen × Form V1 sonnet,
both on 1,110 passages).
"""

import argparse
import sys

from largeliterarymodels.analysis import (
    bh_fdr, fisher_tests, joint_feature_matrix, passage_groups,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--content-version', type=int, default=2)
    ap.add_argument('--form-version', type=int, default=1)
    ap.add_argument('--content-agent', default='qwen3.5-35b-a3b')
    ap.add_argument('--form-agent', default='claude-sonnet-4-6')
    ap.add_argument('--min-tag-passages', type=int, default=30)
    ap.add_argument('--min-feature-passages', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=0.001)
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--out', default=(
        '/Users/rj416/github/largeliterarymodels/data/'
        'analyze_passage_joint_discrimination.csv'))
    ap.add_argument('--no-prose-filter', action='store_true')
    ap.add_argument('--no-pairs', action='store_true')
    ap.add_argument('--no-halfcent', action='store_true')
    ap.add_argument('--no-halfcent-tag', action='store_true')
    ap.add_argument('--no-feature-pairs', action='store_true')
    args = ap.parse_args()

    feats = joint_feature_matrix(
        tasks=['passage-content', 'passage-form'],
        task_versions={'passage-content': args.content_version,
                       'passage-form': args.form_version},
        source_agents={'passage-content': args.content_agent,
                       'passage-form': args.form_agent},
        is_prose_fiction=not args.no_prose_filter,
    )
    n_content = sum(1 for c in feats.columns if c.startswith('content.'))
    n_form = sum(1 for c in feats.columns if c.startswith('form.'))
    print(f"Joint: {len(feats)} passages × {feats.shape[1]} features "
          f"({n_content} content + {n_form} form)", file=sys.stderr)

    # V2 Content has no is_prose_fiction; rely on form's is_prose_fiction if
    # Content version is pre-V3 and Form is available
    if ('form.is_prose_fiction' in feats.columns
            and not args.no_prose_filter
            and args.content_version < 3):
        n_before = len(feats)
        feats = feats[feats['form.is_prose_fiction']]
        feats = feats.drop(columns=['form.is_prose_fiction'])
        print(f"Applied form.is_prose_fiction filter: {n_before} → {len(feats)}",
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
                       ('CROSS-TASK FEATURE PAIRS', 'feature')]:
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
