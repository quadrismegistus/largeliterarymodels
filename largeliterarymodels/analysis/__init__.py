"""largeliterarymodels.analysis — CH-backed cross-task discrimination analysis.

Reads passage annotations from ClickHouse (llmtasks.passage_annotations), joins
across tasks on (_id, scheme, seq), builds boolean feature matrices via
schema-introspection, and runs Fisher-exact discrimination with BH-FDR.

The statistical engine (fisher_tests + bh_fdr + group_matrix) is a temporary
stand-in here. When lltk publishes `lltk.analysis.stats`, the imports in
`stats.py` will swap to `from lltk.analysis.stats import ...` and the local
implementations get deleted.

Public API:
    from largeliterarymodels.analysis import (
        joint_feature_matrix, passage_groups,
        fisher_tests, bh_fdr,  # will be lltk.analysis.stats later
    )

Example:
    feats = joint_feature_matrix(
        tasks=['passage-content', 'passage-form'],
        task_versions={'passage-content': 3, 'passage-form': 1},
    )
    groups = passage_groups(feats.index, include_halfcent=True)
    results = fisher_tests(feats, groups)
    results['q_value'] = bh_fdr(results['p_value'])
"""

from .adapters import wide_to_features, classify_schema_fields
from .features import (
    build_feature_matrix,
    fit_partition_model,
    load_genre_extras,
    period_dummies,
    DEFAULT_ORDINAL_ENCODINGS,
)
from .groups import passage_groups
from .reader import joint_feature_matrix, load_task_annotations
from .registry import TASK_REGISTRY, register_task, resolve_task_class
from .reliability import (
    audit_disagrees_with_reference,
    flagged_for_audit,
    load_agent_annotations,
    majority_consensus,
    pairwise_agreement,
    per_field_trust,
    write_consensus,
)
from .propagate import (
    evaluate_classifiers, calibrate_thresholds, predict_all, write_propagated,
)
from .cross_language import compare_cross_language
from .embeddings import center_by_group, fetch_passage_embeddings, mean_pool_to_text
from .social_networks import (
    SocialNetwork,
    build_dialogue_graph,
    build_directed_graph,
    build_event_graph,
    build_graph,
    character_trajectories,
    compare,
    load_result,
    location_summary,
    network_metrics,
    plot_network,
    relation_type_counts,
    event_verb_counts,
)
from .stats import bh_fdr, fisher_tests, group_matrix

__all__ = [
    'joint_feature_matrix',
    'load_task_annotations',
    'passage_groups',
    'wide_to_features',
    'classify_schema_fields',
    'build_feature_matrix',
    'fit_partition_model',
    'load_genre_extras',
    'period_dummies',
    'DEFAULT_ORDINAL_ENCODINGS',
    'TASK_REGISTRY',
    'register_task',
    'resolve_task_class',
    'fisher_tests',
    'bh_fdr',
    'group_matrix',
    # propagation
    'evaluate_classifiers',
    'calibrate_thresholds',
    'predict_all',
    'write_propagated',
    # reliability / ensemble consensus
    'load_agent_annotations',
    'per_field_trust',
    'pairwise_agreement',
    'majority_consensus',
    'flagged_for_audit',
    'audit_disagrees_with_reference',
    'write_consensus',
    # cross-language comparison
    'compare_cross_language',
    # embeddings
    'fetch_passage_embeddings',
    'mean_pool_to_text',
    'center_by_group',
    # social networks
    'SocialNetwork',
    'build_graph',
    'build_directed_graph',
    'build_dialogue_graph',
    'build_event_graph',
    'compare',
    'load_result',
    'character_trajectories',
    'location_summary',
    'network_metrics',
    'plot_network',
    'relation_type_counts',
    'event_verb_counts',
]
