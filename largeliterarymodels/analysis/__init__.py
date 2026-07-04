"""largeliterarymodels.analysis — cross-task discrimination analysis.

Schema-aware feature extraction, Fisher-exact discrimination with BH-FDR,
ensemble consensus, and social network analysis.

Public API:
    from largeliterarymodels.analysis import (
        joint_feature_matrix, passage_groups,
        fisher_tests, bh_fdr,
    )

Example:
    feats = joint_feature_matrix(
        tasks=['passage-content', 'passage-form'],
        task_versions={'passage-content': 3, 'passage-form': 1},
    )
    groups = passage_groups(feats.index, include_halfcent=True)
    results = fisher_tests(feats, groups)
    results['q_value'] = bh_fdr(results['p_value'])

Submodules are lazy-loaded (PEP 562, same pattern as
largeliterarymodels.tasks) so `import largeliterarymodels.analysis`
doesn't eagerly pull pandas/numpy/clickhouse into every process.
"""

import importlib

_LAZY_IMPORTS = {
    # adapters
    'wide_to_features': '.adapters',
    'classify_schema_fields': '.adapters',
    # features
    'build_feature_matrix': '.features',
    'fit_partition_model': '.features',
    'load_genre_extras': '.features',
    'period_dummies': '.features',
    'DEFAULT_ORDINAL_ENCODINGS': '.features',
    # groups
    'passage_groups': '.groups',
    # reader
    'joint_feature_matrix': '.reader',
    'load_task_annotations': '.reader',
    # registry
    'TASK_REGISTRY': '.registry',
    'register_task': '.registry',
    'resolve_task_class': '.registry',
    # reliability / ensemble consensus
    'audit_disagrees_with_reference': '.reliability',
    'flagged_for_audit': '.reliability',
    'load_agent_annotations': '.reliability',
    'majority_consensus': '.reliability',
    'pairwise_agreement': '.reliability',
    'per_field_trust': '.reliability',
    'write_consensus': '.reliability',
    # propagation
    'evaluate_classifiers': '.propagate',
    'calibrate_thresholds': '.propagate',
    'predict_all': '.propagate',
    'write_propagated': '.propagate',
    # cross-language comparison
    'compare_cross_language': '.cross_language',
    # embeddings
    'center_by_group': '.embeddings',
    'fetch_passage_embeddings': '.embeddings',
    'mean_pool_to_text': '.embeddings',
    # social networks
    'SocialNetwork': '.social_networks',
    'build_dialogue_graph': '.social_networks',
    'build_directed_graph': '.social_networks',
    'build_event_graph': '.social_networks',
    'build_graph': '.social_networks',
    'character_trajectories': '.social_networks',
    'compare': '.social_networks',
    'load_result': '.social_networks',
    'location_summary': '.social_networks',
    'network_metrics': '.social_networks',
    'plot_network': '.social_networks',
    'relation_type_counts': '.social_networks',
    'event_verb_counts': '.social_networks',
    # stats
    'bh_fdr': '.stats',
    'fisher_tests': '.stats',
    'group_matrix': '.stats',
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
