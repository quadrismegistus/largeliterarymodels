"""Re-exports of `lltk.analysis.stats` for callers that import from
`largeliterarymodels.analysis`.

Historical note: until 2026-04-21 this file held stand-in implementations
of fisher_tests / bh_fdr / group_matrix. Those moved to lltk (commit
89ce631 on clickhouse-migration). This module is now a passthrough.
"""

from lltk.analysis.stats import bh_fdr, fisher_tests, group_matrix


__all__ = ['fisher_tests', 'bh_fdr', 'group_matrix']
