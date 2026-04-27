"""Shared ClickHouse client factory for standalone use.

When called from lltk, callers pass `client=lltk.db.client` explicitly.
This fallback is for standalone scripts and tests without lltk installed.
"""


def _default_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host='localhost', port=8123, username='lltk', password='lltk',
    )
