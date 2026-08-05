"""Data-root resolution tests.

The defect these pin (lacan seat's field report, against 78fd632):
STASH_PATH derived from __file__, so a non-editable pip install pointed
the stash, the batch ledger and the raw sidecars into site-packages —
the run reported success, certify() said complete, and the next
--force-reinstall deleted the lot, ledger included. The volume a
coverage check certifies must not be disposable.
"""

import logging
import os

import largeliterarymodels.batch as B
import largeliterarymodels.llm as L
import largeliterarymodels.rawlog as R
from largeliterarymodels.llm import _data_dir


def test_env_override_always_wins(tmp_path):
    root = tmp_path / "clone"
    (root / "data").mkdir(parents=True)
    (root / "data" / "x").write_text("occupied")
    assert _data_dir(env="~/myproj/data", pkg_parent=str(root)) == \
        os.path.expanduser("~/myproj/data"), \
        "LITMOD_DATA_DIR beats even a populated clone data/"


def test_clone_with_data_keeps_its_root(tmp_path):
    """The 9 GB case: an existing repo's data/ stays exactly where it
    is — upgrading must not silently orphan a clone's history."""
    root = tmp_path / "clone"
    (root / "data").mkdir(parents=True)
    (root / "data" / "stash").mkdir()
    assert _data_dir(env="", pkg_parent=str(root)) == str(root / "data")


def test_fresh_install_defaults_to_home(tmp_path):
    root = tmp_path / "clone"
    root.mkdir()
    assert _data_dir(env="", pkg_parent=str(root)) == os.path.join(
        os.path.expanduser("~"), ".largeliterarymodels", "data"), \
        "no data dir at all -> the durable default, not a repo-relative one"


def test_site_packages_data_is_never_used(tmp_path, caplog):
    """pip owns site-packages: data there is disposable by construction
    and must be refused even when NON-EMPTY — with a warning naming
    both paths, because refusing silently would orphan what an earlier
    run of the broken derivation already wrote there."""
    sp = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    (sp / "data" / "stash").mkdir(parents=True)
    (sp / "data" / "stash" / "entry").write_text("paid annotation")
    with caplog.at_level(logging.WARNING):
        got = _data_dir(env="", pkg_parent=str(sp))
    assert got == os.path.join(os.path.expanduser("~"),
                               ".largeliterarymodels", "data")
    assert "site-packages" in caplog.text and "copy it out" in caplog.text

    caplog.clear()
    sp2 = tmp_path / "v2" / "lib" / "python3.12" / "site-packages"
    sp2.mkdir(parents=True)  # empty: nothing to warn about
    with caplog.at_level(logging.WARNING):
        got = _data_dir(env="", pkg_parent=str(sp2))
    assert got.endswith(os.path.join(".largeliterarymodels", "data"))
    assert "site-packages" not in caplog.text


def test_every_persistent_store_derives_from_one_root():
    """STASH_PATH decides where money-backed artifacts live; the
    ledger, the sidecar default and the usage logs must all be its
    siblings — a second independent derivation would reintroduce the
    split-brain this fix removes."""
    data_root = os.path.dirname(L.STASH_PATH)
    assert B.LEDGER_DIR == os.path.join(data_root, "batch_ledger")
    assert R._default_root() == os.path.join(data_root, "raw_responses")
    import inspect
    task_src = inspect.getsource(
        __import__("largeliterarymodels.task", fromlist=["task"]))
    assert 'os.path.dirname(STASH_PATH), "usage_logs"' in task_src
