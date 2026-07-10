"""Tests for SettingsTab (issue #46) — model cache deletion UI + helpers.

Pure-Python helpers (`_format_size`, `_list_cached_models`, `_delete_files`,
`_wipe_cache_directory`) are tested directly. Widget methods are exercised
via AST-extracted method source bound to a MagicMock-backed stub, so the
real production code runs without a live Slicer/Qt environment.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).parent.parent
PRODUCTION_ROOT = ROOT / "ZebrafishEmbryoAnalyzer"
SETTINGS_PATH = PRODUCTION_ROOT / "ZebrafishEmbryoAnalyzerLib" / "settings_tab.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qt_modules(monkeypatch):
    """Install MagicMock shims for qt + slicer so settings_tab can be imported
    without a real Slicer / Qt environment."""
    qt_mock = MagicMock()
    qt_mock.Qt.AlignRight = 0x0001
    qt_mock.Qt.AlignVCenter = 0x0004
    qt_mock.Qt.ScrollBarAlwaysOff = 1
    qt_mock.QSizePolicy = MagicMock()
    qt_mock.QSizePolicy.Expanding = 1
    # QSignalBlocker(w) must be a callable that returns *something* (the
    # production code does `del blocker` afterward, so any object will do).
    # Using `MagicMock` (the class) directly would make calls like
    # `qt.QSignalBlocker(<MagicMock instance>)` raise `InvalidSpecError`
    # under Python 3.13 because spec would be a Mock object.
    qt_mock.QSignalBlocker = lambda *a, **k: MagicMock()
    monkeypatch.setitem(sys.modules, "qt", qt_mock)
    slicer_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "slicer", slicer_mock)
    return qt_mock


@pytest.fixture
def settings_module(qt_modules):
    import importlib
    import ZebrafishEmbryoAnalyzerLib.settings_tab as module
    return importlib.reload(module)


# ---------------------------------------------------------------------------
# AST method extraction (real production code bound to a stub at test time)
# ---------------------------------------------------------------------------

def _extract_method_source(name: str) -> str:
    tree = ast.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise RuntimeError(f"{name} not found in settings_tab.py")


def _bind_method(obj, name, namespace_extra=None):
    src = _extract_method_source(name)
    # The extracted methods may reference `qt` (e.g. qt.QSignalBlocker) as a
    # free variable — production code reads it via the module-level `import qt`.
    # Re-inject the MagicMock shim from sys.modules so exec'd source sees it.
    ns = {
        "qt": sys.modules.get("qt"),
    }
    if namespace_extra:
        ns.update(namespace_extra)
    exec(src, ns)
    setattr(obj, name, ns[name].__get__(obj, type(obj)))


def _bind_settings_tab_methods(obj, names):
    """Bind a set of SettingsTab methods (and their private helpers) onto a
    stub object. Walks the production class AST to find all private methods,
    binds them in dependency-friendly order, and then binds the requested
    public methods last so they overwrite any helpers they share a name with.
    """
    tree = ast.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SettingsTab":
            class_node = node
            break
    if class_node is None:
        raise RuntimeError("SettingsTab class not found")

    # First, bind all private helpers so public methods can find them.
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_") and node.name not in names:
            _bind_method(obj, node.name)
    # Then bind the public method(s) explicitly requested.
    for name in names:
        _bind_method(obj, name)


def _make_stub():
    """Bare object with the attributes SettingsTab methods touch. Avoids
    instantiating a real qt.QWidget (which fails under the MagicMock shim
    because QWidget's metaclass dance doesn't survive in our test rig).
    """
    class _Stub:
        pass
    obj = _Stub()
    # Pre-populate the widgets each method touches with MagicMocks so tests
    # can override just the few that matter for their assertion.
    obj._select_all = MagicMock()
    obj._btn_delete_selected = MagicMock()
    obj._btn_delete_all = MagicMock()
    obj._row_widgets = []
    return obj


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_format_size_zero_or_negative_is_missing(settings_module):
    assert settings_module._format_size(0) == "missing"
    assert settings_module._format_size(-1) == "missing"


def test_format_size_kb_and_mb(settings_module):
    assert settings_module._format_size(512).endswith("KB")
    assert settings_module._format_size(1024 * 1024 * 5).endswith("MB")


def test_list_cached_models_returns_one_row_per_entry(settings_module, tmp_path, monkeypatch):
    """Every manifest entry is listed; size is read from disk."""
    entries = {
        "a": {"id": "a", "label": "A label", "filename": "a.pth", "size_bytes": 100},
        "b": {"id": "b", "label": "B label", "filename": "b.pth", "size_bytes": 200},
        "c": {"id": "c", "label": "C label", "filename": "c.pth", "size_bytes": 300},
    }
    monkeypatch.setattr(settings_module, "collect_all_model_entries", lambda: entries)
    monkeypatch.setattr(
        settings_module, "get_cached_path", lambda e: tmp_path / e["filename"]
    )
    (tmp_path / "a.pth").write_bytes(b"x" * 1024)         # 1 KB
    (tmp_path / "b.pth").write_bytes(b"x" * (1024 * 1024 * 2))  # 2 MB

    rows = settings_module._list_cached_models()
    assert len(rows) == 3
    by_id = {r["id"]: r for r in rows}
    assert by_id["a"]["exists"] is True
    assert by_id["a"]["size_text"].endswith("KB")
    assert by_id["b"]["exists"] is True
    assert by_id["b"]["size_text"].endswith("MB")
    # Missing-on-disk rows still appear in the list (visibility matters).
    assert by_id["c"]["exists"] is False
    assert by_id["c"]["size_text"] == "missing"


def test_delete_files_unlinks_only_existing(settings_module, tmp_path):
    """`_delete_files` returns the count actually removed; missing paths are skipped."""
    p1 = tmp_path / "a.pth"
    p2 = tmp_path / "b.pth"
    p3 = tmp_path / "missing.pth"
    p1.write_bytes(b"x")
    p2.write_bytes(b"x")

    n = settings_module._delete_files([p1, p2, p3])
    assert n == 2
    assert not p1.exists()
    assert not p2.exists()


def test_delete_files_handles_directory_path_gracefully(settings_module, tmp_path):
    """Passing a directory path should not delete the directory itself."""
    d = tmp_path / "sub"
    d.mkdir()
    n = settings_module._delete_files([d])
    assert n == 0
    assert d.exists()


def test_wipe_cache_directory_removes_only_top_level_files(settings_module, tmp_path):
    """`_wipe_cache_directory` clears the top-level files; subdirs and their
    contents are intentionally left intact (sibling-tooling protection)."""
    (tmp_path / "a.pth").write_bytes(b"x")
    (tmp_path / "b.pth").write_bytes(b"x")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    nested = subdir / "c.pth"
    nested.write_bytes(b"x")

    n = settings_module._wipe_cache_directory(tmp_path)
    assert n == 2
    assert not (tmp_path / "a.pth").exists()
    assert not (tmp_path / "b.pth").exists()
    assert subdir.exists()
    assert nested.exists()


def test_wipe_cache_directory_missing_dir_is_safe(settings_module, tmp_path):
    """Calling wipe on a non-existent directory returns 0 (no exception)."""
    n = settings_module._wipe_cache_directory(tmp_path / "does-not-exist")
    assert n == 0


# ---------------------------------------------------------------------------
# Widget behaviour: AST-extracted methods on a stub
# ---------------------------------------------------------------------------

def test_on_select_all_toggled_checks_every_enabled_row(settings_module):
    obj = _make_stub()
    cb1 = MagicMock(); cb1.isEnabled.return_value = True
    cb2 = MagicMock(); cb2.isEnabled.return_value = True
    obj._row_widgets = [
        ("a", cb1, MagicMock(), True),
        ("b", cb2, MagicMock(), True),
    ]
    _bind_settings_tab_methods(obj, ["_on_select_all_toggled"])

    obj._on_select_all_toggled(True)

    assert cb1.setChecked.call_args.args == (True,)
    assert cb2.setChecked.call_args.args == (True,)


def test_on_select_all_toggled_skips_missing_rows(settings_module):
    """Missing-on-disk rows have cb.setEnabled(False) and must not be flipped."""
    obj = _make_stub()
    cb_existing = MagicMock(); cb_existing.isEnabled.return_value = True
    cb_missing = MagicMock(); cb_missing.isEnabled.return_value = False
    obj._row_widgets = [
        ("a", cb_existing, MagicMock(), True),
        ("b", cb_missing, MagicMock(), False),
    ]
    _bind_settings_tab_methods(obj, ["_on_select_all_toggled"])

    obj._on_select_all_toggled(True)

    cb_existing.setChecked.assert_called_once_with(True)
    cb_missing.setChecked.assert_not_called()


def test_update_button_labels_reflects_selection_count(settings_module):
    obj = _make_stub()
    btn_sel = MagicMock()
    btn_all = MagicMock()
    master = MagicMock()
    blocker_mock = MagicMock()
    obj._btn_delete_selected = btn_sel
    obj._btn_delete_all = btn_all
    obj._select_all = master
    # Two existing rows, one checked.
    cb1 = MagicMock(); cb1.isChecked.return_value = True
    cb2 = MagicMock(); cb2.isChecked.return_value = False
    obj._row_widgets = [
        ("a", cb1, MagicMock(), True),
        ("b", cb2, MagicMock(), True),
    ]
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "ZebrafishEmbryoAnalyzerLib.settings_tab.qt.QSignalBlocker",
        return_value=blocker_mock,
    ):
        _bind_settings_tab_methods(obj, ["_update_button_labels"])
        obj._update_button_labels()

    btn_sel.setText.assert_called_with("Delete selected models (1)")
    btn_sel.setEnabled.assert_called_with(True)
    btn_all.setEnabled.assert_called_with(True)
    master.setText.assert_called_with("Select all")  # not all selected → "Select all"


def test_update_button_labels_all_selected_flips_master_label(settings_module):
    obj = _make_stub()
    btn_sel = MagicMock()
    btn_all = MagicMock()
    master = MagicMock()
    blocker_mock = MagicMock()
    obj._btn_delete_selected = btn_sel
    obj._btn_delete_all = btn_all
    obj._select_all = master
    cb1 = MagicMock(); cb1.isChecked.return_value = True
    cb2 = MagicMock(); cb2.isChecked.return_value = True
    obj._row_widgets = [
        ("a", cb1, MagicMock(), True),
        ("b", cb2, MagicMock(), True),
    ]
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "ZebrafishEmbryoAnalyzerLib.settings_tab.qt.QSignalBlocker",
        return_value=blocker_mock,
    ):
        _bind_settings_tab_methods(obj, ["_update_button_labels"])
        obj._update_button_labels()

    master.setText.assert_called_with("Deselect all")
    master.setChecked.assert_called_with(True)


def test_set_actions_enabled_disables_all_when_false(settings_module):
    obj = _make_stub()
    btn_sel = MagicMock()
    btn_all = MagicMock()
    master = MagicMock()
    cb_existing = MagicMock(); cb_existing.isChecked.return_value = True  # selected
    obj._btn_delete_selected = btn_sel
    obj._btn_delete_all = btn_all
    obj._select_all = master
    obj._row_widgets = [
        ("a", cb_existing, MagicMock(), True),
    ]
    _bind_settings_tab_methods(obj, ["set_actions_enabled"])

    obj.set_actions_enabled(False)

    btn_sel.setEnabled.assert_called_with(False)
    btn_all.setEnabled.assert_called_with(False)
    master.setEnabled.assert_called_with(False)
    cb_existing.setEnabled.assert_called_with(False)


def test_set_actions_enabled_true_with_selection_keeps_button_enabled(settings_module):
    obj = _make_stub()
    btn_sel = MagicMock()
    btn_all = MagicMock()
    master = MagicMock()
    cb = MagicMock(); cb.isChecked.return_value = True
    obj._btn_delete_selected = btn_sel
    obj._btn_delete_all = btn_all
    obj._select_all = master
    obj._row_widgets = [
        ("a", cb, MagicMock(), True),
    ]
    _bind_settings_tab_methods(obj, ["set_actions_enabled"])

    obj.set_actions_enabled(True)

    btn_sel.setEnabled.assert_called_with(True)
    btn_all.setEnabled.assert_called_with(True)
    master.setEnabled.assert_called_with(True)
    cb.setEnabled.assert_called_with(True)
