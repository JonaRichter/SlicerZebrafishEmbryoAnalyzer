"""Tests for GalleryTab layout behaviour (issues #28, #16)."""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).parent.parent
PRODUCTION_ROOT = ROOT / "ZebrafishEmbryoAnalyzer"
GALLERY_PATH = PRODUCTION_ROOT / "ZebrafishEmbryoAnalyzerLib" / "gallery_tab.py"


def _extract_reflow_source():
    """Return the source of `_reflow` from gallery_tab.py as a string.

    We can't import GalleryTab as a real class (qt is a MagicMock in this
    test), so we extract the method body via AST and exec it against a
    stub instance. This keeps the test in lockstep with production without
    requiring Slicer / Qt.
    """
    tree = ast.parse(GALLERY_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_reflow":
            return ast.unparse(node)
    raise RuntimeError("_reflow not found in gallery_tab.py")


def _install_reflow_on_stub():
    """Define `_reflow` as a module-level function that operates on `self`."""
    from ZebrafishEmbryoAnalyzerLib.gallery_tab import THUMB_SIZE
    src = _extract_reflow_source()
    namespace = {"THUMB_SIZE": THUMB_SIZE}
    exec(src, namespace)
    return namespace["_reflow"]


class _StubGalleryTab:
    """Bare-bones stand-in for GalleryTab that exposes the same attributes
    _reflow touches but skips Qt's metaclass dance (which fails when qt is
    a MagicMock — QWidget's __init_subclass__/__new__ chokes)."""

    def __init__(self, cells, width):
        self._cells = list(cells)
        self.width = width
        self._n_cols = 0
        self._thumbnails = []
        grid_mock = MagicMock()
        grid_mock.spacing = 6
        self._grid = grid_mock


@pytest.fixture
def qt_modules(monkeypatch):
    """Install lightweight MagicMock shims for qt + slicer so the gallery
    module can be imported without a real Slicer install."""
    qt_mock = MagicMock()
    qt_mock.Qt.AlignCenter = 0
    qt_mock.Qt.ElideRight = 1
    monkeypatch.setitem(sys.modules, "qt", qt_mock)
    slicer_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "slicer", slicer_mock)
    return qt_mock


@pytest.fixture
def gallery_module(qt_modules):
    import importlib
    import ZebrafishEmbryoAnalyzerLib.gallery_tab as module
    return importlib.reload(module)


def _make_gallery(cells, width):
    """Build a stub GalleryTab-like object and bind the real `_reflow` method.

    We need the actual _reflow function (not a re-implementation) so we exercise
    exactly what production runs. The stub class skips the Qt metaclass.
    """
    g = _StubGalleryTab(cells, width)
    g._reflow = _install_reflow_on_stub().__get__(g, _StubGalleryTab)
    return g


def _stub_cells(n):
    """Return n MagicMock cell widgets."""
    from ZebrafishEmbryoAnalyzerLib.gallery_tab import THUMB_SIZE
    cells = []
    for _ in range(n):
        cell = MagicMock()
        cell.width = THUMB_SIZE + 4
        cells.append(cell)
    return cells


def test_reflow_sets_zero_column_stretch_to_left_align_thumbnails(gallery_module):
    """Issue #28: thumbnails should left-align, not spread across the row.

    When fewer images than columns are loaded, _reflow must explicitly set
    column stretch factors to 0 so columns size to content and any remaining
    horizontal space accumulates on the right side of the row instead of
    being distributed evenly between cells.
    """
    g = _make_gallery(_stub_cells(3), width=2000)
    g._reflow()
    # 3 cells in 2000px width with THUMB_SIZE=150 + spacing=6 → 2000 // 156 = 12 cols
    cols = g._n_cols
    assert cols > 3, f"test fixture expects more columns than cells, got cols={cols}"

    # setColumnStretch must be called for every column 0..cols-1 with stretch=0
    calls = g._grid.setColumnStretch.call_args_list
    assert len(calls) == cols, (
        f"setColumnStretch must be called once per column (cols={cols}), "
        f"got {len(calls)} calls"
    )
    columns_set = []
    stretches = []
    for call in calls:
        col = call.args[0]
        stretch = call.args[1] if len(call.args) > 1 else call.kwargs.get("stretch")
        columns_set.append(col)
        stretches.append(stretch)
    assert sorted(columns_set) == list(range(cols))
    assert all(s == 0 for s in stretches), (
        f"All column stretches must be 0 (left-align), got {stretches}"
    )


def test_reflow_left_aligns_with_many_columns_few_cells(gallery_module):
    """Tighter regression for issue #28: 3 cells, very wide container."""
    g = _make_gallery(_stub_cells(3), width=3000)
    g._reflow()
    assert g._n_cols >= 10, "fixture sanity: many more columns than cells"

    call_columns = [c.args[0] for c in g._grid.setColumnStretch.call_args_list]
    assert call_columns == list(range(g._n_cols)), (
        f"setColumnStretch must be called in column order, got {call_columns}"
    )
    for call in g._grid.setColumnStretch.call_args_list:
        stretch = call.args[1] if len(call.args) > 1 else call.kwargs.get("stretch")
        assert stretch == 0, f"every column stretch must be 0, got {stretch}"


def test_reflow_skipped_when_column_count_unchanged(gallery_module):
    """Smoke: same-column-count reflow should not redo layout work.

    _reflow has an early-return when column count is unchanged. This test
    guards that behaviour so the column-stretch loop in issue #28 doesn't
    accidentally re-run on every resize tick.
    """
    g = _make_gallery(_stub_cells(3), width=2000)
    g._reflow()  # first call populates n_cols
    g._grid.reset_mock()
    g._reflow()  # second call: should early-return
    g._grid.setColumnStretch.assert_not_called()
    g._grid.addWidget.assert_not_called()


def test_reflow_does_not_call_set_row_stretch(gallery_module):
    """Issue #16: setRowStretch on an empty row caused uneven vertical spacing.

    The previous implementation called `setRowStretch(rows, 1)` on a
    notional empty row to "push content up" — but this interacted
    inconsistently with rows of different heights (single-line vs
    multi-line captions). QGridLayout positions content at the top of
    each row's allotted space by default, so the explicit stretch is
    unnecessary.

    Regression guard: _reflow must never call setRowStretch.
    """
    for n_cells in (0, 1, 3, 5, 12):
        g = _make_gallery(_stub_cells(n_cells), width=2000)
        g._reflow()
        g._grid.setRowStretch.assert_not_called()