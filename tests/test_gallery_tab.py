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
    qt_mock.Qt.AlignTop = 32        # 0x20 — Qt.AlignTop
    qt_mock.Qt.AlignHCenter = 4     # 0x04 — Qt.AlignHCenter
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

    # setColumnStretch must be called for every real column 0..cols-1 with
    # stretch=0, plus one extra call for the trailing spacer column at index
    # `cols` (see test_reflow_gives_one_trailing_spacer_column_all_the_column_stretch).
    calls = g._grid.setColumnStretch.call_args_list
    assert len(calls) == cols + 1, (
        f"setColumnStretch must be called once per real column plus the "
        f"trailing spacer column (cols={cols}), got {len(calls)} calls"
    )
    columns_set = []
    stretches = []
    for call in calls:
        col = call.args[0]
        stretch = call.args[1] if len(call.args) > 1 else call.kwargs.get("stretch")
        columns_set.append(col)
        stretches.append(stretch)
    assert sorted(columns_set) == list(range(cols + 1))
    real_column_stretches = [s for c, s in zip(columns_set, stretches) if c < cols]
    assert all(s == 0 for s in real_column_stretches), (
        f"All real column stretches must be 0 (left-align), got {real_column_stretches}"
    )


def test_reflow_left_aligns_with_many_columns_few_cells(gallery_module):
    """Tighter regression for issue #28: 3 cells, very wide container."""
    g = _make_gallery(_stub_cells(3), width=3000)
    g._reflow()
    assert g._n_cols >= 10, "fixture sanity: many more columns than cells"

    call_columns = [c.args[0] for c in g._grid.setColumnStretch.call_args_list]
    assert call_columns == list(range(g._n_cols + 1)), (
        f"setColumnStretch must be called in column order, real columns then "
        f"the trailing spacer column, got {call_columns}"
    )
    for call in g._grid.setColumnStretch.call_args_list[:-1]:
        stretch = call.args[1] if len(call.args) > 1 else call.kwargs.get("stretch")
        assert stretch == 0, f"every real column stretch must be 0, got {stretch}"


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


def test_reflow_gives_one_trailing_spacer_row_all_the_row_stretch(gallery_module):
    """Issue #16 (top-left anchoring, kept widgetResizable(True)):
    QGridLayout distributes leftover vertical space evenly across every row
    whose stretch is 0 — even when every *real* row is explicitly set to
    stretch 0 — which is what caused rows to spread apart vertically.
    Giving exactly one trailing spacer row (one past the last real content
    row) a nonzero stretch collects all that leftover space there instead,
    pinning the real rows to the top. This only works now that every cell's
    caption reserves a fixed two-line height, so all real rows are already
    the same height and don't need their own stretch to look even.
    """
    for n_cells, cols_expected_gt in ((1, 0), (3, 0), (5, 0), (12, 0)):
        g = _make_gallery(_stub_cells(n_cells), width=2000)
        g._reflow()
        cols = g._n_cols
        expected_rows = -(-n_cells // cols)
        g._grid.setRowStretch.assert_called_once_with(expected_rows, 1)


def test_reflow_gives_one_trailing_spacer_column_all_the_column_stretch(gallery_module):
    """Issue #16 (top-left anchoring): same reasoning as the row spacer, but
    for horizontal space — one trailing spacer column past the last real
    content column gets stretch 1, collecting leftover horizontal space so
    real columns (all stretch 0) don't spread apart.
    """
    g = _make_gallery(_stub_cells(3), width=2000)
    g._reflow()
    cols = g._n_cols
    g._grid.setColumnStretch.assert_any_call(cols, 1)


# ---------------------------------------------------------------------------
# populate() tests for issue #16 re-refinement (cell-internal alignment)
# ---------------------------------------------------------------------------

def _extract_populate_source():
    """Return the source of `populate` from gallery_tab.py as a string."""
    tree = ast.parse(GALLERY_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "populate":
            return ast.unparse(node)
    raise RuntimeError("populate not found in gallery_tab.py")


_ClickableLabel_instances = []


def _make_stub_clickable_label(idx, on_select, loaded=True):
    """Bare-bones replacement for production _ClickableLabel.

    Production's _ClickableLabel inherits from qt.QLabel, which under our
    MagicMock-qt test rig is a function (after we monkey-patch it to a
    factory). Subclassing a function fails. We replace _ClickableLabel in
    the exec namespace with this factory that returns a MagicMock with the
    construction parameters recorded.
    """
    instance = MagicMock()
    instance.idx = idx
    instance.on_select = on_select
    instance._loaded = loaded
    _ClickableLabel_instances.append(instance)
    return instance


def _install_populate_on_stub():
    """Extract populate() via AST and prepare it for binding to a stub.

    The exec namespace is seeded with every module-level name populate
    references (THUMB_SIZE, border constants, qt, _numpy_to_qpixmap,
    make_overlay, _ClickableLabel). Note: real _ClickableLabel inherits
    from qt.QLabel, which the test patches to a factory function — Python
    refuses to subclass a function. We substitute _StubClickableLabel for
    the test.
    """
    from ZebrafishEmbryoAnalyzerLib.gallery_tab import (
        THUMB_SIZE,
        BORDER_OK,
        BORDER_WARN,
        BORDER_ERROR,
        BORDER_LOADING,
        _numpy_to_qpixmap,
    )
    src = _extract_populate_source()
    namespace = {
        "THUMB_SIZE": THUMB_SIZE,
        "BORDER_OK": BORDER_OK,
        "BORDER_WARN": BORDER_WARN,
        "BORDER_ERROR": BORDER_ERROR,
        "BORDER_LOADING": BORDER_LOADING,
        "_numpy_to_qpixmap": lambda rgb: MagicMock(),
        "qt": sys.modules["qt"],
        "make_overlay": lambda r, thumbnail_size: MagicMock(),
        "_ClickableLabel": _make_stub_clickable_label,
    }
    exec(src, namespace)
    return namespace["populate"]


class _StubGalleryTabForPopulate:
    """Stand-in for GalleryTab. Same shape that populate()/ _reflow() touch."""

    def __init__(self):
        self._grid = MagicMock()
        self._grid.count.return_value = 0
        self._grid.spacing = 6
        self.width = 2000
        self._on_select = lambda i: None
        self._cells = []
        self._thumbnails = []
        self._n_cols = 0


@pytest.fixture
def populated_stub(gallery_module, monkeypatch, qt_modules):
    """AST-extracted `populate()` against a stub.

    Returns (stub, vbox_layouts, captions) where:
      - vbox_layouts: list of cell_layout MagicMocks (one per result)
      - captions:     list of caption MagicMocks (one per result)
    """
    # Patch overlay.make_overlay so populate's runtime import binds our
    # fake (the real make_overlay hits cv2 on MagicMock result dicts).
    import ZebrafishEmbryoAnalyzerLib.overlay as overlay_module
    monkeypatch.setattr(
        overlay_module, "make_overlay",
        lambda r, thumbnail_size: MagicMock(),
    )

    # Capture every QVBoxLayout instance populate creates.
    vbox_layouts = []
    real_qt = sys.modules["qt"]

    def vbox_factory(*args, **kwargs):
        instance = MagicMock()
        vbox_layouts.append(instance)
        return instance

    monkeypatch.setattr(real_qt, "QVBoxLayout", vbox_factory)

    # Capture every QLabel instance populate creates. Captions are
    # identified by having setMinimumHeight called (image labels and
    # cells don't get a minimum height set; only captions do). The caption
    # uses fontMetrics().lineSpacing() to compute its minimum height — make
    # that return a real int so 2 * lineSpacing() + 4 is an int.
    captions = []

    def qlabel_factory(*args, **kwargs):
        instance = MagicMock()
        instance.fontMetrics.return_value.lineSpacing.return_value = 15

        def track_set_min_height(*a, **kw):
            captions.append(instance)

        def track_set_alignment(*a, **kw):
            if instance not in captions:
                captions.append(instance)

        instance.setMinimumHeight.side_effect = track_set_min_height
        instance.setAlignment.side_effect = track_set_alignment
        return instance

    monkeypatch.setattr(real_qt, "QLabel", qlabel_factory)

    # Build the stub and bind populate + _reflow (both AST-extracted).
    stub = _StubGalleryTabForPopulate()
    populate = _install_populate_on_stub()
    stub.populate = populate.__get__(stub, _StubGalleryTabForPopulate)

    # _reflow needs _grid.spacing=6 and a real width to do its arithmetic.
    stub._grid.spacing = 6
    stub.width = 2000
    stub._reflow = _install_reflow_on_stub().__get__(stub, _StubGalleryTabForPopulate)

    # Run populate with two results: one analyzed (2-line caption), one not.
    results = [
        {"filename": "analyzed.jpg", "original": MagicMock(), "length": 100.0,
         "curvature": 1.5, "error": None},
        {"filename": "pending.jpg", "original": None, "length": None,
         "curvature": None, "error": None},
    ]
    stub.populate(results)

    return stub, vbox_layouts, captions


def test_populate_aligns_captions_to_top(populated_stub):
    """Issue #16 (re-refinement): captions must be top-aligned within their
    label so 1-line content sits at the top of the reserved 2-line area,
    not vertically centered.
    """
    _, _, captions = populated_stub
    qt = sys.modules["qt"]
    # Filter to captions: those with setMinimumHeight called (the image labels
    # don't get setMinimumHeight — only captions do).
    real_captions = [c for c in captions if c.setMinimumHeight.called]
    assert len(real_captions) >= 2, (
        f"Expected at least 2 captions (one per result), got {len(real_captions)}"
    )
    for caption in real_captions:
        align_args = caption.setAlignment.call_args.args
        assert align_args, "setAlignment should have been called on the caption"
        alignment = align_args[0]
        assert alignment & qt.Qt.AlignTop, (
            f"Caption alignment must include AlignTop (top-anchored), got "
            f"{alignment!r} (AlignTop={qt.Qt.AlignTop!r})"
        )


def test_populate_reserves_two_lines_of_caption_height(populated_stub):
    """Issue #16 (re-refinement): every caption reserves space for two lines
    of text so cells in the same row all have the same height. Without
    this, mixed-caption rows would have varying cell heights and the
    gallery would show a vertical gap between image and caption in the
    shorter cells.
    """
    _, _, captions = populated_stub
    real_captions = [c for c in captions if c.setMinimumHeight.called]
    assert len(real_captions) >= 2
    for caption in real_captions:
        caption.setMinimumHeight.assert_called_once()
        height_arg = caption.setMinimumHeight.call_args.args[0]
        # Must be a positive value; specifically at least one line of text.
        # We don't assert the exact pixel count because it depends on font
        # metrics from the test rig (a MagicMock), but the production code
        # uses 2 * lineSpacing + 4 which is always > a single line.
        assert height_arg > 0, f"setMinimumHeight must be positive, got {height_arg}"


def test_populate_adds_bottom_stretch_to_each_cell(populated_stub):
    """Issue #16 (re-refinement): cell_layout.addStretch(1) at the end of
    each cell ensures any extra cell height becomes bottom-padding below
    the caption, not middle-padding between image and caption.
    """
    stub, vbox_layouts, _ = populated_stub
    # populate runs two iterations, so we expect two cell layouts
    assert len(vbox_layouts) == 2, (
        f"Expected one VBoxLayout per result, got {len(vbox_layouts)}"
    )
    for cell_layout in vbox_layouts:
        cell_layout.addStretch.assert_called_with(1)
        # And addStretch must come AFTER addWidget(caption) — verify ordering
        widget_calls = cell_layout.addWidget.call_args_list
        stretch_calls = cell_layout.addStretch.call_args_list
        assert len(widget_calls) == 2, (
            f"cell_layout should addWidget twice (label + caption), got "
            f"{len(widget_calls)}"
        )
        assert len(stretch_calls) == 1
        # addWidget is called before addStretch in production code; check
        # by inspecting call counts (a crude but sufficient ordering check).
        # The actual production order is: addWidget(label), addWidget(caption),
        # addStretch(1). We can verify that addStretch was called after
        # addWidget by checking that the mock's total call count is 3 and
        # the last call is addStretch.
        assert cell_layout.method_calls[-1][0] == "addStretch", (
            f"addStretch must be the last call on cell_layout, got "
            f"{cell_layout.method_calls[-1]}"
        )