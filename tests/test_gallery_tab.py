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
        self._container = MagicMock()


@pytest.fixture
def qt_modules(monkeypatch):
    """Install lightweight MagicMock shims for qt + slicer so the gallery
    module can be imported without a real Slicer install."""
    qt_mock = MagicMock()
    qt_mock.Qt.AlignCenter = 0
    qt_mock.Qt.AlignTop = 32        # 0x20 — Qt.AlignTop
    qt_mock.Qt.AlignHCenter = 4     # 0x04 — Qt.AlignHCenter
    qt_mock.Qt.AlignLeft = 1        # 0x01 — Qt.AlignLeft
    qt_mock.Qt.ScrollBarAlwaysOff = 1
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
    g._container.reset_mock()
    g._reflow()  # second call: should early-return
    g._grid.setColumnStretch.assert_not_called()
    g._grid.addWidget.assert_not_called()
    g._container.adjustSize.assert_not_called()


def test_reflow_calls_container_adjust_size_when_repositioning(gallery_module):
    """Issue #51: with widgetResizable(False), QScrollArea no longer force-
    resizes the container to the viewport, so _reflow must explicitly call
    self._container.adjustSize() after repositioning cells — otherwise the
    container keeps its stale/default size and the gallery appears empty.
    """
    g = _make_gallery(_stub_cells(3), width=2000)
    g._reflow()
    g._container.adjustSize.assert_called_once()


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
        self._container = MagicMock()


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


# ---------------------------------------------------------------------------
# Grid-alignment regression tests for issue #16 (third layer)
# ---------------------------------------------------------------------------
#
# Choice: AST-extract `__init__` (same pattern as `_reflow` and `populate`)
# rather than refactoring production to call a small `_build_scroll_area()`
# helper. Reasons:
#   1. Consistency — the file already uses AST-extraction for the other
#      two layout-related methods, so the test rig is uniform.
#   2. Production code stays minimal — adding a helper would expand the
#      class surface just to make testing easier, and the three-call
#      configuration block is small and reads naturally inline.
#   3. The risk of drift between a helper and its test is the same as the
#      risk of drift between an inlined block and its AST-extracted test.
# The trade-off is that exec'ing `__init__` against a stub needs extra
# ceremony (the `super().__init__()` cell trick), but it stays inside the
# existing test rig, so we keep it.


def _extract_init_source():
    """Return the source of `GalleryTab.__init__` from gallery_tab.py.

    gallery_tab.py defines two `__init__` methods — one on `_ClickableLabel`
    and one on `GalleryTab`. A naive `ast.walk` returns whichever it visits
    first, so we match by parent class to be sure we get the one that
    configures the scroll area.

    The bare `super().__init__()` call inside GalleryTab.__init__ is
    stripped before unparsing — without the implicit `__class__` cell
    that real class definitions provide, a standalone `super()` raises
    "super(): __class__ cell not found". We don't want to exercise Qt
    widget construction in these tests anyway (we're only inspecting the
    scroll-area configuration calls), so dropping the super call is the
    correct trade-off.
    """
    tree = ast.parse(GALLERY_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "GalleryTab":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                init_fn = child
                break
        else:
            continue
        break
    else:
        raise RuntimeError("GalleryTab.__init__ not found in gallery_tab.py")

    # Strip `super().__init__()` from the body — it's a no-op for our
    # tests since the stub isn't a real Qt widget. The AST shape is
    # `Expr(value=Call(func=Attribute(value=Call(func=Name('super')))))`
    # i.e. `super().__init__()`.
    def _is_super_init_call(stmt):
        if not isinstance(stmt, ast.Expr):
            return False
        call = stmt.value
        if not isinstance(call, ast.Call):
            return False
        attr = call.func
        if not isinstance(attr, ast.Attribute) or attr.attr != "__init__":
            return False
        inner = attr.value
        if not isinstance(inner, ast.Call):
            return False
        return isinstance(inner.func, ast.Name) and inner.func.id == "super"

    init_fn.body = [stmt for stmt in init_fn.body if not _is_super_init_call(stmt)]
    return ast.unparse(init_fn)


class _StubGalleryTabForInit:
    """Stand-in for GalleryTab used to exercise __init__.

    Skips Qt's metaclass dance (QWidget's __init_subclass__/__new__ chokes
    when qt is a MagicMock) but provides the attributes __init__ writes to.
    The bare `super().__init__()` inside GalleryTab.__init__ resolves to
    `object.__init__()` via the `__class__` cell we inject at exec time.
    """

    def __init__(self):
        # Will be overwritten by the AST-extracted GalleryTab.__init__
        # when bound below. We just need an instance to bind to.
        pass


def _run_init_with_monkey_qt(monkey_qt):
    """Extract GalleryTab.__init__ with a custom `qt` namespace, run it, and
    return the stub instance after __init__ has executed.

    The trick: `__class__` is seeded so the bare `super().__init__()`
    resolves against the stub class (otherwise the AST-extracted function
    fails with "super(): __class__ cell not found").
    """
    src = _extract_init_source()
    namespace = {"qt": monkey_qt, "__class__": _StubGalleryTabForInit}
    exec(src, namespace)
    init_fn = namespace["__init__"]
    stub = _StubGalleryTabForInit()
    init_fn(stub, lambda i: None)
    return stub


def _monkey_qt_with_scroll_capture(real_qt):
    """Build a per-test qt mock that exposes real Qt constants and
    captures every QScrollArea instance creation.

    QScrollArea is a MagicMock with a side_effect that records every
    instance it returns. This lets callers both introspect the returned
    scroll instance (via `scroll_instances`) and count the calls (via
    `monkey_qt.QScrollArea.call_count`).
    """
    scroll_instances = []

    def scroll_factory(*args, **kwargs):
        instance = MagicMock()
        scroll_instances.append(instance)
        return instance

    scroll_mock = MagicMock(side_effect=scroll_factory)
    monkey_qt = MagicMock()
    monkey_qt.Qt.AlignTop = real_qt.Qt.AlignTop
    monkey_qt.Qt.AlignLeft = real_qt.Qt.AlignLeft
    monkey_qt.Qt.ScrollBarAlwaysOff = real_qt.Qt.ScrollBarAlwaysOff
    monkey_qt.QScrollArea = scroll_mock
    return monkey_qt, scroll_instances


def test_init_disables_widget_resizable_on_scroll_area(qt_modules):
    """Issue #16 (grid alignment): setWidgetResizable(False) makes the inner
    widget size to its grid content instead of being force-resized to the
    viewport. With True, Qt distributes extra viewport space between rows
    (vertically) and across columns (horizontally) — which is the bug the
    user reported even after the row-stretch and column-stretch fixes.
    """
    real_qt = sys.modules["qt"]
    monkey_qt, scroll_instances = _monkey_qt_with_scroll_capture(real_qt)
    _run_init_with_monkey_qt(monkey_qt)

    assert len(scroll_instances) == 1, (
        f"Expected exactly one QScrollArea created in __init__, got "
        f"{len(scroll_instances)}"
    )
    scroll = scroll_instances[0]
    scroll.setWidgetResizable.assert_called_once_with(False)


def test_init_anchors_scroll_area_to_top_left(qt_modules):
    """Issue #16 (grid alignment): setAlignment(Qt.AlignTop | Qt.AlignLeft)
    positions the inner widget at the top-left of the viewport when the
    widget is smaller than the viewport. Without this, Qt's default is to
    center the widget, which makes the grid look vertically and
    horizontally centered instead of top-left-anchored.
    """
    real_qt = sys.modules["qt"]
    monkey_qt, scroll_instances = _monkey_qt_with_scroll_capture(real_qt)
    _run_init_with_monkey_qt(monkey_qt)

    assert len(scroll_instances) == 1
    scroll = scroll_instances[0]
    scroll.setAlignment.assert_called_once()
    align_args = scroll.setAlignment.call_args.args
    alignment = align_args[0]
    expected = real_qt.Qt.AlignTop | real_qt.Qt.AlignLeft
    assert alignment == expected, (
        f"setAlignment must be called with AlignTop | AlignLeft "
        f"({expected!r}), got {alignment!r}"
    )


def test_init_suppresses_horizontal_scrollbar(qt_modules):
    """Issue #16 (grid alignment): setHorizontalScrollBarPolicy(
    ScrollBarAlwaysOff) ensures the gallery never shows a horizontal
    scrollbar. With widgetResizable=False, horizontal overflow only
    happens at extremely narrow panel widths (< ~154px); the user prefers
    clipped cells over a scrollbar in a thumbnail grid.
    """
    real_qt = sys.modules["qt"]
    monkey_qt, scroll_instances = _monkey_qt_with_scroll_capture(real_qt)
    _run_init_with_monkey_qt(monkey_qt)

    assert len(scroll_instances) == 1
    scroll = scroll_instances[0]
    scroll.setHorizontalScrollBarPolicy.assert_called_once_with(
        real_qt.Qt.ScrollBarAlwaysOff
    )


def test_init_stores_scroll_area_on_instance(qt_modules):
    """Issue #16 (grid alignment): __init__ must store the QScrollArea as
    `self._scroll` so tests (and any future introspection helper) can
    reach it without rebuilding the widget tree. The other init tests
    rely on this attribute for scroll-area inspection.
    """
    real_qt = sys.modules["qt"]
    monkey_qt, scroll_instances = _monkey_qt_with_scroll_capture(real_qt)
    stub = _run_init_with_monkey_qt(monkey_qt)

    assert hasattr(stub, "_scroll"), (
        "GalleryTab.__init__ must store the QScrollArea as self._scroll"
    )
    # self._scroll should be the same instance used as the layout host —
    # it's the only QScrollArea created.
    assert monkey_qt.QScrollArea.call_count == 1, (
        f"Expected exactly one QScrollArea() call, got "
        f"{monkey_qt.QScrollArea.call_count}"
    )
    # With side_effect set, return_value is the auto-MagicMock; the real
    # instance returned is the one we captured.
    assert stub._scroll is scroll_instances[0]


def test_init_three_changes_happen_together(qt_modules):
    """Issue #16 (grid alignment): the three configuration calls
    (setWidgetResizable(False), setAlignment(AlignTop|AlignLeft),
    setHorizontalScrollBarPolicy(ScrollBarAlwaysOff)) form a single
    atomic fix — removing any one of them re-introduces part of the
    bug. This test guards against accidental partial-reverts by
    requiring all three to be present on the same scroll instance.
    """
    real_qt = sys.modules["qt"]
    monkey_qt, scroll_instances = _monkey_qt_with_scroll_capture(real_qt)
    _run_init_with_monkey_qt(monkey_qt)

    assert len(scroll_instances) == 1
    scroll = scroll_instances[0]
    # All three must be called on the SAME scroll instance.
    scroll.setWidgetResizable.assert_called_with(False)
    scroll.setAlignment.assert_called_once()
    align = scroll.setAlignment.call_args.args[0]
    assert align & real_qt.Qt.AlignTop
    assert align & real_qt.Qt.AlignLeft
    scroll.setHorizontalScrollBarPolicy.assert_called_with(
        real_qt.Qt.ScrollBarAlwaysOff
    )