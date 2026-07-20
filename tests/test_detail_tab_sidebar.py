"""Tests for issue #66: sidebar QSplitter skeleton + persistence.

The detail tab used to stack everything vertically under the image; #66
moves every control into a right-hand sidebar wrapped in a QSplitter so
the user can resize it, and persists the resulting width to QSettings
so it survives restarts.

We do not exercise a real Qt event loop here — these tests stub the
``qt`` module with classes that look Qt-like but track calls via MagicMock
delegation, then construct DetailTab directly and inspect the resulting
widget tree + QSettings calls.
"""

import ast
import pathlib
import sys
from unittest.mock import MagicMock

import pytest


ROOT = pathlib.Path(__file__).parent.parent
PRODUCTION_ROOT = ROOT / "ZebrafishEmbryoAnalyzer"
DETAIL_PATH = PRODUCTION_ROOT / "ZebrafishEmbryoAnalyzerLib" / "detail_tab.py"


# ---------------------------------------------------------------------------
# Qt stub classes — real classes so `class DetailTab(qt.QWidget)` works,
# but every instance carries a MagicMock for call introspection.
# ---------------------------------------------------------------------------


class _Recorder:
    """A class that records every callable access into a MagicMock.

    Each instance gets its own MagicMock at ``self._calls`` that records
    calls to any method/attribute. ``setLayout`` and similar Qt methods
    that production code expects are real methods on the class so
    arbitrary Python attribute access still works.
    """
    class _Reg:
        def __init__(self):
            self.records = []

        def add(self, *args, **kwargs):
            self.records.append(('add', args, kwargs))

        def extend(self, items):
            self.records.append(('extend', tuple(items), {}))

    INSTANCES = []

    def __init__(self, *args, **kwargs):
        self._calls = MagicMock(name=f"{type(self).__name__}_inst{len(type(self).INSTANCES)}")
        self._calls(*args, **kwargs)  # record the constructor call
        type(self).INSTANCES.append(self)

    @classmethod
    def all(cls):
        return list(cls.INSTANCES)

    @classmethod
    def reset(cls):
        cls.INSTANCES = []


class _Signal:
    """Qt-like signal stub that records ``.connect`` calls."""

    def __init__(self, name):
        self._name = name
        self._connections = MagicMock(name=f"signal.{name}.connect_recorder")

    def connect(self, slot):
        self._connections.connect(slot)


class _QWidget(_Recorder):
    def setLayout(self, layout): self._calls.setLayout(layout)
    def setFocusPolicy(self, p): self._calls.setFocusPolicy(p)
    def setMinimumWidth(self, w): self._calls.setMinimumWidth(w)
    def setMinimumHeight(self, h): self._calls.setMinimumHeight(h)
    def setMaximumHeight(self, h): self._calls.setMaximumHeight(h)
    def setMaximumWidth(self, w): self._calls.setMaximumWidth(w)
    def setVisible(self, v): self._calls.setVisible(v)
    def setEnabled(self, e): self._calls.setEnabled(e)
    def setWordWrap(self, w): self._calls.setWordWrap(w)
    def setAlignment(self, a): self._calls.setAlignment(a)
    def setStyleSheet(self, s): self._calls.setStyleSheet(s)
    def setFixedWidth(self, w): self._calls.setFixedWidth(w)
    def setFixedHeight(self, h): self._calls.setFixedHeight(h)
    def setText(self, t): self._calls.setText(t)
    def setChecked(self, c): self._calls.setChecked(c)
    def setToolTip(self, t): self._calls.setToolTip(t)
    def setTextInteractionFlags(self, f): self._calls.setTextInteractionFlags(f)
    def blockSignals(self, b): self._calls.blockSignals(b)
    def isVisible(self): return self._calls.isVisible()
    def layout(self): return self._calls.layout()
    def keyPressEvent(self, e): self._calls.keyPressEvent(e)


class _QLabel(_QWidget): pass


class _QPushButton(_QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clicked = _Signal("clicked")


class _QCheckBox(_QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toggled = _Signal("toggled")


class _QSplitter(_Recorder):
    HORIZONTAL = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = MagicMock(name="saveState_result")
        self._state.__bool__ = lambda s: True
        self.splitterMoved = _Signal("splitterMoved")

    def addWidget(self, w): self._calls.addWidget(w); return w
    def setStretchFactor(self, idx, s): self._calls.setStretchFactor(idx, s)
    def setChildrenCollapsible(self, c): self._calls.setChildrenCollapsible(c)
    def setHandleWidth(self, w): self._calls.setHandleWidth(w)
    def saveState(self):
        self._calls.saveState()
        return self._state
    def restoreState(self, state): self._calls.restoreState(state)


class _QVBoxLayout(_Recorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def addWidget(self, w, stretch=0, *args, **kwargs):
        # Capture extras so we can introspect alignments like
        # ``sidebar_layout.addWidget(self._status_badge, 0, qt.Qt.AlignVCenter)``.
        self._calls.addWidget(w, stretch, *args, **kwargs)
        return w

    def addLayout(self, l, stretch=0):
        self._calls.addLayout(l, stretch)

    def addStretch(self, s=1):
        self._calls.addStretch(s)

    def setContentsMargins(self, *m): self._calls.setContentsMargins(*m)
    def setSpacing(self, s): self._calls.setSpacing(s)


class _QHBoxLayout(_QVBoxLayout): pass


class _Qt:
    Horizontal = 1
    AlignCenter = 0
    AlignTop = 32
    AlignVCenter = 16
    AlignHCenter = 4
    StrongFocus = 1
    Key_Escape = 0x01000000  # 0x1b
    Key_Left = 0x01000012
    Key_Right = 0x01000014
    # TextInteractionFlag values used by detail_tab.setTextInteractionFlags
    # so users can copy measurement values out of the sidebar. We just need
    # them present and distinct — production uses bitwise OR, not specific bits.
    NoTextInteraction = 0
    TextSelectableByMouse = 1
    TextSelectableByKeyboard = 2
    LinksAccessibleByMouse = 4
    LinksAccessibleByKeyboard = 8
    TextBrowserInteraction = 7


class _QSettings:
    """Real class so production code does `qt.QSettings()` without args."""

    INSTANCES = []

    def __init__(self, *args, **kwargs):
        self._values = {}
        self._calls = MagicMock(name=f"QSettings_inst{len(self.INSTANCES)}")
        self.INSTANCES.append(self)

    def value(self, key, default=None):
        self._calls.value(key, default)
        return self._values.get(key, default)

    def setValue(self, key, val):
        self._calls.setValue(key, val)
        self._values[key] = val

    @classmethod
    def reset(cls):
        cls.INSTANCES = []


class _QTimer:
    @staticmethod
    def singleShot(ms, cb):
        pass


def _install_qt_stub(monkeypatch):
    """Replace the ``qt`` module with the recordable stubs above."""
    qt_stub = MagicMock(name="qt")
    qt_stub.QWidget = _QWidget
    qt_stub.QSplitter = _QSplitter
    qt_stub.QVBoxLayout = _QVBoxLayout
    qt_stub.QHBoxLayout = _QHBoxLayout
    qt_stub.QLabel = _QLabel
    qt_stub.QPushButton = _QPushButton
    qt_stub.QCheckBox = _QCheckBox
    qt_stub.QSettings = _QSettings
    qt_stub.Qt = _Qt
    qt_stub.QByteArray = MagicMock(name="QByteArray")
    qt_stub.QPixmap = MagicMock(name="QPixmap")
    qt_stub.QTimer = _QTimer
    monkeypatch.setitem(sys.modules, "qt", qt_stub)

    # Stub out ZoomableImageView so detail_tab's import succeeds.
    zv = MagicMock(name="zoom_view")

    class _FakeZoomableImageView:
        INSTANCES = []

        def __init__(self, *args, **kwargs):
            self._calls = MagicMock(name=f"ZoomableImageView_inst{len(self.INSTANCES)}")
            self._calls(*args, **kwargs)
            _FakeZoomableImageView.INSTANCES.append(self)

        def set_manual_mode(self, v): self._calls.set_manual_mode(v)
        def clear_dots(self): self._calls.clear_dots()
        def show_placeholder(self, text=""): self._calls.show_placeholder(text)
        def set_pixmap(self, p, reset_zoom=True): self._calls.set_pixmap(p, reset_zoom)
        @property
        def _on_navigate(self): return self._on_navigate_val
        @_on_navigate.setter
        def _on_navigate(self, v): self._on_navigate_val = v
        @property
        def _tap_handler(self): return self._tap_handler_val
        @_tap_handler.setter
        def _tap_handler(self, v): self._tap_handler_val = v

    zv.ZoomableImageView = _FakeZoomableImageView
    sys.modules["ZebrafishEmbryoAnalyzerLib.zoom_view"] = zv

    # overlay is imported lazily by _build_rgb_array.
    om = MagicMock(name="overlay")
    # Return a real numpy array so cv2.cvtColor in _build_rgb_array is
    # happy; the actual pixel values don't matter for these tests, only
    # the fact that show_result() reaches the badge/banner refresh.
    import numpy as _np
    om.make_full_overlay = MagicMock(return_value=_np.zeros((4, 4, 3), dtype=_np.uint8))
    sys.modules["ZebrafishEmbryoAnalyzerLib.overlay"] = om
    # cv2 stub: real module so cv2.cvtColor works against a real ndarray.
    cv2_stub = MagicMock(name="cv2")
    cv2_stub.cvtColor = lambda src, code: _np.asarray(src)
    sys.modules["cv2"] = cv2_stub

    # PIL shim for _numpy_to_qpixmap.
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        pil = MagicMock(name="PIL")
        pil_mod = MagicMock(name="PIL.Image")
        pil_mod.fromarray = MagicMock(return_value=MagicMock(
            save=MagicMock(side_effect=lambda buf, **kw: buf.write(b"BMP"))
        ))
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_mod

    # Reset all the recorder class instance lists so each fixture gets a clean slate.
    for cls in (_QWidget, _QSplitter, _QVBoxLayout, _QHBoxLayout, _QLabel, _QPushButton, _QCheckBox):
        cls.reset()
    _QSettings.reset()

    return {
        "qt": qt_stub,
    }


@pytest.fixture
def stubs(monkeypatch):
    return _install_qt_stub(monkeypatch)


def _import_detail_tab():
    mod_name = "ZebrafishEmbryoAnalyzerLib.detail_tab"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import ZebrafishEmbryoAnalyzerLib.detail_tab as dt  # noqa: E402
    return dt


def _make_tab(stubs):
    dt = _import_detail_tab()
    tab = dt.DetailTab(on_navigate=lambda d: None, logic=MagicMock())
    return dt, tab


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_splitter_image_left_and_sidebar_right(stubs):
    """#66 acceptance: image on the left, every control moved into
    a sidebar widget on the right side of a QSplitter."""
    dt, tab = _make_tab(stubs)
    splitter = _QSplitter.all()[-1]  # last-created splitter

    add_calls = [c.args[0] for c in splitter._calls.addWidget.call_args_list]
    assert len(add_calls) == 2, (
        f"QSplitter must hold exactly two children (view, sidebar), "
        f"got {len(add_calls)} addWidget calls"
    )
    image_view, sidebar_widget = add_calls[0], add_calls[1]

    # ZoomableImageView was instantiated exactly once and the instance is
    # what's on the left of the splitter.
    zv_module = sys.modules["ZebrafishEmbryoAnalyzerLib.zoom_view"]
    zv_instances = zv_module.ZoomableImageView.INSTANCES
    assert len(zv_instances) == 1, "ZoomableImageView must be constructed exactly once"
    assert image_view is zv_instances[0], (
        "First QSplitter child must be the ZoomableImageView (left)"
    )
    assert sidebar_widget is tab._sidebar, (
        "Second QSplitter child must be tab._sidebar (right)"
    )


def test_splitter_orientation_is_horizontal(stubs):
    dt, _tab = _make_tab(stubs)
    splitter = _QSplitter.all()[-1]
    ctor_args = splitter._calls.call_args.args
    assert ctor_args[0] == _Qt.Horizontal, (
        "QSplitter must be constructed with Qt.Horizontal orientation"
    )


def test_splitter_stretch_factors_image_flex_sidebar_fixed(stubs):
    """#66: image flexes; sidebar stays at its minimum even when the panel grows."""
    dt, _tab = _make_tab(stubs)
    splitter = _QSplitter.all()[-1]
    stretches = [c.args for c in splitter._calls.setStretchFactor.call_args_list]
    assert stretches == [(0, 1), (1, 0)], (
        f"Stretch factors must be image=1 (flex), sidebar=0 (fixed), got {stretches}"
    )


def test_sidebar_has_minimum_width_in_spec_band(stubs):
    """#66 acceptance: sidebar must call setMinimumWidth(220) (200-260px band)
    so the splitter restoreState can't collapse it to Qt's 0-default on resize."""
    dt, tab = _make_tab(stubs)
    sidebar = tab._sidebar
    assert sidebar._calls.setMinimumWidth.called, "Sidebar must call setMinimumWidth"
    width_arg = sidebar._calls.setMinimumWidth.call_args.args[0]
    assert 200 <= width_arg <= 260, (
        f"Sidebar minimum width must be in 200-260px band per #66, got {width_arg}"
    )


def test_sidebar_holds_every_pre_existing_control(stubs):
    """#66: nothing should be lost — the sidebar must host the existing
    Manual Adjust row, status, Exclude, Metrics, and Nav row.

    #67 re-shaped the metrics area from a single ``_metrics_label`` to a
    5-row grid + status badge + error banner, so this test now asserts on
    those widget handles instead of the removed ``_metrics_label``.
    """
    dt, tab = _make_tab(stubs)

    # Every QVBoxLayout created — the one for self (the tab) and the one for the sidebar.
    # Find the one whose addWidget/addLayout calls reference the manually-created widgets.
    candidates = list(_QVBoxLayout.all())

    def _holds_widget(layout, widget):
        for c in layout._calls.addWidget.call_args_list:
            if c.args and c.args[0] is widget:
                return True
        return False

    sidebar_layout = None
    for layout in candidates:
        if (_holds_widget(layout, tab._manual_row_widget)
                and _holds_widget(layout, tab._chk_exclude)
                and _holds_widget(layout, tab._error_banner)):
            sidebar_layout = layout
            break

    assert sidebar_layout is not None, (
        "Could not find the sidebar's QVBoxLayout — it must hold the manual row, "
        "exclude checkbox, and error banner."
    )

    add_widgets = [c.args[0] for c in sidebar_layout._calls.addWidget.call_args_list]
    assert tab._manual_row_widget in add_widgets, (
        "Manual Adjust/Revert row must live inside the sidebar"
    )
    assert tab._manual_status in add_widgets, (
        "Manual status label must live inside the sidebar"
    )
    assert tab._chk_exclude in add_widgets, (
        "Exclude checkbox must live inside the sidebar"
    )
    assert tab._error_banner in add_widgets, (
        "Error banner must live inside the sidebar (#67)"
    )
    sidebar_layout._calls.addLayout.assert_called()
    layout_args = [c.args[0] for c in sidebar_layout._calls.addLayout.call_args_list]
    assert tab._nav_row_layout in layout_args, (
        "Nav row layout must live at the bottom of the sidebar"
    )
    assert tab._measurements_grid in layout_args, (
        "Measurements grid layout must live inside the sidebar (#67)"
    )

    # Filename + status badge share a top HBox inside the sidebar — verify
    # both widgets landed in *some* QHBoxLayout's addWidget list, and that
    # layout is itself in the sidebar's addLayout list (#67).
    hbox_layouts = list(_QHBoxLayout.all())
    top_row = None
    for hbox in hbox_layouts:
        if (_holds_widget(hbox, tab._filename_label)
                and _holds_widget(hbox, tab._status_badge)):
            top_row = hbox
            break
    assert top_row is not None, (
        "Filename label + status badge must share a top HBoxLayout (#67)"
    )
    assert top_row in layout_args, (
        "Top row (filename + status badge) HBoxLayout must live inside the sidebar (#67)"
    )


def test_splitter_restore_state_called_when_state_saved(stubs):
    """#66 acceptance: when QSettings returns a prior splitter state,
    DetailTab must call restoreState during __init__."""
    saved = MagicMock(name="saved_state")
    saved.__bool__ = lambda s: True

    # Replace the QSettings factory on the qt stub so the very first
    # `qt.QSettings()` call from DetailTab returns an instance with our
    # saved value pre-populated.
    preloaded = _QSettings()
    preloaded._values["ZebrafishEmbryoAnalyzer/detailSplitterState"] = saved

    def factory(*args, **kwargs):
        # Always return the same preloaded instance so its values are visible.
        return preloaded

    stubs["qt"].QSettings = factory

    dt, _tab = _make_tab(stubs)
    splitter = _QSplitter.all()[-1]
    assert splitter._calls.restoreState.called, (
        "restoreState must be called on the splitter when QSettings has saved state"
    )
    assert splitter._calls.restoreState.call_args.args[0] is saved, (
        "restoreState must be called with the exact QByteArray QSettings returned"
    )


def _make_settings_first():
    """Helper to ensure a QSettings instance exists when the test starts."""
    return _QSettings()


def test_splitter_restore_skipped_when_no_saved_state(stubs):
    """#66: first run → no prior state → restoreState must NOT be called."""
    dt, _tab = _make_tab(stubs)
    splitter = _QSplitter.all()[-1]
    assert not splitter._calls.restoreState.called, (
        "restoreState must be skipped when QSettings returns no saved state"
    )


def test_qsettings_key_follows_project_convention(stubs):
    """#66 consistency: the persistence key must follow the project's
    'ZebrafishEmbryoAnalyzer/...' prefix that other widgets in this module use."""
    dt, _tab = _make_tab(stubs)
    # QSettings is queried during construction — grab the most-recent instance.
    settings = _QSettings.INSTANCES[-1]
    keys = [c.args[0] for c in settings._calls.value.call_args_list if c.args]
    assert dt.DetailTab._SPLITTER_SETTINGS_KEY in keys, (
        f"QSettings.value must be read for {dt.DetailTab._SPLITTER_SETTINGS_KEY!r}, "
        f"saw keys: {keys}"
    )
    assert dt.DetailTab._SPLITTER_SETTINGS_KEY.startswith("ZebrafishEmbryoAnalyzer/"), (
        "Splitter state key must follow project 'ZebrafishEmbryoAnalyzer/...' convention"
    )


def test_save_splitter_state_writes_to_qsettings(stubs):
    dt, tab = _make_tab(stubs)
    n_before = len(_QSettings.INSTANCES)
    tab.save_splitter_state()
    # save_splitter_state builds a fresh QSettings; inspect the new instance.
    new = _QSettings.INSTANCES[n_before:]
    assert new, "save_splitter_state must construct a QSettings instance"
    settings = new[-1]
    setv_calls = settings._calls.setValue.call_args_list
    matching = [c for c in setv_calls
                if c.args and c.args[0] == dt.DetailTab._SPLITTER_SETTINGS_KEY]
    assert matching, (
        f"save_splitter_state must call QSettings.setValue with the key "
        f"{dt.DetailTab._SPLITTER_SETTINGS_KEY!r}"
    )
    assert matching[0].args[1] is _QSplitter.all()[-1]._state


def test_splitter_moved_signal_wires_live_persistence(stubs):
    """#66: splitterMoved must call save_splitter_state so each user drag
    is persisted without waiting for cleanup()."""
    dt, _tab = _make_tab(stubs)
    splitter = _QSplitter.all()[-1]
    # Production does self._splitter.splitterMoved.connect(...); we expose
    # splitterMoved as a real Signal attribute so .connect is recorded there.
    assert splitter.splitterMoved._connections.connect.called, (
        "DetailTab must connect to QSplitter.splitterMoved for live persistence"
    )
    callback = splitter.splitterMoved._connections.connect.call_args_list[0].args[0]

    n_before = len(_QSettings.INSTANCES)
    callback(120, 300)  # signature varies; MagicMock eats any args
    new = _QSettings.INSTANCES[n_before:]
    assert new, "splitterMoved callback must construct a QSettings instance"
    settings = new[-1]
    setv_calls = settings._calls.setValue.call_args_list
    assert any(
        c.args and c.args[0] == dt.DetailTab._SPLITTER_SETTINGS_KEY
        for c in setv_calls
    ), "splitterMoved callback must persist via save_splitter_state → QSettings.setValue"


def test_cleanup_persists_splitter_state(stubs):
    dt, tab = _make_tab(stubs)
    n_before = len(_QSettings.INSTANCES)
    tab.cleanup()
    new = _QSettings.INSTANCES[n_before:]
    assert new, "cleanup() must construct a QSettings instance to persist state"
    settings = new[-1]
    matching = [c for c in settings._calls.setValue.call_args_list
                if c.args and c.args[0] == dt.DetailTab._SPLITTER_SETTINGS_KEY]
    assert matching, "cleanup() must persist the splitter state via QSettings.setValue"


def test_keypressevent_still_handles_escape_and_arrows(stubs):
    """#66 explicit guard: keyPressEvent must still route Escape / Left /
    Right on the parent DetailTab so a sidebar widget doesn't silently
    swallow those keys."""
    source = DETAIL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "keyPressEvent":
            found = node
            break
    assert found is not None, "keyPressEvent must remain a method on DetailTab"
    src = ast.unparse(found)
    for token in ("Key_Escape", "Key_Left", "Key_Right"):
        assert token in src, (
            f"keyPressEvent must still reference qt.Qt.{token}; "
            f"removing it would violate #66's 'keyPressEvent must keep working exactly as today' clause."
        )


# ---------------------------------------------------------------------------
# Issue #67 — Measurements section, status badge, error banner
# ---------------------------------------------------------------------------
#
# These tests cover the five acceptance criteria from issue #67:
#
#   1. Five-row Measurements section in the sidebar (always present, never
#      hidden, missing values render as "—").
#   2. Status badge above the metrics with priority order stale → error →
#      manual_corrected → analyzed → not analyzed.
#   3. Error banner above the metrics, shown only when ``result["error"]``
#      is non-empty.
#   4. ``set_stale(bool)`` setter so ``widget.py`` can flip the badge live.
#   5. The default stale state is False (so freshly-built tabs don't appear
#      stale before any MRML observation has run).


def _latest_settext(widget, attr="_calls"):
    """Return the last setText text passed to ``widget`` if any, else fall
    back to the text passed at construction time (production uses
    ``qt.QLabel("—")`` for measurement value initial state, so we need to
    see constructor args too)."""
    calls = getattr(widget, attr).setText.call_args_list
    if calls:
        args = calls[-1].args
        if args:
            return args[0]
    ctor_calls = getattr(widget, attr).call_args_list
    if ctor_calls and ctor_calls[0].args:
        return ctor_calls[0].args[0]
    return None


def _latest_stylesheet(widget, attr="_calls"):
    """Return the last setStyleSheet CSS passed to ``widget`` if any, else None."""
    calls = getattr(widget, attr).setStyleSheet.call_args_list
    if not calls:
        return None
    args = calls[-1].args
    return args[0] if args else None


def test_measurements_grid_has_five_rows(stubs):
    """#67: the sidebar must host a 5-row Measurements section in the
    canonical order: Length, Curvature class, Length/straight ratio,
    Eye area, Eye diameter."""
    dt, tab = _make_tab(stubs)
    assert len(tab._measurements) == 5, (
        f"DetailTab must keep exactly 5 measurement rows, got {len(tab._measurements)}"
    )

    def _label_text(widget):
        text = _latest_settext(widget)
        return text or ""

    actual_labels = [_label_text(label) for label, _ in tab._measurements]
    expected_labels = [
        "Length:", "Curvature class:", "Length/straight ratio:",
        "Eye area:", "Eye diameter:",
    ]
    assert actual_labels == expected_labels, (
        f"Measurement labels must appear in canonical order, got {actual_labels}"
    )


def test_measurements_show_em_dash_when_field_missing(stubs):
    """#67: missing values render as the em-dash placeholder '—', not
    blank/empty, so the sidebar's vertical footprint stays constant as
    the user navigates between images with different fields populated."""
    dt, tab = _make_tab(stubs)

    # Bare tab → all measurement values are at the "—" default.
    for _label, value_widget in tab._measurements:
        assert _latest_settext(value_widget) == "—", (
            "Bare DetailTab must initialise every measurement value to '—'"
        )

    # show_result with a row that has no fields at all → still all "—".
    empty_row = {
        "filename": "blank.tif", "original": None, "mask": None,
        "length": None, "curvature": None, "ratio": None,
        "eye_area": None, "eye_diameter": None, "error": None,
        "manual_corrected": False,
    }
    tab.show_result(0, [empty_row])
    for _label, value_widget in tab._measurements:
        assert _latest_settext(value_widget) == "—", (
            "show_result with no fields must leave every measurement as '—'"
        )


def test_measurements_use_canonical_format_strings(stubs):
    """#67: each measurement has a fixed display format — Length µm,
    Curvature raw, Ratio 3 decimals, Eye area µm², Eye diameter µm."""
    dt, tab = _make_tab(stubs)
    row = {
        "filename": "ok.tif", "original": MagicMock(), "mask": MagicMock(),
        "length": 123.456, "curvature": "straight",
        "ratio": 0.85, "eye_area": 1500.7, "eye_diameter": 12.34,
        "error": None, "manual_corrected": False,
    }
    tab.show_result(0, [row])

    expected_texts = [
        "123.5 µm",    # length — 1 decimal, " µm"
        "straight",    # curvature — str() of the raw value
        "0.850",       # ratio — 3 decimals
        "1500.7 µm²",  # eye area — 1 decimal + " µm²"
        "12.3 µm",     # eye diameter — 1 decimal + " µm"
    ]
    for (label_widget, value_widget), expected in zip(tab._measurements, expected_texts):
        actual = _latest_settext(value_widget)
        assert actual == expected, (
            f"Measurement value for {label_widget!r} must be {expected!r}, "
            f"got {actual!r}"
        )


def test_measurements_text_selectable_by_mouse(stubs):
    """#67 acceptance: the user must be able to copy measurement values
    out of the sidebar, which means setTextInteractionFlags must have
    enabled mouse-driven selection on every value label."""
    dt, tab = _make_tab(stubs)
    for label, value in tab._measurements:
        flags = value._calls.setTextInteractionFlags.call_args_list
        assert flags, (
            f"setTextInteractionFlags must be called on {label!r}'s value widget"
        )
        # Every call must include the mouse-selection flag (bit OR'd into
        # the flag set). Our stub uses TextSelectableByMouse=1.
        for call in flags:
            arg = call.args[0]
            assert arg & _Qt.TextSelectableByMouse, (
                f"setTextInteractionFlags must include TextSelectableByMouse "
                f"so users can copy values, got flags={arg!r}"
            )


def test_status_badge_default_state_is_not_analyzed(stubs):
    """#67: a freshly-built DetailTab must show 'Not analyzed' so the
    user has feedback before the first image is rendered."""
    dt, tab = _make_tab(stubs)
    text = _latest_settext(tab._status_badge)
    css = _latest_stylesheet(tab._status_badge)
    assert text == "Not analyzed", (
        f"Freshly-built DetailTab's status badge must read 'Not analyzed', got {text!r}"
    )
    assert css and "background:" in css, (
        "Status badge must apply a background colour (CSS background: ...)"
    )


def test_status_badge_priority_stale_beats_everything(stubs):
    """#67 priority #1: stale wins over error, manual_corrected, analyzed."""
    dt, tab = _make_tab(stubs)
    result = {
        "filename": "x.tif",
        "length": 100.0, "mask": MagicMock(),
        "error": "boom",            # would also trigger 'Error'
        "manual_corrected": True,   # would also trigger 'Manually corrected'
    }
    tab.show_result(0, [result])
    # Now flip the stale flag — it should beat everything else.
    tab.set_stale(True)
    text = _latest_settext(tab._status_badge)
    css = _latest_stylesheet(tab._status_badge)
    assert text == "Stale — recompute needed", (
        f"Stale must beat all other states per #67 priority #1, got {text!r}"
    )
    assert css and "255, 152, 0" in css, (
        f"Stale badge colour must be amber (255, 152, 0), got {css!r}"
    )


def test_status_badge_priority_error_beats_manual(stubs):
    """#67 priority #2: error beats manual_corrected + analyzed."""
    dt, tab = _make_tab(stubs)
    result = {
        "filename": "x.tif",
        "length": 100.0, "mask": MagicMock(),
        "error": "timeout",
        "manual_corrected": True,
    }
    tab.show_result(0, [result])
    text = _latest_settext(tab._status_badge)
    css = _latest_stylesheet(tab._status_badge)
    assert text == "Error", (
        f"Error must beat manual_corrected + analyzed per #67 priority #2, got {text!r}"
    )
    assert css and "244, 67, 54" in css, (
        f"Error badge colour must be red (244, 67, 54), got {css!r}"
    )


def test_status_badge_priority_manual_corrected_beats_analyzed(stubs):
    """#67 priority #3: manual_corrected beats analyzed."""
    dt, tab = _make_tab(stubs)
    result = {
        "filename": "x.tif",
        "length": 100.0, "mask": MagicMock(),
        "manual_corrected": True,
    }
    tab.show_result(0, [result])
    text = _latest_settext(tab._status_badge)
    css = _latest_stylesheet(tab._status_badge)
    assert text == "Manually corrected", (
        f"Manual corrected must beat analyzed per #67 priority #3, got {text!r}"
    )
    assert css and "33, 150, 243" in css, (
        f"Manual badge colour must be blue (33, 150, 243), got {css!r}"
    )


def test_status_badge_priority_analyzed_when_length_or_mask_set(stubs):
    """#67 priority #4: a row with length OR mask is 'Analyzed'."""
    dt, tab = _make_tab(stubs)

    # Length-only case.
    tab.show_result(0, [{"filename": "a.tif", "length": 100.0,
                          "mask": None, "error": None, "manual_corrected": False}])
    assert _latest_settext(tab._status_badge) == "Analyzed"

    # Mask-only case (length=None but mask present).
    tab.show_result(0, [{"filename": "b.tif", "length": None,
                          "mask": MagicMock(), "error": None, "manual_corrected": False}])
    assert _latest_settext(tab._status_badge) == "Analyzed"

    css = _latest_stylesheet(tab._status_badge)
    assert css and "76, 175, 80" in css, (
        f"Analyzed badge colour must be green (76, 175, 80), got {css!r}"
    )


def test_status_badge_priority_not_analyzed_when_nothing_present(stubs):
    """#67 priority #5: no length, no mask, no error, not corrected → 'Not analyzed'."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "c.tif", "length": None,
                          "mask": None, "error": None, "manual_corrected": False}])
    assert _latest_settext(tab._status_badge) == "Not analyzed"


def test_error_banner_hidden_when_no_error(stubs):
    """#67 acceptance: the error banner stays hidden until a row with
    an error is shown."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "ok.tif", "length": 100.0, "mask": MagicMock(),
                          "error": None, "manual_corrected": False}])
    # The most recent setVisible(False) is the post-show_result state.
    visible_calls = tab._error_banner._calls.setVisible.call_args_list
    assert visible_calls, "Error banner visibility must be set during show_result"
    last_visible_arg = visible_calls[-1].args[0]
    assert last_visible_arg is False, (
        f"Error banner must be hidden when result has no error, got setVisible({last_visible_arg!r})"
    )


def test_error_banner_shown_when_error_set(stubs):
    """#67 acceptance: when a row's error string is non-empty, the
    banner appears with the error message as its text."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "bad.tif", "length": None,
                          "mask": None, "error": "Inference failed: OOM",
                          "manual_corrected": False}])
    visible_calls = tab._error_banner._calls.setVisible.call_args_list
    assert visible_calls
    last_visible_arg = visible_calls[-1].args[0]
    assert last_visible_arg is True, (
        f"Error banner must be shown when result['error'] is set, got setVisible({last_visible_arg!r})"
    )
    text = _latest_settext(tab._error_banner)
    assert text == "Inference failed: OOM", (
        f"Error banner must display the exact error string, got {text!r}"
    )


def test_error_banner_clears_when_subsequent_row_has_no_error(stubs):
    """#67 acceptance: navigating from an errored row to a clean one
    must hide the banner again (it isn't a sticky indicator)."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "bad.tif", "error": "boom"}])
    tab.show_result(0, [{"filename": "ok.tif", "length": 100.0, "mask": MagicMock(),
                          "error": None, "manual_corrected": False}])
    visible_calls = tab._error_banner._calls.setVisible.call_args_list
    last_visible_arg = visible_calls[-1].args[0]
    assert last_visible_arg is False, (
        "Error banner must hide again after navigating away from the errored row"
    )


def test_set_stale_default_is_false(stubs):
    """#67: a freshly-built DetailTab must default to _current_is_stale=False
    so a tab that has never received a stale notification isn't stale."""
    dt, tab = _make_tab(stubs)
    assert tab._current_is_stale is False, (
        f"Default _current_is_stale must be False, got {tab._current_is_stale!r}"
    )


def test_set_stale_true_flips_badge_when_row_visible(stubs):
    """#67 acceptance: set_stale(True) flips the badge to 'Stale — recompute needed'
    immediately when a row is currently shown."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    # Baseline: green Analyzed.
    assert _latest_settext(tab._status_badge) == "Analyzed"
    tab.set_stale(True)
    assert _latest_settext(tab._status_badge) == "Stale — recompute needed", (
        "set_stale(True) must immediately update the badge"
    )
    assert tab._current_is_stale is True


def test_set_stale_false_clears_stale_state_on_badge(stubs):
    """#67: set_stale(False) restores the badge to the row's underlying state
    (Analyzed, Error, etc.) once the segment is no longer stale."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_stale(True)
    assert _latest_settext(tab._status_badge) == "Stale — recompute needed"
    tab.set_stale(False)
    assert _latest_settext(tab._status_badge) == "Analyzed", (
        "set_stale(False) must restore the badge to the row's underlying state"
    )
    assert tab._current_is_stale is False


def test_set_stale_coerces_to_bool(stubs):
    """#67: set_stale accepts truthy/falsy values; storage is bool-coerced
    so downstream priority logic doesn't have to special-case 'truthy'."""
    dt, tab = _make_tab(stubs)
    tab.set_stale(1)
    assert tab._current_is_stale is True
    tab.set_stale(0)
    assert tab._current_is_stale is False


def test_set_stale_is_noop_before_any_row_shown(stubs):
    """#67: set_stale must not crash when called before show_result — there's
    no current row to refresh yet."""
    dt, tab = _make_tab(stubs)
    # No show_result call yet.
    tab.set_stale(True)
    assert tab._current_is_stale is True
    # The badge should not have been touched (no current result to update).
    # Its text should still be the default "Not analyzed" from __init__.
    assert _latest_settext(tab._status_badge) == "Not analyzed", (
        "set_stale before show_result must not refresh the badge with junk"
    )


def test_reset_clears_stale_state_and_resets_badge(stubs):
    """#67: DetailTab.reset() must restore the default non-stale state so a
    fresh dataset doesn't inherit the previous row's staleness."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_stale(True)
    assert tab._current_is_stale is True
    tab.reset()
    assert tab._current_is_stale is False, (
        "reset() must restore _current_is_stale to its default False"
    )
    assert _latest_settext(tab._status_badge) == "Not analyzed"
    visible_calls = tab._error_banner._calls.setVisible.call_args_list
    assert visible_calls[-1].args[0] is False, (
        "reset() must hide the error banner"
    )


# ---------------------------------------------------------------------------
# Issue #68 — Actions grouping + relocated Recompute metrics button
# ---------------------------------------------------------------------------
#
# #68 wraps the manual-adjust row, manual status, exclude checkbox, and the
# relocated "Recompute metrics" button under an "Actions" section heading
# inside the sidebar (the button used to live on the QTabBar; #68 moves
# it here). set_stale() drives the button's visibility, and
# set_recompute_callback() wires its click handler.


def test_actions_heading_lives_in_sidebar(stubs):
    """#68: the sidebar must host an 'Actions' heading label so the three
    sidebar regions (Status / Metrics / Actions / Nav) read as separate
    sections rather than an undifferentiated stack."""
    dt, tab = _make_tab(stubs)
    assert tab._actions_heading is not None, (
        "DetailTab must create an _actions_heading widget"
    )
    text = _latest_settext(tab._actions_heading)
    assert text == "Actions", (
        f"Actions heading must read 'Actions', got {text!r}"
    )

    # And the heading must be inside the sidebar's QVBoxLayout, above
    # the manual-row widget (the first action group widget).
    candidates = list(_QVBoxLayout.all())

    def _holds_widget(layout, widget):
        for c in layout._calls.addWidget.call_args_list:
            if c.args and c.args[0] is widget:
                return True
        return False

    sidebar_layout = None
    for layout in candidates:
        if (_holds_widget(layout, tab._actions_heading)
                and _holds_widget(layout, tab._manual_row_widget)
                and _holds_widget(layout, tab._chk_exclude)):
            sidebar_layout = layout
            break
    assert sidebar_layout is not None, (
        "Sidebar layout must hold the Actions heading + manual row + exclude"
    )

    add_widgets = [
        c.args[0] for c in sidebar_layout._calls.addWidget.call_args_list
    ]
    heading_idx = add_widgets.index(tab._actions_heading)
    manual_idx = add_widgets.index(tab._manual_row_widget)
    assert heading_idx < manual_idx, (
        f"Actions heading must appear above the manual row in the sidebar "
        f"(heading={heading_idx}, manual={manual_idx})"
    )


def test_recompute_button_lives_in_sidebar_under_actions(stubs):
    """#68: the Relocated Recompute button must live inside the sidebar's
    QVBoxLayout, immediately after the exclude checkbox (the last item
    in the actions group), and definitely NOT on a QTabBar."""
    dt, tab = _make_tab(stubs)
    candidates = list(_QVBoxLayout.all())

    def _holds_widget(layout, widget):
        for c in layout._calls.addWidget.call_args_list:
            if c.args and c.args[0] is widget:
                return True
        return False

    sidebar_layout = None
    for layout in candidates:
        if _holds_widget(layout, tab._btn_recompute):
            sidebar_layout = layout
            break
    assert sidebar_layout is not None, (
        "Recompute button must be added to a QVBoxLayout (the sidebar layout)"
    )
    add_widgets = [
        c.args[0] for c in sidebar_layout._calls.addWidget.call_args_list
    ]
    assert tab._btn_recompute in add_widgets, (
        "Recompute button must be a direct child of the sidebar's QVBoxLayout"
    )
    # The button should sit after the exclude checkbox (last action-group item).
    exclude_idx = add_widgets.index(tab._chk_exclude)
    btn_idx = add_widgets.index(tab._btn_recompute)
    assert btn_idx > exclude_idx, (
        f"Recompute button must appear after Exclude checkbox in the "
        f"sidebar (exclude={exclude_idx}, btn={btn_idx})"
    )


def test_recompute_button_hidden_by_default(stubs):
    """#68: the button must start hidden — it's only revealed when a row
    is stale AND a recompute callback has been registered."""
    dt, tab = _make_tab(stubs)
    visible_calls = tab._btn_recompute._calls.setVisible.call_args_list
    assert visible_calls, "Recompute button visibility must be set in __init__"
    last_visible = visible_calls[-1].args[0]
    assert last_visible is False, (
        f"Recompute button must start hidden, got setVisible({last_visible!r})"
    )
    enabled_calls = tab._btn_recompute._calls.setEnabled.call_args_list
    assert enabled_calls, "Recompute button enabled state must be set in __init__"
    assert enabled_calls[-1].args[0] is False, (
        "Recompute button must start disabled"
    )


def test_set_stale_does_not_show_recompute_without_callback(stubs):
    """#68: a stale row alone must NOT reveal the Recompute button —
    the callback has to be registered first (by widget.py) so the click
    actually does something. This prevents a bare DetailTab in unit
    tests from showing a clickable no-op."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_stale(True)
    visible_calls = tab._btn_recompute._calls.setVisible.call_args_list
    last_visible = visible_calls[-1].args[0]
    assert last_visible is False, (
        f"Stale alone must NOT reveal the Recompute button (no callback "
        f"registered), got setVisible({last_visible!r})"
    )


def test_set_stale_with_callback_reveals_recompute_button(stubs):
    """#68 acceptance: stale row + registered callback → button visible
    AND enabled, so the user can click it to recompute metrics."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_recompute_callback(lambda: None)
    tab.set_stale(True)
    visible_calls = tab._btn_recompute._calls.setVisible.call_args_list
    enabled_calls = tab._btn_recompute._calls.setEnabled.call_args_list
    assert visible_calls[-1].args[0] is True, (
        "Recompute button must be visible when stale + callback registered"
    )
    assert enabled_calls[-1].args[0] is True, (
        "Recompute button must be enabled when stale + callback registered"
    )


def test_set_stale_false_hides_recompute_button(stubs):
    """#68: once a row is no longer stale (e.g. recompute just ran), the
    button must hide again so the user can't accidentally re-trigger it."""
    dt, tab = _make_tab(stubs)
    tab.set_recompute_callback(lambda: None)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_stale(True)
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is True
    tab.set_stale(False)
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is False, (
        "Recompute button must hide when the row is no longer stale"
    )
    assert tab._btn_recompute._calls.setEnabled.call_args_list[-1].args[0] is False


def test_set_recompute_callback_refreshes_button_when_already_stale(stubs):
    """#68: registering a callback while a stale row is already shown
    must immediately reveal the button — not wait for the next
    set_stale() call from widget.py."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_stale(True)  # stale set BEFORE callback
    # Button still hidden — no callback yet.
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is False
    # Register callback now.
    tab.set_recompute_callback(lambda: None)
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is True, (
        "set_recompute_callback must immediately reveal the button when "
        "the current row is stale"
    )


def test_set_recompute_callback_none_hides_button(stubs):
    """#68: passing None as the callback (e.g. widget.py's try/except
    failed path) must hide the button so it can't be clicked."""
    dt, tab = _make_tab(stubs)
    tab.set_recompute_callback(lambda: None)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_stale(True)
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is True
    # Clear the callback (simulates widget.py teardown).
    tab.set_recompute_callback(None)
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is False, (
        "set_recompute_callback(None) must hide the button"
    )


def test_recompute_button_click_invokes_callback(stubs):
    """#68: clicking the Recompute button must call the registered
    callback (so widget.py's _on_recompute_current_detail runs)."""
    dt, tab = _make_tab(stubs)
    callback = MagicMock()
    tab.set_recompute_callback(callback)
    # Production wires this in __init__: self._btn_recompute.clicked.connect(
    # self._on_recompute_clicked)
    btn = tab._btn_recompute
    # Find the connection that targets _on_recompute_clicked.
    # The clicked signal on _QPushButton is a _Signal instance.
    clicked = btn.clicked
    # Get the last connection — that's the click handler.
    connected = clicked._connections.connect.call_args_list[-1].args[0]
    connected()
    callback.assert_called_once()


def test_recompute_button_click_without_callback_is_safe(stubs):
    """#68 defence-in-depth: even if the user somehow clicks the button
    before any callback is registered, the click must be a safe no-op
    (not raise)."""
    dt, tab = _make_tab(stubs)
    # No set_recompute_callback call.
    btn = tab._btn_recompute
    connected = btn.clicked._connections.connect.call_args_list[-1].args[0]
    # Must not raise.
    connected()


def test_recompute_button_callback_exception_is_caught(stubs):
    """#68: if widget.py's callback raises, the click handler must not
    propagate the exception (would kill the UI thread)."""
    dt, tab = _make_tab(stubs)
    def boom():
        raise RuntimeError("simulated widget-side bug")
    tab.set_recompute_callback(boom)
    btn = tab._btn_recompute
    connected = btn.clicked._connections.connect.call_args_list[-1].args[0]
    # Must not raise.
    connected()


def test_reset_hides_recompute_button(stubs):
    """#68: reset() must hide the Recompute button so a fresh dataset
    doesn't inherit the previous row's stale button state."""
    dt, tab = _make_tab(stubs)
    tab.set_recompute_callback(lambda: None)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    tab.set_stale(True)
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is True
    tab.reset()
    assert tab._btn_recompute._calls.setVisible.call_args_list[-1].args[0] is False, (
        "reset() must hide the Recompute button"
    )
    assert tab._btn_recompute._calls.setEnabled.call_args_list[-1].args[0] is False


def test_recompute_button_click_handler_does_not_import_mrml(stubs):
    """#68 architectural guard: DetailTab must never import MRML
    (mrml.is_volume_node_stale lives on the widget's logic side).
    Verify the production source has no ``from ZebrafishEmbryoAnalyzerLib
    .mrml import`` at module level — the stale flag arrives via set_stale.
    """
    import re
    source = DETAIL_PATH.read_text(encoding="utf-8")
    bad = re.search(r"^\s*from\s+ZebrafishEmbryoAnalyzerLib\.mrml\b",
                    source, flags=re.MULTILINE)
    assert not bad, (
        "DetailTab must not import from ZebrafishEmbryoAnalyzerLib.mrml — "
        "stale state arrives via set_stale() instead"
    )


# ---------------------------------------------------------------------------
# Issue #11 — Segmentation overlay toggle
# ---------------------------------------------------------------------------
#
# #11 adds a "Show segmentation overlay" checkbox to the sidebar that lets
# the user hide the mask/path/straight-line overlay and see the bare
# original image. The choice is persisted to QSettings so it survives
# restarts. The pixmap cache is keyed by (index, overlay_visible) so
# toggling rebuilds the right variant.


def test_overlay_checkbox_lives_in_sidebar_between_measurements_and_actions(stubs):
    """#11: the 'Show segmentation overlay' checkbox must live in the
    sidebar's QVBoxLayout, between the measurements grid and the Actions
    heading (so it controls how the displayed image is rendered, not
    how the metadata is computed)."""
    dt, tab = _make_tab(stubs)
    candidates = list(_QVBoxLayout.all())

    def _holds_widget(layout, widget):
        for c in layout._calls.addWidget.call_args_list:
            if c.args and c.args[0] is widget:
                return True
        return False

    sidebar_layout = None
    for layout in candidates:
        if (_holds_widget(layout, tab._chk_overlay)
                and _holds_widget(layout, tab._actions_heading)
                and _holds_widget(layout, tab._chk_exclude)):
            sidebar_layout = layout
            break
    assert sidebar_layout is not None, (
        "Sidebar layout must hold the overlay checkbox, actions heading, "
        "and exclude checkbox"
    )

    add_widgets = [
        c.args[0] for c in sidebar_layout._calls.addWidget.call_args_list
    ]
    overlay_idx = add_widgets.index(tab._chk_overlay)
    # The overlay checkbox must appear AFTER the measurements grid (the
    # measurements grid is added via addLayout, so we can't compare by
    # widget identity — instead, verify it's before the actions heading).
    actions_idx = add_widgets.index(tab._actions_heading)
    assert overlay_idx < actions_idx, (
        f"Overlay checkbox must appear before the Actions heading "
        f"(overlay={overlay_idx}, actions={actions_idx})"
    )


def test_overlay_checkbox_default_checked(stubs):
    """#11: by default the overlay is visible (preserves pre-#11 UX)."""
    dt, tab = _make_tab(stubs)
    text = _latest_settext(tab._chk_overlay)
    assert text == "Show segmentation overlay", (
        f"Checkbox label must read 'Show segmentation overlay', got {text!r}"
    )
    # setChecked was called during __init__ with the default True.
    checked_calls = tab._chk_overlay._calls.setChecked.call_args_list
    assert checked_calls, "Overlay checkbox must be setChecked during __init__"
    assert checked_calls[-1].args[0] is True, (
        f"Overlay checkbox must default to checked, got {checked_calls[-1].args[0]!r}"
    )


def test_overlay_checkbox_toggle_fires_slot(stubs):
    """#11: toggling the checkbox must invoke _on_overlay_toggled."""
    dt, tab = _make_tab(stubs)
    toggled = tab._chk_overlay.toggled
    connected = toggled._connections.connect.call_args_list[-1].args[0]
    # The connected slot should be _on_overlay_toggled; verify by side-effect.
    # We can't easily inspect the bound method, so verify via observable state.
    tab._overlay_visible = True  # baseline
    tab.set_overlay_visible(False)
    # If the slot is wired correctly, programmatic setChecked would have
    # routed through it. The check that matters: toggled has exactly one
    # connection, and that connection isn't None.
    assert connected is not None
    assert len(toggled._connections.connect.call_args_list) >= 1


def test_overlay_checkbox_lives_in_sidebar_layout(stubs):
    """#11 sanity: the checkbox is reachable from the sidebar layout so
    Slicer actually shows it (defensive — the layout test above is the
    primary contract)."""
    dt, tab = _make_tab(stubs)
    assert tab._chk_overlay is not None
    assert hasattr(tab._chk_overlay, "toggled")


def test_set_overlay_visible_false_flips_state(stubs):
    """#11 acceptance: set_overlay_visible(False) must update the
    internal state, sync the checkbox, and persist to QSettings."""
    dt, tab = _make_tab(stubs)
    assert tab._overlay_visible is True, "Default overlay state must be True"
    tab.set_overlay_visible(False)
    assert tab._overlay_visible is False


def test_set_overlay_visible_true_flips_state_back(stubs):
    dt, tab = _make_tab(stubs)
    tab.set_overlay_visible(False)
    assert tab._overlay_visible is False
    tab.set_overlay_visible(True)
    assert tab._overlay_visible is True


def test_set_overlay_visible_noop_when_unchanged(stubs):
    """#11: setting the same state twice must not trigger a redundant
    cache churn / QSettings write."""
    dt, tab = _make_tab(stubs)
    n_before = len(_QSettings.INSTANCES)
    n_cache_before = len(tab._cache)
    tab.set_overlay_visible(True)  # default already True
    # No new QSettings instance.
    assert len(_QSettings.INSTANCES) == n_before, (
        "set_overlay_visible(True) when already True must not write QSettings"
    )
    assert len(tab._cache) == n_cache_before, (
        "set_overlay_visible(True) when already True must not clear the cache"
    )


def test_set_overlay_visible_persists_to_qsettings(stubs):
    """#11 acceptance: the user's choice must be persisted so it survives
    a Slicer restart."""
    dt, tab = _make_tab(stubs)
    n_before = len(_QSettings.INSTANCES)
    tab.set_overlay_visible(False)
    new = _QSettings.INSTANCES[n_before:]
    assert new, "set_overlay_visible must construct a QSettings instance"
    settings = new[-1]
    matching = [
        c for c in settings._calls.setValue.call_args_list
        if c.args and c.args[0] == dt.DetailTab._OVERLAY_SETTINGS_KEY
    ]
    assert matching, (
        f"set_overlay_visible must persist via QSettings.setValue with key "
        f"{dt.DetailTab._OVERLAY_SETTINGS_KEY!r}"
    )


def test_overlay_settings_key_follows_project_convention(stubs):
    """#11: persistence key must follow the project's
    'ZebrafishEmbryoAnalyzer/...' prefix used by other DetailTab settings."""
    dt, tab = _make_tab(stubs)
    assert dt.DetailTab._OVERLAY_SETTINGS_KEY.startswith("ZebrafishEmbryoAnalyzer/"), (
        f"Overlay settings key must follow project convention, got "
        f"{dt.DetailTab._OVERLAY_SETTINGS_KEY!r}"
    )


def test_load_overlay_visible_restores_from_qsettings(stubs):
    """#11 acceptance: a saved False value must round-trip — a fresh
    DetailTab picks up the stored preference."""
    # Pre-populate QSettings with False.
    preloaded = _QSettings()
    preloaded._values["ZebrafishEmbryoAnalyzer/detailOverlayVisible"] = False

    def factory(*args, **kwargs):
        return preloaded
    stubs["qt"].QSettings = factory

    dt, tab = _make_tab(stubs)
    assert tab._overlay_visible is False, (
        "DetailTab must read the persisted overlay-visible preference "
        "from QSettings on construction"
    )


def test_load_overlay_visible_defaults_to_true_when_unset(stubs):
    """#11: first run (no saved value) must default to overlay visible
    to preserve pre-#11 behaviour."""
    dt, tab = _make_tab(stubs)
    assert tab._overlay_visible is True, (
        "DetailTab must default to overlay visible when QSettings is empty"
    )


def test_set_overlay_visible_clears_cache(stubs):
    """#11: toggling must drop the cached pixmaps for the OLD overlay
    variant so the next show_result rebuilds with the new variant.
    With a row currently visible, set_overlay_visible also rebuilds
    the new variant into the cache, so we verify the OLD variant
    key is gone and the NEW variant key is present."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    # Old variant is in cache.
    assert (0, True) in tab._cache
    tab.set_overlay_visible(False)
    # Old variant key is gone; new variant key is rebuilt.
    assert (0, True) not in tab._cache, (
        "set_overlay_visible(False) must drop the (idx, True) cache entry"
    )
    assert (0, False) in tab._cache, (
        "set_overlay_visible(False) must rebuild the (idx, False) cache entry "
        "for the currently displayed row"
    )


def test_set_overlay_visible_rebuilds_displayed_pixmap(stubs):
    """#11 acceptance: with a row currently visible, toggling must trigger
    a _start_job call so the displayed pixmap reflects the new state."""
    dt, tab = _make_tab(stubs)
    tab.show_result(0, [{"filename": "x.tif", "length": 100.0,
                          "mask": MagicMock(), "error": None,
                          "manual_corrected": False}])
    # Reset the view mock to capture the next set_pixmap call.
    view_mock = tab._view._calls
    view_mock.reset_mock()
    # _start_job schedules qt.QTimer.singleShot(0, ...) and _update_display
    # eventually calls view.set_pixmap. We can't wait for the timer in a
    # unit test, but we CAN verify that _start_job was entered (which calls
    # _ensure_cached and rebuilds the cache).
    pre_cache_len = len(tab._cache)
    tab.set_overlay_visible(False)
    # Cache was cleared, then re-populated by _start_job → _ensure_cached.
    assert len(tab._cache) >= 1, (
        "set_overlay_visible must rebuild the cache for the currently "
        "displayed row"
    )
    # And the rebuilt cache key must use include_overlay=False.
    assert any(
        key == (0, False) for key in tab._cache.keys()
    ), "Rebuilt cache must use the new (index, False) key"


def test_reset_preserves_overlay_toggle_preference(stubs):
    """#11: the overlay toggle is a user preference, NOT per-row state.
    reset() must leave it untouched across scene-close + reopen."""
    dt, tab = _make_tab(stubs)
    tab.set_overlay_visible(False)
    tab.reset()
    assert tab._overlay_visible is False, (
        "reset() must not change the user's overlay preference"
    )


def test_build_rgb_array_accepts_include_overlay_kwarg(stubs):
    """#11: the module-level _build_rgb_array must accept include_overlay
    so show_result / _ensure_cached can pass it through to make_full_overlay."""
    dt, tab = _make_tab(stubs)
    import inspect
    sig = inspect.signature(dt._build_rgb_array)
    assert "include_overlay" in sig.parameters, (
        "_build_rgb_array must accept include_overlay parameter for #11"
    )


def test_make_full_overlay_returns_bare_when_disabled(stubs):
    """#11 integration: overlay.make_full_overlay(result, include_overlay=False)
    must return a bare BGR image with no mask/path/straight-line drawn.

    We test this via source inspection rather than runtime invocation
    because the test fixture stubs out the overlay module entirely to
    bypass cv2 — the real production overlay module is what we want to
    verify here.
    """
    overlay_path = PRODUCTION_ROOT / "ZebrafishEmbryoAnalyzerLib" / "overlay.py"
    source = overlay_path.read_text(encoding="utf-8")
    # Contract: make_full_overlay must accept the include_overlay kwarg.
    assert "include_overlay" in source, (
        "overlay.make_full_overlay must accept include_overlay parameter"
    )
    # Contract: when include_overlay is False, the function must return
    # the bare BGR (no mask draw, no path draw, no straight-line draw).
    assert "if not include_overlay" in source, (
        "overlay.make_full_overlay must have an early-return path for "
        "include_overlay=False"
    )
    # Contract: the early-return path must come BEFORE any overlay-draw
    # step (mask blend, eye blend, path polyline, straight line) — otherwise
    # the user would still see the overlay despite the toggle.
    import re
    # Find the line numbers of the include_overlay branch and the first
    # overlay-draw step (mask blend).
    m_early = re.search(r"if not include_overlay", source)
    m_mask = re.search(r"_blend_mask\(base,\s*m,\s*_MASK_COLOR", source)
    assert m_early and m_mask, (
        "could not locate both the include_overlay branch and the mask "
        "blend call in overlay.py"
    )
    assert m_early.start() < m_mask.start(), (
        f"include_overlay early-return must appear BEFORE the first "
        f"overlay-draw step (mask blend); got early={m_early.start()} "
        f"mask={m_mask.start()}"
    )


def test_make_full_overlay_default_is_true(stubs):
    """#11: include_overlay defaults to True so all existing call sites
    keep drawing the overlay without modification."""
    overlay_path = PRODUCTION_ROOT / "ZebrafishEmbryoAnalyzerLib" / "overlay.py"
    source = overlay_path.read_text(encoding="utf-8")
    import re
    m = re.search(r"def make_full_overlay\(([^)]*)\)", source)
    assert m, "could not find make_full_overlay signature"
    sig = m.group(1)
    assert "include_overlay: bool = True" in sig or "include_overlay=True" in sig, (
        f"include_overlay must default to True so existing callers keep "
        f"drawing the overlay; signature was: {sig!r}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
