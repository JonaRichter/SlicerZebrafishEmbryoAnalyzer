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

    def addWidget(self, w, stretch=0):
        self._calls.addWidget(w, stretch)

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
    AlignHCenter = 4
    StrongFocus = 1
    Key_Escape = 0x01000000  # 0x1b
    Key_Left = 0x01000012
    Key_Right = 0x01000014


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
    om.make_full_overlay = MagicMock(return_value=MagicMock(name="full_overlay"))
    sys.modules["ZebrafishEmbryoAnalyzerLib.overlay"] = om

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
    Manual Adjust row, status, Exclude, Metrics, and Nav row."""
    dt, tab = _make_tab(stubs)

    # Every QVBoxLayout created — the one for self (the tab) and the one for the sidebar.
    # Find the one whose addWidget calls reference the manually-created widgets.
    candidates = list(_QVBoxLayout.all())

    def _holds(layout, widget):
        for c in layout._calls.addWidget.call_args_list:
            if c.args and c.args[0] is widget:
                return True
        return False

    sidebar_layout = None
    for layout in candidates:
        if (_holds(layout, tab._manual_row_widget)
                and _holds(layout, tab._chk_exclude)
                and _holds(layout, tab._metrics_label)):
            sidebar_layout = layout
            break

    assert sidebar_layout is not None, (
        "Could not find the sidebar's QVBoxLayout — it must hold the manual row, "
        "exclude checkbox, and metrics label."
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
    assert tab._metrics_label in add_widgets, (
        "Metrics label must live inside the sidebar"
    )
    sidebar_layout._calls.addLayout.assert_called()
    layout_args = [c.args[0] for c in sidebar_layout._calls.addLayout.call_args_list]
    assert tab._nav_row_layout in layout_args, (
        "Nav row layout must live at the bottom of the sidebar"
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
