"""
Detail tab — full-resolution overlay + metrics for the selected image.

show_result(index, results) — display result at index.
"""

import qt
import numpy as np
from ZebrafishEmbryoAnalyzerLib.zoom_view import ZoomableImageView


def _numpy_to_qpixmap(rgb_array: np.ndarray) -> "qt.QPixmap":
    from PIL import Image as PILImage
    import io
    arr = np.ascontiguousarray(rgb_array.clip(0, 255).astype("uint8"))
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="BMP")  # BMP: no compression, fast encode
    data = qt.QByteArray(buf.getvalue())
    pixmap = qt.QPixmap()
    pixmap.loadFromData(data)
    return pixmap


def _build_rgb_array(result: dict, include_overlay: bool = True) -> np.ndarray:
    """Build the RGB overlay array synchronously on the main thread.

    Issue #11: pass ``include_overlay=False`` to get the bare original
    image (no mask, no path, no straight-line guide). Used by the
    "Show segmentation overlay" checkbox in the sidebar — when the user
    toggles it off, we rebuild each cached pixmap without the overlay.
    """
    from ZebrafishEmbryoAnalyzerLib.overlay import make_full_overlay
    import cv2
    bgr = make_full_overlay(result, include_overlay=include_overlay)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class _ElidedLabel(qt.QLabel):
    """Single-line QLabel that elides with '…' at the right edge.

    QLabel has no built-in single-line + auto-elide mode (the Qt.ElideRight
    flag only exists for QLineEdit/QListView/QPainter), so we keep the
    full text and re-render the elided form whenever the widget resizes
    (e.g. user drags the sidebar splitter wider/narrower) or the text is
    replaced via setText().
    """
    def __init__(self, text=""):
        qt.QLabel.__init__(self, text)
        self.setWordWrap(False)
        self._full_text = text

    def setText(self, text):
        self._full_text = str(text) if text is not None else ""
        try:
            self._apply_elision()
        except Exception:
            # Unit-test stubs (tests/test_detail_tab_sidebar.py::_QLabel)
            # only mock a small subset of QLabel's surface — they don't
            # implement font() / width(), so QFontMetrics.elidedText would
            # raise. Fall back to the raw text so the test path still
            # receives the full filename it wants to inspect.
            qt.QLabel.setText(self, self._full_text)

    def _apply_elision(self):
        fm = qt.QFontMetrics(self.font())
        # width() can be 0 before the widget is shown — guard so Qt doesn't
        # elide to empty for an unparented / not-yet-laid-out label.
        w = max(self.width(), 1)
        qt.QLabel.setText(
            self, fm.elidedText(self._full_text, qt.Qt.ElideRight, w)
        )

    def resizeEvent(self, event):
        qt.QLabel.resizeEvent(self, event)
        try:
            self._apply_elision()
        except Exception:
            pass


class DetailTab(qt.QWidget):
    _SPLITTER_SETTINGS_KEY = "ZebrafishEmbryoAnalyzer/detailSplitterState"
    _OVERLAY_SETTINGS_KEY = "ZebrafishEmbryoAnalyzer/detailOverlayVisible"

    def __init__(self, on_navigate=None, on_back=None, logic=None, on_exclude_change=None):
        super().__init__()
        self._on_navigate = on_navigate
        self._on_back = on_back
        self._logic = logic
        self._on_exclude_change = on_exclude_change  # callable(filename, checked)
        self._current_filename = None
        self._full_pixmap = None
        self._results = []
        self._current_idx = 0
        self._cache = {}          # index → QPixmap  (main thread only)
        self._pending_reset_zoom = True  # True=reset zoom on next pixmap update
        self.setFocusPolicy(qt.Qt.StrongFocus)

        # Issue #11: overlay visibility state, persisted to QSettings so it
        # survives restarts. Default True to match pre-#11 behaviour
        # (segmentation overlay drawn by default). ``set_overlay_visible``
        # flushes the entire pixmap cache on toggle because each cached
        # pixmap is overlay-state-specific.
        self._overlay_visible = self._load_overlay_visible()

        self._manual_mode = False
        self._manual_points = []   # list of (row, col) in original image space
        self._params_getter = None  # set by widget after construction

        # Main image viewer
        self._view = ZoomableImageView()
        self._view._on_navigate = self._on_navigate
        self._view._tap_handler = self._on_image_tap

        self._btn_manual_adjust = qt.QPushButton("✏ Manual Adjust")
        self._btn_revert_auto = qt.QPushButton("↩ Revert to Auto")
        self._btn_revert_auto.setVisible(False)
        self._manual_status = qt.QLabel("")
        self._manual_status.setAlignment(qt.Qt.AlignCenter)
        self._manual_status.setVisible(False)

        self._btn_manual_adjust.clicked.connect(self._on_manual_adjust_clicked)
        self._btn_revert_auto.clicked.connect(self._on_revert_auto_clicked)

        # Navigation buttons
        self._btn_prev = qt.QPushButton("◄")
        self._btn_next = qt.QPushButton("►")
        self._nav_label = qt.QLabel("")
        self._nav_label.setAlignment(qt.Qt.AlignCenter)

        for btn in (self._btn_prev, self._btn_next):
            btn.setFixedWidth(48)
            btn.setFixedHeight(32)

        self._btn_prev.clicked.connect(lambda: self._on_navigate and self._on_navigate(-1))
        self._btn_next.clicked.connect(lambda: self._on_navigate and self._on_navigate(1))

        # Exclude checkbox
        self._chk_exclude = qt.QCheckBox("Exclude from export")
        self._chk_exclude.setEnabled(False)
        self._chk_exclude.toggled.connect(self._on_exclude_toggled)

        # Issue #68: "Recompute metrics" button relocated out of the tab
        # bar (#42 used to place it on the QTabBar) and into the Actions
        # group of the sidebar. Visibility + enabled state are driven by
        # ``set_stale()`` — the button only lights up when the current
        # row's segmentation has been edited since the last analysis.
        # No callback is registered until ``widget.py`` calls
        # ``set_recompute_callback`` during construction, so a bare
        # DetailTab (used in unit tests) stays click-safe.
        self._btn_recompute = qt.QPushButton("⟳ Recompute metrics")
        self._btn_recompute.setToolTip(
            "Segmentation was edited in the Segment Editor — "
            "recompute metrics from the new segmentation."
        )
        self._btn_recompute.setVisible(False)
        self._btn_recompute.setEnabled(False)
        self._recompute_callback = None
        self._btn_recompute.clicked.connect(self._on_recompute_clicked)

        # Issue #68: small section heading above the actions block so the
        # sidebar reads as three labelled regions (Status / Metrics /
        # Actions / Nav) rather than an undifferentiated stack.
        self._actions_heading = qt.QLabel("Actions")
        self._actions_heading.setStyleSheet("font-weight: bold;")

        # Issue #67: filename label + status badge live at the top of the
        # sidebar so the user always sees the current row's identity + state.
        # _ElidedLabel keeps the filename on one line and clips with '…' at
        # the right edge when the sidebar is narrower than the name — QLabel
        # itself has no single-line + auto-elide mode.
        self._filename_label = _ElidedLabel("")
        self._filename_label.setStyleSheet("font-weight: bold;")

        # Canonical source of truth for badge / error banner / Recompute
        # button is the volume node's MRML attributes (ADR 0001 — issue
        # #67). set_current_volume_node() is the entry point used by
        # widget.py at every navigation / scene-rebuild / recompute site.
        # _current_is_stale is a legacy bool fallback kept so the older
        # set_stale(bool) wrapper still drives the badge for tests +
        # bootstrap before any row is shown; it is ignored once a volume
        # node has been set (MRML wins).
        self._current_volume_node = None
        self._current_is_stale = False

        self._status_badge = qt.QLabel("")
        self._status_badge.setAlignment(qt.Qt.AlignCenter)
        # Initial badge state — hidden until a row with an explicit signal
        # (stale / error / manual_corrected) is shown. show_result() will
        # overwrite via _refresh_status_badge() once a row is shown. The
        # badge used to say "Not analyzed" as a default; that wording was
        # removed because it conflated absence-of-data with
        # absence-of-signal and caused the #67 inconsistency across scene
        # save/reload.
        if hasattr(self, "_status_badge"):
            self._status_badge.setVisible(False)

        # Issue #11: "Show segmentation overlay" checkbox — when unchecked,
        # the user sees the bare original image with no mask/path/straight
        # line drawn. State is persisted to QSettings (see
        # _load_overlay_visible / _save_overlay_visible). blockSignals
        # during the initial setChecked so the toggle handler doesn't fire
        # during __init__ (it would clear the cache for nothing).
        self._chk_overlay = qt.QCheckBox("Show segmentation overlay")
        self._chk_overlay.blockSignals(True)
        self._chk_overlay.setChecked(self._overlay_visible)
        self._chk_overlay.blockSignals(False)
        self._chk_overlay.toggled.connect(self._on_overlay_toggled)

        # Error banner shown only when the current row has an error (or is
        # stale — STALE_ERROR_MESSAGE rides the same channel, see mrml.py:440).
        # Bold text in the theme's default colour — no hard-coded background,
        # so it reads correctly in both light and dark Slicer themes.
        self._error_banner = qt.QLabel("")
        self._error_banner.setWordWrap(True)
        self._error_banner.setVisible(False)
        self._error_banner.setStyleSheet("font-weight: bold;")

        # Issue #67: 5-row Measurements grid with `—` placeholders so the
        # sidebar height stays constant across images — never hide a row.
        # Tracked as a list of (field, value) pairs in display order to make
        # populating them in show_result() a simple loop. No per-label
        # setStyleSheet — labels take the theme's default muted colour
        # automatically (Qt's QPalette.Mid/PlaceholderText role).
        self._measurements = []
        for _label_text in ("Length", "Curvature class", "Length/straight ratio",
                            "Eye area", "Eye diameter"):
            label = qt.QLabel(f"{_label_text}:")
            value = qt.QLabel("—")
            value.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
            self._measurements.append((label, value))

        _nav_row = qt.QHBoxLayout()
        _nav_row.addStretch(1)
        _nav_row.addWidget(self._btn_prev)
        _nav_row.addWidget(self._nav_label)
        _nav_row.addWidget(self._btn_next)
        _nav_row.addStretch(1)
        self._nav_row_layout = _nav_row

        _manual_row = qt.QHBoxLayout()
        _manual_row.addStretch(1)
        _manual_row.addWidget(self._btn_manual_adjust)
        _manual_row.addWidget(self._btn_revert_auto)
        _manual_row.addStretch(1)

        # Container widget so we can hide the entire row reliably in PythonQt
        self._manual_row_widget = qt.QWidget()
        self._manual_row_widget.setLayout(_manual_row)
        self._manual_row_widget.setVisible(False)

        # Issue #67: 5-row Measurements grid — label/value pairs, never hidden,
        # so the sidebar height is constant as the user navigates between
        # images that lack different metric fields.
        self._measurements_grid = qt.QGridLayout()
        self._measurements_grid.setHorizontalSpacing(12)
        self._measurements_grid.setVerticalSpacing(4)
        for _row_i, (_lbl, _val) in enumerate(self._measurements):
            self._measurements_grid.addWidget(_lbl, _row_i, 0)
            self._measurements_grid.addWidget(_val, _row_i, 1)

        # Issue #67 header — "#67 will further split this" is the issue's own wording;
        # #68 will label the manual/exclude block as "Actions" with visual headings.
        self._sidebar = qt.QWidget()
        sidebar_layout = qt.QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        # Top of the sidebar: filename + status badge (one row, badge right-aligned)
        top_row = qt.QHBoxLayout()
        top_row.addWidget(self._filename_label, 1)
        top_row.addWidget(self._status_badge, 0, qt.Qt.AlignVCenter)
        sidebar_layout.addLayout(top_row, 0)
        sidebar_layout.addWidget(self._error_banner, 0)
        sidebar_layout.addLayout(self._measurements_grid, 0)
        # Issue #11: overlay toggle sits between the Measurements section
        # and the Actions section — it controls how the displayed image
        # is rendered (overlay on/off), not how the metadata is computed.
        sidebar_layout.addWidget(self._chk_overlay, 0)

        # Issue #68: Actions group — Manual Adjust/Revert row, manual status
        # label, the Exclude checkbox, and the relocated Recompute metrics
        # button all live under the "Actions" heading. Keeping these as
        # individual widgets (not nested in a child QWidget) lets the
        # existing ``_manual_row_widget`` / ``_chk_exclude`` tests in
        # test_lifecycle.py continue to assert on the same handles.
        sidebar_layout.addWidget(self._actions_heading, 0)
        sidebar_layout.addWidget(self._manual_row_widget, 0)
        sidebar_layout.addWidget(self._manual_status, 0)
        sidebar_layout.addWidget(self._chk_exclude, 0)
        sidebar_layout.addWidget(self._btn_recompute, 0)

        sidebar_layout.addStretch(1)
        sidebar_layout.addLayout(self._nav_row_layout, 0)
        self._sidebar.setMinimumWidth(220)

        # Image on the left, sidebar on the right, user-resizable, persisted.
        self._splitter = qt.QSplitter(qt.Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(4)
        self._splitter.addWidget(self._view)
        self._splitter.addWidget(self._sidebar)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        try:
            self._splitter.splitterMoved.connect(
                lambda *_: self.save_splitter_state()
            )
        except Exception:
            pass
        self._restore_splitter_state()

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_result(self, index: int, results: list) -> None:
        # Exit manual mode when navigating to a new image
        if self._manual_mode:
            self._manual_mode = False
            self._manual_points = []
            self._manual_status.setVisible(False)
            self._view.set_manual_mode(False)
            self._view.clear_dots()

        self._results = results
        self._current_idx = index
        result = results[index]
        self._current_filename = result["filename"]
        self._chk_exclude.setEnabled(True)

        if hasattr(self, "_filename_label"):
            self._filename_label.setText(str(self._current_filename))
        _populate_measurements(getattr(self, "_measurements", []), result)
        # #67 — drive the badge / error banner / Recompute button from the
        # row's MRML volume node (ADR 0001). Centralised in
        # set_current_volume_node so widget.py and the manual-adjust paths
        # share the same refresh entry point. _current_is_stale is cleared
        # so the legacy bool fallback can't shadow a fresh MRML read.
        if hasattr(self, "_status_badge"):
            self.set_current_volume_node(result.get("_volume_node"))

        # Sync button state — only show after analysis (stubs have length=None)
        analyzed = result.get("length") is not None or result.get("mask") is not None
        self._manual_row_widget.setVisible(analyzed)
        is_corrected = bool(result.get("manual_corrected"))
        self._btn_revert_auto.setVisible(is_corrected)
        self._btn_manual_adjust.setText(
            "✏ Redo Manual" if is_corrected else "✏ Manual Adjust"
        )
        self._manual_status.setText("")
        self._manual_status.setVisible(False)

        self._pending_reset_zoom = True  # navigation → always reset zoom
        # Issue #11: include_overlay tracks the user's "Show segmentation
        # overlay" toggle. The cache is keyed by (index, include_overlay) so
        # toggling doesn't return a stale pixmap from the other variant.
        # ``getattr`` keeps lifecycle tests that bypass __init__ via
        # ``object.__new__`` working — they pre-populate ``_cache[0]``
        # with an integer key and rely on the lookup falling through.
        overlay_visible = getattr(self, "_overlay_visible", True)
        cache_key = (index, overlay_visible)
        if cache_key in self._cache:
            self._full_pixmap = self._cache[cache_key]
        elif index in self._cache:
            # Backwards-compat: legacy integer-keyed cache from before #11.
            self._full_pixmap = self._cache[index]
        else:
            self._cache[cache_key] = _numpy_to_qpixmap(
                _build_rgb_array(results[index], include_overlay=overlay_visible)
            )
            self._full_pixmap = self._cache[cache_key]
        qt.QTimer.singleShot(0, self._update_display)

        self._update_nav_state()

    def show_raw_image(self, rgb: np.ndarray, caption: str = "") -> None:
        """Display an arbitrary RGB numpy array — used for scalebar preview.

        Issue #69: the scale-bar preview used to wipe the sidebar state
        (filename, badge, measurements, results list, pixmap cache,
        manual-mode state, nav buttons). That was a footgun — when the
        user clicked "Detect scalebar" in the ScaleBar widget, the
        Detail tab's right-hand sidebar would silently reset to a blank
        placeholder, and the user would lose context (they couldn't see
        which row was being previewed, and any unsaved manual correction
        dots would disappear).

        #69 keeps the sidebar/results state intact and only swaps the
        displayed image. The next ``show_result`` call (from widget.py
        on the next navigation, recompute, etc.) will rebuild the
        sidebar for the actual current row.
        """
        # Don't blow away the sidebar / results / cache / nav state.
        # We only swap the displayed pixmap.
        self._pending_reset_zoom = True  # preview always resets zoom to fit
        self._full_pixmap = _numpy_to_qpixmap(rgb)
        # ``caption`` was previously used to clear the metrics label;
        # #69 drops it silently since the ScaleBar widget already has
        # its own status label next to the Detect button (avoid duplicating
        # the same text into the sidebar). Accept the argument so existing
        # callers don't break.
        del caption
        qt.QTimer.singleShot(0, self._update_display)

    def invalidate_cache(self):
        """Call after a new batch run so stale pixmaps are discarded."""
        self._cache.clear()

    def sync_exclude(self, is_excluded: bool) -> None:
        """Update exclude checkbox state without firing callbacks."""
        self._chk_exclude.blockSignals(True)
        self._chk_exclude.setChecked(is_excluded)
        self._chk_exclude.blockSignals(False)

    def reset(self):
        """Clear all visible and internal state after scene close."""
        self.invalidate_cache()
        self._results = []
        self._current_idx = 0
        self._current_filename = None
        self._full_pixmap = None
        self._manual_mode = False
        self._manual_points = []
        self._view.show_placeholder("Select an image from the Gallery.")
        self._view.set_manual_mode(False)
        # #67 — guard each new sidebar widget with getattr so existing
        # lifecycle tests that bypass __init__ via ``object.__new__``
        # (and only set up the original sidebar widgets) keep working.
        if hasattr(self, "_filename_label"):
            self._filename_label.setText("")
        if hasattr(self, "_measurements"):
            _clear_measurements(self._measurements)
        # #67 — clear the volume node + legacy bool override so the badge /
        # error banner / Recompute button all start from a hidden state
        # rather than inheriting the previous row's signals.
        self._current_volume_node = None
        if hasattr(self, "_current_is_stale"):
            self._current_is_stale = False
        if hasattr(self, "_status_badge"):
            self._refresh_status_badge()
        if hasattr(self, "_error_banner"):
            self._refresh_error_banner()
        # #68 — hide the relocated Recompute button too so a fresh dataset
        # doesn't inherit a stale button state.
        if hasattr(self, "_btn_recompute"):
            self._refresh_recompute_button()
        # #11 — overlay toggle is a user preference, NOT per-row state, so
        # reset() leaves it untouched. The checkbox keeps its current
        # checked state across scene-close + reopen.
        self._nav_label.setText("")
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._manual_row_widget.setVisible(False)
        self._manual_status.setVisible(False)
        self._chk_exclude.blockSignals(True)
        self._chk_exclude.setChecked(False)
        self._chk_exclude.setEnabled(False)
        self._chk_exclude.blockSignals(False)

    def cleanup(self):
        """Invalidate cache."""
        self.save_splitter_state()
        self.invalidate_cache()

    def _on_exclude_toggled(self, checked: bool) -> None:
        if self._current_filename and self._on_exclude_change:
            self._on_exclude_change(self._current_filename, checked)

    # ------------------------------------------------------------------
    # Issue #68: Recompute metrics callback registration + click handler
    # ------------------------------------------------------------------

    def set_recompute_callback(self, callback) -> None:
        """Register the click handler for the Recompute metrics button.

        ``widget.py`` is the only caller — it passes its
        ``_on_recompute_current_detail`` method (or any callable with no
        arguments) so this class stays MRML-agnostic. Storing ``None``
        hides the button (used when widget.py's try/except around button
        creation failed, so the button never becomes a clickable no-op).
        After registration, refresh the button visibility in case a
        row is currently shown and stale.
        """
        self._recompute_callback = callback
        # Refresh the button's enabled/visible state under the new
        # callback, in case a stale row is currently displayed.
        self._refresh_recompute_button()

    def _on_recompute_clicked(self) -> None:
        """Internal click handler — delegates to the registered callback.

        No-op when no callback has been registered yet (e.g. the user
        somehow managed to click the button before widget.py finished
        wiring up). The button is already hidden by ``set_stale`` in
        that case, so this is just defence-in-depth.
        """
        if self._recompute_callback is None:
            return
        try:
            self._recompute_callback()
        except Exception:
            # Don't let a widget-side bug kill the UI thread — the caller
            # already logs exceptions internally.
            pass

    # ------------------------------------------------------------------
    # Issue #11: segmentation overlay toggle + QSettings persistence
    # ------------------------------------------------------------------

    def set_overlay_visible(self, visible: bool) -> None:
        """Programmatically show/hide the segmentation overlay.

        Updates the checkbox state (without re-firing its signal — that
        would cause an infinite loop), persists the choice to QSettings,
        drops every cached pixmap so the next ``show_result`` rebuilds
        the right variant, and refreshes the display if a row is currently
        visible. ``widget.py`` rarely calls this directly; the checkbox's
        own ``toggled`` signal drives it in the normal case.
        """
        visible = bool(visible)
        if visible == self._overlay_visible:
            return  # no-op, avoid pointless cache churn
        self._overlay_visible = visible
        # Sync the checkbox without re-emitting toggled().
        if hasattr(self, "_chk_overlay"):
            self._chk_overlay.blockSignals(True)
            self._chk_overlay.setChecked(visible)
            self._chk_overlay.blockSignals(False)
        # Persist the user's choice.
        self._save_overlay_visible(visible)
        # Drop cached pixmaps — every key includes the overlay flag now,
        # but clearing is cheaper than picking through both variants.
        self._cache.clear()
        # Refresh the currently displayed image, if any.
        if (self._current_filename is not None
                and 0 <= self._current_idx < len(self._results)):
            self._start_job(self._current_idx)

    def _on_overlay_toggled(self, checked: bool) -> None:
        """Checkbox slot — defers to ``set_overlay_visible``."""
        self.set_overlay_visible(checked)

    def _load_overlay_visible(self) -> bool:
        """Read the persisted overlay-visible preference from QSettings.

        Defaults to True (overlay shown) on first run / corrupted value.
        Wrapped in try/except so a stale bytes value from an older build
        doesn't break the constructor.
        """
        try:
            settings = qt.QSettings()
            stored = settings.value(self._OVERLAY_SETTINGS_KEY)
        except Exception:
            return True
        if stored is None:
            return True
        # QSettings returns strings for some backends; coerce defensively.
        if isinstance(stored, bool):
            return stored
        if isinstance(stored, str):
            return stored.strip().lower() not in ("false", "0", "no", "")
        return bool(stored)

    def _save_overlay_visible(self, visible: bool) -> None:
        """Persist the overlay-visible preference to QSettings.

        Best-effort — QSettings may not be available under every test rig,
        so we swallow exceptions and let the next ``cleanup()`` re-attempt.
        """
        try:
            settings = qt.QSettings()
            settings.setValue(self._OVERLAY_SETTINGS_KEY, bool(visible))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Issue #67: MRML-driven status badge + error banner + Recompute button
    # ------------------------------------------------------------------

    def set_current_volume_node(self, vol) -> None:
        """Set the MRML volume node for the currently-shown row.

        Re-derives the status badge, error banner, and Recompute button
        visibility from the volume node's MRML attributes (ADR 0001).
        This is the canonical refresh entry point — widget.py calls it
        at every ``show_result`` site (gallery click, scene reload,
        recompute, results-ready) so the sidebar stays consistent across
        scene save/reload and across rows.

        Priority (issue #67):
          1. ``ZebrafishAnalysis.stale``           → "Stale — recompute needed"
          2. ``ZebrafishAnalysis.error``           → "Error: {message}"
          3. ``ZebrafishAnalysis.manualCorrected`` → "Manually corrected"
          4. none of the above                     → badge hidden

        The "Analyzed" / "Not analyzed" states were dropped: they
        conflated absence-of-data with absence-of-signal and were the
        cause of the #67 inconsistency across scene save/reload. The
        badge now surfaces only when something specific needs the user's
        attention.
        """
        self._current_volume_node = vol
        # Clear the legacy bool override — once a volume node is set, MRML
        # is the source of truth and the bool can't shadow it.
        self._current_is_stale = False
        self._refresh_status_badge()
        self._refresh_error_banner()
        self._refresh_recompute_button()

    def set_stale(self, is_stale: bool) -> None:
        """DEPRECATED — use :meth:`set_current_volume_node` so the badge
        and Recompute button derive from MRML attributes (ADR 0001).

        Kept as a backward-compatible wrapper for any caller that drives
        the badge from a bool. When a volume node has been set, the bool
        is ignored — the truth comes from MRML and we just re-derive.
        When no volume node is set yet (legacy unit tests, bootstrap
        before any row is shown), the bool is honoured.
        """
        if getattr(self, "_current_volume_node", None) is not None:
            # Volume node set → MRML wins, bool is ignored.
            self._refresh_status_badge()
            self._refresh_error_banner()
            self._refresh_recompute_button()
            return
        # No volume node → honour the bool (legacy path).
        self._current_is_stale = bool(is_stale)
        self._refresh_status_badge()
        self._refresh_recompute_button()

    def _refresh_status_badge(self) -> None:
        """Recompute badge text + visibility from MRML (canonical).

        Reads from ``self._current_volume_node`` when set, otherwise
        falls back to ``self._current_is_stale`` (legacy bool used by
        ``set_stale(bool)`` callers that haven't migrated to
        ``set_current_volume_node`` yet + bootstrap before any row is
        shown). When no signal is present the badge is hidden — the old
        "Not analyzed" / "Analyzed" wording was removed in #67 because
        it conflated absence-of-data with absence-of-signal and caused
        the cross-reload inconsistency.
        """
        if not hasattr(self, "_status_badge"):
            return
        vol = getattr(self, "_current_volume_node", None)
        if vol is not None and hasattr(vol, "GetAttribute"):
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                is_volume_node_stale,
                is_volume_node_manual_corrected,
            )
            if is_volume_node_stale(vol):
                self._apply_badge_text("Stale — recompute needed")
                return
            err = self._read_error_attr(vol)
            if err:
                self._apply_badge_text(f"Error: {err}")
                return
            if is_volume_node_manual_corrected(vol):
                self._apply_badge_text("Manually corrected")
                return
            # No MRML signal — hide the badge.
            self._status_badge.setVisible(False)
            return
        # No volume node — legacy bool fallback (tests + bootstrap).
        if getattr(self, "_current_is_stale", False):
            self._apply_badge_text("Stale — recompute needed")
            return
        self._status_badge.setVisible(False)

    def _apply_badge_text(self, text: str) -> None:
        """Set the badge text + bold style + visible in one go.

        Centralised so the badge always reads as part of Slicer's native
        UI (no coloured pill background) per the #67 design decision.
        """
        self._status_badge.setStyleSheet("font-weight: bold;")
        self._status_badge.setText(text)
        self._status_badge.setVisible(True)

    def _read_error_attr(self, vol) -> str:
        """Read the error attribute from a volume node.

        Returns the message string, or ``""`` if the attribute is
        missing/empty or the read raises. Defensive against stubs that
        only implement a subset of the MRML node surface.
        """
        try:
            err = vol.GetAttribute("ZebrafishAnalysis.error")
        except Exception:
            return ""
        return str(err) if err else ""

    def _refresh_error_banner(self) -> None:
        """Show or hide the error banner from the current volume node.

        Note: stale rows always have an error message set by
        :func:`mrml.mark_volume_node_stale` (see ``STALE_ERROR_MESSAGE``),
        so the banner naturally surfaces stale rows too — the badge and
        the banner share the same source of truth.
        """
        if not hasattr(self, "_error_banner"):
            return
        vol = getattr(self, "_current_volume_node", None)
        if vol is not None and hasattr(vol, "GetAttribute"):
            err = self._read_error_attr(vol)
        else:
            # No volume node — banner stays hidden until one is set.
            err = ""
        if err:
            self._error_banner.setText(err)
            self._error_banner.setVisible(True)
        else:
            self._error_banner.setVisible(False)

    def _refresh_recompute_button(self) -> None:
        """Show the Recompute button iff the current row is stale AND a
        callback has been registered.

        ``set_recompute_callback(None)`` hides the button even for stale
        rows — used when widget.py's try/except around button creation
        failed and the button should never be reachable.
        """
        if not hasattr(self, "_btn_recompute"):
            return
        show = self._compute_is_stale() and self._recompute_callback is not None
        self._btn_recompute.setVisible(show)
        self._btn_recompute.setEnabled(show)

    def _compute_is_stale(self) -> bool:
        """Single source of truth for "is the current row stale?".

        Reads from MRML when a volume node is set; falls back to the
        legacy ``_current_is_stale`` bool for un-migrated callers and
        bootstrap before any row is shown.
        """
        vol = getattr(self, "_current_volume_node", None)
        if vol is not None and hasattr(vol, "GetAttribute"):
            try:
                from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale
                return bool(is_volume_node_stale(vol))
            except Exception:
                return False
        return bool(getattr(self, "_current_is_stale", False))

    # ------------------------------------------------------------------
    # Legacy aliases — kept so any external caller still importing the
    # pre-#67 method names keeps working. They forward to the new
    # MRML-driven refreshers so behaviour stays consistent.
    # ------------------------------------------------------------------

    def _update_status_badge(self, result: dict) -> None:
        """Legacy alias — calls :meth:`_refresh_status_badge`.

        Accepts the pre-#67 ``result`` dict but ignores it: badge state
        is now derived from ``self._current_volume_node`` instead.
        """
        self._refresh_status_badge()

    def _update_error_banner(self, result: dict) -> None:
        """Legacy alias — calls :meth:`_refresh_error_banner`.

        Accepts the pre-#67 ``result`` dict but ignores it: banner state
        is now derived from ``self._current_volume_node`` instead.
        """
        self._refresh_error_banner()

    # ------------------------------------------------------------------
    # Splitter width persistence
    # ------------------------------------------------------------------

    def _restore_splitter_state(self) -> None:
        """Issue #66: restore the sidebar's saved width from QSettings, if any.

        Silently no-ops when no saved state exists yet (first run) or when
        the stored value isn't a usable QByteArray (corrupted storage from
        an older build) — the splitter falls back to its sizeHint() layout.
        """
        try:
            settings = qt.QSettings()
            state = settings.value(self._SPLITTER_SETTINGS_KEY)
        except Exception:
            return
        if not state:
            return
        try:
            self._splitter.restoreState(state)
        except Exception:
            # Stale or invalid bytes — leave the splitter on its default sizeHint layout.
            pass

    def save_splitter_state(self) -> None:
        """Persist the current splitter sizes to QSettings.

        Called from :meth:`cleanup` so the user's resize survives widget
        teardown. Also wired live to ``splitterMoved`` so a SIGKILL of
        Slicer doesn't lose the most recent drag.
        """
        if not hasattr(self, "_splitter") or self._splitter is None:
            return
        try:
            settings = qt.QSettings()
            settings.setValue(
                self._SPLITTER_SETTINGS_KEY, self._splitter.saveState()
            )
        except Exception:
            pass

    def _update_nav_state(self) -> None:
        n = len(self._results)
        self._btn_prev.setEnabled(self._current_idx > 0)
        self._btn_next.setEnabled(self._current_idx < n - 1)
        if n > 0:
            self._nav_label.setText(f"{self._current_idx + 1} / {n}")
        else:
            self._nav_label.setText("")

    def _ensure_cached(self, index: int) -> None:
        """Build and cache the pixmap for index synchronously if not yet cached.

        Issue #11: the cache key now includes the overlay-visibility flag so
        toggling the overlay doesn't reuse the wrong variant.
        """
        cache_key = (index, getattr(self, "_overlay_visible", True))
        if cache_key in self._cache:
            return
        if index < 0 or index >= len(self._results):
            return
        self._cache[cache_key] = _numpy_to_qpixmap(
            _build_rgb_array(self._results[index],
                             include_overlay=self._overlay_visible)
        )

    def _start_job(self, index: int) -> None:
        """Synchronously rebuild the overlay for index and update the display."""
        # Issue #11: drop BOTH overlay variants for this index — we want a
        # clean rebuild regardless of which variant the toggle currently picks.
        for key in list(self._cache.keys()):
            if isinstance(key, tuple) and key[0] == index:
                self._cache.pop(key, None)
            elif key == index:
                # Backwards-compat: legacy integer keys from before #11.
                self._cache.pop(key, None)
        self._ensure_cached(index)
        cache_key = (index, self._overlay_visible)
        if cache_key in self._cache:
            self._full_pixmap = self._cache[cache_key]
            qt.QTimer.singleShot(0, self._update_display)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        self._view.set_pixmap(self._full_pixmap, reset_zoom=self._pending_reset_zoom)

    def keyPressEvent(self, event):
        if event.key() == qt.Qt.Key_Escape:
            if self._on_back:
                self._on_back()
            return
        if self._on_navigate:
            if event.key() == qt.Qt.Key_Right:
                self._on_navigate(1)
                return
            if event.key() == qt.Qt.Key_Left:
                self._on_navigate(-1)
                return
        qt.QWidget.keyPressEvent(self, event)

    # ------------------------------------------------------------------
    # Manual correction — tap mode
    # ------------------------------------------------------------------

    def _on_manual_adjust_clicked(self):
        """Enter tap mode to place head/tail points."""
        if not self._results:
            return
        self._manual_mode = True
        self._manual_points = []
        self._manual_status.setText("Click HEAD point (1/2)")
        self._manual_status.setVisible(True)
        self._view.set_manual_mode(True)
        self._view.clear_dots()

    def _on_revert_auto_clicked(self):
        """Restore auto-computed values for current fish."""
        if not self._results:
            return
        result = self._results[self._current_idx]
        self._logic.revert_manual_correction(result)
        # Mirror the flag on the MRML volume node so it survives scene
        # save/reload — previously it only lived in transient ``_results``.
        try:
            from ZebrafishEmbryoAnalyzerLib.mrml import set_volume_node_manual_corrected
            set_volume_node_manual_corrected(result.get("_volume_node"), False)
        except Exception:
            pass

        self._cache.pop(self._current_idx, None)

        self._manual_mode = False
        self._manual_points = []
        self._manual_status.setText("Reverted to auto.")
        self._manual_status.setVisible(True)
        self._btn_revert_auto.setVisible(False)
        self._btn_manual_adjust.setText("✏ Manual Adjust")
        _populate_measurements(self._measurements, result)
        # Re-derive badge + banner from MRML (the manual_corrected flag was
        # just cleared on the volume node above; re-reading it now makes
        # the badge disappear so the user sees the revert reflected).
        if hasattr(self, "_status_badge"):
            self.set_current_volume_node(result.get("_volume_node"))
        self._view.set_manual_mode(False)
        self._view.clear_dots()

        self._pending_reset_zoom = False  # preserve zoom after correction rebuild
        self._start_job(self._current_idx)

    def _on_image_tap(self, row: int, col: int) -> None:
        """Called by ZoomableImageView tap handler with pre-mapped (row, col) coords."""
        if not self._manual_mode:
            return

        self._manual_points.append((row, col))
        self._view.add_dot(
            row, col,
            qt.QColor(0, 220, 0) if len(self._manual_points) == 1 else qt.QColor(220, 0, 0)
        )

        if len(self._manual_points) == 1:
            self._manual_status.setText("Click TAIL point (2/2)")
        elif len(self._manual_points) >= 2:
            self._manual_mode = False
            self._view.set_manual_mode(False)
            self._manual_status.setText("Computing…")
            qt.QTimer.singleShot(0, self._apply_correction)

    def _apply_correction(self):
        """Apply 2-point manual correction to current result and refresh."""
        if len(self._manual_points) < 2:
            return
        result = self._results[self._current_idx]
        params = self._params_getter() if callable(self._params_getter) else {}

        self._logic.apply_manual_correction(
            result, self._manual_points[0], self._manual_points[1], params
        )
        # Mirror the flag on the MRML volume node so it survives scene
        # save/reload — previously it only lived in transient ``_results``.
        try:
            from ZebrafishEmbryoAnalyzerLib.mrml import set_volume_node_manual_corrected
            set_volume_node_manual_corrected(result.get("_volume_node"), True)
        except Exception:
            pass

        self._cache.pop(self._current_idx, None)
        self._manual_points = []
        self._full_pixmap = None
        self._pending_reset_zoom = False  # preserve zoom — user was zoomed in for precision
        self._view.clear_dots()           # remove placement dots before overlay rebuild

        self._manual_status.setText("Manual correction applied.")
        self._btn_revert_auto.setVisible(True)
        self._btn_manual_adjust.setText("✏ Redo Manual")
        _populate_measurements(self._measurements, result)
        # Re-derive badge + banner from MRML (the manual_corrected flag was
        # just set on the volume node above; re-reading it now makes the
        # badge flip to "Manually corrected" so the user sees the change
        # without needing to navigate away and back).
        if hasattr(self, "_status_badge"):
            self.set_current_volume_node(result.get("_volume_node"))

        self._start_job(self._current_idx)


# ---------------------------------------------------------------------------
# Issue #67: Measurements helpers
# ---------------------------------------------------------------------------

# Display order matches self._measurements (built in __init__).
_MEASUREMENT_FORMATS = [
    # key, format callable (value) -> display string. None value → "—".
    ("length",        lambda v: f"{v:.1f} µm"),
    ("curvature",     lambda v: str(v)),
    ("ratio",         lambda v: f"{v:.3f}"),
    ("eye_area",      lambda v: f"{v:.1f} µm²"),
    ("eye_diameter",  lambda v: f"{v:.1f} µm"),
]


def _populate_measurements(rows, result: dict) -> None:
    """Update each (label, value) pair in ``rows`` from the result dict.

    Missing fields render as ``—`` per issue #67 — rows are never hidden
    so the sidebar height stays constant across navigations.
    """
    for row_i, (_lbl, val_widget) in enumerate(rows):
        key, formatter = _MEASUREMENT_FORMATS[row_i]
        value = result.get(key)
        if value is None:
            val_widget.setText("—")
        else:
            try:
                val_widget.setText(formatter(value))
            except Exception:
                # Defensive: a malformed numeric (e.g. from a half-built
                # result dict) shouldn't crash the show_result path —
                # fall back to the raw repr.
                val_widget.setText(repr(value))


def _clear_measurements(rows) -> None:
    """Reset every value cell to ``—``. Used by :meth:`DetailTab.reset`."""
    for _lbl, val_widget in rows:
        val_widget.setText("—")
