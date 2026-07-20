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
        self._manual_status.setStyleSheet("font-size: 11px; color: #aaa; padding: 2px;")
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
            btn.setStyleSheet("font-size: 16px;")

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
        self._actions_heading.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #888;"
            " padding-top: 4px;"
        )

        # Default state — widget.py will overwrite via set_stale() right
        # after show_result() for each navigation. Tests that construct a
        # bare DetailTab() without MRML state stay covered by this default.
        self._current_is_stale = False

        # Issue #67: filename label + status badge live at the top of the
        # sidebar so the user always sees the current row's identity + state.
        self._filename_label = qt.QLabel("")
        self._filename_label.setWordWrap(True)
        self._filename_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px;")

        self._status_badge = qt.QLabel("")
        self._status_badge.setAlignment(qt.Qt.AlignCenter)
        self._status_badge.setMaximumHeight(24)
        # Initial badge state — "Not analyzed" so the user has feedback
        # before any image is selected. show_result() will overwrite via
        # _update_status_badge() once a row is shown.
        self._update_status_badge({"filename": "", "error": None})

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

        # Translucent error banner shown only when the current row has an error
        # (or is stale — STALE_ERROR_MESSAGE rides the same channel, see
        # mrml.py:440). Translucent so it reads correctly in both light and
        # dark Slicer themes.
        self._error_banner = qt.QLabel("")
        self._error_banner.setWordWrap(True)
        self._error_banner.setVisible(False)
        self._error_banner.setStyleSheet(
            "font-size: 11px; padding: 6px; border-radius: 3px;"
            " background: rgba(244, 67, 54, 180); color: white;"
        )

        # Issue #67: 5-row Measurements grid with `—` placeholders so the
        # sidebar height stays constant across images — never hide a row.
        # Tracked as a list of (field, value) pairs in display order to make
        # populating them in show_result() a simple loop.
        self._measurements = []
        for _label_text in ("Length", "Curvature class", "Length/straight ratio",
                            "Eye area", "Eye diameter"):
            label = qt.QLabel(f"{_label_text}:")
            label.setStyleSheet("font-size: 11px; color: #888;")
            value = qt.QLabel("—")
            value.setStyleSheet("font-size: 12px;")
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
        # #67 — guard the new sidebar widgets with hasattr so lifecycle tests
        # that bypass __init__ via ``object.__new__`` (and only set up the
        # original sidebar widgets) keep working.
        if hasattr(self, "_update_status_badge"):
            self._update_status_badge(result)
        if hasattr(self, "_update_error_banner"):
            self._update_error_banner(result)

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
        """Display an arbitrary RGB numpy array — used for scalebar preview."""
        self._results = []
        self._current_idx = 0
        self._cache.clear()
        self._pending_reset_zoom = True  # preview always resets zoom to fit
        self._manual_mode = False
        self._manual_points = []
        self._view.set_manual_mode(False)
        self._view.clear_dots()
        self._full_pixmap = _numpy_to_qpixmap(rgb)
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._nav_label.setText("")
        # Issue #67/#69: the scale-bar preview should leave the sidebar
        # showing whatever the current image's real state is (this issue
        # silently clears it to a placeholder). #69 reverses this — for
        # now, leave filename/badge/measurements untouched here so this
        # commit is non-invasive; the caller will re-call show_result()
        # afterwards in the normal flow.
        if caption:
            # No caption reaches the sidebar in #67; #69 confirms it's
            # duplicated against the ScaleBar widget's status text and
            # drops it. We accept the argument silently so existing callers
            # don't break.
            pass
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
        if hasattr(self, "_current_is_stale"):
            self._current_is_stale = False
        if hasattr(self, "_status_badge"):
            self._update_status_badge({"filename": "", "error": None})
        if hasattr(self, "_error_banner"):
            self._error_banner.setVisible(False)
        # #68 — hide the relocated Recompute button too so a fresh dataset
        # doesn't inherit a stale button state.
        if hasattr(self, "_btn_recompute"):
            self._btn_recompute.setVisible(False)
            self._btn_recompute.setEnabled(False)
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
        if hasattr(self, "_btn_recompute"):
            show = (
                self._current_is_stale
                and self._recompute_callback is not None
            )
            self._btn_recompute.setVisible(show)
            self._btn_recompute.setEnabled(show)

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
    # Issue #67: stale setter + status badge + error banner
    # ------------------------------------------------------------------

    def set_stale(self, is_stale: bool) -> None:
        """Set whether the currently shown row's segmentation is stale.

        ``widget.py`` is the only caller — it computes staleness from
        :func:`mrml.is_volume_node_stale` (which DetailTab never imports
        directly to keep this class MRML-agnostic + unit-testable without
        a running Slicer scene). Stores the flag; the badge refreshes
        when ``show_result`` runs next, or on demand if a row is updated
        mid-lifetime.

        Issue #68 also drives the relocated "Recompute metrics" button's
        visibility + enabled state — the button is only reachable when
        the row is stale AND a recompute callback has been registered
        (i.e. widget.py has wired up its click handler).
        """
        self._current_is_stale = bool(is_stale)
        # Refresh the badge immediately if a row is currently visible.
        if self._current_filename is not None and self._results:
            self._update_status_badge(self._results[self._current_idx])
            self._update_error_banner(self._results[self._current_idx])
        # Toggle the Recompute button — visible only when both stale AND
        # a callback is registered (a bare DetailTab in unit tests has no
        # callback, so the button stays hidden even when set_stale(True)
        # is called). setEnabled follows visibility so the user can't
        # accidentally click a button that's been hidden.
        if hasattr(self, "_btn_recompute"):
            self._btn_recompute.setVisible(
                self._current_is_stale and self._recompute_callback is not None
            )
            self._btn_recompute.setEnabled(
                self._current_is_stale and self._recompute_callback is not None
            )

    def _update_status_badge(self, result: dict) -> None:
        """Recompute the status badge text + colour from ``result``.

        Priority (issue #67 clarification, evaluated in this exact order):
          1. ``self._current_is_stale`` → "Stale — recompute needed"
          2. ``result.get("error")``   → "Error"
          3. ``result.get("manual_corrected")`` → "Manually corrected"
          4. ``result.get("length")`` or ``result.get("mask")`` non-None → "Analyzed"
          5. otherwise → "Not analyzed"
        """
        # #67 — guard against stub objects in lifecycle tests that bypass
        # __init__ and never set _status_badge. Default to "not stale" so
        # the rest of the priority logic evaluates cleanly.
        is_stale = getattr(self, "_current_is_stale", False)
        if is_stale:
            text = "Stale — recompute needed"
            colour = "rgba(255, 152, 0, 200)"  # amber, translucent
        elif result.get("error"):
            text = "Error"
            colour = "rgba(244, 67, 54, 200)"  # red, translucent
        elif result.get("manual_corrected"):
            text = "Manually corrected"
            colour = "rgba(33, 150, 243, 200)"  # blue, translucent
        elif (result.get("length") is not None
              or result.get("mask") is not None):
            text = "Analyzed"
            colour = "rgba(76, 175, 80, 200)"  # green, translucent
        else:
            text = "Not analyzed"
            colour = "rgba(127, 127, 127, 180)"  # grey, translucent

        # Translucent pill — keep the existing CSS structure; only swap colours.
        if hasattr(self, "_status_badge"):
            self._status_badge.setStyleSheet(
                f"font-size: 11px; padding: 3px 6px; border-radius: 8px;"
                f" background: {colour}; color: white;"
            )
            self._status_badge.setText(text)

    def _update_error_banner(self, result: dict) -> None:
        """Show or hide the error banner based on the current result.

        Note: stale rows always have an error message string set by
        :func:`mrml.mark_volume_node_stale` (see ``STALE_ERROR_MESSAGE``),
        so the banner naturally surfaces stale rows too — the badge and
        the banner share the same source of truth (``result['error']``).
        """
        # #67 — defensive for lifecycle tests that bypass __init__.
        if not hasattr(self, "_error_banner"):
            return
        message = result.get("error") or ""
        if message:
            self._error_banner.setText(str(message))
            self._error_banner.setVisible(True)
        else:
            self._error_banner.setVisible(False)

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

        self._cache.pop(self._current_idx, None)

        self._manual_mode = False
        self._manual_points = []
        self._manual_status.setText("Reverted to auto.")
        self._manual_status.setVisible(True)
        self._btn_revert_auto.setVisible(False)
        self._btn_manual_adjust.setText("✏ Manual Adjust")
        _populate_measurements(self._measurements, result)
        self._update_status_badge(result)
        self._update_error_banner(result)
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

        self._cache.pop(self._current_idx, None)
        self._manual_points = []
        self._full_pixmap = None
        self._pending_reset_zoom = False  # preserve zoom — user was zoomed in for precision
        self._view.clear_dots()           # remove placement dots before overlay rebuild

        self._manual_status.setText("Manual correction applied.")
        self._btn_revert_auto.setVisible(True)
        self._btn_manual_adjust.setText("✏ Redo Manual")
        _populate_measurements(self._measurements, result)
        self._update_status_badge(result)
        self._update_error_banner(result)

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
