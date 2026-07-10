"""
Gallery tab — scrollable grid of result thumbnails.

Click a thumbnail -> emits index via on_select callback.
populate(results)  — rebuild grid from result list.
update_thumb(index, rgb_array) — update single thumbnail in-place.
"""

import qt
import numpy as np


THUMB_SIZE   = 150
BORDER_OK    = "2px solid #4CAF50"
BORDER_WARN  = "2px solid #FFC107"
BORDER_ERROR = "2px solid #F44336"
BORDER_LOADING = "2px solid #555555"


class _ClickableLabel(qt.QLabel):
    def __init__(self, idx, on_select, loaded=True):
        super().__init__()
        self._idx = idx
        self._on_select = on_select
        self._loaded = loaded

    def mousePressEvent(self, event):
        if self._loaded:
            self._on_select(self._idx)


def _numpy_to_qpixmap(rgb_array: np.ndarray) -> "qt.QPixmap":
    from PIL import Image as PILImage
    import io
    arr = np.ascontiguousarray(rgb_array.clip(0, 255).astype("uint8"))
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="BMP")
    data = qt.QByteArray(buf.getvalue())
    pixmap = qt.QPixmap()
    pixmap.loadFromData(data)
    return pixmap


class GalleryTab(qt.QWidget):
    def __init__(self, on_select):
        super().__init__()
        self._on_select = on_select
        self._thumbnails = []
        self._cells = []
        self._n_cols = 0

        scroll = qt.QScrollArea()
        # Issue #16 (grid-alignment layer): with widgetResizable=True Qt
        # resizes the inner widget to the viewport size and distributes
        # any extra vertical space between rows and any extra horizontal
        # space across columns — even when setColumnStretch(col, 0) and
        # no setRowStretch are set. The visible effect is rows spread far
        # apart vertically and, when fewer cells than columns, cells
        # spread across the full row width. Setting widgetResizable=False
        # makes the inner widget size to its grid content instead; the
        # alignment flag then pins the content to the top-left of the
        # viewport, and the horizontal scrollbar is suppressed because
        # cells in a thumbnail grid should clip at the edge rather than
        # scroll sideways (vertical scrolling remains enabled).
        scroll.setWidgetResizable(False)
        scroll.setAlignment(qt.Qt.AlignTop | qt.Qt.AlignLeft)
        scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self._scroll = scroll

        self._container = qt.QWidget()
        self._grid      = qt.QGridLayout(self._container)
        self._grid.setSpacing(6)
        scroll.setWidget(self._container)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(scroll)

    def populate(self, results: list) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbnails = []
        self._cells = []
        self._n_cols = 0

        from ZebrafishEmbryoAnalyzerLib.overlay import make_overlay

        for i in range(len(results)):
            r = results[i]
            loaded = r.get("original") is not None
            thumb_rgb = make_overlay(r, thumbnail_size=THUMB_SIZE)
            pixmap    = _numpy_to_qpixmap(thumb_rgb)

            label = _ClickableLabel(i, self._on_select, loaded=loaded)
            label.setPixmap(pixmap)
            label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
            label.setScaledContents(True)

            if not loaded:
                border = BORDER_LOADING
            elif r.get("error"):
                border = BORDER_ERROR
            elif r.get("length") is None:
                border = BORDER_WARN
            else:
                border = BORDER_OK
            label.setStyleSheet(f"border: {border};")

            caption = qt.QLabel()
            caption.setFixedWidth(THUMB_SIZE)
            caption.setAlignment(qt.Qt.AlignTop | qt.Qt.AlignHCenter)
            caption.setWordWrap(False)
            caption.setStyleSheet("font-size: 10px;")
            # Issue #16 (re-refinement): always reserve space for two lines of
            # caption, even if the second line carries no text. This makes every
            # cell in any row the same height, so the gallery never shows a
            # vertical gap between thumbnail and caption in mixed-caption rows.
            caption.setMinimumHeight(2 * caption.fontMetrics().lineSpacing() + 4)
            caption.setToolTip(r["filename"])
            _elided = caption.fontMetrics().elidedText(
                r["filename"], qt.Qt.ElideRight, THUMB_SIZE
            )
            _mparts = []
            if r.get("error"):
                _mparts.append("ERROR")
            else:
                if r.get("length")    is not None: _mparts.append(f"{r['length']:.0f} µm")
                if r.get("curvature") is not None: _mparts.append(f"Cls {r['curvature']}")
            caption.setText(_elided + ("\n" + " | ".join(_mparts) if _mparts else ""))

            cell = qt.QWidget()
            cell_layout = qt.QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            cell_layout.setSpacing(2)
            cell_layout.addWidget(label)
            cell_layout.addWidget(caption)
            # Issue #16 (re-refinement): bottom-stretch keeps any extra cell
            # height (shouldn't happen now that captions reserve two lines,
            # but defensive against future size variation) below the caption,
            # never between image and caption.
            cell_layout.addStretch(1)

            self._cells.append(cell)
            self._thumbnails.append(label)

        self._reflow()
        # Issue #51 (follow-up): on the very first populate() after opening
        # the module, self.width can still reflect a stale/default size —
        # Qt hasn't yet processed the module panel's own pending resize/show
        # events at this point in the synchronous Load Folder... handler.
        # _reflow() then locks in a wrong column count from that stale
        # width, and nothing corrects it until some later event (e.g.
        # leaving and re-entering the module) forces a real resizeEvent.
        # Scheduling a second reflow for the next event-loop tick lets it
        # re-run once the real geometry is available; _reflow()'s own
        # early-return makes this a no-op if the column count already
        # matched.
        qt.QTimer.singleShot(0, self._reflow)

    def update_thumb(self, index: int, rgb_array: np.ndarray) -> None:
        """Update a single thumbnail — builds thumb from full-res rgb on main thread."""
        if index >= len(self._thumbnails):
            return
        from ZebrafishEmbryoAnalyzerLib.overlay import make_overlay
        stub = {"original": rgb_array, "mask": None, "length": None, "error": None}
        thumb_rgb = make_overlay(stub, thumbnail_size=THUMB_SIZE)
        self.update_thumb_prebuilt(index, thumb_rgb)

    def update_thumb_prebuilt(self, index: int, thumb_rgb: np.ndarray) -> None:
        """Update a single thumbnail from a pre-built 150px RGB array (fast, no resize)."""
        if index >= len(self._thumbnails):
            return
        pixmap = _numpy_to_qpixmap(thumb_rgb)
        label = self._thumbnails[index]
        label.setPixmap(pixmap)
        label._loaded = True
        label.setStyleSheet(f"border: {BORDER_WARN};")

    def _reflow(self):
        if not self._cells:
            return
        spacing = self._grid.spacing
        cols = max(1, self.width // (THUMB_SIZE + spacing))
        if cols == self._n_cols:
            return
        self._n_cols = cols
        for i, cell in enumerate(self._cells):
            row, col = divmod(i, cols)
            self._grid.addWidget(cell, row, col)
        # Issue #28: when fewer images are loaded than columns, thumbnails
        # were spread across the full row width with large gaps instead of
        # sitting next to each other on the left. Setting column stretch to
        # 0 disables horizontal expansion so columns size to their content
        # (THUMB_SIZE + 2*2px margin) and any remaining horizontal space
        # accumulates on the right side of the row.
        for col in range(cols):
            self._grid.setColumnStretch(col, 0)
        # Issue #51: with widgetResizable(False) the scroll area no longer
        # auto-resizes the container to fit the grid. addWidget() only
        # invalidates the layout — actual geometry recompute is deferred to a
        # posted LayoutRequest event, which a hidden widget (Gallery not the
        # active tab yet, e.g. right after Load Folder...) never receives. So
        # adjustSize() alone can read a stale size hint. activate() forces
        # the grid to recompute immediately regardless of visibility, then
        # adjustSize() resizes the container to the now-current hint.
        self._grid.activate()
        self._container.adjustSize()
        # Issue #51 (root cause): with widgetResizable(False), QScrollArea
        # only repositions/repaints its scrolled widget from inside its own
        # resizeEvent() handler — it never notices when *we* resize the
        # container from here. A genuine top-level resize that happens to
        # change the scroll area's own size (e.g. Slicer's shell-layout dock
        # resize on module re-entry) triggers that handler and "fixes" the
        # gallery; a plain window resize that leaves the module panel's own
        # width unchanged does not, and neither does the container resize
        # above. Sending the scroll area a synthetic resize event forces the
        # same internal repositioning without depending on an unrelated
        # ancestor happening to resize it.
        scroll_size = self._scroll.size
        qt.QApplication.sendEvent(self._scroll, qt.QResizeEvent(scroll_size, scroll_size))
        # Issue #16: the previous code called setRowStretch(rows, 1) on a
        # notional empty row to "push content up". This interacted
        # inconsistently with variable row heights (multi-line captions are
        # taller than single-line ones): the first row ended up with
        # noticeably more space below it than subsequent rows. QGridLayout
        # already positions each row's content at the top of its allotted
        # space by default, so the explicit stretch is unnecessary — and
        # harmful when row heights differ. Removing it gives uniform
        # spacing between rows.

    def resizeEvent(self, event):
        self._reflow()
