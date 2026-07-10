"""
Settings tab — destructive actions that don't belong in the fix-workflow:
model cache deletion (issue #46). Python package uninstall is tracked in #50
and will reuse this tab's "Python dependencies" section once implemented.

Per project convention (feedback_slicer_no_excessive_popups.md), this tab
uses persistent buttons, never modal confirmation pop-ups. The dedicated
tab placement is itself the "are you sure?" affordance.
"""

from pathlib import Path

import qt

from ZebrafishEmbryoAnalyzerLib.model_manifest import (
    _CACHE_DIR,
    collect_all_model_entries,
    get_cached_path,
)


# ---------------------------------------------------------------------------
# Pure helpers (no Qt widgets touched) — testable in isolation
# ---------------------------------------------------------------------------

def _format_size(n_bytes: int) -> str:
    """Return a human-readable size string. Returns 'missing' for non-positive sizes."""
    if n_bytes <= 0:
        return "missing"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def _list_cached_models() -> list:
    """Return one row per manifest entry describing its on-disk state.

    Each row is a dict: {id, label, filename, size_text, exists}.
    'missing' rows remain listed (so the user sees what would be re-downloaded
    after a fresh delete-all) but cannot be checked for individual deletion.
    """
    rows = []
    for entry_id, entry in collect_all_model_entries().items():
        cached = get_cached_path(entry)
        try:
            if cached.exists() and cached.stat().st_size > 0:
                rows.append({
                    "id": entry_id,
                    "label": entry.get("label", entry_id),
                    "filename": entry["filename"],
                    "size_text": _format_size(cached.stat().st_size),
                    "exists": True,
                })
            else:
                rows.append({
                    "id": entry_id,
                    "label": entry.get("label", entry_id),
                    "filename": entry["filename"],
                    "size_text": "missing",
                    "exists": False,
                })
        except OSError:
            rows.append({
                "id": entry_id,
                "label": entry.get("label", entry_id),
                "filename": entry["filename"],
                "size_text": "missing",
                "exists": False,
            })
    return rows


def _delete_files(paths) -> int:
    """Unlink each existing file in `paths`. Returns count actually deleted."""
    n = 0
    for p in paths:
        try:
            p = Path(p)
            if p.exists() and p.is_file():
                p.unlink()
                n += 1
        except OSError:
            pass
    return n


def _wipe_cache_directory(cache_dir=None) -> int:
    """Delete every regular file directly inside cache_dir (not the dir itself,
    not nested files inside subdirectories). Returns count actually deleted.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(_CACHE_DIR)
    n = 0
    try:
        for entry in cache_dir.iterdir():
            try:
                if entry.is_file():
                    entry.unlink()
                    n += 1
            except OSError:
                pass
    except OSError:
        pass
    return n


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class SettingsTab(qt.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # List of (entry_id, checkbox, size_label_widget, exists)
        self._row_widgets = []
        self._build_ui()

    def _build_ui(self):
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)

        container = qt.QWidget()
        layout = qt.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = qt.QLabel("Model cache")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        intro = qt.QLabel(
            "Files downloaded from HuggingFace are cached here. Removing a "
            "model file requires re-download on the next analysis run."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(intro)

        self._select_all = qt.QCheckBox("Select all")
        layout.addWidget(self._select_all)
        self._select_all.toggled.connect(self._on_select_all_toggled)

        self._rows_container = qt.QWidget()
        self._rows_layout = qt.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addWidget(self._rows_container)

        self._btn_delete_selected = qt.QPushButton("Delete selected models")
        layout.addWidget(self._btn_delete_selected)
        self._btn_delete_all = qt.QPushButton("Delete all cached models")
        layout.addWidget(self._btn_delete_all)

        layout.addStretch(1)
        scroll.setWidget(container)

        outer = qt.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._btn_delete_selected.clicked.connect(self._on_delete_selected)
        self._btn_delete_all.clicked.connect(self._on_delete_all)

        self.refresh()

    def refresh(self):
        """Rebuild the per-model list from current disk state."""
        # Drop existing rows cleanly.
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._row_widgets = []

        rows = _list_cached_models()
        for row in rows:
            row_widget = qt.QWidget()
            hbox = qt.QHBoxLayout(row_widget)
            hbox.setContentsMargins(2, 2, 2, 2)
            hbox.setSpacing(6)
            cb = qt.QCheckBox()
            # Only allow per-row checking when the file actually exists;
            # "missing" rows stay visible but un-actionable.
            cb.setEnabled(row["exists"])
            label = qt.QLabel(f"{row['label']}")
            label.setToolTip(row["filename"])
            size = qt.QLabel(row["size_text"])
            size.setStyleSheet("color: #666;")
            size.setAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
            hbox.addWidget(cb)
            hbox.addWidget(label, 1)
            hbox.addWidget(size)
            self._rows_layout.addWidget(row_widget)
            cb.toggled.connect(self._update_button_labels)
            self._row_widgets.append((row["id"], cb, size, row["exists"]))

        # Reset the master toggle to "Select all" / unchecked (avoid cascading
        # toggled() emissions back into _update_button_labels during rebuild).
        blocker = qt.QSignalBlocker(self._select_all)
        try:
            self._select_all.setChecked(False)
        finally:
            del blocker

        self._update_button_labels()

    # ----- signal handlers -----

    def _on_select_all_toggled(self, checked):
        for _id, cb, _size, exists in self._row_widgets:
            if exists and cb.isEnabled():
                cb.setChecked(checked)
        self._update_button_labels()

    def _on_delete_selected(self):
        entries = collect_all_model_entries()
        targets = [
            get_cached_path(entries[entry_id])
            for entry_id, cb, _size, exists in self._row_widgets
            if exists and cb.isChecked()
        ]
        if not targets:
            return
        n = _delete_files(targets)
        try:
            import slicer
            slicer.util.showStatusMessage(
                f"Deleted {n} cached model file(s). They will be re-downloaded on next analysis.",
                5000,
            )
        except Exception:
            pass
        self.refresh()

    def _on_delete_all(self):
        n = _wipe_cache_directory()
        try:
            import slicer
            slicer.util.showStatusMessage(
                f"Cleared model cache ({n} file(s)). They will be re-downloaded on next analysis.",
                5000,
            )
        except Exception:
            pass
        self.refresh()

    # ----- state-driven enable / disable -----

    def set_actions_enabled(self, enabled):
        """Disable both destructive buttons (and per-row checkboxes) while
        inference or model download is in flight. Called from the parent
        widget at every start/finish point of those operations.
        """
        has_selection = self._has_selection()
        has_any = self._has_any_existing()
        self._btn_delete_selected.setEnabled(enabled and has_selection)
        self._btn_delete_all.setEnabled(enabled and has_any)
        for _id, cb, _size, exists in self._row_widgets:
            cb.setEnabled(enabled and exists)
        self._select_all.setEnabled(enabled and has_any)

    # ----- private helpers -----

    def _update_button_labels(self):
        n_selected = sum(
            1 for _id, cb, _size, exists in self._row_widgets
            if exists and cb.isChecked()
        )
        n_enabled = sum(
            1 for _id, _cb, _size, exists in self._row_widgets if exists
        )
        all_selected = (n_selected == n_enabled) and n_enabled > 0

        blocker = qt.QSignalBlocker(self._select_all)
        try:
            self._select_all.setText("Deselect all" if all_selected else "Select all")
            self._select_all.setChecked(all_selected)
        finally:
            del blocker

        self._btn_delete_selected.setText(
            f"Delete selected models ({n_selected})" if n_selected > 0
            else "Delete selected models"
        )

        self._btn_delete_selected.setEnabled(n_selected > 0)
        self._btn_delete_all.setEnabled(n_enabled > 0)

    def _has_selection(self):
        return any(
            exists and cb.isChecked()
            for _id, cb, _size, exists in self._row_widgets
        )

    def _has_any_existing(self):
        return any(exists for _id, _cb, _size, exists in self._row_widgets)
