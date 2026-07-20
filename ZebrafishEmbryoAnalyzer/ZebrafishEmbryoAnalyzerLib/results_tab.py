"""
Results tab — QTableWidget showing all measurements.
"""

import qt

from ZebrafishEmbryoAnalyzerLib.export import METRIC_KEYS


COLUMNS = [
    ("Filename",              "filename",     str),
    ("Length (µm)",           "length",       lambda v: f"{v:.1f}" if v is not None else ""),
    ("Curvature class",       "curvature",    lambda v: str(v) if v is not None else ""),
    ("Length/straight ratio", "ratio",        lambda v: f"{v:.3f}" if v is not None else ""),
    ("Eye area (µm²)",        "eye_area",     lambda v: f"{v:.1f}" if v is not None else ""),
    ("Eye diameter (µm)",     "eye_diameter", lambda v: f"{v:.1f}" if v is not None else ""),
    ("Edema area (µm²)",      "edema_area",   lambda v: f"{v:.1f}" if v is not None else ""),
    ("Error",                 "error",        lambda v: v or ""),
]

_METRIC_LABELS = {key: label for label, key, _ in COLUMNS}


class ResultsTab(qt.QWidget):
    def __init__(self, on_exclude_change=None):
        super().__init__()
        self._on_exclude_change = on_exclude_change  # callable(filename, metric_key, checked)
        # rows: list of (filename, {metric_key: QCheckBox})
        self._rows = []

        # Issue #74: one exclude checkbox column per active metric key,
        # derived from METRIC_KEYS (itself derived from export.HEADERS) —
        # a new metric column added there automatically gets its own
        # exclude column here without any change to this module.
        self._excl_cols = {key: len(COLUMNS) + i for i, key in enumerate(METRIC_KEYS)}

        self._table = qt.QTableWidget(0, len(COLUMNS) + len(METRIC_KEYS))
        headers = [c[0] for c in COLUMNS] + [
            f"Excl: {_METRIC_LABELS.get(k, k)}" for k in METRIC_KEYS
        ]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.horizontalHeader().setSectionResizeMode(
            0, qt.QHeaderView.Stretch
        )
        self._table.editTriggers = qt.QAbstractItemView.NoEditTriggers
        self._table.selectionBehavior = qt.QAbstractItemView.SelectRows

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self._table)

    def populate(self, results, excluded=None) -> None:
        """``excluded``: ``{filename: set(excluded_metric_keys)}`` (issue #74).

        A plain ``set`` of filenames is still accepted for backward
        compatibility with any remaining caller that has not migrated yet —
        treated as "every metric excluded for that filename".
        """
        excluded = _normalize_excluded(excluded)
        n = len(results)
        self._table.rowCount = n
        self._rows = []
        for row in range(n):
            r = results[row]
            filename = r["filename"]
            for col in range(len(COLUMNS)):
                _, key, fmt = COLUMNS[col]
                val = r.get(key)
                self._table.setItem(row, col, qt.QTableWidgetItem(fmt(val)))
            row_excluded = excluded.get(filename, set())
            is_error = bool(r.get("error"))
            checkboxes = {}
            for key in METRIC_KEYS:
                chk = qt.QCheckBox()
                chk.setChecked(is_error or key in row_excluded)
                chk.toggled.connect(
                    lambda checked, fn=filename, k=key: (
                        self._on_exclude_change and self._on_exclude_change(fn, k, checked)
                    )
                )
                self._table.setCellWidget(row, self._excl_cols[key], chk)
                checkboxes[key] = chk
            self._rows.append((filename, checkboxes))

    def sync_exclude(self, excluded) -> None:
        """Update checkbox states from outside without firing callbacks.

        ``excluded``: ``{filename: set(excluded_metric_keys)}`` (or a plain
        ``set`` of filenames, normalized the same way as ``populate``).
        """
        excluded = _normalize_excluded(excluded)
        for fn, checkboxes in self._rows:
            row_excluded = excluded.get(fn, set())
            for key, chk in checkboxes.items():
                chk.blockSignals(True)
                chk.setChecked(key in row_excluded)
                chk.blockSignals(False)

    def get_excluded(self) -> dict:
        """Return ``{filename: set(excluded_metric_keys)}`` for every row
        that has at least one metric excluded."""
        result = {}
        for fn, checkboxes in self._rows:
            excluded_keys = {k for k, chk in checkboxes.items() if chk.isChecked()}
            if excluded_keys:
                result[fn] = excluded_keys
        return result


def _normalize_excluded(excluded):
    """Accept either the new ``{filename: set(metric_keys)}`` shape or the
    legacy flat ``set(filenames)`` shape and return the new shape."""
    if not excluded:
        return {}
    if isinstance(excluded, dict):
        return excluded
    # Legacy flat set of filenames: every metric counts as excluded.
    return {fn: set(METRIC_KEYS) for fn in excluded}
