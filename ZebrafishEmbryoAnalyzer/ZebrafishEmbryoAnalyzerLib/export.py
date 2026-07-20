"""
Export analysis results to Excel or CSV.

Adding a new format: add a function export_<fmt>(results, path) here
and wire it to a new button in widget.py.
"""

HEADERS = [
    ("Filename",              "filename"),
    ("Length (µm)",           "length"),
    ("Curvature class",       "curvature"),
    ("Length/straight ratio", "ratio"),
    ("Eye area (µm²)",        "eye_area"),
    ("Eye diameter (µm)",     "eye_diameter"),
    ("Edema area (µm²)",      "edema_area"),
    ("Swim bladder area (µm²)", "swim_area"),
    ("Swim bladder width (µm)", "swim_width"),
    ("Error",                 "error"),
]

# Keys in HEADERS that represent a per-image metric a user can individually
# exclude from statistics/export — everything except the identifying
# "filename" column and the "error" column (errored rows are always fully
# excluded via a different mechanism, not a per-metric checkbox).
# Derived automatically from HEADERS so a new metric column (e.g. a future
# swim-bladder area) becomes excludable without touching this module.
METRIC_KEYS = [key for _, key in HEADERS if key not in ("filename", "error")]

# Same source, reused for boxplot titles/axis labels and statistics row
# prefixes (issue #75) — one label per metric, no separate hand-maintained
# copy to drift out of sync with HEADERS.
_LABEL_BY_KEY = {key: label for label, key in HEADERS}

EXCLUDED_TEXT = "Excluded"

_EXCEL_MAX_SHEET_NAME = 31
_EXCEL_FORBIDDEN_SHEET_CHARS = str.maketrans("", "", r"/\?*[]:'")
_DEFAULT_SHEET_NAME = "Zebrafish Results"


def _cell_value(result: dict, key: str, excluded_metrics) -> object:
    if key in ("filename", "error"):
        return result.get(key)
    if excluded_metrics and key in excluded_metrics:
        return EXCLUDED_TEXT
    return result.get(key)


def sanitize_sheet_name(name: str, default: str = _DEFAULT_SHEET_NAME) -> str:
    """Strip Excel-forbidden characters and cap length at 31 characters
    (issue #75, ported from the live reference webapp's own
    ``_sanitize_sheet_name``). Falls back to ``default`` when the result
    would otherwise be empty."""
    cleaned = (name or "").strip().translate(_EXCEL_FORBIDDEN_SHEET_CHARS)
    return cleaned[:_EXCEL_MAX_SHEET_NAME] if cleaned else default


def _is_included(row_excluded: set, key: str) -> bool:
    return not (row_excluded and key in row_excluded)


def _clean_numeric_values(results: list, excluded: dict, key: str) -> list:
    """Values for ``key`` across ``results`` that are both numeric and not
    per-metric-excluded (issue #75's exclusion-aware statistics)."""
    out = []
    for r in results:
        if r.get("error"):
            continue
        row_excluded = excluded.get(r.get("filename"), set())
        if not _is_included(row_excluded, key):
            continue
        v = r.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _stats(values: list):
    """Return (median, p25, p75, mean, stdev), or ("N/A",)*5 if empty."""
    if not values:
        return ("N/A",) * 5
    import statistics
    sorted_v = sorted(values)
    n = len(sorted_v)

    def _percentile(p):
        if n == 1:
            return sorted_v[0]
        k = (n - 1) * p
        f, c = int(k), min(int(k) + 1, n - 1)
        if f == c:
            return sorted_v[f]
        return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)

    return (
        statistics.median(sorted_v),
        _percentile(0.25),
        _percentile(0.75),
        statistics.mean(sorted_v),
        statistics.pstdev(sorted_v) if n > 1 else 0.0,
    )


def _make_boxplots_image(results: list, excluded: dict) -> bytes:
    """PNG bytes of one boxplot subplot per metric with at least one
    numeric (non-excluded) value — issue #75, ported from the live
    reference webapp's ``_make_boxplots_image``, but generic over
    ``METRIC_KEYS`` instead of one hand-written subplot per metric, so a
    new metric column needs no change here."""
    import matplotlib
    matplotlib.use("Agg")  # headless — no GUI backend needed/available
    import matplotlib.pyplot as plt
    import io

    per_metric = {k: _clean_numeric_values(results, excluded, k) for k in METRIC_KEYS}
    active = [(k, v) for k, v in per_metric.items() if v]
    num_plots = max(1, len(active))

    fig = plt.figure(figsize=(5 * num_plots, 5))
    for i, (key, values) in enumerate(active, start=1):
        plt.subplot(1, num_plots, i)
        plt.boxplot(values, orientation="vertical", patch_artist=True)
        label = _LABEL_BY_KEY.get(key, key)
        plt.title(label)
        plt.ylabel(label)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _class_distribution(results: list, excluded: dict) -> list:
    """[(label, count), ...] for curvature classes 1-4 plus "Not Classified"
    (class 5 / confidence-gated uncertain), excluding per-metric-excluded
    and errored rows — issue #75."""
    labels = ["Class 1", "Class 2", "Class 3", "Class 4", "Not Classified"]
    counts = [0, 0, 0, 0, 0]
    for r in results:
        if r.get("error"):
            continue
        row_excluded = excluded.get(r.get("filename"), set())
        if not _is_included(row_excluded, "curvature"):
            continue
        c = r.get("curvature")
        if c is None:
            continue
        idx = 4 if c == 5 else int(c) - 1
        if 0 <= idx < 5:
            counts[idx] += 1
    return list(zip(labels, counts))


def export_excel(results: list, path: str, excluded: dict = None, sheet_name: str = None) -> None:
    """Write ``results`` to an .xlsx file.

    ``excluded`` is an optional ``{filename: set(excluded_metric_keys)}``
    mapping (see ``widget.py``'s per-metric exclude model). A row is never
    dropped for being excluded — only the specific excluded cell renders as
    "Excluded" instead of its value, so the row stays visible and every
    other metric on it is still exported normally.

    Issue #75 additions beyond the plain per-row dump: an embedded boxplot
    image per active metric, exclusion-aware summary statistics
    (median/25th/75th percentile/mean/stdev), a curvature class-distribution
    breakdown, and a sanitized, user-configurable ``sheet_name``.
    """
    import openpyxl
    from openpyxl.drawing.image import Image as ExcelImage
    import io

    excluded = excluded or {}
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sanitize_sheet_name(sheet_name)
    ws.append([h for h, _ in HEADERS])
    for r in results:
        row_excluded = excluded.get(r.get("filename"), set())
        ws.append([_cell_value(r, k, row_excluded) for _, k in HEADERS])

    ws.append([])

    excluded_counts = {
        key: sum(
            1 for r in results
            if not _is_included(excluded.get(r.get("filename"), set()), key)
        )
        for key in METRIC_KEYS
    }
    excl_note_parts = [
        f"{_LABEL_BY_KEY.get(k, k)}: {v}" for k, v in excluded_counts.items() if v > 0
    ]
    if excl_note_parts:
        ws.append(["Excluded from statistics — " + ", ".join(excl_note_parts)])

    ws.append(["Statistics (excluded values not counted)"])
    for key in METRIC_KEYS:
        values = _clean_numeric_values(results, excluded, key)
        if not values:
            continue
        label = _LABEL_BY_KEY.get(key, key)
        med, p25, p75, mean, std = _stats(values)
        ws.append([f"Median {label}", med])
        ws.append([f"25th Percentile {label}", p25])
        ws.append([f"75th Percentile {label}", p75])
        ws.append([f"Mean {label}", mean])
        ws.append([f"Standard Deviation {label}", std])

    ws.append([])
    ws.append(["Class Distribution"])
    for label, count in _class_distribution(results, excluded):
        ws.append([label, count])

    try:
        boxplot_png = _make_boxplots_image(results, excluded)
        ws.add_image(ExcelImage(io.BytesIO(boxplot_png)), "E2")
    except ImportError:
        # matplotlib not installed — degrade gracefully, still write the
        # rest of the workbook (matches export_csv's dependency-free path).
        pass

    wb.save(path)


def export_csv(results: list, path: str, excluded: dict = None) -> None:
    import csv
    excluded = excluded or {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([h for h, _ in HEADERS])
        for r in results:
            row_excluded = excluded.get(r.get("filename"), set())
            w.writerow([_cell_value(r, k, row_excluded) for _, k in HEADERS])
