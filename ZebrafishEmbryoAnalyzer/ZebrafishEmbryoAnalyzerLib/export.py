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
    ("Error",                 "error"),
]

# Keys in HEADERS that represent a per-image metric a user can individually
# exclude from statistics/export — everything except the identifying
# "filename" column and the "error" column (errored rows are always fully
# excluded via a different mechanism, not a per-metric checkbox).
# Derived automatically from HEADERS so a new metric column (e.g. a future
# swim-bladder area) becomes excludable without touching this module.
METRIC_KEYS = [key for _, key in HEADERS if key not in ("filename", "error")]

EXCLUDED_TEXT = "Excluded"


def _cell_value(result: dict, key: str, excluded_metrics) -> object:
    if key in ("filename", "error"):
        return result.get(key)
    if excluded_metrics and key in excluded_metrics:
        return EXCLUDED_TEXT
    return result.get(key)


def export_excel(results: list, path: str, excluded: dict = None) -> None:
    """Write ``results`` to an .xlsx file.

    ``excluded`` is an optional ``{filename: set(excluded_metric_keys)}``
    mapping (see ``widget.py``'s per-metric exclude model). A row is never
    dropped for being excluded — only the specific excluded cell renders as
    "Excluded" instead of its value, so the row stays visible and every
    other metric on it is still exported normally.
    """
    import openpyxl
    excluded = excluded or {}
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Zebrafish Results"
    ws.append([h for h, _ in HEADERS])
    for r in results:
        row_excluded = excluded.get(r.get("filename"), set())
        ws.append([_cell_value(r, k, row_excluded) for _, k in HEADERS])
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
