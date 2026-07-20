import os


RESULTS = [
    {
        "filename": "fish1.png", "length": 1200.0, "curvature": 2,
        "ratio": 1.05, "eye_area": None, "eye_diameter": None, "error": None,
    },
    {
        "filename": "fish2.png", "length": None, "curvature": None,
        "ratio": None, "eye_area": None, "eye_diameter": None, "error": "Segmentation failed",
    },
]


def test_export_excel_creates_file(tmp_path):
    from ZebrafishEmbryoAnalyzerLib.export import export_excel
    out = str(tmp_path / "results.xlsx")
    export_excel(RESULTS, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_export_csv_creates_file(tmp_path):
    from ZebrafishEmbryoAnalyzerLib.export import export_csv
    out = str(tmp_path / "results.csv")
    export_csv(RESULTS, out)
    assert os.path.exists(out)
    lines = open(out).readlines()
    assert len(lines) == 3  # header + 2 rows


def test_export_excel_includes_error_column(tmp_path):
    import openpyxl
    from ZebrafishEmbryoAnalyzerLib.export import export_excel
    out = str(tmp_path / "results.xlsx")
    export_excel(RESULTS, out)
    wb = openpyxl.load_workbook(out)
    headers = [cell.value for cell in wb.active[1]]
    assert "Error" in headers


def test_export_filtered_skips_excluded(tmp_path):
    from ZebrafishEmbryoAnalyzerLib.export import export_excel
    import openpyxl
    out = str(tmp_path / "filtered.xlsx")
    filtered = [r for r in RESULTS if r["error"] is None]
    export_excel(filtered, out)
    wb = openpyxl.load_workbook(out)
    assert wb.active.max_row == 2  # header + 1 result row


def test_metric_keys_derived_from_headers_excludes_filename_and_error():
    from ZebrafishEmbryoAnalyzerLib.export import METRIC_KEYS, HEADERS
    keys_from_headers = {k for _, k in HEADERS}
    assert set(METRIC_KEYS) == keys_from_headers - {"filename", "error"}
    assert "filename" not in METRIC_KEYS
    assert "error" not in METRIC_KEYS


def test_export_excel_renders_excluded_cell_not_dropped_row(tmp_path):
    """Issue #74/#75: a per-metric-excluded cell must render as literal
    "Excluded" text, and the row must still be present — not dropped."""
    from ZebrafishEmbryoAnalyzerLib.export import export_excel, HEADERS
    import openpyxl
    out = str(tmp_path / "partial_exclude.xlsx")
    excluded = {"fish1.png": {"curvature"}}
    export_excel(RESULTS, out, excluded=excluded)
    wb = openpyxl.load_workbook(out)
    ws = wb.active
    assert ws.max_row == 3  # header + both result rows, none dropped
    header_row = [c.value for c in ws[1]]
    curvature_col = header_row.index("Curvature class") + 1
    length_col = header_row.index("Length (µm)") + 1
    assert ws.cell(row=2, column=curvature_col).value == "Excluded"
    assert ws.cell(row=2, column=length_col).value == 1200.0  # untouched


def test_export_csv_renders_excluded_cell_not_dropped_row(tmp_path):
    from ZebrafishEmbryoAnalyzerLib.export import export_csv
    out = str(tmp_path / "partial_exclude.csv")
    excluded = {"fish1.png": {"curvature"}}
    export_csv(RESULTS, out, excluded=excluded)
    lines = open(out).readlines()
    assert len(lines) == 3  # header + both rows
    assert "Excluded" in lines[1]
