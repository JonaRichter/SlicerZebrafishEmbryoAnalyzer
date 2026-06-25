"""Tests for the pure Python conversion layer in ZebrafishAnalysisLib.mrml.

All tests run in the normal pytest process. conftest.py adds the ZebrafishAnalysis
directory to sys.path, so ZebrafishAnalysisLib.mrml imports without slicer or vtk.
"""

import math

import pytest

from ZebrafishAnalysisLib.mrml import results_to_rows, TABLE_SCHEMA, ROLE_RESULTS_TABLE


# ---------------------------------------------------------------------------
# TABLE_SCHEMA contract
# ---------------------------------------------------------------------------

def test_schema_has_exactly_seven_columns():
    assert len(TABLE_SCHEMA) == 7


def test_schema_column_names_in_order():
    names = [col for col, _, _ in TABLE_SCHEMA]
    assert names == [
        "Filename",
        "Length_um",
        "CurvatureClass",
        "LengthStraightRatio",
        "EyeArea_um2",
        "EyeDiameter_um",
        "Error",
    ]


# ---------------------------------------------------------------------------
# results_to_rows — basic cases
# ---------------------------------------------------------------------------

def test_empty_results_returns_empty_rows():
    assert results_to_rows([]) == []


def test_single_complete_result_produces_one_row():
    r = {
        "filename": "fish.png",
        "length": 1234.5,
        "curvature": 2,
        "ratio": 1.05,
        "eye_area": 567.8,
        "eye_diameter": 12.3,
        "error": None,
    }
    rows = results_to_rows([r])
    assert len(rows) == 1
    row = rows[0]
    assert row["Filename"] == "fish.png"
    assert row["Length_um"] == pytest.approx(1234.5)
    assert row["CurvatureClass"] == "2"
    assert row["LengthStraightRatio"] == pytest.approx(1.05)
    assert row["EyeArea_um2"] == pytest.approx(567.8)
    assert row["EyeDiameter_um"] == pytest.approx(12.3)
    assert row["Error"] == ""


def test_numeric_values_remain_float():
    r = {
        "filename": "fish.png",
        "length": 100.0,
        "curvature": 1,
        "ratio": 1.1,
        "eye_area": 50.0,
        "eye_diameter": 8.0,
        "error": None,
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert isinstance(row["Length_um"], float)
    assert isinstance(row["LengthStraightRatio"], float)
    assert isinstance(row["EyeArea_um2"], float)
    assert isinstance(row["EyeDiameter_um"], float)


def test_missing_numeric_values_become_nan():
    r = {
        "filename": "fish.png",
        "length": None,
        "curvature": None,
        "ratio": None,
        "eye_area": None,
        "eye_diameter": None,
        "error": None,
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert math.isnan(row["Length_um"])
    assert math.isnan(row["LengthStraightRatio"])
    assert math.isnan(row["EyeArea_um2"])
    assert math.isnan(row["EyeDiameter_um"])
    assert row["CurvatureClass"] == ""
    assert row["Error"] == ""


def test_curvature_int_becomes_string():
    r = {
        "filename": "fish.png", "length": None, "curvature": 2,
        "ratio": None, "eye_area": None, "eye_diameter": None, "error": None,
    }
    rows = results_to_rows([r])
    assert rows[0]["CurvatureClass"] == "2"


def test_error_row_preserves_filename_and_error():
    r = {
        "filename": "bad.png",
        "length": None, "curvature": None, "ratio": None,
        "eye_area": None, "eye_diameter": None,
        "error": "Could not read image.",
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert row["Filename"] == "bad.png"
    assert row["Error"] == "Could not read image."


def test_error_row_numerics_are_nan():
    r = {
        "filename": "bad.png",
        "length": None, "curvature": None, "ratio": None,
        "eye_area": None, "eye_diameter": None,
        "error": "Could not read image.",
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert math.isnan(row["Length_um"])
    assert math.isnan(row["LengthStraightRatio"])
    assert math.isnan(row["EyeArea_um2"])
    assert math.isnan(row["EyeDiameter_um"])


def test_error_row_with_actual_values_forces_nan_and_blank_curvature():
    """Error row with non-None numeric and curvature values must still be normalized."""
    r = {
        "filename": "partial.png",
        "length": 999.9,
        "curvature": 2,
        "ratio": 1.5,
        "eye_area": 300.0,
        "eye_diameter": 20.0,
        "error": "Segmentation collapsed.",
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert row["Filename"] == "partial.png"
    assert row["Error"] == "Segmentation collapsed."
    assert math.isnan(row["Length_um"]), "length should be NaN on error row"
    assert math.isnan(row["LengthStraightRatio"]), "ratio should be NaN on error row"
    assert math.isnan(row["EyeArea_um2"]), "eye_area should be NaN on error row"
    assert math.isnan(row["EyeDiameter_um"]), "eye_diameter should be NaN on error row"
    assert row["CurvatureClass"] == "", "curvature should be blank on error row"


def test_multiple_results_preserve_order():
    results = [
        {
            "filename": "a.png", "length": 1.0, "curvature": 0, "ratio": 1.0,
            "eye_area": None, "eye_diameter": None, "error": None,
        },
        {
            "filename": "b.png", "length": 2.0, "curvature": 1, "ratio": 1.1,
            "eye_area": None, "eye_diameter": None, "error": None,
        },
        {
            "filename": "c.png", "length": 3.0, "curvature": 2, "ratio": 1.2,
            "eye_area": None, "eye_diameter": None, "error": None,
        },
    ]
    rows = results_to_rows(results)
    assert len(rows) == 3
    assert rows[0]["Filename"] == "a.png"
    assert rows[1]["Filename"] == "b.png"
    assert rows[2]["Filename"] == "c.png"
    assert rows[0]["Length_um"] == pytest.approx(1.0)
    assert rows[1]["Length_um"] == pytest.approx(2.0)
    assert rows[2]["Length_um"] == pytest.approx(3.0)


def test_input_dicts_not_mutated():
    r = {
        "filename": "fish.png", "length": 1.0, "curvature": 2,
        "ratio": 1.05, "eye_area": None, "eye_diameter": None, "error": None,
    }
    original = dict(r)
    original_keys = set(r.keys())
    results_to_rows([r])
    assert r == original, "input dict was mutated"
    assert set(r.keys()) == original_keys, "input dict keys changed"


def test_mrml_module_importable_without_slicer_or_vtk():
    """mrml.py must be importable without installing slicer or vtk."""
    from ZebrafishAnalysisLib import mrml
    assert hasattr(mrml, "results_to_rows")
    assert hasattr(mrml, "TABLE_SCHEMA")
    assert hasattr(mrml, "ROLE_RESULTS_TABLE")
    assert hasattr(mrml, "get_or_create_table_node")
    assert hasattr(mrml, "build_vtk_table")
    assert hasattr(mrml, "populate_table_node")
