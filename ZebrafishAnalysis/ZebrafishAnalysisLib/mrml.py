"""
MRML adapter for ZebrafishAnalysis.

``results_to_rows`` is pure Python with no Slicer or VTK dependency and is
testable with standard pytest.

``get_or_create_table_node`` and ``populate_table_node`` require the Slicer
runtime.  ``populate_table_node`` imports vtk lazily inside its body so this
module is importable in plain Python test environments.
"""

import math

# Central table schema: (column_name, result_dict_key, vtk_array_type)
# vtk_array_type is "double" or "string".
# All tests, VTK column creation, and conversion use this single definition.
TABLE_SCHEMA = [
    ("Filename",            "filename",     "string"),
    ("Length_um",           "length",       "double"),
    ("CurvatureClass",      "curvature",    "string"),
    ("LengthStraightRatio", "ratio",        "double"),
    ("EyeArea_um2",         "eye_area",     "double"),
    ("EyeDiameter_um",      "eye_diameter", "double"),
    ("Error",               "error",        "string"),
]

ROLE_RESULTS_TABLE = "ResultsTable"

# String columns whose values are always preserved verbatim, even on error rows.
_PRESERVE_ON_ERROR = frozenset({"filename", "error"})


def results_to_rows(results):
    """Convert analysis result dicts to row dicts for the MRML table.

    Pure Python — no vtk or slicer imports. Testable with standard pytest.
    Input dicts are not mutated.

    Conversion rules (applied per column):
    - error row (non-empty "error" key): numeric → NaN, CurvatureClass → "",
      Filename and Error are always preserved
    - numeric field, value is None → math.nan
    - numeric field, value present → float(value)
    - string field, value is None  → ""
    - string field, value present  → str(value)

    Parameters
    ----------
    results : list[dict]
        List of result dicts from analyse_images().

    Returns
    -------
    list[dict]
        One dict per result, keyed by TABLE_SCHEMA column names, in schema order.
    """
    rows = []
    for r in results:
        has_error = bool(r.get("error"))
        row = {}
        for col_name, key, vtk_type in TABLE_SCHEMA:
            val = r.get(key)
            if vtk_type == "double":
                row[col_name] = math.nan if (has_error or val is None) else float(val)
            else:
                if key in _PRESERVE_ON_ERROR or not has_error:
                    row[col_name] = str(val) if val is not None else ""
                else:
                    row[col_name] = ""
        rows.append(row)
    return rows


def get_or_create_table_node(param_node, scene):
    """Return the existing ResultsTable node or create exactly one new node.

    Looks up the node by the stored node-reference role, not by display name.
    If no valid reference exists (missing or wrong node type), creates a new
    vtkMRMLTableNode, sets its initial display name, and registers its ID on
    the parameter node.  A wrong-type foreign node is left in the scene
    unchanged.

    Parameters
    ----------
    param_node : vtkMRMLScriptedModuleNode
        The module parameter node that owns the ResultsTable reference.
    scene : vtkMRMLScene
        The active MRML scene.

    Returns
    -------
    vtkMRMLTableNode
    """
    existing = param_node.GetNodeReference(ROLE_RESULTS_TABLE)
    if existing is not None and existing.IsA("vtkMRMLTableNode"):
        return existing

    node = scene.AddNewNodeByClass("vtkMRMLTableNode")
    node.SetName("ZebrafishAnalysis Results")
    param_node.SetNodeReferenceID(ROLE_RESULTS_TABLE, node.GetID())
    return node


def build_vtk_table(rows):
    """Build a complete vtkTable from conversion rows. No MRML side effects.

    Parameters
    ----------
    rows : list[dict]
        Output of results_to_rows().

    Returns
    -------
    vtk.vtkTable
    """
    import vtk  # lazy — not available in plain Python test environments

    n = len(rows)
    table = vtk.vtkTable()

    for col_name, _, vtk_type in TABLE_SCHEMA:
        if vtk_type == "double":
            arr = vtk.vtkDoubleArray()
        else:
            arr = vtk.vtkStringArray()
        arr.SetName(col_name)
        arr.SetNumberOfTuples(n)
        for i, row in enumerate(rows):
            arr.SetValue(i, row[col_name])
        table.AddColumn(arr)

    return table


def populate_table_node(rows, node):
    """Replace node content atomically with data from results_to_rows().

    Builds a complete vtk.vtkTable in memory first; applies it to the node
    only after all columns and values have been set successfully.  If
    construction fails the existing node content is preserved unchanged.

    Parameters
    ----------
    rows : list[dict]
        Output of results_to_rows().
    node : vtkMRMLTableNode
        Target node to update.
    """
    table = build_vtk_table(rows)
    node.SetAndObserveTable(table)
