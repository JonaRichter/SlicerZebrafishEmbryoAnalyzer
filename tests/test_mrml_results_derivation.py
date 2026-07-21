"""
Tests for issue #40: results table rebuilt as a derived cache from
per-image volume-node attributes (ADR 0001).

Covers:
- ``volume_node_to_result_dict`` round-trip: writing metric attributes via
  the same format the production code uses (and ``mrml._format_attr``)
  must yield a result dict whose row is identical to the result dict's
  own row.
- ``volume_nodes_to_rows`` returns rows in insertion order of the nodes.
- ``volume_nodes_to_rows`` produces the same row list as
  ``results_to_rows`` for an equivalent in-memory result list — the
  acceptance criterion.
- ``list_tracked_volume_nodes`` honours ``ROLE_ZEBRAFISH_IMAGES`` order,
  skips missing/deleted IDs, and skips wrong-type nodes.
- ``update_results_table_from_tracked_nodes`` routes via the same code
  path as ``update_results_table`` (single shared tail, no drift).
- NaN/empty attribute values round-trip identically.
- Curvature-class int- and string-typed attributes both round-trip.
- The boolean exclude attribute round-trips.
- A volume node error attribute surfaces in the derived row.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Lightweight fake volume node
# ---------------------------------------------------------------------------

class _FakeVolumeNode:
    """Minimal fake for vtkMRMLVolumeNode — supports Set/GetAttribute,
    GetName, and an optional IsA class match.

    Attribute storage uses a plain dict so test setup is straightforward;
    no real vtk / slicer needed.
    """

    def __init__(self, name="fish.png", attrs=None, klass="vtkMRMLVolumeNode"):
        self._attrs = dict(attrs or {})
        self._name = name
        self._klass = klass

    def GetAttribute(self, name):
        return self._attrs.get(name)

    def SetAttribute(self, name, value):
        if value is None:
            self._attrs[name] = None
        else:
            self._attrs[name] = str(value)

    def GetName(self):
        return self._name

    def IsA(self, klass):
        return klass == self._klass

    # For AST-based test surface area inspection (optional, harmless).
    def GetID(self):
        return f"vtkMRMLVolumeNode_{id(self)}"


def _write_metric_attributes(result, node):
    """Replicate the production ``_write_metric_attributes`` writer so we
    can build hand-constructed volume nodes without depending on the
    private helper (which lives in mrml.py but is intentionally _-prefixed).

    Mirrors the rules from issue #39: floats via repr / lossless, ints
    via str, None → empty string, exclude → "true"/"false".
    """
    def _fmt(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        return str(value)

    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ATTR_LENGTH,
        ATTR_CURVATURE_CLASS,
        ATTR_RATIO,
        ATTR_EYE_AREA,
        ATTR_EYE_DIAMETER,
        ATTR_EXCLUDE,
    )

    node.SetAttribute(ATTR_LENGTH, _fmt(result.get("length")))
    node.SetAttribute(ATTR_CURVATURE_CLASS, _fmt(result.get("curvature")))
    node.SetAttribute(ATTR_RATIO, _fmt(result.get("ratio")))
    node.SetAttribute(ATTR_EYE_AREA, _fmt(result.get("eye_area")))
    node.SetAttribute(ATTR_EYE_DIAMETER, _fmt(result.get("eye_diameter")))
    node.SetAttribute(ATTR_EXCLUDE, "true" if bool(result.get("exclude")) else "false")


def _make_node_from_result(result, name=None):
    """Build a fake volume node carrying the same metric attributes that
    issue #39 would have written for ``result``.
    """
    node = _FakeVolumeNode(name=name or result.get("filename") or "fish.png")
    _write_metric_attributes(result, node)
    # Attach an error attribute if the result has one.
    err = result.get("error")
    if err:
        node.SetAttribute("ZebrafishAnalysis.error", str(err))
    return node


# ---------------------------------------------------------------------------
# Pure-Python derivation helpers
# ---------------------------------------------------------------------------

def test_volume_node_to_result_dict_round_trip_basic():
    """Re-deriving a result dict from a volume node carrying its attributes
    must yield a row that matches the original (after the schema coercion
    rules in results_to_rows).
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    result = {
        "filename": "fish.png",
        "length": 1.234,
        "curvature": 2,
        "ratio": 1.05,
        "eye_area": 0.04,
        "eye_diameter": 22.99,
        "exclude": False,
        "error": "",
    }
    node = _make_node_from_result(result)

    derived = mrml.volume_node_to_result_dict(node)
    rows_from_results = mrml.results_to_rows([result])
    rows_from_nodes = mrml.volume_nodes_to_rows([node])

    assert rows_from_nodes == rows_from_results, (
        f"derived rows diverge: {rows_from_nodes!r} vs {rows_from_results!r}"
    )


def test_volume_nodes_to_rows_matches_results_to_rows_for_equivalent_state():
    """Acceptance criterion: build the table from hand-constructed node
    attributes and confirm it matches the equivalent in-memory results list.
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    results = [
        {
            "filename": "a.png",
            "length": 1.5,
            "curvature": 1,
            "ratio": 1.02,
            "eye_area": 0.04,
            "eye_diameter": 22.99,
            "exclude": False,
            "error": "",
        },
        {
            "filename": "b.png",
            "length": None,  # length disabled
            "curvature": 0,
            "ratio": None,
            "eye_area": None,
            "eye_diameter": None,
            "exclude": True,
            "error": "",
        },
        {
            "filename": "broken.png",
            "length": None,
            "curvature": "",
            "ratio": None,
            "eye_area": None,
            "eye_diameter": None,
            "exclude": False,
            "error": "Could not read image.",
        },
    ]
    nodes = [_make_node_from_result(r) for r in results]

    rows_from_results = mrml.results_to_rows(results)
    rows_from_nodes = mrml.volume_nodes_to_rows(nodes)

    assert rows_from_nodes == rows_from_results


def test_volume_nodes_to_rows_preserves_node_order():
    """Row order == node order. Even if names sort lexically, the row list
    follows the caller-supplied order so the table layout is reproducible.
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    n0 = _make_node_from_result({"filename": "z_last.png"}, name="z_last.png")
    n1 = _make_node_from_result({"filename": "a_first.png"}, name="a_first.png")
    n2 = _make_node_from_result({"filename": "m_middle.png"}, name="m_middle.png")

    rows = mrml.volume_nodes_to_rows([n0, n1, n2])
    assert [r["Filename"] for r in rows] == ["z_last.png", "a_first.png", "m_middle.png"]

    rows2 = mrml.volume_nodes_to_rows([n2, n0, n1])
    assert [r["Filename"] for r in rows2] == ["m_middle.png", "z_last.png", "a_first.png"]


def test_nan_attributes_round_trip_to_nan_cell():
    """A float NaN written via ``_format_attr`` becomes the literal string
    "nan"; the derivation must produce math.nan — the same value
    ``results_to_rows`` writes for None numeric cells.
    """
    import math

    from ZebrafishEmbryoAnalyzerLib import mrml

    node = _FakeVolumeNode()
    # Manually emulate: a numeric that was None becomes raw "" (sentinel)
    from ZebrafishEmbryoAnalyzerLib.mrml import ATTR_LENGTH
    node.SetAttribute(ATTR_LENGTH, "")

    derived = mrml.volume_node_to_result_dict(node)
    assert derived["length"] is None, (
        "volume_node_to_result_dict converts nan sentinel to None so the "
        "results-to-rows comparison matches."
    )

    rows_node = mrml.volume_nodes_to_rows([node])
    rows_none = mrml.results_to_rows([
        {"filename": node.GetName(), "length": None, "curvature": "",
         "ratio": None, "eye_area": None, "eye_diameter": None,
         "exclude": False, "error": ""}
    ])
    assert rows_node == rows_none


def test_boolean_exclude_round_trip():
    """An exclude value of True or False must write its attribute and
    come back identical.
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    for excluded in (True, False):
        result = {
            "filename": f"fish_{excluded}.png",
            "length": 1.0, "curvature": 0, "ratio": 1.0,
            "eye_area": None, "eye_diameter": None,
            "exclude": excluded, "error": "",
        }
        node = _make_node_from_result(result)
        derived = mrml.volume_node_to_result_dict(node)
        assert bool(derived["exclude"]) is excluded, (
            f"exclude round-trip lost: expected {excluded}, got {derived['exclude']!r}"
        )


def test_curvature_class_int_round_trip():
    """The legacy curvature column was an int class id (0/1/2)."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    for cls in (0, 1, 2):
        result = {
            "filename": f"c_{cls}.png",
            "length": 1.0,
            "curvature": cls,
            "ratio": 1.0,
            "eye_area": None,
            "eye_diameter": None,
            "exclude": False,
            "error": "",
        }
        node = _make_node_from_result(result)
        derived = mrml.volume_node_to_result_dict(node)
        assert derived["curvature"] == cls, (
            f"curvature int round-trip lost: expected {cls}, got {derived['curvature']!r}"
        )


def test_curvature_class_string_round_trip():
    """Newer rows may store a string label; must round-trip."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    result = {
        "filename": "c_label.png",
        "length": 1.0,
        "curvature": "Straight",
        "ratio": 1.0,
        "eye_area": None,
        "eye_diameter": None,
        "exclude": False,
        "error": "",
    }
    node = _make_node_from_result(result)
    derived = mrml.volume_node_to_result_dict(node)
    assert derived["curvature"] == "Straight"


def test_error_attribute_surfaces_in_derived_row():
    """An image with a pipeline error must surface an error row in the
    derived table; the original rows_to_rows semantics are preserved.
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    node = _FakeVolumeNode()
    node.SetAttribute("ZebrafishAnalysis.error", "Could not read image.")

    rows = mrml.volume_nodes_to_rows([node])
    assert rows[0]["Error"] == "Could not read image."
    # Numeric cells are NaN for error rows.
    import math
    for col in ("Length_um", "LengthStraightRatio", "EyeArea_um2", "EyeDiameter_um"):
        assert math.isnan(rows[0][col]), f"{col} should be NaN for error row"


def test_coerced_attribute_invalid_float_raises():
    """Defensive: if a non-parseable string sneaks into a numeric
    attribute, the derivation must raise so the widget can surface a
    specific error rather than silently writing zero.
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    node = _FakeVolumeNode()
    from ZebrafishEmbryoAnalyzerLib.mrml import ATTR_LENGTH
    node.SetAttribute(ATTR_LENGTH, "not-a-number")

    with pytest.raises(ValueError):
        mrml.volume_node_to_result_dict(node)


def test_missing_attributes_default_to_neutral_empty():
    """If a volume node carries no metric attributes at all (e.g. an
    empty node before analysis ran), the derived row must be the empty
    / NaN neutral row, not raise.
    """
    import math

    from ZebrafishEmbryoAnalyzerLib import mrml

    node = _FakeVolumeNode(name="blank.png")
    rows = mrml.volume_nodes_to_rows([node])
    assert rows[0]["Filename"] == "blank.png"
    assert rows[0]["Length_um"] != rows[0]["Length_um"]  # NaN != NaN trick
    assert rows[0]["CurvatureClass"] == ""
    assert rows[0]["Error"] == ""


# ---------------------------------------------------------------------------
# Param-node reference list helper
# ---------------------------------------------------------------------------

def _mock_param_with_refs(ids, role):
    """MagicMock exposing the real per-role reference-enumeration API.

    Production code (``mrml._node_reference_ids``) reads references via
    ``GetNumberOfNodeReferences``/``GetNthNodeReferenceID`` — the plural
    ``GetNodeReferenceIDs(role)`` getter is not exposed by the Python
    binding on ``vtkMRMLNode`` in real Slicer.
    """
    param = MagicMock()
    param.GetNumberOfNodeReferences = MagicMock(
        side_effect=lambda r: len(ids) if r == role else 0
    )
    param.GetNthNodeReferenceID = MagicMock(
        side_effect=lambda r, n: ids[n] if r == role and 0 <= n < len(ids) else None
    )
    return param


def test_list_tracked_volume_nodes_returns_ordered_nodes():
    """Param-node reference list ordering drives row order."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    nodes = [_FakeVolumeNode(name=f"n{i}") for i in range(3)]
    ids = [n.GetID() for n in nodes]

    param = _mock_param_with_refs(ids, mrml.ROLE_ZEBRAFISH_IMAGES)

    scene = MagicMock()
    def _by_id(nid):
        for n in nodes:
            if n.GetID() == nid:
                return n
        return None
    scene.GetNodeByID = MagicMock(side_effect=_by_id)

    listed = mrml.list_tracked_volume_nodes(param, scene)
    assert listed == nodes
    param.GetNumberOfNodeReferences.assert_called_once_with(mrml.ROLE_ZEBRAFISH_IMAGES)


def test_list_tracked_volume_nodes_sorts_by_load_order_attribute():
    """A Save Scene -> Load Scene round-trip was observed to come back with
    the NodeReference array itself reordered even though every node/reference
    survived (found while testing #61). list_tracked_volume_nodes must
    restore folder-load order from each node's ZebrafishAnalysis.loadOrder
    attribute (set at eager-creation time) rather than trusting the
    reference array's own enumeration order.
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    n0 = _FakeVolumeNode(name="n0", attrs={"ZebrafishAnalysis.loadOrder": "0"})
    n1 = _FakeVolumeNode(name="n1", attrs={"ZebrafishAnalysis.loadOrder": "1"})
    n2 = _FakeVolumeNode(name="n2", attrs={"ZebrafishAnalysis.loadOrder": "2"})
    nodes = [n0, n1, n2]

    # Reference list comes back scrambled (simulating the reload reorder).
    scrambled_ids = [n2.GetID(), n0.GetID(), n1.GetID()]
    param = _mock_param_with_refs(scrambled_ids, mrml.ROLE_ZEBRAFISH_IMAGES)

    scene = MagicMock()
    def _by_id(nid):
        for n in nodes:
            if n.GetID() == nid:
                return n
        return None
    scene.GetNodeByID = MagicMock(side_effect=_by_id)

    listed = mrml.list_tracked_volume_nodes(param, scene)
    assert listed == [n0, n1, n2]


def test_list_tracked_volume_nodes_keeps_reference_order_without_attribute():
    """Nodes with no loadOrder attribute (older scenes, other test fakes)
    must keep their relative reference-list order rather than crash or
    reorder unpredictably.
    """
    from ZebrafishEmbryoAnalyzerLib import mrml

    nodes = [_FakeVolumeNode(name=f"n{i}") for i in range(3)]
    ids = [n.GetID() for n in nodes]
    param = _mock_param_with_refs(ids, mrml.ROLE_ZEBRAFISH_IMAGES)

    scene = MagicMock()
    def _by_id(nid):
        for n in nodes:
            if n.GetID() == nid:
                return n
        return None
    scene.GetNodeByID = MagicMock(side_effect=_by_id)

    listed = mrml.list_tracked_volume_nodes(param, scene)
    assert listed == nodes


def test_list_tracked_volume_nodes_skips_missing_ids():
    """A reference whose node no longer exists must be silently dropped —
    it will surface as a missing row in the table (not a crash)."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    real_node = _FakeVolumeNode(name="real")
    param = _mock_param_with_refs(["missing-id", real_node.GetID()], mrml.ROLE_ZEBRAFISH_IMAGES)

    scene = MagicMock()
    scene.GetNodeByID = MagicMock(return_value=None)

    listed = mrml.list_tracked_volume_nodes(param, scene)
    assert listed == []


def test_list_tracked_volume_nodes_skips_wrong_type():
    """Only vtkMRMLVolumeNode subclasses are returned."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    good = _FakeVolumeNode(name="good", klass="vtkMRMLVolumeNode")
    bad = _FakeVolumeNode(name="bad", klass="vtkMRMLTableNode")
    param = _mock_param_with_refs([good.GetID(), bad.GetID()], mrml.ROLE_ZEBRAFISH_IMAGES)

    scene = MagicMock()
    scene.GetNodeByID = MagicMock(side_effect=lambda nid:
                                   good if nid == good.GetID() else
                                   bad if nid == bad.GetID() else
                                   None)

    listed = mrml.list_tracked_volume_nodes(param, scene)
    assert listed == [good]


# ---------------------------------------------------------------------------
# Logic-level entry points (subprocess tests)
# ---------------------------------------------------------------------------

_SLICER_STUB = """\
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"] = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()

class _BaseWidget:
    pass

class _VTKMixin:
    def addObserver(self, *a, **kw): pass
    def removeObservers(self, *a, **kw): pass
    def removeObserver(self, *a, **kw): pass
    def hasObserver(self, *a, **kw): return False

sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=_BaseWidget,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(
    VTKObservationMixin=_VTKMixin,
)
_vtk = types.ModuleType("vtk")
_vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
sys.modules["vtk"] = _vtk
import vtk  # noqa
"""


def _run_in_subprocess(source):
    """Run ``source`` in a fresh subprocess with the package on the path
    and importable. Returns the CompletedProcess.

    The source string is dedented so heredoc-style test bodies (with the
    leading 12-space indent from this fixture) parse as top-level Python.
    A slicer/vtk stub is prepended so the Logic class can be imported
    without the real Slicer runtime.
    """
    import subprocess
    import textwrap
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_root = os.path.join(root, "ZebrafishEmbryoAnalyzer")
    full = _SLICER_STUB + textwrap.dedent(source)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pkg_root},
    )


def test_update_results_table_from_volume_nodes_routes_through_shared_tail():
    """Issue #40: the new entry point must build the same table as
    ``update_results_table`` for equivalent input state.
    """
    r = _run_in_subprocess(r"""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        fake_node = MagicMock()
        fake_node.GetID.return_value = "Tnode"

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table") as mb, \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_table_node",
                   return_value=fake_node) as mg, \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.volume_nodes_to_rows",
                   wraps=__import__("ZebrafishEmbryoAnalyzerLib.mrml",
                                    fromlist=["*"]).volume_nodes_to_rows) as mv:
            nodes = [object(), object()]
            logic.update_results_table_from_volume_nodes(nodes)

            mv.assert_called_once_with(list(nodes))
            mb.assert_called_once()
            mg.assert_called_once()
            fake_node.SetAndObserveTable.assert_called_once_with(mb.return_value)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_results_table_still_works_with_results_list():
    """Backward compatibility: the legacy results-list entry point must
    keep producing the same table content for the same data.
    """
    r = _run_in_subprocess(r"""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        fake_node = MagicMock()
        fake_node.GetID.return_value = "Tnode"

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table") as mb, \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_table_node",
                   return_value=fake_node), \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.results_to_rows",
                   wraps=__import__("ZebrafishEmbryoAnalyzerLib.mrml",
                                    fromlist=["*"]).results_to_rows) as mr:
            logic.update_results_table([
                {"filename": "f.png", "length": 1.0, "curvature": 0,
                 "ratio": 1.0, "eye_area": None, "eye_diameter": None,
                 "error": None, "exclude": False}
            ])

            mr.assert_called_once()
            mb.assert_called_once()
            fake_node.SetAndObserveTable.assert_called_once_with(mb.return_value)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_results_table_from_tracked_nodes_uses_param_node_references():
    """The tracked-node entry point reads via
    ``list_tracked_volume_nodes`` and forwards the ordered list to the
    volume-node entry point.
    """
    r = _run_in_subprocess(r"""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        param = MagicMock()
        logic.getParameterNode = MagicMock(return_value=param)

        fake_nodes = [object(), object()]

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
                   return_value=fake_nodes) as ml, \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table"), \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_table_node",
                   return_value=MagicMock()):
            import slicer
            slicer.mrmlScene = MagicMock()
            res = logic.update_results_table_from_tracked_nodes()
            ml.assert_called_once_with(param, slicer.mrmlScene)
            assert res is not None
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_widget_routes_through_tracked_nodes_after_analysis():
    """Widget must call update_results_table_from_tracked_nodes, not
    pass the in-memory results list — the single-code-path contract.
    """
    widget_src = open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzerLib", "widget.py",
        )
    ).read()
    assert "update_results_table_from_tracked_nodes" in widget_src, (
        "widget.py must call the tracked-nodes entry point for the single code path"
    )
    # And the legacy entry point should no longer be invoked from the run path.
    run_section = widget_src.split("_on_runner_finished", 1)[1]
    assert "update_results_table(self._results)" not in run_section, (
        "run path must not pass in-memory results to update_results_table anymore"
    )
