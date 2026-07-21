"""
Tests for MRML node creation, reuse, and update_results_table orchestration.

Static checks verify source-level contracts without subprocess overhead.
Behavioral node tests use small fake objects directly — conftest.py adds the
ZebrafishEmbryoAnalyzer directory to sys.path so mrml.py imports cleanly.
Subprocess tests cover the full update_results_table flow, which requires
the Slicer module stub so ZebrafishEmbryoAnalyzer.py can be imported.
"""

import math
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ZebrafishEmbryoAnalyzer",
)
_MAIN_PY   = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzer.py")
_WIDGET_PY = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "widget.py")
_LOGIC_PY  = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "logic.py")
_MRML_PY   = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "mrml.py")
_CMAKE     = os.path.join(
    os.path.dirname(_MODULE_DIR), "ZebrafishEmbryoAnalyzer", "CMakeLists.txt"
)


# ---------------------------------------------------------------------------
# Subprocess helper (used for update_results_table integration tests)
# ---------------------------------------------------------------------------

_SLICER_STUB = """\
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"]  = MagicMock()
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


def _run(code: str) -> subprocess.CompletedProcess:
    full = _SLICER_STUB + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


# ---------------------------------------------------------------------------
# Fake MRML objects for direct (non-subprocess) node tests
# ---------------------------------------------------------------------------

class _FakeTableNode:
    _counter = 0

    def __init__(self):
        _FakeTableNode._counter += 1
        self._id = f"vtkMRMLTableNode{_FakeTableNode._counter}"
        self._name = ""
        self._table = None

    def GetID(self):
        return self._id

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLTableNode"

    def SetAndObserveTable(self, vtk_table):
        self._table = vtk_table


class _FakeNonTableNode:
    """Simulates a foreign node (e.g. volume node) stored under the ResultsTable role."""

    def __init__(self):
        self._id = "vtkMRMLVolumeNode1"

    def GetID(self):
        return self._id

    def IsA(self, class_name):
        return class_name == "vtkMRMLVolumeNode"


class _FakeScene:
    def __init__(self):
        self._nodes = []
        self._add_count = 0

    def AddNewNodeByClass(self, class_name):
        self._add_count += 1
        node = _FakeTableNode()
        self._nodes.append(node)
        return node


class _FakeParamNode:
    def __init__(self, existing_node=None):
        self._existing = existing_node
        self._stored_role = None
        self._stored_id = None
        self._set_ref_calls = 0

    def GetNodeReference(self, role):
        return self._existing

    def SetNodeReferenceID(self, role, node_id):
        self._stored_role = role
        self._stored_id = node_id
        self._set_ref_calls += 1


# Fake vtk module for populate_table_node / build_vtk_table tests
class _FakeVTKArray:
    def __init__(self):
        self._name = ""
        self._data = {}
        self._n = 0

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def SetNumberOfTuples(self, n):
        self._n = n

    def SetValue(self, i, val):
        self._data[i] = val

    def GetValue(self, i):
        return self._data.get(i)

    def GetNumberOfTuples(self):
        return self._n


class _FakeVTKTable:
    def __init__(self):
        self._cols = []

    def AddColumn(self, col):
        self._cols.append(col)

    def GetNumberOfColumns(self):
        return len(self._cols)

    def GetColumn(self, i):
        return self._cols[i]


def _make_fake_vtk():
    fake = types.ModuleType("vtk")
    fake.vtkTable = _FakeVTKTable
    fake.vtkDoubleArray = _FakeVTKArray
    fake.vtkStringArray = _FakeVTKArray
    return fake


# ---------------------------------------------------------------------------
# Fake MRML objects for image node tests
# ---------------------------------------------------------------------------
#
# Issue #56: the singleton ``CurrentImage`` / ``CurrentSegmentation`` helpers
# were retired in favour of the per-image volume + segmentation nodes
# (issues #38 / #39). The behavioural tests that needed direct
# get_or_create_* coverage were replaced by behaviour tests against the new
# ``set_slice_viewer_background`` / ``set_segmentation_visibility`` /
# ``find_tracked_volume_node_by_filename`` helpers, which use ``MagicMock``
# instead of bespoke fakes.

# ---------------------------------------------------------------------------
# Source-level static checks
# ---------------------------------------------------------------------------

def test_logic_py_does_not_import_mrml():
    """ZebrafishEmbryoAnalyzerLib.logic must not import the mrml adapter."""
    src = open(_LOGIC_PY).read()
    import re
    assert not re.search(r'^(?:import|from)\s+.*mrml', src, re.MULTILINE), \
        "logic.py must not import mrml"


def test_mrml_in_cmake():
    """ZebrafishEmbryoAnalyzer/CMakeLists.txt must list ZebrafishEmbryoAnalyzerLib/mrml.py."""
    content = open(_CMAKE).read()
    assert "ZebrafishEmbryoAnalyzerLib/mrml.py" in content, (
        "CMakeLists.txt does not include ZebrafishEmbryoAnalyzerLib/mrml.py"
    )


def test_mrml_in_reload_eviction_list():
    """ZebrafishEmbryoAnalyzer.py _LIB_MODULES must include ZebrafishEmbryoAnalyzerLib.mrml."""
    src = open(_MAIN_PY).read()
    assert '"ZebrafishEmbryoAnalyzerLib.mrml"' in src, (
        "_LIB_MODULES must include 'ZebrafishEmbryoAnalyzerLib.mrml'"
    )


def test_core_submodules_in_reload_eviction_list():
    """All five ZebrafishEmbryoAnalyzerCore submodules must be evicted on reload.

    Regression guard for issue #49: without explicit eviction, plain Python
    import caching keeps stale Core code in sys.modules across Developer Tools
    reloads, so editing seg.py / seg_helper.py / length.py / manual.py /
    scalebar.py requires a full Slicer restart.
    """
    src = open(_MAIN_PY).read()
    expected = (
        '"ZebrafishEmbryoAnalyzerCore.seg"',
        '"ZebrafishEmbryoAnalyzerCore.seg_helper"',
        '"ZebrafishEmbryoAnalyzerCore.length"',
        '"ZebrafishEmbryoAnalyzerCore.manual"',
        '"ZebrafishEmbryoAnalyzerCore.scalebar"',
    )
    for module in expected:
        assert module in src, (
            f"{module} must be in the reload eviction list in "
            f"ZebrafishEmbryoAnalyzer.py (issue #49)"
        )


def test_no_get_first_node_by_name_in_mrml():
    """mrml.py must not use GetFirstNodeByName for ownership lookups."""
    src = open(_MRML_PY).read()
    assert "GetFirstNodeByName" not in src, (
        "mrml.py must not use GetFirstNodeByName — use node references instead"
    )


def test_widget_has_no_persistent_table_node_pointer():
    """widget.py must not store a persistent _table_node attribute."""
    src = open(_WIDGET_PY).read()
    assert "self._table_node" not in src, (
        "widget.py must not keep a persistent _table_node pointer — "
        "ownership is via parameter node reference"
    )


def test_widget_calls_update_results_table_not_mrml_directly():
    """widget.py must call update_results_table via logic, not import mrml directly."""
    src = open(_WIDGET_PY).read()
    assert "update_results_table" in src, (
        "widget.py must call self._logic.update_results_table()"
    )
    assert "from ZebrafishEmbryoAnalyzerLib.mrml" not in src, (
        "widget.py must not import ZebrafishEmbryoAnalyzerLib.mrml directly"
    )
    assert "from ZebrafishEmbryoAnalyzerLib import mrml" not in src, (
        "widget.py must not import ZebrafishEmbryoAnalyzerLib.mrml directly"
    )


def test_mrml_module_has_no_global_slicer_import():
    """mrml.py must not have a module-level 'import slicer'."""
    src = open(_MRML_PY).read()
    lines = src.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(def |class )", stripped):
            in_function = True
        if not in_function and re.match(r"^import slicer\b", stripped):
            pytest.fail("mrml.py has a module-level 'import slicer'")


def test_mrml_module_has_no_global_vtk_import():
    """mrml.py must not have a module-level 'import vtk'."""
    src = open(_MRML_PY).read()
    lines = src.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(def |class )", stripped):
            in_function = True
        if not in_function and re.match(r"^import vtk\b", stripped):
            pytest.fail("mrml.py has a module-level 'import vtk'")


# ---------------------------------------------------------------------------
# Behavioral: get_or_create_table_node (direct, using fake objects)
# ---------------------------------------------------------------------------

def test_existing_node_reference_is_reused():
    """get_or_create_table_node returns the existing node without creating a new one."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node, ROLE_RESULTS_TABLE

    existing = _FakeTableNode()
    existing.SetName("My renamed table")
    param_node = _FakeParamNode(existing_node=existing)
    scene = _FakeScene()

    result = get_or_create_table_node(param_node, scene)

    assert result is existing, "existing node reference not reused"
    assert scene._add_count == 0, "new node created despite existing reference"
    assert param_node._set_ref_calls == 0, "SetNodeReferenceID called unexpectedly"


def test_missing_reference_creates_node_with_display_name():
    """get_or_create_table_node creates exactly one new node with the canonical name."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node

    param_node = _FakeParamNode(existing_node=None)
    scene = _FakeScene()

    node = get_or_create_table_node(param_node, scene)

    assert node is not None
    assert scene._add_count == 1, f"expected 1 new node, got {scene._add_count}"
    assert node.GetName() == "ZebrafishEmbryoAnalyzer Results"


def test_new_node_id_stored_in_param_node():
    """get_or_create_table_node stores the new node ID in the parameter node."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node, ROLE_RESULTS_TABLE

    param_node = _FakeParamNode(existing_node=None)
    scene = _FakeScene()

    node = get_or_create_table_node(param_node, scene)

    assert param_node._set_ref_calls == 1, "SetNodeReferenceID not called"
    assert param_node._stored_role == ROLE_RESULTS_TABLE
    assert param_node._stored_id == node.GetID()


def test_renamed_node_is_reused():
    """A node renamed by the user is still found via the stored reference."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node

    existing = _FakeTableNode()
    existing.SetName("User renamed this")
    param_node = _FakeParamNode(existing_node=existing)
    scene = _FakeScene()

    result = get_or_create_table_node(param_node, scene)

    assert result is existing
    assert result.GetName() == "User renamed this", "user name was overwritten"
    assert scene._add_count == 0


def test_wrong_node_type_creates_new_table_node():
    """A reference to a non-table node triggers creation of a new table node."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node

    wrong_node = _FakeNonTableNode()
    param_node = _FakeParamNode(existing_node=wrong_node)
    scene = _FakeScene()

    result = get_or_create_table_node(param_node, scene)

    assert result is not wrong_node, "wrong-type foreign node must not be reused"
    assert scene._add_count == 1, "expected exactly one new table node"
    assert param_node._stored_id != wrong_node.GetID(), (
        "reference must point to the new node, not the wrong-type node"
    )
    assert result.IsA("vtkMRMLTableNode"), "new node must be a table node"


# ---------------------------------------------------------------------------
# Behavioral: populate_table_node / build_vtk_table (direct, fake vtk)
# ---------------------------------------------------------------------------

def test_populate_table_node_columns_and_names(monkeypatch):
    """populate_table_node creates one correctly named column per TABLE_SCHEMA entry."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod
    from ZebrafishEmbryoAnalyzerLib.mrml import TABLE_SCHEMA

    fake_vtk = _make_fake_vtk()
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)

    rows = [{"Filename": "a.png", "Length_um": 1.0, "CurvatureClass": "1",
              "LengthStraightRatio": 1.05, "EyeArea_um2": math.nan,
              "EyeDiameter_um": math.nan, "Error": ""}]
    node = _FakeTableNode()
    mrml_mod.populate_table_node(rows, node)

    assert node._table is not None
    assert node._table.GetNumberOfColumns() == len(TABLE_SCHEMA)
    expected_names = [col for col, _, _ in TABLE_SCHEMA]
    actual_names = [node._table.GetColumn(i).GetName()
                    for i in range(node._table.GetNumberOfColumns())]
    assert actual_names == expected_names


def test_populate_table_node_applies_atomically(monkeypatch):
    """populate_table_node only calls SetAndObserveTable after full construction."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod
    from ZebrafishEmbryoAnalyzerLib.mrml import TABLE_SCHEMA

    fake_vtk = _make_fake_vtk()
    set_observe_calls = []

    class _TrackingNode(_FakeTableNode):
        def SetAndObserveTable(self, t):
            set_observe_calls.append(t.GetNumberOfColumns())
            super().SetAndObserveTable(t)

    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)

    rows = [{"Filename": "a.png", "Length_um": 1.0, "CurvatureClass": "1",
              "LengthStraightRatio": 1.05, "EyeArea_um2": math.nan,
              "EyeDiameter_um": math.nan, "Error": ""}]
    node = _TrackingNode()
    mrml_mod.populate_table_node(rows, node)

    assert len(set_observe_calls) == 1, "SetAndObserveTable must be called exactly once"
    assert set_observe_calls[0] == len(TABLE_SCHEMA), (
        "SetAndObserveTable called with incomplete table"
    )


def test_populate_table_node_existing_table_preserved_on_failure(monkeypatch):
    """If vtk construction fails, the existing table on the node is not replaced."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod

    class _BrokenVTK:
        def vtkTable(self):
            raise RuntimeError("vtk construction failed")

    monkeypatch.setitem(sys.modules, "vtk", _BrokenVTK())

    original_sentinel = object()
    node = _FakeTableNode()
    node._table = original_sentinel

    rows = [{"Filename": "a.png", "Length_um": 1.0, "CurvatureClass": "1",
              "LengthStraightRatio": 1.05, "EyeArea_um2": math.nan,
              "EyeDiameter_um": math.nan, "Error": ""}]

    with pytest.raises(Exception):
        mrml_mod.populate_table_node(rows, node)

    assert node._table is original_sentinel, "existing table was overwritten on error"


def test_input_results_not_mutated_by_update(monkeypatch):
    """update_results_table must not mutate the input results list or dicts."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod

    fake_vtk = _make_fake_vtk()
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)

    results = [
        {
            "filename": "fish.png", "length": 1.0, "curvature": 2, "ratio": 1.05,
            "eye_area": None, "eye_diameter": None, "error": None,
        }
    ]
    original_results = [dict(r) for r in results]

    rows = mrml_mod.results_to_rows(results)
    node = _FakeTableNode()
    mrml_mod.populate_table_node(rows, node)

    assert results[0] == original_results[0], "input result dict was mutated"


# ---------------------------------------------------------------------------
# Subprocess: update_results_table integration
# ---------------------------------------------------------------------------

def test_update_results_table_calls_mrml_functions():
    """update_results_table builds the vtk table then resolves/creates the MRML node."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        fake_table = MagicMock()
        fake_node = MagicMock()
        fake_node.GetID.return_value = "nodeID1"

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table",
                   return_value=fake_table) as mock_build, \\
             patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_table_node",
                   return_value=fake_node) as mock_get:
            import slicer
            result = logic.update_results_table([
                {"filename": "a.png", "length": 1.0, "curvature": 1, "ratio": 1.0,
                 "eye_area": None, "eye_diameter": None, "error": None}
            ])

        assert mock_build.called, "build_vtk_table not called"
        assert mock_get.called, "get_or_create_table_node not called"
        fake_node.SetAndObserveTable.assert_called_once_with(fake_table)
        assert result is fake_node
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_vtk_build_failure_creates_no_node():
    """If build_vtk_table raises, get_or_create_table_node must not be called."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        node_creation_calls = []

        def _record_create(param_node, scene):
            node_creation_calls.append(1)
            return MagicMock()

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table",
                   side_effect=RuntimeError("vtk unavailable")), \\
             patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_table_node",
                   _record_create):
            try:
                logic.update_results_table([
                    {"filename": "a.png", "length": None, "curvature": None,
                     "ratio": None, "eye_area": None, "eye_diameter": None, "error": None}
                ])
            except MRMLAdapterError:
                pass

        assert not node_creation_calls, (
            f"get_or_create_table_node was called despite build failure: {node_creation_calls}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_results_table_wraps_unexpected_exception_as_mrml_error():
    """update_results_table wraps unexpected exceptions as MRMLAdapterError."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table",
                   side_effect=RuntimeError("vtk broke")):
            try:
                logic.update_results_table([
                    {"filename": "a.png", "length": None, "curvature": None,
                     "ratio": None, "eye_area": None, "eye_diameter": None, "error": None}
                ])
                print("NO_ERROR")
            except MRMLAdapterError as exc:
                print(f"OK:{exc}")
            except Exception as exc:
                print(f"WRONG_TYPE:{type(exc).__name__}:{exc}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK:"), r.stdout
    assert "NO_ERROR" not in r.stdout


def test_widget_mrml_failure_preserves_results_via_helper():
    """_try_update_mrml_table must not affect self._results on MRMLAdapterError."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._results = [{"filename": "fish.png"}]

        mock_logic = MagicMock()
        mock_logic.update_results_table.side_effect = MRMLAdapterError("simulated")
        w._logic = mock_logic

        import slicer

        # Call the actual production method, not a hand-written copy
        w._try_update_mrml_table(w._results)

        assert w._results == [{"filename": "fish.png"}], (
            f"_results changed: {w._results!r}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_run_analysis_has_no_mrml_calls():
    """run_analysis() must not call update_results_table or any MRML function."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()

        calls = []

        def _fake_update(results):
            calls.append("update_results_table")

        logic.update_results_table = _fake_update

        with patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images",
                   return_value=[{"filename": "x.png"}]):
            logic.run_analysis(["/x.png"], {"um_per_px": 1.0})

        assert not calls, (
            f"run_analysis() called update_results_table: {calls}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Static checks for E2b
# ---------------------------------------------------------------------------

def test_mrml_module_has_no_global_numpy_import():
    """mrml.py must not have a module-level 'import numpy'."""
    src = open(_MRML_PY).read()
    lines = src.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(def |class )", stripped):
            in_function = True
        if not in_function and re.match(r"^import numpy\b", stripped):
            pytest.fail("mrml.py has a module-level 'import numpy'")


def test_widget_source_does_not_contain_persistent_image_node_pointer():
    """widget.py must not store a persistent node pointer like self._image_node."""
    src = open(_WIDGET_PY).read()
    assert "self._image_node" not in src, (
        "widget.py must not keep a persistent _image_node pointer — "
        "ownership is via parameter node reference"
    )


def test_widget_source_calls_show_gallery_selection_in_slice_view():
    """widget.py must call show_gallery_selection_in_slice_view for each gallery click.

    Issue #56: gallery selection drives slice-view background + segmentation
    visibility through this single logic-method call (replaces the old
    singleton ``update_current_image_node`` / ``update_current_segmentation_node``
    pair).
    """
    src = open(_WIDGET_PY).read()
    assert "show_gallery_selection_in_slice_view" in src, (
        "widget.py must call self._logic.show_gallery_selection_in_slice_view()"
    )
    # The old singleton-helper calls must be gone — otherwise the new
    # path runs alongside the old one and the singleton node resurrects.
    assert "update_current_image_node" not in src, (
        "widget.py must not call the removed update_current_image_node"
    )
    assert "update_current_segmentation_node" not in src, (
        "widget.py must not call the removed update_current_segmentation_node"
    )
    assert "_try_update_mrml_image" not in src
    assert "_try_update_mrml_segmentation" not in src


def test_mrml_module_exports_image_functions():
    """mrml.py must export all required E2b symbols."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    assert hasattr(mrml, "image_geometry")
    assert hasattr(mrml, "update_image_node")
    assert hasattr(mrml, "create_image_volume_node")
    # Issue #56: gallery-selection replaces the singleton mechanism, so the
    # removed helpers must no longer exist.
    assert not hasattr(mrml, "ROLE_CURRENT_IMAGE"), (
        "ROLE_CURRENT_IMAGE was retired in issue #56"
    )
    assert not hasattr(mrml, "get_or_create_image_node"), (
        "get_or_create_image_node was retired in issue #56"
    )


# ---------------------------------------------------------------------------
# Static checks for E2c
# ---------------------------------------------------------------------------

def test_mrml_module_exports_segmentation_symbols():
    """mrml.py must export all required E2c symbols."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    assert hasattr(mrml, "resample_mask_to_original")
    assert hasattr(mrml, "update_segmentation_node")
    # Issue #56: the singleton CurrentSegmentation mechanism is gone.
    assert not hasattr(mrml, "ROLE_CURRENT_SEGMENTATION"), (
        "ROLE_CURRENT_SEGMENTATION was retired in issue #56"
    )
    assert not hasattr(mrml, "get_or_create_segmentation_node"), (
        "get_or_create_segmentation_node was retired in issue #56"
    )


def test_widget_source_does_not_contain_persistent_segmentation_node_pointer():
    """widget.py must not store a persistent node pointer like self._segmentation_node."""
    src = open(_WIDGET_PY).read()
    assert "self._segmentation_node" not in src, (
        "widget.py must not keep a persistent _segmentation_node pointer — "
        "ownership is via parameter node reference"
    )


# ---------------------------------------------------------------------------
# Behavioral: set_slice_viewer_background (direct, with fake slicer)
# ---------------------------------------------------------------------------

def test_set_slice_viewer_background_calls_slicer_with_volume_node():
    """set_slice_viewer_background forwards volume_node to slicer.util.setSliceViewerLayers."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    from unittest.mock import MagicMock, patch

    volume = object()
    fake_slicer = MagicMock()
    with patch.dict(
        "sys.modules",
        {"slicer": fake_slicer, "ZebrafishEmbryoAnalyzerLib.mrml": mrml},
        clear=False,
    ):
        ok = mrml.set_slice_viewer_background(volume)

    assert ok is True
    fake_slicer.util.setSliceViewerLayers.assert_called_once_with(background=volume)


def test_set_slice_viewer_background_returns_false_when_slicer_missing():
    """No-slicer environments (e.g. plain pytest) must no-op without raising."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    from unittest.mock import patch

    with patch.dict("sys.modules", {"slicer": None}, clear=False):
        ok = mrml.set_slice_viewer_background(object())

    assert ok is False


def test_set_slice_viewer_background_swallows_slicer_errors():
    """A slicer.util failure must not propagate to the caller."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    from unittest.mock import MagicMock, patch

    fake_slicer = MagicMock()
    fake_slicer.util.setSliceViewerLayers.side_effect = RuntimeError("boom")
    with patch.dict("sys.modules", {"slicer": fake_slicer}, clear=False):
        ok = mrml.set_slice_viewer_background(object())

    assert ok is False


def test_set_segmentation_visibility_toggles_display_node():
    """set_segmentation_visibility forwards to the segmentation display node."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    display = MagicMock()
    seg = MagicMock()
    seg.GetDisplayNode.return_value = display

    assert mrml.set_segmentation_visibility(seg, True) is True
    display.SetVisibility.assert_called_once_with(True)

    display2 = MagicMock()
    seg2 = MagicMock()
    seg2.GetDisplayNode.return_value = display2
    assert mrml.set_segmentation_visibility(seg2, False) is True
    display2.SetVisibility.assert_called_once_with(False)


def test_set_segmentation_visibility_handles_missing_display_node():
    """A segmentation without a display node must no-op cleanly."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    seg = MagicMock()
    seg.GetDisplayNode.return_value = None
    assert mrml.set_segmentation_visibility(seg, True) is False
    assert mrml.set_segmentation_visibility(None, True) is False


def test_find_tracked_volume_node_by_filename_returns_matching_node():
    """find_tracked_volume_node_by_filename returns the node whose name equals filename."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    from unittest.mock import MagicMock

    node_a = MagicMock(); node_a.GetName.return_value = "first.png"
    node_b = MagicMock(); node_b.GetName.return_value = "second.png"
    param = MagicMock()
    scene = MagicMock()
    with patch.object(
        mrml, "list_tracked_volume_nodes", return_value=[node_a, node_b]
    ):
        found = mrml.find_tracked_volume_node_by_filename(param, scene, "second.png")
    assert found is node_b


def test_find_tracked_volume_node_by_filename_returns_none_on_miss():
    """No match -> None; an empty filename -> None; missing scene -> None."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    from unittest.mock import MagicMock, patch

    other = MagicMock(); other.GetName.return_value = "other.png"
    param = MagicMock()
    scene = MagicMock()

    with patch.object(mrml, "list_tracked_volume_nodes", return_value=[other]):
        assert mrml.find_tracked_volume_node_by_filename(param, scene, "absent.png") is None
        assert mrml.find_tracked_volume_node_by_filename(param, scene, "") is None
        assert mrml.find_tracked_volume_node_by_filename(None, scene, "x.png") is None
        assert mrml.find_tracked_volume_node_by_filename(param, None, "x.png") is None


# ---------------------------------------------------------------------------
# Behavioral: show_gallery_selection_in_slice_view (subprocess)
# ---------------------------------------------------------------------------

def test_show_gallery_selection_in_slice_view_no_op_when_param_node_missing():
    """No parameter node -> silent no-op, no exception propagates."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=None)

        result = logic.show_gallery_selection_in_slice_view({"filename": "x.png"})
        assert result is None, f"expected None, got {result!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_show_gallery_selection_in_slice_view_no_op_when_result_none():
    """None result -> silent no-op, no exception propagates."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        result = logic.show_gallery_selection_in_slice_view(None)
        assert result is None, f"expected None, got {result!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_show_gallery_selection_in_slice_view_no_op_when_no_matching_volume_node():
    """A result whose filename does not match any tracked volume node must no-op silently."""
    r = _run("""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        # No tracked nodes -> find_tracked_volume_node_by_filename returns None.
        with patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.find_tracked_volume_node_by_filename",
            return_value=None,
        ):
            result = logic.show_gallery_selection_in_slice_view({"filename": "absent.png"})
        assert result is None
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_show_gallery_selection_in_slice_view_sets_background_and_toggles_visibility():
    """Clicking a gallery thumbnail sets the slice-view background and toggles
    the per-image segmentation visibility on (and the previously-visible one
    off)."""
    r = _run("""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        # Real Slicer returns "" for unset parameter strings; mimic that
        # so the "no previous seg id" branch is taken on the first click.
        param = MagicMock(); param.GetParameter.return_value = ""
        logic.getParameterNode = MagicMock(return_value=param)

        seg = MagicMock(); seg.GetID.return_value = "SegId1"
        volume = MagicMock(); volume.GetNodeReference.return_value = seg

        with patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.find_tracked_volume_node_by_filename",
            return_value=volume,
        ), patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.set_slice_viewer_background",
            return_value=True,
        ) as set_bg, patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.set_segmentation_visibility",
        ) as set_vis:
            logic.show_gallery_selection_in_slice_view({"filename": "x.png"})

        set_bg.assert_called_once_with(volume)
        # First click: no previous segmentation -> only the new one is shown.
        set_vis.assert_called_once_with(seg, True)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_show_gallery_selection_in_slice_view_hides_previous_segmentation():
    """A second click must hide the previously-visible segmentation before showing the new one."""
    r = _run("""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        param = MagicMock()
        param.GetParameter.return_value = "SegIdPrev"
        logic.getParameterNode = MagicMock(return_value=param)

        prev_seg = MagicMock()
        new_seg  = MagicMock(); new_seg.GetID.return_value = "SegIdNew"
        volume   = MagicMock(); volume.GetNodeReference.return_value = new_seg

        scene = MagicMock()
        scene.GetNodeByID.return_value = prev_seg
        # Patch slicer.mrmlScene lookup at import time.
        fake_slicer = MagicMock()
        fake_slicer.mrmlScene = scene

        with patch.dict("sys.modules", {"slicer": fake_slicer}), patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.find_tracked_volume_node_by_filename",
            return_value=volume,
        ), patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.set_slice_viewer_background",
        ), patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.set_segmentation_visibility",
        ) as set_vis:
            logic.show_gallery_selection_in_slice_view({"filename": "y.png"})

        # Visibility toggles: prev -> False, new -> True (in this order).
        # The two nodes are different MagicMock instances in the test
        # scope; assert by visibility value (the second positional arg),
        # which is what differentiates them.
        visibilities = [c.args[1] for c in set_vis.call_args_list]
        assert visibilities == [False, True], visibilities
        # Old seg-id parameter was overwritten with the new one.
        param.SetParameter.assert_called_with(
            "ZebrafishAnalysis.previousVisibleSegmentationId", "SegIdNew"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_show_gallery_selection_in_slice_view_handles_no_segmentation_reference():
    """A volume node without a segmentation reference (analysis not run yet)
    must still set the slice-view background — it must just skip the
    segmentation toggle entirely."""
    r = _run("""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        param = MagicMock(); param.GetParameter.return_value = ""
        logic.getParameterNode = MagicMock(return_value=param)

        volume = MagicMock(); volume.GetNodeReference.return_value = None

        with patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.find_tracked_volume_node_by_filename",
            return_value=volume,
        ), patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.set_slice_viewer_background",
        ) as set_bg, patch(
            "ZebrafishEmbryoAnalyzerLib.mrml.set_segmentation_visibility",
        ) as set_vis:
            logic.show_gallery_selection_in_slice_view({"filename": "z.png"})

        set_bg.assert_called_once_with(volume)
        # No seg reference -> no toggle, no error.
        set_vis.assert_not_called()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: update_image_node with fake VTK (numpy.int64 round-trip)
# ---------------------------------------------------------------------------

def test_update_image_node_calls_set_dimensions_with_correct_values():
    """update_image_node derives (W, H, 1) dimensions from the image array."""
    import numpy as np
    import importlib
    from unittest.mock import MagicMock

    # Build fake VTK objects
    fake_vtk_array = MagicMock()
    fake_numpy_support = MagicMock()
    fake_numpy_support.numpy_to_vtk.return_value = fake_vtk_array

    fake_image_data = MagicMock()
    fake_vtk_module = MagicMock()
    fake_vtk_module.vtkImageData.return_value = fake_image_data
    fake_vtk_module.vtkMatrix4x4.return_value = MagicMock()
    fake_vtk_module.VTK_UNSIGNED_CHAR = 3

    fake_vtk_util = MagicMock()
    fake_vtk_util.numpy_support = fake_numpy_support

    fake_node = MagicMock()

    # Inject fake vtk into sys.modules for the duration of this test
    original_vtk = sys.modules.pop("vtk", None)
    original_vtk_util = sys.modules.pop("vtk.util", None)
    original_vtk_util_ns = sys.modules.pop("vtk.util.numpy_support", None)
    try:
        sys.modules["vtk"] = fake_vtk_module
        sys.modules["vtk.util"] = fake_vtk_util
        sys.modules["vtk.util.numpy_support"] = fake_numpy_support

        # Force reimport of mrml to pick up fake vtk
        import ZebrafishEmbryoAnalyzerLib.mrml as mrml_mod
        importlib.reload(mrml_mod)

        image = np.zeros((10, 8, 3), dtype="uint8")
        mrml_mod.update_image_node(image, 22.99, fake_node)

        # Dimensions must be (W=8, H=10, 1)
        fake_image_data.SetDimensions.assert_called_once_with((8, 10, 1))
        # SetSpacing must be called before SetAndObserveImageData
        assert fake_node.SetSpacing.called
        assert fake_node.SetAndObserveImageData.called
        # update_image_node calls node.SetSpacing(*spacing) which unpacks the tuple
        spacing_args = fake_node.SetSpacing.call_args[0]
        assert spacing_args[0] == pytest.approx(22.99 / 1000.0)
        assert spacing_args[1] == pytest.approx(22.99 / 1000.0)
        assert spacing_args[2] == pytest.approx(1.0)
    finally:
        sys.modules.pop("vtk", None)
        sys.modules.pop("vtk.util", None)
        sys.modules.pop("vtk.util.numpy_support", None)
        if original_vtk is not None:
            sys.modules["vtk"] = original_vtk
        if original_vtk_util is not None:
            sys.modules["vtk.util"] = original_vtk_util
        if original_vtk_util_ns is not None:
            sys.modules["vtk.util.numpy_support"] = original_vtk_util_ns
        # Reload mrml again with real (absent) vtk to restore state
        importlib.reload(mrml_mod)


# ---------------------------------------------------------------------------
# Issue #56 follow-up: Data-module deletions are ground truth
# ---------------------------------------------------------------------------

def test_logic_clear_stale_flag_for_volume_node_calls_mrml_helper():
    """Logic.clear_stale_flag_for_volume_node forwards to mrml.clear_volume_node_stale.

    Issue #56 follow-up: when the user deletes a segmentation node
    inside the Data module, the stale flag must be cleared silently
    (without recreating the segmentation). The widget relies on this
    Logic method so it does not need to import mrml directly.
    """
    r = _run("""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = lambda: MagicMock()
        vol = MagicMock()
        with patch.object(mrml_mod, "clear_volume_node_stale") as clear_helper:
            logic.clear_stale_flag_for_volume_node(vol)
        clear_helper.assert_called_once_with(vol)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout, r.stderr


def test_logic_clear_stale_flag_for_volume_node_swallows_import_failure():
    """A broken import must not propagate to the widget.

    Forces ``mrml.clear_volume_node_stale`` to be missing by stubbing
    a fake mrml module without it — verifies the logic layer fails
    closed (no exception) instead of crashing the recompute-prompt
    loop.
    """
    r = _run("""
        import sys, types
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = lambda: MagicMock()

        fake_mrml = types.ModuleType("ZebrafishEmbryoAnalyzerLib.mrml")
        # Deliberately no clear_volume_node_stale attribute.
        sys.modules["ZebrafishEmbryoAnalyzerLib.mrml"] = fake_mrml
        # Should swallow the AttributeError from the missing helper.
        logic.clear_stale_flag_for_volume_node(MagicMock())
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout, r.stderr


def test_logic_volume_node_references_existing_seg():
    """volume_node_references_existing_seg resolves the
    ROLE_ZEBRAFISH_SEGMENTATION reference against the live scene.

    Returns True only when the referenced segmentation node is still
    in the scene; False for any error condition or when the reference
    id is empty. Issue #56 follow-up hook for the recompute-prompt
    and detail-recompute-button paths.
    """
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        import slicer

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = lambda: MagicMock()

        vol = MagicMock()
        vol.GetNodeReferenceID.return_value = "Seg42"
        slicer.mrmlScene.GetNodeByID.return_value = MagicMock(name="present")
        assert logic.volume_node_references_existing_seg(vol) is True

        slicer.mrmlScene.GetNodeByID.return_value = None
        assert logic.volume_node_references_existing_seg(vol) is False

        vol.GetNodeReferenceID.return_value = ""
        assert logic.volume_node_references_existing_seg(vol) is False

        assert logic.volume_node_references_existing_seg(None) is False
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout, r.stderr


def test_prompt_recompute_stale_images_skips_volumes_with_deleted_seg():
    """prompt_recompute_stale_images must NOT prompt for or recreate a
    volume whose segmentation node has been removed from the scene.

    Issue #56 follow-up: data-module deletions are ground truth, so the
    auto-prompt must silently drop the stale volume (and clear the
    stale flag) rather than run the recompute pipeline that
    recreates the segmentation.
    """
    r = _run("""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        import ZebrafishEmbryoAnalyzerLib.widget as w_mod

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = lambda: MagicMock()

        # Two stale volumes — one whose seg is gone, one whose seg
        # is still present. Only the latter should be prompted.
        deleted_vol = MagicMock(); deleted_vol.GetName.return_value = "deleted.png"
        deleted_vol.GetNodeReferenceID.return_value = "SegGone"
        present_vol = MagicMock(); present_vol.GetName.return_value = "present.png"
        present_vol.GetNodeReferenceID.return_value = "SegStill"

        scene = MagicMock()
        def _get_node(id_):
            return None if id_ == "SegGone" else MagicMock()
        scene.GetNodeByID.side_effect = _get_node
        fake_slicer = MagicMock(); fake_slicer.mrmlScene = scene

        prompt_calls = []
        def _policy(_names):
            # Issue #83: one prompt for the whole batch, so the policy gets a
            # list of filenames rather than a single name.
            prompt_calls.append(list(_names))
            return "yes"

        # Construct the widget without running __init__ so we do not
        # need a Qt parent / layout. Only inject the dependencies the
        # prompt loop actually touches.
        widget = w_mod.ZebrafishEmbryoAnalyzerMainWidget.__new__(
            w_mod.ZebrafishEmbryoAnalyzerMainWidget
        )
        widget._logic = logic
        widget._stale_recompute_prompt_policy = _policy
        recompute_calls = []
        widget._recompute_for_volume_node = lambda vol: recompute_calls.append(vol)

        with patch.dict("sys.modules", {"slicer": fake_slicer}), patch.object(
            logic, "list_stale_tracked_volume_nodes", return_value=[deleted_vol, present_vol]
        ), patch.object(logic, "clear_stale_flag_for_volume_node") as clear_fn:
            widget.prompt_recompute_stale_images()

        # 1) The deleted volume is cleared silently and never reaches
        #    the prompt or the recompute pipeline.
        clear_fn.assert_called_once_with(deleted_vol)
        # 2) The prompt policy is called exactly once — for the
        #    volume whose segmentation is still in the scene.
        assert prompt_calls == [["present.png"]], (
            f"Prompt called with {prompt_calls!r}; exactly one prompt naming "
            "only the still-present seg was expected."
        )
        # 3) Recompute runs only for the still-present seg.
        assert recompute_calls == [present_vol], (
            f"Recompute called with {recompute_calls!r}; the deleted-seg "
            "volume must not be re-created."
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout, r.stderr


# ---------------------------------------------------------------------------
# Issue #85 — segmentations must be nested under their volume in the Data tree
# ---------------------------------------------------------------------------

class _FakeSubjectHierarchy:
    """Models the part of vtkMRMLSubjectHierarchyNode the reparenting uses.

    Item ids are ints; 0 means "no item", which is what
    ``GetItemByDataNode`` returns for a node the hierarchy has not picked up
    yet — the case that used to make the reparent silently do nothing.
    """

    SCENE_ITEM = 1

    def __init__(self, known=()):
        self._items = {}
        self._parents = {}
        self._next = 100
        for node in known:
            self._add(node)

    def _add(self, node):
        self._next += 1
        self._items[id(node)] = self._next
        self._parents[self._next] = self.SCENE_ITEM
        return self._next

    def GetSceneItemID(self):
        return self.SCENE_ITEM

    def GetItemByDataNode(self, node):
        return self._items.get(id(node), 0)

    def CreateItem(self, parent_item, node):
        item = self._add(node)
        self._parents[item] = parent_item
        return item

    def GetItemParent(self, item):
        return self._parents.get(item, 0)

    def SetItemParent(self, item, parent_item):
        self._parents[item] = parent_item


def _reparent_with(sh, child, parent):
    """Run the real helper against a fake hierarchy."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    fake_slicer = MagicMock()
    fake_slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode.return_value = sh
    with patch.dict("sys.modules", {"slicer": fake_slicer}):
        mrml._reparent_in_subject_hierarchy(MagicMock(name="scene"), child, parent)


def test_reparent_creates_the_item_when_the_hierarchy_has_not_seen_the_node():
    """The defect behind #85: a freshly created segmentation has no Subject
    Hierarchy item yet, so the old ``if parent_item and child_item`` guard
    skipped it and the node stayed at the scene root — while the markups
    created moments later, which the hierarchy had caught up with, nested
    correctly.
    """
    volume, seg = MagicMock(name="volume"), MagicMock(name="seg")
    sh = _FakeSubjectHierarchy(known=[volume])          # seg unknown on purpose
    assert sh.GetItemByDataNode(seg) == 0

    _reparent_with(sh, seg, volume)

    seg_item = sh.GetItemByDataNode(seg)
    assert seg_item, "an item must be created for the segmentation"
    assert sh.GetItemParent(seg_item) == sh.GetItemByDataNode(volume)


def test_reparent_nests_a_node_the_hierarchy_already_knows():
    volume, seg = MagicMock(name="volume"), MagicMock(name="seg")
    sh = _FakeSubjectHierarchy(known=[volume, seg])

    _reparent_with(sh, seg, volume)

    assert sh.GetItemParent(sh.GetItemByDataNode(seg)) == sh.GetItemByDataNode(volume)


def test_reparent_leaves_a_node_the_user_filed_elsewhere_alone():
    """Idempotent and non-destructive: re-running an analysis must not drag a
    node back that the user deliberately moved somewhere else in the tree.
    """
    volume, seg, folder = MagicMock(), MagicMock(), MagicMock()
    sh = _FakeSubjectHierarchy(known=[volume, seg, folder])
    folder_item = sh.GetItemByDataNode(folder)
    sh.SetItemParent(sh.GetItemByDataNode(seg), folder_item)

    _reparent_with(sh, seg, volume)

    assert sh.GetItemParent(sh.GetItemByDataNode(seg)) == folder_item
