"""
Tests for MRML node creation, reuse, and update_results_table orchestration.

Static checks verify source-level contracts without subprocess overhead.
Behavioral node tests use small fake objects directly — conftest.py adds the
ZebrafishAnalysis directory to sys.path so mrml.py imports cleanly.
Subprocess tests cover the full update_results_table flow, which requires
the Slicer module stub so ZebrafishAnalysis.py can be imported.
"""

import math
import os
import re
import subprocess
import sys
import textwrap
import types

import pytest

_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ZebrafishAnalysis",
)
_MAIN_PY   = os.path.join(_MODULE_DIR, "ZebrafishAnalysis.py")
_WIDGET_PY = os.path.join(_MODULE_DIR, "ZebrafishAnalysisLib", "widget.py")
_LOGIC_PY  = os.path.join(_MODULE_DIR, "ZebrafishAnalysisLib", "logic.py")
_MRML_PY   = os.path.join(_MODULE_DIR, "ZebrafishAnalysisLib", "mrml.py")
_CMAKE     = os.path.join(
    os.path.dirname(_MODULE_DIR), "ZebrafishAnalysis", "CMakeLists.txt"
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
# Source-level static checks
# ---------------------------------------------------------------------------

def test_logic_py_does_not_import_mrml():
    """ZebrafishAnalysisLib.logic must not import the mrml adapter."""
    src = open(_LOGIC_PY).read()
    assert "mrml" not in src, (
        "ZebrafishAnalysisLib.logic must not import ZebrafishAnalysisLib.mrml"
    )


def test_mrml_in_cmake():
    """ZebrafishAnalysis/CMakeLists.txt must list ZebrafishAnalysisLib/mrml.py."""
    content = open(_CMAKE).read()
    assert "ZebrafishAnalysisLib/mrml.py" in content, (
        "CMakeLists.txt does not include ZebrafishAnalysisLib/mrml.py"
    )


def test_mrml_in_reload_eviction_list():
    """ZebrafishAnalysis.py _LIB_MODULES must include ZebrafishAnalysisLib.mrml."""
    src = open(_MAIN_PY).read()
    assert '"ZebrafishAnalysisLib.mrml"' in src, (
        "_LIB_MODULES must include 'ZebrafishAnalysisLib.mrml'"
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
    assert "from ZebrafishAnalysisLib.mrml" not in src, (
        "widget.py must not import ZebrafishAnalysisLib.mrml directly"
    )
    assert "from ZebrafishAnalysisLib import mrml" not in src, (
        "widget.py must not import ZebrafishAnalysisLib.mrml directly"
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
    from ZebrafishAnalysisLib.mrml import get_or_create_table_node, ROLE_RESULTS_TABLE

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
    from ZebrafishAnalysisLib.mrml import get_or_create_table_node

    param_node = _FakeParamNode(existing_node=None)
    scene = _FakeScene()

    node = get_or_create_table_node(param_node, scene)

    assert node is not None
    assert scene._add_count == 1, f"expected 1 new node, got {scene._add_count}"
    assert node.GetName() == "ZebrafishAnalysis Results"


def test_new_node_id_stored_in_param_node():
    """get_or_create_table_node stores the new node ID in the parameter node."""
    from ZebrafishAnalysisLib.mrml import get_or_create_table_node, ROLE_RESULTS_TABLE

    param_node = _FakeParamNode(existing_node=None)
    scene = _FakeScene()

    node = get_or_create_table_node(param_node, scene)

    assert param_node._set_ref_calls == 1, "SetNodeReferenceID not called"
    assert param_node._stored_role == ROLE_RESULTS_TABLE
    assert param_node._stored_id == node.GetID()


def test_renamed_node_is_reused():
    """A node renamed by the user is still found via the stored reference."""
    from ZebrafishAnalysisLib.mrml import get_or_create_table_node

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
    from ZebrafishAnalysisLib.mrml import get_or_create_table_node

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
    from ZebrafishAnalysisLib import mrml as mrml_mod
    from ZebrafishAnalysisLib.mrml import TABLE_SCHEMA

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
    from ZebrafishAnalysisLib import mrml as mrml_mod
    from ZebrafishAnalysisLib.mrml import TABLE_SCHEMA

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
    from ZebrafishAnalysisLib import mrml as mrml_mod

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
    from ZebrafishAnalysisLib import mrml as mrml_mod

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
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        fake_table = MagicMock()
        fake_node = MagicMock()
        fake_node.GetID.return_value = "nodeID1"

        with patch("ZebrafishAnalysisLib.mrml.build_vtk_table",
                   return_value=fake_table) as mock_build, \\
             patch("ZebrafishAnalysisLib.mrml.get_or_create_table_node",
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
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        from ZebrafishAnalysisLib.errors import MRMLAdapterError

        logic = ZebrafishAnalysisLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        node_creation_calls = []

        def _record_create(param_node, scene):
            node_creation_calls.append(1)
            return MagicMock()

        with patch("ZebrafishAnalysisLib.mrml.build_vtk_table",
                   side_effect=RuntimeError("vtk unavailable")), \\
             patch("ZebrafishAnalysisLib.mrml.get_or_create_table_node",
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
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        from ZebrafishAnalysisLib.errors import MRMLAdapterError

        logic = ZebrafishAnalysisLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        with patch("ZebrafishAnalysisLib.mrml.build_vtk_table",
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
        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget
        from ZebrafishAnalysisLib.errors import MRMLAdapterError

        w = object.__new__(ZebrafishAnalysisMainWidget)
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
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()

        calls = []

        def _fake_update(results):
            calls.append("update_results_table")

        logic.update_results_table = _fake_update

        with patch("ZebrafishAnalysisLib.logic.analyse_images",
                   return_value=[{"filename": "x.png"}]):
            logic.run_analysis(["/x.png"], {"um_per_px": 1.0})

        assert not calls, (
            f"run_analysis() called update_results_table: {calls}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
