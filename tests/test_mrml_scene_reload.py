"""
Tests for issue #41: scene reload auto-detects and rebuilds the full
module state from the saved MRML scene.

Covers:
- ``volume_node_to_pixels`` round-trips a uint8 (H, W, 3) array, applying
  the inverse of update_image_node's flipud+fliplr transform.
- ``volume_node_to_pixels`` returns None for nodes without image data.
- ``validate_volume_node`` returns ("", "") for healthy nodes and for
  never-analyzed nodes (pending, not an error); a descriptive error only
  for a node that was analyzed but lost its segmentation reference.
- ``volume_node_to_result_dict_with_validation`` flags broken-seg-ref rows
  via the existing error-row auto-exclude mechanism, and leaves
  never-analyzed rows unflagged.
- ``logic.rebuild_results_from_scene`` walks ``ROLE_ZEBRAFISH_IMAGES`` in
  insertion order and produces one result dict per node with the right
  fields populated.
- ``rebuild_results_from_scene`` skips non-volume nodes and missing IDs
  silently (acceptance: do not crash on broken references).
- ``widget.rebuild_from_scene`` populates the in-memory state without
  triggering an inference run.
- A node whose segmentation reference is broken surfaces as an
  auto-excluded error row.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

@pytest.fixture
def mrml_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "qt", MagicMock())
    monkeypatch.setitem(sys.modules, "ctk", MagicMock())
    monkeypatch.setitem(sys.modules, "slicer", MagicMock())
    # vtk.util.numpy_support is the only piece of vtk that volume_node_to_pixels
    # actually uses; we leave vtk itself mock-able so each test can patch the
    # numpy conversion via sys.modules["vtk.util"] / ["vtk.util.numpy_support"].
    monkeypatch.setitem(sys.modules, "vtk", MagicMock())
    monkeypatch.setitem(sys.modules, "vtk.util", MagicMock())
    monkeypatch.setitem(sys.modules, "vtk.util.numpy_support", MagicMock())
    import importlib
    import ZebrafishEmbryoAnalyzerLib.mrml as module
    return importlib.reload(module)


# ---------------------------------------------------------------------------
# Pixel-array round-trip
# ---------------------------------------------------------------------------

def test_volume_node_to_pixels_round_trip(monkeypatch, mrml_module):
    """A numpy uint8 (H, W, 3) array written via update_image_node must
    come back out via volume_node_to_pixels in visual orientation
    (the inverse flip)."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    h, w = 12, 18
    rgb_visual = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_visual[..., 0] = np.arange(w, dtype=np.uint8)        # R varies by col
    rgb_visual[..., 1] = np.arange(h, dtype=np.uint8)[:, None]  # G varies by row

    # Fake volume node that returns a vtkImageData whose scalars are the
    # VTK-flipped version of the visual array (the order update_image_node
    # writes).
    flipped = np.flipud(np.fliplr(rgb_visual)).copy()
    flat = flipped.reshape(-1, 3)
    scalars_array = flat.reshape(h * w, 3)

    fake_scalars = MagicMock()
    fake_scalars.GetNumberOfComponents.return_value = 3
    fake_scalars._numpy = scalars_array

    fake_image_data = MagicMock()
    fake_image_data.GetDimensions.return_value = (w, h, 1)
    fake_image_data.GetPointData().GetScalars.return_value = fake_scalars

    # Patch vtk.util.numpy_support.vtk_to_numpy to return our pre-flipped
    # array. numpy_support is imported lazily inside the helper.
    fake_numpy_support = types.SimpleNamespace(
        vtk_to_numpy=lambda _: scalars_array,
        numpy_to_vtk=lambda *a, **kw: None,
    )
    fake_vtk_util = types.SimpleNamespace(numpy_support=fake_numpy_support)
    monkeypatch.setitem(sys.modules, "vtk", MagicMock())
    monkeypatch.setitem(sys.modules, "vtk.util", fake_vtk_util)

    fake_node = MagicMock()
    fake_node.GetImageData.return_value = fake_image_data

    pixels = mrml.volume_node_to_pixels(fake_node)
    assert pixels is not None
    assert pixels.shape == (h, w, 3)
    assert pixels.dtype == np.uint8
    # After the inverse flip, the [0,0] pixel of the visual array must be
    # at pixels[0, 0].
    assert np.array_equal(pixels[0, 0], rgb_visual[0, 0])


def test_volume_node_to_pixels_returns_none_when_no_image_data(mrml_module):
    """Nodes without image data must return None, not crash."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    fake_node = MagicMock()
    fake_node.GetImageData.return_value = None

    pixels = mrml.volume_node_to_pixels(fake_node)
    assert pixels is None


def test_volume_node_to_pixels_returns_none_on_exception(monkeypatch, mrml_module):
    """Defensive: a node whose GetImageData raises must not crash the caller."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    fake_node = MagicMock()
    fake_node.GetImageData.side_effect = RuntimeError("VTK exploded")

    pixels = mrml.volume_node_to_pixels(fake_node)
    assert pixels is None


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validate_volume_node_healthy(mrml_module):
    """Healthy node: returns empty error string."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    node = MagicMock()
    node.GetAttribute.return_value = "1.234"
    node.GetNodeReferenceID.return_value = "segNodeId"

    err_field, _msg = mrml.validate_volume_node(node)
    assert err_field == ""


def test_validate_volume_node_never_analyzed_is_not_an_error(mrml_module):
    """A tracked-but-never-analyzed node (no ZebrafishAnalysis.* attributes
    at all — issue #38 eager-load, before "Run Analysis") is a normal
    pending state, not an error."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    node = MagicMock()
    node.GetAttribute.return_value = None
    node.GetNodeReferenceID.return_value = "segNodeId"

    err_field, _msg = mrml.validate_volume_node(node)
    assert err_field == ""


def test_validate_volume_node_no_seg_ref(mrml_module):
    """A volume node that was analyzed but lost its segmentation reference
    is flagged."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    # ATTR_EXCLUDE present marks this node as "analysis has run" — the
    # signal validate_volume_node uses to distinguish a broken analyzed
    # node from a never-analyzed one.
    attrs = {"ZebrafishAnalysis.length": "1.234", "ZebrafishAnalysis.exclude": "false"}
    node = MagicMock()
    node.GetAttribute.side_effect = attrs.__getitem__
    node.GetNodeReferenceID.return_value = None  # …but seg ref missing

    err_field, _msg = mrml.validate_volume_node(node)
    assert err_field == "Segmentation node missing"


def test_validate_volume_node_none(mrml_module):
    """None node → 'Missing volume node'."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    err_field, _msg = mrml.validate_volume_node(None)
    assert err_field == "Missing volume node"


# ---------------------------------------------------------------------------
# Result dict with validation
# ---------------------------------------------------------------------------

def test_volume_node_to_result_dict_with_validation_healthy(mrml_module):
    """Healthy node: validation passes; dict matches plain reconstruction."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    attrs = {
        "ZebrafishAnalysis.length": "1.234",
        "ZebrafishAnalysis.curvature_class": "0",
        "ZebrafishAnalysis.ratio": "1.05",
        "ZebrafishAnalysis.eye_area": "",
        "ZebrafishAnalysis.eye_diameter": "",
        "ZebrafishAnalysis.edema_area": "",
        "ZebrafishAnalysis.exclude": "false",
        "ZebrafishAnalysis.error": "",  # explicit empty
    }
    node = MagicMock()
    node.GetName.return_value = "fish.png"
    node.GetAttribute.side_effect = attrs.__getitem__
    node.GetNodeReferenceID.return_value = "segNodeId"

    row = mrml.volume_node_to_result_dict_with_validation(node)
    assert row["error"] == ""
    assert row["exclude"] is False


def test_volume_node_to_result_dict_with_validation_never_analyzed(mrml_module):
    """Never-analyzed node: no error, not excluded — a normal pending row."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    node = MagicMock()
    node.GetName.return_value = "blank.png"
    # GetAttribute returns None for every key — nothing was ever written,
    # i.e. analysis was never run for this node.
    node.GetAttribute.return_value = None
    node.GetNodeReferenceID.return_value = "segNodeId"

    row = mrml.volume_node_to_result_dict_with_validation(node)
    assert row["error"] == ""
    assert row["exclude"] is False


def test_volume_node_to_result_dict_with_validation_broken_seg_ref(mrml_module):
    """Segmentation reference missing: error field set, exclude forced True."""
    from ZebrafishEmbryoAnalyzerLib import mrml

    attrs = {
        "ZebrafishAnalysis.length": "1.234",
        "ZebrafishAnalysis.curvature_class": "0",
        "ZebrafishAnalysis.ratio": "1.05",
        "ZebrafishAnalysis.eye_area": "",
        "ZebrafishAnalysis.eye_diameter": "",
        "ZebrafishAnalysis.edema_area": "",
        "ZebrafishAnalysis.exclude": "false",
        "ZebrafishAnalysis.error": "",
    }
    node = MagicMock()
    node.GetName.return_value = "halfway.png"
    node.GetAttribute.side_effect = attrs.__getitem__
    node.GetNodeReferenceID.return_value = None

    row = mrml.volume_node_to_result_dict_with_validation(node)
    assert row["error"] == "Segmentation node missing"
    assert row["exclude"] is True


# ---------------------------------------------------------------------------
# Logic-layer scene rebuild
# ---------------------------------------------------------------------------

class _FakeVolumeNode:
    """Minimal fake that behaves like a tracked volume node — supports
    GetAttribute/SetAttribute, GetName, GetID, GetNodeReferenceID,
    GetImageData (returns None → no pixel data, covered separately).
    """
    def __init__(self, name, attrs=None, seg_id="segNode"):
        self._name = name
        self._attrs = dict(attrs or {})
        self._seg_id = seg_id
        self._id = f"vtkMRMLVolumeNode_{name}"

    def GetAttribute(self, k): return self._attrs.get(k)
    def SetAttribute(self, k, v): self._attrs[k] = v
    def GetName(self): return self._name
    def GetID(self): return self._id
    def GetNodeReferenceID(self, role):
        from ZebrafishEmbryoAnalyzerLib import mrml
        return self._seg_id if role == mrml.ROLE_ZEBRAFISH_SEGMENTATION else None
    def GetImageData(self): return None


def _run_in_subprocess(source):
    """Run a Python source string with the package + slicer stub on sys.path."""
    import subprocess, textwrap
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_root = os.path.join(root, "ZebrafishEmbryoAnalyzer")
    full = _SLICER_STUB + textwrap.dedent(source)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": pkg_root},
    )


_SLICER_STUB = """\
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"] = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()

class _Base:
    pass

class _VTKMixin:
    def addObserver(self, *a, **kw): pass
    def removeObservers(self, *a, **kw): pass
    def removeObserver(self, *a, **kw): pass
    def hasObserver(self, *a, **kw): return False

sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=_Base,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(VTKObservationMixin=_VTKMixin)
_vtk = types.ModuleType("vtk")
_vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
sys.modules["vtk"] = _vtk
import vtk  # noqa
"""


def test_logic_rebuild_results_from_scene_walks_tracked_nodes():
    """Logic layer walks ROLE_ZEBRAFISH_IMAGES in insertion order."""
    r = _run_in_subprocess(r"""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        param = MagicMock()
        logic.getParameterNode = MagicMock(return_value=param)

        n1 = MagicMock(); n1.GetName.return_value="a.png"
        n1.GetAttribute.return_value="1.0"
        n1.GetID.return_value="id1"
        n1.GetNodeReferenceID.return_value="seg1"
        n1.GetImageData.return_value=None

        n2 = MagicMock(); n2.GetName.return_value="b.png"
        n2.GetAttribute.return_value="2.0"
        n2.GetID.return_value="id2"
        n2.GetNodeReferenceID.return_value="seg2"
        n2.GetImageData.return_value=None

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
                   return_value=[n1, n2]) as ml, \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.volume_node_to_pixels",
                   return_value=None), \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.volume_node_to_result_dict_with_validation",
                   wraps=__import__("ZebrafishEmbryoAnalyzerLib.mrml",
                                    fromlist=["*"]).volume_node_to_result_dict_with_validation):
            import slicer
            slicer.mrmlScene = MagicMock()
            results = logic.rebuild_results_from_scene()
            ml.assert_called_once_with(param, slicer.mrmlScene)
            assert [r["filename"] for r in results] == ["a.png", "b.png"]
            for r in results:
                assert "_volume_node" in r
                assert "_volume_node_id" in r
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_rebuild_results_from_scene_returns_empty_on_no_param_node():
    """No parameter node → empty list, no exception."""
    r = _run_in_subprocess(r"""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=None)

        import slicer
        slicer.mrmlScene = MagicMock()
        results = logic.rebuild_results_from_scene()
        assert results == []
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_rebuild_results_from_scene_skips_broken_refs_silently():
    """A volume-node list with wrong-type / missing entries must not crash
    the rebuild — they are filtered out by ``list_tracked_volume_nodes``.
    """
    r = _run_in_subprocess(r"""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        good = MagicMock(); good.GetName.return_value="good.png"
        good.GetAttribute.return_value="1.0"
        good.GetID.return_value="id1"
        good.GetNodeReferenceID.return_value="seg1"
        good.GetImageData.return_value=None

        # list_tracked_volume_nodes already filtered out the bad ones; only
        # ``good`` reaches the per-row conversion.
        with patch("ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
                   return_value=[good]), \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.volume_node_to_pixels",
                   return_value=None), \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.volume_node_to_result_dict_with_validation",
                   wraps=__import__("ZebrafishEmbryoAnalyzerLib.mrml",
                                    fromlist=["*"]).volume_node_to_result_dict_with_validation):
            import slicer
            slicer.mrmlScene = MagicMock()
            results = logic.rebuild_results_from_scene()
            assert len(results) == 1
            assert results[0]["filename"] == "good.png"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_rebuild_results_from_scene_pixels_attached():
    """When ``volume_node_to_pixels`` returns an array, the result dict's
    ``original`` field is populated for gallery rebuild.
    """
    r = _run_in_subprocess(r"""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        import numpy as np

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        n = MagicMock(); n.GetName.return_value="img.png"
        n.GetAttribute.return_value="1.0"
        n.GetID.return_value="id1"
        n.GetNodeReferenceID.return_value="seg1"
        n.GetImageData.return_value=None

        fake_pixels = np.zeros((8, 12, 3), dtype=np.uint8)

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
                   return_value=[n]), \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.volume_node_to_pixels",
                   return_value=fake_pixels), \
             patch("ZebrafishEmbryoAnalyzerLib.mrml.volume_node_to_result_dict_with_validation",
                   wraps=__import__("ZebrafishEmbryoAnalyzerLib.mrml",
                                    fromlist=["*"]).volume_node_to_result_dict_with_validation):
            import slicer
            slicer.mrmlScene = MagicMock()
            results = logic.rebuild_results_from_scene()
            assert len(results) == 1
            assert "original" in results[0]
            assert results[0]["original"].shape == (8, 12, 3)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_scene_end_import_invokes_rebuild_from_scene():
    """The slice's ``_on_scene_end_import`` handler must call
    ``widget.rebuild_from_scene`` so a saved scene restores the full UI.

    Uses a file-level scan rather than ``inspect.getsource`` because
    importing the module would pull in vtk/Slicer-side dependencies that
    aren't installed in the plain test environment.
    """
    src = open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzer.py",
        )
    ).read()
    # Look for the handler body that follows the EndImportEvent observer
    # registration. We check both that the handler exists and that it
    # mentions rebuild_from_scene.
    assert "_on_scene_end_import" in src
    # Find the handler definition and inspect the next ~30 lines.
    handler_idx = src.find("def _on_scene_end_import")
    assert handler_idx >= 0
    snippet = src[handler_idx:handler_idx + 800]
    assert "rebuild_from_scene" in snippet, (
        "_on_scene_end_import must call rebuild_from_scene"
    )


def test_widget_rebuild_from_scene_is_no_op_on_empty_scene():
    """An empty scene (no tracked nodes) must leave the widget empty —
    no crash, no overwriting of the just-cleared UI."""
    r = _run_in_subprocess(r"""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzerLib import widget as w

        w_mod = w
        inst = object.__new__(w.ZebrafishEmbryoAnalyzerMainWidget)
        inst._logic = MagicMock()
        inst._logic.rebuild_results_from_scene.return_value = []
        # If anything tries to touch these we want it to crash so the test
        # catches over-eager widget-side logic.
        for attr in ("_results", "_image_paths", "_excluded", "_gallery",
                     "_results_tab", "_detail", "_queue_list",
                     "_refresh_run_button", "_current_detail_idx"):
            setattr(inst, attr, MagicMock())

        inst.rebuild_from_scene()

        # No results → no assignment to self._results
        inst._results.__setitem__.assert_not_called()
        inst._gallery.populate.assert_not_called()
        inst._results_tab.populate.assert_not_called()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_widget_rebuild_from_scene_populates_ui():
    """A scene with tracked nodes must populate the widget without
    triggering an inference run.
    """
    r = _run_in_subprocess(r"""
        from unittest.mock import MagicMock, patch
        from ZebrafishEmbryoAnalyzerLib import widget as w

        # cv2 is imported lazily inside rebuild_from_scene; install a
        # mock so the thumbnail loop is a no-op (the test data has no
        # pixel arrays).
        import sys, types
        sys.modules["cv2"] = MagicMock()

        inst = object.__new__(w.ZebrafishEmbryoAnalyzerMainWidget)
        inst._logic = MagicMock()
        inst._results = []
        inst._image_paths = []
        inst._excluded = set()
        inst._current_detail_idx = -1
        inst._queue_list = MagicMock()
        inst._gallery = MagicMock()
        inst._results_tab = MagicMock()
        inst._detail = MagicMock()
        inst._refresh_run_button = MagicMock()
        inst._scale_status = MagicMock()
        inst._bar_um_edit = MagicMock()
        inst._main_widget = MagicMock()

        # Two rebuilt rows: one healthy, one with an error.
        inst._logic.rebuild_results_from_scene.return_value = [
            {"filename": "ok.png", "length": 1.0, "curvature": 0,
             "ratio": 1.0, "eye_area": None, "eye_diameter": None,
             "exclude": False, "error": "", "original": None,
             "_volume_node": MagicMock(), "_volume_node_id": "id1"},
            {"filename": "broken.png", "length": None, "curvature": "",
             "ratio": None, "eye_area": None, "eye_diameter": None,
             "exclude": True, "error": "Segmentation node missing",
             "original": None,
             "_volume_node": MagicMock(), "_volume_node_id": "id2"},
        ]

        with patch("ZebrafishEmbryoAnalyzerLib.widget.qt") as mq, \
             patch("ZebrafishEmbryoAnalyzerLib.widget.slicer", create=True) as ms:
            inst.rebuild_from_scene()

        # The widget must populate all sub-views
        inst._gallery.populate.assert_called_once()
        inst._results_tab.populate.assert_called_once()
        inst._detail.invalidate_cache.assert_called()
        inst._detail.show_result.assert_called_once()
        inst._refresh_run_button.assert_called()
        inst._logic.update_results_table_from_tracked_nodes.assert_called_once()
        # The error row must be auto-excluded
        assert "broken.png" in inst._excluded
        assert "ok.png" not in inst._excluded
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout