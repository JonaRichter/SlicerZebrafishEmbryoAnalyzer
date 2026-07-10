"""Tests for issue #42: stale-segmentation flag + recompute prompt.

The module covers three layers:

* the helpers in :mod:`ZebrafishEmbryoAnalyzerLib.mrml` — ``mark_volume_node_stale``,
  ``is_volume_node_stale``, ``clear_volume_node_stale``;
* the Logic class methods on :class:`ZebrafishEmbryoAnalyzerWidget`'s logic —
  ``setup_segmentation_staleness_observers``, ``list_stale_tracked_volume_nodes``,
  ``recompute_metrics_for_volume_node``;
* the widget-side glue — ``prompt_recompute_stale_images``,
  ``_stale_recompute_prompt_policy``, ``_on_recompute_current_detail``, and
  ``_refresh_detail_recompute_button`` — all of which rely on a clean
  per-row volume-node reference and the ``ATTR_STALE`` attribute.

The Slicer runtime is unavailable under pytest; both ``vtk`` and ``slicer``
are stubbed before importing the implementation. Each test constructs its
own light-weight fake nodes (no real ``vtkMRMLVectorVolumeNode`` objects).
"""

import os
import sys
import types
from unittest.mock import MagicMock


def _install_slicer_stub():
    """Insert a Slicer stub into ``sys.modules`` so importing the module
    under test does not require a real Slicer runtime.
    """
    if "slicer" in sys.modules and hasattr(sys.modules["slicer"], "_zea_stub"):
        return sys.modules["slicer"]

    # vtk stub — every module under test only references vtkCommand.
    vtk = types.ModuleType("vtk")
    vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=22)
    sys.modules["vtk"] = vtk
    sys.modules["vtk"]._zea_stub = True

    # Minimum slicer surface — must be a real module (not MagicMock) so
    # attribute access like ``slicer.ScriptedLoadableModule`` resolves.
    slicer = types.ModuleType("slicer")
    mrml_scene = MagicMock(name="mrmlScene")
    slicer.mrmlScene = mrml_scene

    util = types.SimpleNamespace(
        warningDisplay=MagicMock(),
        errorDisplay=MagicMock(),
        showStatusMessage=MagicMock(),
        loadVolume=MagicMock(return_value=MagicMock()),
    )
    slicer.util = util

    class _Base:
        pass

    class _VTKMixin:
        def addObserver(self, *a, **kw):
            pass
        def removeObservers(self, *a, **kw):
            pass
        def removeObserver(self, *a, **kw):
            pass
        def hasObserver(self, *a, **kw):
            return False

    slicer.ScriptedLoadableModule = types.SimpleNamespace(
        ScriptedLoadableModule=object,
        ScriptedLoadableModuleWidget=_Base,
        ScriptedLoadableModuleLogic=object,
        ScriptedLoadableModuleTest=object,
    )
    slicer._zea_stub = True
    sys.modules["slicer"] = slicer
    # `from slicer.ScriptedLoadableModule import (...)` requires an
    # actual module of that dotted name to be on sys.modules.
    sys.modules["slicer.ScriptedLoadableModule"] = slicer.ScriptedLoadableModule
    sys.modules["slicer.util"] = types.SimpleNamespace(VTKObservationMixin=_VTKMixin)

    # ctk module is referenced by ZebrafishEmbryoAnalyzerLib imports.
    sys.modules["ctk"] = MagicMock()

    # qt stub for widget imports.
    qt = types.ModuleType("qt")
    QMessageBox = MagicMock()
    QMessageBox.Question = 0
    QMessageBox.Yes = 1
    QMessageBox.No = 0
    QPushButton = MagicMock()
    QFileDialog = MagicMock()
    QSettings = MagicMock()
    QTabWidget = MagicMock()
    qt.QMessageBox = QMessageBox
    qt.QPushButton = QPushButton
    qt.QFileDialog = QFileDialog
    qt.QSettings = QSettings
    qt.QTabWidget = QTabWidget
    sys.modules["qt"] = qt

    # numpy is heavy but the module under test imports it unconditionally.
    import numpy  # noqa: F401

    return slicer


_SLICER = _install_slicer_stub()

# --------------------------------------------------------------------------- #
# Helpers / fakes
# --------------------------------------------------------------------------- #


class _FakeNodeRef:
    def __init__(self, attrs=None, refs=None):
        self._attrs = dict(attrs or {})
        self._refs = list(refs or [])

    def GetAttribute(self, key):
        return self._attrs.get(key)

    def SetAttribute(self, key, value):
        self._attrs[key] = value

    def RemoveAttribute(self, key):
        self._attrs.pop(key, None)

    def GetID(self):
        return "vtkMRMLVectorVolumeNode_" + str(id(self))

    def GetName(self):
        return self._attrs.get("ZebrafishAnalysis.filename") or "Image"

    def AddNodeReferenceID(self, role, node_id):
        self._refs.append((role, node_id))

    def GetNodeReferenceID(self, role):
        for r, nid in self._refs:
            if r == role:
                return nid
        return None

    def GetNodeReferenceIDs(self, role):
        return [nid for r, nid in self._refs if r == role]


class _FakeSegmentationNode:
    def __init__(self):
        self._modified = False
        self.observers = []

    def AddObserver(self, event, fn):
        self.observers.append((event, fn))
        return len(self.observers)

    def Modified(self):
        for _e, fn in self.observers:
            fn()


def _make_attr_node(filename, seg_id=None, **extras):
    """Build a fake volume-like node carrying only the attributes the
    module under test reads / writes.
    """
    attrs = {"ZebrafishAnalysis.filename": filename}
    attrs.update(extras)
    n = _FakeNodeRef(attrs=attrs)
    if seg_id:
        # Match the role the production code uses.
        from ZebrafishEmbryoAnalyzerLib.mrml import ROLE_ZEBRAFISH_SEGMENTATION
        n.AddNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION, seg_id)
    return n


# --------------------------------------------------------------------------- #
# Layer 1: pure helpers in mrml.py
# --------------------------------------------------------------------------- #


def test_mark_volume_node_stale_round_trip():
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        mark_volume_node_stale, is_volume_node_stale, clear_volume_node_stale,
        ATTR_STALE,
    )
    n = _make_attr_node("embryo.tif")
    assert is_volume_node_stale(n) is False
    mark_volume_node_stale(n)
    assert n.GetAttribute(ATTR_STALE) == "true"
    assert is_volume_node_stale(n) is True
    # Auto-excluded + error message set on the row so the table surfaces it.
    assert n.GetAttribute("ZebrafishAnalysis.exclude") == "true"
    assert "recompute" in (n.GetAttribute("ZebrafishAnalysis.error") or "").lower()
    clear_volume_node_stale(n)
    assert is_volume_node_stale(n) is False


def test_is_volume_node_stale_handles_missing_node():
    from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale
    assert is_volume_node_stale(None) is False


def test_clear_volume_node_stale_is_noop_when_not_stale():
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        is_volume_node_stale, clear_volume_node_stale,
    )
    n = _make_attr_node("embryo.tif")
    # Should not raise on a never-marked node.
    clear_volume_node_stale(n)
    assert is_volume_node_stale(n) is False


# --------------------------------------------------------------------------- #
# Layer 2: Logic class — observer setup + list_stale + recompute plumbing
# --------------------------------------------------------------------------- #


class _FakeSelf:
    """Stand-in for the Slicer instance slice that the Logic methods use.

    The Logic methods call ``self.getParameterNode()`` and ``self.addObserver`` /
    ``self.removeObserver`` via the Slicer VTKObservationMixin. We give the
    fake just those plus a ``_stale_observer_tags`` slot so the setup can run
    idempotently.
    """

    def __init__(self, param_node, tracked, seg_nodes_by_id):
        self._pn = param_node
        self._tracked = tracked
        self._seg_nodes_by_id = seg_nodes_by_id
        self._stale_observer_tags = []
        self.removed_tags = []
        self.added = []
        self._fn_by_tag = {}

    def getParameterNode(self):
        return self._pn

    def addObserver(self, seg_node, event, fn):
        # Match Slicer's tag-based mixin: returns an integer tag and
        # actually registers the callback on the seg_node so firing the
        # seg node's Modified() invokes our callback.
        tag = len(self.added) + 1
        self.added.append((seg_node, event, fn))
        self._fn_by_tag[tag] = fn
        # Attach callback to seg_node so Modified() fires it.
        if hasattr(seg_node, "AddObserver"):
            seg_node.AddObserver(event, fn)
        self._stale_observer_tags.append(tag)
        return tag

    def removeObserver(self, tag):
        self.removed_tags.append(tag)
        # mimic Slicer's idempotent remove
        self._stale_observer_tags = [t for t in self._stale_observer_tags if t != tag]


def _install_logic_methods():
    """Import the Widget class via the slicer.shadow so the test can call
    the real ``setup_segmentation_staleness_observers`` method against a
    stubbed ``self``. The method lives on the Widget, not the Logic —
    it needs the VTKObservationMixin that the Widget subclasses.
    """
    from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget
    return ZebrafishEmbryoAnalyzerWidget


def test_setup_observers_calls_add_observer_for_each_tracked_segment():
    import slicer
    from unittest.mock import patch
    Cls = _install_logic_methods()
    param = MagicMock(name="parameterNode")
    seg_node = _FakeSegmentationNode()
    seg_id = "seg_1"
    volume = _make_attr_node("embryo.tif", seg_id=seg_id)
    self_ = _FakeSelf(param, [volume], {seg_id: seg_node})
    # Make slicer.mrmlScene.GetNodeByID resolve our fake seg node.
    slicer.mrmlScene.GetNodeByID = MagicMock(return_value=seg_node)
    with patch(
        "ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
        return_value=[volume],
    ):
        Cls.setup_segmentation_staleness_observers(self_)
    assert len(self_.added) == 1
    added_seg, added_event, added_fn = self_.added[0]
    assert added_seg is seg_node
    assert added_event == 22  # vtkCommand.ModifiedEvent


def test_setup_observers_is_idempotent():
    from unittest.mock import patch
    Cls = _install_logic_methods()
    param = MagicMock(name="parameterNode")
    seg_node = _FakeSegmentationNode()
    seg_id = "seg_1"
    volume = _make_attr_node("embryo.tif", seg_id=seg_id)
    self_ = _FakeSelf(param, [volume], {seg_id: seg_node})
    with patch(
        "ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
        return_value=[volume],
    ):
        Cls.setup_segmentation_staleness_observers(self_)
        first_count = len(self_.added)
        assert first_count == 1
        # Second call should not stack observers.
        Cls.setup_segmentation_staleness_observers(self_)
    # The fake tracks _stale_observer_tags; the production impl calls
    # removeObserver(tag) before re-adding, so the fake records at least one
    # removal from the second pass.
    assert self_.removed_tags
    # Net observers added: first call +1, second call replaces it.
    assert len(self_.added) == first_count + 1


def test_setup_observers_skips_volume_without_segmentation_ref():
    from unittest.mock import patch
    Cls = _install_logic_methods()
    param = MagicMock(name="parameterNode")
    # No seg reference on this volume.
    volume = _make_attr_node("embryo.tif")
    self_ = _FakeSelf(param, [volume], {})
    with patch(
        "ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
        return_value=[volume],
    ):
        Cls.setup_segmentation_staleness_observers(self_)
    assert self_.added == []


def test_setup_observers_skips_when_param_node_missing():
    Cls = _install_logic_methods()
    self_ = _FakeSelf(None, [], {})
    # Should not raise.
    Cls.setup_segmentation_staleness_observers(self_)


def test_observer_callback_marks_associated_volume_stale():
    """End-to-end: the observer we register fires ``Modified`` on the seg
    node and the volume's stale attribute flips on.
    """
    import slicer
    from unittest.mock import patch
    Cls = _install_logic_methods()
    from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale
    param = MagicMock(name="parameterNode")
    seg_node = _FakeSegmentationNode()
    seg_id = "seg_1"
    volume = _make_attr_node("embryo.tif", seg_id=seg_id)
    self_ = _FakeSelf(param, [volume], {seg_id: seg_node})
    slicer.mrmlScene.GetNodeByID = MagicMock(return_value=seg_node)
    with patch(
        "ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
        return_value=[volume],
    ):
        Cls.setup_segmentation_staleness_observers(self_)
    # Fire the observer — this is what Segment Editor triggers when the
    # user paints a stroke.
    seg_node.Modified()
    assert is_volume_node_stale(volume) is True


# --------------------------------------------------------------------------- #
# Layer 3: widget-side glue
# --------------------------------------------------------------------------- #


def _build_widget_prompt_double(policy_result):
    """Return a fake widget with just enough surface for prompt tests."""
    w = MagicMock(name="widget")
    w._logic = MagicMock()
    # Build a fake volume node per stale list entry.
    stale = [
        _make_attr_node("embryo_a.tif"),
        _make_attr_node("embryo_b.tif"),
    ]
    w._logic.list_stacked_volume_nodes = lambda: stale  # not used
    w._logic.list_stale_tracked_volume_nodes = MagicMock(return_value=stale)
    w._logic.recompute_metrics_for_volume_node = MagicMock(return_value=None)
    # Stub the policy + per-volume call.
    w._stale_recompute_prompt_policy = MagicMock(return_value=policy_result)
    w._recompute_for_volume_node = MagicMock()
    w._detail = MagicMock()
    w._gallery = MagicMock()
    w._results_tab = MagicMock()
    w._results = []
    w._excluded = set()
    return w


def test_prompt_recompute_calls_recompute_per_stale_when_yes():
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    w = _build_widget_prompt_double(policy_result="yes")
    widget_mod.ZebrafishEmbryoAnalyzerMainWidget.prompt_recompute_stale_images(w)
    assert w._stale_recompute_prompt_policy.call_count == 2
    assert w._recompute_for_volume_node.call_count == 2


def test_prompt_recompute_skips_recompute_when_no():
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    w = _build_widget_prompt_double(policy_result="no")
    widget_mod.ZebrafishEmbryoAnalyzerMainWidget.prompt_recompute_stale_images(w)
    assert w._stale_recompute_prompt_policy.call_count == 2
    assert w._recompute_for_volume_node.call_count == 0


def test_prompt_recompute_stops_on_dismiss():
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    w = _build_widget_prompt_double(policy_result="dismiss")
    widget_mod.ZebrafishEmbryoAnalyzerMainWidget.prompt_recompute_stale_images(w)
    # First image prompted -> "dismiss" breaks the loop.
    assert w._stale_recompute_prompt_policy.call_count == 1
    assert w._recompute_for_volume_node.call_count == 0


def test_recompute_for_volume_node_uses_logic_output_and_refreshes_ui():
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod

    class _W:
        _logic = MagicMock()
        _detail = MagicMock()
        _gallery = MagicMock()
        _results_tab = MagicMock()
        _excluded = set()
        _current_detail_idx = 0
        _results = []
        _main_widget = MagicMock()

    new_row = {
        "filename": "embryo_a.tif",
        "_volume_node_id": "fake_id_a",
        "_volume_node": MagicMock(),
        "error": "",
    }
    vol = MagicMock()
    vol.GetID = MagicMock(return_value="fake_id_a")
    _W._logic.recompute_metrics_for_volume_node = MagicMock(return_value=new_row)
    _W._results = [
        {"filename": "embryo_a.tif", "_volume_node_id": "fake_id_a",
         "_volume_node": vol, "original": MagicMock()},
    ]

    widget_mod.ZebrafishEmbryoAnalyzerMainWidget._recompute_for_volume_node(_W, vol)

    # UI refresh path:
    _W._gallery.populate.assert_called_once()
    _W._results_tab.populate.assert_called_once()
    _W._logic.update_results_table_from_tracked_nodes.assert_called_once()


def test_on_recompute_current_detail_noop_when_not_stale():
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale

    class _W:
        _logic = MagicMock()
        _logic.is_volume_node_stale = MagicMock(return_value=False)
        _detail = MagicMock()
        _gallery = MagicMock()
        _results_tab = MagicMock()
        _excluded = set()
        _current_detail_idx = 0
        _results = [{"filename": "healthy.tif", "_volume_node": _FakeNodeRef()}]

    widget_mod.ZebrafishEmbryoAnalyzerMainWidget._on_recompute_current_detail(_W)
    # The healthy node has no stale attribute, so no recompute kicks off.
    _W._logic.recompute_metrics_for_volume_node.assert_not_called()


def test_refresh_recompute_button_enables_only_for_stale():
    from ZebrafishEmbryoAnalyzerLib.mrml import mark_volume_node_stale

    class _W:
        _current_detail_idx = 0
        _results = [{"_volume_node": _make_attr_node("ok.tif")}]
        _recompute_btn = MagicMock()
        _logic = MagicMock()
        _logic.is_volume_node_stale = MagicMock(return_value=False)

    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    MainWidget = widget_mod.ZebrafishEmbryoAnalyzerMainWidget

    # Healthy state -> button disabled.
    MainWidget._refresh_detail_recompute_button(_W)
    _W._logic.is_volume_node_stale.assert_called_with(
        _W._results[0]["_volume_node"]
    )
    _W._recompute_btn.setEnabled.assert_called_with(False)

    # Mark stale and re-check.
    mark_volume_node_stale(_W._results[0]["_volume_node"])
    _W._logic.is_volume_node_stale = MagicMock(return_value=True)
    MainWidget._refresh_detail_recompute_button(_W)
    _W._recompute_btn.setEnabled.assert_called_with(True)


# --------------------------------------------------------------------------- #
# Final assertion — the test file alone is still importable.
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
