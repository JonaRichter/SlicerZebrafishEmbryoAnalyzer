"""Tests for issue #42: stale-segmentation flag + recompute prompt.

The module covers three layers:

* the helpers in :mod:`ZebrafishEmbryoAnalyzerLib.mrml` — ``mark_volume_node_stale``,
  ``is_volume_node_stale``, ``clear_volume_node_stale``;
* the content-digest staleness detection in :mod:`ZebrafishEmbryoAnalyzerLib.mrml`
  — ``segmentation_content_hash``, ``record_segmentation_hash``,
  ``refresh_stale_flag`` (issue #81);
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

import numpy as np


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


class _FakeSegmentation:
    """The ``vtkSegmentation`` contained in a segmentation node. Holds one
    voxel array per segment id — content, not timestamps, is what the
    staleness check reads (issue #81).
    """

    def __init__(self, segments):
        self._segments = dict(segments)

    def GetSegmentIDs(self):
        return list(self._segments)


class _FakeSegmentationNode:
    def __init__(self, segments=None):
        if segments is None:
            segments = {"Body": np.array([[0, 1], [1, 1]], dtype=np.uint8)}
        self._segmentation = _FakeSegmentation(segments)

    def GetSegmentation(self):
        return self._segmentation

    def paint(self, segment_id="Body"):
        """Simulate a brush stroke: flip one voxel."""
        arr = self._segmentation._segments[segment_id].copy()
        arr[0, 0] = 0 if arr[0, 0] else 1
        self._segmentation._segments[segment_id] = arr

    def set_segment(self, segment_id, arr):
        self._segmentation._segments[segment_id] = arr


def _stub_labelmap_reader():
    """Point ``slicer.util.arrayFromSegmentBinaryLabelmap`` at the fakes.

    Installed on whatever ``slicer`` currently sits in ``sys.modules`` —
    another test file may have swapped in its own stub, and a reader that is
    missing makes ``segmentation_content_hash`` return "" so every assertion
    here would silently pass for the wrong reason.
    """
    import slicer
    util = getattr(slicer, "util", None)
    if util is None:
        util = types.SimpleNamespace()
        try:
            setattr(slicer, "util", util)
        except (AttributeError, TypeError):
            return

    def _read(node, segment_id, *_args, **_kwargs):
        try:
            return node.GetSegmentation()._segments.get(segment_id)
        except Exception:
            return None

    try:
        setattr(util, "arrayFromSegmentBinaryLabelmap", _read)
    except (AttributeError, TypeError):
        pass


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
    clear_volume_node_stale(n)
    assert is_volume_node_stale(n) is False


def test_marking_stale_does_not_touch_exclude_or_error():
    """Issue #81: staleness is its own state.

    Borrowing ``exclude`` and ``error`` is what made a reloaded scene report
    "could not be fully restored", suppressed the overlay, and left a stale
    row indistinguishable from a broken or hand-excluded one.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import mark_volume_node_stale
    n = _make_attr_node("embryo.tif")

    mark_volume_node_stale(n)

    assert n.GetAttribute("ZebrafishAnalysis.exclude") is None
    assert n.GetAttribute("ZebrafishAnalysis.error") is None


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
# Layer 2: content-digest staleness detection (issue #81)
# --------------------------------------------------------------------------- #


class _FakeScene:
    def __init__(self, nodes_by_id):
        self._nodes = dict(nodes_by_id)

    def GetNodeByID(self, node_id):
        return self._nodes.get(node_id)


def _analysed_row(seg_node, seg_id="seg_1"):
    """A tracked volume node whose metrics were computed from ``seg_node``."""
    from ZebrafishEmbryoAnalyzerLib.mrml import record_segmentation_hash
    _stub_labelmap_reader()
    volume = _make_attr_node("embryo.tif", seg_id=seg_id)
    record_segmentation_hash(volume, seg_node)
    return volume, _FakeScene({seg_id: seg_node})


def test_content_hash_is_stable_across_repeated_reads():
    """The digest must depend only on voxels — reading twice without touching
    anything has to produce the same value, or every check would report a
    change.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import segmentation_content_hash
    _stub_labelmap_reader()
    seg_node = _FakeSegmentationNode()
    assert segmentation_content_hash(seg_node) == segmentation_content_hash(seg_node)


def test_content_hash_changes_when_voxels_change():
    from ZebrafishEmbryoAnalyzerLib.mrml import segmentation_content_hash
    _stub_labelmap_reader()
    seg_node = _FakeSegmentationNode()
    before = segmentation_content_hash(seg_node)
    seg_node.paint()
    assert segmentation_content_hash(seg_node) != before


def test_content_hash_ignores_label_value_relabelling():
    """Segments are compared as set/not-set. A conversion that renumbers
    labels without moving the mask must not read as a user edit.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import segmentation_content_hash
    _stub_labelmap_reader()
    seg_node = _FakeSegmentationNode({"Body": np.array([[0, 1], [1, 1]], dtype=np.uint8)})
    before = segmentation_content_hash(seg_node)
    seg_node.set_segment("Body", np.array([[0, 7], [7, 7]], dtype=np.uint8))
    assert segmentation_content_hash(seg_node) == before


def test_content_hash_changes_when_a_segment_is_added():
    from ZebrafishEmbryoAnalyzerLib.mrml import segmentation_content_hash
    _stub_labelmap_reader()
    seg_node = _FakeSegmentationNode()
    before = segmentation_content_hash(seg_node)
    seg_node.set_segment("Eye", np.array([[1, 0], [0, 0]], dtype=np.uint8))
    assert segmentation_content_hash(seg_node) != before


def test_content_hash_is_empty_without_a_readable_labelmap():
    """"Cannot tell" must be distinguishable from "unchanged" — callers key on
    the empty string to abstain rather than guess.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import segmentation_content_hash
    _stub_labelmap_reader()
    assert segmentation_content_hash(None) == ""
    assert segmentation_content_hash(_FakeSegmentationNode({})) == ""


def test_refresh_marks_stale_when_the_segmentation_was_edited():
    from ZebrafishEmbryoAnalyzerLib.mrml import refresh_stale_flag, is_volume_node_stale
    seg_node = _FakeSegmentationNode()
    volume, scene = _analysed_row(seg_node)
    assert refresh_stale_flag(volume, scene) is False

    seg_node.paint()

    assert refresh_stale_flag(volume, scene) is True
    assert is_volume_node_stale(volume) is True


def test_refresh_leaves_an_untouched_row_clean():
    """The regression this whole issue exists for: saving, importing and
    redisplaying a scene all move VTK timestamps without changing a voxel.
    Repeated refreshes on unchanged content must stay silent.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import refresh_stale_flag, is_volume_node_stale
    seg_node = _FakeSegmentationNode()
    volume, scene = _analysed_row(seg_node)

    for _ in range(3):
        assert refresh_stale_flag(volume, scene) is False
    assert is_volume_node_stale(volume) is False


def test_refresh_clears_a_stale_marking_once_the_content_matches_again():
    """Covers both a recompute and a scene saved with wrongly-set flags by an
    earlier version — the row must come back clean rather than stay stuck.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        refresh_stale_flag, mark_volume_node_stale, is_volume_node_stale,
        ATTR_EXCLUDE,
    )
    seg_node = _FakeSegmentationNode()
    volume, scene = _analysed_row(seg_node)
    mark_volume_node_stale(volume)
    assert is_volume_node_stale(volume) is True

    assert refresh_stale_flag(volume, scene) is False
    assert is_volume_node_stale(volume) is False


def test_refresh_repairs_a_row_marked_by_the_pre_decoupling_version():
    """Scenes saved before #81 carry ``stale`` + ``exclude=true`` + the stale
    error string. Those must come back clean, or an existing ``.mrb`` keeps
    reporting "could not be fully restored" and hiding its overlay forever.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        refresh_stale_flag, is_volume_node_stale,
        ATTR_STALE, ATTR_EXCLUDE, ATTR_PREFIX, STALE_ERROR_MESSAGE,
    )
    seg_node = _FakeSegmentationNode()
    volume, scene = _analysed_row(seg_node)
    volume.SetAttribute(ATTR_STALE, "true")
    volume.SetAttribute(ATTR_EXCLUDE, "true")
    volume.SetAttribute(ATTR_PREFIX + "error", STALE_ERROR_MESSAGE)

    assert refresh_stale_flag(volume, scene) is False

    assert is_volume_node_stale(volume) is False
    assert volume.GetAttribute(ATTR_EXCLUDE) == "false"
    assert volume.GetAttribute(ATTR_PREFIX + "error") is None


def test_refresh_preserves_a_genuine_user_exclusion():
    """A hand-excluded row carries no stale-error signature, so the clear path
    must leave it excluded.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import refresh_stale_flag, ATTR_EXCLUDE
    seg_node = _FakeSegmentationNode()
    volume, scene = _analysed_row(seg_node)
    volume.SetAttribute(ATTR_EXCLUDE, "true")

    refresh_stale_flag(volume, scene)

    assert volume.GetAttribute(ATTR_EXCLUDE) == "true"


def test_refresh_abstains_without_a_recorded_digest():
    """An unanalysed row, or one from a scene written before this attribute
    existed, must keep whatever flag it has instead of being guessed at.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import refresh_stale_flag
    _stub_labelmap_reader()
    seg_node = _FakeSegmentationNode()
    volume = _make_attr_node("embryo.tif", seg_id="seg_1")   # no hash recorded
    scene = _FakeScene({"seg_1": seg_node})

    assert refresh_stale_flag(volume, scene) is False
    assert volume.GetAttribute("ZebrafishAnalysis.stale") is None


def test_refresh_abstains_when_the_segmentation_is_gone():
    """Data module is ground truth: a deleted segmentation must not be read as
    an edit.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import refresh_stale_flag, is_volume_node_stale
    seg_node = _FakeSegmentationNode()
    volume, _scene = _analysed_row(seg_node)

    assert refresh_stale_flag(volume, _FakeScene({})) is False
    assert is_volume_node_stale(volume) is False


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


def test_elide_filename_keeps_short_names_untouched():
    from ZebrafishEmbryoAnalyzerLib.widget import elide_filename
    assert elide_filename("short.png") == "short.png"
    assert elide_filename("") == ""
    assert elide_filename(None) == ""


def test_elide_filename_drops_from_the_middle_within_the_budget():
    """Both ends carry information — the head tells two dataset images apart,
    the tail holds the extension — so the cut goes in the middle.
    """
    from ZebrafishEmbryoAnalyzerLib.widget import elide_filename
    a = "fish_000001_jpg.rf.f9e4338f9fdce1d85c4fdbe1e177ecce.jpg"
    b = "fish_000002_jpg.rf.29a95f68033583013088fc4f25d98967.jpg"

    ea, eb = elide_filename(a), elide_filename(b)

    assert len(ea) == 44 and len(eb) == 44
    assert ea.startswith("fish_000001") and eb.startswith("fish_000002")
    assert ea.endswith(".jpg") and eb.endswith(".jpg")
    assert ea != eb, "elided names must stay distinguishable"


def test_prompt_recompute_asks_once_and_recomputes_every_stale_row():
    """Issue #83: two edited images must produce one dialog, not two, and a
    yes must cover both.
    """
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    w = _build_widget_prompt_double(policy_result="yes")
    widget_mod.ZebrafishEmbryoAnalyzerMainWidget.prompt_recompute_stale_images(w)
    assert w._stale_recompute_prompt_policy.call_count == 1
    assert w._recompute_for_volume_node.call_count == 2


def test_prompt_recompute_passes_every_filename_to_the_single_prompt():
    """The one dialog has to name all affected images, so the policy receives
    the whole list rather than one name.
    """
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    w = _build_widget_prompt_double(policy_result="no")
    widget_mod.ZebrafishEmbryoAnalyzerMainWidget.prompt_recompute_stale_images(w)
    (names,), _kwargs = w._stale_recompute_prompt_policy.call_args
    assert sorted(names) == ["embryo_a.tif", "embryo_b.tif"]


def test_prompt_recompute_skips_recompute_when_no():
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod
    w = _build_widget_prompt_double(policy_result="no")
    widget_mod.ZebrafishEmbryoAnalyzerMainWidget.prompt_recompute_stale_images(w)
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


# --------------------------------------------------------------------------- #
# Manual exclusion must survive a save/reload
# --------------------------------------------------------------------------- #


def test_set_volume_node_exclude_round_trips_through_the_attribute():
    """A hand-excluded fish came back included after save/reload because the
    decision lived only in the widget's in-memory set. It has to reach the
    node, which is what the scene carries.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        set_volume_node_exclude, volume_node_to_result_dict, ATTR_EXCLUDE,
    )
    n = _make_attr_node("embryo.tif")

    assert set_volume_node_exclude(n, True) is True
    assert n.GetAttribute(ATTR_EXCLUDE) == "true"
    assert volume_node_to_result_dict(n)["exclude"] is True

    assert set_volume_node_exclude(n, False) is True
    # Written as "false", not removed — validate_volume_node's "was analysed"
    # check keys on the attribute being present.
    assert n.GetAttribute(ATTR_EXCLUDE) == "false"
    assert volume_node_to_result_dict(n)["exclude"] is False


def test_set_volume_node_exclude_handles_a_missing_node():
    from ZebrafishEmbryoAnalyzerLib.mrml import set_volume_node_exclude
    assert set_volume_node_exclude(None, True) is False


def test_exclude_change_persists_to_the_volume_node():
    """The widget handler must write the decision through, not only update its
    own set — that gap is what made the state session-local.
    """
    from ZebrafishEmbryoAnalyzerLib import widget as widget_mod

    w = MagicMock(name="widget")
    w._excluded = set()
    w._results = [{"filename": "a.tif"}, {"filename": "b.tif"}]
    w._logic = MagicMock()
    w._results_tab = MagicMock()
    w._detail = MagicMock()

    widget_mod.ZebrafishEmbryoAnalyzerMainWidget._on_exclude_change(w, "b.tif", True)

    assert w._results[1]["exclude"] is True
    assert w._results[0].get("exclude") is None, "only the named row may change"
    w._logic.set_row_exclusion.assert_called_once_with(w._results[1], True)

    widget_mod.ZebrafishEmbryoAnalyzerMainWidget._on_exclude_change(w, "b.tif", False)

    assert w._results[1]["exclude"] is False
    assert w._logic.set_row_exclusion.call_args[0][1] is False
