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


# Stand-ins for the Slicer/VTK constants the staleness path needs. The real
# values are opaque enum ints; only identity between "registered" and "fired"
# matters here.
BINARY_LABELMAP_NAME = "Binary labelmap"
SEGMENT_MODIFIED = 4001
SOURCE_REPRESENTATION_MODIFIED = 4002
SEGMENT_ADDED = 4003
SEGMENT_REMOVED = 4004


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

    # Issue #81: staleness is decided on the binary-labelmap representation,
    # so the converter (for the representation's name) and the vtkSegmentation
    # event constants both have to exist outside the Slicer runtime.
    slicer.vtkSegmentationConverter = types.SimpleNamespace(
        GetSegmentationBinaryLabelmapRepresentationName=staticmethod(
            lambda: BINARY_LABELMAP_NAME
        ),
    )
    vtk_seg_core = types.ModuleType("vtkSegmentationCore")
    vtk_seg_core.vtkSegmentation = types.SimpleNamespace(
        SegmentModified=SEGMENT_MODIFIED,
        SourceRepresentationModified=SOURCE_REPRESENTATION_MODIFIED,
        SegmentAdded=SEGMENT_ADDED,
        SegmentRemoved=SEGMENT_REMOVED,
    )
    sys.modules["vtkSegmentationCore"] = vtk_seg_core
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


class _FakeLabelmap:
    """The ``vtkOrientedImageData`` holding a segment's voxels.

    Issue #81: this is the *only* object whose MTime advances when the user
    paints in the Segment Editor — measured against a live scene, the node,
    its segmentation and the segment itself all stayed put.
    """

    def __init__(self, mtime=1000):
        self._mtime = mtime

    def GetMTime(self):
        return self._mtime

    def Modified(self):
        self._mtime += 1


class _FakeSegment:
    def __init__(self, labelmap):
        self._labelmap = labelmap
        self._mtime = 500

    def GetRepresentation(self, name):
        return self._labelmap if name == BINARY_LABELMAP_NAME else None

    def GetMTime(self):
        return self._mtime


class _FakeSegmentation:
    """The ``vtkSegmentation`` contained in a segmentation node — this is what
    the production code observes, not the node itself.
    """

    def __init__(self, segments):
        self._segments = dict(segments)
        self.observers = []
        self._mtime = 700

    def GetSegmentIDs(self):
        return list(self._segments)

    def GetSegment(self, segment_id):
        return self._segments.get(segment_id)

    def GetMTime(self):
        return self._mtime

    def AddObserver(self, event, fn):
        self.observers.append((event, fn))
        return len(self.observers)

    def InvokeEvent(self, event):
        """Test helper: fire one registered event, as Slicer would."""
        for registered_event, fn in list(self.observers):
            if registered_event == event:
                fn()


class _FakeSegmentationNode:
    def __init__(self, segment_ids=("Body",)):
        self._segmentation = _FakeSegmentation(
            {sid: _FakeSegment(_FakeLabelmap()) for sid in segment_ids}
        )
        # Node MTime exists but must never drive the staleness decision.
        self._mtime = 1000

    def GetSegmentation(self):
        return self._segmentation

    def GetMTime(self):
        return self._mtime

    def Modified(self):
        """Node-level modification — what Slicer's import/display pipeline
        does. Must NOT read as a user edit.
        """
        self._mtime += 1
        self._segmentation.InvokeEvent(SOURCE_REPRESENTATION_MODIFIED)

    def paint(self, segment_id="Body"):
        """Simulate a Segment Editor brush stroke: only the labelmap moves,
        then the segmentation announces it.
        """
        self._segmentation.GetSegment(segment_id)._labelmap.Modified()
        self._segmentation.InvokeEvent(SEGMENT_MODIFIED)


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


def _ensure_scene_with_getnode(seg_node):
    """Set ``slicer.mrmlScene.GetNodeByID`` to a MagicMock that returns
    ``seg_node``. Tolerant of a SimpleNamespace stub left over by
    another test file (just installs the attribute when possible).

    Returns the (possibly newly-installed) mrmlScene so callers can
    introspect it.
    """
    import slicer
    scene = getattr(slicer, "mrmlScene", None)
    if scene is not None and hasattr(scene, "GetNodeByID"):
        try:
            scene.GetNodeByID = MagicMock(return_value=seg_node)
            return scene
        except (AttributeError, TypeError):
            pass
    # Either there was no mrmlScene at all or it was a read-only stub;
    # install a fresh MagicMock scene directly via setattr (works on
    # SimpleNamespace too — pytest's ``monkeypatch.setattr`` does not).
    fake_scene = MagicMock(name="mrmlScene")
    fake_scene.GetNodeByID = MagicMock(return_value=seg_node)
    try:
        setattr(slicer, "mrmlScene", fake_scene)
    except (AttributeError, TypeError):
        pass
    return fake_scene


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
    # Issue #56 follow-up: use the tolerant helper so the test also
    # passes when sys.modules["slicer"] was replaced with a
    # SimpleNamespace stub by an earlier test file.
    _ensure_scene_with_getnode(seg_node)
    with patch(
        "ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
        return_value=[volume],
    ):
        Cls.setup_segmentation_staleness_observers(self_)
    # Issue #81: the observer goes on the vtkSegmentation object, not on the
    # MRML node — painting never touches the node, so observing it missed
    # every real edit while firing on every scene import.
    assert self_.added, "at least one observer must be registered"
    targets = {id(target) for target, _event, _fn in self_.added}
    assert targets == {id(seg_node.GetSegmentation())}
    events = {event for _target, event, _fn in self_.added}
    assert events == {
        SEGMENT_MODIFIED, SOURCE_REPRESENTATION_MODIFIED,
        SEGMENT_ADDED, SEGMENT_REMOVED,
    }


def test_setup_observers_is_idempotent():
    from unittest.mock import patch
    Cls = _install_logic_methods()
    param = MagicMock(name="parameterNode")
    seg_node = _FakeSegmentationNode()
    seg_id = "seg_1"
    volume = _make_attr_node("embryo.tif", seg_id=seg_id)
    self_ = _FakeSelf(param, [volume], {seg_id: seg_node})
    # Issue #56 follow-up: ensure mrmlScene is installed before the
    # second call into setup_segmentation_staleness_observers (the
    # production impl calls GetNodeByID to verify the seg reference).
    _ensure_scene_with_getnode(seg_node)
    with patch(
        "ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
        return_value=[volume],
    ):
        Cls.setup_segmentation_staleness_observers(self_)
        first_count = len(self_.added)
        assert first_count == 4  # one per observed vtkSegmentation event
        # Second call should not stack observers.
        Cls.setup_segmentation_staleness_observers(self_)
    # The fake tracks _stale_observer_tags; the production impl calls
    # removeObserver(tag) before re-adding, so the fake records at least one
    # removal from the second pass.
    assert self_.removed_tags
    # Net observers added: first call +4, second call replaces them.
    assert len(self_.added) == first_count * 2


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


def _ensure_segmentation_stubs():
    """Re-install the segmentation stubs on whatever ``slicer`` is currently
    in ``sys.modules``.

    Another test file may have replaced the module with its own stub that has
    no ``vtkSegmentationConverter``; ``mrml.segment_labelmap_mtimes`` then
    reads no MTimes and silently reports "nothing changed", so these tests
    would pass alone and fail in a full run. Same tolerance rationale as
    ``_ensure_scene_with_getnode``.
    """
    import slicer
    if getattr(slicer, "vtkSegmentationConverter", None) is None:
        try:
            setattr(slicer, "vtkSegmentationConverter", types.SimpleNamespace(
                GetSegmentationBinaryLabelmapRepresentationName=staticmethod(
                    lambda: BINARY_LABELMAP_NAME
                ),
            ))
        except (AttributeError, TypeError):
            pass
    if "vtkSegmentationCore" not in sys.modules:
        mod = types.ModuleType("vtkSegmentationCore")
        mod.vtkSegmentation = types.SimpleNamespace(
            SegmentModified=SEGMENT_MODIFIED,
            SourceRepresentationModified=SOURCE_REPRESENTATION_MODIFIED,
            SegmentAdded=SEGMENT_ADDED,
            SegmentRemoved=SEGMENT_REMOVED,
        )
        sys.modules["vtkSegmentationCore"] = mod


def _armed_observer(seg_node, seg_id="seg_1"):
    """Run the real observer setup against fakes; return (self_, volume)."""
    from unittest.mock import patch
    Cls = _install_logic_methods()
    _ensure_segmentation_stubs()
    volume = _make_attr_node("embryo.tif", seg_id=seg_id)
    self_ = _FakeSelf(MagicMock(name="parameterNode"), [volume], {seg_id: seg_node})
    _ensure_scene_with_getnode(seg_node)
    with patch(
        "ZebrafishEmbryoAnalyzerLib.mrml.list_tracked_volume_nodes",
        return_value=[volume],
    ):
        Cls.setup_segmentation_staleness_observers(self_)
    return self_, volume


def test_segment_labelmap_mtimes_reads_the_representation_not_the_node():
    """The helper must report the binary labelmap's MTime per segment — the
    only value that moves when the user paints (issue #81).
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import segment_labelmap_mtimes

    _ensure_segmentation_stubs()
    seg_node = _FakeSegmentationNode(segment_ids=("Body", "Eye"))
    before = segment_labelmap_mtimes(seg_node)
    assert set(before) == {"Body", "Eye"}

    seg_node.paint("Body")
    after = segment_labelmap_mtimes(seg_node)
    assert after["Body"] > before["Body"]
    assert after["Eye"] == before["Eye"]


def test_segment_labelmap_mtimes_returns_empty_for_unusable_input():
    from ZebrafishEmbryoAnalyzerLib.mrml import segment_labelmap_mtimes
    assert segment_labelmap_mtimes(None) == {}
    assert segment_labelmap_mtimes(object()) == {}


def test_observer_marks_stale_when_the_labelmap_changes():
    """A Segment Editor brush stroke moves the labelmap MTime and nothing
    else — that must flip the row to stale.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale

    seg_node = _FakeSegmentationNode()
    _self, volume = _armed_observer(seg_node)
    assert is_volume_node_stale(volume) is False

    seg_node.paint()

    assert is_volume_node_stale(volume) is True


def test_observer_ignores_node_level_modification_without_a_labelmap_change():
    """Regression guard for the defect this issue exists for.

    Slicer's scene-import and display pipeline modifies the segmentation node
    itself without touching any labelmap. The previous implementation keyed on
    the node's MTime and therefore marked every row stale on every reload —
    which is what produced the per-fish popup storm — while missing real edits
    entirely, because painting leaves the node MTime untouched.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale

    seg_node = _FakeSegmentationNode()
    _self, volume = _armed_observer(seg_node)
    mtime_before = seg_node.GetMTime()

    seg_node.Modified()  # pipeline event: node moves, voxels do not

    assert seg_node.GetMTime() > mtime_before, "fake must model the node bump"
    assert is_volume_node_stale(volume) is False


def test_observer_does_not_remark_a_row_that_was_recomputed():
    """After a real edit the baseline moves forward, so a later pipeline
    event on the same segmentation must not resurrect the stale flag the
    recompute just cleared.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        is_volume_node_stale, clear_volume_node_stale,
    )

    seg_node = _FakeSegmentationNode()
    _self, volume = _armed_observer(seg_node)

    seg_node.paint()
    assert is_volume_node_stale(volume) is True
    clear_volume_node_stale(volume)          # what a recompute does

    seg_node.Modified()

    assert is_volume_node_stale(volume) is False


def test_observer_does_not_mark_stale_without_readable_labelmaps():
    """"Cannot tell" must never mean "changed" — a segmentation whose
    representation is unavailable must leave the row alone rather than
    marking it, which is how the reload storm started.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale

    seg_node = _FakeSegmentationNode()
    _self, volume = _armed_observer(seg_node)
    # Drop the representation, then fire an event.
    seg_node.GetSegmentation()._segments["Body"]._labelmap = None

    seg_node.GetSegmentation().InvokeEvent(SEGMENT_MODIFIED)

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
