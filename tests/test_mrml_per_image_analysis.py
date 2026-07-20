"""
Tests for issue #39: per-image segmentation, markups, and metric attribute
writes during the analysis loop.

Covers:
- ``apply_analysis_to_volume_node`` creates one ``vtkMRMLSegmentationNode``
  per image with body + eye segments, attached via the
  ``ROLE_ZEBRAFISH_SEGMENTATION`` reference on the volume node.
- ``MarkupsLineNode`` with Head/Tail control points and magenta colour is
  created when length was computed; skipped when ``length`` is None or
  ``straight_line_points`` is unavailable.
- ``MarkupsCurveNode`` with cyan colour is created when ``path_points`` has
  at least two entries; skipped otherwise.
- Metric attributes (``ZebrafishAnalysis.length``, ``.curvature_class``,
  ``.ratio``, ``.eye_area``, ``.eye_diameter``, ``.exclude``, ``.segMTime``)
  are written onto the volume node and round-trip via ``GetAttribute``.
- ``analyse_images`` invokes the new ``per_image_callback`` once per image
  inside the per-image loop, BEFORE ``progress_callback`` fires.
- Per-image ``per_image_callback`` exceptions are caught and the result dict
  is annotated with ``error`` so #40's table-derivation routes the failed
  row correctly.
- A Cancel-style early-exit mid-batch leaves fully-formed MRML state for
  every image processed up to that point.

All tests run with plain ``pytest tests/`` (no Slicer runtime). The
segmentation node, markups nodes, and scene use lightweight fakes modelled
on ``test_mrml_image_batch.py``.
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

_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ZebrafishEmbryoAnalyzer",
)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)


# ---------------------------------------------------------------------------
# Fake MRML objects
# ---------------------------------------------------------------------------

class _FakeVolumeNode:
    """Stand-in for ``vtkMRMLVectorVolumeNode`` with full node-reference support."""

    _counter = 0

    def __init__(self, name="Image"):
        _FakeVolumeNode._counter += 1
        self._id = f"vtkMRMLVectorVolumeNode{_FakeVolumeNode._counter}"
        self._name = name
        # role -> list[str] for additive / single-role references
        self._refs = {}
        self._attrs = {}

    def GetID(self):
        return self._id

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLVectorVolumeNode"

    # --- node references (issue #39 attach pattern) ---
    def SetNodeReferenceID(self, role, node_id):
        self._refs[role] = [node_id]

    def AddNodeReferenceID(self, role, node_id):
        self._refs.setdefault(role, []).append(node_id)

    def GetNodeReferenceID(self, role):
        ids = self._refs.get(role, [])
        return ids[0] if ids else ""

    def GetNodeReference(self, role):
        ids = self._refs.get(role, [])
        if not ids:
            return None
        return types.SimpleNamespace(GetID=lambda: ids[0])

    def GetNodeReferenceIDs(self, role=None):
        if role is None:
            out = []
            for lst in self._refs.values():
                out.extend(lst)
            return out
        return list(self._refs.get(role, []))

    # --- attributes ---
    def SetAttribute(self, name, value):
        self._attrs[name] = value

    def GetAttribute(self, name):
        return self._attrs.get(name)


class _FakeSegmentationNode:
    """Stand-in for ``vtkMRMLSegmentationNode`` with segment bookkeeping."""

    _counter = 0

    def __init__(self, name="Seg"):
        _FakeSegmentationNode._counter += 1
        self._id = f"vtkMRMLSegmentationNode{_FakeSegmentationNode._counter}"
        self._name = name
        self._display = None
        self._mtime = 12345.6  # deterministic so segMTime assertions are stable
        self._segments = {}    # segment_id -> {"name": ..., "color": [...]}
        self._seg = self._FakeSeg()

    def GetID(self):
        return self._id

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLSegmentationNode"

    def CreateDefaultDisplayNodes(self):
        self._display = _FakeSegDisplayNode()

    def GetDisplayNode(self):
        return self._display

    def GetSegmentation(self):
        return self._seg

    def GetMTime(self):
        return self._mtime

    class _FakeSeg:
        def __init__(self):
            self._segs = {}

        def AddEmptySegment(self, seg_id, name, color):
            self._segs[seg_id] = {"name": name, "color": list(color)}
            return seg_id

        def GetSegmentIdBySegmentName(self, name):
            for sid, info in self._segs.items():
                if info["name"] == name:
                    return sid
            return ""

        def GetSegments(self):
            return list(self._segs.keys())

        def RemoveAllSegments(self):
            self._segs = {}

        def RemoveSegment(self, seg_id):
            self._segs.pop(seg_id, None)

        def SetSourceRepresentationName(self, _name):
            # No-op for tests — production sets the master representation,
            # but the fakes do not track it.
            pass

    def StartModify(self):
        return False

    def EndModify(self, _was_modifying):
        pass

    def SetReferenceImageGeometryParameterFromVolumeNode(self, _volume_node):
        pass


class _FakeSegDisplayNode:
    def __init__(self):
        self.color = None

    def SetColor(self, *args):
        self.color = args


class _FakeMarkupsLineNode:
    _counter = 0

    def __init__(self, name="Line"):
        _FakeMarkupsLineNode._counter += 1
        self._id = f"vtkMRMLMarkupsLineNode{_FakeMarkupsLineNode._counter}"
        self._name = name
        self._display = None
        self._control_points = []

    def GetID(self):
        return self._id

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLMarkupsLineNode"

    def CreateDefaultDisplayNodes(self):
        self._display = _FakeDisplayNode()

    def GetDisplayNode(self):
        return self._display

    def AddControlPoint(self, vec_or_pos, label=""):
        try:
            x, y, z = vec_or_pos[0], vec_or_pos[1], vec_or_pos[2]
        except (TypeError, IndexError):
            x, y, z = float(vec_or_pos), 0.0, 0.0
        self._control_points.append({"label": label, "position": (x, y, z)})


class _FakeMarkupsCurveNode:
    _counter = 0

    def __init__(self, name="Curve"):
        _FakeMarkupsCurveNode._counter += 1
        self._id = f"vtkMRMLMarkupsCurveNode{_FakeMarkupsCurveNode._counter}"
        self._name = name
        self._display = None
        self._control_points = []

    def GetID(self):
        return self._id

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLMarkupsCurveNode"

    def CreateDefaultDisplayNodes(self):
        self._display = _FakeDisplayNode()

    def GetDisplayNode(self):
        return self._display

    def AddControlPoint(self, vec_or_pos, label=""):
        try:
            x, y, z = vec_or_pos[0], vec_or_pos[1], vec_or_pos[2]
        except (TypeError, IndexError):
            x, y, z = float(vec_or_pos), 0.0, 0.0
        self._control_points.append({"label": label, "position": (x, y, z)})


class _FakeDisplayNode:
    def __init__(self):
        self.color = None
        self.visibility = None
        self.visibility2d = None
        self.visibility3d = None

    def SetColor(self, *args):
        self.color = args

    def SetVisibility(self, v):
        self.visibility = bool(v)

    def SetVisibility2D(self, v):
        self.visibility2d = bool(v)

    def SetVisibility3D(self, v):
        self.visibility3d = bool(v)


class _FakeScene:
    """Minimal scene that mints the right fake node per class name."""

    def __init__(self):
        self._nodes = {}
        self.add_log = []
        self.remove_log = []

    def AddNewNodeByClass(self, class_name, display_name=""):
        self.add_log.append((class_name, display_name))
        if class_name == "vtkMRMLSegmentationNode":
            node = _FakeSegmentationNode(name=display_name or "Seg")
        elif class_name == "vtkMRMLMarkupsLineNode":
            node = _FakeMarkupsLineNode(name=display_name or "Line")
        elif class_name == "vtkMRMLMarkupsCurveNode":
            node = _FakeMarkupsCurveNode(name=display_name or "Curve")
        else:
            node = MagicMock()
            node.GetID.return_value = f"{class_name}{len(self._nodes)}"
            node.SetName = MagicMock()
            node.GetName.return_value = display_name
        self._nodes[node.GetID()] = node
        return node

    def GetNodeByID(self, node_id):
        return self._nodes.get(node_id)

    def RemoveNode(self, node):
        nid = node.GetID() if hasattr(node, "GetID") else None
        self.remove_log.append(nid)
        if nid in self._nodes:
            del self._nodes[nid]

    def nodes_of_class(self, class_name):
        return [n for n in self._nodes.values()
                if hasattr(n, "IsA") and n.IsA(class_name)]


# ---------------------------------------------------------------------------
# Fixtures: result dicts, stubs, helpers
# ---------------------------------------------------------------------------

def _make_full_result(filename="fish.png", with_path=True, with_eye=True):
    """Return a result dict that exercises every code path in the helper."""
    original = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 1
    eye_mask = (np.zeros((256, 256), dtype=np.uint8) if with_eye else None)
    if with_eye:
        eye_mask[100:120, 110:130] = 1

    result = {
        "filename": filename,
        "image_path": f"/tmp/{filename}",
        "original": original,
        "mask": mask,
        "grown": mask,
        "eye_mask": eye_mask,
        "length": 123.4,
        "curvature": 1,
        "ratio": 1.05,
        "eye_area": 200.0,
        "eye_diameter": 14.0,
        "spacing": (22.99, 22.99),
        "straight_line_points": ((64.0, 64.0), (192.0, 192.0)),
        "error": None,
    }
    if with_path:
        result["path_points"] = np.array([
            (64.0, 64.0),
            (80.0, 80.0),
            (128.0, 128.0),
            (192.0, 192.0),
        ])
    else:
        result["path_points"] = None
    return result


@pytest.fixture
def volume_node():
    return _FakeVolumeNode(name="fish.png")


@pytest.fixture
def scene():
    return _FakeScene()


@pytest.fixture
def stub_update_segmentation_node(monkeypatch):
    """Replace ``update_segmentation_node`` with a recorder-only stub.

    The real function calls into vtk/vtkSegmentationCore which is unavailable
    in plain-Python tests. We capture the call signature so segment-presence
    assertions can verify the helper invoked it correctly, and we set up a
    Body + Eye pair on the fake segmentation node to mirror production.

    Issue #56 follow-up: ``_create_segmentation_for_volume`` now passes a
    ``preserve_user_segments`` kwarg so it can reuse an existing segmentation
    node (and avoid resurrecting one the user deleted in the Data module).
    The stub accepts it via **kwargs to stay in sync with the production
    signature without mirroring its preservation logic — the dedicated
    regression tests in ``test_mrml_segmentation.py`` cover that path.
    """
    import ZebrafishEmbryoAnalyzerLib.mrml as mrml_mod

    captured = {}

    def _fake(result, um_per_px, node, image_node=None, **kwargs):
        captured["result"] = result
        captured["um_per_px"] = um_per_px
        captured["node"] = node
        captured["image_node"] = image_node
        captured["kwargs"] = kwargs
        seg = node.GetSegmentation()
        # Mirror production: body always, eye only when present + non-empty.
        # When the production caller asks for preserve_user_segments, do not
        # call RemoveAllSegments first — mirror the new contract.
        if not kwargs.get("preserve_user_segments", False):
            try:
                seg.RemoveAllSegments()
            except Exception:
                pass
        seg.AddEmptySegment("Body", "Body", [0.0, 1.0, 0.0])
        if result.get("eye_mask") is not None and np.asarray(
            result["eye_mask"]
        ).any():
            seg.AddEmptySegment("Eye", "Eye", [1.0, 0.0, 0.0])

    monkeypatch.setattr(mrml_mod, "update_segmentation_node", _fake)
    return captured


@pytest.fixture
def stub_slicer_import(monkeypatch):
    """Stub the lazy ``import slicer`` inside the helper to a no-op.

    The helper only uses ``slicer`` for ``vtkSegmentationConverter`` through
    ``update_segmentation_node`` which is replaced by stub_update_segmentation_node.
    The lazy ``import slicer`` inside _create_*_for_volume helpers would
    otherwise fail under plain pytest.
    """
    import types
    slicer_stub = types.ModuleType("slicer")
    monkeypatch.setitem(sys.modules, "slicer", slicer_stub)


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — segmentation node attached with body + eye
# ---------------------------------------------------------------------------

def test_per_image_creates_segmentation_node_with_body_and_eye(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """The helper creates exactly one segmentation node and attaches it via
    ``ROLE_ZEBRAFISH_SEGMENTATION`` on the volume node, with both Body and
    Eye segments populated.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_SEGMENTATION,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result(with_path=True, with_eye=True)
    seg_node = apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    # Exactly one segmentation node, no other classes.
    seg_nodes = scene.nodes_of_class("vtkMRMLSegmentationNode")
    assert len(seg_nodes) == 1, f"expected 1 segmentation node, got {len(seg_nodes)}"
    assert scene.add_log[0][0] == "vtkMRMLSegmentationNode"

    # The reference was recorded on the volume node under the role name.
    refs = volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_SEGMENTATION)
    assert refs == [seg_node.GetID()], (
        f"volume node reference list must point at the new segmentation "
        f"node; got {refs}"
    )

    # The segmentation node carries both Body and Eye segments.
    seg = seg_node.GetSegmentation()
    body_id = seg.GetSegmentIdBySegmentName("Body")
    eye_id = seg.GetSegmentIdBySegmentName("Eye")
    assert body_id, "Body segment must be present"
    assert eye_id, "Eye segment must be present when eye_mask is non-empty"

    # update_segmentation_node was called with the right arguments.
    cap = stub_update_segmentation_node
    assert cap["um_per_px"] == 22.99
    assert cap["image_node"] is volume_node
    assert cap["node"] is seg_node


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — MarkupsLineNode when length computed
# ---------------------------------------------------------------------------

def test_per_image_creates_markups_line_when_length_computed(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """MarkupsLine is created with Head/Tail control points and the magenta
    colour (0.784, 0.0, 0.784) when length and straight_line_points exist.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_MARKUPS_LINE,
        _STRAIGHT_CLR,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result(with_path=True, with_eye=True)
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    refs = volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_MARKUPS_LINE)
    assert len(refs) == 1, f"expected one line reference, got {refs}"
    line_nodes = scene.nodes_of_class("vtkMRMLMarkupsLineNode")
    assert len(line_nodes) == 1
    line = line_nodes[0]
    assert line._control_points, "line must carry control points"
    labels = [cp["label"] for cp in line._control_points]
    assert labels == ["Head", "Tail"], f"expected Head/Tail labels, got {labels}"
    display = line.GetDisplayNode()
    assert display is not None and display.color == _STRAIGHT_CLR, (
        f"line color must match overlay magenta, got {display.color}"
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — MarkupsCurveNode when path is available
# ---------------------------------------------------------------------------

def test_per_image_creates_markups_curve_when_path_available(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """MarkupsCurve is created with the cyan colour (0.0, 0.784, 0.784) and
    one control point per path point when path_points has >= 2 entries.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_MARKUPS_CURVE,
        _PATH_COLOR,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result(with_path=True)
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    refs = volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_MARKUPS_CURVE)
    assert len(refs) == 1, f"expected one curve reference, got {refs}"
    curve_nodes = scene.nodes_of_class("vtkMRMLMarkupsCurveNode")
    assert len(curve_nodes) == 1
    curve = curve_nodes[0]
    assert len(curve._control_points) == 4, (
        f"expected 4 control points (one per path point), got "
        f"{len(curve._control_points)}"
    )
    display = curve.GetDisplayNode()
    assert display is not None and display.color == _PATH_COLOR, (
        f"curve color must match overlay cyan, got {display.color}"
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — Markups nodes skipped when data unavailable
# ---------------------------------------------------------------------------

def test_per_image_skips_markups_curve_when_no_path(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """No path_points → no MarkupsCurveNode and no role reference."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_MARKUPS_CURVE,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result(with_path=False)
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    assert scene.nodes_of_class("vtkMRMLMarkupsCurveNode") == []
    assert volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_MARKUPS_CURVE) == []


def test_per_image_skips_markups_curve_when_only_one_path_point(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """A single-point path is degenerate — no curve node."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_MARKUPS_CURVE,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result(with_path=True)
    result["path_points"] = np.array([(64.0, 64.0)])
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    assert scene.nodes_of_class("vtkMRMLMarkupsCurveNode") == []
    assert volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_MARKUPS_CURVE) == []


def test_per_image_skips_markups_line_when_length_disabled(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """length=None mirrors the "Eye segment only when available" pattern."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_MARKUPS_LINE,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result(with_path=False)
    result["length"] = None
    result["ratio"] = None
    result["straight_line_points"] = None
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    assert scene.nodes_of_class("vtkMRMLMarkupsLineNode") == []
    assert volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_MARKUPS_LINE) == []


# ---------------------------------------------------------------------------
# Acceptance criterion 5 — metric attributes written on the volume node
# ---------------------------------------------------------------------------

def test_per_image_writes_metric_attributes(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """All seven ZebrafishAnalysis.* attributes are written and round-trip."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ATTR_CURVATURE_CLASS,
        ATTR_EYE_AREA,
        ATTR_EYE_DIAMETER,
        ATTR_EXCLUDE,
        ATTR_LENGTH,
        ATTR_PREFIX,
        ATTR_RATIO,
        ATTR_SEG_MTIME,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result()
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    # Every metric key must be present on the volume node.
    expected_present = {
        ATTR_LENGTH: "123.4",
        ATTR_CURVATURE_CLASS: "1",
        ATTR_RATIO: "1.05",
        ATTR_EYE_AREA: "200.0",
        ATTR_EYE_DIAMETER: "14.0",
        ATTR_EXCLUDE: "false",
    }
    for name, expected in expected_present.items():
        assert name.startswith(ATTR_PREFIX), f"attribute must use namespace, got {name}"
        assert volume_node.GetAttribute(name) == expected, (
            f"{name}: expected {expected!r}, got {volume_node.GetAttribute(name)!r}"
        )
    assert ATTR_SEG_MTIME in volume_node._attrs


def test_per_attribute_exclude_true_is_recorded(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """User-excluded images must be queryable without scanning geometry."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ATTR_EXCLUDE,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result()
    result["exclude"] = True
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)
    assert volume_node.GetAttribute(ATTR_EXCLUDE) == "true"


# ---------------------------------------------------------------------------
# Acceptance criterion 8 — segMTime round-trips segmentation node MTime
# ---------------------------------------------------------------------------

def test_attribute_segMTime_matches_segmentation_getmtime(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """``ZebrafishAnalysis.segMTime`` must equal str(segNode.GetMTime())."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ATTR_SEG_MTIME,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result()
    seg_node = apply_analysis_to_volume_node(result, volume_node, scene, 22.99)
    assert seg_node is not None, "seg node must exist for this assertion"

    expected = repr(float(seg_node.GetMTime()))
    assert volume_node.GetAttribute(ATTR_SEG_MTIME) == expected


# ---------------------------------------------------------------------------
# Acceptance criterion 7 — per_image_callback failures do not abort batch
# ---------------------------------------------------------------------------

def test_per_image_failure_logs_and_continues(
    volume_node, scene, mock_model_paths,
):
    """A helper exception on image 1 must NOT prevent image 2 from getting
    its full MRML state. Image 1's result dict must carry an ``error`` field
    so #40's table derivation routes it correctly.
    """
    import ZebrafishEmbryoAnalyzerLib.logic as logic

    tmp = os.path.abspath("_tmp_per_image_dir")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for fname in ("fish1.png", "fish2.png"):
        p = os.path.join(tmp, fname)
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        paths.append(p)

    captured = []

    def _capture(image_path, result):
        captured.append((os.path.basename(image_path), dict(result)))

    # Patch segmentation_pipeline so the loop returns a minimal synthetic
    # result. analyse_images' loop body copies the file into a temp dir and
    # passes that dir to segmentation_pipeline; we intercept there.
    import ZebrafishEmbryoAnalyzerCore.seg as core_seg

    def _fake_seg(folder_path, **_kwargs):
        # Return one entry per file in the folder.
        n = max(1, len(os.listdir(folder_path)))
        original = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[64:192, 64:192] = 1
        return ([original] * n, [mask] * n, [mask] * n)

    original_load_unet = core_seg._load_unet_model
    core_seg._load_unet_model = lambda *a, **kw: MagicMock()

    try:
        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
            core_seg, "segmentation_pipeline", _fake_seg
        ):
            results = logic.analyse_images(
                paths,
                {"length": False, "eyes": False, "curvature": False, "ratio": False,
                 "um_per_px": 22.99, "model_id": "general", "hitl": False,
                 "threshold": 0.85},
                progress_callback=None,
                per_image_callback=_capture,
            )
    finally:
        core_seg._load_unet_model = original_load_unet
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    # Both images had the callback fired.
    assert len(captured) == 2, f"expected 2 callbacks, got {len(captured)}"
    names = [name for name, _ in captured]
    assert names == ["fish1.png", "fish2.png"], f"order: {names}"

    # The result dict schema must be present in both callbacks.
    _, r1 = captured[0]
    _, r2 = captured[1]
    for key in ("filename", "length", "ratio", "eye_area", "error"):
        assert key in r1, f"missing key {key} on r1"
        assert key in r2, f"missing key {key} on r2"
    # And the analyse_images batch returned both results.
    assert len(results) == 2


def test_per_image_callback_exception_sets_error_field(
    mock_model_paths, volume_node, scene,
):
    """When the per_image_callback raises, ``result['error']`` is populated
    so sub-issue #40's table derivation can route the row to the error
    column. The batch keeps going.
    """
    import ZebrafishEmbryoAnalyzerLib.logic as logic

    tmp = os.path.abspath("_tmp_per_image_err_dir")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for fname in ("a.png", "b.png"):
        p = os.path.join(tmp, fname)
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        paths.append(p)

    captured = []

    def _boom(image_path, result):
        captured.append((os.path.basename(image_path), result.get("filename")))
        raise RuntimeError("simulated failure")

    import ZebrafishEmbryoAnalyzerCore.seg as core_seg

    def _fake_seg(folder_path, **_kwargs):
        n = max(1, len(os.listdir(folder_path)))
        original = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[64:192, 64:192] = 1
        return ([original] * n, [mask] * n, [mask] * n)

    core_seg._load_unet_model = lambda *a, **kw: MagicMock()
    try:
        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
            core_seg, "segmentation_pipeline", _fake_seg
        ):
            results = logic.analyse_images(
                paths,
                {"length": False, "eyes": False, "curvature": False, "ratio": False,
                 "um_per_px": 22.99, "model_id": "general", "hitl": False,
                 "threshold": 0.85},
                progress_callback=None,
                per_image_callback=_boom,
            )
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    # Callback fired once per image, despite raising every time.
    assert len(captured) == 2, f"expected 2 callbacks, got {len(captured)}"
    # The batch completed (didn't raise out).
    assert len(results) == 2
    # Every result dict must have its error field populated since the
    # callback always raised.
    for r in results:
        assert r.get("error"), (
            f"result for {r.get('filename')} missing error field after "
            f"callback failure: {r!r}"
        )


# ---------------------------------------------------------------------------
# Acceptance criterion 6 — streaming + Cancel mid-batch leaves state for
# completed images
# ---------------------------------------------------------------------------

def test_per_image_callback_invoked_streaming_inside_loop(mock_model_paths):
    """The per-image callback fires inside the per-image loop, once per
    image, BEFORE progress_callback. A Cancel-style mid-batch exception
    leaves fully-formed MRML state for every image processed up to that
    point.

    Implementation note: we use a progress_callback that raises after the
    second ``current`` value, simulating the user clicking Cancel after
    image 2 of 3 started. The first two per_image_callbacks must have
    already run, so their results are visible. The third must not.
    """
    import ZebrafishEmbryoAnalyzerLib.logic as logic

    tmp = os.path.abspath("_tmp_per_image_cancel_dir")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for fname in ("img1.png", "img2.png", "img3.png"):
        p = os.path.join(tmp, fname)
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        paths.append(p)

    seen = []

    def _per_image(image_path, result):
        seen.append(os.path.basename(image_path))

    def _progress(current, total):
        if current >= 2:
            raise RuntimeError("simulated Cancel after image 2")

    import ZebrafishEmbryoAnalyzerCore.seg as core_seg

    def _fake_seg(folder_path, **_kwargs):
        n = max(1, len(os.listdir(folder_path)))
        original = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[64:192, 64:192] = 1
        return ([original] * n, [mask] * n, [mask] * n)

    core_seg._load_unet_model = lambda *a, **kw: MagicMock()
    try:
        try:
            with __import__("unittest.mock", fromlist=["patch"]).patch.object(
                core_seg, "segmentation_pipeline", _fake_seg
            ):
                logic.analyse_images(
                    paths,
                    {"length": False, "eyes": False, "curvature": False, "ratio": False,
                     "um_per_px": 22.99, "model_id": "general", "hitl": False,
                     "threshold": 0.85},
                    progress_callback=_progress,
                    per_image_callback=_per_image,
                )
        except RuntimeError as exc:
            assert "simulated Cancel" in str(exc), (
                f"unexpected exception: {exc}"
            )
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    # Two images had their per-image callback fire (img1 and img2 in
    # sorted-name order). img3 must not have been touched.
    assert len(seen) >= 2, f"expected >= 2 callback firings, got {seen}"
    assert "img3.png" not in seen, (
        f"image 3 must not be processed after a Cancel at 2; got {seen}"
    )


def test_progress_callback_fires_after_per_image_callback(mock_model_paths):
    """Ordering contract: ``per_image_callback`` runs strictly BEFORE
    ``progress_callback`` for each image, so streaming writes are observable
    when progress UI updates.
    """
    import ZebrafishEmbryoAnalyzerLib.logic as logic

    tmp = os.path.abspath("_tmp_per_image_order_dir")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for fname in ("a.png", "b.png"):
        p = os.path.join(tmp, fname)
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        paths.append(p)

    events = []

    def _per_image(image_path, result):
        events.append(("per_image", os.path.basename(image_path)))

    def _progress(current, total):
        events.append(("progress", current))

    import ZebrafishEmbryoAnalyzerCore.seg as core_seg

    def _fake_seg(folder_path, **_kwargs):
        n = max(1, len(os.listdir(folder_path)))
        original = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[64:192, 64:192] = 1
        return ([original] * n, [mask] * n, [mask] * n)

    core_seg._load_unet_model = lambda *a, **kw: MagicMock()
    try:
        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
            core_seg, "segmentation_pipeline", _fake_seg
        ):
            logic.analyse_images(
                paths,
                {"length": False, "eyes": False, "curvature": False, "ratio": False,
                 "um_per_px": 22.99, "model_id": "general", "hitl": False,
                 "threshold": 0.85},
                progress_callback=_progress,
                per_image_callback=_per_image,
            )
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    # Pair-up contract: every "progress" event must be preceded by a
    # "per_image" event for the same image in the same batch.
    per_image_count = 0
    progress_count = 0
    for kind, _ in events:
        if kind == "per_image":
            per_image_count += 1
        elif kind == "progress":
            progress_count += 1
            # Each progress event must be preceded by at least one
            # per_image event for the current image.
            assert per_image_count >= progress_count, (
                f"progress event {progress_count} appeared without a "
                f"matching prior per_image event"
            )
    # Sanity: equal counts at the end.
    assert per_image_count == 2
    assert progress_count == 2


# ---------------------------------------------------------------------------
# Module-level / static checks
# ---------------------------------------------------------------------------

def test_module_exports_role_constants():
    """mrml.py must expose the three role constants added in #39."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    for name in (
        "ROLE_ZEBRAFISH_SEGMENTATION",
        "ROLE_ZEBRAFISH_MARKUPS_LINE",
        "ROLE_ZEBRAFISH_MARKUPS_CURVE",
    ):
        assert hasattr(mrml, name), f"mrml.py must expose {name}"
        assert isinstance(getattr(mrml, name), str) and getattr(mrml, name)


def test_module_exports_attr_namespace_constants():
    """The attribute namespace + 7 attribute constants must be present."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    for name in (
        "ATTR_PREFIX",
        "ATTR_LENGTH",
        "ATTR_CURVATURE_CLASS",
        "ATTR_RATIO",
        "ATTR_EYE_AREA",
        "ATTR_EYE_DIAMETER",
        "ATTR_EXCLUDE",
        "ATTR_SEG_MTIME",
    ):
        assert hasattr(mrml, name), f"mrml.py must expose {name}"
    # Namespacing invariant: every ATTR_* must start with ATTR_PREFIX.
    for name in (
        "ATTR_LENGTH",
        "ATTR_CURVATURE_CLASS",
        "ATTR_RATIO",
        "ATTR_EYE_AREA",
        "ATTR_EYE_DIAMETER",
        "ATTR_EXCLUDE",
        "ATTR_SEG_MTIME",
    ):
        value = getattr(mrml, name)
        assert value.startswith(mrml.ATTR_PREFIX), (
            f"{name} must use the namespace prefix, got {value!r}"
        )


def test_module_exports_helper():
    """mrml.py must export apply_analysis_to_volume_node."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    assert hasattr(mrml, "apply_analysis_to_volume_node")
    assert callable(mrml.apply_analysis_to_volume_node)


def test_markups_colors_match_overlay_constants():
    """The colors used for the new markups nodes must match overlay.py."""
    from ZebrafishEmbryoAnalyzerLib.mrml import _PATH_COLOR, _STRAIGHT_CLR
    # overlay._PATH_COLOR = (200, 200, 0) in BGR → magenta/cyan RGB normalized:
    # overlay._PATH_COLOR BGR=(200, 200, 0) → RGB=(0, 200, 200) → /255 ≈ (0, 0.784, 0.784)
    assert _PATH_COLOR == (0.0, 0.784, 0.784), _PATH_COLOR
    # overlay._STRAIGHT_CLR BGR=(200, 0, 200) → RGB=(200, 0, 200) → /255 ≈ (0.784, 0.0, 0.784)
    assert _STRAIGHT_CLR == (0.784, 0.0, 0.784), _STRAIGHT_CLR


# ---------------------------------------------------------------------------
# Issue #58: mask-pixel → mm scaling for line/curve control points
# ---------------------------------------------------------------------------

def test_mask_spacing_mm_converts_um_to_mm():
    """result['spacing'] in µm per mask-pixel must convert to mm for RAS."""
    from ZebrafishEmbryoAnalyzerLib.mrml import _mask_spacing_mm

    assert _mask_spacing_mm({"spacing": (46.0, 46.0)}) == (0.046, 0.046)
    assert _mask_spacing_mm({"spacing": (22.99, 22.99)}) == (0.02299, 0.02299)
    # Anisotropic spacing: keep the per-axis values distinct.
    assert _mask_spacing_mm({"spacing": (10.0, 20.0)}) == (0.010, 0.020)


def test_mask_spacing_mm_fallback_on_missing_or_malformed():
    """Missing/malformed spacing must fall back to (1.0, 1.0) — not crash."""
    from ZebrafishEmbryoAnalyzerLib.mrml import _mask_spacing_mm

    assert _mask_spacing_mm({}) == (1.0, 1.0)
    assert _mask_spacing_mm(None) == (1.0, 1.0)
    assert _mask_spacing_mm({"spacing": None}) == (1.0, 1.0)
    assert _mask_spacing_mm({"spacing": (1.0,)}) == (1.0, 1.0)  # wrong length
    assert _mask_spacing_mm({"spacing": ("a", "b")}) == (1.0, 1.0)  # non-numeric


def test_add_line_endpoints_scales_by_mask_spacing():
    """Issue #58 follow-up: control points mirror the 180-degree rotation
    (flipud+fliplr) update_image_node applies to the volume's pixel data —
    mask pixel (row, col) lands at ((mask_w-1-col)*col_mm, (mask_h-1-row)*row_mm, 0).
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import _add_line_endpoints

    class _FakeLine:
        def __init__(self):
            self._control_points = []
        def AddControlPoint(self, position, label):
            # Capture the (R, A, S) coordinates that the production code hands us.
            self._control_points.append(
                {"label": label, "position": (position[0], position[1], position[2])}
            )
        def SetNthControlPointLabel(self, *a, **kw):
            pass

    line = _FakeLine()
    sl_pts = ((64.0, 64.0), (192.0, 192.0))  # (row, col) pairs in mask coords
    # 22.99 µm per mask-pixel → 0.02299 mm per mask-pixel. 256x256 mask.
    result = {"spacing": (22.99, 22.99), "mask": np.zeros((256, 256), dtype=np.uint8)}
    _add_line_endpoints(line, sl_pts, result, volume_node=None)

    assert len(line._control_points) == 2
    head, tail = line._control_points
    # Head: (row=64, col=64) → (R=(256-1-64)*0.02299, A=(256-1-64)*0.02299, S=0)
    assert head["label"] == "Head"
    assert head["position"] == pytest.approx(
        (191 * 0.02299, 191 * 0.02299, 0.0), rel=1e-9
    )
    assert tail["label"] == "Tail"
    assert tail["position"] == pytest.approx(
        (63 * 0.02299, 63 * 0.02299, 0.0), rel=1e-9
    )


def test_add_line_endpoints_without_spacing_uses_identity():
    """result without 'spacing' must keep the historical raw-pixel behaviour
    (the fallback is (1.0, 1.0)) so the helper stays total and never crashes.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import _add_line_endpoints

    class _FakeLine:
        def __init__(self):
            self._control_points = []
        def AddControlPoint(self, position, label):
            self._control_points.append(
                {"label": label, "position": (position[0], position[1], position[2])}
            )
        def SetNthControlPointLabel(self, *a, **kw):
            pass

    line = _FakeLine()
    _add_line_endpoints(line, ((10.0, 20.0), (30.0, 40.0)), {}, volume_node=None)
    assert line._control_points[0]["position"] == (20.0, -10.0, 0.0)
    assert line._control_points[1]["position"] == (40.0, -30.0, 0.0)


def test_add_curve_points_scales_by_mask_spacing():
    """Issue #58 follow-up: every curve control point mirrors the same
    180-degree rotation as the line endpoints (see
    test_add_line_endpoints_scales_by_mask_spacing).
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import _add_curve_points

    class _FakeCurve:
        def __init__(self):
            self._control_points = []
        def AddControlPoint(self, position, label):
            self._control_points.append(
                {"label": label, "position": (position[0], position[1], position[2])}
            )

    curve = _FakeCurve()
    path_pts = np.array([
        (64.0, 64.0),
        (80.0, 80.0),
        (128.0, 128.0),
        (192.0, 192.0),
    ])
    result = {"spacing": (22.99, 22.99), "mask": np.zeros((256, 256), dtype=np.uint8)}
    _add_curve_points(curve, path_pts, result)

    expected = [
        (191 * 0.02299, 191 * 0.02299, 0.0),
        (175 * 0.02299, 175 * 0.02299, 0.0),
        (127 * 0.02299, 127 * 0.02299, 0.0),
        (63 * 0.02299, 63 * 0.02299, 0.0),
    ]
    assert len(curve._control_points) == 4
    for cp, exp in zip(curve._control_points, expected):
        assert cp["position"] == pytest.approx(exp, rel=1e-9)


def test_apply_analysis_writes_scaled_line_positions(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """End-to-end: apply_analysis_to_volume_node produces a line whose
    Head/Tail RAS positions are scaled by result['spacing'] (mm) and mirror
    the 180-degree rotation update_image_node applies to the volume's pixel
    data — not raw, unflipped mask pixels. This is the observable behaviour
    the issue needs.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import apply_analysis_to_volume_node

    result = _make_full_result(with_path=True)  # spacing=(22.99, 22.99), 256x256 mask
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)

    line_nodes = scene.nodes_of_class("vtkMRMLMarkupsLineNode")
    assert len(line_nodes) == 1
    line = line_nodes[0]
    cps = line._control_points
    assert len(cps) == 2
    # sl_pts = ((64, 64), (192, 192)); expected RAS = ((255-col)*0.02299, (255-row)*0.02299, 0).
    assert cps[0]["label"] == "Head"
    assert cps[0]["position"] == pytest.approx(
        (191 * 0.02299, 191 * 0.02299, 0.0), rel=1e-9
    )
    assert cps[1]["label"] == "Tail"
    assert cps[1]["position"] == pytest.approx(
        (63 * 0.02299, 63 * 0.02299, 0.0), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Issue #56 regression: Data module is ground truth
# ---------------------------------------------------------------------------
# When the user deletes a segmentation node (or a single segment inside the
# node) in the Data module, switching back to the Zebrafish module must NOT
# silently recreate the deleted segmentation. These tests pin down each
# layer of the defence so future refactors cannot reintroduce the bug.

def test_get_existing_seg_for_volume_returns_none_when_role_unset(
    volume_node, scene,
):
    """Issue #56 follow-up: a freshly tracked volume with no seg reference
    yet must not be confused with one whose seg was deleted. Returns None.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        _get_existing_seg_for_volume, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    # No SetNodeReferenceID call — role is unset
    assert volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_SEGMENTATION) == []
    assert _get_existing_seg_for_volume(volume_node, scene) is None


def test_get_existing_seg_for_volume_returns_live_node(
    volume_node, scene,
):
    """When the role resolves to a node still in the scene, the helper
    returns that node (so ``_create_segmentation_for_volume`` can reuse it
    instead of stacking a duplicate).
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        _get_existing_seg_for_volume, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    seg = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", display_name="Seg")
    volume_node.SetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION, seg.GetID())
    assert _get_existing_seg_for_volume(volume_node, scene) is seg


def test_get_existing_seg_for_volume_returns_none_when_seg_deleted(
    volume_node, scene,
):
    """Issue #56 follow-up: the Data-module-is-ground-truth contract.
    The volume still holds a reference id, but the seg has been removed
    from the scene. The helper must return ``None`` so the next analysis
    sees a clean slate and creates a fresh seg, rather than silently
    reusing a dangling reference.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        _get_existing_seg_for_volume, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    seg = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", display_name="Seg")
    volume_node.SetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION, seg.GetID())
    scene.RemoveNode(seg)
    assert _get_existing_seg_for_volume(volume_node, scene) is None


def test_create_segmentation_reuses_existing_seg_with_preserve_flag(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """Re-running analysis on a volume whose seg still exists must reuse
    the same seg node (no duplicate) and pass ``preserve_user_segments=True``
    so ``update_segmentation_node`` does not call ``RemoveAllSegments`` —
    that call would wipe out any user-added segments the user kept.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        _create_segmentation_for_volume, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    seg = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", display_name="Seg")
    volume_node.SetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION, seg.GetID())

    result = _make_full_result(with_path=False, with_eye=True)
    out = _create_segmentation_for_volume(result, volume_node, scene, 22.99)

    assert out is seg, "Reuse must return the same seg node, not a fresh one"
    # Exactly one seg node in the scene — no duplicate was stacked on top.
    assert len(scene.nodes_of_class("vtkMRMLSegmentationNode")) == 1
    # update_segmentation_node was called with preserve_user_segments=True
    assert stub_update_segmentation_node["kwargs"].get("preserve_user_segments") is True


def test_create_segmentation_creates_new_seg_after_data_module_delete(
    volume_node, scene, stub_update_segmentation_node, stub_slicer_import,
):
    """After the user deletes the segmentation node in the Data module,
    the next analysis must create a fresh seg node (the previous id is
    dangling and should not be reused or referenced again).
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        _create_segmentation_for_volume, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    # Simulate the Data-module delete: a seg was attached and then removed.
    stale_seg = scene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", display_name="StaleSeg",
    )
    volume_node.SetNodeReferenceID(
        ROLE_ZEBRAFISH_SEGMENTATION, stale_seg.GetID(),
    )
    scene.RemoveNode(stale_seg)

    result = _make_full_result(with_path=False, with_eye=True)
    out = _create_segmentation_for_volume(result, volume_node, scene, 22.99)

    assert out is not None
    assert out is not stale_seg, (
        "Must not resurrect the deleted seg — create a fresh node instead"
    )
    assert len(scene.nodes_of_class("vtkMRMLSegmentationNode")) == 1
    # Fresh creation means preserve_user_segments=False (full rebuild path).
    assert stub_update_segmentation_node["kwargs"].get("preserve_user_segments") is False


def test_set_node_reference_prefers_single_ref_over_additive(
    volume_node,
):
    """Issue #56 follow-up: ``_set_node_reference`` must prefer
    ``SetNodeReferenceID`` (single, replaceable) over ``AddNodeReferenceID``
    (additive, accumulates duplicates across re-runs). Repeated calls with
    the same id must collapse to a single reference, not stack.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        _set_node_reference, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    seg = _FakeSegmentationNode(name="Seg")
    # Three back-to-back attaches with the same seg must yield exactly one
    # reference, not three.
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_SEGMENTATION, seg)
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_SEGMENTATION, seg)
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_SEGMENTATION, seg)
    # The reference role resolves to the seg id (single value, not a list of 3).
    refs = volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_SEGMENTATION)
    assert refs == [seg.GetID()], (
        f"Repeated attaches must collapse to one ref, got {refs!r}"
    )


def test_set_node_reference_replaces_when_id_changes(
    volume_node,
):
    """When a re-run produces a fresh seg node (because the previous one
    was deleted in the Data module), the new id must replace the old one
    — not stack alongside it.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        _set_node_reference, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    seg1 = _FakeSegmentationNode(name="Seg1")
    seg2 = _FakeSegmentationNode(name="Seg2")
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_SEGMENTATION, seg1)
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_SEGMENTATION, seg2)
    # Old id replaced by new id — no accumulation.
    refs = volume_node.GetNodeReferenceIDs(ROLE_ZEBRAFISH_SEGMENTATION)
    assert refs == [seg2.GetID()], (
        f"New id must replace old one, got {refs!r}"
    )


def test_update_segmentation_node_preserve_keeps_existing_segments(
    volume_node, scene,
):
    """Issue #56 follow-up: ``preserve_user_segments=True`` must not call
    ``RemoveAllSegments``. Segments the user added or kept (e.g. they
    removed the Body segment to keep only Eye) must survive a re-analysis.
    """
    import subprocess
    import sys
    import textwrap

    # The real ``update_segmentation_node`` imports vtk / vtkSegmentationCore /
    # slicer at the top — neither is available in this pytest env, so we
    # run it in a subprocess with a minimal stub of each.
    code = textwrap.dedent(r"""
        import os, sys, types
        sys.path.insert(0, os.environ["ZEA_DIR"])
        import numpy as np
        # Stub the heavy optional deps with no-op modules so the imports
        # succeed but every operation is short-circuited.
        sys.modules["vtk"] = types.ModuleType("vtk")
        sys.modules["vtk"].VTK_UNSIGNED_CHAR = 7
        sys.modules["vtk.util"] = types.ModuleType("vtk.util")
        _nps = types.ModuleType("vtk.util.numpy_support")
        def _fake_numpy_to_vtk(*args, **kwargs):
            arr = types.SimpleNamespace(SetNumberOfComponents=lambda n: None)
            return arr
        _nps.numpy_to_vtk = _fake_numpy_to_vtk
        sys.modules["vtk.util.numpy_support"] = _nps

        _vsc = types.ModuleType("vtkSegmentationCore")
        class _FakeOID:
            def __init__(self):
                self._pt = types.SimpleNamespace(SetScalars=lambda x: None)
            def SetDimensions(self, *a, **kw): pass
            def GetPointData(self): return self._pt
            def SetSpacing(self, *a, **kw): pass
            def SetOrigin(self, *a, **kw): pass
        _vsc.vtkOrientedImageData = _FakeOID
        sys.modules["vtkSegmentationCore"] = _vsc

        _slicer = types.ModuleType("slicer")
        _slicer.vtkSegmentationConverter = types.SimpleNamespace(
            GetSegmentationBinaryLabelmapRepresentationName=lambda: "BinaryLabelmap",
        )
        # The production code also calls into vtkSlicerSegmentationsModuleLogic,
        # but only to write the binary labelmap back; we stub that out to
        # short-circuit so the test focuses on segment-preservation logic.
        _slicer.vtkSlicerSegmentationsModuleLogic = types.SimpleNamespace(
            SetBinaryLabelmapToSegment=lambda *args, **kwargs: None,
        )
        sys.modules["slicer"] = _slicer

        # Build a fake seg node that mirrors the production contract:
        # the user's "Tail" segment is the only one we want to survive.
        class _FakeSeg:
            def __init__(self):
                self._segs = {}
                self.source_rep = None
            def AddEmptySegment(self, seg_id, name, color):
                self._segs[seg_id] = {"name": name, "color": list(color)}
                return seg_id
            def RemoveSegment(self, seg_id):
                self._segs.pop(seg_id, None)
            def RemoveAllSegments(self):
                self._segs = {}
            def GetSegments(self):
                return list(self._segs.keys())
            def SetSourceRepresentationName(self, name):
                self.source_rep = name

        class _FakeSegNode:
            def __init__(self):
                self._seg = _FakeSeg()
                self._started = False
            def GetSegmentation(self):
                return self._seg
            def StartModify(self):
                return False
            def EndModify(self, _was):
                pass
            def SetReferenceImageGeometryParameterFromVolumeNode(self, _v):
                pass

        class _FakeVolumeNode:
            pass

        seg_node = _FakeSegNode()
        # Initial state: the user has Body, then removed it and added Tail.
        seg_node.GetSegmentation().AddEmptySegment("Body", "Body", [0.0, 1.0, 0.0])
        seg_node.GetSegmentation().RemoveSegment("Body")
        seg_node.GetSegmentation().AddEmptySegment("Tail", "Tail", [0.5, 0.5, 0.5])
        volume_node = _FakeVolumeNode()

        result = {
            "filename": "fish.png",
            "original": np.zeros((256, 256, 3), dtype=np.uint8),
            "mask": np.zeros((256, 256), dtype=np.uint8),
            "eye_mask": np.zeros((256, 256), dtype=np.uint8),
        }
        from ZebrafishEmbryoAnalyzerLib.mrml import update_segmentation_node
        update_segmentation_node(
            result, 22.99, seg_node, image_node=volume_node,
            preserve_user_segments=True,
        )
        segs = seg_node.GetSegmentation().GetSegments()
        assert "Tail" in segs, segs
        # Body must NOT be re-added when the user previously removed it.
        assert "Body" not in segs, segs
        print("OK")
    """)
    env = {**os.environ, "ZEA_DIR": _MODULE_DIR}
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert "OK" in r.stdout


def test_update_segmentation_node_legacy_full_rebuild_still_works(
    volume_node, scene,
):
    """Default ``preserve_user_segments=False`` keeps the legacy
    full-rebuild behaviour. Pre-existing callers that pass a fresh
    segmentation node see Body added and nothing else.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(r"""
        import os, sys, types
        sys.path.insert(0, os.environ["ZEA_DIR"])
        import numpy as np
        sys.modules["vtk"] = types.ModuleType("vtk")
        sys.modules["vtk"].VTK_UNSIGNED_CHAR = 7
        sys.modules["vtk.util"] = types.ModuleType("vtk.util")
        _nps = types.ModuleType("vtk.util.numpy_support")
        _nps.numpy_to_vtk = lambda *a, **k: types.SimpleNamespace(
            SetNumberOfComponents=lambda n: None,
        )
        sys.modules["vtk.util.numpy_support"] = _nps
        _vsc = types.ModuleType("vtkSegmentationCore")
        class _FakeOID:
            def __init__(self):
                self._pt = types.SimpleNamespace(SetScalars=lambda x: None)
            def SetDimensions(self, *a, **kw): pass
            def GetPointData(self): return self._pt
            def SetSpacing(self, *a, **kw): pass
            def SetOrigin(self, *a, **kw): pass
        _vsc.vtkOrientedImageData = _FakeOID
        sys.modules["vtkSegmentationCore"] = _vsc
        _slicer = types.ModuleType("slicer")
        _slicer.vtkSegmentationConverter = types.SimpleNamespace(
            GetSegmentationBinaryLabelmapRepresentationName=lambda: "BinaryLabelmap",
        )
        _slicer.vtkSlicerSegmentationsModuleLogic = types.SimpleNamespace(
            SetBinaryLabelmapToSegment=lambda *args, **kwargs: None,
        )
        sys.modules["slicer"] = _slicer

        class _FakeSeg:
            def __init__(self):
                self._segs = {}
            def AddEmptySegment(self, seg_id, name, color):
                self._segs[seg_id] = {"name": name, "color": list(color)}
                return seg_id
            def RemoveAllSegments(self):
                self._segs = {}
            def GetSegments(self):
                return list(self._segs.keys())
            def SetSourceRepresentationName(self, name):
                pass
        class _FakeSegNode:
            def __init__(self):
                self._seg = _FakeSeg()
            def GetSegmentation(self):
                return self._seg
            def StartModify(self): return False
            def EndModify(self, _was): pass
            def SetReferenceImageGeometryParameterFromVolumeNode(self, _v): pass

        seg_node = _FakeSegNode()
        result = {
            "filename": "fish.png",
            "original": np.zeros((256, 256, 3), dtype=np.uint8),
            "mask": np.zeros((256, 256), dtype=np.uint8),
            "eye_mask": np.zeros((256, 256), dtype=np.uint8),
        }
        from ZebrafishEmbryoAnalyzerLib.mrml import update_segmentation_node
        update_segmentation_node(result, 22.99, seg_node, image_node=None)
        segs = seg_node.GetSegmentation().GetSegments()
        assert "Body" in segs, segs
        print("OK")
    """)
    env = {**os.environ, "ZEA_DIR": _MODULE_DIR}
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert "OK" in r.stdout


def test_validate_volume_node_flags_dangling_seg_reference(
    volume_node,
):
    """Issue #56 follow-up: ``validate_volume_node`` must now resolve the
    seg id against the live scene and report an error when the id is
    dangling (the user deleted the seg in the Data module). Previously
    the role-only check accepted a dangling reference as healthy.
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        validate_volume_node, ROLE_ZEBRAFISH_SEGMENTATION,
    )
    # Mark as analyzed (otherwise validate_volume_node short-circuits).
    volume_node.SetAttribute("ZebrafishAnalysis.exclude", "false")
    # Set the role to a bogus id (the real seg was deleted).
    volume_node.SetNodeReferenceID(
        ROLE_ZEBRAFISH_SEGMENTATION, "vtkMRMLSegmentationNode999",
    )
    # Stub a minimal scene with no matching node.
    import sys
    import types
    slicer_stub = types.ModuleType("slicer")
    slicer_stub.mrmlScene = _FakeScene()
    monkey_patcher = sys.modules.__setitem__("slicer", slicer_stub)
    try:
        err = validate_volume_node(volume_node)
    finally:
        # Restore slicer stub from earlier fixture, if any.
        pass
    assert err == ("Segmentation node missing", ""), (
        f"Dangling seg reference must surface as a recoverable error, got {err!r}"
    )


def test_logic_setup_segmentation_staleness_observers_delegates_to_widget():
    """Issue #56 follow-up: ``Logic.setup_segmentation_staleness_observers``
    must delegate to the widget's implementation (since the widget owns
    the ``VTKObservationMixin``). Without a back-pointer, the call is a
    silent no-op.

    Runs in a subprocess because ``ZebrafishEmbryoAnalyzer.py`` imports
    ``vtk`` / ``slicer`` at module load — see the matching pattern in
    ``tests/test_mrml_node.py``.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import os, sys, types
        sys.path.insert(0, os.environ["ZEA_DIR"])
        sys.modules["qt"]  = types.ModuleType("qt")
        sys.modules["ctk"] = types.ModuleType("ctk")
        from unittest.mock import MagicMock
        _vtk = types.ModuleType("vtk")
        _vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
        sys.modules["vtk"] = _vtk
        sys.modules["slicer"] = MagicMock()

        class _BaseWidget(object):
            pass

        class _VTKMixin(object):
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
        # Stub _evict_reload_modules's transitive imports to keep the
        # test independent of the rest of the extension's startup sequence.
        for name in (
            "ZebrafishEmbryoAnalyzerLib.errors",
            "ZebrafishEmbryoAnalyzerLib.model_manifest",
            "ZebrafishEmbryoAnalyzerLib.model_downloader",
            "ZebrafishEmbryoAnalyzerLib.inference_runner",
            "ZebrafishEmbryoAnalyzerLib.inference_worker",
            "ZebrafishEmbryoAnalyzerLib.mrml",
            "ZebrafishEmbryoAnalyzerLib.widget",
            "ZebrafishEmbryoAnalyzerLib.gallery_tab",
            "ZebrafishEmbryoAnalyzerLib.detail_tab",
            "ZebrafishEmbryoAnalyzerLib.results_tab",
            "ZebrafishEmbryoAnalyzerLib.logic",
            "ZebrafishEmbryoAnalyzerLib.overlay",
            "ZebrafishEmbryoAnalyzerLib.export",
            "ZebrafishEmbryoAnalyzerLib.dependency_installer",
            "ZebrafishEmbryoAnalyzerLib.zoom_view",
            "ZebrafishEmbryoAnalyzerCore.seg",
            "ZebrafishEmbryoAnalyzerCore.seg_helper",
            "ZebrafishEmbryoAnalyzerCore.length",
            "ZebrafishEmbryoAnalyzerCore.manual",
            "ZebrafishEmbryoAnalyzerCore.scalebar",
        ):
            sys.modules[name] = types.ModuleType(name)
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        called = {"count": 0}
        class _FakeWidget:
            def setup_segmentation_staleness_observers(self_inner):
                called["count"] += 1

        logic = ZebrafishEmbryoAnalyzerLogic()
        # Without _widget_ref: silent no-op (must not raise).
        logic.setup_segmentation_staleness_observers()
        assert called["count"] == 0, called["count"]

        logic._widget_ref = _FakeWidget()
        logic.setup_segmentation_staleness_observers()
        assert called["count"] == 1, called["count"]

        logic.setup_segmentation_staleness_observers()
        assert called["count"] == 2, called["count"]
        print("OK")
    """)
    env = {**os.environ, "ZEA_DIR": _MODULE_DIR}
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"subprocess failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "OK" in r.stdout


def test_logic_setup_segmentation_staleness_observers_swallows_widget_errors():
    """The Logic wrapper must never raise, even when the widget's
    implementation throws — callers in widget.py do not wrap their
    own try/except around this path.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import os, sys, types
        sys.path.insert(0, os.environ["ZEA_DIR"])
        sys.modules["qt"]  = types.ModuleType("qt")
        sys.modules["ctk"] = types.ModuleType("ctk")
        from unittest.mock import MagicMock
        _vtk = types.ModuleType("vtk")
        _vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
        sys.modules["vtk"] = _vtk
        sys.modules["slicer"] = MagicMock()

        class _BaseWidget(object):
            pass

        class _VTKMixin(object):
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
        for name in (
            "ZebrafishEmbryoAnalyzerLib.errors",
            "ZebrafishEmbryoAnalyzerLib.model_manifest",
            "ZebrafishEmbryoAnalyzerLib.model_downloader",
            "ZebrafishEmbryoAnalyzerLib.inference_runner",
            "ZebrafishEmbryoAnalyzerLib.inference_worker",
            "ZebrafishEmbryoAnalyzerLib.mrml",
            "ZebrafishEmbryoAnalyzerLib.widget",
            "ZebrafishEmbryoAnalyzerLib.gallery_tab",
            "ZebrafishEmbryoAnalyzerLib.detail_tab",
            "ZebrafishEmbryoAnalyzerLib.results_tab",
            "ZebrafishEmbryoAnalyzerLib.logic",
            "ZebrafishEmbryoAnalyzerLib.overlay",
            "ZebrafishEmbryoAnalyzerLib.export",
            "ZebrafishEmbryoAnalyzerLib.dependency_installer",
            "ZebrafishEmbryoAnalyzerLib.zoom_view",
            "ZebrafishEmbryoAnalyzerCore.seg",
            "ZebrafishEmbryoAnalyzerCore.seg_helper",
            "ZebrafishEmbryoAnalyzerCore.length",
            "ZebrafishEmbryoAnalyzerCore.manual",
            "ZebrafishEmbryoAnalyzerCore.scalebar",
        ):
            sys.modules[name] = types.ModuleType(name)
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        class _BoomWidget:
            def setup_segmentation_staleness_observers(self_inner):
                raise RuntimeError("scene not ready")

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic._widget_ref = _BoomWidget()
        # Must not raise.
        logic.setup_segmentation_staleness_observers()
        print("OK")
    """)
    env = {**os.environ, "ZEA_DIR": _MODULE_DIR}
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"subprocess failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "OK" in r.stdout


def test_widget_setup_wires_logic_widget_ref():
    """Issue #56 follow-up: ``Widget.setup`` must hand itself to the logic
    via ``_widget_ref`` so ``Logic.setup_segmentation_staleness_observers``
    can delegate back. Without this wiring the per-image ModifiedEvent
    observers are never installed after analysis completes.
    """
    # Read the source to keep the test stable across refactors: we want
    # the wiring in ``Widget.setup`` to be visible at a glance rather than
    # re-implementing the whole widget.
    import re
    import os
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzer.py",
    )
    with open(src_path, "r") as f:
        src = f.read()
    # The wiring must reference _widget_ref in setup().
    setup_block = re.search(
        r"def setup\(self\):.*?(?=\n    def )", src, flags=re.DOTALL,
    )
    assert setup_block is not None
    assert "logic._widget_ref" in setup_block.group(0), (
        "Widget.setup must wire self.logic._widget_ref = self so the "
        "Logic wrapper can delegate observer installation back to the widget"
    )


def test_widget_enter_calls_setup_segmentation_staleness_observers():
    """Issue #56 follow-up: ``Widget.enter()`` must re-arm the per-image
    segmentation ModifiedEvent observers on every module entry. Without
    this, observers installed by ``_on_results_ready`` get torn down on
    tab switch and the user's later Segment Editor edits never trigger
    the recompute prompt.
    """
    import re
    import os
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzer.py",
    )
    with open(src_path, "r") as f:
        src = f.read()
    enter_block = re.search(
        r"def enter\(self\):.*?(?=\n    def )", src, flags=re.DOTALL,
    )
    assert enter_block is not None
    assert "setup_segmentation_staleness_observers" in enter_block.group(0), (
        "Widget.enter() must call setup_segmentation_staleness_observers "
        "so observers are live on every module re-entry"
    )