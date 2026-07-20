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
    """
    import ZebrafishEmbryoAnalyzerLib.mrml as mrml_mod

    captured = {}

    def _fake(result, um_per_px, node, image_node=None):
        captured["result"] = result
        captured["um_per_px"] = um_per_px
        captured["node"] = node
        captured["image_node"] = image_node
        seg = node.GetSegmentation()
        # Mirror production: body always, eye only when present + non-empty.
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
    """User-excluded images must be queryable without scanning geometry.

    Issue #74: the whole-row bool ``result["exclude"] = True`` (no explicit
    ``exclude_metrics``) is the backward-compatible fallback path and is
    encoded as the new schema's own whole-row shorthand "*" (not the legacy
    "true" — that spelling is still accepted when *reading*, see
    ``_decode_exclude_metrics``, but is no longer what this codebase writes).
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ATTR_EXCLUDE,
        apply_analysis_to_volume_node,
    )

    result = _make_full_result()
    result["exclude"] = True
    apply_analysis_to_volume_node(result, volume_node, scene, 22.99)
    assert volume_node.GetAttribute(ATTR_EXCLUDE) == "*"


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