"""
Tests for issue #38: per-image volume node batch helpers.

Covers:
- ``create_image_volume_node`` creates one new ``vtkMRMLVectorVolumeNode``
  per call (no reuse), and registers its ID via ``AddNodeReferenceID``
  under ``ROLE_ZEBRAFISH_IMAGES``.
- ``create_image_volume_node`` rolls back the half-constructed node when
  the image population step raises.
- ``remove_all_image_volume_nodes`` clears every tracked volume node and
  any node references reachable from them (recursive cleanup is a no-op
  today but the structure is in place for sub-issue #39).
- Helper accepts both list and single-id forms for ``RemoveNodeReferenceIDs``.
- Widget helpers: ``_format_readability_message`` caps the list at 10
  entries and adds a ``"... and N more"`` tail. ``_filter_readable_paths``
  routes bad files into the failed bucket.

Tests run with plain ``pytest tests/`` (no Slicer runtime). Mocks follow
the project's existing MagicMock-based pattern from ``test_mrml_node.py``.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ZebrafishEmbryoAnalyzer",
)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)


# ---------------------------------------------------------------------------
# Fake scene / param node / volume node
# ---------------------------------------------------------------------------

class _FakeVolumeNode:
    """Minimal vtkMRMLVectorVolumeNode substitute."""

    _counter = 0

    def __init__(self, name="Image"):
        _FakeVolumeNode._counter += 1
        self._id = f"vtkMRMLVectorVolumeNode{_FakeVolumeNode._counter}"
        self._name = name
        self._ref_ids = []  # child node-reference IDs across all roles
        self._attrs = {}

    def GetID(self):
        return self._id

    def GetAttribute(self, name):
        return self._attrs.get(name)

    def SetAttribute(self, name, value):
        self._attrs[name] = value

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLVectorVolumeNode"

    # Child references — used to verify the recursive cleanup path.
    # Real vtkMRMLNode does not expose a plural GetNodeReferenceIDs(role)
    # getter in the Python binding; production code enumerates via
    # GetNumberOfNodeReferences/GetNthNodeReferenceID and GetNodeReferenceRoles.
    def GetNodeReferenceRoles(self, out_roles):
        out_roles.extend(["_all"] if self._ref_ids else [])

    def GetNumberOfNodeReferences(self, role):
        return len(self._ref_ids) if role == "_all" else 0

    def GetNthNodeReferenceID(self, role, n):
        return self._ref_ids[n] if role == "_all" else None

    def AddReference(self, child_id):
        self._ref_ids.append(child_id)


class _FakeChildNode:
    """Generic foreign node reachable through a volume node's references.

    Used to verify recursive cleanup handles referenced children."""

    _counter = 0

    def __init__(self, class_name):
        _FakeChildNode._counter += 1
        self._id = f"{class_name}{_FakeChildNode._counter}"
        self._class = class_name
        self._ref_ids = []

    def GetID(self):
        return self._id

    def IsA(self, class_name):
        return class_name == self._class

    def GetNodeReferenceRoles(self, out_roles):
        out_roles.extend(["_all"] if self._ref_ids else [])

    def GetNumberOfNodeReferences(self, role):
        return len(self._ref_ids) if role == "_all" else 0

    def GetNthNodeReferenceID(self, role, n):
        return self._ref_ids[n] if role == "_all" else None


class _FakeScene:
    """Minimal vtkMRMLScene substitute."""

    def __init__(self):
        self._nodes = {}
        self.add_calls = []
        self.remove_log = []

    def AddNewNodeByClass(self, class_name, display_name=""):
        self.add_calls.append((class_name, display_name))
        node = _FakeVolumeNode(name=display_name or class_name)
        self._nodes[node.GetID()] = node
        return node

    def GetNodeByID(self, node_id):
        return self._nodes.get(node_id)

    def RemoveNode(self, node):
        nid = node.GetID() if hasattr(node, "GetID") else None
        self.remove_log.append(nid)
        if nid in self._nodes:
            del self._nodes[nid]

    def add_child(self, child):
        """Test helper: register a foreign node with the scene directly."""
        self._nodes[child.GetID()] = child
        return child

    def node_count(self):
        return len(self._nodes)


class _FakeBatchParamNode:
    """Parameter node with additive-reference-list semantics (issue #38)."""

    def __init__(self):
        self._refs = {}  # role -> list[str]
        self.add_calls = 0
        self.remove_calls = []

    def GetNodeReference(self, role):
        ids = self._refs.get(role, [])
        if not ids:
            return None
        # Caller is expected to resolve via scene.GetNodeByID.
        return types.SimpleNamespace(GetID=lambda: ids[0])

    def GetNodeReferenceRoles(self, out_roles):
        out_roles.extend(r for r, ids in self._refs.items() if ids)

    def GetNumberOfNodeReferences(self, role):
        return len(self._refs.get(role, []))

    def GetNthNodeReferenceID(self, role, n):
        ids = self._refs.get(role, [])
        return ids[n] if 0 <= n < len(ids) else None

    def AddNodeReferenceID(self, role, node_id):
        self.add_calls += 1
        self._refs.setdefault(role, []).append(node_id)

    def SetNodeReferenceID(self, role, node_id):
        self._refs[role] = [node_id]

    def RemoveNodeReferenceIDs(self, role, ids):
        self.remove_calls.append((role, list(ids) if isinstance(ids, list) else [ids]))
        existing = self._refs.get(role, [])
        if isinstance(ids, str):
            ids = [ids]
        for i in ids:
            if i in existing:
                existing.remove(i)

    def ref_count(self, role):
        return len(self._refs.get(role, []))


# ---------------------------------------------------------------------------
# create_image_volume_node — direct behavior
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_populate(monkeypatch):
    """Patch _populate_image_node to a no-op so update_image_node never
    runs. Tests of remove/recursion don't care about pixel contents; the
    actual image-data path is covered by test_update_image_node_* tests.
    """
    import ZebrafishEmbryoAnalyzerLib.mrml as mrml_mod
    monkeypatch.setattr(mrml_mod, "_populate_image_node", lambda *_a, **_kw: None)


def test_create_image_volume_node_creates_one_node_per_call(stub_populate):
    """Each call creates a brand-new vtkMRMLVectorVolumeNode (never reused)."""
    from ZebrafishEmbryoAnalyzerLib.mrml import create_image_volume_node

    param_node = _FakeBatchParamNode()
    scene = _FakeScene()
    image_rgb = MagicMock()  # stand-in for the pixel array

    n1 = create_image_volume_node(image_rgb, 22.99, "a.png", param_node, scene)
    n2 = create_image_volume_node(image_rgb, 22.99, "b.png", param_node, scene)

    assert n1 is not n2, "second call must not reuse first node"
    assert scene.add_calls[0] == ("vtkMRMLVectorVolumeNode", "a.png")
    assert scene.add_calls[1] == ("vtkMRMLVectorVolumeNode", "b.png")
    assert param_node.add_calls == 2, "AddNodeReferenceID must fire once per call"
    assert param_node.ref_count("ZebrafishImage") == 2, (
        "two entries expected in the additive reference list"
    )


def test_create_image_volume_node_appends_via_AddNodeReferenceID(stub_populate):
    """The new node ID is registered through AddNodeReferenceID (additive)."""
    from ZebrafishEmbryoAnalyzerLib.mrml import create_image_volume_node

    param_node = _FakeBatchParamNode()
    scene = _FakeScene()
    image_rgb = MagicMock()

    node = create_image_volume_node(image_rgb, 22.99, "fish01.tif", param_node, scene)

    ids = list(param_node._refs.get("ZebrafishImage", []))
    assert ids == [node.GetID()], f"single append expected, got {ids}"


def test_create_image_volume_node_stamps_load_order_attribute(stub_populate):
    """Each node gets ZebrafishAnalysis.loadOrder = its batch position,
    read from the reference count *before* this call's AddNodeReferenceID —
    list_tracked_volume_nodes uses this to restore folder-load order even
    if the reference array itself comes back reordered after a scene
    reload (found while testing #61).
    """
    from ZebrafishEmbryoAnalyzerLib.mrml import create_image_volume_node

    param_node = _FakeBatchParamNode()
    scene = _FakeScene()
    image_rgb = MagicMock()

    n0 = create_image_volume_node(image_rgb, 22.99, "a.png", param_node, scene)
    n1 = create_image_volume_node(image_rgb, 22.99, "b.png", param_node, scene)
    n2 = create_image_volume_node(image_rgb, 22.99, "c.png", param_node, scene)

    assert n0.GetAttribute("ZebrafishAnalysis.loadOrder") == "0"
    assert n1.GetAttribute("ZebrafishAnalysis.loadOrder") == "1"
    assert n2.GetAttribute("ZebrafishAnalysis.loadOrder") == "2"


def test_create_image_volume_node_uses_filename_as_display_name(stub_populate):
    """The display name should reflect the original filename for the Data module."""
    from ZebrafishEmbryoAnalyzerLib.mrml import create_image_volume_node

    param_node = _FakeBatchParamNode()
    scene = _FakeScene()
    create_image_volume_node(MagicMock(), 22.99, "zebrafish_01.png", param_node, scene)

    cls, name = scene.add_calls[-1]
    assert cls == "vtkMRMLVectorVolumeNode"
    assert name == "zebrafish_01.png"


def test_create_image_volume_node_falls_back_to_default_display_name(stub_populate):
    """An empty name hint still yields a usable display name."""
    from ZebrafishEmbryoAnalyzerLib.mrml import create_image_volume_node

    param_node = _FakeBatchParamNode()
    scene = _FakeScene()
    create_image_volume_node(MagicMock(), 22.99, "", param_node, scene)

    _, name = scene.add_calls[-1]
    assert name, "display name must be non-empty"


def test_create_image_volume_node_rolls_back_half_built_node_on_failure(monkeypatch):
    """If image population raises, the empty node must be removed from the scene."""
    import ZebrafishEmbryoAnalyzerLib.mrml as mrml_mod
    from ZebrafishEmbryoAnalyzerLib.mrml import create_image_volume_node

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated update_image_node failure")

    monkeypatch.setattr(mrml_mod, "_populate_image_node", _boom)

    scene = _FakeScene()
    param_node = _FakeBatchParamNode()

    with pytest.raises(RuntimeError, match="simulated update_image_node failure"):
        create_image_volume_node(MagicMock(), 22.99, "bad.tif", param_node, scene)

    # Rollback: RemoveNode was called for the half-built node.
    assert len(scene.remove_log) == 1, (
        f"expected half-built node to be removed, got {scene.remove_log}"
    )
    # Failed image must NOT have been registered in the reference list.
    assert param_node.ref_count("ZebrafishImage") == 0, (
        "failed node ID must not be added to the parameter node reference list"
    )
    assert scene.node_count() == 0, (
        "scene must not retain an orphaned empty volume node"
    )


def test_create_image_volume_node_invokes_populate_with_image_and_spacing(monkeypatch):
    """The image and um_per_px must reach update_image_node unmodified."""
    import ZebrafishEmbryoAnalyzerLib.mrml as mrml_mod

    captured = {}

    def _capture(image_rgb, um_per_px, node):
        captured["image_rgb"] = image_rgb
        captured["um_per_px"] = um_per_px
        captured["node"] = node

    monkeypatch.setattr(mrml_mod, "_populate_image_node", _capture)

    param_node = _FakeBatchParamNode()
    scene = _FakeScene()
    image_rgb = MagicMock()
    node = mrml_mod.create_image_volume_node(image_rgb, 12.34, "x.png", param_node, scene)

    assert captured["image_rgb"] is image_rgb
    assert captured["um_per_px"] == 12.34
    assert captured["node"] is node


# ---------------------------------------------------------------------------
# remove_all_image_volume_nodes — direct behavior
# ---------------------------------------------------------------------------

def test_remove_all_image_volume_nodes_removes_only_tracked_nodes(stub_populate):
    """Removal touches only nodes stored under ROLE_ZEBRAFISH_IMAGES, never foreign nodes."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_IMAGES,
        create_image_volume_node,
        remove_all_image_volume_nodes,
    )

    scene = _FakeScene()
    param_node = _FakeBatchParamNode()
    img = MagicMock()

    owned_a = create_image_volume_node(img, 22.99, "a.png", param_node, scene)
    owned_b = create_image_volume_node(img, 22.99, "b.png", param_node, scene)
    # Foreign node the user added directly; must survive.
    foreign = scene.add_child(_FakeVolumeNode("user-volume"))

    assert param_node.ref_count(ROLE_ZEBRAFISH_IMAGES) == 2

    removed = remove_all_image_volume_nodes(param_node, scene)

    assert removed == 2, f"expected 2 tracked nodes removed, got {removed}"
    assert owned_a.GetID() in scene.remove_log
    assert owned_b.GetID() in scene.remove_log
    assert foreign.GetID() not in scene.remove_log, "foreign node must be preserved"
    assert scene.GetNodeByID(foreign.GetID()) is foreign, (
        "foreign node must remain in the scene"
    )
    assert param_node.ref_count(ROLE_ZEBRAFISH_IMAGES) == 0, (
        "reference list must be cleared after removal"
    )


def test_remove_all_image_volume_nodes_recurses_into_children(stub_populate):
    """When #39 lands, segmentation / markups references on the volume node
    must also be cleaned up. Today the structure is exercised by attaching
    a child node to a volume node and verifying it is removed too."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_IMAGES,
        create_image_volume_node,
        remove_all_image_volume_nodes,
    )

    scene = _FakeScene()
    param_node = _FakeBatchParamNode()

    owned = create_image_volume_node(MagicMock(), 22.99, "a.png", param_node, scene)
    child = scene.add_child(_FakeChildNode("vtkMRMLSegmentationNode"))
    owned.AddReference(child.GetID())

    # Sanity: scene knows about both nodes, reference is recorded.
    assert child.GetID() in [id for id in scene._nodes]
    assert child.GetID() in owned._ref_ids

    removed = remove_all_image_volume_nodes(param_node, scene)

    assert removed == 1
    assert owned.GetID() in scene.remove_log
    assert child.GetID() in scene.remove_log, (
        "child node reachable through volume node references must be removed too"
    )
    assert scene.GetNodeByID(child.GetID()) is None
    assert scene.GetNodeByID(owned.GetID()) is None


def test_remove_all_image_volume_nodes_empty_list_is_noop():
    """Calling remove with an empty reference list returns 0 and touches nothing."""
    from ZebrafishEmbryoAnalyzerLib.mrml import remove_all_image_volume_nodes

    scene = _FakeScene()
    param_node = _FakeBatchParamNode()

    removed = remove_all_image_volume_nodes(param_node, scene)

    assert removed == 0
    assert scene.remove_log == []


def test_remove_all_image_volume_nodes_none_inputs_are_noop():
    """None param_node / scene must not raise — caller may be mid teardown."""
    from ZebrafishEmbryoAnalyzerLib.mrml import remove_all_image_volume_nodes

    assert remove_all_image_volume_nodes(None, MagicMock()) == 0
    assert remove_all_image_volume_nodes(MagicMock(), None) == 0
    assert remove_all_image_volume_nodes(None, None) == 0


def test_remove_all_image_volume_nodes_skips_missing_ids():
    """If the scene no longer holds a tracked ID, the cleanup must skip it gracefully."""
    from ZebrafishEmbryoAnalyzerLib.mrml import remove_all_image_volume_nodes

    scene = _FakeScene()
    param_node = _FakeBatchParamNode()
    # Manually register an ID that does not correspond to any live scene node.
    param_node.AddNodeReferenceID("ZebrafishImage", "vtkMRMLVectorVolumeNode999")

    removed = remove_all_image_volume_nodes(param_node, scene)

    assert removed == 0, "missing scene nodes must not be counted as removed"
    assert scene.remove_log == [], "missing nodes must not trigger RemoveNode"


def test_remove_all_image_volume_nodes_clears_reference_list(stub_populate):
    """The parameter node reference list must be cleared after removal."""
    from ZebrafishEmbryoAnalyzerLib.mrml import (
        ROLE_ZEBRAFISH_IMAGES,
        create_image_volume_node,
        remove_all_image_volume_nodes,
    )

    scene = _FakeScene()
    param_node = _FakeBatchParamNode()

    for i in range(3):
        create_image_volume_node(MagicMock(), 22.99, f"img{i}.png", param_node, scene)

    assert param_node.ref_count(ROLE_ZEBRAFISH_IMAGES) == 3
    remove_all_image_volume_nodes(param_node, scene)
    assert param_node.ref_count(ROLE_ZEBRAFISH_IMAGES) == 0, (
        "post-removal, the parameter node must not retain any ZebrafishImage refs"
    )


# ---------------------------------------------------------------------------
# Order-of-operations for `_set_queue` (issue #38)
# ---------------------------------------------------------------------------

@pytest.fixture
def widget_module(monkeypatch):
    """Reload widget module with qt/ctk/slicer stubbed, mirroring the fixture
    in test_h2_error_clarity.py."""
    monkeypatch.setitem(sys.modules, "qt", MagicMock())
    monkeypatch.setitem(sys.modules, "ctk", MagicMock())
    slicer = MagicMock()
    slicer.util.mainWindow.return_value = None
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    import importlib
    import ZebrafishEmbryoAnalyzerLib.widget as module
    return importlib.reload(module)


# ---------------------------------------------------------------------------
# Static / module-level checks
# ---------------------------------------------------------------------------

def test_module_exports_role_zebrafish_images():
    """mrml.py must expose ROLE_ZEBRAFISH_IMAGES (issue #38 public constant)."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    assert hasattr(mrml, "ROLE_ZEBRAFISH_IMAGES"), (
        "mrml.py must expose ROLE_ZEBRAFISH_IMAGES"
    )
    assert isinstance(mrml.ROLE_ZEBRAFISH_IMAGES, str)
    assert mrml.ROLE_ZEBRAFISH_IMAGES  # non-empty


def test_module_exports_batch_helpers():
    """mrml.py must export the per-image batch helpers used by the widget/logic."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    for name in ("create_image_volume_node", "remove_all_image_volume_nodes"):
        assert hasattr(mrml, name), f"mrml.py must expose {name}"
        assert callable(getattr(mrml, name))


# ---------------------------------------------------------------------------
# Widget helpers: _format_readability_message and _filter_readable_paths
# ---------------------------------------------------------------------------

def _w_import_widget():
    """Import the widget module under a Slicer MagicMock stub.

    Mirrors the pattern in tests/test_h2_error_clarity.py: stub qt, ctk,
    slicer; reload widget so the module sees the stub.
    """
    sys.modules.setdefault("qt", MagicMock())
    sys.modules.setdefault("ctk", MagicMock())
    slicer = MagicMock()
    slicer.util.mainWindow.return_value = None
    sys.modules["slicer"] = slicer
    import importlib
    import ZebrafishEmbryoAnalyzerLib.widget as widget_mod
    return importlib.reload(widget_mod)


def test_format_readability_message_single_file():
    widget_mod = _w_import_widget()
    msg = widget_mod.ZebrafishEmbryoAnalyzerMainWidget._format_readability_message(
        1, ["a.png"],
    )
    assert msg == "1 image could not be read and were not imported: a.png", msg


def test_format_readability_message_multiple_files_no_cap():
    widget_mod = _w_import_widget()
    names = [f"img{i}.png" for i in range(5)]
    msg = widget_mod.ZebrafishEmbryoAnalyzerMainWidget._format_readability_message(
        5, names,
    )
    assert msg.startswith("5 images could not be read and were not imported: ")
    for n in names:
        assert n in msg, f"{n} missing from {msg}"


def test_format_readability_message_caps_at_ten_with_more_tail():
    widget_mod = _w_import_widget()
    names = [f"img{i:02d}.png" for i in range(25)]
    msg = widget_mod.ZebrafishEmbryoAnalyzerMainWidget._format_readability_message(
        25, names,
    )
    # First 10 names appear, the rest do not, and a `... and N more` tail is appended.
    shown = names[:10]
    not_shown = names[10:]
    for n in shown:
        assert n in msg, f"expected {n} in capped list: {msg}"
    for n in not_shown:
        assert n not in msg, f"cap should hide {n}: {msg}"
    assert "... and 15 more" in msg, f"missing tail in {msg}"


def test_format_readability_message_exact_cap_no_tail():
    """When total == cap, no `... and N more` tail is appended."""
    widget_mod = _w_import_widget()
    names = [f"img{i}.png" for i in range(10)]
    msg = widget_mod.ZebrafishEmbryoAnalyzerMainWidget._format_readability_message(
        10, names,
    )
    assert "... and" not in msg, f"no tail expected at exact cap: {msg}"


def test_filter_readable_paths_routes_unreadable_files(tmp_path, monkeypatch):
    """Files cv2.imread fails on go into the 'failed' bucket, not 'readable'."""
    widget_mod = _w_import_widget()

    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    # File existence alone is not enough: cv2.imread returns None when it
    # cannot decode the bytes. Drive the contract through a fully-stubbed
    # cv2.imread so the test is hermetic.
    import cv2

    def _fake_imread(path, *a, **kw):
        return None if str(path) == str(bad) else np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "imread", _fake_imread)

    widget = object.__new__(widget_mod.ZebrafishEmbryoAnalyzerMainWidget)
    readable, failed, decoded = widget._filter_readable_paths([str(good), str(bad)])

    assert str(good) in readable
    assert str(bad) not in readable
    assert "bad.png" in failed
    assert "good.png" not in failed
    # Issue #38 acceptance: pre-flight also produces the decoded array.
    assert str(good) in decoded, (
        f"decoded dict must contain the readable path; got {list(decoded.keys())}"
    )
    assert decoded[str(good)].shape[-1] == 3, (
        f"decoded RGB must be 3-channel; shape was {decoded[str(good)].shape}"
    )
    assert str(bad) not in decoded, "failed paths must not appear in decoded dict"


def test_filter_readable_paths_keeps_all_when_everything_loads(tmp_path, monkeypatch):
    """When every cv2.imread returns a non-None array, no failures are reported."""
    widget_mod = _w_import_widget()

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    import cv2
    monkeypatch.setattr(cv2, "imread", lambda *a, **kw: np.zeros((10, 10, 3), dtype=np.uint8))

    widget = object.__new__(widget_mod.ZebrafishEmbryoAnalyzerMainWidget)
    readable, failed, decoded = widget._filter_readable_paths([str(a), str(b)])

    assert len(readable) == 2
    assert failed == []
    assert set(decoded) == {str(a), str(b)}, (
        f"decoded must carry every readable path; got {list(decoded.keys())}"
    )


def test_set_queue_calls_replace_image_volume_nodes_before_loading(widget_module):
    """Replace-on-load must fire BEFORE _load_originals so the previous
    scene contents cannot linger when a new batch is queued."""
    calls = []

    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 0
    w._active_runner = None

    w._logic = MagicMock()
    w._logic.replace_image_volume_nodes.side_effect = lambda: calls.append("replace")

    w._filter_readable_paths = MagicMock(return_value=([], [], {}))
    w._refresh_run_button = lambda: calls.append("refresh")
    w._load_originals = MagicMock(side_effect=lambda *a, **kw: calls.append("load"))

    # Other attributes that _set_queue touches — stub minimally.
    w._image_paths = []
    w._queue_list = MagicMock()
    w._results = []
    w._excluded = set()
    w._detail = MagicMock()
    w._results_tab = MagicMock()
    w._gallery = MagicMock()
    w._tabs = MagicMock()
    w._um_per_px = MagicMock()
    w._deps_ok = True
    w._load_result_label = MagicMock()  # Issue #62

    w._set_queue([])

    assert calls[:1] == ["replace"], (
        f"replace-on-load must run first; got {calls}"
    )
