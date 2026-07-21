"""
Tests for merging a saved scene into an already-open session (issue #36).

Slicer's scene *import* adds the imported nodes to the current scene, but the
module's parameter node is a singleton: MRML copies the imported one onto the
existing node, replacing the ``ZebrafishImage`` reference list instead of
extending it. The previous batch's volume nodes survive with no reference
pointing at them, so the module renders only the imported images while the
Data module shows both.

These tests exercise the reconciliation that repairs the list, using small
fakes — conftest.py puts the module directory on sys.path so mrml.py imports
without the Slicer runtime.
"""

import os

from ZebrafishEmbryoAnalyzerLib.mrml import (
    ATTR_LOAD_ORDER,
    ROLE_ZEBRAFISH_IMAGES,
    find_zebrafish_volume_nodes_in_scene,
    list_tracked_volume_nodes,
    reconcile_tracked_volume_nodes,
)


class _FakeVolumeNode:
    def __init__(self, node_id, load_order=None, node_class="vtkMRMLVolumeNode"):
        self._id = node_id
        self._class = node_class
        self._attrs = {}
        if load_order is not None:
            self._attrs[ATTR_LOAD_ORDER] = str(load_order)

    def GetID(self):
        return self._id

    def GetName(self):
        return self._id

    def GetAttribute(self, key):
        return self._attrs.get(key)

    def SetAttribute(self, key, value):
        self._attrs[key] = value

    def IsA(self, class_name):
        return class_name == self._class

    @property
    def load_order(self):
        raw = self._attrs.get(ATTR_LOAD_ORDER)
        return None if raw is None else int(raw)


class _FakeScene:
    def __init__(self, nodes):
        self._nodes = list(nodes)

    def _of_class(self, class_name):
        return [n for n in self._nodes if n.IsA(class_name)]

    def GetNumberOfNodesByClass(self, class_name):
        return len(self._of_class(class_name))

    def GetNthNodeByClass(self, index, class_name):
        return self._of_class(class_name)[index]

    def GetNodeByID(self, node_id):
        for node in self._nodes:
            if node.GetID() == node_id:
                return node
        return None


class _FakeParamNode:
    def __init__(self, ids=()):
        self._refs = {ROLE_ZEBRAFISH_IMAGES: list(ids)}

    def GetNumberOfNodeReferences(self, role):
        return len(self._refs.get(role, []))

    def GetNthNodeReferenceID(self, role, index):
        return self._refs.get(role, [])[index]

    def AddNodeReferenceID(self, role, node_id):
        self._refs.setdefault(role, []).append(node_id)

    @property
    def tracked_ids(self):
        return list(self._refs.get(ROLE_ZEBRAFISH_IMAGES, []))


def _batch(prefix, count, start_order=0):
    return [
        _FakeVolumeNode(f"{prefix}{i}", load_order=start_order + i)
        for i in range(count)
    ]


def test_find_zebrafish_volume_nodes_skips_foreign_volumes():
    """Only nodes this module stamped are ours — a CT the user loaded
    separately must not be swept into the gallery.
    """
    mine = _batch("mine", 2)
    foreign = _FakeVolumeNode("someCT", load_order=None)
    scene = _FakeScene([mine[0], foreign, mine[1]])

    found = find_zebrafish_volume_nodes_in_scene(scene)

    assert [n.GetID() for n in found] == ["mine0", "mine1"]


def test_find_zebrafish_volume_nodes_tolerates_a_scene_without_the_api():
    assert find_zebrafish_volume_nodes_in_scene(None) == []
    assert find_zebrafish_volume_nodes_in_scene(object()) == []


def test_reconcile_is_a_noop_when_every_discovered_node_is_registered():
    """The overwhelmingly common case: a plain reload, references intact.
    Nothing may be added and load order must be left alone.
    """
    nodes = _batch("img", 3)
    scene = _FakeScene(nodes)
    param = _FakeParamNode([n.GetID() for n in nodes])

    added = reconcile_tracked_volume_nodes(param, scene)

    assert added == []
    assert param.tracked_ids == ["img0", "img1", "img2"]
    assert [n.load_order for n in nodes] == [0, 1, 2]


def test_reconcile_reregisters_orphans_and_keeps_the_old_batch_in_front():
    """The merge case. After the singleton copy the references name only the
    imported batch; the pre-import snapshot is what puts the existing images
    back in front of it.
    """
    old = _batch("old", 2)
    new = _batch("new", 3)
    scene = _FakeScene(old + new)
    # Post-import state: references were replaced by the imported batch.
    param = _FakeParamNode([n.GetID() for n in new])
    snapshot = [n.GetID() for n in old]

    added = reconcile_tracked_volume_nodes(param, scene, snapshot)

    assert added == ["old0", "old1"]
    assert sorted(param.tracked_ids) == ["new0", "new1", "new2", "old0", "old1"]
    # The user-visible contract: gallery order is old first, imported after.
    assert [n.GetID() for n in list_tracked_volume_nodes(param, scene)] == [
        "old0", "old1", "new0", "new1", "new2",
    ]


def test_reconcile_renumbers_load_order_across_both_batches():
    """Both scenes number their own images from zero, and
    ``list_tracked_volume_nodes`` sorts on that attribute — without
    renumbering the two batches interleave.
    """
    old = _batch("old", 2)
    new = _batch("new", 2)
    assert [n.load_order for n in old] == [0, 1]
    assert [n.load_order for n in new] == [0, 1]
    scene = _FakeScene(old + new)
    param = _FakeParamNode([n.GetID() for n in new])

    reconcile_tracked_volume_nodes(param, scene, [n.GetID() for n in old])

    assert [n.load_order for n in old] == [0, 1]
    assert [n.load_order for n in new] == [2, 3]


def test_reconcile_without_a_snapshot_appends_orphans_after_the_references():
    """No StartImportEvent was observed (the merge happened before the module
    was ever opened). Every image still comes back — only the order between
    the batches falls back to references-first.
    """
    orphans = _batch("orphan", 2)
    tracked = _batch("tracked", 2)
    scene = _FakeScene(orphans + tracked)
    param = _FakeParamNode([n.GetID() for n in tracked])

    added = reconcile_tracked_volume_nodes(param, scene, None)

    assert added == ["orphan0", "orphan1"]
    assert [n.GetID() for n in list_tracked_volume_nodes(param, scene)] == [
        "tracked0", "tracked1", "orphan0", "orphan1",
    ]


def test_reconcile_ignores_snapshot_ids_whose_nodes_are_gone():
    """A node the user deleted from the Data module between the snapshot and
    the import must not resurrect as a dangling reference.
    """
    surviving = _FakeVolumeNode("old0", load_order=0)
    new = _batch("new", 1)
    scene = _FakeScene([surviving] + new)
    param = _FakeParamNode([n.GetID() for n in new])

    added = reconcile_tracked_volume_nodes(param, scene, ["old0", "deletedNode"])

    assert added == ["old0"]
    assert "deletedNode" not in param.tracked_ids
    assert [n.GetID() for n in list_tracked_volume_nodes(param, scene)] == [
        "old0", "new0",
    ]


def test_reconcile_handles_missing_param_node_or_scene():
    assert reconcile_tracked_volume_nodes(None, _FakeScene([])) == []
    assert reconcile_tracked_volume_nodes(_FakeParamNode(), None) == []


def _module_source():
    return open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzer.py",
        )
    ).read()


def test_start_import_observer_is_registered():
    """Without the StartImportEvent observer there is no pre-import snapshot,
    and the merge silently degrades to arbitrary batch order.

    Source-level scan — registering real MRML observers needs the Slicer
    runtime (same rationale as the scene-reload ordering tests).
    """
    src = _module_source()
    idx = src.find("def _register_scene_observers")
    assert idx >= 0
    end = src.find("\n    def ", idx + 1)
    body = src[idx:end if end != -1 else len(src)]

    assert "slicer.mrmlScene.StartImportEvent" in body
    assert "self._on_scene_start_import" in body


def test_rebuild_reconciles_before_reading_the_reference_list():
    """The reconcile must run BEFORE ``list_tracked_volume_nodes`` — it
    exists precisely to repair that list, so reading first would return the
    imported batch only.
    """
    src = _module_source()
    idx = src.find("def rebuild_results_from_scene")
    assert idx >= 0
    end = src.find("\n    def ", idx + 1)
    body = src[idx:end if end != -1 else len(src)]

    # Match call expressions, not bare names: both functions also appear in
    # the method's import block a few lines above.
    reconcile_idx = body.find("reconcile_tracked_volume_nodes(\n")
    read_idx = body.find("list_tracked_volume_nodes(param_node, scene)")
    assert reconcile_idx >= 0, (
        "rebuild_results_from_scene must call reconcile_tracked_volume_nodes()"
    )
    assert read_idx >= 0
    assert reconcile_idx < read_idx, (
        "reconcile must run before the reference list is read (issue #36)"
    )
