"""
MRML adapter for ZebrafishEmbryoAnalyzer.

``results_to_rows`` is pure Python with no Slicer or VTK dependency and is
testable with standard pytest.

``get_or_create_table_node`` and ``populate_table_node`` require the Slicer
runtime.  ``populate_table_node`` imports vtk lazily inside its body so this
module is importable in plain Python test environments.
"""

import math

# Central table schema: (column_name, result_dict_key, vtk_array_type)
# vtk_array_type is "double" or "string".
# All tests, VTK column creation, and conversion use this single definition.
TABLE_SCHEMA = [
    ("Filename",            "filename",     "string"),
    ("Length_um",           "length",       "double"),
    ("CurvatureClass",      "curvature",    "string"),
    ("LengthStraightRatio", "ratio",        "double"),
    ("EyeArea_um2",         "eye_area",     "double"),
    ("EyeDiameter_um",      "eye_diameter", "double"),
    ("Error",               "error",        "string"),
]

ROLE_RESULTS_TABLE = "ResultsTable"
# Role used for the per-image volume node reference list on the parameter
# node (issue #38). Each successfully loaded image contributes one entry via
# AddNodeReferenceID; replace-on-load clears the list before the next batch.
ROLE_ZEBRAFISH_IMAGES = "ZebrafishImage"
# Per-image node-reference roles attached to each volume node (issue #39).
# Each successful analysis writes one segmentation node, one optional
# MarkupsLineNode (when length was computed), and one optional
# MarkupsCurveNode (when path_points exist). Sub-issue #5 needs to know the
# seg-node role to detect external edits.
ROLE_ZEBRAFISH_SEGMENTATION = "ZebrafishSegmentation"
ROLE_ZEBRAFISH_MARKUPS_LINE = "ZebrafishMarkupsLine"
ROLE_ZEBRAFISH_MARKUPS_CURVE = "ZebrafishMarkupsCurve"

# Per-image metric attributes (issue #39, ADR 0001). All attributes share the
# ``ZebrafishAnalysis.`` namespace so they do not collide with other modules
# storing data on the same volume node. Values are written verbatim; missing
# values are stored as the empty string so downstream readers can distinguish
# "not computed" (length disabled) from "computed and zero".
ATTR_PREFIX = "ZebrafishAnalysis."
ATTR_LENGTH = ATTR_PREFIX + "length"
ATTR_CURVATURE_CLASS = ATTR_PREFIX + "curvature_class"
ATTR_RATIO = ATTR_PREFIX + "ratio"
ATTR_EYE_AREA = ATTR_PREFIX + "eye_area"
ATTR_EYE_DIAMETER = ATTR_PREFIX + "eye_diameter"
ATTR_EXCLUDE = ATTR_PREFIX + "exclude"
# Issue #42: a segmentation node's ``ModifiedEvent`` observer sets
# ``ZebrafishAnalysis.stale = "true"`` whenever the user edits a Body
# mask in the Segment Editor. Cleared on successful recompute.
ATTR_STALE = ATTR_PREFIX + "stale"

# Issue #38: batch position, stamped on every image node at eager-creation
# time. Doubles as this module's ownership marker on a volume node — issue
# #36 uses it to find our images in a scene that also holds foreign volumes.
ATTR_LOAD_ORDER = ATTR_PREFIX + "loadOrder"

# Markups colors mirror ``overlay.py`` so the real MRML nodes match the custom
# Detail-tab overlay visually. Stored as RGB floats in [0, 1] — VTK's expected
# range for ``vtkMRMLDisplayNode.SetColor``.
_STRAIGHT_CLR = (0.784, 0.0, 0.784)     # magenta (overlay._STRAIGHT_CLR)
_PATH_COLOR = (0.0, 0.784, 0.784)       # cyan    (overlay._PATH_COLOR)

# String columns whose values are always preserved verbatim, even on error rows.
_PRESERVE_ON_ERROR = frozenset({"filename", "error"})


def results_to_rows(results):
    """Convert analysis result dicts to row dicts for the MRML table.

    Pure Python — no vtk or slicer imports. Testable with standard pytest.
    Input dicts are not mutated.

    Conversion rules (applied per column):
    - error row (non-empty "error" key): numeric → NaN, CurvatureClass → "",
      Filename and Error are always preserved
    - numeric field, value is None → math.nan
    - numeric field, value present → float(value)
    - string field, value is None  → ""
    - string field, value present  → str(value)

    Parameters
    ----------
    results : list[dict]
        List of result dicts from analyse_images().

    Returns
    -------
    list[dict]
        One dict per result, keyed by TABLE_SCHEMA column names, in schema order.
    """
    rows = []
    for r in results:
        has_error = bool(r.get("error"))
        row = {}
        for col_name, key, vtk_type in TABLE_SCHEMA:
            val = r.get(key)
            if vtk_type == "double":
                row[col_name] = math.nan if (has_error or val is None) else float(val)
            else:
                if key in _PRESERVE_ON_ERROR or not has_error:
                    row[col_name] = str(val) if val is not None else ""
                else:
                    row[col_name] = ""
        rows.append(row)
    return rows


def get_or_create_table_node(param_node, scene):
    """Return the existing ResultsTable node or create exactly one new node.

    Looks up the node by the stored node-reference role, not by display name.
    If no valid reference exists (missing or wrong node type), creates a new
    vtkMRMLTableNode, sets its initial display name, and registers its ID on
    the parameter node.  A wrong-type foreign node is left in the scene
    unchanged.

    Parameters
    ----------
    param_node : vtkMRMLScriptedModuleNode
        The module parameter node that owns the ResultsTable reference.
    scene : vtkMRMLScene
        The active MRML scene.

    Returns
    -------
    vtkMRMLTableNode
    """
    existing = param_node.GetNodeReference(ROLE_RESULTS_TABLE)
    if existing is not None and existing.IsA("vtkMRMLTableNode"):
        return existing

    node = scene.AddNewNodeByClass("vtkMRMLTableNode")
    node.SetName("ZebrafishEmbryoAnalyzer Results")
    param_node.SetNodeReferenceID(ROLE_RESULTS_TABLE, node.GetID())
    return node


def build_vtk_table(rows):
    """Build a complete vtkTable from conversion rows. No MRML side effects.

    Parameters
    ----------
    rows : list[dict]
        Output of results_to_rows().

    Returns
    -------
    vtk.vtkTable
    """
    import vtk  # lazy — not available in plain Python test environments

    n = len(rows)
    table = vtk.vtkTable()

    for col_name, _, vtk_type in TABLE_SCHEMA:
        if vtk_type == "double":
            arr = vtk.vtkDoubleArray()
        else:
            arr = vtk.vtkStringArray()
        arr.SetName(col_name)
        arr.SetNumberOfTuples(n)
        for i, row in enumerate(rows):
            arr.SetValue(i, row[col_name])
        table.AddColumn(arr)

    return table


def populate_table_node(rows, node):
    """Replace node content atomically with data from results_to_rows().

    Builds a complete vtk.vtkTable in memory first; applies it to the node
    only after all columns and values have been set successfully.  If
    construction fails the existing node content is preserved unchanged.

    Parameters
    ----------
    rows : list[dict]
        Output of results_to_rows().
    node : vtkMRMLTableNode
        Target node to update.
    """
    table = build_vtk_table(rows)
    node.SetAndObserveTable(table)


# ---------------------------------------------------------------------------
# Issue #40: results table as a derived cache from node attributes (ADR 0001)
# ---------------------------------------------------------------------------
#
# The table is *never* an authoritative store — every cell must be derivable
# from the per-image volume node attributes written in #39. This module
# exposes a single conversion path that builds the same in-memory rows
# whether it is called after a fresh analysis run or during a scene reload
# (issue #41), so the two callers cannot drift apart.
#
# Row↔node linkage no longer relies on the ``Filename`` column. We iterate
# the caller-supplied volume-node list in order — for the run-after-analysis
# path the widget passes the order in ``ROLE_ZEBRAFISH_IMAGES``; for the
# reload path issue #41 passes the same ordering rebuilt from the
# parameter-node reference list. The volume node's display name is preserved
# as the row's ``filename`` so existing export scripts keep working.


def _coerce_attr_float(node, attr_name):
    """Return the float value of ``node.GetAttribute(attr_name)`` or math.nan.

    Empty string (the sentinel set by :func:`_format_attr` for "not
    computed") becomes ``math.nan`` — same value that ``results_to_rows``
    writes for ``None`` numeric fields, so the table cell renders identically
    regardless of which side of the derivation pair produced it.

    Raises ``ValueError`` for non-empty values that do not parse as float
    (e.g. mistakenly-written strings), so the caller can attribute the
    problem to the right volume node.
    """
    raw = node.GetAttribute(attr_name) if hasattr(node, "GetAttribute") else None
    if raw is None or raw == "":
        return math.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"Volume node attribute {attr_name!r} is not a valid float: {raw!r}"
        )


def _coerce_attr_int_or_str(node, attr_name):
    """Return int(node.GetAttribute(attr_name)) if parseable, else the raw string.

    The ``curvature`` column is stored under
    ``ZebrafishAnalysis.curvature_class``; for older rows it may be the
    class id (``0``/``1``/``2``) and for newer ones a string label. We
    attempt int first and fall back to the raw string so both schemas
    round-trip.
    """
    raw = node.GetAttribute(attr_name) if hasattr(node, "GetAttribute") else None
    if raw is None or raw == "":
        return ""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return str(raw)


def volume_node_to_result_dict(node):
    """Reconstruct one result dict from a volume node's own attributes.

    Reads each metric key under :data:`ATTR_PREFIX` and converts back to the
    Python type the rest of the codebase expects:

    * ``length``, ``ratio``, ``eye_area``, ``eye_diameter`` → ``float`` or
      ``None`` when the attribute is absent/empty.
    * ``curvature`` → ``int`` if parseable, otherwise the raw string.
    * ``exclude`` → ``bool`` (default ``False`` when missing).
    * ``error`` → ``str`` or empty string.
    * ``filename`` → display name of the volume node (preserves
      provenance without depending on the source path; reload works even
      when the original folder is gone — issue #41).
    * ``original`` → ``None``. Callers that need the pixel array for
      gallery rebuilds must read it from the volume node directly via
      issue #41's reload path; this helper only covers the
      table-derivation contract.
    """
    length = _coerce_attr_float(node, ATTR_LENGTH)
    ratio = _coerce_attr_float(node, ATTR_RATIO)
    eye_area = _coerce_attr_float(node, ATTR_EYE_AREA)
    eye_diameter = _coerce_attr_float(node, ATTR_EYE_DIAMETER)

    # math.nan is the canonical "missing" sentinel in results_to_rows; mirror
    # it here so a run-after-analysis reload comparison is value-equal.
    def _nn(x):
        return None if isinstance(x, float) and math.isnan(x) else x

    exclude_raw = node.GetAttribute(ATTR_EXCLUDE) if hasattr(node, "GetAttribute") else None
    exclude_val = exclude_raw == "true" if exclude_raw in (None, "", "true", "false") else False

    error_raw = node.GetAttribute("ZebrafishAnalysis.error") if hasattr(node, "GetAttribute") else None
    error_val = error_raw if isinstance(error_raw, str) else ""

    name = node.GetName() if hasattr(node, "GetName") else ""

    return {
        "filename": name,
        "length": _nn(length),
        "curvature": _coerce_attr_int_or_str(node, ATTR_CURVATURE_CLASS),
        "ratio": _nn(ratio),
        "eye_area": _nn(eye_area),
        "eye_diameter": _nn(eye_diameter),
        "exclude": exclude_val,
        "error": error_val,
        # No ``original`` — see docstring.
    }


def volume_node_to_pixels(node):
    """Read the volume node's RGB pixel array (uint8 (H, W, 3)) or None.

    Used by the scene-reload path (#41) to rebuild gallery thumbnails
    without depending on the original source folder. Pixel data is stored
    in Slicer's scene (.mrb) so this works even when the original image
    files have been moved or deleted.

    The returned array is the unflipped visual image — i.e. the inverse of
    the flipud/fliplr transform applied by :func:`update_image_node` on
    write. Storing the visual-orientation numpy array here keeps the
    gallery widget's ``_load_originals`` path unchanged.

    Returns ``None`` for nodes that do not expose image data (empty nodes,
    stubs, or wrong-type nodes). The caller treats ``None`` as "no
    thumbnail available" rather than an error.
    """
    import numpy as np
    import vtk

    try:
        image_data = node.GetImageData() if hasattr(node, "GetImageData") else None
    except Exception:
        image_data = None
    if image_data is None:
        return None
    try:
        scalars = image_data.GetPointData().GetScalars()
    except Exception:
        return None
    if scalars is None:
        return None
    try:
        from vtk.util import numpy_support
        arr = numpy_support.vtk_to_numpy(scalars)
    except Exception:
        return None
    try:
        dims = image_data.GetDimensions()
    except Exception:
        dims = None
    if dims is None or len(dims) < 2 or dims[0] <= 0 or dims[1] <= 0:
        return None
    w_vtk, h_vtk = int(dims[0]), int(dims[1])
    try:
        comps = int(scalars.GetNumberOfComponents())
    except Exception:
        comps = 1
    try:
        flat = arr.reshape(h_vtk * w_vtk, comps)
        rgb_vtk = flat[:, :3].reshape(h_vtk, w_vtk, 3)
        # Inverse of update_image_node's flipud+fliplr: bottom-left-origin
        # back to top-left-origin, radiological back to anatomical left.
        rgb_visual = np.flipud(np.fliplr(rgb_vtk)).copy()
        return np.ascontiguousarray(rgb_visual, dtype=np.uint8)
    except Exception:
        return None


def validate_volume_node(node):
    """Return ``("", "")`` if the node has the expected attribute set,
    otherwise a descriptive ``(error_field, error_message)`` tuple.

    Used by the scene-reload path to surface per-image robustness:
    - tracked but never analyzed (no ``ZebrafishAnalysis.*`` attributes at
      all — issue #38's eager per-image loading tracks a volume node
      before "Run Analysis" is ever clicked) → ``("", "")``, i.e. not an
      error. The row reconstructs with metrics as ``None``/not-computed,
      same as right after folder-load, ready for the run queue.
    - was analyzed but its segmentation node reference is missing or
      broken → ``("Segmentation node missing", "")``.

    Issue #41 acceptance: a row with a broken segmentation reference must
    auto-exclude via the existing error-row mechanism rather than crash
    the module.

    Issue #56 follow-up: a *dangling* reference — the role is set but the
    referenced id no longer resolves in the scene (the user deleted the
    segmentation in the Data module) — is now also surfaced as a
    recoverable error instead of silently being treated as healthy. This
    matches the "Data module is ground truth" rule: a row whose seg was
    removed must auto-exclude so a future analysis re-run sees the volume
    as needing a fresh segmentation, not as already-attached.
    """
    if node is None:
        return ("Missing volume node", "")
    if not hasattr(node, "GetAttribute"):
        return ("Invalid volume node", "")
    # ATTR_EXCLUDE is always written (as "true" or "false") the moment
    # analysis runs once for this node — see _write_metric_attributes.
    # Its absence means analysis was never attempted, which is a normal
    # pending state, not an error.
    try:
        was_analyzed = node.GetAttribute(ATTR_EXCLUDE) is not None
    except Exception:
        was_analyzed = False
    if not was_analyzed:
        return ("", "")

    # segmentation reference must resolve (otherwise the metrics may be
    # stale relative to a deleted segmentation node).
    seg_id = None
    try:
        if hasattr(node, "GetNodeReferenceID"):
            seg_id = node.GetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION)
    except Exception:
        seg_id = None
    if not seg_id:
        # Not an error per se — could be a half-finished image where
        # analysis set metrics but seg-node attachment failed. Surface as a
        # specific recoverable error so the user can decide.
        return ("Segmentation node missing", "")

    # Issue #56 follow-up: resolve the id against the live scene. If the
    # user deleted the segmentation node in the Data module, the role on
    # the volume node still holds the now-dangling id — a naive check
    # accepts that as healthy and the next analysis silently re-attaches
    # the freshly-created seg, resurrecting the node. Resolve against
    # ``slicer.mrmlScene`` (best-effort; the test fakes carry the role but
    # no real scene, so a scene-resolution failure is treated as a
    # warning, not an auto-exclude).
    seg_live = None
    try:
        import slicer  # lazy: tests never import slicer
        scene = getattr(slicer, "mrmlScene", None)
        if scene is not None:
            seg_live = scene.GetNodeByID(seg_id)
    except Exception:
        seg_live = None
    if scene is not None and seg_live is None:
        return ("Segmentation node missing", "")
    return ("", "")


def volume_node_to_result_dict_with_validation(node):
    """Like :func:`volume_node_to_result_dict`, plus a robustness check.

    If :func:`validate_volume_node` returns a non-empty error, the
    returned dict's ``error`` key is set to that message and ``exclude``
    is forced to ``True`` (the existing error-row auto-exclude will catch
    it). The original ``error`` field on the attribute — if any — is
    overwritten, because validator errors take precedence over the
    "Could not read image." error from the pipeline.
    """
    row = volume_node_to_result_dict(node)
    err_field, _msg = validate_volume_node(node)
    if err_field:
        row["error"] = err_field
        row["exclude"] = True
    return row


# ---------------------------------------------------------------------------
# Issue #42: Segment Editor staleness flag
# ---------------------------------------------------------------------------
#
# A per-image segmentation node's ``ModifiedEvent`` triggers the cheap
# bookkeeping in :func:`mark_volume_node_stale` so the volume node's
# ``ZebrafishAnalysis.stale`` attribute is set immediately on every
# brush stroke (synchronous, no recomputation). The user is then asked on
# every module re-entry whether to recompute (policy in widget.py).

# Standard user-facing error message for stale rows; centralised here so
# the wording stays consistent between the auto-exclude flow and the
# detail-view helper text.
STALE_ERROR_MESSAGE = "Segmentation modified — recompute needed"


def mark_volume_node_stale(volume_node):
    """Set the stale attribute and force ``exclude`` + a stable error message.

    Cheap — no recomputation, no model call. Safe to call from an
    observer that fires many times per brush stroke (issue #42 explicit
    requirement: the observer must not trigger a perceptible per-stroke
    delay).

    The error message is set via ``ZebrafishAnalysis.error`` so it
    surfaces in the existing error-row auto-exclude path that #41 uses
    for scene-reload robustness — no new schema column.
    """
    if volume_node is None or not hasattr(volume_node, "SetAttribute"):
        return
    try:
        volume_node.SetAttribute(ATTR_STALE, "true")
    except Exception:
        return
    # The exclude + error set is what makes the row visibly stale in the
    # gallery/results table. The user can still see the metric values
    # (preserved per the "Existing metric values are not deleted, only
    # flagged" decision).
    try:
        volume_node.SetAttribute(ATTR_EXCLUDE, "true")
    except Exception:
        pass
    try:
        volume_node.SetAttribute(ATTR_PREFIX + "error", STALE_ERROR_MESSAGE)
    except Exception:
        pass


def segment_labelmap_mtimes(seg_node):
    """Return ``{segment_id: labelmap MTime}`` for every segment on ``seg_node``.

    This is the only reliable "did the user actually edit this segmentation"
    signal. Painting in the Segment Editor changes the voxels inside each
    segment's ``vtkOrientedImageData`` binary-labelmap representation, and
    that modification does **not** propagate upwards: measured against a live
    scene, a brush stroke left ``segNode.GetMTime()``,
    ``segNode.GetSegmentation().GetMTime()`` and ``segment.GetMTime()`` all
    unchanged while only the representation's MTime advanced (issue #81).
    Slicer's own ``segNode.GetMTime()`` does advance for display-node and
    representation-conversion work during scene import, i.e. exactly the
    events that must *not* count as user edits — so comparing node MTimes
    gets the answer backwards in both directions.

    Mirrors what Slicer does internally in
    ``qt-scripted-modules/SegmentEditorEffects/AbstractScriptedSegmentEditorAutoCompleteEffect.py``
    (``selectedSegmentModifiedTimes``), which keeps a per-segment-ID map of
    exactly this MTime to decide whether a segment really changed.

    Note that Slicer may keep several segments in one shared labelmap, so
    editing Eye can advance Body's MTime too. For "any edit invalidates this
    row's metrics" that is correct, if slightly coarse.

    Returns an empty dict when the node, its segmentation or the Slicer
    runtime is unavailable — callers treat "no readings" as "cannot tell",
    never as "changed". Never raises.
    """
    if seg_node is None:
        return {}
    try:
        import slicer  # lazy: tests never import slicer
    except Exception:
        return {}
    try:
        segmentation = seg_node.GetSegmentation()
    except Exception:
        return {}
    if segmentation is None:
        return {}
    try:
        representation_name = (
            slicer.vtkSegmentationConverter
            .GetSegmentationBinaryLabelmapRepresentationName()
        )
    except Exception:
        return {}
    try:
        segment_ids = list(segmentation.GetSegmentIDs())
    except Exception:
        return {}

    mtimes = {}
    for segment_id in segment_ids:
        try:
            segment = segmentation.GetSegment(segment_id)
        except Exception:
            continue
        if segment is None:
            continue
        try:
            labelmap = segment.GetRepresentation(representation_name)
        except Exception:
            continue
        if labelmap is None:
            continue
        try:
            mtimes[segment_id] = int(labelmap.GetMTime())
        except Exception:
            continue
    return mtimes


def is_volume_node_stale(volume_node):
    """Return True if the volume node's ``ATTR_STALE`` attribute is "true"."""
    if volume_node is None or not hasattr(volume_node, "GetAttribute"):
        return False
    try:
        return volume_node.GetAttribute(ATTR_STALE) == "true"
    except Exception:
        return False


def clear_volume_node_stale(volume_node):
    """Clear the stale flag after a successful recompute.

    Does NOT auto-clear ``error`` / ``exclude`` — the widget decides
    whether the user is still excluded (they may have manually excluded
    before the recompute). The recompute function explicitly resets
    both via :func:`write_metric_attributes` and the widget's exclude set.
    """
    if volume_node is None or not hasattr(volume_node, "RemoveAttribute"):
        return
    try:
        volume_node.RemoveAttribute(ATTR_STALE)
    except Exception:
        pass


def volume_nodes_to_results(volume_nodes):
    """Map a list of volume nodes to the canonical results list shape.

    Order is preserved — the caller is responsible for passing the same
    order both after a fresh run (insertion order of the loaded folder)
    and after a scene reload (the ``ROLE_ZEBRAFISH_IMAGES`` reference
    order on the parameter node).
    """
    return [volume_node_to_result_dict(n) for n in volume_nodes]


def volume_nodes_to_rows(volume_nodes):
    """Single conversion path used by both the run-after-analysis table
    build and the scene-reload table build (issue #41).

    Returns rows in the same format as :func:`results_to_rows` so the
    downstream ``build_vtk_table`` / ``populate_table_node`` pipeline is
    untouched. This is the function that satisfies the "single code path"
    acceptance criterion in issue #40.
    """
    return results_to_rows(volume_nodes_to_results(volume_nodes))


def _node_reference_ids(node, role):
    """Return every reference ID under ``role`` on ``node``, in order.

    ``GetNodeReferenceIDs(role)`` (plural) is not exposed by the Python
    binding on ``vtkMRMLNode`` in this Slicer build — only
    ``GetNumberOfNodeReferences(role)`` and ``GetNthNodeReferenceID(role, n)``
    are. Every multi-value reference list (``ROLE_ZEBRAFISH_IMAGES``) must
    go through this helper instead of calling the plural getter directly.
    """
    if node is None or not hasattr(node, "GetNumberOfNodeReferences"):
        return []
    try:
        n = node.GetNumberOfNodeReferences(role)
    except Exception:
        return []
    ids = []
    for i in range(n):
        try:
            nid = node.GetNthNodeReferenceID(role, i)
        except Exception:
            continue
        if nid:
            ids.append(nid)
    return ids


def list_tracked_volume_nodes(param_node, scene):
    """Return the volume nodes currently registered on ``param_node`` under
    :data:`ROLE_ZEBRAFISH_IMAGES`, in insertion order.

    The order matters: ``volume_nodes_to_rows`` produces a row per node in
    the order received, and the table's Filename column inherits each
    node's display name. Issue #38 sets up the references in folder-load
    order; #41 must reproduce the same order from the scene after reload.

    Looks up each ID through ``scene.GetNodeByID``. Missing IDs (e.g. a
    node the user deleted from the Data module between load and run) are
    silently skipped — they will surface as a missing row in the table
    rather than crash the build.

    Result is then sorted by each node's ``ZebrafishAnalysis.loadOrder``
    attribute (set at eager-creation time, see :func:`create_image_volume_node`)
    rather than trusting the NodeReference array's own enumeration order —
    a Save Scene -> Load Scene round-trip was observed to come back with a
    different gallery/table ordering even though every node and reference
    was still present (found while testing #61). Nodes without the
    attribute (e.g. older scenes, test fakes) keep their relative
    reference-order position, sorted after any attributed nodes.
    """
    if param_node is None or scene is None:
        return []
    ids = _node_reference_ids(param_node, ROLE_ZEBRAFISH_IMAGES)
    out = []
    for nid in ids:
        if not nid:
            continue
        try:
            node = scene.GetNodeByID(nid)
        except Exception:
            node = None
        if node is None:
            continue
        try:
            if hasattr(node, "IsA") and not node.IsA("vtkMRMLVolumeNode"):
                continue
        except Exception:
            continue
        out.append(node)

    def _load_order_key(node):
        try:
            raw = node.GetAttribute(ATTR_LOAD_ORDER)
            return (0, int(raw)) if raw is not None else (1, 0)
        except Exception:
            return (1, 0)

    out.sort(key=_load_order_key)
    return out


def find_zebrafish_volume_nodes_in_scene(scene):
    """Return every volume node in ``scene`` that this module created, in
    scene enumeration order.

    Discovery is by :data:`ATTR_LOAD_ORDER`, stamped on each image node at
    eager-creation time (issue #38). Unlike the parameter node's reference
    list this survives a scene *merge*: the parameter node is a singleton, so
    importing a second saved scene copies the imported one onto the existing
    node and replaces the reference list wholesale, while the volume nodes of
    both scenes remain in the scene side by side (issue #36).

    Foreign volumes (a CT the user loaded separately, another module's
    output) never carry the attribute and are skipped.

    Never raises — a scene stub without the class-enumeration API yields an
    empty list.
    """
    if scene is None:
        return []
    try:
        count = int(scene.GetNumberOfNodesByClass("vtkMRMLVolumeNode"))
    except Exception:
        return []
    out = []
    for i in range(count):
        try:
            node = scene.GetNthNodeByClass(i, "vtkMRMLVolumeNode")
        except Exception:
            continue
        if node is None:
            continue
        try:
            if node.GetAttribute(ATTR_LOAD_ORDER) is None:
                continue
        except Exception:
            continue
        out.append(node)
    return out


def reconcile_tracked_volume_nodes(param_node, scene, known_ids=None):
    """Re-register Zebrafish volume nodes present in ``scene`` but missing
    from ``param_node``'s :data:`ROLE_ZEBRAFISH_IMAGES` list.

    Loading a scene on top of an open session is additive in Slicer: the
    imported volume nodes join the ones already there. The parameter node is
    the exception — being a singleton, MRML copies the imported node onto the
    existing one, which replaces the reference list instead of extending it.
    The previous session's images then still exist in full (segmentations,
    metrics, markups) but nothing references them, so the module shows only
    the imported batch while the Data module shows both (issue #36).

    ``known_ids`` is the ordered list of tracked node IDs captured *before*
    the import (the widget's ``StartImportEvent`` handler). Nodes listed
    there keep their position at the front, so existing images stay put and
    the imported batch appends — matching what the Data module does. Without
    a snapshot (the merge happened before the module was ever opened, so no
    observer was armed) the current references come first and the orphans
    are appended.

    :data:`ATTR_LOAD_ORDER` is renumbered across the whole union afterwards,
    because each scene numbers its own images from zero and
    :func:`list_tracked_volume_nodes` sorts on that attribute — leaving the
    duplicates in place would interleave the two batches.

    Returns the list of node IDs newly registered — empty in the normal
    non-merge case, where the references already cover every discovered
    node. Never raises.
    """
    if param_node is None or scene is None:
        return []

    tracked_ids = _node_reference_ids(param_node, ROLE_ZEBRAFISH_IMAGES)
    discovered = {}
    for node in find_zebrafish_volume_nodes_in_scene(scene):
        try:
            nid = node.GetID()
        except Exception:
            continue
        if nid:
            discovered[nid] = node

    tracked_set = set(tracked_ids)
    if not [nid for nid in discovered if nid not in tracked_set]:
        return []

    def _resolve(nid):
        node = discovered.get(nid)
        if node is not None:
            return node
        try:
            return scene.GetNodeByID(nid)
        except Exception:
            return None

    ordered = []
    seen = set()
    for nid in list(known_ids or []) + tracked_ids + list(discovered):
        if not nid or nid in seen:
            continue
        if _resolve(nid) is None:
            continue
        seen.add(nid)
        ordered.append(nid)

    added = []
    for nid in ordered:
        if nid in tracked_set:
            continue
        try:
            param_node.AddNodeReferenceID(ROLE_ZEBRAFISH_IMAGES, nid)
        except Exception:
            continue
        added.append(nid)

    for position, nid in enumerate(ordered):
        node = _resolve(nid)
        try:
            node.SetAttribute(ATTR_LOAD_ORDER, str(position))
        except Exception:
            pass

    return added


def find_tracked_volume_node_by_filename(param_node, scene, filename):
    """Return the tracked volume node whose display name equals ``filename``.

    Issue #56: replaces the singleton "CurrentImage" lookup. Each image
    in a batch has its own ``vtkMRMLVectorVolumeNode`` registered under
    ``ROLE_ZEBRAFISH_IMAGES`` (issue #38) whose display name was set to
    the basename at eager-creation time — matching the result dict's
    ``filename`` key. Returns ``None`` if no match is found, the
    parameter node / scene is unavailable, or ``filename`` is falsy.

    O(n) over the tracked list is fine for typical gallery sizes (a few
    dozen images per run).
    """
    if not filename or param_node is None or scene is None:
        return None
    for node in list_tracked_volume_nodes(param_node, scene):
        try:
            if node.GetName() == filename:
                return node
        except Exception:
            continue
    return None


def set_slice_viewer_background(volume_node):
    """Set ``volume_node`` as the background of all standard slice viewers.

    Issue #56: drives the slice-view live preview off the already-existing
    per-image volume node (issue #38), replacing the singleton "CurrentImage"
    mechanism. Best-effort — every step is guarded so a missing slicer,
    a stale node, or a binding mismatch never raises into the UI. Returns
    ``True`` on success, ``False`` otherwise.

    Must be called on the Slicer main thread (slicer.util touches MRML).
    """
    if volume_node is None:
        return False
    try:
        import slicer
    except Exception:
        return False
    try:
        slicer.util.setSliceViewerLayers(background=volume_node)
        return True
    except Exception:
        return False


def set_segmentation_visibility(seg_node, visible):
    """Toggle ``seg_node``'s display visibility.

    Issue #56: gallery selection toggles per-image segmentation visibility
    so the slice views show only the selected image's overlays (instead of
    stacking every previously-clicked image's segmentation). Returns
    ``True`` on success, ``False`` when the node is missing or has no
    display node. No-op on slicer / display-property failure.
    """
    if seg_node is None:
        return False
    try:
        display = seg_node.GetDisplayNode()
    except Exception:
        return False
    if display is None:
        return False
    try:
        display.SetVisibility(bool(visible))
        return True
    except Exception:
        return False


def image_geometry(h_orig: int, w_orig: int, um_per_px: float):
    """Return (dims, spacing, origin) for a vtkMRMLVectorVolumeNode.

    dims    = (w_orig, h_orig, 1)              VTK IJK order
    spacing = (um_per_px/1000, um_per_px/1000, 1.0)  mm, isotropic
    origin  = (0.0, 0.0, 0.0)

    Pure Python — no vtk or slicer imports. Testable with standard pytest.

    Raises ValueError for non-positive h_orig, w_orig, or um_per_px
    (including zero, negative, NaN, and inf).
    """
    if not isinstance(h_orig, int) or h_orig <= 0:
        raise ValueError(f"h_orig must be a positive integer, got {h_orig!r}")
    if not isinstance(w_orig, int) or w_orig <= 0:
        raise ValueError(f"w_orig must be a positive integer, got {w_orig!r}")
    if not math.isfinite(um_per_px) or um_per_px <= 0:
        raise ValueError(f"um_per_px must be finite and positive, got {um_per_px!r}")
    spacing_mm = um_per_px / 1000.0
    dims = (w_orig, h_orig, 1)
    spacing = (spacing_mm, spacing_mm, 1.0)
    origin = (0.0, 0.0, 0.0)
    return dims, spacing, origin


def update_image_node(image_rgb, um_per_px, node):
    """Write a uint8 RGB array into an existing vtkMRMLVectorVolumeNode.

    image_rgb must be uint8, shape (H, W, 3).
    um_per_px is the original-image physical scale in micrometers per pixel.
    result["spacing"] must NOT be used here (it is calibrated to 256x256 mask space).

    NOTE: _on_detect_scale / show_raw_image is out of scope for E2b.
    The MRML node intentionally reflects the last gallery selection, not the
    scalebar debug overlay.

    VTK step order:
      1. derive h_orig, w_orig from image_rgb.shape
      2. compute geometry via image_geometry()
      3. flipud + fliplr + copy (corrects VTK bottom-left origin and Slicer radiological convention)
      4. reshape and convert to VTK array (no AllocateScalars)
      5. build vtkImageData: SetDimensions, GetPointData().SetScalars()
      6. reset direction cosines to identity
      7. set spacing and origin on node (before SetAndObserveImageData)
      8. SetAndObserveImageData as final step
    """
    import vtk
    from vtk.util import numpy_support
    import numpy as np

    h_orig, w_orig = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"image_rgb must have shape (H, W, 3), got {image_rgb.shape}"
        )
    dims, spacing, origin = image_geometry(h_orig, w_orig, um_per_px)

    # flipud: numpy row 0 (image top) → VTK last row (visual top)
    # fliplr: compensates for Slicer's radiological convention (R axis → left of screen)
    # .copy() restores C-contiguity after the non-contiguous views
    flipped = np.flipud(np.fliplr(image_rgb)).copy()
    flat = flipped.reshape(-1, 3)

    vtk_array = numpy_support.numpy_to_vtk(
        flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_array.SetNumberOfComponents(3)
    vtk_array.SetName("ImageScalars")

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.GetPointData().SetScalars(vtk_array)  # no AllocateScalars

    identity = vtk.vtkMatrix4x4()
    node.SetIJKToRASDirectionMatrix(identity)
    node.SetSpacing(*spacing)
    node.SetOrigin(*origin)
    node.SetAndObserveImageData(image_data)  # final step — fires observers with complete geometry


def resample_mask_to_original(mask_2d, h_orig, w_orig):
    """Resample a 2-D mask to (h_orig, w_orig) with nearest-neighbour interpolation.

    Pure Python (cv2 + numpy only) — no vtk or slicer. Testable with standard pytest.

    Parameters
    ----------
    mask_2d : ndarray
        2-D array of any dtype. Values > 0 are treated as body / eye pixels.
    h_orig : int
        Target height in pixels.
    w_orig : int
        Target width in pixels.

    Returns
    -------
    ndarray
        uint8 array of shape (h_orig, w_orig) with values 0 or 1.
    """
    import cv2
    import numpy as np

    binary = (mask_2d > 0).astype(np.uint8)
    # cv2.resize expects (width, height)
    return cv2.resize(binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)


def update_segmentation_node(result, um_per_px, node, image_node=None, preserve_user_segments=False):
    """Write body and eye masks from a result dict into an existing vtkMRMLSegmentationNode.

    result["original"]: uint8 ndarray shape (H_orig, W_orig, 3).
    result["mask"]: 2-D ndarray shape (256, 256) — body mask (>0 means body).
    result["eye_mask"]: 2-D bool ndarray shape (256, 256) or None — eye mask.
    um_per_px: physical scale of the original image in micrometres per pixel.
    image_node: optional vtkMRMLVectorVolumeNode — used to set reference geometry
        so Slicer can position the segmentation in slice views.
    preserve_user_segments: when True, only refresh Body/Eye segments that
        already exist on ``node``. Do not call ``RemoveAllSegments()`` and do
        not add Body/Eye if the user removed them in the Segment Editor.
        Default ``False`` (legacy behaviour: full rebuild) so callers that
        always pass a fresh node are unaffected. The per-image batch path
        (``_create_segmentation_for_volume``) flips this to ``True`` for
        re-runs to respect the Data-module-is-ground-truth rule.

    VTK step order:
      1. Lazy-import vtk, numpy, slicer, vtkSegmentationCore inside function.
      2. Derive h_orig, w_orig from result["original"].shape; skip gracefully if absent.
      3. Resample body mask and eye mask (if applicable) to (h_orig, w_orig).
      4. Compute geometry via image_geometry(h_orig, w_orig, um_per_px).
      5. Build vtkOrientedImageData for each segment (uint8, values 0/1).
         - flipud + fliplr + ascontiguousarray to match VTK coordinate convention.
      6. Set master representation to binary labelmap.
      7. Wrap full modification in StartModify/EndModify to suppress intermediate events.
      8. When preserve_user_segments=False: node.GetSegmentation().RemoveAllSegments()
         (legacy full-rebuild behaviour).
      9. Add or refresh "Body" segment (green) — always when present in result;
         preserved mode only adds it when it already existed.
      10. Add or refresh "Eye" segment (red) — only when eye_mask is not None
         and eye_mask.any(); preserved mode only adds it when it already existed.
      11. Populate each segment via SetBinaryLabelmapToSegment.
      12. Set reference image geometry from image_node if provided.
    """
    import vtk
    from vtk.util import numpy_support
    import numpy as np
    import slicer
    import vtkSegmentationCore

    original = result.get("original") if result else None
    if original is None:
        return

    h_orig, w_orig = int(original.shape[0]), int(original.shape[1])
    dims, spacing, origin = image_geometry(h_orig, w_orig, um_per_px)
    spacing_mm = spacing[0]

    mask_2d = result.get("mask")
    eye_mask_2d = result.get("eye_mask")

    body_2d = resample_mask_to_original(mask_2d, h_orig, w_orig) if mask_2d is not None else None
    has_eye = (
        eye_mask_2d is not None
        and hasattr(eye_mask_2d, "any")
        and eye_mask_2d.any()
    )
    eye_2d = resample_mask_to_original(eye_mask_2d, h_orig, w_orig) if has_eye else None

    def _make_oriented_image(arr_2d):
        """Build a vtkOrientedImageData from a 2-D uint8 (0/1) array."""
        flipped = np.ascontiguousarray(np.flipud(np.fliplr(arr_2d)))
        flat = flipped.reshape(-1)
        vtk_array = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_array.SetNumberOfComponents(1)
        oid = vtkSegmentationCore.vtkOrientedImageData()
        oid.SetDimensions(w_orig, h_orig, 1)
        oid.GetPointData().SetScalars(vtk_array)
        oid.SetSpacing(spacing_mm, spacing_mm, 1.0)
        oid.SetOrigin(0.0, 0.0, 0.0)
        return oid

    node.GetSegmentation().SetSourceRepresentationName(
        slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    )

    was_modifying = node.StartModify()
    try:
        seg = node.GetSegmentation()

        # Data module is ground truth for the contents of a segmentation node:
        # if the user removed "Body" or "Eye" inside the Segment Editor, do
        # not silently add it back on the next analysis run. In legacy
        # (preserve_user_segments=False) mode — used when callers know they
        # hold a fresh node — wipe the slate first to match the prior
        # full-rebuild behaviour.
        existing_segments = set()
        try:
            for s in seg.GetSegments():
                existing_segments.add(s)
        except Exception:
            existing_segments = set()

        if not preserve_user_segments:
            seg.RemoveAllSegments()
            existing_segments = set()

        if body_2d is not None and (not preserve_user_segments or "Body" in existing_segments):
            if preserve_user_segments and "Body" in existing_segments:
                body_id = "Body"
            else:
                body_id = seg.AddEmptySegment("Body", "Body", [0.0, 1.0, 0.0])
            slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
                _make_oriented_image(body_2d), node, body_id
            )

        if eye_2d is not None and (not preserve_user_segments or "Eye" in existing_segments):
            if preserve_user_segments and "Eye" in existing_segments:
                eye_id = "Eye"
            else:
                eye_id = seg.AddEmptySegment("Eye", "Eye", [1.0, 0.0, 0.0])
            slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
                _make_oriented_image(eye_2d), node, eye_id
            )

        if image_node is not None:
            node.SetReferenceImageGeometryParameterFromVolumeNode(image_node)
    finally:
        node.EndModify(was_modifying)


# ---------------------------------------------------------------------------
# Per-image volume node batch helpers (issue #38)
# ---------------------------------------------------------------------------

def _populate_image_node(image_rgb, um_per_px, node):
    """Thin wrapper around update_image_node used by create_image_volume_node.

    Patching this in tests lets the rest of the create-image logic be
    exercised without bringing up VTK.
    """
    update_image_node(image_rgb, um_per_px, node)


def create_image_volume_node(image_rgb, um_per_px, name_hint, param_node, scene):
    """Create one new ``vtkMRMLVectorVolumeNode`` for an eagerly-loaded image.

    Unlike the retired :func:`get_or_create_image_node` (singleton
    ``CurrentImage`` removed in issue #56), this function NEVER reuses an
    existing node — each successful image gets its own persistent volume node
    created at folder-load time, before "Run Analysis" is clicked.

    Parameters
    ----------
    image_rgb : numpy.ndarray
        ``uint8`` array of shape ``(H, W, 3)`` already loaded by the widget's
        pre-flight readability check; this function does not read the file
        from disk again.
    um_per_px : float
        Physical scale (micrometres per pixel) used for spacing metadata.
    name_hint : str
        Suggested display name for the node (typically the basename).
    param_node : vtkMRMLScriptedModuleNode
        Module parameter node that owns the batch reference list.
    scene : vtkMRMLScene
        Active MRML scene.

    Returns
    -------
    vtkMRMLVectorVolumeNode
        The newly created node.

    Notes
    -----
    The new node ID is appended to the reference list under
    ``ROLE_ZEBRAFISH_IMAGES`` via ``AddNodeReferenceID``. If image population
    fails, the half-constructed node is removed from the scene to avoid
    leaving an empty/orphan volume node visible in the Data module.
    """
    display_name = name_hint if name_hint else "ZebrafishEmbryoAnalyzer Image"
    node = scene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", display_name)
    try:
        _populate_image_node(image_rgb, um_per_px, node)
    except Exception:
        # Roll back the half-built node so the Data module does not advertise
        # an empty volume that no GUI state references.
        try:
            scene.RemoveNode(node)
        except Exception:
            pass
        raise

    # Stamp the batch position as a plain node attribute — unlike
    # NodeReference array order, attribute strings are known-reliable
    # through this codebase's scene save/reload path (the ZebrafishAnalysis.*
    # metric/staleness attributes already round-trip correctly), so
    # list_tracked_volume_nodes can restore folder-load order after a
    # Save Scene -> Load Scene round-trip even if the reference list itself
    # comes back reordered.
    try:
        load_order = param_node.GetNumberOfNodeReferences(ROLE_ZEBRAFISH_IMAGES)
        node.SetAttribute(ATTR_LOAD_ORDER, str(load_order))
    except Exception:
        pass
    param_node.AddNodeReferenceID(ROLE_ZEBRAFISH_IMAGES, node.GetID())
    return node


def remove_all_image_volume_nodes(param_node, scene):
    """Remove every volume node tracked under ``ROLE_ZEBRAFISH_IMAGES``.

    Cleans up anything reachable from those volume nodes through their node
    references, so the recursive cleanup stays correct after sub-issue #39
    adds segmentation / markups references on the volume node. Today the
    recursion is a no-op (volume nodes currently own no children) but the
    structure is in place.

    Parameters
    ----------
    param_node : vtkMRMLScriptedModuleNode | None
        Module parameter node. ``None`` is a no-op.
    scene : vtkMRMLScene | None
        Active MRML scene. ``None`` is a no-op.

    Returns
    -------
    int
        Number of top-level volume nodes removed from the scene.
    """
    if param_node is None or scene is None:
        return 0

    ids_snapshot = _node_reference_ids(param_node, ROLE_ZEBRAFISH_IMAGES)
    if not ids_snapshot:
        return 0

    removed = 0
    for nid in ids_snapshot:
        node = scene.GetNodeByID(nid)
        if node is None:
            continue
        _recursively_remove(node, scene)
        removed += 1

    # The reference list no longer points at live nodes. Clearing it keeps
    # the parameter node consistent across Slicer restarts (which currently
    # do not see #36 scene-merge behaviour, per issue #38 out-of-scope).
    # ``RemoveAllNodeReferenceIDs`` exists in vtkMRMLNode and clears the list
    # atomically, so we prefer it over the legacy ``RemoveNodeReferenceIDs``
    # list-form path whose argument type varies across Slicer/VTK bindings.
    if hasattr(param_node, "RemoveAllNodeReferenceIDs"):
        try:
            param_node.RemoveAllNodeReferenceIDs(ROLE_ZEBRAFISH_IMAGES)
        except Exception:
            _remove_node_reference_ids_fallback(param_node, ROLE_ZEBRAFISH_IMAGES, ids_snapshot)
    else:
        _remove_node_reference_ids_fallback(param_node, ROLE_ZEBRAFISH_IMAGES, ids_snapshot)

    return removed


def _remove_node_reference_ids_fallback(param_node, role, ids_snapshot):
    """Best-effort fallback for bindings that lack ``RemoveAllNodeReferenceIDs``."""
    try:
        param_node.RemoveNodeReferenceIDs(role, ids_snapshot)
        return
    except TypeError:
        # Single-ID signature in older bindings: remove one at a time.
        for nid in ids_snapshot:
            try:
                param_node.RemoveNodeReferenceIDs(role, nid)
            except Exception:
                pass
    except Exception:
        # Nodes have already been removed from the scene; nothing to do.
        pass


def _recursively_remove(node, scene):
    """Remove ``node`` and every node it references from ``scene``.

    Children are snapshot before their parent is removed, because
    ``RemoveNode`` invalidates references. ``scene.GetNodeByID`` may return
    ``None`` between successive removals; missing nodes are skipped.
    """
    seen = set()

    def _visit(n):
        nid = n.GetID() if hasattr(n, "GetID") else None
        if nid is None or nid in seen:
            return
        seen.add(nid)
        # Snapshot first — RemoveNode may invalidate references on `n`.
        child_ids = _collect_node_reference_ids(n)
        for cid in child_ids:
            child = scene.GetNodeByID(cid) if hasattr(scene, "GetNodeByID") else None
            if child is not None:
                _visit(child)
        if hasattr(scene, "RemoveNode"):
            scene.RemoveNode(n)

    _visit(node)


def _collect_node_reference_ids(node):
    """Return every node-reference ID carried by ``node`` across all roles.

    ``GetNodeReferenceRoles`` fills a ``std::vector<std::string>`` out-param
    (pass a plain Python list); each role is then resolved through
    :func:`_node_reference_ids`, since the plural per-role getter is not
    exposed by the Python binding either.
    """
    if node is None or not hasattr(node, "GetNodeReferenceRoles"):
        return []
    roles = []
    try:
        node.GetNodeReferenceRoles(roles)
    except Exception:
        return []
    ids = []
    for role in roles:
        ids.extend(_node_reference_ids(node, role))
    return ids


# ---------------------------------------------------------------------------
# Per-image analysis streaming helpers (issue #39, ADR 0001)
# ---------------------------------------------------------------------------
#
# analyse_images() in ZebrafishEmbryoAnalyzerLib.logic invokes
# ``apply_analysis_to_volume_node`` once per completed image, so segmentation
# nodes, markups nodes, and metric attributes are streamed into the MRML
# scene as soon as each result is ready — not batched at the end. A Cancel
# mid-batch therefore leaves fully-formed nodes + attributes for every image
# processed so far.
#
# Attribute namespace: every metric is stored on the volume node under the
# ``ZebrafishAnalysis.`` prefix. Downstream code (results table derivation in
# sub-issue #40, segment-editor staleness in sub-issue #5) reads them back
# via ``volumeNode.GetAttribute("ZebrafishAnalysis.<name>")``.
#
# Node-reference roles: the segmentation / markups nodes are attached to the
# volume node via ``volumeNode.SetNodeReferenceID(role, node.GetID())`` —
# the per-image segmentation was historically mirrored into a singleton
# ``ROLE_CURRENT_SEGMENTATION`` node on the parameter node, but issue #56
# retired that mechanism in favour of these per-image references directly.
# The volume node therefore owns all per-image children, which makes
# ``remove_all_image_volume_nodes`` recursive cleanup correct (children are
# reachable through the volume node's references).


def _format_attr(value) -> str:
    """Format a metric value as a string for ``SetAttribute``.

    Floats are written with full precision so the round-trip is lossless;
    integers (curvature class) keep their natural form. ``None`` becomes
    empty string so ``GetAttribute`` returns ``None`` rather than the literal
    string "None" — distinguishing "not computed" from any real value.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _write_metric_attributes(result, volume_node):
    """Write metric attributes onto ``volume_node`` under the ``ZebrafishAnalysis.`` prefix.

    Attribute catalogue (see ADR 0001):

    * ``ZebrafishAnalysis.length``        — float µm or "" if not computed
    * ``ZebrafishAnalysis.curvature_class`` — int class id (string form) or ""
    * ``ZebrafishAnalysis.ratio``         — float length/straight or ""
    * ``ZebrafishAnalysis.eye_area``      — float µm² or ""
    * ``ZebrafishAnalysis.eye_diameter``  — float µm or ""
    * ``ZebrafishAnalysis.exclude``       — "true" / "false" (always present;
      defaults to "false" when the key is missing on the result dict)

    All attributes are written unconditionally so the reader can distinguish
    "the value is the empty string because length was disabled" from "the
    attribute is missing because analysis was never run". ``SetAttribute`` is
    a no-op when the value is already present and identical, so repeat writes
    do not bump the volume node's MTime.
    """
    if volume_node is None or not hasattr(volume_node, "SetAttribute"):
        return
    exclude_raw = result.get("exclude", False)
    exclude_val = bool(exclude_raw) if exclude_raw is not None else False
    volume_node.SetAttribute(ATTR_LENGTH, _format_attr(result.get("length")))
    volume_node.SetAttribute(
        ATTR_CURVATURE_CLASS, _format_attr(result.get("curvature"))
    )
    volume_node.SetAttribute(ATTR_RATIO, _format_attr(result.get("ratio")))
    volume_node.SetAttribute(ATTR_EYE_AREA, _format_attr(result.get("eye_area")))
    volume_node.SetAttribute(
        ATTR_EYE_DIAMETER, _format_attr(result.get("eye_diameter"))
    )
    volume_node.SetAttribute(ATTR_EXCLUDE, "true" if exclude_val else "false")


def _set_node_reference(volume_node, role, child_node):
    """Attach ``child_node`` to ``volume_node`` under ``role`` (no-op if either is None).

    Used by the segmentation, markups-line, and markups-curve attach helpers.
    Each per-image helper writes to a *different* role string, so duplicate
    roles only ever arise when the same helper is called twice (e.g. a
    re-run reusing the same per-image segmentation — see
    :func:`_create_segmentation_for_volume`). For per-image roles we want a
    single, replaceable reference: ``SetNodeReferenceID`` collapses
    re-attachment to the same child id, and when the child id has changed
    (e.g. the user deleted the seg and the helper created a new one) it
    replaces the dangling reference cleanly. ``AddNodeReferenceID`` would
    accumulate duplicate ids across runs, eventually orphaning one entry.
    Falls back to ``AddNodeReferenceID`` only when ``SetNodeReferenceID``
    is unavailable on this MRML node type — purely defensive.
    """
    if volume_node is None or child_node is None:
        return
    nid = child_node.GetID() if hasattr(child_node, "GetID") else None
    if nid is None:
        return
    if hasattr(volume_node, "SetNodeReferenceID"):
        try:
            volume_node.SetNodeReferenceID(role, nid)
            return
        except Exception:
            pass
    if hasattr(volume_node, "AddNodeReferenceID"):
        try:
            volume_node.AddNodeReferenceID(role, nid)
        except Exception:
            pass


def _reparent_in_subject_hierarchy(scene, child_node, parent_node):
    """Nest ``child_node`` under ``parent_node`` in the Data module's tree view.

    Node references (``SetNodeReferenceID``/``AddNodeReferenceID``) do not
    affect Subject Hierarchy placement — every new node defaults to a child
    of the scene root item. Segment Editor achieves "segmentation nested
    under its source volume" by explicitly reparenting the SH item once the
    segmentation is created; this replicates that for the streamed nodes.
    """
    if scene is None or child_node is None or parent_node is None:
        return
    try:
        import slicer  # lazy: tests never import slicer
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(scene)
        if sh_node is None:
            return
        parent_item = sh_node.GetItemByDataNode(parent_node)
        child_item = sh_node.GetItemByDataNode(child_node)
        if parent_item and child_item:
            sh_node.SetItemParent(child_item, parent_item)
    except Exception:
        pass


def _get_existing_seg_for_volume(volume_node, scene):
    """Return the segmentation node already attached to ``volume_node``, if any.

    The Data module is ground truth: a node the user deletes in the Data
    module must not be silently recreated by a later analysis run. This
    helper resolves the per-image ``ROLE_ZEBRAFISH_SEGMENTATION`` reference
    against ``scene`` and returns the live ``vtkMRMLSegmentationNode`` when
    it still exists, otherwise ``None``. Callers (e.g.
    :func:`_create_segmentation_for_volume`) decide whether to reuse the
    existing node or create a new one.

    Returns ``None`` on any failure — the volume node being torn down, the
    reference role missing, or the referenced id not resolving. Never
    raises; callers always fall back to "create new".
    """
    if volume_node is None or scene is None:
        return None
    seg_id = None
    try:
        if hasattr(volume_node, "GetNodeReferenceID"):
            seg_id = volume_node.GetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION)
    except Exception:
        seg_id = None
    if not seg_id:
        return None
    try:
        return scene.GetNodeByID(seg_id)
    except Exception:
        return None


def _create_segmentation_for_volume(result, volume_node, scene, um_per_px):
    """Create one segmentation node for ``volume_node`` and attach via ``ROLE_ZEBRAFISH_SEGMENTATION``.

    Reuses an existing segmentation node attached to ``volume_node`` when one
    is still in the scene — see :func:`_get_existing_seg_for_volume`. The
    Data module is ground truth: deleting a segmentation in the Data module
    drops the reference, so the next call sees no existing node and creates
    a fresh one. Re-running analysis on the same volume also reuses the
    existing node, refreshing the segments it still has while leaving any
    user-added segments untouched (see ``preserve_user_segments`` on
    :func:`update_segmentation_node`).

    Returns the (reused or newly-created) segmentation node, or ``None``
    when the result carries no image (decoding failure / error row) — callers
    must tolerate ``None``.
    """
    if volume_node is None or scene is None:
        return None
    original = result.get("original") if result else None
    if original is None:
        return None
    import slicer  # lazy: tests never import slicer
    seg_node = _get_existing_seg_for_volume(volume_node, scene)
    created_here = False
    if seg_node is None:
        seg_node = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetName(_seg_display_name(result, volume_node))
        created_here = True
    # Preserve any segments the user added (or kept) inside the existing
    # node — only the Body/Eye segments from this analysis result are
    # overwritten. A freshly-created node has no prior segments so the
    # preserve path is identical to a full rebuild.
    update_segmentation_node(
        result,
        um_per_px,
        seg_node,
        image_node=volume_node,
        preserve_user_segments=not created_here,
    )
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_SEGMENTATION, seg_node)
    if created_here:
        _reparent_in_subject_hierarchy(scene, seg_node, volume_node)
        # Every per-image segmentation is created hidden — without this, a
        # multi-image batch shows every segmentation stacked on top of each
        # other in the slice view regardless of which volume is the current
        # background (Slicer doesn't tie segmentation visibility to the
        # background volume by default). Users toggle visibility per-node
        # in the Data module's eye icon as needed.
        display = seg_node.GetDisplayNode() if hasattr(seg_node, "GetDisplayNode") else None
        if display is not None:
            try:
                display.SetVisibility(False)
            except Exception:
                pass
    return seg_node


def _seg_display_name(result, volume_node):
    """Return a display name for the segmentation node derived from the volume node's name.

    Falls back to the result filename when the volume node lacks ``GetName``.
    Keeps the Data module readable when many images are loaded.
    """
    base = None
    if hasattr(volume_node, "GetName"):
        try:
            base = volume_node.GetName()
        except Exception:
            base = None
    if not base:
        base = (result or {}).get("filename") or "ZebrafishEmbryoAnalyzer Segmentation"
    return f"{base} Segmentation"


def _create_markups_line_for_volume(result, volume_node, scene):
    """Create a MarkupsLineNode with Head/Tail control points when length was computed.

    Skipped when ``result["length"]`` is ``None`` (length disabled) or when
    ``result["straight_line_points"]`` is unavailable — mirrors the
    "segment only when available" pattern. The line uses the same two
    endpoints as the persisted manual-correction target and matches
    ``overlay._STRAIGHT_CLR`` so the real MRML view matches the Detail tab.

    Reuses an existing line node already attached to ``volume_node`` when
    one is in the scene (see :func:`_get_existing_markup_for_volume`) — a
    re-run refreshes the two endpoints in place instead of stacking a
    duplicate node on top.

    Returns the new (or reused) node, or ``None`` when skipped.
    """
    if volume_node is None or scene is None:
        return None
    if result.get("length") is None:
        return None
    sl_pts = result.get("straight_line_points")
    if sl_pts is None:
        return None
    import slicer  # lazy: tests never import slicer
    line = _get_existing_markup_for_volume(volume_node, scene, ROLE_ZEBRAFISH_MARKUPS_LINE)
    created_here = False
    if line is None:
        line = scene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        line.CreateDefaultDisplayNodes()
        line.SetName(_markups_display_name(result, volume_node, "Line"))
        created_here = True
    display = line.GetDisplayNode() if hasattr(line, "GetDisplayNode") else None
    if display is not None:
        try:
            display.SetColor(*_STRAIGHT_CLR)
        except Exception:
            pass
        try:
            # Created hidden — see the matching comment in
            # _create_segmentation_for_volume for why.
            display.SetVisibility(False)
        except Exception:
            pass
        try:
            display.SetVisibility2D(True)
        except Exception:
            pass
        try:
            display.SetVisibility3D(True)
        except Exception:
            pass
        _style_thin_line_markup(display)
    # On reuse, drop any leftover endpoints from a previous run before
    # adding the new ones — otherwise AddControlPoint appends and the
    # line slowly accumulates duplicate Head/Tail points across re-runs.
    if not created_here:
        try:
            if hasattr(line, "RemoveAllControlPoints"):
                line.RemoveAllControlPoints()
        except Exception:
            pass
    _add_line_endpoints(line, sl_pts, result, volume_node)
    _lock_markups_node(line)
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_MARKUPS_LINE, line)
    if created_here:
        _reparent_in_subject_hierarchy(scene, line, volume_node)
    return line


def _markups_display_name(result, volume_node, suffix):
    """Return a display name for a markups node derived from the volume node's name."""
    base = None
    if hasattr(volume_node, "GetName"):
        try:
            base = volume_node.GetName()
        except Exception:
            base = None
    if not base:
        base = (result or {}).get("filename") or "ZebrafishEmbryoAnalyzer"
    return f"{base} {suffix}"


def _get_existing_markup_for_volume(volume_node, scene, role):
    """Return the live markups node referenced under ``role``, or ``None``.

    Mirrors :func:`_get_existing_seg_for_volume` for markups roles
    (``ROLE_ZEBRAFISH_MARKUPS_LINE`` / ``ROLE_ZEBRAFISH_MARKUPS_CURVE``).
    Used by ``_create_markups_line_for_volume`` / ``_create_markups_curve_for_volume``
    so a re-analysis reuses the existing line/curve node instead of
    stacking a duplicate on top — also needed because :func:`_set_node_reference`
    now collapses the role to a single id.
    """
    if volume_node is None or scene is None:
        return None
    nid = None
    try:
        if hasattr(volume_node, "GetNodeReferenceID"):
            nid = volume_node.GetNodeReferenceID(role)
    except Exception:
        nid = None
    if not nid:
        return None
    try:
        return scene.GetNodeByID(nid)
    except Exception:
        return None


class _Vec3(tuple):
    """A 3-component position tuple that the real vtk API can index.

    Real ``vtkMRMLMarkupsNode.AddControlPoint`` accepts anything that
    behaves like a 3-vector (``[0]``/``[1]``/``[2]`` access). The test fakes
    accept the same interface. Using this tiny named tuple keeps the helper
    independent of the optional ``vtk`` import — critical because the
    per-image MRML work runs in both the Slicer process (vtk available)
    and the inference subprocess (no vtk).
    """


def _vec3(x, y, z=0.0):
    return _Vec3((float(x), float(y), float(z)))


def _mask_spacing_mm(result):
    """Return ``(row_spacing_mm, col_spacing_mm)`` for mask-pixel-index coordinates.

    ``result["spacing"]`` is ``(row_um_per_maskpx, col_um_per_maskpx)``, written
    by ``logic.py``'s ``analyse_images`` — micrometres per *mask*-pixel (already
    scaled for the mask-vs-original-image resolution ratio). Converts to
    millimetres (Slicer RAS units) for use as a per-axis multiplier on raw
    mask-pixel indices. Falls back to ``(1.0, 1.0)`` only if ``spacing`` is
    missing or malformed (defensive — should not happen for a real result).
    """
    spacing = result.get("spacing") if result else None
    if not spacing or len(spacing) != 2:
        return (1.0, 1.0)
    try:
        return (float(spacing[0]) / 1000.0, float(spacing[1]) / 1000.0)
    except (TypeError, ValueError):
        return (1.0, 1.0)


def _mask_dims(result):
    """Return ``(mask_h, mask_w)`` from ``result["mask"]``'s shape, or ``None``.

    Used to mirror the 180-degree rotation (``flipud`` + ``fliplr``) that
    ``update_image_node`` applies to the volume's pixel data (issue #58
    follow-up): mask pixel (row, col) is displayed at volume voxel
    (mask_h-1-row, mask_w-1-col), not (row, col) unchanged. Returns ``None``
    when ``mask`` is missing/malformed so callers can fall back to the
    historical unflipped formula (degenerate/test inputs only — real
    results always carry ``mask``).
    """
    mask = result.get("mask") if result else None
    if mask is None or not hasattr(mask, "shape") or len(mask.shape) < 2:
        return None
    return int(mask.shape[0]), int(mask.shape[1])


def _add_line_endpoints(line, sl_pts, result, volume_node):
    """Add Head and Tail control points to ``line`` from ``sl_pts``.

    ``sl_pts`` is the straight-line endpoints tuple produced by
    ``tube_length_border2border`` — shape ``((row0, col0), (row1, col1))``
    in mask coordinates. Points are scaled by ``_mask_spacing_mm(result)``
    before being placed in RAS space so they land inside the volume node's
    actual physical extent (issue #58). The same flipud / fliplr transform
    that ``update_image_node`` applies to the volume node's pixel data is
    replicated here so the markups land on the visible fish.

    In Slicer production, ``line.AddControlPoint`` accepts either a
    ``vtkVector3d`` or any object supporting ``[0]``/``[1]``/``[2]``. We pass
    a small ``_Vec3`` named tuple so the helper works under both real Slicer
    (no vtk import here, the runtime picks the right path) and plain pytest
    where vtk is unavailable.

    All point additions are wrapped in a single try/except so a failure
    midway does NOT abort the surrounding apply_analysis_to_volume_node
    step — cancel-safety contract preserved. The markups node itself is
    still attached and visible in the Data module; the control points can
    be re-applied on reload by sub-issue #5.
    """
    try:
        p0, p1 = sl_pts
    except Exception:
        return
    # Mask coords are (row, col). update_image_node writes the volume's
    # pixel data through flipud(fliplr(...)) — a 180-degree rotation — so
    # mask pixel (row, col) is displayed at voxel (mask_h-1-row, mask_w-1-col),
    # not (row, col) unchanged. Mirror that rotation here (issue #58 follow-up)
    # so the control points land on the visible fish instead of off to the
    # side / outside the volume. row_mm/col_mm convert mask-pixel distances to mm.
    try:
        row_mm, col_mm = _mask_spacing_mm(result)
        dims = _mask_dims(result)
        if dims is not None:
            mask_h, mask_w = dims
            pos_head = _vec3((mask_w - 1 - p0[1]) * col_mm, (mask_h - 1 - p0[0]) * row_mm, 0.0)
            pos_tail = _vec3((mask_w - 1 - p1[1]) * col_mm, (mask_h - 1 - p1[0]) * row_mm, 0.0)
        else:
            # Degenerate input (no mask, e.g. missing spacing) — keep the
            # historical unflipped formula rather than guessing dimensions.
            pos_head = _vec3(p0[1] * col_mm, -p0[0] * row_mm, 0.0)
            pos_tail = _vec3(p1[1] * col_mm, -p1[0] * row_mm, 0.0)
    except Exception:
        return
    try:
        if hasattr(line, "AddControlPoint"):
            # In the real Slicer build, line.SetName is the public API for
            # the node display name (already set above); for individual
            # control points we rely on the second positional arg of
            # AddControlPoint. Older bindings exposed only SetNthControlPointLabel;
            # call both when available so labels survive either path.
            line.AddControlPoint(pos_head, "Head")
            line.AddControlPoint(pos_tail, "Tail")
            # Belt-and-braces: also set labels explicitly so any
            # double-add path (e.g. AddControlPoint auto-naming) doesn't
            # overwrite our labels.
            if hasattr(line, "SetNthControlPointLabel"):
                try:
                    line.SetNthControlPointLabel(0, "Head")
                    line.SetNthControlPointLabel(1, "Tail")
                except Exception:
                    pass
        else:
            # Older bindings — record via plain attribute so tests can verify.
            line._control_points = [
                {"label": "Head", "position": pos_head},
                {"label": "Tail", "position": pos_tail},
            ]
    except Exception:
        # AddControlPoint may fail when slicer / vtk is unavailable (subprocess).
        # The markups node itself is still attached and visible in the Data
        # module; the control points can be re-applied on reload by sub-issue #5.
        pass


def _style_thin_line_markup(display):
    """Reduce a MarkupsLine/Curve display node to a thin, non-editable line.

    ``vtkMRMLMarkupsDisplayNode`` defaults (glyph scale ~3% of viewport,
    default control-point balls, floating name+length label) are tuned for
    anatomical-scale volumes; on a ~5 mm zebrafish embryo they render as a
    thick tube with an oversized label dwarfing the segmentation (issue #58
    follow-up).

    Control-point glyphs are shrunk to near-zero (``GlyphScale``) rather than
    hidden outright — there is no separate "points off, line on" toggle on
    this display node. ``LineThickness`` is a *fraction of GlyphScale*, so
    shrinking GlyphScale to make points invisible also starves the line
    (first attempt at issue #58 follow-up: line became nearly invisible
    too). Switching ``CurveLineSizeMode`` to Absolute decouples the tube
    diameter from GlyphScale entirely — ``LineDiameter`` is then a fixed mm
    value independent of point size. Each call wrapped individually since
    the exact API surface varies across Slicer versions.
    """
    try:
        display.SetGlyphScale(0.01)
    except Exception:
        pass
    try:
        display.SetCurveLineSizeMode(getattr(display, "SizeModeAbsolute", 1))
    except Exception:
        pass
    try:
        display.SetLineDiameter(0.012)
    except Exception:
        pass
    try:
        # Without this, the segmentation's 3D surface occludes the thin
        # line/curve whenever it's behind it from the current camera angle.
        display.SetOccludedVisibility(True)
    except Exception:
        pass
    try:
        display.SetOccludedOpacity(1.0)
    except Exception:
        pass
    try:
        display.SetPropertiesLabelVisibility(False)
    except Exception:
        pass
    try:
        display.SetPointLabelsVisibility(False)
    except Exception:
        pass


def _lock_markups_node(node):
    """Lock a MarkupsLine/Curve node so its control points can't be dragged.

    ``vtkMRMLMarkupsNode.SetLocked(True)`` disables interactive point
    manipulation in slice/3D views while leaving visibility, color and
    other display-node properties editable as usual. These markups mirror
    computed analysis results, not manual annotations, so accidental
    dragging should not silently desync them from ``result``.
    """
    try:
        node.SetLocked(True)
    except Exception:
        pass


def _create_markups_curve_for_volume(result, volume_node, scene):
    """Create a MarkupsCurveNode for the centerline when ``path_points`` exist.

    Skipped when ``path_points`` is ``None`` or has fewer than two points —
    the same conditional pattern as ``update_segmentation_node``'s eye segment.
    Color matches ``overlay._PATH_COLOR`` (cyan). Returned node is attached
    via ``ROLE_ZEBRAFISH_MARKUPS_CURVE``.

    Reuses an existing curve node already attached to ``volume_node`` when
    one is in the scene (see :func:`_get_existing_markup_for_volume`) — a
    re-run refreshes the centerline in place. The previous curve's control
    points are dropped first so re-analysis doesn't pile duplicate
    segments on top.

    Returns the new (or reused) node, or ``None`` when skipped.
    """
    if volume_node is None or scene is None:
        return None
    path_pts = result.get("path_points")
    if path_pts is None:
        return None
    try:
        n_pts = len(path_pts)
    except Exception:
        n_pts = 0
    if n_pts < 2:
        return None
    import slicer  # lazy: tests never import slicer
    curve = _get_existing_markup_for_volume(volume_node, scene, ROLE_ZEBRAFISH_MARKUPS_CURVE)
    created_here = False
    if curve is None:
        curve = scene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        curve.CreateDefaultDisplayNodes()
        curve.SetName(_markups_display_name(result, volume_node, "Curve"))
        created_here = True
    display = curve.GetDisplayNode() if hasattr(curve, "GetDisplayNode") else None
    if display is not None:
        try:
            display.SetColor(*_PATH_COLOR)
        except Exception:
            pass
        try:
            # Created hidden — see the matching comment in
            # _create_segmentation_for_volume for why.
            display.SetVisibility(False)
        except Exception:
            pass
        try:
            display.SetVisibility2D(True)
        except Exception:
            pass
        try:
            display.SetVisibility3D(True)
        except Exception:
            pass
        _style_thin_line_markup(display)
    if not created_here:
        try:
            if hasattr(curve, "RemoveAllControlPoints"):
                curve.RemoveAllControlPoints()
        except Exception:
            pass
    _add_curve_points(curve, path_pts, result)
    _lock_markups_node(curve)
    _set_node_reference(volume_node, ROLE_ZEBRAFISH_MARKUPS_CURVE, curve)
    if created_here:
        _reparent_in_subject_hierarchy(scene, curve, volume_node)
    return curve


def _add_curve_points(curve, path_pts, result):
    """Add all ``path_pts`` to ``curve`` as anonymous control points.

    Each point is converted from mask (row, col) to RAS (R, A, S) using the
    same flip applied to the image, and scaled by ``_mask_spacing_mm(result)``
    so the curve lands inside the volume node's actual physical extent
    (issue #58). ``_Vec3`` is used as the position type so the helper works
    under both real Slicer and plain pytest (see :func:`_add_line_endpoints`
    for the cancel-safety rationale).
    """
    try:
        row_mm, col_mm = _mask_spacing_mm(result)
        dims = _mask_dims(result)
        if dims is not None:
            mask_h, mask_w = dims
            positions = [
                _vec3((mask_w - 1 - p[1]) * col_mm, (mask_h - 1 - p[0]) * row_mm, 0.0)
                for p in path_pts
            ]
        else:
            # Degenerate input (no mask) — keep the historical unflipped formula.
            positions = [_vec3(p[1] * col_mm, -p[0] * row_mm, 0.0) for p in path_pts]
    except Exception:
        return
    try:
        if hasattr(curve, "AddControlPoint"):
            for pos in positions:
                curve.AddControlPoint(pos, "")
        else:
            curve._control_points = [
                {"label": "", "position": p} for p in positions
            ]
    except Exception:
        # Half-written curves are still attached to the volume node — see
        # cancel-safety contract in apply_analysis_to_volume_node.
        pass


def _sync_volume_node_spacing(volume_node, um_per_px):
    """Re-apply ``um_per_px`` to ``volume_node``'s spacing (mm), in place.

    Mirrors the isotropic spacing formula in :func:`image_geometry` without
    touching pixel data, dimensions, or origin — only ``SetSpacing`` is
    called, so this is safe to call repeatedly and cheap enough to run
    before every analysis.
    """
    if volume_node is None or not hasattr(volume_node, "SetSpacing"):
        return
    spacing_mm = float(um_per_px) / 1000.0
    volume_node.SetSpacing(spacing_mm, spacing_mm, 1.0)


def apply_analysis_to_volume_node(result, volume_node, scene, um_per_px):
    """Stream one fully-analysed image's MRML state onto ``volume_node``.

    Writes, in order:

    1. A new ``vtkMRMLSegmentationNode`` (body + eye segments via
       :func:`update_segmentation_node`) attached via
       ``ROLE_ZEBRAFISH_SEGMENTATION`` on the volume node.
    2. A ``vtkMRMLMarkupsLineNode`` with Head/Tail control points, attached
       via ``ROLE_ZEBRAFISH_MARKUPS_LINE`` — only when ``result["length"]``
       and ``result["straight_line_points"]`` are present.
    3. A ``vtkMRMLMarkupsCurveNode`` built from ``path_points``, attached
       via ``ROLE_ZEBRAFISH_MARKUPS_CURVE`` — only when ``path_points`` has
       at least two entries.
    4. Metric attributes under the ``ZebrafishAnalysis.`` prefix.

    Must be called on the Slicer main thread; MRML / vtk objects live there.
    Returns ``None`` silently if ``volume_node`` or ``scene`` is ``None`` or
    if the result carries no image — the caller is expected to log + record
    ``error`` on the result dict in that case.

    All four steps are best-effort: a failure in any one step (e.g. Slicer
    is unavailable mid-batch) is logged via ``logging.exception`` and
    swallowed so a single bad image never aborts a batch. The caller is
    expected to check ``result.get("error")`` afterwards and route the row
    to the error column (sub-issue #40).
    """
    import logging

    if volume_node is None or scene is None:
        return None
    if not result or result.get("original") is None:
        return None

    # Volume nodes are created eagerly at folder-load time (#38) using
    # whatever um_per_px the UI showed then (often a rough header-based
    # estimate). If the user recalibrates (e.g. "Auto-detect from first
    # image") before running analysis, that baked-in spacing goes stale —
    # re-syncing it here to the um_per_px actually used for this analysis
    # run keeps the volume and its segmentation/markups geometrically
    # consistent, whatever the calibration timeline was.
    try:
        _sync_volume_node_spacing(volume_node, um_per_px)
    except Exception:
        logging.exception(
            "apply_analysis_to_volume_node: volume node spacing sync failed"
        )

    seg_node = None
    try:
        seg_node = _create_segmentation_for_volume(result, volume_node, scene, um_per_px)
    except Exception:
        logging.exception(
            "apply_analysis_to_volume_node: segmentation node creation failed"
        )

    line_node = None
    try:
        line_node = _create_markups_line_for_volume(result, volume_node, scene)
    except Exception:
        logging.exception(
            "apply_analysis_to_volume_node: MarkupsLineNode creation failed"
        )

    curve_node = None
    try:
        curve_node = _create_markups_curve_for_volume(result, volume_node, scene)
    except Exception:
        logging.exception(
            "apply_analysis_to_volume_node: MarkupsCurveNode creation failed"
        )

    # Attributes last, so a failure here cannot leave a half-written
    # segmentation behind.
    try:
        _write_metric_attributes(result, volume_node)
    except Exception:
        logging.exception(
            "apply_analysis_to_volume_node: attribute write failed"
        )

    return seg_node


# ---------------------------------------------------------------------------
# Issue #56 follow-up: scene-reload overlay reconstruction
# ---------------------------------------------------------------------------
#
# ``volume_node_to_result_dict`` only restores the scalar metric attributes
# (length, curvature, ratio, eye_area, eye_diameter). After a Save Scene
# -> Load Scene round-trip, the segmentation and markups nodes ARE in the
# scene (so the Data module still shows the segmentation, and the user can
# show/hide it via Slicer's default slice views), but the Zebra module's
# ``result`` dicts lack the ``mask`` / ``eye_mask`` / ``path_points`` /
# ``straight_line_points`` keys that ``overlay.make_full_overlay`` draws.
# The gallery therefore shows bare originals after a scene reload even
# though every analysis artefact is sitting right there in the scene.
#
# The helpers below re-derive those four row keys from the existing scene
# nodes referenced via ``ROLE_ZEBRAFISH_SEGMENTATION``,
# ``ROLE_ZEBRAFISH_MARKUPS_CURVE``, and ``ROLE_ZEBRAFISH_MARKUPS_LINE``.
# Each helper is best-effort and never raises so a partially-restored
# scene (e.g. the user removed the Eye segment in the Data module) still
# loads cleanly — the missing key just means that overlay layer is
# silently omitted.


def _extract_segment_mask(seg_node, segment_name):
    """Return the binary labelmap of ``segment_name`` on ``seg_node`` as a
    2-D uint8 ndarray (values 0 / 1), or ``None`` when the segment is
    missing or extraction fails.

    Lazy-imports ``slicer`` so the function is unit-testable without the
    Slicer runtime; tests stub ``slicer.util.arrayFromSegment`` via
    ``sys.modules`` before importing :mod:`ZebrafishEmbryoAnalyzerLib.mrml`.

    The returned array keeps the segmentation's stored geometry — the
    overlay code resizes to the original image dimensions internally, so
    any (H, W) is acceptable as long as it carries the body's footprint.

    Orientation: ``_make_oriented_image`` stores the mask as
    ``flipud(fliplr(...))`` to match VTK's bottom-left origin and Slicer's
    radiological convention, so the array coming back out of
    ``arrayFromSegment`` is rotated 180 degrees relative to the image the
    overlay draws on. Undo it here — the same inverse
    :func:`volume_node_to_pixels` applies to the image itself. Without
    this the restored mask lands mirrored about the image centre instead
    of on the fish.
    """
    if seg_node is None or not segment_name:
        return None
    try:
        import numpy as np  # local — module never imports numpy at top
        import slicer  # lazy: tests never import slicer
    except Exception:
        return None
    try:
        seg = seg_node.GetSegmentation()
    except Exception:
        return None
    if seg is None:
        return None
    try:
        seg_id = seg.GetSegmentIdBySegmentName(segment_name)
    except Exception:
        seg_id = ""
    if not seg_id:
        return None
    try:
        # ``arrayFromSegment`` is a deprecated wrapper that logs a warning on
        # every call — two per row (Body + Eye), so a scene reload floods the
        # Python console. It forwards to ``arrayFromSegmentBinaryLabelmap``
        # with identical semantics; call that directly and keep the old name
        # as a fallback for older Slicer builds and the test doubles.
        read_labelmap = getattr(
            slicer.util, "arrayFromSegmentBinaryLabelmap", None
        ) or getattr(slicer.util, "arrayFromSegment", None)
        if read_labelmap is None:
            return None
        arr = read_labelmap(seg_node, seg_id)
    except Exception:
        return None
    if arr is None or getattr(arr, "size", 0) == 0:
        return None
    try:
        # ``arrayFromSegment`` returns (K, H, W) for labelmap volumes;
        # collapse to a single 2-D mask.
        while arr.ndim > 2:
            arr = arr[0]
        mask = (arr > 0).astype(np.uint8)
        return np.ascontiguousarray(np.flipud(np.fliplr(mask)))
    except Exception:
        return None


def _markups_to_mask_coords(volume_node):
    """Return ``(col_mm, row_mm)`` for converting RAS positions to mask coords.

    The analysis pipeline writes control points into RAS as
    ``(mask_w - 1 - col) * col_mm`` etc. (see :func:`_add_curve_points` and
    :func:`_add_line_endpoints`). To invert on reload we need ``col_mm`` /
    ``row_mm`` expressed in mask-pixel units — i.e. the physical extent of
    the volume mapped onto the 256-pixel mask grid, not the volume's own
    per-pixel spacing.

    Returns ``(col_mm, row_mm)`` or ``None`` when ``volume_node`` does not
    expose the required geometry. Never raises.
    """
    if volume_node is None:
        return None
    try:
        spacing = volume_node.GetSpacing()
        dims = volume_node.GetDimensions()
    except Exception:
        return None
    if spacing is None or dims is None:
        return None
    try:
        col_mm = (float(dims[0]) / 256.0) * float(spacing[0])
        row_mm = (float(dims[1]) / 256.0) * float(spacing[1])
    except Exception:
        return None
    if row_mm <= 0 or col_mm <= 0:
        return None
    return col_mm, row_mm


def _extract_markups_curve_points(curve_node, volume_node):
    """Return ``path_points`` (shape ``(N, 2)``) read back from ``curve_node``.

    Mirrors the conversion :func:`_add_curve_points` performs in reverse:
    each RAS control point ``(R, A, S)`` becomes
    ``(mask_h - 1 - A / row_mm, mask_w - 1 - R / col_mm)``. Returns
    ``None`` when the curve has fewer than two points or extraction
    fails. Never raises.
    """
    if curve_node is None or volume_node is None:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    coords = _markups_to_mask_coords(volume_node)
    if coords is None:
        return None
    col_mm, row_mm = coords
    try:
        n = int(curve_node.GetNumberOfControlPoints())
    except Exception:
        return None
    if n < 2:
        return None
    try:
        out = np.zeros((n, 2), dtype=float)
        pos = [0.0, 0.0, 0.0]
        for i in range(n):
            pos[0] = pos[1] = pos[2] = 0.0
            try:
                curve_node.GetNthControlPointPosition(i, pos)
            except Exception:
                return None
            out[i, 0] = 256.0 - 1.0 - pos[1] / row_mm
            out[i, 1] = 256.0 - 1.0 - pos[0] / col_mm
    except Exception:
        return None
    return out


def _extract_markups_line_endpoints(line_node, volume_node):
    """Return ``straight_line_points`` ``((r0, c0), (r1, c1))`` from ``line_node``.

    Mirrors :func:`_add_line_endpoints`. Returns ``None`` when the line has
    fewer than two control points or extraction fails. Never raises.
    """
    if line_node is None or volume_node is None:
        return None
    coords = _markups_to_mask_coords(volume_node)
    if coords is None:
        return None
    col_mm, row_mm = coords
    try:
        n = int(line_node.GetNumberOfControlPoints())
    except Exception:
        return None
    if n < 2:
        return None
    try:
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        try:
            line_node.GetNthControlPointPosition(0, pos0)
            line_node.GetNthControlPointPosition(1, pos1)
        except Exception:
            return None
        r0 = 256.0 - 1.0 - pos0[1] / row_mm
        c0 = 256.0 - 1.0 - pos0[0] / col_mm
        r1 = 256.0 - 1.0 - pos1[1] / row_mm
        c1 = 256.0 - 1.0 - pos1[0] / col_mm
    except Exception:
        return None
    return ((r0, c0), (r1, c1))


def _populate_row_overlays_from_scene(row, volume_node, scene):
    """In-place: pull overlay inputs from scene nodes into ``row``.

    Walks the ``ROLE_ZEBRAFISH_SEGMENTATION`` /
    ``ROLE_ZEBRAFISH_MARKUPS_CURVE`` / ``ROLE_ZEBRAFISH_MARKUPS_LINE``
    references on ``volume_node`` and writes ``mask`` / ``eye_mask`` /
    ``path_points`` / ``straight_line_points`` keys onto ``row`` so
    :func:`overlay.make_full_overlay` can render the analyzed overlay
    after a saved-scene reload.

    Skips work when ``row`` already carries an ``error`` (the row will be
    auto-excluded and rendered as bare original anyway, by the defensive
    guard in :func:`overlay.make_full_overlay`). Best-effort: every helper
    silently returns ``None`` on failure, so a partially-restored scene
    just leaves the corresponding row key unset instead of crashing the
    widget.
    """
    if row is None or volume_node is None or scene is None:
        return
    if row.get("error"):
        return

    # Body + eye masks from the segmentation node.
    seg_id = ""
    try:
        if hasattr(volume_node, "GetNodeReferenceID"):
            seg_id = volume_node.GetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION) or ""
    except Exception:
        seg_id = ""
    if seg_id:
        seg_node = None
        try:
            seg_node = scene.GetNodeByID(seg_id)
        except Exception:
            seg_node = None
        if seg_node is not None:
            mask = _extract_segment_mask(seg_node, "Body")
            if mask is not None:
                row["mask"] = mask
            eye_mask = _extract_segment_mask(seg_node, "Eye")
            if eye_mask is not None:
                row["eye_mask"] = eye_mask

    # Centerline from the MarkupsCurveNode (if path was computed).
    curve_id = ""
    try:
        if hasattr(volume_node, "GetNodeReferenceID"):
            curve_id = volume_node.GetNodeReferenceID(ROLE_ZEBRAFISH_MARKUPS_CURVE) or ""
    except Exception:
        curve_id = ""
    if curve_id:
        curve_node = None
        try:
            curve_node = scene.GetNodeByID(curve_id)
        except Exception:
            curve_node = None
        if curve_node is not None:
            pts = _extract_markups_curve_points(curve_node, volume_node)
            if pts is not None:
                row["path_points"] = pts

    # Endpoints from the MarkupsLineNode (if length was computed).
    line_id = ""
    try:
        if hasattr(volume_node, "GetNodeReferenceID"):
            line_id = volume_node.GetNodeReferenceID(ROLE_ZEBRAFISH_MARKUPS_LINE) or ""
    except Exception:
        line_id = ""
    if line_id:
        line_node = None
        try:
            line_node = scene.GetNodeByID(line_id)
        except Exception:
            line_node = None
        if line_node is not None:
            sl = _extract_markups_line_endpoints(line_node, volume_node)
            if sl is not None:
                row["straight_line_points"] = sl
