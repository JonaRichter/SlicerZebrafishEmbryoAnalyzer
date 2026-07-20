import logging
import sys

import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin


# Slicer puts this module's directory on sys.path, so ZebrafishEmbryoAnalyzerLib and
# ZebrafishEmbryoAnalyzerCore import as normal packages — no path manipulation needed.

_LIB_MODULES = (
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
)

_CORE_MODULES = (
    "ZebrafishEmbryoAnalyzerCore.seg",
    "ZebrafishEmbryoAnalyzerCore.seg_helper",
    "ZebrafishEmbryoAnalyzerCore.length",
    "ZebrafishEmbryoAnalyzerCore.manual",
    "ZebrafishEmbryoAnalyzerCore.scalebar",
)

_RELOAD_MODULES = _LIB_MODULES + _CORE_MODULES


def _evict_reload_modules():
    for _m in _RELOAD_MODULES:
        sys.modules.pop(_m, None)


_evict_reload_modules()


class ZebrafishEmbryoAnalyzer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Zebrafish Embryo Analyzer"
        self.parent.categories = ["Quantification"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jona Richter", "Mark Daniel Arndt"]
        self.parent.helpText = (
            "Segment zebrafish from 2-D microscopy images and measure "
            "body length, curvature class, length/straight-line ratio, "
            "and eye metrics."
        )
        self.parent.acknowledgementText = (
            "Based on the Zebrafish Webapp "
            "(github.com/MarkDanielArndt/Zebrafish_webapp)."
        )


class ZebrafishEmbryoAnalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self._main = None
        self._parameterNode = None
        self._sceneObserversRegistered = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        _evict_reload_modules()

        self.logic = ZebrafishEmbryoAnalyzerLogic()
        # Issue #56 follow-up: the widget owns ``VTKObservationMixin`` and
        # therefore the segmentation ModifiedEvent observers, but widget
        # code (and ``_on_results_ready``) calls
        # ``self._logic.setup_segmentation_staleness_observers()`` so the
        # logic can be the single facade. Wire the widget as a back-pointer
        # so ``Logic.setup_segmentation_staleness_observers`` can delegate
        # to the real implementation.
        self.logic._widget_ref = self

        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
        self._main = ZebrafishEmbryoAnalyzerMainWidget(self.layout, logic=self.logic)
        self._main._on_settings_changed = self._on_settings_changed
        self._main.refresh_dependency_status()

        self._register_scene_observers()
        self.initializeParameterNode()

        if self._main is not None:
            self._main.apply_shell_layout()

    def _register_scene_observers(self):
        """Register MRML scene observers exactly once per setup."""
        if self._sceneObserversRegistered:
            return
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.StartCloseEvent,
            self._on_scene_start_close,
        )
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.EndCloseEvent,
            self._on_scene_end_close,
        )
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.EndImportEvent,
            self._on_scene_end_import,
        )
        self._sceneObserversRegistered = True

    def enter(self):
        if hasattr(self, "logic"):
            self.initializeParameterNode()
        if self._main is not None:
            self._main.apply_shell_layout()
            # Issue #41 follow-up: EndImportEvent for a saved scene that was
            # loaded BEFORE this module was first opened cannot reach us —
            # scene observers are only registered in setup(), which Slicer
            # calls the first time the module is selected. No-op when the
            # widget already has results (do not rebuild on every tab
            # switch in a long session).
            self._main.try_rebuild_from_scene_if_empty()
            self._main.prompt_install_if_missing()
            # Issue #56 Mode B follow-up: lightweight revalidation of
            # ``self._results`` against the live MRML scene. The Data
            # module is ground truth — when the user deletes a
            # segmentation node there while the Zebrafish module was
            # in the background, the row's dangling-reference status
            # only becomes visible to the widget when we re-validate
            # each row on entry. Without this refresh the gallery
            # still shows the row as if its segmentation were
            # attached, even though the seg node is gone. Cheap (no
            # pixel-array reload, no thumbnail rebuild) so safe to run
            # on every tab switch.
            try:
                self._main.refresh_results_against_scene()
            except Exception:
                logging.exception(
                    "ZebrafishEmbryoAnalyzer: refresh_results_against_scene failed in enter()"
                )
            # Issue #56 follow-up: re-arm the per-image segmentation
            # ModifiedEvent observers on every module entry. Without this,
            # observers installed the first time the module was opened get
            # torn down by Slicer on tab switch, and the user's later
            # Segment Editor edits silently no-op on the staleness path —
            # which means a manual edit never triggers the recompute prompt
            # and the user has no way to learn their edit needs a re-run.
            try:
                self.setup_segmentation_staleness_observers()
            except Exception:
                logging.exception(
                    "ZebrafishEmbryoAnalyzer: setup_segmentation_staleness_observers failed in enter()"
                )
            # Issue #42: ask the user to recompute metrics for every
            # tracked image whose segmentation was edited in the Segment
            # Editor since we last saw it. The policy is "ask once per
            # image per enter()"; changing it to "ask only once" is a
            # one-line tweak in the policy function.
            self._main.prompt_recompute_stale_images()

    def exit(self):
        if self._main is not None:
            self._main.restore_shell_layout()

    def cleanup(self):
        self.setParameterNode(None)
        self.removeObservers()
        self._sceneObserversRegistered = False
        if self._main is not None:
            self._main.restore_shell_layout()  # ensure restore even on reload (exit() not called)
            self._main.cleanup()

    def getParameterNode(self):
        """Delegate to the logic object.

        ``ScriptedLoadableModuleWidget`` (a plain Python class) does not
        provide ``getParameterNode()`` itself — only ``ScriptedLoadableModuleLogic``
        does. Widget-side code that needs the parameter node outside
        ``initializeParameterNode()``/``setParameterNode()`` calls this
        instead of reaching into ``self.logic`` directly at each call site.
        """
        return self.logic.getParameterNode()

    # ------------------------------------------------------------------
    # Parameter node
    # ------------------------------------------------------------------

    def initializeParameterNode(self):
        """Get the scene parameter node, fill missing params and normalize invalid ones."""
        import math
        from ZebrafishEmbryoAnalyzerLib.widget import (
            PARAM_DEFAULTS, _MODEL_BY_ID, _DEFAULT_MODEL_ID,
            PARAM_LENGTH_ENABLED, PARAM_CURVATURE_ENABLED, PARAM_RATIO_ENABLED,
            PARAM_EYES_ENABLED, PARAM_CONFIDENCE_THRESHOLD_ENABLED,
            PARAM_CONFIDENCE_THRESHOLD, PARAM_UM_PER_PX, PARAM_MODEL_ID,
        )
        node = self.logic.getParameterNode()
        wasModified = node.StartModify()
        try:
            for name, default in PARAM_DEFAULTS.items():
                if not node.GetParameter(name):
                    node.SetParameter(name, default)
            for key in (PARAM_LENGTH_ENABLED, PARAM_CURVATURE_ENABLED, PARAM_RATIO_ENABLED,
                        PARAM_EYES_ENABLED, PARAM_CONFIDENCE_THRESHOLD_ENABLED):
                if node.GetParameter(key) not in ("true", "false"):
                    node.SetParameter(key, PARAM_DEFAULTS[key])
            v = node.GetParameter(PARAM_CONFIDENCE_THRESHOLD)
            try:
                f = float(v)
                if not (math.isfinite(f) and 0.0 <= f <= 1.0):
                    raise ValueError
            except (ValueError, TypeError):
                node.SetParameter(PARAM_CONFIDENCE_THRESHOLD, PARAM_DEFAULTS[PARAM_CONFIDENCE_THRESHOLD])
            v = node.GetParameter(PARAM_UM_PER_PX)
            try:
                f = float(v)
                if not (math.isfinite(f) and 0.001 <= f <= 9999.0):
                    raise ValueError
            except (ValueError, TypeError):
                node.SetParameter(PARAM_UM_PER_PX, PARAM_DEFAULTS[PARAM_UM_PER_PX])
            if node.GetParameter(PARAM_MODEL_ID) not in _MODEL_BY_ID:
                node.SetParameter(PARAM_MODEL_ID, _DEFAULT_MODEL_ID)
        finally:
            node.EndModify(wasModified)
        # Compute before setParameterNode so we can detect early return for same node.
        same_node = node is self._parameterNode
        self.setParameterNode(node)
        # setParameterNode early-returns for the same node (no GUI update inside).
        # Always update here so enter() re-entering is reflected in the UI.
        if same_node and self._main is not None:
            self._main.updateGUIFromParameterNode(node)

    def setParameterNode(self, node):
        """Connect to a new parameter node and disconnect from the old one."""
        if node is self._parameterNode:
            return
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._on_parameter_node_modified,
            )
        self._parameterNode = node
        if node is not None:
            self.addObserver(
                node,
                vtk.vtkCommand.ModifiedEvent,
                self._on_parameter_node_modified,
            )
            if self._main is not None:
                self._main.updateGUIFromParameterNode(node)

    def _on_parameter_node_modified(self, caller=None, event=None):
        if self._main is not None:
            self._main.updateGUIFromParameterNode(self._parameterNode)

    def _on_settings_changed(self):
        if self._parameterNode is not None and self._main is not None:
            self._main.updateParameterNodeFromGUI(self._parameterNode)

    # ------------------------------------------------------------------
    # Scene events
    # ------------------------------------------------------------------

    def _on_scene_start_close(self, caller=None, event=None):
        # Disconnect parameter node before scene objects are destroyed.
        # Also cancel active downloads and invalidate transient state early.
        self.setParameterNode(None)
        if self._main is not None:
            self._main._cancel_workers()

    def _on_scene_end_close(self, caller=None, event=None):
        # Reset session UI state, then connect to the fresh scene's parameter node.
        if self._main is not None:
            self._main.reset_for_scene_close()
        self.initializeParameterNode()

    def setup_segmentation_staleness_observers(self):
        """Issue #42: install a ``ModifiedEvent`` observer on every per-image
        segmentation node.

        Each observer is cheap — it sets a ``stale`` attribute on the
        corresponding volume node (via the ``ROLE_ZEBRAFISH_SEGMENTATION``
        reverse lookup) and auto-excludes the row. No recomputation, no
        model call. The widget calls this on every analysis completion
        and on scene reload so newly-tracked nodes are always observed.

        Idempotent: observer tags from a previous setup are removed
        before the new ones are installed, so repeated setup calls don't
        pile up duplicate observers on the same segmentation nodes.
        """
        try:
            import slicer
            import vtk as _vtk
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                list_tracked_volume_nodes,
                ROLE_ZEBRAFISH_SEGMENTATION,
                mark_volume_node_stale,
                clear_volume_node_stale,
            )
        except Exception:
            return
        param_node = self.getParameterNode()
        if param_node is None:
            return
        scene = getattr(slicer, "mrmlScene", None)
        if scene is None:
            return

        # Drop any tags from a previous setup so observers don't stack.
        prev = getattr(self, "_stale_observer_tags", [])
        for tag in prev:
            try:
                if hasattr(self, "removeObserver"):
                    self.removeObserver(tag)
            except Exception:
                pass
        self._stale_observer_tags = []

        for vol in list_tracked_volume_nodes(param_node, scene):
            seg_id = None
            try:
                seg_id = vol.GetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION)
            except Exception:
                seg_id = None
            if not seg_id:
                continue
            try:
                seg_node = scene.GetNodeByID(seg_id)
            except Exception:
                seg_node = None
            if seg_node is None:
                continue

            # Issue #56 follow-up: the seg node's ModifiedEvent also fires
            # during node teardown (Data-module delete). Marking the
            # volume stale in that case prompts the user to recompute on
            # the next module enter, which resurrects the segmentation
            # they just removed. Capture the seg id so the observer can
            # distinguish "edit in place" (still in scene) from "deleted"
            # (no longer in scene) — only the former counts as an external
            # edit that needs the recompute prompt.
            def _on_seg_modified(_caller=None, _event=None, _vol=vol, _seg_id=seg_id, _scene=scene):
                try:
                    current = _scene.GetNodeByID(_seg_id)
                except Exception:
                    current = None
                if current is None:
                    # Seg was deleted from the scene — Data module is
                    # ground truth. Clear any stale flag and the
                    # auto-exclude / error side-effects that the stale
                    # path set, so a future module enter does not offer
                    # to recompute and silently recreate the segmentation.
                    try:
                        clear_volume_node_stale(_vol)
                    except Exception:
                        pass
                    try:
                        _vol.SetAttribute("ZebrafishAnalysis.exclude", "false")
                        _vol.SetAttribute("ZebrafishAnalysis.error", "")
                    except Exception:
                        pass
                    return
                mark_volume_node_stale(_vol)

            tag = None
            try:
                if hasattr(self, "addObserver"):
                    tag = self.addObserver(
                        seg_node, _vtk.vtkCommand.ModifiedEvent, _on_seg_modified
                    )
            except Exception:
                tag = None
            if tag is not None:
                self._stale_observer_tags.append(tag)

    def _on_scene_end_import(self, caller=None, event=None):
        # Pick up parameter node values from the newly loaded scene.
        self.initializeParameterNode()
        # Issue #41: rebuild the widget's UI from the freshly imported
        # scene's volume nodes, seg nodes, and metric attributes — the
        # full-state reconstruction for scene reload. Runs only when the
        # new scene actually carries tracked volume nodes (i.e. a save of
        # our own scene); a fresh empty scene is a no-op.
        if self._main is not None:
            try:
                self._main.rebuild_from_scene()
            except Exception:
                # Reload must never crash the module — log and continue.
                logging.exception("ZebrafishEmbryoAnalyzer: scene-reload rebuild failed")
        # Re-arm segmentation observers on the freshly imported scene.
        self.setup_segmentation_staleness_observers()

class ZebrafishEmbryoAnalyzerLogic(ScriptedLoadableModuleLogic):
    """Orchestrates analysis requests on behalf of the widget.

    Widget calls these methods; each delegates to the corresponding free
    function in ZebrafishEmbryoAnalyzerLib.logic so the widget never imports that
    module directly.  ZebrafishEmbryoAnalyzerCore remains Slicer-independent.
    """

    def dependency_status(self) -> dict:
        """Return availability of optional ML/vision dependencies.

        Thin wrapper so widget.py never imports ZebrafishEmbryoAnalyzerLib.logic directly.
        """
        from ZebrafishEmbryoAnalyzerLib.logic import dependency_status as _ds
        return _ds()

    def run_analysis(self, image_paths, params, progress_callback=None):
        import math
        import os
        from collections.abc import Mapping, Sequence
        from ZebrafishEmbryoAnalyzerLib.errors import AnalysisInputError

        # image_paths: must be a non-string, non-empty Sequence of path-like values
        if not isinstance(image_paths, Sequence) or isinstance(image_paths, (str, bytes)):
            raise AnalysisInputError(
                f"image_paths must be a Sequence of paths (list or tuple), "
                f"got {type(image_paths).__name__!r}"
            )
        if not image_paths:
            raise AnalysisInputError("No images loaded")
        for p in image_paths:
            try:
                os.fspath(p)
            except TypeError:
                raise AnalysisInputError(
                    f"image_paths entries must be path-like strings, "
                    f"got {type(p).__name__!r}"
                )

        # params: must be a Mapping (isinstance check, not duck-typing .get())
        if not isinstance(params, Mapping):
            raise AnalysisInputError(
                f"params must be a Mapping, got {type(params).__name__!r}"
            )

        # um_per_px: numeric, finite, in UI range [0.001, 9999.0]
        raw_um = params.get("um_per_px", 22.99)
        try:
            um_per_px = float(raw_um)
        except (TypeError, ValueError):
            raise AnalysisInputError(
                f"params['um_per_px'] must be numeric, got {type(raw_um).__name__!r}"
            )
        if not math.isfinite(um_per_px):
            raise AnalysisInputError(
                f"params['um_per_px'] must be finite, got {um_per_px!r}"
            )
        if not (0.001 <= um_per_px <= 9999.0):
            raise AnalysisInputError(
                f"params['um_per_px'] must be in [0.001, 9999.0], got {um_per_px!r}"
            )

        # threshold: numeric, finite, in UI range [0.0, 1.0]
        raw_thr = params.get("threshold", 0.85)
        try:
            threshold = float(raw_thr)
        except (TypeError, ValueError):
            raise AnalysisInputError(
                f"params['threshold'] must be numeric, got {type(raw_thr).__name__!r}"
            )
        if not math.isfinite(threshold):
            raise AnalysisInputError(
                f"params['threshold'] must be finite, got {threshold!r}"
            )
        if not (0.0 <= threshold <= 1.0):
            raise AnalysisInputError(
                f"params['threshold'] must be in [0.0, 1.0], got {threshold!r}"
            )

        # Normalize: convert paths to str and write validated floats back.
        # Work on copies so the caller's list and dict are never mutated.
        normalized_paths = [os.fspath(p) for p in image_paths]
        normalized_params = dict(params)
        normalized_params["um_per_px"] = um_per_px
        normalized_params["threshold"] = threshold

        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        return analyse_images(normalized_paths, normalized_params, progress_callback)

    def detect_scalebar(self, image_path, label_um=None):
        from ZebrafishEmbryoAnalyzerLib.logic import detect_scalebar
        return detect_scalebar(image_path, label_um=label_um)

    def preload_models(self, params):
        from ZebrafishEmbryoAnalyzerLib.logic import preload_models
        return preload_models(params)

    def apply_manual_correction(self, result, point1_orig, point2_orig, params=None):
        from ZebrafishEmbryoAnalyzerLib.logic import apply_manual_correction
        return apply_manual_correction(result, point1_orig, point2_orig, params)

    def revert_manual_correction(self, result):
        from ZebrafishEmbryoAnalyzerLib.logic import revert_manual_correction
        return revert_manual_correction(result)

    def update_results_table(self, results):
        """Create or update the MRML table node with raw analysis results.

        Separate from run_analysis() so that a table update failure cannot
        discard a successful analysis.  Raises MRMLAdapterError on failure;
        the existing table content is preserved when possible.

        Returns
        -------
        vtkMRMLTableNode
        """
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                build_vtk_table,
                get_or_create_table_node,
                results_to_rows,
            )
            # Route through the single shared build/observe path: the run
            # path supplies ``results`` here (legacy entry point); the
            # reload path (#41) calls ``update_results_table_from_volume_nodes``
            # which uses the same ``_update_table_with_rows`` tail. Both
            # produce identical table content for identical metric state.
            return self._update_table_with_rows(results_to_rows(results))
        except MRMLAdapterError:
            raise
        except Exception as exc:
            raise MRMLAdapterError(
                f"Failed to update results table: {exc}"
            ) from exc

    def update_results_table_from_volume_nodes(self, volume_nodes):
        """Issue #40 / #41: rebuild the results table from per-image volume nodes.

        Reads each volume node's ``ZebrafishAnalysis.*`` attributes (written
        by ``apply_analysis_to_volume_node`` in #39) and rebuilds the same
        table content as :meth:`update_results_table` would. Used both:

        * after a fresh analysis run (the widget now routes the table
          build through this path instead of the in-memory ``results``
          list, to satisfy the "single code path" acceptance criterion),
        * and during scene reload (#41) where the only state available
          is the volume nodes already in the MRML scene.

        ``volume_nodes`` must be ordered the same way the user expects to
        see in the table; both the run path and the reload path read
        ``ROLE_ZEBRAFISH_IMAGES`` from the parameter node for that order.

        Returns ``vtkMRMLTableNode`` on success, ``None`` if no parameter
        node exists. Raises ``MRMLAdapterError`` on any failure.
        """
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
        try:
            from ZebrafishEmbryoAnalyzerLib.mrml import volume_nodes_to_rows
            rows = volume_nodes_to_rows(list(volume_nodes))
            return self._update_table_with_rows(rows)
        except MRMLAdapterError:
            raise
        except Exception as exc:
            raise MRMLAdapterError(
                f"Failed to update results table from volume nodes: {exc}"
            ) from exc

    def update_results_table_from_tracked_nodes(self):
        """Issue #40: build the table from the volume nodes currently
        registered on the parameter node under ``ROLE_ZEBRAFISH_IMAGES``.

        Used by the widget's post-analysis finish path so the run flow and
        the scene-reload flow share the same code path
        (``update_results_table_from_volume_nodes`` → ``_update_table_with_rows``).
        Issue #41 will call this exact method from the scene-reload handler.
        """
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import list_tracked_volume_nodes
        except Exception:
            return None
        param_node = self.getParameterNode()
        if param_node is None:
            return None
        scene = getattr(slicer, "mrmlScene", None)
        nodes = list_tracked_volume_nodes(param_node, scene)
        return self.update_results_table_from_volume_nodes(nodes)

    def rebuild_results_from_scene(self):
        """Issue #41: rebuild the widget's ``self._results`` list from the
        current MRML scene state.

        Walks the parameter node's ``ROLE_ZEBRAFISH_IMAGES`` reference list,
        reconstructs a result dict per volume node (with ``original``
        populated from the volume node's pixel array when available), and
        validates each via :func:`mrml.validate_volume_node` so broken /
        half-finished entries surface as auto-excluded error rows instead
        of crashing the widget.

        Returns ``list[dict]`` — one entry per tracked volume node. Returns
        an empty list when no parameter node or no tracked nodes exist.
        """
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                list_tracked_volume_nodes,
                volume_node_to_result_dict_with_validation,
                volume_node_to_pixels,
            )
        except Exception:
            return []

        param_node = self.getParameterNode()
        if param_node is None:
            return []
        scene = getattr(slicer, "mrmlScene", None)
        nodes = list_tracked_volume_nodes(param_node, scene)
        results = []
        for node in nodes:
            row = volume_node_to_result_dict_with_validation(node)
            px = volume_node_to_pixels(node)
            if px is not None:
                row["original"] = px
            # stashed so #42's segMTime comparison doesn't need to walk
            # the MRML scene again — keeps the per-row state self-contained.
            row["_volume_node"] = node
            row["_volume_node_id"] = (
                node.GetID() if hasattr(node, "GetID") else ""
            )
            results.append(row)
        return results

    def list_stale_tracked_volume_nodes(self):
        """Issue #42: return the volume nodes currently marked stale.

        Walks ``ROLE_ZEBRAFISH_IMAGES`` on the parameter node and filters
        to those whose ``ZebrafishAnalysis.stale`` attribute is ``"true"``.
        Used by the widget's ``enter()`` recompute-prompt to know which
        images to ask about.
        """
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                list_tracked_volume_nodes,
                is_volume_node_stale,
            )
        except Exception:
            return []
        param_node = self.getParameterNode()
        if param_node is None:
            return []
        scene = getattr(slicer, "mrmlScene", None)
        return [n for n in list_tracked_volume_nodes(param_node, scene)
                if is_volume_node_stale(n)]

    def is_volume_node_stale(self, volume_node):
        """Issue #42: thin re-export of :func:`mrml.is_volume_node_stale`.

        Lets the widget check a single node's staleness without importing
        ``ZebrafishEmbryoAnalyzerLib.mrml`` directly (see the test that
        enforces this rule). Returns False on any error so a missing
        attribute / bad node never raises into the UI.
        """
        try:
            from ZebrafishEmbryoAnalyzerLib.mrml import is_volume_node_stale
        except Exception:
            return False
        try:
            return bool(is_volume_node_stale(volume_node))
        except Exception:
            return False

    def clear_stale_flag_for_volume_node(self, volume_node):
        """Issue #56 follow-up: clear a single volume node's stale flag.

        Used by ``prompt_recompute_stale_images`` when the segmentation
        node a stale volume was once linked to has been removed from the
        scene (e.g. the user deleted it in the Data module). Clearing
        the flag without recreating anything honours "Data module is
        ground truth" — the gallery row stays visible without the
        "Segmentation modified — recompute needed" error, and a later
        enter() will not re-prompt.

        Best-effort: returns silently on any error so a transient scene
        glitch cannot break the recompute-prompt loop.
        """
        try:
            from ZebrafishEmbryoAnalyzerLib.mrml import clear_volume_node_stale
        except Exception:
            return
        try:
            clear_volume_node_stale(volume_node)
        except Exception:
            pass

    def volume_node_references_existing_seg(self, volume_node):
        """Issue #56 follow-up: return True if ``volume_node`` has a
        ``ROLE_ZEBRAFISH_SEGMENTATION`` reference whose target
        segmentation node is still in the scene.

        Used by the recompute-prompt loop to silently drop any
        tracked volume whose segmentation the user deleted in the
        Data module — honouring the deletion rather than offering to
        recompute (which would resurrect the segmentation).

        Returns ``False`` for any error condition so callers can use
        this as a guard without try/except.
        """
        if volume_node is None or not hasattr(volume_node, "GetNodeReferenceID"):
            return False
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import ROLE_ZEBRAFISH_SEGMENTATION
            scene = getattr(slicer, "mrmlScene", None)
            seg_id = volume_node.GetNodeReferenceID(ROLE_ZEBRAFISH_SEGMENTATION)
            if not seg_id or scene is None:
                return False
            return scene.GetNodeByID(seg_id) is not None
        except Exception:
            return False

    def validate_tracked_row_exclusion(self, volume_node):
        """Issue #56 Mode B follow-up: return ``(error_message, should_exclude)``
        for one volume node by delegating to :func:`mrml.validate_volume_node`.

        Lets the widget's ``refresh_results_against_scene`` re-validate
        each row without importing ``ZebrafishEmbryoAnalyzerLib.mrml``
        directly (widget.py is forbidden to do that — see
        ``test_widget_calls_update_results_table_not_mrml_directly``).

        Returns ``("", False)`` for any failure (no error, no exclude)
        so callers can apply the result without their own try/except.
        """
        if volume_node is None:
            return ("", False)
        try:
            from ZebrafishEmbryoAnalyzerLib.mrml import validate_volume_node
            err_field, _msg = validate_volume_node(volume_node)
            return (err_field or "", bool(err_field))
        except Exception:
            return ("", False)

    def setup_segmentation_staleness_observers(self):
        """Issue #56 follow-up: thin wrapper around the widget's
        ``setup_segmentation_staleness_observers`` method.

        ``_on_results_ready`` (and friends) calls
        ``self._logic.setup_segmentation_staleness_observers()`` because
        the widget should not import scene-observation internals. The
        actual observer-installation code lives on the
        ``ZebrafishEmbryoAnalyzerWidget`` instance — its parent owns
        ``addObserver``/``removeObserver`` via ``VTKObservationMixin``.
        The widget hands itself to the logic via ``_widget_ref`` in
        ``Widget.setup`` so this wrapper can delegate cleanly.

        Best-effort: returns silently on any error (no widget ref yet,
        widget's own try/except swallowed something, scene not ready)
        so callers do not need to wrap their own try/except.
        """
        widget = getattr(self, "_widget_ref", None)
        if widget is None:
            return
        install = getattr(widget, "setup_segmentation_staleness_observers", None)
        if install is None:
            return
        try:
            install()
        except Exception:
            pass

    def recompute_metrics_for_volume_node(self, volume_node):
        """Issue #42: rerun the segmentation→measurement pipeline for one
        volume node and update its attributes + segmentation node.

        Synchronous on the Slicer main thread (no threading — same
        pattern as the Run Analysis button). Reads pixel data from the
        volume node (no original-file dependency), runs the same
        ``analyse_images`` → ``apply_analysis_to_volume_node`` chain that
        #39 uses, then clears the stale flag.

        Returns the updated result dict (filename + new metric fields),
        or ``None`` if the recompute fails (e.g. model unavailable). The
        widget surfaces a status message on None.
        """
        try:
            import slicer
            import numpy as np
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                volume_node_to_pixels,
                apply_analysis_to_volume_node,
                clear_volume_node_stale,
            )
        except Exception:
            return None
        px = volume_node_to_pixels(volume_node)
        if px is None:
            return None
        params = self._recompute_params()
        try:
            from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
            results = analyse_images(
                ["__volume_node__"],
                params,
                per_image_callback=lambda _path, r: apply_analysis_to_volume_node(
                    r, volume_node, slicer.mrmlScene, params.get("um_per_px", 22.99)
                ),
            )
        except Exception:
            logging.exception(
                "ZebrafishEmbryoAnalyzer: recompute_metrics_for_volume_node failed"
            )
            return None
        if not results or results[0].get("error"):
            return None
        clear_volume_node_stale(volume_node)
        # Restore the un-excluded state — the user explicitly asked for
        # recompute, so the row no longer counts as "excluded because of
        # stale segmentation".
        try:
            volume_node.SetAttribute("ZebrafishAnalysis.exclude", "false")
            volume_node.SetAttribute("ZebrafishAnalysis.error", "")
        except Exception:
            pass
        # Surface the recomputed metrics in the widget-visible shape.
        r = results[0]
        return {
            "filename": volume_node.GetName() if hasattr(volume_node, "GetName") else "",
            "length": r.get("length"),
            "curvature": r.get("curvature"),
            "ratio": r.get("ratio"),
            "eye_area": r.get("eye_area"),
            "eye_diameter": r.get("eye_diameter"),
            "exclude": False,
            "error": "",
            "_volume_node": volume_node,
            "_volume_node_id": (
                volume_node.GetID() if hasattr(volume_node, "GetID") else ""
            ),
        }

    def _recompute_params(self):
        """Build the params dict for a single-image recompute.

        Mirrors the module defaults that the Run button uses — no
        inference-time override. Um-per-px comes from the parameter
        node's ``UM_PER_PX`` so the recompute matches the scale the
        original run used.
        """
        try:
            from ZebrafishEmbryoAnalyzerLib.widget import (
                PARAM_UM_PER_PX, PARAM_THRESHOLD,
                PARAM_LENGTH_ENABLED, PARAM_CURVATURE_ENABLED,
                PARAM_RATIO_ENABLED, PARAM_EYES_ENABLED,
                PARAM_MODEL_ID, _DEFAULT_MODEL_ID,
            )
            node = self.getParameterNode()
            params = {}
            if node is not None and hasattr(node, "GetParameter"):
                try:
                    params["um_per_px"] = float(node.GetParameter(PARAM_UM_PER_PX) or 22.99)
                except (TypeError, ValueError):
                    params["um_per_px"] = 22.99
                try:
                    params["threshold"] = float(node.GetParameter(PARAM_THRESHOLD) or 0.85)
                except (TypeError, ValueError):
                    params["threshold"] = 0.85
                params["length_enabled"] = (node.GetParameter(PARAM_LENGTH_ENABLED) == "true")
                params["curvature_enabled"] = (node.GetParameter(PARAM_CURVATURE_ENABLED) == "true")
                params["ratio_enabled"] = (node.GetParameter(PARAM_RATIO_ENABLED) == "true")
                params["eyes_enabled"] = (node.GetParameter(PARAM_EYES_ENABLED) == "true")
                params["model_id"] = node.GetParameter(PARAM_MODEL_ID) or _DEFAULT_MODEL_ID
            else:
                params = {
                    "um_per_px": 22.99,
                    "threshold": 0.85,
                    "length_enabled": True,
                    "curvature_enabled": True,
                    "ratio_enabled": True,
                    "eyes_enabled": True,
                    "model_id": _DEFAULT_MODEL_ID,
                }
            return params
        except Exception:
            return {"um_per_px": 22.99, "threshold": 0.85, "length_enabled": True,
                    "curvature_enabled": True, "ratio_enabled": True,
                    "eyes_enabled": True, "model_id": "general"}

    def _update_table_with_rows(self, rows):
        """Shared tail used by ``update_results_table`` and the reload path.

        Builds the vtk table before touching the MRML scene so a build
        failure cannot leave the existing table in a torn state.
        ``MRMLAdapterError`` from mrml.build_vtk_table (e.g. invalid
        values) propagates as-is.
        """
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                build_vtk_table,
                get_or_create_table_node,
            )
            completed_table = build_vtk_table(rows)
            param_node = self.getParameterNode()
            if param_node is None:
                return None
            table_node = get_or_create_table_node(param_node, slicer.mrmlScene)
            table_node.SetAndObserveTable(completed_table)
            return table_node
        except MRMLAdapterError:
            raise

    def show_gallery_selection_in_slice_view(self, result):
        """Mirror the selected gallery image into Slicer's slice views.

        Issue #56: replaces the singleton ``CurrentImage`` / ``CurrentSegmentation``
        mechanism. Resolves ``result`` to its already-existing per-image
        volume node (issue #38), sets it as the slice-view background, and
        toggles its segmentation display visibility on — hiding any
        previously-shown segmentation so the slice views do not accumulate
        stacked overlays across gallery clicks.

        Returns ``None`` (no exceptions propagate). Tolerates:

        * a result without a volume node (decode-failure / error row),
        * a result whose analysis hasn't run yet (volume exists but no
          segmentation reference attached),
        * slicer / mrml / display-node bindings that disagree across
          Slicer versions — every step is guarded individually.

        Parameters
        ----------
        result : dict | None
            A result dict from ``self._results``. May carry
            ``"_volume_node"`` (scene-reload path) or just ``"filename"``
            (post-Run Analysis path — volume node looked up by display
            name match).
        """
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                ROLE_ZEBRAFISH_SEGMENTATION,
                find_tracked_volume_node_by_filename,
                set_slice_viewer_background,
                set_segmentation_visibility,
            )
        except Exception:
            return None

        param_node = self.getParameterNode()
        scene = getattr(slicer, "mrmlScene", None)
        if param_node is None or scene is None or not result:
            return None

        volume_node = result.get("_volume_node")
        if volume_node is None:
            volume_node = find_tracked_volume_node_by_filename(
                param_node, scene, (result or {}).get("filename")
            )
        if volume_node is None:
            return None

        set_slice_viewer_background(volume_node)

        seg_node = None
        try:
            seg_node = volume_node.GetNodeReference(ROLE_ZEBRAFISH_SEGMENTATION)
        except Exception:
            seg_node = None

        # Re-read the "previously visible" id from a parameter-node
        # attribute when present, so a fresh ``ZebrafishEmbryoAnalyzerLogic``
        # instance on the same scene (e.g. after a Slicer restart) does
        # not show two segmentations stacked on the first click.
        prev_seg_id = None
        try:
            prev_seg_id = param_node.GetParameter(
                "ZebrafishAnalysis.previousVisibleSegmentationId"
            )
        except Exception:
            prev_seg_id = None
        # Slicer returns "" for unset parameter strings — coerce everything
        # falsy to None so the "hide previous" branch only runs when an id
        # was actually recorded.
        if not prev_seg_id:
            prev_seg_id = None

        new_seg_id = None
        try:
            new_seg_id = seg_node.GetID() if seg_node is not None else None
        except Exception:
            new_seg_id = None

        if prev_seg_id and prev_seg_id != new_seg_id:
            try:
                prev_seg = scene.GetNodeByID(prev_seg_id)
            except Exception:
                prev_seg = None
            set_segmentation_visibility(prev_seg, False)

        if seg_node is not None:
            set_segmentation_visibility(seg_node, True)

        try:
            param_node.SetParameter(
                "ZebrafishAnalysis.previousVisibleSegmentationId", new_seg_id or ""
            )
        except Exception:
            pass
        return None

    def create_image_volume_nodes(self, batches):
        """Create one ``vtkMRMLVectorVolumeNode`` per readable image (issue #38).

        Called by the widget after its pre-flight readability check has
        already loaded every pixel array into a result stub. No second
        disk read happens here; the pixel arrays are reused directly.

        Each batch is attempted independently. A failure in one image
        never aborts the remaining batches (CLAUDE.md "avoid partial
        results appearing as successful" rule): per-image failures are
        collected into ``failed``, and a fully fatal setup failure
        (slicer / MRML unavailable) raises ``MRMLAdapterError`` so the
        widget can surface it.

        Parameters
        ----------
        batches : list[dict]
            Each dict has keys:
              - ``"filename"`` (str) — display-name hint for the node.
              - ``"image_rgb"`` (``numpy.ndarray``) — uint8 (H, W, 3).
              - ``"um_per_px"`` (float) — physical scale metadata.

        Returns
        -------
        tuple ``(created, failed)``
            ``created`` (int) — number of nodes successfully added to the
            scene; may be 0 when the parameter node was unavailable.
            ``failed`` (list[str]) — basenames of images whose volume node
            creation raised; the widget surfaces them in one capped summary.
        """
        if not batches:
            return 0, []
        # Local imports so the module loads even if slicer / mrml are
        # unavailable (e.g. during interpreter startup outside Slicer).
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import create_image_volume_node
        except Exception as exc:
            from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
            raise MRMLAdapterError(
                f"Failed to create image volume nodes: {exc}"
            ) from exc

        param_node = self.getParameterNode()
        if param_node is None:
            return 0, [
                b.get("filename") or "ZebrafishEmbryoAnalyzer Image"
                for b in batches
            ]
        scene = slicer.mrmlScene

        import logging
        created = 0
        failed = []
        for batch in batches:
            image_rgb = batch.get("image_rgb")
            filename = batch.get("filename") or "ZebrafishEmbryoAnalyzer Image"
            if image_rgb is None:
                failed.append(filename)
                continue
            um_per_px = float(batch.get("um_per_px", 22.99))
            try:
                create_image_volume_node(image_rgb, um_per_px, filename, param_node, scene)
                created += 1
            except Exception as exc:
                # Half-success guard: keep going, accumulate the failed
                # filenames, and let the caller surface them all at once.
                logging.exception(
                    "ZebrafishEmbryoAnalyzer: image volume node failed for %s: %s",
                    filename, exc,
                )
                failed.append(filename)

        return created, failed

    def replace_image_volume_nodes(self):
        """Remove every volume node owned by this batch from the scene (#38).

        Used by ``ZebrafishEmbryoAnalyzerMainWidget._set_queue`` so that
        loading a new folder or file selection replaces the previous one in
        the MRML scene rather than accumulating orphans in the Data module.
        Returns the number of top-level image nodes removed. Never raises —
        scene-cleanup failures are logged and swallowed so a transient scene
        glitch cannot break the user-facing load flow.
        """
        try:
            import logging
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import remove_all_image_volume_nodes
            param_node = self.getParameterNode()
            if param_node is None:
                return 0
            return remove_all_image_volume_nodes(param_node, slicer.mrmlScene)
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup, log only
            try:
                import logging
                logging.exception(
                    "ZebrafishEmbryoAnalyzer: replace_image_volume_nodes failed: %s", exc
                )
            except Exception:
                pass
            return 0

    def apply_results_to_tracked_volume_nodes(self, results, um_per_px):
        """Issue #39 follow-up: write per-image segmentation/markups/attributes
        for a batch of results produced by the subprocess-based Run Analysis
        flow (``inference_runner``).

        ``analyse_images``'s ``per_image_callback`` (the original streaming
        hook #39 was scoped against) only fires for the in-process
        ``recompute_metrics_for_volume_node`` path — the main Run Analysis
        button runs inference out-of-process via ``inference_runner`` and only
        gets results back in one batch once the subprocess exits, so there is
        no per-image callback to hook there. This applies the same
        ``apply_analysis_to_volume_node`` write, once per result, matched to
        its already-existing (#38 eager) volume node by filename.

        Returns the number of results successfully applied. Never raises —
        a missing match or per-node failure is logged and skipped so one bad
        image cannot block the rest of the batch.
        """
        try:
            import logging
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                list_tracked_volume_nodes,
                apply_analysis_to_volume_node,
            )
        except Exception:
            return 0
        param_node = self.getParameterNode()
        if param_node is None:
            return 0
        scene = slicer.mrmlScene
        nodes_by_name = {}
        for node in list_tracked_volume_nodes(param_node, scene):
            try:
                nodes_by_name[node.GetName()] = node
            except Exception:
                continue
        applied = 0
        for result in results or []:
            filename = (result or {}).get("filename")
            node = nodes_by_name.get(filename)
            if node is None:
                continue
            try:
                apply_analysis_to_volume_node(result, node, scene, um_per_px)
                applied += 1
            except Exception:
                logging.exception(
                    "ZebrafishEmbryoAnalyzer: apply_results_to_tracked_volume_nodes "
                    "failed for %s", filename,
                )
        return applied
