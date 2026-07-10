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
            self._main.prompt_install_if_missing()

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

    def _on_scene_end_import(self, caller=None, event=None):
        # Pick up parameter node values from the newly loaded scene.
        self.initializeParameterNode()

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

    def update_current_image_node(self, result, um_per_px):
        """Create or update the MRML vector volume node for the current image.

        Returns None silently if result["original"] is None (stub or error row).
        Raises MRMLAdapterError on MRML or VTK failure.
        Must be called on the Slicer main thread only.
        Does not use result["spacing"] — that is calibrated to 256x256 mask space.
        """
        original = result.get("original") if result else None
        if original is None:
            return None
        try:
            from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                get_or_create_image_node,
                update_image_node,
            )
            param_node = self.getParameterNode()
            if param_node is None:
                return None
            node = get_or_create_image_node(param_node, slicer.mrmlScene)
            update_image_node(original, um_per_px, node)
            return node
        except MRMLAdapterError:
            raise
        except Exception as exc:
            raise MRMLAdapterError(
                f"Failed to update current image node: {exc}"
            ) from exc

    def update_current_segmentation_node(self, result, um_per_px):
        """Create or update the MRML segmentation node for the current image's masks.

        Returns None silently if result["original"] is None (stub or error row).
        Raises MRMLAdapterError on MRML or VTK failure.
        Must be called on the Slicer main thread only.
        """
        original = result.get("original") if result else None
        if original is None:
            return None
        try:
            from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                get_or_create_segmentation_node,
                update_segmentation_node,
            )
            param_node = self.getParameterNode()
            if param_node is None:
                return None
            image_node = param_node.GetNodeReference("CurrentImage")
            node = get_or_create_segmentation_node(param_node, slicer.mrmlScene)
            update_segmentation_node(result, um_per_px, node, image_node=image_node)
            return node
        except MRMLAdapterError:
            raise
        except Exception as exc:
            raise MRMLAdapterError(
                f"Failed to update current segmentation node: {exc}"
            ) from exc

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
