"""Safety tests for G1 no-thread and no-hidden-preload behavior."""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).parent.parent
PRODUCTION_ROOT = ROOT / "ZebrafishEmbryoAnalyzer"
WIDGET_PATH = PRODUCTION_ROOT / "ZebrafishEmbryoAnalyzerLib" / "widget.py"
MAIN_PATH = PRODUCTION_ROOT / "ZebrafishEmbryoAnalyzer.py"


@pytest.fixture
def widget_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "qt", MagicMock())
    monkeypatch.setitem(sys.modules, "ctk", MagicMock())
    slicer = MagicMock()
    slicer.util.mainWindow.return_value = None
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    import importlib
    import ZebrafishEmbryoAnalyzerLib.widget as module
    return importlib.reload(module)


def _method_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _calls_in_file(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_no_python_thread_creation_in_production_code():
    """Production extension code must not create Python background threads."""
    violations = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_thread_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "threading":
                for alias in node.names:
                    if alias.name == "Thread":
                        imported_thread_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "Thread"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "threading"
                ):
                    violations.append((path, node.lineno, "threading.Thread"))
                elif isinstance(func, ast.Name) and func.id in imported_thread_names:
                    violations.append((path, node.lineno, "Thread"))
    assert violations == []


def test_startup_prewarm_imports_removed():
    assert "_prewarm_imports" not in _method_names(MAIN_PATH)


def test_widget_model_preload_methods_removed():
    names = _method_names(WIDGET_PATH)
    assert "_start_preload" not in names
    assert "_preload_cached_models" not in names


def test_no_startup_model_preload_timer_or_signal():
    calls = _calls_in_file(WIDGET_PATH)
    forbidden = []
    for call in calls:
        text = ast.unparse(call)
        if "_preload_cached_models" in text or "_start_preload" in text:
            forbidden.append((call.lineno, text))
        if "singleShot" in text and "500" in text:
            forbidden.append((call.lineno, text))
    assert forbidden == []


def test_model_selection_only_updates_parameter_node():
    source = WIDGET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    connect_body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_connect_signals":
            connect_body = "".join(lines[node.lineno - 1:node.end_lineno])
            break
    assert connect_body is not None
    assert "_notify_settings_changed" in connect_body
    assert "preload" not in connect_body.lower()


def test_run_analysis_starts_download_before_analysis_when_models_missing(widget_module):
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 0
    w._image_paths = ["/tmp/fish.png"]
    w._model_combo = MagicMock()
    w._model_combo.currentData = "general"
    w._chk_length = MagicMock()
    w._chk_curvature = MagicMock()
    w._chk_ratio = MagicMock()
    w._chk_eyes = MagicMock()
    w._chk_hitl = MagicMock()
    for chk in (w._chk_length, w._chk_curvature, w._chk_ratio, w._chk_eyes, w._chk_hitl):
        chk.isChecked.return_value = True
    w._threshold_slider = MagicMock()
    w._threshold_slider.value = 0.85
    w._um_per_px = MagicMock()
    w._um_per_px.value = 22.99
    w._prompt_download_models = MagicMock(return_value=True)
    w._missing_required_models = MagicMock(return_value=[{"label": "Body"}])
    w._start_model_download = MagicMock()
    w._start_inference_process = MagicMock()

    w._on_run()

    w._start_model_download.assert_called_once()
    w._start_inference_process.assert_not_called()


def test_run_analysis_starts_inference_when_models_cached(widget_module):
    """When models are already cached, _start_inference_process is called directly."""
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 0
    w._image_paths = ["/tmp/fish.png"]
    w._model_combo = MagicMock()
    w._model_combo.currentData = "general"
    w._chk_length = MagicMock()
    w._chk_curvature = MagicMock()
    w._chk_ratio = MagicMock()
    w._chk_eyes = MagicMock()
    w._chk_hitl = MagicMock()
    for chk in (w._chk_length, w._chk_curvature, w._chk_ratio, w._chk_eyes, w._chk_hitl):
        chk.isChecked.return_value = True
    w._threshold_slider = MagicMock()
    w._threshold_slider.value = 0.85
    w._um_per_px = MagicMock()
    w._um_per_px.value = 22.99
    w._missing_required_models = MagicMock(return_value=[])
    w._start_inference_process = MagicMock()

    w._on_run()

    w._start_inference_process.assert_called_once()


def test_downloader_success_rechecks_cache_before_analysis(widget_module):
    import qt as _qt
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = 1
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._run_status_label = MagicMock()
    w._missing_required_models = MagicMock(return_value=[])
    w._start_inference_process = MagicMock()
    missing = [{"label": "Body"}]
    params = {"model_id": "general"}

    controller = MagicMock()

    def fake_start(entries, callback, parent=None):
        w._active_downloader = controller
        callback(True, "succeeded", None, controller)
        return controller

    with patch("ZebrafishEmbryoAnalyzerLib.model_downloader.start_model_download", fake_start):
        w._start_model_download(missing, "general", params, token=1)

    w._missing_required_models.assert_called_with("general")
    # Analysis is now deferred via QTimer.singleShot, not called directly.
    w._start_inference_process.assert_not_called()
    _qt.QTimer.singleShot.assert_called_once()
    call_args = _qt.QTimer.singleShot.call_args
    assert call_args[0][0] == 0
    deferred = call_args[0][1]
    assert callable(deferred)
    # Call the deferred — token matches so analysis runs.
    deferred()
    w._start_inference_process.assert_called_once_with("general", params, 1)


def test_downloader_failure_does_not_start_analysis(widget_module):
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = 1
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._run_status_label = MagicMock()
    w._missing_required_models = MagicMock(return_value=[])
    w._start_inference_process = MagicMock()
    controller = MagicMock()

    def fake_start(entries, callback, parent=None):
        w._active_downloader = controller
        callback(False, "failed", "offline", controller)
        return controller

    with patch("ZebrafishEmbryoAnalyzerLib.model_downloader.start_model_download", fake_start):
        w._start_model_download([{"label": "Body"}], "general", {}, token=1)

    w._start_inference_process.assert_not_called()


# ---------------------------------------------------------------------------
# New G1 token / deferred-analysis tests
# ---------------------------------------------------------------------------

def _make_download_widget(widget_module, token=1):
    """Return a minimal widget shell wired for _start_model_download tests."""
    import qt as _qt
    _qt.QTimer.singleShot.reset_mock()
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = token
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._run_status_label = MagicMock()
    w._missing_required_models = MagicMock(return_value=[])
    w._start_inference_process = MagicMock()
    return w


def _fire_download_success(widget_module, w, token=1):
    """Call _start_model_download with a fake downloader that immediately fires success."""
    controller = MagicMock()

    def fake_start(entries, callback, parent=None):
        w._active_downloader = controller
        callback(True, "succeeded", None, controller)
        return controller

    with patch("ZebrafishEmbryoAnalyzerLib.model_downloader.start_model_download", fake_start):
        w._start_model_download([{"label": "Body"}], "general", {"model_id": "general"}, token=token)

    return controller


def test_download_success_schedules_deferred_continuation_not_direct_call(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    # QTimer.singleShot must have been called exactly once with delay=0 and a callable.
    _qt.QTimer.singleShot.assert_called_once()
    args = _qt.QTimer.singleShot.call_args[0]
    assert args[0] == 0
    assert callable(args[1])

    # Analysis must NOT have been called synchronously inside _finished.
    w._start_inference_process.assert_not_called()


def test_download_success_does_not_call_analysis_inside_finished_callback(widget_module):
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)
    assert w._start_inference_process.call_count == 0


def test_deferred_continuation_calls_analysis_when_token_matches(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    deferred = _qt.QTimer.singleShot.call_args[0][1]
    deferred()

    w._start_inference_process.assert_called_once_with(
        "general", {"model_id": "general"}, 1
    )


def test_disposed_before_deferred_continuation_prevents_analysis(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    deferred = _qt.QTimer.singleShot.call_args[0][1]
    w._disposed = True
    deferred()

    w._start_inference_process.assert_not_called()
    # UI must be reset to idle.
    w._run_stack.setCurrentIndex.assert_called_with(0)


def test_stale_token_before_deferred_continuation_prevents_analysis(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    deferred = _qt.QTimer.singleShot.call_args[0][1]
    # Simulate cancel / newer run invalidating the token.
    w._run_token = 2
    deferred()

    w._start_inference_process.assert_not_called()
    w._run_stack.setCurrentIndex.assert_called_with(0)


def test_start_model_download_exception_restores_run_ui(widget_module):
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = 1
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()

    def raising_start(entries, callback, parent=None):
        raise RuntimeError("network init failed")

    with patch("ZebrafishEmbryoAnalyzerLib.model_downloader.start_model_download", raising_start):
        w._start_model_download([{"label": "Body"}], "general", {}, token=1)

    w._run_stack.setCurrentIndex.assert_called_with(0)
    assert w._active_downloader is None


def test_no_processevents_in_download_to_analysis_path():
    source = WIDGET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    target_functions = {"_start_model_download", "_start_inference_process"}
    found_violations = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in target_functions:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    text = ast.unparse(child)
                    if "processEvents" in text:
                        found_violations.append((node.name, child.lineno, text))

    assert found_violations == [], f"processEvents found in: {found_violations}"


def test_set_queue_cancels_active_runner(widget_module):
    """_set_queue must cancel any in-flight InferenceController."""
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 0
    w._deps_ok = True
    w._results = []
    w._run_stack = MagicMock()
    runner = MagicMock()
    w._active_runner = runner
    w._gallery = MagicMock()
    w._queue_list = MagicMock()
    w._detail = MagicMock()
    w._excluded = set()
    w._results_tab = MagicMock()
    w._tabs = MagicMock()
    w._btn_run = MagicMock()
    w._load_originals = MagicMock()
    w._load_result_label = MagicMock()  # Issue #62

    w._set_queue([])

    runner.cancel.assert_called_once()
    assert w._active_runner is None


def test_set_queue_bumps_token_before_cancel(widget_module):
    """Token increment happens before runner cancel so callback sees stale token."""
    tokens_at_cancel = []

    def _cancel():
        tokens_at_cancel.append(w._run_token)

    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 1
    w._deps_ok = True
    w._results = []
    w._run_stack = MagicMock()
    w._gallery = MagicMock()
    w._queue_list = MagicMock()
    w._detail = MagicMock()
    w._excluded = set()
    w._results_tab = MagicMock()
    w._tabs = MagicMock()
    w._btn_run = MagicMock()
    w._load_originals = MagicMock()
    w._load_result_label = MagicMock()  # Issue #62
    runner = MagicMock()
    runner.cancel.side_effect = _cancel
    w._active_runner = runner

    w._set_queue([])

    assert tokens_at_cancel == [2]  # token already bumped when cancel() was called


def test_set_queue_increments_run_token(widget_module):
    """_set_queue must increment _run_token as its first action."""
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 5
    w._deps_ok = True
    w._image_paths = []
    w._queue_list = MagicMock()
    w._results = []
    w._excluded = set()
    w._detail = MagicMock()
    w._results_tab = MagicMock()
    w._gallery = MagicMock()
    w._tabs = MagicMock()
    w._um_per_px = MagicMock()
    w._btn_run = MagicMock()
    w._load_result_label = MagicMock()  # Issue #62

    # _load_originals needs to be a no-op (it calls cv2 etc.)
    w._load_originals = MagicMock()

    w._set_queue([])

    assert w._run_token == 6


# ---------------------------------------------------------------------------
# Issue #59: cancel mid-batch preserves completed-image results
# ---------------------------------------------------------------------------

def _make_runner_finished_widget(widget_module, token=1):
    """Minimal widget shell with the attributes _handle_runner_finished reads."""
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._disposed = False
    w._run_token = token
    w._active_runner = MagicMock(name="active_runner")
    w._refresh_settings_actions = MagicMock()
    w._run_stack = MagicMock()
    w._results = []
    w._on_results_ready = MagicMock()
    w._try_update_mrml_table = MagicMock()
    w._categorize_inference_error = MagicMock(return_value="formatted error")
    w._run_temp_files = []  # issue #61: _handle_runner_finished always clears these
    return w


def test_handle_runner_finished_cancel_with_results_applies_partial(widget_module):
    """Cancelled batch with non-empty controller.results must trigger apply path.

    Issue #59: prior to this, a cancel mid-batch wiped controller.results and
    the widget returned early — segmentation/attributes for finished images
    were silently dropped. Now the same apply-path as a successful run fires.
    """
    import slicer as _slicer
    w = _make_runner_finished_widget(widget_module, token=1)

    runner = MagicMock()
    runner.results = [{"filename": "fish_0.png", "length": 100.0}]
    w._active_runner = runner

    w._handle_runner_finished(success=False, state="cancelled", message=None,
                              controller=runner, token=1)

    # Token guard passes, partial results were applied via the success path.
    w._on_results_ready.assert_called_once()
    w._try_update_mrml_table.assert_called_once_with(runner.results)
    assert w._results == runner.results
    # UI returned to idle stack.
    w._run_stack.setCurrentIndex.assert_called_with(0)
    # No error dialog for a cancel.
    _slicer.util.errorDisplay.assert_not_called()


def test_handle_runner_finished_restores_queue_order_and_original_filenames(widget_module):
    """analyse_images (logic.py) iterates sorted(image_paths) internally, so
    a successful run's results come back sorted by whatever path was sent
    to the worker, not the user's original queue order — and for a
    materialized post-reload image (issue #61), that sent path's basename
    is a random temp filename, not the real one. _handle_runner_finished
    must restore both the original queue order and the original filenames
    (bug found while testing #61: gallery order changed and captions were
    renamed to temp filenames after Run Analysis on a reloaded scene).
    """
    w = _make_runner_finished_widget(widget_module, token=1)
    w._try_apply_results_to_volume_nodes = MagicMock()
    # Queue order: fish_0, fish_1, fish_2. fish_1 was a reload-only (no file
    # on disk) row, materialized to a random-basename temp file.
    w._image_paths = ["/real/fish_0.png", "/real/fish_1.png", "/real/fish_2.png"]
    w._run_sent_paths = [
        "/real/fish_0.png",
        "/tmp/zebrafish_reload_ab12cd.png",
        "/real/fish_2.png",
    ]
    w._results = [{"filename": "fish_0.png"}, {"filename": "fish_1.png"}, {"filename": "fish_2.png"}]

    runner = MagicMock()
    # Worker returns results sorted by sent path, NOT queue order — and the
    # reload row's filename is the temp file's basename.
    runner.results = [
        {"filename": "fish_0.png", "image_path": "/real/fish_0.png", "length": 1.0},
        {"filename": "fish_2.png", "image_path": "/real/fish_2.png", "length": 3.0},
        {"filename": "zebrafish_reload_ab12cd.png",
         "image_path": "/tmp/zebrafish_reload_ab12cd.png", "length": 2.0},
    ]
    w._active_runner = runner

    w._handle_runner_finished(success=True, state="succeeded", message=None,
                              controller=runner, token=1)

    assert [r["filename"] for r in w._results] == ["fish_0.png", "fish_1.png", "fish_2.png"]
    assert [r["length"] for r in w._results] == [1.0, 2.0, 3.0]


def test_handle_runner_finished_cancel_keeps_unprocessed_images_raw(widget_module):
    """Issue #59 follow-up: images the worker never reached before Cancel
    must stay in self._results (raw stub, no segmentation/metrics) instead
    of disappearing from the gallery/table — but must NOT be passed to
    _try_apply_results_to_volume_nodes, since running apply_analysis on a
    mask-less stub would create an empty segmentation node.
    """
    w = _make_runner_finished_widget(widget_module, token=1)
    w._try_apply_results_to_volume_nodes = MagicMock()
    w._image_paths = ["/tmp/fish_0.png", "/tmp/fish_1.png", "/tmp/fish_2.png"]
    w._run_sent_paths = list(w._image_paths)

    # Three images were queued; only the first had finished when Cancel hit.
    stub_1 = {"filename": "fish_1.png", "original": "raw1", "mask": None, "error": None, "length": None}
    stub_2 = {"filename": "fish_2.png", "original": "raw2", "mask": None, "error": None, "length": None}
    w._results = [
        {"filename": "fish_0.png", "original": "raw0", "mask": None, "error": None, "length": None},
        stub_1,
        stub_2,
    ]

    runner = MagicMock()
    runner.results = [{"filename": "fish_0.png", "image_path": "/tmp/fish_0.png",
                        "length": 100.0, "mask": "computed"}]
    w._active_runner = runner

    w._handle_runner_finished(success=False, state="cancelled", message=None,
                              controller=runner, token=1)

    # self._results keeps all three: the completed one plus the two untouched stubs.
    assert w._results == [runner.results[0], stub_1, stub_2]
    # Only the completed result was handed to the MRML-apply step.
    w._try_apply_results_to_volume_nodes.assert_called_once_with(runner.results)
    w._on_results_ready.assert_called_once()
    w._try_update_mrml_table.assert_called_once_with(w._results)


def test_handle_runner_finished_cancel_with_empty_results_is_noop(widget_module):
    """Cancelled before any image completed — nothing to apply, no error shown.

    The widget must NOT call _on_results_ready or _try_update_mrml_table in
    this case (there's no segmentation to render), but also must not raise.
    """
    import slicer as _slicer
    w = _make_runner_finished_widget(widget_module, token=1)

    runner = MagicMock()
    runner.results = []  # worker hadn't finished any image yet
    w._active_runner = runner

    w._handle_runner_finished(success=False, state="cancelled", message=None,
                              controller=runner, token=1)

    w._on_results_ready.assert_not_called()
    w._try_update_mrml_table.assert_not_called()
    # UI still returned to idle.
    w._run_stack.setCurrentIndex.assert_called_with(0)
    _slicer.util.errorDisplay.assert_not_called()


def test_handle_runner_finished_failure_shows_error(widget_module):
    """Non-cancel failure (state != 'cancelled', success=False) → error dialog."""
    import slicer as _slicer
    w = _make_runner_finished_widget(widget_module, token=1)

    runner = MagicMock()
    runner.results = []
    w._active_runner = runner

    w._handle_runner_finished(success=False, state="failed", message="boom",
                              controller=runner, token=1)

    w._on_results_ready.assert_not_called()
    w._try_update_mrml_table.assert_not_called()
    w._categorize_inference_error.assert_called_once_with("boom", runner)
    _slicer.util.errorDisplay.assert_called_once_with("formatted error")


def test_handle_runner_finished_stale_token_does_not_apply(widget_module):
    """Stale runner (token mismatch) must discard results without applying."""
    w = _make_runner_finished_widget(widget_module, token=2)

    runner = MagicMock()
    runner.results = [{"filename": "fish_0.png"}]
    w._active_runner = runner

    # Token at finish time is 1, but current _run_token is 2 → stale.
    w._handle_runner_finished(success=True, state="succeeded", message=None,
                              controller=runner, token=1)

    w._on_results_ready.assert_not_called()
    w._try_update_mrml_table.assert_not_called()
    assert w._results == []  # unchanged
    w._run_stack.setCurrentIndex.assert_called_with(0)


# ---------------------------------------------------------------------------
# Issue #77: Detail tab must sync when reached via the tab bar, not just a
# gallery click — _on_results_ready must call show_result the same way the
# gallery-click and scene-reload paths already do.
# ---------------------------------------------------------------------------

def _make_results_ready_widget(widget_module, results):
    """Minimal widget shell with the attributes the real _on_results_ready reads."""
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._results = results
    w._detail = MagicMock()
    w._gallery = MagicMock()
    w._results_tab = MagicMock()
    w._tabs = MagicMock()
    w._logic = MagicMock()
    w._refresh_detail_recompute_button = MagicMock()
    w._excluded = set()
    return w


def test_on_results_ready_syncs_detail_tab_when_results_present(widget_module):
    """_on_results_ready must call _detail.show_result(0, results) and set
    _current_detail_idx, mirroring the gallery-click (_on_gallery_select) and
    scene-reload paths — otherwise Detail tab shows stale/placeholder state
    when reached via the tab bar right after a fresh Run Analysis.
    """
    import slicer as _slicer
    _slicer.util.warningDisplay = MagicMock()
    results = [{"filename": "fish_0.png"}, {"filename": "fish_1.png"}]
    w = _make_results_ready_widget(widget_module, results)

    w._on_results_ready()

    w._detail.show_result.assert_called_once_with(0, results)
    assert w._current_detail_idx == 0


def test_on_results_ready_no_detail_sync_when_no_results(widget_module):
    """Empty results (e.g. an all-failed batch) must not call show_result
    with an out-of-range index."""
    import slicer as _slicer
    _slicer.util.warningDisplay = MagicMock()
    w = _make_results_ready_widget(widget_module, [])

    w._on_results_ready()

    w._detail.show_result.assert_not_called()


def test_handle_runner_finished_cancel_with_partial_results_syncs_detail_tab(widget_module):
    """The cancel-with-partial-results path shares _on_results_ready with the
    success path (issue #59), so it must get the same Detail-tab sync fix —
    not a separately-maintained code path that could drift out of sync again.
    """
    import slicer as _slicer
    w = _make_runner_finished_widget(widget_module, token=1)
    w._on_results_ready = widget_module.ZebrafishEmbryoAnalyzerMainWidget._on_results_ready.__get__(w)
    w._detail = MagicMock()
    w._gallery = MagicMock()
    w._results_tab = MagicMock()
    w._tabs = MagicMock()
    w._logic = MagicMock()
    w._refresh_detail_recompute_button = MagicMock()
    w._try_apply_results_to_volume_nodes = MagicMock()
    w._reorder_and_rename_results = lambda results, fill_missing_from=None: results
    _slicer.util.warningDisplay = MagicMock()

    runner = MagicMock()
    runner.results = [{"filename": "fish_0.png"}]
    w._active_runner = runner

    w._handle_runner_finished(success=False, state="cancelled", message=None,
                              controller=runner, token=1)

    w._detail.show_result.assert_called_once_with(0, [{"filename": "fish_0.png"}])
    assert w._current_detail_idx == 0
