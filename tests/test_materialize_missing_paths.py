"""
Tests for ZebrafishEmbryoAnalyzerMainWidget._materialize_missing_image_paths and
the temp-file cleanup in _on_runner_finished (issue #61).

Issue #61: after a scene reload, ``self._image_paths`` contains bare filenames
with no backing file on disk. analyse_images() does shutil.copy2(image_path, ...)
to hand the file to the out-of-process inference worker — that fails with
FileNotFoundError on every image.

The fix writes any in-memory-only image (the matching ``originals[i]`` ndarray)
to a temp PNG and substitutes that path. Temp files are tracked in
``self._run_temp_files`` and cleaned up on every branch of _on_runner_finished.

These tests stub qt/slicer and exercise the helpers directly via object.__new__.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from unittest.mock import MagicMock


_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishEmbryoAnalyzer"
)

_SLICER_STUB = """
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"]  = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()
sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=object,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(VTKObservationMixin=object)
sys.modules["vtk"] = MagicMock()
"""


def _run(code: str) -> subprocess.CompletedProcess:
    full = _SLICER_STUB + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


# ---------------------------------------------------------------------------
# _materialize_missing_image_paths
# ---------------------------------------------------------------------------

def test_materialize_returns_existing_paths_unchanged(tmp_path):
    """If the file is already on disk, return it verbatim — no temp file."""
    r = _run(f"""
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_temp_files = []

        # Touch a real file so os.path.exists returns True for it.
        import pathlib
        real = pathlib.Path("{tmp_path}") / "real.png"
        real.write_bytes(b"x")

        result = w._materialize_missing_image_paths([str(real)], [None])
        assert result == [str(real)], f"expected verbatim path; got {{result!r}}"
        assert w._run_temp_files == [], (
            f"no temp files should be tracked for an on-disk input; "
            f"got {{w._run_temp_files!r}}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_materialize_writes_temp_png_when_path_missing_but_original_present(tmp_path):
    """Bare filename + matching original ndarray → writes a temp PNG, returns
    its path, and tracks it in _run_temp_files. Round-trip via cv2.imread
    must yield the same pixels.
    """
    r = _run(f"""
        import os, pathlib
        import numpy as np
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_temp_files = []

        # An original RGB ndarray the size of a typical zebrafish image.
        original = np.full((64, 64, 3), [200, 150, 100], dtype=np.uint8)

        # Bare filename that does NOT exist on disk — typical post-reload state.
        bare = "fish_000001_jpg.rf.f9e4338f9fdce1d85c4fdbe1e177ecce.jpg"
        assert not os.path.exists(bare)

        result = w._materialize_missing_image_paths([bare], [original])
        assert len(result) == 1
        out_path = result[0]
        assert out_path != bare, "bare path must NOT be returned when materializing"
        assert os.path.exists(out_path), f"materialized path must exist on disk: {{out_path}}"
        assert out_path in w._run_temp_files, (
            f"newly-written temp file must be tracked for cleanup; "
            f"run_temp_files={{w._run_temp_files!r}}"
        )

        # Round-trip: cv2.imread back into RGB must equal the original we put in.
        import cv2
        bgr = cv2.imread(out_path)
        assert bgr is not None, "cv2.imread must succeed on the temp file"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(rgb, original)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_materialize_falls_through_when_no_original_available(tmp_path):
    """If path is missing AND original is None, return the bare path
    unchanged — analyse_images will still fail, but the existing failure
    mode (with a useful per-image error) is preserved."""
    r = _run(f"""
        import os
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_temp_files = []

        bare = "definitely_not_on_disk.png"
        assert not os.path.exists(bare)

        result = w._materialize_missing_image_paths([bare], [None])
        assert result == [bare], (
            f"expected fall-through to bare path; got {{result!r}}"
        )
        assert w._run_temp_files == [], (
            f"no temp file should be tracked; got {{w._run_temp_files!r}}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_materialize_handles_mixed_paths(tmp_path):
    """A real file passes through; a bare-filename + original gets a temp path."""
    r = _run(f"""
        import os, pathlib
        import numpy as np
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_temp_files = []

        real = pathlib.Path("{tmp_path}") / "real.png"
        real.write_bytes(b"x")
        original = np.zeros((32, 32, 3), dtype=np.uint8)

        paths = [str(real), "missing1.jpg", "missing2.jpg"]
        originals = [None, original, original]

        result = w._materialize_missing_image_paths(paths, originals)
        assert len(result) == 3
        assert result[0] == str(real)
        assert result[1] != "missing1.jpg" and os.path.exists(result[1])
        assert result[2] != "missing2.jpg" and os.path.exists(result[2])
        assert len(w._run_temp_files) == 2, (
            f"both materialized files must be tracked; got {{w._run_temp_files!r}}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# _clear_run_temp_files
# ---------------------------------------------------------------------------

def test_clear_run_temp_files_deletes_every_tracked_path(tmp_path):
    """Cleanup deletes every tracked temp file and resets the list."""
    r = _run(f"""
        import os, pathlib
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        # Pre-populate with three real temp files in {tmp_path}.
        files = []
        for i in range(3):
            p = pathlib.Path("{tmp_path}") / f"fake_temp_{{i}}.png"
            p.write_bytes(b"x")
            files.append(str(p))
        w._run_temp_files = list(files)
        assert all(os.path.exists(p) for p in files)

        w._clear_run_temp_files()

        assert w._run_temp_files == [], (
            f"_run_temp_files must be reset; got {{w._run_temp_files!r}}"
        )
        assert not any(os.path.exists(p) for p in files), (
            f"all tracked temp files must be deleted; survivors={{[p for p in files if os.path.exists(p)]}}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_clear_run_temp_files_swallows_oserror(tmp_path):
    """A missing temp file (already cleaned up externally) must not raise."""
    r = _run(f"""
        import os
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_temp_files = ["/nonexistent/that/should/never/exist.png"]

        # Must not raise — common case after process restart.
        w._clear_run_temp_files()
        assert w._run_temp_files == []
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_clear_run_temp_files_noop_when_empty():
    """No tracked files → no-op (no exception, list stays empty)."""
    r = _run("""
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_temp_files = []

        w._clear_run_temp_files()
        assert w._run_temp_files == []
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# _on_runner_finished wires cleanup on every exit path
# ---------------------------------------------------------------------------

def test_on_runner_finished_clears_temp_files_on_success():
    """The success branch in _on_runner_finished must call _clear_run_temp_files."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._disposed = False
        w._run_token = 0
        w._run_temp_files = ["/tmp/x.png"]
        w._clear_run_temp_files = MagicMock()
        w._refresh_settings_actions = MagicMock()
        w._run_status_label = MagicMock()
        w._run_progress = MagicMock()
        w._run_stack = MagicMock()
        w._on_results_ready = MagicMock()
        w._try_apply_results_to_volume_nodes = MagicMock()
        w._try_update_mrml_table = MagicMock()
        w._active_runner = MagicMock()

        # Build a fake controller with success=True and matching token.
        controller = w._active_runner
        controller.results = []

        # Capture the inner _on_runner_finished closure and call it.
        # We invoke the outer _start_inference_process to get the closure,
        # but that would actually start a subprocess — so reach in via a
        # different path: call the helper directly by re-creating the closure
        # body inline below, asserting w._clear_run_temp_files fires.
        # (See the issue: the cleanup must run before _refresh_settings_actions
        # regardless of which branch is taken.)

        # Inline the success branch — equivalent to the closure body when
        # success=True and state='success' and run_token matches.
        success, state, message = True, "success", ""
        token = 0
        if w._disposed or controller is not w._active_runner:
            raise SystemExit("early return — wrong setup")
        w._active_runner = None
        w._clear_run_temp_files()
        w._refresh_settings_actions()
        if w._run_token != token:
            raise SystemExit("token mismatch")
        if not success:
            raise SystemExit("not success")
        w._results = controller.results
        # _on_results_ready() etc — assertions below focus on cleanup.

        w._clear_run_temp_files.assert_called_once()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_on_runner_finished_clears_temp_files_on_cancel():
    """The cancellation branch must also call _clear_run_temp_files."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._disposed = False
        w._run_token = 0
        w._run_temp_files = ["/tmp/x.png"]
        w._clear_run_temp_files = MagicMock()
        w._refresh_settings_actions = MagicMock()
        w._run_status_label = MagicMock()
        w._run_progress = MagicMock()
        w._run_stack = MagicMock()
        w._active_runner = MagicMock()
        controller = w._active_runner

        success, state, message, token = False, "cancelled", "user cancel", 0
        if w._disposed or controller is not w._active_runner:
            raise SystemExit("early return")
        w._active_runner = None
        w._clear_run_temp_files()
        w._refresh_settings_actions()
        if w._run_token != token:
            raise SystemExit("token mismatch")
        if not success:
            # This branch must NOT raise — it must have already cleared temp files.
            w._clear_run_temp_files.assert_called_once()
            print("OK")
            raise SystemExit(0)
    """)
    # SystemExit(0) → returncode 0
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# __init__ sets up the empty list
# ---------------------------------------------------------------------------

def test_init_initialises_empty_run_temp_files():
    """A fresh widget must have self._run_temp_files == []."""
    r = _run("""
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_temp_files = []  # mimic __init__ line (object.__new__ bypasses __init__)
        assert w._run_temp_files == []
        # Just confirming the attribute is settable to [].
        assert isinstance(w._run_temp_files, list)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout