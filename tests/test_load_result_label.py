"""
Tests for the persistent in-panel load result label (issue #62).

Issue #62: when _filter_readable_paths drops unreadable images, the only
feedback the user gets is slicer.util.showStatusMessage — an 8-second flash
in the global status bar, easy to miss while attention is on the module panel.

The fix adds a ``_load_result_label`` QLabel that stays visible in the panel
until the next load, with three states:
- empty (no load yet / nothing loaded)
- green ("Loaded N image(s).") on full success
- amber ("Loaded X of Y image(s) — N unreadable and skipped: ...") on partial

The existing transient showStatusMessage call is preserved (additive — not
a replacement) for users who do look at the status bar.
"""

import os
import subprocess
import sys
import textwrap

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


def _make_widget():
    """Build a stub widget with everything _set_queue touches mocked."""
    return _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_token = 0
        w._deps_ok = True
        w._image_paths = []
        w._btn_run = MagicMock()
        w._queue_list = MagicMock()
        w._results = []
        w._excluded = set()
        w._detail = MagicMock()
        w._results_tab = MagicMock()
        w._gallery = MagicMock()
        w._tabs = MagicMock()
        w._um_per_px = MagicMock()
        w._load_originals = MagicMock()
        w._active_runner = None
        w._logic = MagicMock()
        # Issue #62 — the subject under test.
        w._load_result_label = MagicMock()
        w._format_readability_message = MagicMock(side_effect=lambda n, names: f"unreadable: {names!r}")
        globals()['__w'] = w
    """)


# ---------------------------------------------------------------------------
# Outcome cases — three states the issue spec calls out
# ---------------------------------------------------------------------------

def test_set_queue_full_success_sets_green_label():
    """All readable → label reads 'Loaded N image(s).' in green."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_token = 0
        w._deps_ok = True
        w._image_paths = []
        w._btn_run = MagicMock()
        w._queue_list = MagicMock()
        w._results = []
        w._excluded = set()
        w._detail = MagicMock()
        w._results_tab = MagicMock()
        w._gallery = MagicMock()
        w._tabs = MagicMock()
        w._um_per_px = MagicMock()
        w._load_originals = MagicMock()
        w._active_runner = None
        w._logic = MagicMock()
        w._load_result_label = MagicMock()
        w._format_readability_message = MagicMock()

        # All three images readable — nothing failed.
        w._filter_readable_paths = MagicMock(
            return_value=(["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"], [], {})
        )

        w._set_queue(["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"])

        # Label text reads "Loaded 3 image(s)."
        last_text = w._load_result_label.setText.call_args.args[0]
        assert last_text == "Loaded 3 image(s).", (
            f"expected full-success message; got {last_text!r}"
        )
        # Green style applied.
        last_style = w._load_result_label.setStyleSheet.call_args.args[0]
        assert "4CAF50" in last_style, (
            f"full-success label must use green color; got {last_style!r}"
        )
        # _format_readability_message not called (no failures).
        w._format_readability_message.assert_not_called()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_set_queue_partial_success_sets_amber_label_with_failures():
    """Some unreadable → label shows 'Loaded X of Y' in amber with failed names."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_token = 0
        w._deps_ok = True
        w._image_paths = []
        w._btn_run = MagicMock()
        w._queue_list = MagicMock()
        w._results = []
        w._excluded = set()
        w._detail = MagicMock()
        w._results_tab = MagicMock()
        w._gallery = MagicMock()
        w._tabs = MagicMock()
        w._um_per_px = MagicMock()
        w._load_originals = MagicMock()
        w._active_runner = None
        w._logic = MagicMock()
        w._load_result_label = MagicMock()
        w._format_readability_message = MagicMock(return_value="corrupt1.png, corrupt2.png")

        # 1 readable, 2 unreadable.
        w._filter_readable_paths = MagicMock(
            return_value=(["/tmp/good.png"], ["corrupt1.png", "corrupt2.png"], {})
        )

        w._set_queue(["/tmp/good.png", "corrupt1.png", "corrupt2.png"])

        last_text = w._load_result_label.setText.call_args.args[0]
        # Must show the partial count, the failed count, and the failed names.
        assert "Loaded 1 of 3 image(s)" in last_text, last_text
        assert "2 unreadable and skipped" in last_text, last_text
        assert "corrupt1.png, corrupt2.png" in last_text, last_text
        # Amber style applied.
        last_style = w._load_result_label.setStyleSheet.call_args.args[0]
        assert "FFC107" in last_style, (
            f"partial-success label must use amber color; got {last_style!r}"
        )
        # _format_readability_message is called twice — once for the persistent
        # label text, once for the transient showStatusMessage (issue #62 is
        # additive: it does NOT replace the existing 8s global flash).
        assert w._format_readability_message.call_count == 2
        expected_call = w._format_readability_message.call_args_list[0]
        assert expected_call.args == (2, ["corrupt1.png", "corrupt2.png"])
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_set_queue_all_unreadable_still_shows_amber_label():
    """Zero readable, N unreadable → amber label still shown (not green)."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_token = 0
        w._deps_ok = True
        w._image_paths = []
        w._btn_run = MagicMock()
        w._queue_list = MagicMock()
        w._results = []
        w._excluded = set()
        w._detail = MagicMock()
        w._results_tab = MagicMock()
        w._gallery = MagicMock()
        w._tabs = MagicMock()
        w._um_per_px = MagicMock()
        w._load_originals = MagicMock()
        w._active_runner = None
        w._logic = MagicMock()
        w._load_result_label = MagicMock()
        w._format_readability_message = MagicMock(return_value="bad.png")

        w._filter_readable_paths = MagicMock(
            return_value=([], ["bad.png"], {})
        )

        w._set_queue(["bad.png"])

        last_text = w._load_result_label.setText.call_args.args[0]
        assert "Loaded 0 of 1 image(s)" in last_text, last_text
        assert "1 unreadable and skipped" in last_text, last_text
        last_style = w._load_result_label.setStyleSheet.call_args.args[0]
        assert "FFC107" in last_style, last_style  # amber, not green
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_set_queue_empty_input_clears_label():
    """Empty input list → empty text and no style."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_token = 0
        w._deps_ok = True
        w._image_paths = []
        w._btn_run = MagicMock()
        w._queue_list = MagicMock()
        w._results = []
        w._excluded = set()
        w._detail = MagicMock()
        w._results_tab = MagicMock()
        w._gallery = MagicMock()
        w._tabs = MagicMock()
        w._um_per_px = MagicMock()
        w._load_originals = MagicMock()
        w._active_runner = None
        w._logic = MagicMock()
        w._load_result_label = MagicMock()
        w._format_readability_message = MagicMock()

        w._filter_readable_paths = MagicMock(return_value=([], [], {}))
        w._set_queue([])

        # Label text must be cleared (empty string) on empty input.
        last_text = w._load_result_label.setText.call_args.args[0]
        assert last_text == "", (
            f"empty input must clear the label; got {last_text!r}"
        )
        last_style = w._load_result_label.setStyleSheet.call_args.args[0]
        assert last_style == "", f"empty input must clear the style; got {last_style!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_set_queue_preserves_existing_transient_status_message():
    """The 8-second global showStatusMessage call must still fire on partial."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
        from unittest.mock import patch
        import sys

        # Inject a sentinel object we can spy on.
        sentinel = MagicMock()
        sys.modules['slicer'].util.showStatusMessage = MagicMock(
            side_effect=lambda msg, ms: sentinel(msg=msg, ms=ms)
        )

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._run_token = 0
        w._deps_ok = True
        w._image_paths = []
        w._btn_run = MagicMock()
        w._queue_list = MagicMock()
        w._results = []
        w._excluded = set()
        w._detail = MagicMock()
        w._results_tab = MagicMock()
        w._gallery = MagicMock()
        w._tabs = MagicMock()
        w._um_per_px = MagicMock()
        w._load_originals = MagicMock()
        w._active_runner = None
        w._logic = MagicMock()
        w._load_result_label = MagicMock()
        w._format_readability_message = MagicMock(return_value="bad.png")

        w._filter_readable_paths = MagicMock(
            return_value=(["/tmp/good.png"], ["bad.png"], {})
        )
        w._set_queue(["/tmp/good.png", "bad.png"])

        # The transient 8s showStatusMessage must still be called with the
        # existing readability summary (issue #62 = additive, not replacement).
        sys.modules['slicer'].util.showStatusMessage.assert_called_once()
        args = sys.modules['slicer'].util.showStatusMessage.call_args.args
        assert args[1] == 8000, f"existing 8s timeout must be preserved; got {args}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout