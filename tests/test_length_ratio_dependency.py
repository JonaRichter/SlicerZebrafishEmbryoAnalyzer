"""
Tests for ZebrafishEmbryoAnalyzerMainWidget._enforce_length_ratio_dependency (issue #60).

Issue #60: "Length/straight ratio" can be enabled while "Body length" is disabled —
silently no-ops because logic.py's ratio computation is nested inside the length block.

The fix is a UI-level guard: when _chk_length is unchecked, _chk_ratio is disabled
and force-unchecked.  Re-checking _chk_length re-enables ratio (the previous checked
state is intentionally not restored — see the issue for the explicit "do not
over-engineer" note).

These tests stub qt/slicer and exercise the helper directly via object.__new__ so
they run in plain pytest without Slicer.
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


def _make_widget(length_checked: bool, ratio_checked: bool):
    """Build a stub widget with mocked checkboxes matching the given state."""
    return _run(f"""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._chk_length    = MagicMock(); w._chk_length.isChecked.return_value    = {str(length_checked)}
        w._chk_ratio     = MagicMock(); w._chk_ratio.isChecked.return_value     = {str(ratio_checked)}
        print("OK")
        globals()['__w'] = w
    """)


# ---------------------------------------------------------------------------
# _enforce_length_ratio_dependency — direct behaviour
# ---------------------------------------------------------------------------

def test_enforce_disables_ratio_when_length_is_unchecked():
    """Issue #60: ratio must be disabled and force-unchecked when length is off."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._chk_length = MagicMock(); w._chk_length.isChecked.return_value = False
        w._chk_ratio  = MagicMock()

        w._enforce_length_ratio_dependency()

        # Ratio must be disabled (length is off).
        w._chk_ratio.setEnabled.assert_called_with(False)
        # And forced unchecked, regardless of its prior state.
        w._chk_ratio.setChecked.assert_called_with(False)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_enforce_reenables_ratio_when_length_is_checked():
    """Issue #60: ratio must be re-enabled (not auto-checked) when length is on."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._chk_length = MagicMock(); w._chk_length.isChecked.return_value = True
        w._chk_ratio  = MagicMock()

        w._enforce_length_ratio_dependency()

        w._chk_ratio.setEnabled.assert_called_with(True)
        # When length is on, we deliberately do NOT touch setChecked —
        # the user's current ratio selection is preserved.
        w._chk_ratio.setChecked.assert_not_called()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_enforce_does_not_toggle_ratio_when_length_is_off_but_already_disabled():
    """If the GUI never called enforce before (mock returns True), the uncheck
    branch still fires — there is no skip-on-already-disabled optimisation.
    Documents the chosen semantics: idempotent but not over-smart.
    """
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._chk_length = MagicMock(); w._chk_length.isChecked.return_value = False
        w._chk_ratio  = MagicMock(); w._chk_ratio.isChecked.return_value = True

        w._enforce_length_ratio_dependency()

        w._chk_ratio.setEnabled.assert_called_with(False)
        w._chk_ratio.setChecked.assert_called_with(False)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# updateGUIFromParameterNode — restores consistency on load
# ---------------------------------------------------------------------------

def test_update_gui_clears_stale_ratio_when_length_disabled_in_node():
    """Issue #60 acceptance: loading a node with length=False, ratio=True
    must leave _chk_ratio.setChecked(False) as the final call, so the saved
    invalid combination is corrected on next module load.
    """
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import (
            ZebrafishEmbryoAnalyzerMainWidget,
            PARAM_LENGTH_ENABLED,
            PARAM_RATIO_ENABLED,
            PARAM_CURVATURE_ENABLED,
            PARAM_EYES_ENABLED,
            PARAM_CONFIDENCE_THRESHOLD_ENABLED,
            PARAM_CONFIDENCE_THRESHOLD,
            PARAM_UM_PER_PX,
            PARAM_MODEL_ID,
        )

        class _FakeNode:
            def __init__(self):
                self._p = {
                    PARAM_LENGTH_ENABLED: "false",
                    PARAM_RATIO_ENABLED: "true",
                    PARAM_CURVATURE_ENABLED: "true",
                    PARAM_EYES_ENABLED: "false",
                    PARAM_CONFIDENCE_THRESHOLD_ENABLED: "false",
                    PARAM_CONFIDENCE_THRESHOLD: "0.85",
                    PARAM_UM_PER_PX: "22.99",
                    PARAM_MODEL_ID: "general",
                }
            def GetParameter(self, k): return self._p.get(k)
            def StartModify(self): return False
            def EndModify(self, *a, **kw): pass

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock(); w._chk_length.isChecked.return_value    = False
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock(); w._chk_ratio.isChecked.return_value     = True
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock(); w._model_combo.count = 1
        w._model_combo.itemData.return_value = "general"

        w.updateGUIFromParameterNode(_FakeNode())

        # The final setChecked call on _chk_ratio must be (False,) — the
        # dependency hook overrides the stale saved "true".
        last_call = w._chk_ratio.setChecked.call_args
        assert last_call.args == (False,), (
            f"ratio must be force-unchecked by _enforce_length_ratio_dependency "
            f"after loading length=false, ratio=true; got call_args={last_call}"
        )
        # And ratio must be disabled (greyed out) in the UI.
        w._chk_ratio.setEnabled.assert_called_with(False)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_keeps_ratio_when_both_enabled_in_node():
    """Sanity check: a previously saved length=true, ratio=true combination
    must survive updateGUIFromParameterNode intact.
    """
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import (
            ZebrafishEmbryoAnalyzerMainWidget,
            PARAM_LENGTH_ENABLED,
            PARAM_RATIO_ENABLED,
            PARAM_CURVATURE_ENABLED,
            PARAM_EYES_ENABLED,
            PARAM_CONFIDENCE_THRESHOLD_ENABLED,
            PARAM_CONFIDENCE_THRESHOLD,
            PARAM_UM_PER_PX,
            PARAM_MODEL_ID,
        )

        class _FakeNode:
            def __init__(self):
                self._p = {
                    PARAM_LENGTH_ENABLED: "true",
                    PARAM_RATIO_ENABLED: "true",
                    PARAM_CURVATURE_ENABLED: "true",
                    PARAM_EYES_ENABLED: "false",
                    PARAM_CONFIDENCE_THRESHOLD_ENABLED: "false",
                    PARAM_CONFIDENCE_THRESHOLD: "0.85",
                    PARAM_UM_PER_PX: "22.99",
                    PARAM_MODEL_ID: "general",
                }
            def GetParameter(self, k): return self._p.get(k)
            def StartModify(self): return False
            def EndModify(self, *a, **kw): pass

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock(); w._chk_length.isChecked.return_value    = True
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock(); w._chk_ratio.isChecked.return_value     = True
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock(); w._model_combo.count = 1
        w._model_combo.itemData.return_value = "general"

        w.updateGUIFromParameterNode(_FakeNode())

        # ratio must be enabled and last setChecked(True) wins.
        w._chk_ratio.setEnabled.assert_called_with(True)
        last_call = w._chk_ratio.setChecked.call_args
        assert last_call.args == (True,), (
            f"ratio must stay checked when length is also enabled; got {last_call}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout