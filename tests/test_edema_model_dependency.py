"""
Tests for ZebrafishEmbryoAnalyzerMainWidget._enforce_edema_model_dependency
(issue #73).

Edema segmentation is only available for the DESY model (model_manifest's
MODEL_SETS["desy"] has an "edema" role; MODEL_SETS["general"] does not,
matching the live reference webapp's own model registry). The UI-level
guard mirrors issue #60's _enforce_length_ratio_dependency pattern: grey
out + force-uncheck the edema checkbox for any model that doesn't support
it.
"""

import os
import subprocess
import sys
import textwrap


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


def test_edema_disabled_and_unchecked_for_general_model():
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._model_combo = MagicMock(); w._model_combo.currentData = "general"
        w._chk_edema = MagicMock()

        w._enforce_edema_model_dependency()

        w._chk_edema.setEnabled.assert_called_with(False)
        w._chk_edema.setChecked.assert_called_with(False)
        assert w._chk_edema.setToolTip.call_args[0][0] != ""
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_edema_enabled_for_desy_model():
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._model_combo = MagicMock(); w._model_combo.currentData = "desy"
        w._chk_edema = MagicMock()

        w._enforce_edema_model_dependency()

        w._chk_edema.setEnabled.assert_called_with(True)
        # Enabling must not force-uncheck an existing user selection.
        w._chk_edema.setChecked.assert_not_called()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_edema_defaults_to_general_model_id_when_combo_empty():
    """currentData falsy (e.g. combo not yet populated) must not crash and
    must fall back to _DEFAULT_MODEL_ID (general -> no edema)."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._model_combo = MagicMock(); w._model_combo.currentData = None
        w._chk_edema = MagicMock()

        w._enforce_edema_model_dependency()

        w._chk_edema.setEnabled.assert_called_with(False)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
