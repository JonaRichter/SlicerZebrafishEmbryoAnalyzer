import os
import subprocess
import sys
import textwrap


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )


def test_core_length_imports_without_matplotlib():
    # Block matplotlib so `import matplotlib[.pyplot]` raises ImportError, regardless
    # of whether matplotlib is installed in the running environment. The normal
    # analysis entry points used by the Slicer extension must still import.
    result = _run_in_subprocess(
        """
        import sys
        sys.modules["matplotlib"] = None
        sys.modules["matplotlib.pyplot"] = None
        from zebrafish_analysis.core.length import (
            load_model,
            tube_length_border2border,
            classification_curvature,
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "core.length must import without matplotlib.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
