"""
Tests for ensure_dependencies() — the on-demand dependency install flow.

Verifies:
  - opening the module never triggers an install (no call site in enter())
  - nothing missing → the caller proceeds
  - user declines → the caller does not proceed and nothing is installed
  - install succeeds → restart dialog, and the caller still does not proceed,
    because the new packages only become importable after a restart
  - a failed install is reported and does not proceed
  - the confirmation uses Slicer's standard dialog, not a hand-built QDialog

Pure Python — no Slicer, Qt, or torch required.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock


@contextmanager
def _stub_slicer_env():
    """Inject minimal slicer/qt/ctk stubs so widget.py can be imported."""
    saved = {k: sys.modules[k] for k in ("slicer", "qt", "ctk") if k in sys.modules}
    sys.modules["slicer"] = MagicMock()
    sys.modules["qt"] = MagicMock()
    sys.modules["ctk"] = MagicMock()
    sys.modules.pop("ZebrafishEmbryoAnalyzerLib.widget", None)
    try:
        yield
    finally:
        for k in ("slicer", "qt", "ctk", "ZebrafishEmbryoAnalyzerLib.widget"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def _widget_class():
    from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
    return ZebrafishEmbryoAnalyzerMainWidget


def _shell(cls):
    w = object.__new__(cls)
    w._show_restart_dialog = MagicMock()
    return w


def test_show_restart_dialog_takes_no_parameters():
    """_show_restart_dialog must accept zero extra arguments beyond self."""
    import inspect
    with _stub_slicer_env():
        sig = inspect.signature(_widget_class()._show_restart_dialog)
        params = [p for p in sig.parameters if p != "self"]
    assert params == [], (
        f"_show_restart_dialog should have no parameters beyond self; got {params}"
    )


def test_module_entry_does_not_check_dependencies():
    """Opening the module must not ask the user to install anything — browsing the
    results of an existing scene needs none of the packages."""
    from pathlib import Path
    src = Path("ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzer.py").read_text()
    enter_body = src.split("def enter(self)")[1].split("def exit(self)")[0]
    # Strip comments — enter() explains in prose why it does not call this.
    code = "\n".join(line.split("#")[0] for line in enter_body.splitlines())
    assert "ensure_dependencies(" not in code
    assert "prompt_install_if_missing" not in src


def test_proceeds_when_nothing_missing(monkeypatch):
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": []})
        sys.modules["slicer"].app.testingEnabled.return_value = False

        assert cls.ensure_dependencies(w, "analysis") is True
        w._show_restart_dialog.assert_not_called()


def test_declining_stops_the_caller(monkeypatch):
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = False

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": ["timm"]})
        install = MagicMock()
        monkeypatch.setattr(di, "install_packages", install)

        assert cls.ensure_dependencies(w, "analysis") is False
        install.assert_not_called()
        w._show_restart_dialog.assert_not_called()


def test_successful_install_shows_restart_and_still_stops_the_caller(monkeypatch):
    """Installed packages only become importable after a restart, so the action that
    triggered the install must not continue in this session."""
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = True

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": ["timm"]})
        monkeypatch.setattr(di, "install_packages", MagicMock(return_value=True))

        assert cls.ensure_dependencies(w, "analysis") is False
        w._show_restart_dialog.assert_called_once_with()


def test_failed_install_is_reported(monkeypatch):
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = True

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": ["timm"]})
        monkeypatch.setattr(di, "install_packages",
                            MagicMock(side_effect=RuntimeError("boom")))

        assert cls.ensure_dependencies(w, "analysis") is False
        slicer.util.errorDisplay.assert_called_once()
        w._show_restart_dialog.assert_not_called()


def test_confirmation_uses_the_standard_dialog_with_package_details(monkeypatch):
    """Tier 3 asks for minimal popups and no unnecessary custom GUI — the package list
    belongs in the standard dialog's detail area, not in a hand-built QDialog."""
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = False

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(
            di, "get_missing_packages",
            lambda purpose="analysis": {"torch": ["torch"], "general": ["timm"]})

        cls.ensure_dependencies(w, "analysis")

        slicer.util.confirmOkCancelDisplay.assert_called_once()
        detail = slicer.util.confirmOkCancelDisplay.call_args.kwargs["detailedText"]
        assert "timm" in detail
        assert "torch" in detail


def test_no_custom_setup_dialog_remains():
    """The hand-built setup dialog and its model checkboxes are gone; models download
    on demand at the point they are first needed."""
    from pathlib import Path
    src = Path("ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzerLib/widget.py").read_text()
    assert "ZebrafishEmbryoAnalyzer — Setup" not in src
    assert "_start_initial_model_download" not in src
    assert "_install_declined" not in src
