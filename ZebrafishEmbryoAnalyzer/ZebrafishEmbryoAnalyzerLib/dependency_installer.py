"""
Dependency status check and install helpers for ZebrafishEmbryoAnalyzer.

get_missing_packages()  -- pure Python, safe to call anywhere
install_packages()      -- Slicer-only, called only from explicit user action

Torch is never pip-installed from here. It comes from the PyTorch extension via
PyTorchUtils, which selects the build matching the user's hardware; installing it
ourselves from a fixed wheel index forced a CPU-only build on every platform.
"""

REQUIRED_PACKAGES = [
    "segmentation_models_pytorch",
    "timm",
    "scikit-image",
    "opencv-python-headless",
    "huggingface_hub",
    "openpyxl",
    "pytesseract",
]

# Detection only — provided by the PyTorch extension, never installed via pip here.
TORCH_PACKAGES = ["torch", "torchvision"]

PYTORCH_EXTENSION_NAME = "PyTorch"


def _is_importable(name: str) -> bool:
    import importlib.util
    import_name = {
        "scikit-image":                "skimage",
        "opencv-python-headless":      "cv2",
        "huggingface_hub":             "huggingface_hub",
        "segmentation_models_pytorch": "segmentation_models_pytorch",
    }.get(name, name)
    return importlib.util.find_spec(import_name) is not None


def get_missing_packages() -> dict:
    """
    Return {"torch": [...], "general": [...]}.
    Pure Python — no slicer/qt import. Safe at any call site.
    """
    return {
        "torch":   [p for p in TORCH_PACKAGES    if not _is_importable(p)],
        "general": [p for p in REQUIRED_PACKAGES if not _is_importable(p)],
    }


def _pytorch_utils_logic():
    """PyTorchUtils logic from the PyTorch extension, or None if it is not installed."""
    try:
        import PyTorchUtils
    except ModuleNotFoundError:
        return None
    return PyTorchUtils.PyTorchUtilsLogic()


def install_pytorch_extension() -> bool:
    """Install the PyTorch extension from the extension server.

    Returns True once it is installed. Slicer must be restarted before
    PyTorchUtils becomes importable.
    """
    import slicer
    manager = slicer.app.extensionsManagerModel()
    if manager.isExtensionInstalled(PYTORCH_EXTENSION_NAME):
        return True
    return bool(manager.installExtensionFromServer(PYTORCH_EXTENSION_NAME))


def _install_torch() -> str:
    """Install torch and torchvision through the PyTorch extension.

    Returns "ok" when torch is installed, or "restart" when the PyTorch extension
    itself had to be installed first — PyTorchUtils only becomes importable after
    a Slicer restart, so torch follows on the next run.
    Raises RuntimeError if neither can be installed.
    """
    torch_logic = _pytorch_utils_logic()
    if torch_logic is None:
        if not install_pytorch_extension():
            raise RuntimeError(
                "The PyTorch extension could not be installed from the extension server. "
                "Install it manually via the Extensions Manager and restart Slicer."
            )
        return "restart"

    if torch_logic.installTorch(askConfirmation=False) is None:
        raise RuntimeError("PyTorch could not be installed through the PyTorch extension.")
    return "ok"


def install_packages(missing: dict, pip_fn=None, torch_fn=None) -> bool:
    """
    Install missing packages. Only called from an explicit user action.

    missing:  dict from get_missing_packages()
    pip_fn:   injectable for testing (default: slicer.util.pip_install)
    torch_fn: injectable for testing (default: _install_torch)

    Returns True only when every requested package was installed and a plain restart is
    all that remains. Returns False when the caller must not continue: testing mode, a
    failed install, or the PyTorch extension having just been installed, since its own
    restart has to happen before torch itself can follow.
    """
    import slicer
    if slicer.app.testingEnabled():
        return False

    import logging

    if pip_fn is None:
        pip_fn = slicer.util.pip_install
    if torch_fn is None:
        torch_fn = _install_torch

    # Torch must be settled before anything else is installed. Several of the remaining
    # packages (segmentation_models_pytorch above all) declare torch as a dependency, so
    # pip would happily resolve and install its own torch build if we got here first —
    # bypassing the PyTorch extension and the platform-specific constraints it applies.
    # On macOS that produced a torch compiled against NumPy 1.x sitting next to NumPy 2:
    # it imports, then fails at the first array conversion with "Numpy is not available".
    if missing.get("torch"):
        slicer.util.showStatusMessage("ZebrafishEmbryoAnalyzer: installing PyTorch…")
        try:
            outcome = torch_fn()
        except Exception as exc:
            logging.exception("Failed to install PyTorch: %s", exc)
            slicer.util.errorDisplay(
                f"PyTorch could not be installed:\n\n{exc}\n\n"
                "No further packages were installed."
            )
            return False

        if outcome == "restart":
            slicer.util.infoDisplay(
                "The PyTorch extension has been installed.\n\n"
                "Restart Slicer and open this module again to install the remaining "
                "packages. Models you want can be selected again then."
            )
            return False

    errors = []
    for pkg in missing.get("general", []):
        slicer.util.showStatusMessage(f"ZebrafishEmbryoAnalyzer: installing {pkg}…")
        try:
            pip_fn(pkg)
        except Exception as exc:
            logging.exception("Failed to install %s: %s", pkg, exc)
            errors.append(f"{pkg}: {exc}")

    if errors:
        slicer.util.errorDisplay(
            "Some packages could not be installed:\n" + "\n".join(f"  • {e}" for e in errors)
        )
        return False

    slicer.util.showStatusMessage(
        "ZebrafishEmbryoAnalyzer: dependencies installed — restart required."
    )

    return True
