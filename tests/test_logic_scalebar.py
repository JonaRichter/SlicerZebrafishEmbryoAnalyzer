"""
Tests for ZebrafishEmbryoAnalyzerLib.logic.detect_scalebar img_rgb parameter (issue #57).

Issue #57: after a scene reload, the image_path may be a bare filename with no
backing file on disk.  detect_scalebar must accept an in-memory RGB ndarray via
img_rgb so the caller can pass the already-decoded image instead of forcing a
re-read from disk.

These tests:
- require no model download, no Slicer runtime, no graphical desktop
- patch cv2.imread to verify whether (and when) the disk-read path is taken
- patch ZebrafishEmbryoAnalyzerCore.scalebar.detect_scalebar to capture the
  array the production code actually hands to the core detector
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishEmbryoAnalyzer"
)


def test_detect_scalebar_uses_img_rgb_without_reading_disk(tmp_path, synthetic_fish_image):
    """When img_rgb is supplied, cv2.imread must NOT be called.

    This is the core #57 guarantee: scene-reload paths can hand the function
    an already-decoded RGB array, and detect_scalebar must not reach for the
    (possibly nonexistent) disk file.
    """
    from ZebrafishEmbryoAnalyzerLib.logic import detect_scalebar

    captured = []

    def fake_core_detect(img, label_um=None):
        captured.append(img)
        return {"success": True, "bar_found": True, "scale_um_per_px": 22.99}

    fake_core_detect_mock = MagicMock(side_effect=fake_core_detect)

    # image_path is intentionally a path that doesn't exist; if cv2.imread is
    # called, the test fails loudly via the AssertionError side_effect.
    missing_path = str(tmp_path / "does_not_exist.png")

    with patch("cv2.imread", side_effect=AssertionError(
            "cv2.imread must not be called when img_rgb is provided"
    )), patch("ZebrafishEmbryoAnalyzerCore.scalebar.detect_scalebar",
              fake_core_detect_mock):
        result = detect_scalebar(missing_path, img_rgb=synthetic_fish_image)

    assert result["bar_found"] is True
    assert result["scale_um_per_px"] == 22.99
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], synthetic_fish_image)


def test_detect_scalebar_falls_back_to_disk_when_img_rgb_is_none(tmp_path,
                                                                 synthetic_fish_image):
    """When img_rgb is None, detect_scalebar must read image_path from disk."""
    from ZebrafishEmbryoAnalyzerLib.logic import detect_scalebar

    img_path = str(tmp_path / "fish.png")
    import cv2
    cv2.imwrite(img_path, synthetic_fish_image)

    captured = []
    def fake_core_detect(img, label_um=None):
        captured.append(img)
        return {"success": True, "bar_found": False}

    fake_core_detect_mock = MagicMock(side_effect=fake_core_detect)

    with patch("cv2.imread", wraps=cv2.imread) as mock_imread, \
         patch("ZebrafishEmbryoAnalyzerCore.scalebar.detect_scalebar",
               fake_core_detect_mock):
        result = detect_scalebar(img_path)

    assert mock_imread.called, "cv2.imread must be called when img_rgb is None"
    assert mock_imread.call_args.args == (img_path,)
    assert len(captured) == 1
    # The array passed to core must be RGB (not BGR): cv2 returns BGR, the
    # function converts before handing off.
    np.testing.assert_array_equal(captured[0], synthetic_fish_image)
    assert captured[0].shape == synthetic_fish_image.shape


def test_detect_scalebar_returns_failure_when_disk_read_fails(tmp_path):
    """When img_rgb is None AND cv2.imread returns None, return failure dict
    rather than raising.  Preserves the original error contract.
    """
    from ZebrafishEmbryoAnalyzerLib.logic import detect_scalebar

    missing_path = str(tmp_path / "does_not_exist.png")
    with patch("cv2.imread", return_value=None):
        result = detect_scalebar(missing_path)

    assert result == {"success": False, "bar_found": False,
                      "message": "Could not read image."}


def test_detect_scalebar_passes_label_um_through(tmp_path, synthetic_fish_image):
    """label_um must reach the core detector unchanged regardless of img_rgb path."""
    from ZebrafishEmbryoAnalyzerLib.logic import detect_scalebar

    captured_label = []

    def fake_core_detect(img, label_um=None):
        captured_label.append(label_um)
        return {"success": True, "bar_found": True}

    fake_core_detect_mock = MagicMock(side_effect=fake_core_detect)

    with patch("ZebrafishEmbryoAnalyzerCore.scalebar.detect_scalebar",
               fake_core_detect_mock):
        detect_scalebar("/irrelevant", label_um=500.0, img_rgb=synthetic_fish_image)

    assert captured_label == [500.0]


def test_detect_scalebar_signature_accepts_img_rgb_kwarg():
    """The public detect_scalebar function must accept img_rgb as a keyword
    argument.  Guarded by signature inspection so a future refactor that
    silently drops the parameter is caught at test time.
    """
    import inspect
    from ZebrafishEmbryoAnalyzerLib.logic import detect_scalebar

    sig = inspect.signature(detect_scalebar)
    assert "img_rgb" in sig.parameters, (
        "detect_scalebar must accept img_rgb keyword for issue #57 scene-reload "
        "support; signature was %s" % sig
    )
    assert sig.parameters["img_rgb"].default is None, (
        "img_rgb must default to None so the disk-read path stays the default "
        "behavior for callers that don't know about the new parameter"
    )


# ---------------------------------------------------------------------------
# Issue #76: manual two-point scale-bar calibration wrapper
# ---------------------------------------------------------------------------

def test_calibrate_scalebar_from_endpoints_delegates_to_core():
    """logic.calibrate_scalebar_from_endpoints must be a thin wrapper around
    ZebrafishEmbryoAnalyzerCore.scalebar.calibrate_from_endpoints — same
    role as detect_scalebar's wrapper, keeping widget.py free of a direct
    core import."""
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scalebar_from_endpoints

    captured = {}

    def fake_calibrate(pt1, pt2, img_shape, label_um=None):
        captured["args"] = (pt1, pt2, img_shape, label_um)
        return {"success": True, "scale_um_per_px": 2.5}

    with patch("ZebrafishEmbryoAnalyzerCore.scalebar.calibrate_from_endpoints", fake_calibrate):
        result = calibrate_scalebar_from_endpoints((10, 20), (110, 20), (200, 300), label_um=250.0)

    assert captured["args"] == ((10, 20), (110, 20), (200, 300), 250.0)
    assert result == {"success": True, "scale_um_per_px": 2.5}


def test_calibrate_scalebar_from_endpoints_real_core_math():
    """End-to-end (no mocking): 100px apart, 500um bar -> 5.0 um/px."""
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scalebar_from_endpoints

    result = calibrate_scalebar_from_endpoints((0, 0), (100, 0), (200, 300), label_um=500.0)
    assert result["success"] is True
    assert result["scale_um_per_px"] == pytest.approx(5.0)
