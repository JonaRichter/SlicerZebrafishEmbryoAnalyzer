import sys

import numpy as np

# test_exclude_integration.py stubs out ZebrafishEmbryoAnalyzerLib.overlay to satisfy
# its own imports. Pop the stub so this file loads the real module.
sys.modules.pop("ZebrafishEmbryoAnalyzerLib.overlay", None)


def test_make_overlay_returns_rgb_array(synthetic_fish_image, synthetic_fish_mask):
    from ZebrafishEmbryoAnalyzerLib.overlay import make_overlay

    result_dict = {
        "original":  synthetic_fish_image,
        "mask":      synthetic_fish_mask,
        "eye_mask":  None,
        "path_points":          np.array([[64, 128], [128, 128], [192, 128]]),
        "straight_line_points": ((64, 128), (192, 128)),
        "length":    1200.0,
        "curvature": 2,
        "ratio":     1.05,
    }

    overlay = make_overlay(result_dict, thumbnail_size=150)
    assert overlay.ndim == 3
    assert overlay.shape[2] == 3
    assert overlay.dtype == np.uint8
    assert overlay.shape[0] <= 150
    assert overlay.shape[1] <= 150


def test_make_overlay_handles_none_mask(synthetic_fish_image):
    from ZebrafishEmbryoAnalyzerLib.overlay import make_overlay

    result_dict = {
        "original":             synthetic_fish_image,
        "mask":                 None,
        "eye_mask":             None,
        "path_points":          None,
        "straight_line_points": None,
        "length":               None,
        "curvature":            None,
        "ratio":                None,
    }
    overlay = make_overlay(result_dict, thumbnail_size=150)
    assert overlay is not None
    assert overlay.dtype == np.uint8


def test_make_full_overlay_still_draws_for_a_hand_excluded_row(
    synthetic_fish_image, synthetic_fish_mask,
):
    """Excluding a fish is a statistics decision, not a statement about its
    mask. Blanking the overlay made a hand-excluded image look identical to a
    failed one, and after a scene reload left no way to see what had been
    excluded or why.
    """
    import cv2
    from ZebrafishEmbryoAnalyzerLib.overlay import make_full_overlay

    row = {
        "original": synthetic_fish_image,
        "mask": synthetic_fish_mask,
        "eye_mask": None,
        "exclude": True,
        "error": "",
    }

    drawn = make_full_overlay(row)
    bare = cv2.cvtColor(synthetic_fish_image, cv2.COLOR_RGB2BGR)

    assert not np.array_equal(drawn, bare), (
        "a hand-excluded row must still show its segmentation"
    )


def test_make_full_overlay_stays_bare_for_an_error_row(
    synthetic_fish_image, synthetic_fish_mask,
):
    """The dangling-segmentation case keeps its behaviour: a deleted seg node
    leaves a stale mask in the row dict that must not reach the thumbnail.
    """
    import cv2
    from ZebrafishEmbryoAnalyzerLib.overlay import make_full_overlay

    row = {
        "original": synthetic_fish_image,
        "mask": synthetic_fish_mask,
        "eye_mask": None,
        "exclude": True,
        "error": "Segmentation node missing",
    }

    drawn = make_full_overlay(row)
    bare = cv2.cvtColor(synthetic_fish_image, cv2.COLOR_RGB2BGR)

    assert np.array_equal(drawn, bare)
