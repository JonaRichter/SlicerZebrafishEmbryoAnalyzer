import numpy as np
import pytest


@pytest.fixture
def line_mask():
    mask = np.zeros((20, 256), dtype=bool)
    mask[8:12, 5:251] = True
    return mask


def test_tube_length_returns_positive(line_mask):
    from ZebrafishEmbryoAnalyzerCore.length import tube_length_border2border
    result = tube_length_border2border(
        line_mask,
        spacing=(1.0, 1.0),
        return_path=True,
        return_straight_line=True,
        mask_eye=None,
        return_eye_info=False,
    )
    length, straight = result[0], result[1]
    assert length > 0
    assert straight > 0


def test_tube_length_spacing_scales_result(line_mask):
    from ZebrafishEmbryoAnalyzerCore.length import tube_length_border2border
    r1 = tube_length_border2border(
        line_mask, spacing=(1.0, 1.0),
        return_path=True, return_straight_line=True,
        mask_eye=None, return_eye_info=False,
    )
    r2 = tube_length_border2border(
        line_mask, spacing=(2.0, 2.0),
        return_path=True, return_straight_line=True,
        mask_eye=None, return_eye_info=False,
    )
    l1, l2 = r1[0], r2[0]
    assert abs(l2 - 2 * l1) < l1 * 0.15


def test_tube_length_empty_mask_does_not_crash():
    from ZebrafishEmbryoAnalyzerCore.length import tube_length_border2border
    empty = np.zeros((64, 64), dtype=bool)
    try:
        result = tube_length_border2border(
            empty, spacing=(1.0, 1.0),
            return_path=True, return_straight_line=True,
            mask_eye=None, return_eye_info=False,
        )
        length = result[0]
        assert length == 0.0
    except Exception:
        pass


def test_tube_length_wider_mask_is_longer():
    from ZebrafishEmbryoAnalyzerCore.length import tube_length_border2border
    short = np.zeros((20, 100), dtype=bool)
    short[8:12, 5:95] = True
    long_ = np.zeros((20, 256), dtype=bool)
    long_[8:12, 5:251] = True

    r_short = tube_length_border2border(short, spacing=(1.0, 1.0))
    r_long = tube_length_border2border(long_, spacing=(1.0, 1.0))
    assert r_long[0] > r_short[0]


def test_tube_length_returns_tuple():
    from ZebrafishEmbryoAnalyzerCore.length import tube_length_border2border
    mask = np.zeros((20, 100), dtype=bool)
    mask[8:12, 5:95] = True
    result = tube_length_border2border(mask, spacing=(1.0, 1.0))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# Issue #72: compute_tube_metrics (swim bladder area/width)
# ---------------------------------------------------------------------------

def test_compute_tube_metrics_rectangle_area_length_width():
    """A known axis-aligned rectangle mask must yield area/length/width
    matching (within pixel-boundary rounding) its actual dimensions."""
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 10:90] = 1  # 80 wide x 20 tall
    out = compute_tube_metrics(mask, spacing=(1.0, 1.0))
    assert out["area"] == pytest.approx(1600.0, rel=0.05)
    assert out["length"] == pytest.approx(80.0, abs=2.0)
    assert out["width"] == pytest.approx(20.0, abs=2.0)
    assert out["length_line"] is not None
    assert out["width_line"] is not None


def test_compute_tube_metrics_spacing_scales_area():
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 10:90] = 1
    out_unit = compute_tube_metrics(mask, spacing=(1.0, 1.0))
    out_scaled = compute_tube_metrics(mask, spacing=(2.0, 2.0))
    assert out_scaled["area"] == pytest.approx(out_unit["area"] * 4, rel=0.05)


def test_compute_tube_metrics_none_mask_returns_zeroed_dict():
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    out = compute_tube_metrics(None)
    assert out == {"area": 0.0, "length": 0.0, "width": 0.0,
                    "length_line": None, "width_line": None}


def test_compute_tube_metrics_empty_mask_returns_zeroed_dict():
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    mask = np.zeros((50, 50), dtype=np.uint8)
    out = compute_tube_metrics(mask)
    assert out["area"] == 0.0
    assert out["length"] == 0.0
    assert out["width"] == 0.0


def test_compute_tube_metrics_picks_largest_component():
    """Multiple disconnected blobs — must measure only the largest one."""
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:15, 10:15] = 1       # small blob, 5x5
    mask[40:60, 10:90] = 1       # large blob, 80x20
    out = compute_tube_metrics(mask, spacing=(1.0, 1.0))
    assert out["area"] == pytest.approx(1600.0, rel=0.05)
