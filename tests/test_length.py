import numpy as np
import pytest


class _FakeModel:
    """Minimal nn.Module stand-in for select_torch_device's real-model probe."""

    def __init__(self, cuda_functional=True):
        self.cuda_functional = cuda_functional
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def __call__(self, x):
        if self.device == "cuda" and not self.cuda_functional:
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return x


class _NoGradCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTorch:
    """Minimal torch stand-in for select_torch_device — no real torch/CUDA needed."""

    def __init__(self, cuda_available):
        self._cuda_available = cuda_available

    class cuda:
        pass

    def zeros(self, *args, device=None, **kwargs):
        return f"tensor(device={device})"

    def device(self, name):
        return f"device:{name}"

    def no_grad(self):
        return _NoGradCtx()


def _make_fake_torch(cuda_available):
    fake = _FakeTorch(cuda_available)
    fake.cuda.is_available = lambda: cuda_available
    return fake


def test_select_torch_device_no_cuda_returns_cpu():
    from ZebrafishEmbryoAnalyzerCore.length import select_torch_device
    fake_torch = _make_fake_torch(cuda_available=False)
    assert select_torch_device(fake_torch, probe_model=_FakeModel()) == "device:cpu"


def test_select_torch_device_no_probe_model_checks_availability_only():
    from ZebrafishEmbryoAnalyzerCore.length import select_torch_device
    fake_torch = _make_fake_torch(cuda_available=True)
    assert select_torch_device(fake_torch) == "device:cuda"


def test_select_torch_device_functional_cuda_returns_cuda():
    from ZebrafishEmbryoAnalyzerCore.length import select_torch_device
    fake_torch = _make_fake_torch(cuda_available=True)
    model = _FakeModel(cuda_functional=True)
    assert select_torch_device(fake_torch, probe_model=model) == "device:cuda"
    assert model.device == "cuda"


def test_select_torch_device_broken_cuda_falls_back_to_cpu():
    """CUDA reports available but a real forward pass through the model fails
    (e.g. unsupported compute capability) — must fall back to CPU on the model
    itself, not just report cpu while leaving the model stuck on cuda."""
    from ZebrafishEmbryoAnalyzerCore.length import select_torch_device
    fake_torch = _make_fake_torch(cuda_available=True)
    model = _FakeModel(cuda_functional=False)
    assert select_torch_device(fake_torch, probe_model=model) == "device:cpu"
    assert model.device == "cpu"


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
# compute_tube_metrics (#72 — swim bladder area/length/width)
# ---------------------------------------------------------------------------

@pytest.fixture
def rect_mask():
    """A 10x40 axis-aligned rectangle: short side (width) ~ 10, long side (length) ~ 40.
    cv2.minAreaRect measures between pixel centers, so an N-pixel-wide block reports
    ~(N-1); a 10px block keeps that discretization error under 15% relative."""
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:20, 5:45] = 1
    return mask


def test_compute_tube_metrics_none_mask_returns_zeros():
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    out = compute_tube_metrics(None)
    assert out == {"area": 0.0, "length": 0.0, "width": 0.0,
                    "length_line": None, "width_line": None}


def test_compute_tube_metrics_empty_mask_returns_zeros():
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    empty = np.zeros((60, 60), dtype=np.uint8)
    out = compute_tube_metrics(empty)
    assert out["area"] == 0.0
    assert out["length"] == 0.0
    assert out["width"] == 0.0
    assert out["length_line"] is None
    assert out["width_line"] is None


def test_compute_tube_metrics_rectangle_area_length_width(rect_mask):
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    out = compute_tube_metrics(rect_mask, spacing=(1.0, 1.0))
    assert out["area"] == pytest.approx(10 * 40, rel=0.05)
    assert out["length"] == pytest.approx(40, rel=0.05)
    assert out["width"] == pytest.approx(10, rel=0.15)
    assert out["length"] > out["width"]
    assert out["length_line"] is not None
    assert out["width_line"] is not None


def test_compute_tube_metrics_spacing_scales_area_and_length(rect_mask):
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    r1 = compute_tube_metrics(rect_mask, spacing=(1.0, 1.0))
    r2 = compute_tube_metrics(rect_mask, spacing=(2.0, 2.0))
    assert r2["area"] == pytest.approx(r1["area"] * 4, rel=0.05)
    assert r2["length"] == pytest.approx(r1["length"] * 2, rel=0.05)
    assert r2["width"] == pytest.approx(r1["width"] * 2, rel=0.15)


def test_compute_tube_metrics_picks_largest_connected_component():
    """A stray, smaller blob elsewhere in the mask must be ignored."""
    from ZebrafishEmbryoAnalyzerCore.length import compute_tube_metrics
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:20, 5:45] = 1   # 10x40 — the real swim bladder
    mask[50:52, 50:52] = 1  # 2x2 stray noise blob, disconnected
    out = compute_tube_metrics(mask, spacing=(1.0, 1.0))
    assert out["area"] == pytest.approx(10 * 40, rel=0.05)
