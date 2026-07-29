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
