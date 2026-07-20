"""
Tests for inference_worker.run_worker().

All tests that require torch or real model weights are skipped when those
dependencies are absent (via pytest.importorskip or pytest.mark.skipif).
Tests that only test protocol handling and error paths do not need torch.
"""

import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_request(tmp_path, image_paths=None, model_id="general", params=None,
                   result_json=None, arrays_dir=None, protocol_version=1):
    request = {
        "protocol_version": protocol_version,
        "model_id": model_id,
        "image_paths": image_paths or [],
        "params": params or {},
        "result_json": str(result_json or (tmp_path / "result.json")),
        "arrays_dir": str(arrays_dir or (tmp_path / "arrays")),
    }
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(request))
    return str(req_path)


def _make_png(tmp_path, name="fish.png", size=(64, 64)):
    import cv2
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.ellipse(img, (size[0] // 2, size[1] // 2), (size[0] // 3, size[1] // 6),
                0, 0, 360, (200, 200, 200), -1)
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# Protocol / error path tests (no torch needed)
# ---------------------------------------------------------------------------

def test_run_worker_bad_request_exits_3_nonexistent_file():
    """Non-existent request file → exit 3."""
    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    code = run_worker("/nonexistent/request.json")
    assert code == 3


def test_run_worker_bad_request_exits_3_wrong_protocol(tmp_path):
    """protocol_version != 1 → exit 3."""
    req_path = _write_request(tmp_path, protocol_version=99)
    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    code = run_worker(req_path)
    assert code == 3


def test_run_worker_bad_request_exits_3_invalid_json(tmp_path):
    """Malformed JSON → exit 3."""
    req_path = tmp_path / "request.json"
    req_path.write_text("NOT JSON {{{")
    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    code = run_worker(str(req_path))
    assert code == 3


def test_run_worker_model_not_cached_exits_2(tmp_path):
    """ModelNotCachedError during preload → exit 2."""
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError
    req_path = _write_request(tmp_path, image_paths=["/tmp/fish.png"])

    def raise_not_cached(params):
        raise ModelNotCachedError("not cached")

    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    with patch("ZebrafishEmbryoAnalyzerLib.logic.preload_models", raise_not_cached):
        code = run_worker(req_path)
    assert code == 2


def test_run_worker_analysis_exception_exits_1(tmp_path):
    """analyse_images raising → exit 1."""
    img_path = _make_png(tmp_path)
    req_path = _write_request(tmp_path, image_paths=[img_path])

    def noop_preload(params):
        pass

    def raise_analysis(paths, params, cb, per_image_callback=None):
        raise RuntimeError("segmentation failed")

    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    with patch("ZebrafishEmbryoAnalyzerLib.logic.preload_models", noop_preload), \
         patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images", raise_analysis):
        code = run_worker(req_path)
    assert code == 1

    result_json = tmp_path / "result.json"
    assert result_json.exists()
    data = json.loads(result_json.read_text())
    assert data["status"] == "error"
    assert data["error_code"] == 1
    assert "segmentation failed" in data["error_message"]


def test_original_not_in_result(tmp_path):
    """Worker must not include 'original' key in serialised result."""
    img_path = _make_png(tmp_path)
    req_path = _write_request(tmp_path, image_paths=[img_path])

    original_array = np.zeros((64, 64, 3), dtype=np.uint8)
    fake_result = [{
        "filename": "fish.png",
        "image_path": img_path,
        "original": original_array,  # worker must strip this
        "mask": np.zeros((256, 256), dtype=np.uint8),
        "grown": None,
        "eye_mask": None,
        "path_points": None,
        "straight_line_points": None,
        "length": 1234.5,
        "curvature": 0,
        "ratio": 0.95,
        "eye_area": None,
        "eye_diameter": None,
        "spacing": [22.99, 22.99],
        "error": None,
    }]

    def noop_preload(params):
        pass

    def fake_analyse(paths, params, cb, per_image_callback=None):
        cb(1, 1)
        if per_image_callback is not None:
            for r in fake_result:
                per_image_callback(r.get("image_path"), r)
        return fake_result

    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    with patch("ZebrafishEmbryoAnalyzerLib.logic.preload_models", noop_preload), \
         patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images", fake_analyse):
        code = run_worker(req_path)

    assert code == 0
    result_json = tmp_path / "result.json"
    data = json.loads(result_json.read_text())
    assert data["status"] == "ok"
    for r in data["results"]:
        assert "original" not in r


def test_run_worker_writes_result_json_on_success(tmp_path):
    """Successful run writes result.json with status ok."""
    img_path = _make_png(tmp_path)
    req_path = _write_request(tmp_path, image_paths=[img_path])

    fake_result = [{
        "filename": "fish.png",
        "image_path": img_path,
        "original": np.zeros((64, 64, 3), dtype=np.uint8),
        "mask": np.zeros((256, 256), dtype=np.uint8),
        "grown": None,
        "eye_mask": None,
        "path_points": None,
        "straight_line_points": None,
        "length": 500.0,
        "curvature": 1,
        "ratio": 0.90,
        "eye_area": None,
        "eye_diameter": None,
        "spacing": [22.99, 22.99],
        "error": None,
    }]

    def noop_preload(params):
        pass

    def fake_analyse(paths, params, cb, per_image_callback=None):
        cb(1, 1)
        if per_image_callback is not None:
            for r in fake_result:
                per_image_callback(r.get("image_path"), r)
        return fake_result

    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    with patch("ZebrafishEmbryoAnalyzerLib.logic.preload_models", noop_preload), \
         patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images", fake_analyse):
        code = run_worker(req_path)

    assert code == 0
    result_json = tmp_path / "result.json"
    assert result_json.exists()
    data = json.loads(result_json.read_text())
    assert data["status"] == "ok"
    assert data["protocol_version"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["length_um"] == pytest.approx(500.0)


def test_run_worker_writes_progress_to_stdout(tmp_path, capsys):
    """Progress lines written to stdout as PROGRESS i/n."""
    img_path = _make_png(tmp_path)
    req_path = _write_request(tmp_path, image_paths=[img_path, img_path])

    call_log = []

    def noop_preload(params):
        pass

    def fake_analyse(paths, params, cb, per_image_callback=None):
        results = []
        for i, p in enumerate(paths, start=1):
            r = {
                "filename": "fish.png", "image_path": p,
                "original": None, "mask": None, "grown": None,
                "eye_mask": None, "path_points": None, "straight_line_points": None,
                "length": None, "curvature": None, "ratio": None,
                "eye_area": None, "eye_diameter": None, "spacing": None, "error": None,
            }
            results.append(r)
            if per_image_callback is not None:
                per_image_callback(p, r)
            cb(i, len(paths))
        return results

    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    with patch("ZebrafishEmbryoAnalyzerLib.logic.preload_models", noop_preload), \
         patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images", fake_analyse):
        code = run_worker(req_path)

    assert code == 0
    captured = capsys.readouterr()
    assert "PROGRESS 1/2" in captured.out
    assert "PROGRESS 2/2" in captured.out


# ---------------------------------------------------------------------------
# Issue #59: incremental per-image result.json write
# ---------------------------------------------------------------------------

def test_run_worker_writes_result_json_incrementally_after_each_image(tmp_path, monkeypatch):
    """After each per_image_callback, result.json contains exactly that many results.

    Simulates an interrupted batch: per_image_callback fires for images 0..K-1,
    then analyse_images returns/never completes for the rest. The runner
    (ZebrafishEmbryoAnalyzerLib.inference_runner.cancel) reads this file to
    recover completed images — so it must reflect reality after every callback,
    not just at full-batch completion.
    """
    img_paths = [_make_png(tmp_path, name=f"fish_{i}.png") for i in range(4)]
    req_path = _write_request(tmp_path, image_paths=img_paths)
    result_json = tmp_path / "result.json"

    pending_images = list(img_paths)
    fired_callbacks = []

    def noop_preload(params):
        pass

    def fake_analyse(paths, params, cb, per_image_callback=None):
        # Feed images one by one to per_image_callback, then return early to
        # simulate an interrupted batch — without per_image_callback firing
        # for every input image the worker would block forever on analyse_images.
        results = []
        for i, p in enumerate(paths):
            r = {
                "filename": f"fish_{i}.png",
                "image_path": p,
                "original": None, "mask": None, "grown": None,
                "eye_mask": None, "path_points": None, "straight_line_points": None,
                "length": 100.0 + i,
                "curvature": i,
                "ratio": 0.9 + i * 0.01,
                "eye_area": None, "eye_diameter": None,
                "spacing": [22.99, 22.99],
                "error": None,
            }
            if per_image_callback is not None:
                per_image_callback(p, r)
                fired_callbacks.append(p)
            results.append(r)
            # Stop after 2 images to simulate a partial / cancelled batch
            if i == 1:
                return results
        return results

    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    with patch("ZebrafishEmbryoAnalyzerLib.logic.preload_models", noop_preload), \
         patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images", fake_analyse):
        code = run_worker(str(req_path))

    # Worker should exit 0 (the worker process itself succeeded; cancellation
    # happens from the runner side and is a separate concern).
    assert code == 0
    assert len(fired_callbacks) == 2

    # Final file must reflect exactly the two results that completed.
    assert result_json.exists()
    data = json.loads(result_json.read_text())
    assert data["status"] == "ok"
    assert len(data["results"]) == 2
    assert data["results"][0]["length_um"] == pytest.approx(100.0)
    assert data["results"][1]["length_um"] == pytest.approx(101.0)


def test_run_worker_result_json_present_after_first_image(tmp_path):
    """Verify that a single completed image produces a valid result.json file
    readable from disk — the runner's _read_partial_results depends on this
    file existing after any image finishes, not only at batch completion.
    """
    img_path = _make_png(tmp_path, name="fish_0.png")
    req_path = _write_request(tmp_path, image_paths=[img_path])
    result_json = tmp_path / "result.json"

    captured = {}

    def noop_preload(params):
        pass

    def fake_analyse(paths, params, cb, per_image_callback=None):
        # Capture the file content inside the per_image_callback to prove the
        # atomic write happened *before* analyse_images returns.
        r = {
            "filename": "fish_0.png", "image_path": paths[0],
            "original": None, "mask": None, "grown": None,
            "eye_mask": None, "path_points": None, "straight_line_points": None,
            "length": 42.5, "curvature": 0, "ratio": 0.5,
            "eye_area": None, "eye_diameter": None,
            "spacing": [22.99, 22.99], "error": None,
        }
        per_image_callback(paths[0], r)
        captured["during_callback"] = result_json.exists() and json.loads(result_json.read_text())
        return [r]

    from ZebrafishEmbryoAnalyzerLib.inference_worker import run_worker
    with patch("ZebrafishEmbryoAnalyzerLib.logic.preload_models", noop_preload), \
         patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images", fake_analyse):
        code = run_worker(req_path)

    assert code == 0
    assert "during_callback" in captured
    assert captured["during_callback"]["status"] == "ok"
    assert len(captured["during_callback"]["results"]) == 1
    assert captured["during_callback"]["results"][0]["length_um"] == pytest.approx(42.5)
