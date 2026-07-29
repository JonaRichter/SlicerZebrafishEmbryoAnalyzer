"""
Model manifest for ZebrafishEmbryoAnalyzer.

Pure Python — no slicer, qt, vtk, ctk, or torch imports.
Describes all model files, their provenance, and cache locations.

sha256 values are lowercase hex SHA-256 digests of the file at the pinned revision.
Revisions are immutable commit SHAs, not floating branch names.
size_bytes values are approximate estimates used only for pre-download UI;
real sizes come from Content-Length headers during download.
Licenses are "LICENSE_PENDING" until confirmed.
"""

import hashlib
import os
from pathlib import Path


def _default_cache_dir() -> Path:
    """Return a platform-appropriate cache directory for model files.

    Prefers ``platformdirs.user_cache_dir`` (cross-platform).  Falls back to
    ``~/.cache/zebrafish_models`` if platformdirs is not installed so that
    existing deployments on macOS/Linux keep working without an additional
    dependency.
    """
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("zebrafish_models"))
    except Exception:
        # Catch ImportError (platformdirs absent) and runtime errors such as
        # PermissionError / OSError on Windows with a restricted LOCALAPPDATA.
        return Path.home() / ".cache" / "zebrafish_models"


_CACHE_DIR = _default_cache_dir()

# ---------------------------------------------------------------------------
# All known model entries
# ---------------------------------------------------------------------------

MODELS: dict = {
    # The webapp offers three presets, not two: "Fast & Easy" (256px, body only,
    # no eye/edema/swim bladder — this entry), "Complex & Slower" (512px, see
    # general_body below), and "Fine-tuned DESY" (512px). Body-only — the webapp's
    # own Fast & Easy preset has no eye model either (eye_hf_filename=None).
    "fast_body": {
        "id": "fast_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_3400_vgg19.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Body segmentation model (256px, Fast & Easy)",
        "encoder": "vgg19",
        "sha256": "624e9ef0ab447aee7b95a058596c048f033a8255bc850f3a238b5606ea71ae65",
        "size_bytes": 116_289_435,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_body": {
        "id": "general_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_512.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Body segmentation model (512px)",
        "encoder": "vgg19",
        "sha256": "030fd29623467a12eb5d913460fd778d8773345f98f152bbaeae1ebbbbb9ecf2",
        "size_bytes": 116_289_323,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_eye": {
        "id": "general_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_512.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Eye segmentation model (512px)",
        "encoder": "vgg16",
        "sha256": "9493f3ae60ecddbe1f16717b9c28b2a43491adc8b6d8d066031ae83d3b5cd383",
        "size_bytes": 95_048_133,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    # Not yet wired into any MODEL_SET; kept for future edema analysis support.
    "general_edema": {
        "id": "general_edema",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_edema_3400_focal.pth",
        "revision": "673bc5d60e786a8413ecefbcc1701e1ec6ed6ae1",
        "label": "Edema segmentation model",
        "encoder": "vgg19",
        "sha256": "3622392fc8a65d9de1f49554770422cf3661deee8381a4fbbd62c48d01c6dfaf",
        "size_bytes": 116_290_283,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    # Swim bladder uses FPN (segmentation_models_pytorch), not Unet like the other
    # segmentation roles — see model_type below, consumed by seg.py's _load_unet_model.
    # Offered under both presets (unlike edema, which is DESY-only in the webapp).
    "general_swimbladder": {
        "id": "general_swimbladder",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_swimmbladder_512_09072026.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Swim bladder segmentation model",
        "encoder": "vgg19",
        "model_type": "FPN",
        "sha256": "d11e41c7504bbd388f29b53d2a31a731190e4b1b26f036326f4a3c104334d5ab",
        "size_bytes": 88_459_594,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "curvature": {
        "id": "curvature",
        "repo_id": "markdanielarndt/Classification",
        "filename": "best_model_class.pth",
        "revision": "926bea8cec2898e6eb313f8748318f6053876ed8",
        "label": "Curvature classification model",
        "encoder": None,
        "sha256": "7b9c029ed1b8887fca2fe42197d010422b8a822e3aa86da6ffcdbdb530ebdc6c",
        "size_bytes": 352_517_483,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "desy_body": {
        "id": "desy_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "desy_body_512_finetuned.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "DESY body segmentation model (512px)",
        "encoder": "vgg19",
        "sha256": "5e2ee9c72fd0f3a452123bcfa9fcceedd10c5d58ba5b9e70694ccc2227d35340",
        "size_bytes": 116_290_507,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "desy_eye": {
        "id": "desy_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "desy_eye_512_finetuned.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "DESY eye segmentation model (512px)",
        "encoder": "vgg16",
        "sha256": "661308b9be5ff31d386c67abfa80f9b897ed437ceca01b7c046b66704ee5fdb3",
        "size_bytes": 95_049_257,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    # DESY-only — the webapp's General preset has no edema model. Distinct from
    # the unused general_edema entry above (different filename, different revision).
    "desy_edema": {
        "id": "desy_edema",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "desy_edema_512_finetuned.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "DESY edema segmentation model",
        "encoder": "vgg19",
        "sha256": "5c4c99299da84842bc2efa8aa42ae693b5d3bc5e30ad675b2772e4096e09728b",
        "size_bytes": 116_289_947,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "desy_swimbladder": {
        "id": "desy_swimbladder",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "desy_swimmbladder_512_finetuned.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "DESY swim bladder segmentation model",
        "encoder": "vgg19",
        "model_type": "FPN",
        "sha256": "92a47377cf450d3fcb7ec52c7065d1e5b74e36b125460752496ff66837655d1c",
        "size_bytes": 88_459_870,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
}

# ---------------------------------------------------------------------------
# Model sets per variant: variant_id -> {role -> model_entry}
# ---------------------------------------------------------------------------

MODEL_SETS: dict = {
    # No "eye"/"edema"/"swimbladder" keys — the webapp's Fast & Easy preset offers
    # none of those (body + curvature only). widget.py disables and unchecks those
    # checkboxes when this preset is selected, mirroring the existing DESY-only
    # edema gating.
    "fast": {
        "body": MODELS["fast_body"],
        "curvature": MODELS["curvature"],
    },
    "general": {
        "body": MODELS["general_body"],
        "eye": MODELS["general_eye"],
        "curvature": MODELS["curvature"],
        "swimbladder": MODELS["general_swimbladder"],
    },
    "desy": {
        "body": MODELS["desy_body"],
        "eye": MODELS["desy_eye"],
        "curvature": MODELS["curvature"],
        "edema": MODELS["desy_edema"],
        "swimbladder": MODELS["desy_swimbladder"],
    },
}

# Per-preset segmentation_pipeline() target_size, matching the webapp's own
# per-preset resolution (see SEG_MODEL_OPTIONS in the webapp's app.py). Kept
# separate from MODEL_SETS so every value there stays a model-entry dict —
# get_missing_models()/verify_checksum() iterate MODEL_SETS[...].values()
# expecting exactly that shape.
MODEL_TARGET_SIZE: dict = {
    "fast": (256, 256),
    "general": (512, 512),
    "desy": (512, 512),
}


def get_cached_path(entry: dict) -> Path:
    """Return the local cache Path for a model entry."""
    return _CACHE_DIR / entry["filename"]


def verify_checksum(path, sha256: str) -> bool:
    """
    Verify SHA-256 checksum of a file.

    Raises ValueError for missing, malformed, or placeholder hash values
    so that configuration errors are caught early rather than silently skipped.

    Returns True when the file hash matches sha256.
    Raises ValueError for placeholder ("PENDING") or empty sha256.
    Returns False when the file hash does not match or the file cannot be read.
    """
    if not sha256 or sha256 == "PENDING":
        raise ValueError(
            f"Model at {path!r} has a placeholder or missing SHA-256 checksum. "
            "Update model_manifest.py with the real checksum before using this model."
        )
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != sha256:
            return False
        return True
    except OSError:
        return False


def checksum_mismatch_error(entry: dict, path, actual_sha256: str) -> str:
    """Return a human-readable checksum mismatch error string."""
    return (
        f"Checksum mismatch for model {entry['id']!r} at {path!r}.\n"
        f"  Expected: {entry['sha256']}\n"
        f"  Actual:   {actual_sha256}\n"
        "The file may be corrupted or tampered. Delete it and re-download."
    )


def collect_all_model_entries() -> dict:
    """Return all unique model entries across all MODEL_SETS, deduplicated by id.

    Returns
    -------
    dict
        Mapping of entry id -> model_entry for every entry that appears in any
        MODEL_SET.  Entries shared across sets (e.g. curvature) appear once.
    """
    result = {}
    for variant in MODEL_SETS.values():
        for entry in variant.values():
            result.setdefault(entry["id"], entry)
    return result


def get_missing_models(model_set_dict: dict) -> list:
    """
    Return model entries whose cached file is missing or fails checksum verification.

    A truncated or corrupted download (present on disk, non-zero size, wrong
    content — e.g. an interrupted transfer) must be treated the same as a
    genuinely missing file. Otherwise it looks "cached" until deserialization
    fails deep inside analysis with a cryptic pickle error instead of the normal
    download-prompt flow.

    Parameters
    ----------
    model_set_dict : dict
        Mapping of str -> model_entry, e.g. ``MODEL_SETS["general"]``.

    Returns
    -------
    list[dict]
        Subset of model_set_dict.values() that are not yet correctly cached.
    """
    return [
        entry for entry in model_set_dict.values()
        if not verify_checksum(get_cached_path(entry), entry["sha256"])
    ]
