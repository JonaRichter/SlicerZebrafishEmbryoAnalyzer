# Core source provenance

`ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzerCore/` is a manually-maintained
port of the analysis engine from the reference webapp
(`markdanielarndt/Zebrafish_webapp`, live deployment at
`https://huggingface.co/spaces/markdanielarndt/Zebrafish`). This file exists
so a future manual re-sync (webapp adds/changes a feature, this extension
needs to catch up, or vice versa) is a targeted diff against a known state
instead of a full re-read of both codebases.

**Ground truth for "current webapp behavior" is always the live HF Space**
(`.../raw/main/<file>.py`), not any local clone of the webapp repo — local
clones have been observed to drift out of date (e.g. missing swim bladder
segmentation, the edema UI, and the brush-based mask editor that are live in
the deployed app as of 2026-07-20).

## Provenance table

| Core file (this repo) | Webapp source | Ported functions | State at last sync | Deliberate deviations |
|---|---|---|---|---|
| `seg.py` | `seg.py` | `_load_unet_model`, `segmentation_pipeline` | 2026-07-20 (webapp `app.py`/`seg.py` as deployed on the live HF Space) | No Hugging Face Hub download in this layer — `_load_unet_model` only accepts a local `model_path`/`filename`; `repo_id`/`revision`/`force_download` params are accepted for call-site compatibility but ignored. Downloading is `ZebrafishEmbryoAnalyzerLib/model_downloader.py`'s job, driven by `ZebrafishEmbryoAnalyzerLib/model_manifest.py` (webapp downloads directly via `huggingface_hub.hf_hub_download` inline). |
| `seg_helper.py` | `seg.py` (helpers were inline in the webapp's single seg module at the time of the original port) | `load_images_from_path`, `segment_fish`, `fill_holes`, `grow_mask` | 2026-07-20 | None known — pure numpy/opencv, kept close to the original logic. |
| `length.py` | `length.py` | `compute_eye_metrics`, `compute_eye_diameters`, `tube_length_border2border`, `classification_curvature`, `load_model`, `preprocess_masked_image` | 2026-07-20 | Webapp's `compute_tube_metrics` (minimum-area rotated rectangle fit, used for swim bladder area/width) is not yet ported — tracked as part of issue #72. Webapp's older, unused geometric curvature-profile functions (`compute_curvature_profile`/`compute_curvature`) were intentionally not ported since the webapp itself doesn't wire them into its own pipeline either. |
| `manual.py` | `manual.py` | `compute_manual_length` | 2026-07-20 | None — module docstring already states this is shared logic between the webapp and this extension. |
| `scalebar.py` | `scalebar.py` | `detect_scalebar`, `calibrate_from_endpoints`, `draw_scalebar_endpoints` | 2026-07-20 | None known. `calibrate_from_endpoints`/`draw_scalebar_endpoints` exist here but (until issue #76) have no Slicer UI wired up to call them — the webapp exposes them via its manual scale-bar entry accordion. |

## Convention for future ports

See `CLAUDE.md` → "Code style" for the naming/signature convention: keep
ported functions close to the webapp original (same name, same signature)
unless there's a concrete reason to diverge, and note the reason inline as a
short comment when there is one. Update the table above whenever a function
is newly ported or re-synced, including the date and — where practical — a
pointer to which webapp state (a git SHA, if the webapp source is inspected
via its git history, or "as deployed on `<date>`" if inspected via the live
Space) the port was taken from.
