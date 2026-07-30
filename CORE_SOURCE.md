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
| `seg.py` | `seg.py` | `_load_unet_model`, `segmentation_pipeline` | 2026-07-29 (webapp `app.py`/`seg.py`; the live HF Space and the webapp's GitHub repo were byte-identical for both files at that date) | `model_type` support (`"Unet"`/`"FPN"`) and the `include_swimbladder` path were re-synced on 2026-07-29; the pipeline's own defaults are kept identical to the webapp's so a future diff stays short. Return arity differs: the webapp enumerates the flag combinations explicitly, this port appends the requested masks in a fixed order (eyes, edema, swim bladder), which the caller unpacks by the flags it passed. No Hugging Face Hub download in this layer — `_load_unet_model` only accepts a local `model_path`/`filename`; `repo_id`/`revision`/`force_download` params are accepted for call-site compatibility but ignored. Downloading is `ZebrafishEmbryoAnalyzerLib/model_downloader.py`'s job, driven by `ZebrafishEmbryoAnalyzerLib/model_manifest.py` (webapp downloads directly via `huggingface_hub.hf_hub_download` inline). |
| `seg_helper.py` | `seg.py` (helpers were inline in the webapp's single seg module at the time of the original port) | `load_images_from_path`, `segment_fish`, `fill_holes`, `grow_mask` | 2026-07-20 | None known — pure numpy/opencv, kept close to the original logic. |
| `length.py` | `length.py` | `compute_eye_metrics`, `compute_eye_diameters`, `tube_length_border2border`, `classification_curvature`, `load_model`, `preprocess_masked_image`, `compute_tube_metrics` | 2026-07-29 (`compute_tube_metrics`; rest 2026-07-20) | `compute_tube_metrics` ported near-verbatim on 2026-07-29 from the live Space; only deviation is a deferred `import cv2` inside the function body, matching this repo's lazy-import convention for compiled extensions. `select_torch_device` is ours, not the webapp's — it probes the real model on a dummy input to catch a CUDA kernel/compute-capability mismatch that `torch.cuda.is_available()` does not detect, and `classification_curvature`/`load_model` were adjusted to read the device off the model rather than recomputing availability. Webapp's older, unused geometric curvature-profile functions (`compute_curvature_profile`/`compute_curvature`) were intentionally not ported since the webapp itself doesn't wire them into its own pipeline either. |
| `manual.py` | `manual.py` | `compute_manual_length` | 2026-07-20 | None — module docstring already states this is shared logic between the webapp and this extension. |
| `scalebar.py` | `scalebar.py` | `detect_scalebar`, `calibrate_from_endpoints`, `draw_scalebar_endpoints` | 2026-07-20 | None known. `calibrate_from_endpoints`/`draw_scalebar_endpoints` exist here but (until issue #76) have no Slicer UI wired up to call them — the webapp exposes them via its manual scale-bar entry accordion. |

## Model presets and weights

Not a core `.py` port, but the same re-sync problem: `ZebrafishEmbryoAnalyzerLib/model_manifest.py`
mirrors the webapp's `SEG_MODEL_OPTIONS` table. Synced 2026-07-29.

Read that table together with the comment directly above it and with the call
site in `process()`. **A `None` filename there means "use the pipeline default",
not "this preset has no such model."** Reading it the other way produced two
wrong implementations on 2026-07-29 (edema wrongly restricted to the DESY
preset, and the Fast & Easy body model discarded as a legacy file). Ports of
this table should be checked against the *resolved* filenames, not the literal
cell contents.

| Preset (webapp label) | `MODEL_SETS` key | Input size | Notes |
|---|---|---|---|
| `Fast & Easy (256 px, ~2s/image)` | `fast` | 256 | Body/eye/edema/swim-bladder all resolve to the pipeline defaults. Its swim bladder model is Unet + vgg16, unlike the 512px presets' FPN + vgg19. |
| `Complex & Slower (512 px, ~7s/image)` | `general` | 512 | Names body, eye and swim bladder explicitly; edema falls through to the default. |
| `Fine-tuned DESY` | `desy` | 512 | Names all four explicitly. |

The combo-box labels are the webapp's verbatim so users recognise the same
presets in both tools. The stable ids (`fast`/`general`/`desy`) are **not** the
webapp's strings — they are persisted in the MRML parameter node and travel
inside saved scenes, so they must not be renamed to match a display label.

**Deliberate deviation — edema on the 512px general preset.** The webapp offers
edema there and, having no 512px general edema model, falls back to the 256px
default and feeds it 512px input. This port omits the role for that preset
instead and greys the checkbox out.

Reason: Unet and FPN are fully convolutional and accept any input size divisible
by 32, so a resolution mismatch raises nothing — it yields a mask from which a
plausible-looking µm² figure is computed, indistinguishable from a correct one in
a result table. For a measurement tool a silently wrong number is worse than an
unavailable feature.

Rollback: if a 512px general edema model appears upstream, add it to `MODELS` and
wire it into `MODEL_SETS["general"]` under `"edema"`. The checkbox re-enables
itself — `widget.py` only tests `"edema" in model_set` — and no other change is
needed.

## Convention for future ports

See `CLAUDE.md` → "Code style" for the naming/signature convention: keep
ported functions close to the webapp original (same name, same signature)
unless there's a concrete reason to diverge, and note the reason inline as a
short comment when there is one. Update the table above whenever a function
is newly ported or re-synced, including the date and — where practical — a
pointer to which webapp state (a git SHA, if the webapp source is inspected
via its git history, or "as deployed on `<date>`" if inspected via the live
Space) the port was taken from.
