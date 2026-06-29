# 🐟 Zebrafish Embryo Analyzer

A 3D Slicer extension for batch offline zebrafish embryo morphometry from 2-D
microscopy images. Deep-learning models run entirely on your machine — no cloud,
no data upload.

![CI](https://github.com/MarkDanielArndt/SlicerZebrafishEmbryoAnalyzer/actions/workflows/ci.yml/badge.svg)

**Research use only. Not a medical device.**

---

## Table of Contents

- [What it does](#what-it-does)
- [How to use](#how-to-use)
  - [Installation](#installation)
  - [Python dependencies](#python-dependencies)
  - [Model download](#model-download)
  - [Loading images](#loading-images)
  - [Setting the scale](#setting-the-scale)
  - [Choosing measurements](#choosing-measurements)
  - [Running analysis](#running-analysis)
  - [Browsing results — Gallery tab](#browsing-results--gallery-tab)
  - [Inspecting a single image — Detail tab](#inspecting-a-single-image--detail-tab)
  - [Manual point correction](#manual-point-correction)
  - [Excluding images](#excluding-images)
  - [Exporting results](#exporting-results)
- [Measurements reference](#measurements-reference)
- [Curvature classes](#curvature-classes)
- [MRML integration](#mrml-integration)
- [Tests](#tests)
- [Platform support](#platform-support)
- [Known limitations](#known-limitations)
- [Contributors](#contributors)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## What it does

- Batch-loads 2-D microscopy images and measures each one without manual
  tracing
- Segments zebrafish body and eyes with deep-learning models (runs locally)
- Measures body length (µm), curvature class (1–4), length/straight-line ratio,
  eye area (µm²), and eye diameter (µm)
- Shows results in four tabs: **Gallery**, **Detail**, **Results**, **Exclude**
- Exports the measurements table to CSV and Excel
- Creates MRML nodes (table, volume, segmentation) that are visible in Slicer's
  **Data** module and slice views

---

## How to use

### Installation

The extension is loaded from a source checkout — it is not yet distributed
through the Extensions Manager.

1. Open **3D Slicer** (version 5.x).
2. Go to **Edit → Application Settings → Modules**.
3. Under **Additional module paths**, add the `ZebrafishAnalysis/` directory
   from this repository.
4. Click **OK** and restart Slicer.
5. On first open a dialog appears listing the Python packages that will be
   installed into Slicer's interpreter. Review the list and confirm. Nothing
   installs silently.
6. After installation finishes, restart Slicer a second time.
7. Open the **ZebrafishAnalysis** module from the **Modules** dropdown.

<!-- TODO: add screenshot -->
![Module selected in the Modules dropdown](Documentation_images/placeholder.png)

---

### Python dependencies

On first open you will be prompted to install:

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision`, `timm`, `segmentation-models-pytorch` | ML inference |
| `opencv-python`, `scipy`, `scikit-image`, `pillow` | Image processing |
| `openpyxl`, `matplotlib` | Export |
| `numpy<2` | Pinned for torch compatibility |
| `platformdirs` | Cache path lookup (soft dependency, falls back gracefully) |

Total download is several GB (torch alone is approximately 2 GB). Installation
can take several minutes depending on your connection.

---

### Model download

Models are **not** downloaded at startup. When you click **Run Analysis** for
the first time, a dialog lists the required models and their file sizes. The
download begins only after you confirm. Models are cached locally; no further
network access is needed for subsequent runs.

<!-- TODO: add screenshot -->
![Model download confirmation dialog](Documentation_images/placeholder.png)

---

### Loading images

Add one or more microscopy images for batch processing.

1. Click **Add images** in the module panel.
2. Select the image files you want to analyze (multiple selection is
   supported).
3. The loaded images appear in the image list. You can add more files at any
   time before running the analysis.

<!-- TODO: add screenshot -->
![Image list after loading several files](Documentation_images/placeholder.png)

---

### Setting the scale

Every measurement is reported in micrometres. The extension needs to know the
physical size of one pixel.

1. Enter the **µm/px** value in the scale field, or use the **scalebar
   detection** option to have the extension read the scale from an embedded
   scalebar in the image.
2. Verify the displayed scale before running analysis — all length and area
   measurements depend on it.

---

### Choosing measurements

Toggle the measurements you need before running:

- **Length** — body length in µm along the midline
- **Curvature** — curvature class (1–4; see [Curvature classes](#curvature-classes))
- **Ratio** — body length divided by the straight-line head-to-tail distance
- **Eye segmentation** — eye area (µm²) and eye diameter (µm)

You can also set a **confidence threshold** and select the inference model
(**General** or a fine-tuned variant) to match your imaging conditions.

---

### Running analysis

1. Click **Run Analysis**.
2. On the first run the extension loads models into memory — expect **10–30 s**
   before per-image progress begins. Subsequent runs start immediately.
3. A progress indicator shows which image is being processed.
4. When analysis is complete, the **Gallery**, **Results**, and other tabs
   populate automatically.

<!-- TODO: add screenshot -->
![Progress indicator during analysis](Documentation_images/placeholder.png)

---

### Browsing results — Gallery tab

The **Gallery** tab shows a thumbnail grid of every analyzed image with its
overlay drawn on top.

1. Switch to the **Gallery** tab after analysis completes.
2. Scroll through the thumbnails to get an overview of all results.
3. Click any thumbnail to open that image in the **Detail** tab.

<!-- TODO: add screenshot -->
![Gallery tab with thumbnail grid](Documentation_images/placeholder.png)

---

### Inspecting a single image — Detail tab

The **Detail** tab shows the selected image at full resolution with the
segmentation overlay and detected body axis.

1. Open the **Detail** tab (or click a thumbnail in the Gallery).
2. The overlay shows the segmented body outline, eye regions, and head/tail
   endpoints.
3. The measurement values for the selected image are displayed alongside the
   image.

<!-- TODO: add screenshot -->
![Detail tab with segmentation overlay](Documentation_images/placeholder.png)

---

### Manual point correction

If the automatic head/tail detection is incorrect for a particular image, you
can set the endpoints manually.

1. Open the image in the **Detail** tab.
2. Click on the image to place the **head** endpoint, then click again to place
   the **tail** endpoint.
3. Click **Apply Manual Points** to recompute the measurements using the
   corrected endpoints.

---

### Excluding images

Images that failed quality control can be excluded from the summary statistics
without deleting them from the session.

1. Open the **Exclude** tab.
2. Check the box next to each image you want to exclude.
3. Excluded rows are marked in the **Results** tab and are omitted from CSV and
   Excel exports.

<!-- TODO: add screenshot -->
![Exclude tab with checkboxes](Documentation_images/placeholder.png)

---

### Exporting results

1. Click **Export CSV** to save the measurements table as a comma-separated
   file, or click **Export Excel** to save as an `.xlsx` workbook.
2. Both exports respect the exclusions you set in the **Exclude** tab —
   excluded images are not written to the output file.

<!-- TODO: add screenshot -->
![Export buttons in the module panel](Documentation_images/placeholder.png)

---

## Measurements reference

| Measurement | Unit | Description |
|-------------|------|-------------|
| Body length | µm | Length along the detected midline |
| Curvature class | — | 1 (most severe) to 4 (minimal) |
| Length/straight-line ratio | — | Midline length ÷ head-to-tail distance |
| Eye area | µm² | Area of each segmented eye region |
| Eye diameter | µm | Diameter of each segmented eye region |

---

## Curvature classes

| Class | Severity |
|-------|---------|
| 1 | Most severe curvature |
| 2 | Moderate-severe |
| 3 | Mild |
| 4 | Minimal curvature (most healthy) |

Classes are determined automatically from the body shape detected by the
segmentation model.

---

## MRML integration

After each analysis the extension creates or updates the following nodes,
which appear in Slicer's **Data** module and slice views:

| Node type | Content |
|-----------|---------|
| `vtkMRMLTableNode` | Full measurements table |
| `vtkMRMLVectorVolumeNode` | Currently selected image |
| `vtkMRMLSegmentationNode` | Body and eye segments |

These nodes can be used in downstream Slicer workflows or saved with the scene.
Note: saving and reloading a scene restores the MRML nodes but not the
Gallery/Detail/Results tab state — this is expected behavior.

---

## Tests

The suite under `tests/` runs outside Slicer using plain Python:

```bash
python -m pytest tests/ -q
```

Lightweight dependencies only:

```
pytest
numpy<2
opencv-python-headless
platformdirs
```

Slicer integration tests live in `ZebrafishAnalysis/Testing/Python/` and
require a running Slicer instance.

CI runs the plain-Python suite on Ubuntu, macOS, and Windows with Python 3.11
and 3.12 via GitHub Actions.

---

## Platform support

| Platform | Status |
|----------|--------|
| macOS | Development platform — verified |
| Windows | Plain-Python CI green; full Slicer runtime not yet verified |
| Linux | Plain-Python CI green; full Slicer runtime not yet verified |

---

## Known limitations

- First analysis run is slow (10–30 s) due to model loading into memory.
- Scene save/reload restores MRML nodes but not tab state.
- Windows and Linux Slicer runtime not yet fully verified.

---

## Contributors

- Mark Daniel Arndt
- Jona Richter

Issues and questions: please open an issue in this repository and include your
Slicer version, OS, and a description of the problem.

---

## Acknowledgement

Based on the
[Zebrafish_webapp](https://github.com/MarkDanielArndt/Zebrafish_webapp) by Mark
Daniel Arndt.

---

## License

License not yet determined.
