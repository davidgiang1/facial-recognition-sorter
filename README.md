# Facial Recognition Sorter

A Rust desktop application that uses deep learning to detect faces in a large image collection and find matches against a set of reference (target) images. It features a GPU-accelerated inference pipeline, a persistent face database, and an interactive GUI for browsing and exporting results.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Core Modules](#core-modules)
  - [main.rs — Pipeline Orchestrator](#mainrs--pipeline-orchestrator)
  - [face.rs — Detection & Recognition](#facers--detection--recognition)
  - [gui.rs — Desktop Interface](#guirs--desktop-interface)
  - [examples/debug_face.rs — Debug Utility](#examplesdebug_facers--debug-utility)
- [ML Models](#ml-models)
- [Data Schemas](#data-schemas)
- [Key Algorithms](#key-algorithms)
- [Processing Pipeline](#processing-pipeline)
- [Performance Design](#performance-design)
- [Setup & Usage](#setup--usage)
- [Python Utilities](#python-utilities)
- [Configuration & Persistence](#configuration--persistence)
- [Output Structure](#output-structure)

---

## Overview

Given a folder of **target reference images** (photos of the person you are looking for) and a large **input image directory**, the application will:

1. Detect and embed every face it finds in the input directory into a persistent database.
2. Compute representative embeddings for the target person using cross-image majority voting.
3. Score every database entry against the target using cosine similarity.
4. Surface the best matches in a ranked, paginated GUI where you can preview, filter, copy, and export results.

The face database is persisted to disk (`faces_db.bin`) so subsequent runs against the same input directory are fast — only new or changed images are reprocessed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUI (egui / eframe)                     │
│  Directory picker │ Threshold slider │ Image grid │ Export btns │
└──────────────────────────────┬──────────────────────────────────┘
                               │ UiMessage channel
┌──────────────────────────────▼──────────────────────────────────┐
│                    Pipeline Orchestrator (main.rs)               │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ Target Prep  │    │  DB Population   │    │   Matching    │  │
│  │              │    │                  │    │               │  │
│  │ Detect faces │    │ Walk input dir   │    │ Cosine sim    │  │
│  │ Vote on best │    │ CPU preprocess   │    │ Top-K avg     │  │
│  │ Cache to disk│    │ GPU detect+embed │    │ Threshold     │  │
│  └──────────────┘    │ Write to DB      │    │ filter+rank   │  │
│                      └──────────────────┘    └───────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      face.rs  (AI core)                         │
│                                                                 │
│   FaceDetector (SCRFD)          FaceRecognizer (ArcFace)        │
│   ┌──────────────────────┐      ┌──────────────────────────┐    │
│   │ Preprocess 640×640   │      │ Align face  112×112      │    │
│   │ ONNX infer (DirectML)│      │ ONNX infer (DirectML)    │    │
│   │ Decode anchors       │      │ L2-norm 512-dim embedding │    │
│   │ NMS                  │      └──────────────────────────┘    │
│   │ Landmark extraction  │                                      │
│   │ Rotation correction  │                                      │
│   └──────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
```

The application is single-binary. ML inference runs on the GPU via **ONNX Runtime with DirectML** (Windows GPU acceleration that works across NVIDIA, AMD, and Intel GPUs without requiring CUDA).

---

## Technology Stack

| Category | Library / Version | Purpose |
|---|---|---|
| Language | Rust (Edition 2024) | Core implementation |
| GUI | egui 0.28 + eframe 0.28 | Desktop window + UI widgets |
| ML Inference | ort 2.0.0-rc.11 (DirectML) | ONNX Runtime GPU inference |
| Numerics | ndarray 0.17 | Tensor manipulation |
| Image I/O | image 0.25 | Decode/encode JPEG, PNG, etc. |
| Parallelism | rayon 1.10 | CPU-side parallel preprocessing |
| Threading | crossbeam-channel 0.5 | Pipelined producer/consumer |
| Async | tokio 1.38 | Async runtime (GUI background tasks) |
| Serialization | serde + bincode + serde_json | Binary DB + JSON fallback |
| File traversal | walkdir 2.5 | Recursive directory scan |
| CLI | clap 4.5 | Argument parsing |
| File dialogs | rfd 0.14 | Native OS open/save dialogs |
| EXIF | kamadak-exif 0.5 | Image metadata reading |
| Progress | indicatif 0.17 | Terminal progress bars |
| Error handling | anyhow 1.0 | Ergonomic error propagation |

---

## Project Structure

```
facial-recognition-sorter/
│
├── src/
│   ├── main.rs              # Entry point: pipeline logic, DB management, matching
│   ├── face.rs              # FaceDetector (SCRFD) + FaceRecognizer (ArcFace)
│   └── gui.rs               # egui application: layout, state, image grid, export
│
├── examples/
│   └── debug_face.rs        # Standalone debug tool for the detection pipeline
│
├── scripts/
│   ├── download_models.py   # Downloads DINOv2 + exports YOLO26 ONNX models
│   └── quantize_models.py   # INT8/FP16 quantization of ONNX models
│
├── models/                  # ONNX model files (not in git)
│   ├── det.onnx             # SCRFD face detector (FP32, ~16 MB)
│   └── rec.onnx             # ArcFace face recognizer (FP32, ~174 MB)
│
├── people/                  # Target reference images (not in git)
├── output/                  # Generated outputs (not in git)
│   ├── debug_targets/       # Aligned face crops from target images
│   └── target_matches/      # Exported matched images
│
├── dumps/                   # Comprehensive export collections (not in git)
│
├── faces_db.bin             # Persistent face database (not in git)
├── target_cache.bin         # Cached target embeddings (not in git)
│
├── Cargo.toml               # Rust manifest + dependencies
├── Cargo.lock               # Pinned dependency versions
└── .gitignore
```

---

## Core Modules

### main.rs — Pipeline Orchestrator

The entry point and the glue between the GUI, face processing, and persistence layers.

**Key responsibilities:**

- **`PersonInfo`** — The fundamental data unit stored per detected face:
  ```rust
  struct PersonInfo {
      face_bbox: [f32; 4],       // [x1, y1, x2, y2] in pixels
      face_score: f32,            // detector confidence
      face_embedding: Vec<f32>,   // 512-dim ArcFace embedding (L2 normalized)
  }
  ```

- **`Database`** — A versioned `HashMap<image_path, Vec<PersonInfo>>` serialized to `faces_db.bin`. Version constant (`DB_VERSION = 6`) forces automatic cache invalidation when the schema changes.

- **`process_directory()`** — Main pipeline function (called from the GUI thread):
  1. Loads or initializes the face DB.
  2. Extracts and caches target embeddings (hash-based invalidation).
  3. Scans the input directory, identifies new images, runs the pipelined detection+embedding loop, and saves results to the DB.
  4. Runs matching and returns ranked results to the GUI.

- **`compute_target_hash()`** — Hashes target file paths + sizes to detect changes and invalidate `target_cache.bin`.

- **`cosine_similarity()` / `calculate_similarity()`** — Dot product of L2-normalized vectors, then top-K averaging over all target face embeddings for robustness.

---

### face.rs — Detection & Recognition

All ML inference and image math lives here.

#### `FaceDetector` (SCRFD model)

SCRFD (Sample and Computation Redistribution for Efficient Face Detection) is a lightweight multi-scale face detector.

**Detection pipeline:**
1. **Preprocess**: Resize input to 640×640, convert RGB→BGR, subtract 127.5, divide by 128.0. Output: `[1, 3, 640, 640]` ONNX tensor.
2. **Infer**: ONNX Runtime call via DirectML.
3. **Postprocess**: Decode three feature map scales (stride 8, 16, 32), each with 2 anchors per grid cell. Produces bounding boxes (via distance prediction) and 5 facial landmark points. Apply confidence threshold (0.5) + NMS (IoU 0.4). Minimum face size: 40px.

**Rotation correction (`correct_rotation()`):**
- Computes the eye-line tilt angle from the two eye landmarks.
- Validates landmark spatial plausibility (e.g., nose below eyes, mouth below nose).
- If tilt > 15°, rotates the image and re-detects.
- Accepts the correction only if the re-detected landmarks score higher on the plausibility metric.
- This handles tilted selfies, rotated scans, etc.

#### `FaceRecognizer` (ArcFace model)

ArcFace produces a 512-dimensional identity embedding optimized for face verification.

**Recognition pipeline:**
1. **`align_face()`**: Warp the detected face region to a canonical 112×112 pose using the 5 detected landmarks and a Umeyama similarity transformation (scale + rotation + translation, no shear).
2. **Preprocess**: Normalize to `[0, 1]`, rearrange to `[1, 3, 112, 112]`.
3. **Infer**: ONNX Runtime.
4. **Postprocess**: L2-normalize the 512-dim output vector. Cosine similarity is then just a dot product.

**`recognize_batch()`**: Processes faces in chunks of 128 images at a time to bound GPU memory usage.

#### Supporting math

- **`umeyama_similarity()`**: Computes the optimal affine transform (similarity, not full affine) aligning detected landmarks to canonical landmark positions. Closed-form solution.
- **`rotate_image()`**: Bilinear-interpolated image rotation.
- **`nms()`**: Standard greedy NMS by IoU.

---

### gui.rs — Desktop Interface

Built with egui (immediate-mode GUI). The GUI runs on the main thread; background processing runs in a `tokio` task that communicates back via a `crossbeam-channel`.

**Application state (`FaceSearchApp`):**
- Input directory path, target person directory
- Min/max similarity distance threshold (slider)
- Current search results (sorted `Vec` of `(path, distance)`)
- Pagination state (current page, page size)
- Texture cache for thumbnail display
- Log buffer for status messages
- Background task join handle + message channel

**UI panels:**
- **Top bar**: Input directory selector, target directory selector, "Search" button, processing status.
- **Settings sidebar**: Distance threshold range sliders, page size control.
- **Results grid**: Paginated image thumbnails with distance scores overlaid. Click to open, multi-select for batch ops.
- **Bottom bar**: "Copy selected to output", "Export to dump", "Select all/none", log output.

**Settings persistence**: JSON file in the OS config directory (via `directories` crate). Saves/restores directory paths and threshold values across sessions.

---

### examples/debug_face.rs — Debug Utility

Run with:
```sh
cargo run --example debug_face -- <path/to/image.jpg>
```

Traces the full detection pipeline on a single image and writes annotated outputs to `output/debug_targets/`:
- Bounding boxes and landmark points drawn on the original image.
- Aligned 112×112 face crops for each detected face.
- Rotation-corrected versions where applicable.

Useful for diagnosing detection failures, alignment quality, or rotation correction behavior.

---

## ML Models

Both models must be placed in the `models/` directory before running.

| File | Architecture | Input | Output | Size |
|---|---|---|---|---|
| `det.onnx` | SCRFD (face detection) | `[1, 3, 640, 640]` BGR float | Bounding boxes + landmarks across 3 scales | ~16 MB |
| `rec.onnx` | ArcFace (face recognition) | `[1, 3, 112, 112]` RGB float | `[1, 512]` identity embedding | ~174 MB |

**Why FP32?** The application deliberately uses full-precision models. INT8 quantization introduces accuracy loss that degrades verification reliability, which is unacceptable for the primary use case of finding a specific person.

**DirectML**: ONNX Runtime's DirectML execution provider is used for GPU acceleration on Windows. It supports NVIDIA, AMD, and Intel GPUs without requiring CUDA, making the application broadly compatible.

---

## Data Schemas

### Face Database (`faces_db.bin`)

Binary (bincode) encoding of:
```rust
struct Database {
    version: u32,                              // Must match DB_VERSION (currently 6)
    images: HashMap<String, Vec<PersonInfo>>,  // image_path → detected faces
}

struct PersonInfo {
    face_bbox: [f32; 4],       // [x1, y1, x2, y2]
    face_score: f32,            // SCRFD confidence [0, 1]
    face_embedding: Vec<f32>,   // 512-dim L2-normalized ArcFace embedding
}
```

A JSON fallback (`faces_db.json`) is also supported for inspection. Version mismatch triggers automatic full reprocessing of all images.

### Target Cache (`target_cache.bin`)

Binary (bincode) encoding of:
```rust
struct TargetCache {
    hash: u64,              // Hash of target file paths + sizes
    entries: Vec<PersonInfo>, // Representative target face embeddings
}
```

Invalidated automatically when any target image is added, removed, or changed in size.

---

## Key Algorithms

### Face Alignment — Umeyama Similarity Transform

The 5 detected landmark points (left eye, right eye, nose tip, left mouth corner, right mouth corner) are mapped to fixed canonical positions in a 112×112 image. The Umeyama algorithm finds the optimal similarity transform (uniform scale + rotation + translation) minimizing the sum of squared distances. This produces a pose-normalized face crop that the ArcFace model expects.

### Similarity Scoring — Top-K Averaging

When the target person has multiple reference images, each containing one or more detected faces, the system accumulates a pool of target embeddings. For each candidate image, every detected face is compared against every target embedding (cosine similarity). The **top-K (K=10) similarities** are averaged as the final score. This is more robust than a single max or mean: it ignores poor matches but rewards consistent strong matches.

Distance metric: `distance = 1.0 - cosine_similarity` → range [0, 2], where 0 is identical and 2 is maximally dissimilar. Typical match thresholds are in the 0.3–0.7 range.

### Target Representative Selection — Cross-Image Majority Voting

Target embeddings are deduplicated before use. If two detected faces in different target images have similarity > 0.95, they are considered the same face and only the higher-confidence one is retained. This prevents a single over-represented pose or lighting condition from dominating the target embedding pool.

### Rotation Correction

```
detect faces
  → compute eye-line angle
  → if |angle| > 15°:
      rotate image by -angle
      re-detect
      score landmark plausibility (spatial ordering of eye/nose/mouth)
      if plausibility(rotated) > plausibility(original):
          use rotated detection result
```

Plausibility is measured as the fraction of expected spatial ordering relationships that hold (e.g., left eye is to the left of right eye, nose is between eyes vertically, mouth is below nose).

---

## Processing Pipeline

### Phase 1 — Target Preparation

```
For each image in target directory:
    → Decode image
    → Run SCRFD detection (640×640)
    → For each detected face:
        → Correct rotation if needed
        → Align to 112×112 (Umeyama)
        → Run ArcFace → 512-dim embedding
    → Cross-image deduplication (similarity > 0.95)
→ Save representative embeddings to target_cache.bin
```

### Phase 2 — Database Population (Pipelined)

```
Producer thread (CPU, parallel via rayon):
    For each new image file:
        → Decode + resize to 640×640
        → Push preprocessed tensor to bounded channel (capacity=2)

Consumer thread (GPU):
    For each batch:
        → SCRFD detection
        → For each detected face:
            → Rotation correction (CPU)
            → Align to 112×112
        → ArcFace recognition (batched, 128 faces/chunk)
        → Write PersonInfo entries to database
→ Save database to faces_db.bin
```

The bounded channel (capacity 2) applies backpressure so the fast CPU producer doesn't build up unbounded memory when the GPU consumer is slower.

### Phase 3 — Matching & Ranking

```
For each image in database:
    For each detected face:
        → Compute cosine similarity against all target embeddings
    → Score = mean of top-K similarities (K=10)
    → Apply distance threshold filter [min, max]
→ Sort results ascending by distance (best match first)
→ Deduplicate against dumps/ directory (byte-level comparison)
→ Return ranked list to GUI
```

---

## Performance Design

| Technique | Detail |
|---|---|
| GPU inference | DirectML via ONNX Runtime; works on any modern Windows GPU |
| Batched detection | 50 images per detection batch |
| Batched recognition | 128 face crops per ArcFace inference call |
| CPU parallelism | Rayon work-stealing for image decoding and preprocessing |
| Pipelined CPU+GPU | Producer/consumer with bounded crossbeam channel |
| Incremental DB | Only unprocessed images are analyzed; existing entries reused |
| Target caching | Hash-based invalidation; target embeddings computed once |
| Thumbnail caching | egui texture cache prevents redundant GPU uploads |

---

## Setup & Usage

### Prerequisites

- Rust toolchain (stable, Edition 2024)
- Windows with a DirectML-compatible GPU (NVIDIA, AMD, or Intel)
- ONNX model files: `models/det.onnx` and `models/rec.onnx`

The SCRFD and ArcFace models are available from InsightFace's model zoo. They are not included in this repository due to size.

### Build & Run

```sh
# Debug build (slower inference)
cargo run

# Release build (recommended for performance)
cargo run --release
```

### Workflow

1. Launch the application.
2. Select the **target directory** containing reference photos of the person you want to find.
3. Select the **input directory** containing the image collection to search.
4. Adjust the **distance threshold** sliders (lower = stricter match; range 0–2).
5. Click **Search**. The application will:
   - Build/update the face database for the input directory.
   - Score all entries against the target.
   - Display results in the image grid, sorted best-match-first.
6. Select images and use **Copy** or **Export to Dump** to save results.

### Debug Tool

```sh
cargo run --example debug_face -- path/to/image.jpg
```

Annotated output images are written to `output/debug_targets/`.

---

## Python Utilities

These scripts are optional helpers, not required for normal operation.

### `scripts/download_models.py`

Downloads auxiliary models that are not part of the core pipeline:
- **DINOv2** (ViT-S/14) from HuggingFace — experimental, not currently integrated in main pipeline.
- **YOLO26m** — exports from the `ultralytics` package to ONNX format. Was explored as an alternative person detector; not used in production pipeline.

```sh
pip install onnxruntime onnx ultralytics
python scripts/download_models.py
```

### `scripts/quantize_models.py`

Quantizes ONNX models to INT8 or FP16 for faster inference or smaller size. Not used for the production `det.onnx` / `rec.onnx` because quantization degrades recognition accuracy.

---

## Configuration & Persistence

| File | Format | Purpose |
|---|---|---|
| `faces_db.bin` | bincode | Face database (image path → detected faces + embeddings) |
| `target_cache.bin` | bincode | Cached target embeddings with hash for invalidation |
| `faces_db.json` | JSON | Human-readable fallback/export of the face database |
| `<OS config dir>/facial-recognition-sorter/settings.json` | JSON | GUI settings: directory paths, threshold values |

All persistent files are in the project root or the OS config directory. The `.gitignore` excludes `faces_db.bin`, `target_cache.bin`, `models/`, `output/`, `people/`, and `dumps/`.

---

## Output Structure

```
output/
├── debug_targets/          # From debug_face example
│   ├── annotated_<name>.jpg    # Original image with bbox + landmarks drawn
│   └── aligned_<name>_<n>.jpg  # 112×112 aligned face crops
│
└── target_matches/         # From GUI "Copy" operation
    └── <matched images>

dumps/                      # From GUI "Export to Dump" operation
    └── <comprehensive export collections>
```
