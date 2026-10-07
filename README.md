# Layer 2 -- 2D-to-3D Object Association & 3D Relationship System

## Overview

This project is **Layer 2** of the Drone Data Augmentation Pipeline.

Layer 1 produces a 3D Gaussian Splatting reconstruction (a `.ply` file) from
drone footage.  Layer 2 takes that reconstruction and builds a graph of 3D
objects, then uses a Graph Neural Network (GNN) to predict semantic
relationships between them (e.g. *Car ON Road*, *Pole ON Pavement*).

---

## Milestone 1 -- PLY Inspection [COMPLETE]

**Script**: `src/inspect_ply.py`  
**Run**:    `python src/inspect_ply.py`

Inspects the demo 3DGS PLY and reports all 59 properties without assuming
any names.

**Demo PLY facts** (`data/ply/3dgs_sr_demo.ply`, ~3.8 MB):

| Attribute | Present? | Notes |
|-----------|----------|-------|
| XYZ position | YES | world coordinates |
| Opacity | YES | logit space |
| Scale | YES | log space, 3 axes |
| Rotation | YES | quaternion (rot_0..3) |
| SH (f_dc + f_rest) | YES | degree-3 (48 channels) |
| Object ID | NO | assigned by Layer 2 |
| Semantic Class ID | NO | assigned by Layer 2 |

---

## Milestone 2 -- Gaussian Data Model [COMPLETE]

**Module**: `src/gaussian/scene.py`  
**Validate**: `python src/test_gaussian_scene.py`

### What was implemented

- `GaussianScene` class backed by NumPy arrays (no per-Gaussian Python objects).
- Factory method `GaussianScene.from_ply(path)` loads any standard 3DGS PLY.
- Fields stored:

  | Field | Shape | Notes |
  |-------|-------|-------|
  | `ids` | (N,) int64 | Gaussian IDs = PLY vertex indices 0..N-1 |
  | `xyz` | (N, 3) float32 | world-space positions |
  | `opacity_raw` | (N,) float32 | raw logit value from PLY |
  | `scale_raw` | (N, 3) float32 | raw log value from PLY |
  | `rotation` | (N, 4) float32 | quaternion (rot_0..3), PLY convention |
  | `ply_vertex_index` | (N,) int64 | link back to source PLY row |
  | `sh_dc` | (N, 3) float32 | DC SH coefficients (optional) |
  | `sh_rest` | (N, K) float32 | higher-order SH (optional) |

### Key design decisions

- **Raw values are never overwritten.**  The PLY stores opacity in logit space
  and scale in log space; these are preserved exactly.
- **Physical conversions** are provided as helper methods that return new
  arrays:
  - `physical_opacity()` -- `sigmoid(opacity_raw)` -> [0, 1]
  - `physical_scale()`   -- `exp(scale_raw)`       -> real scale in scene units
- **Gaussian IDs** are currently the PLY vertex indices (0..17019 for the demo).
  They will be used as stable keys when the association pipeline assigns
  object memberships.
- `object_id` and `semantic_class_id` are **not present** -- they will be
  derived by later pipeline steps.
- Architecture is NumPy-based and compatible with the ~4 GB real IITB PLY.

### Known limitations (prototype)

- Quaternion ordering (`rot_0..3`) must be confirmed against the renderer
  before use in projection steps.
- SH coefficients are loaded but not yet consumed by any downstream module.
- Scale / rotation are not yet used for Gaussian footprint estimation.

---

## Project Structure

```
Layer2/
├── data/
│   ├── ply/            # Raw .ply files from Layer 1
│   │   └── 3dgs_sr_demo.ply
│   ├── colmap/         # COLMAP sparse reconstruction outputs
│   ├── frames/         # Extracted video frames (images)
│   ├── masks/          # Segmentation masks
│   └── results/        # Pipeline outputs
├── src/
│   ├── __init__.py
│   ├── inspect_ply.py          # M1: PLY inspection
│   ├── test_gaussian_scene.py  # M2: validation
│   └── gaussian/
│       ├── __init__.py
│       └── scene.py            # M2: GaussianScene class
├── requirements.txt
└── README.md
```

---

## Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | DONE | Project structure + PLY inspection |
| M2 | DONE | Gaussian data model (NumPy-backed) |
| M3 | DONE | Camera representation + projection test |
| M4 | DONE | 2D object / mask representation |
| M5 | DONE | Projection-based association baseline |
| M6 | DONE | Alpha-contribution association (prototype) |
| M7 | DONE | Depth-aware filtering |
| M8 | DONE | Multi-view evidence fusion |
| M9 | DONE | DBSCAN spatial cleanup |
| M10 | DONE | 3D object representation |
| M11 | DONE | ON relationship extraction |
| M12 | DONE | Final Layer-2 JSON |
| M13 | DONE | Raw drone frame download (datasets) |
| M14 | -- | YOLO26-L + SAM3 integration |
| M15 | -- | Real IITB Layer-1 PLY integration |

---

## Running the Demo

```bash
# Step 1 -- inspect PLY
python src/inspect_ply.py

# Step 2 -- validate Gaussian data model
python src/test_gaussian_scene.py

# Step 3 -- validate camera module (synthetic camera only)
python src/test_camera.py

# Step 4 -- validate 2D object representation (synthetic data only)
python src/test_object.py

# Step 5 -- validate projection association baseline (synthetic data only)
python src/test_projection_association.py

# Step 6 -- validate alpha-contribution prototype (synthetic data only)
python src/test_alpha_association.py

# Step 7 -- validate depth-aware filtering (synthetic data only)
python src/test_depth_filter.py

# Step 8 -- validate multi-view evidence fusion (synthetic data only)
python src/test_multiview.py

# Step 9 -- validate DBSCAN spatial cleanup (synthetic data only)
python src/test_dbscan_cleanup.py

# Step 10 -- validate 3D object representation (synthetic data only)
python src/test_object_3d.py

# Step 11 -- validate ON relationship extraction (synthetic data only)
python src/test_on_relation.py

# Step 12 -- validate final Layer-2 JSON output (synthetic data only)
python src/test_layer2_json.py

# Datasets -- download raw drone frames from Google Drive
python src/datasets/download_drive_frames.py

# Datasets -- probe access without downloading
python src/datasets/download_drive_frames.py --probe-only

# Datasets -- overwrite existing files
python src/datasets/download_drive_frames.py --overwrite
```

---

## Milestone 5 -- Projection-Based 2D->3D Association Baseline [COMPLETE]

**Module**: `src/association/projection_association.py`  
**Validate**: `python src/test_projection_association.py`  
**IMPORTANT**: All tests use **synthetic data only**. No PLY file, YOLO26-L, or SAM3 executed.

### How It Works

```
Gaussian 3D centre (world coords)
      |
      v  Camera W2C extrinsic  (R_w2c, t_w2c)
Camera-space point  [X_cam, Y_cam, Z_cam]
      |
      v  Depth check:  Z_cam > 0  (reject if behind camera)
      |
      v  Perspective projection
         u = fx * (X_cam / Z_cam) + cx   [float]
         v = fy * (Y_cam / Z_cam) + cy   [float]
      |
      v  Nearest-pixel rounding  ->  (u_int, v_int)
      |
      v  Integer bounds check:  0 <= u_int < W,  0 <= v_int < H
         (rounding can push float-valid pixels out of bounds -- rejected, not clamped)
      |
      v  Binary mask lookup:  Object2D.mask[v_int, u_int]
         (row = v = y,  col = u = x  -- NumPy convention)
      |
      v  Gaussian ID associated if mask is True
```

### Output

```json
{"object_id": 17, "gaussian_ids": [0, 1, 6, 7], "count": 4}
```

### This is a BASELINE Method

> **This baseline association method does not yet model Gaussian footprint,
> alpha contribution, occlusion, depth-aware filtering, or multi-view evidence.**

Known limitations:

| # | Limitation | Planned fix |
|---|-----------|-------------|
| 1 | Centre-only projection (full Gaussian extent ignored) | M6 alpha contribution |
| 2 | No alpha compositing | M6 |
| 3 | No occlusion / depth buffering | M7 depth filtering |
| 4 | Single-view only | M8 multi-view fusion |
| 5 | No spatial cleanup of floating outliers | M9 DBSCAN |

### Key Implementation Decisions

- **Pixel indexing**: `mask[v, u]` — rows = v = y, cols = u = x (explicitly enforced)
- **Rounding**: nearest-integer (`np.round`), documented as "round half to even"
- **Post-round bounds check**: pixels that round out of bounds are **rejected**, not clamped
- **Deduplication**: `np.unique` ensures sorted, unique Gaussian IDs in output
- **No hidden hard-coding**: the algorithm genuinely projects each point through the camera

---

## Dependencies

| Package | Version | Reason |
|---------|---------|--------|
| plyfile | 1.1.5 | Read/write PLY files |
| numpy | >=2.0 | Array-backed Gaussian storage |

Install: `pip install plyfile`

---

## Important Notes

- The **demo PLY** (`3dgs_sr_demo.ply`, ~3.8 MB, 17,020 Gaussians) is used
  for all development.  The actual **IITB PLY** (~4 GB) is NOT touched during
  this stage.
- All prototype methods are clearly labelled as unvalidated until
  experimentally confirmed against real data.
- Coordinate conventions are now documented in Milestone 3 below.

---

## Milestone 3 -- Camera and Coordinate System [COMPLETE]

**Modules**: `src/cameras/camera.py`, `src/cameras/projection.py`  
**Validate**: `python src/test_camera.py`  
**IMPORTANT**: Tested with a **synthetic camera only**. Real IITB camera
parameters must come from the Layer-1 / COLMAP / Nerfstudio output and must
not be invented.

### Coordinate Systems

#### World Space
A right-handed coordinate system used by the 3DGS reconstruction.  The exact
up direction depends on the reconstruction tool and must be verified when real
data is connected.

#### Camera Space (OpenCV convention)
```
+X  ->  right   (image column direction, u increases)
+Y  ->  down    (image row direction,    v increases)
+Z  ->  forward (into the scene, the "look" direction)
```
A point with `Z_cam > 0` is in front of the camera.

### Extrinsic Convention

Stored internally as **W2C (world-to-camera)**:

```
X_cam = R_w2c @ X_world + t_w2c

T_w2c = | R_w2c   t_w2c |
        |  0  0  0   1  |
```

The C2W (camera-to-world) form is always available as a derived property:

```
T_c2w = inv(T_w2c)
R_c2w = R_w2c^T
t_c2w = -R_w2c^T @ t_w2c  (= camera centre in world coords)
```

The `Camera` constructor accepts **either** `convention="w2c"` (COLMAP style)
or `convention="c2w"` (Nerfstudio / NeRF style) and converts internally.

### Intrinsic Matrix K

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

### Projection Equations

```
X_cam = R_w2c @ X_world + t_w2c          # world -> camera

u = fx * (X_cam / Z_cam) + cx            # camera -> pixel
v = fy * (Y_cam / Z_cam) + cy
```

### Handling Points Behind the Camera

Points with `Z_cam <= 0` are **invalid**.  `project_points()` sets their pixel
coordinates to `NaN` and returns `valid=False` for those entries.  No
division-by-zero occurs.

### Visibility vs Image Bounds

| Condition | Meaning |
|-----------|---------|
| `valid = True` | Point is in front of the camera (`Z > 0`) |
| `in_image_bounds = True` | Projected pixel is inside `[0,W) x [0,H)` |

These are separate boolean masks.  A point can be valid but projected outside
the image frame.

### Known Limitations (prototype)

- Only the **centre** of each Gaussian is projected.  The full Gaussian
  footprint / splatting radius is not yet considered.
- Quaternion ordering in the PLY (`rot_0..3`) must be confirmed against the
  renderer before use in rendering-based steps (M6+).
- No distortion model is implemented (not required for 3DGS scenes).

---

## Milestone 4 -- 2D Object Representation [COMPLETE]

**Module**: `src/objects/object.py`  
**Validate**: `python src/test_object.py`  
**IMPORTANT**: All tests use **synthetic data only**. The demo "car" object
is a development fixture, not a real detection result.

### Purpose of Object2D

`Object2D` represents one detected and segmented object instance within a
single drone image frame.  It is the bridge between the 2D detection/
segmentation stage and the 2D->3D Gaussian association stage.

### Field Reference

| Field | Type | Meaning |
|-------|------|---------|
| `object_id` | int | Persistent instance identity across frames |
| `semantic_class_id` | int | Numeric class from upstream detector |
| `class_name` | str | Human-readable label (e.g. "car") |
| `confidence` | float [0,1] | Detector/segmenter confidence score |
| `bbox` | [x1,y1,x2,y2] | Pixel bounding box (x1<x2, y1<y2) |
| `mask` | ndarray (H,W) bool | Binary instance mask |
| `source_frame` | str | Source image filename / identifier |

### object_id

Represents the **persistent identity** of an object instance across frames.
The same physical car seen in frame 001 and frame 002 should share the same
`object_id`.  The final tracking mechanism (connected in a later step) is
responsible for assigning and maintaining these IDs.

### semantic_class_id

**Dataset-specific numeric class ID.**  VisDrone, UAVid, and IITB campus
data may use different numbering for the same category.  A dataset-to-project
semantic mapping layer is planned (Milestone 14).  Do NOT assume IDs are
interchangeable across datasets.

### Bounding Box Format

`[x1, y1, x2, y2]` in **pixel coordinates** (not normalised).  
If YOLO26-L outputs normalised `[0,1]` coordinates, convert to pixels before
constructing `Object2D`.

### Mask Format

A NumPy `bool` array of shape `(image_height, image_width)`.  
`True` = pixel belongs to this object.  
SAM3 output must be converted to this format before storage.

### Helper Properties

| Property | Returns |
|----------|---------|
| `mask_area` | Count of True pixels |
| `mask_coverage` | `mask_area / (H * W)` |
| `mask_bbox` | `[x_min, y_min, x_max, y_max]` tight box around True pixels |
| `image_height` | Rows in mask |
| `image_width` | Columns in mask |

### Serialisation

`to_dict()` returns a JSON-serialisable dict.  The raw mask array is **not**
included (too large); `mask_shape` and `mask_area` are stored instead.

### Why YOLO26-L / SAM3 Are Not Executed Yet

`Object2D` is designed as a **detector-agnostic** data class.  It can be
populated from:
- A manual/synthetic fixture (current stage)
- YOLO26-L + SAM3 output (later milestone)
- Any other detection/segmentation pipeline

There is no import dependency on either model inside `Object2D`.

### Connection to 2D->3D Association (next step)

In Milestone 5 (projection baseline), the `Object2D.mask` will be used to
check which Gaussian projections land inside the object's pixel region.  The
`Object2D.bbox` provides a fast coarse filter before the per-pixel mask test.

### What Object2D Does NOT Contain

- Gaussian IDs (assigned in M5+)
- 3D position / 3D bounding box (derived in M11)
- Spatial relationships (extracted in M11)
- Layer classification / motion state (later milestones)

---

## Milestone 6 -- Alpha-Contribution Association [COMPLETE]

**Module**: `src/association/alpha_association.py`  
**Validate**: `python src/test_alpha_association.py`  
**IMPORTANT**: This is a CPU reference approximation for research prototyping. It is not an exact 3D Gaussian Splatting renderer.

### How It Works

While Step 5 asked *"Where is the Gaussian center?"*, Step 6 asks *"How much of this Gaussian's image-space contribution falls inside the object's mask?"*

```text
physical opacity & scale
      |
projected footprint (radius derived from scale & depth)
      |
per-pixel Gaussian weight ( G(dx, dy) = exp(-0.5 * (dx^2 + dy^2) / radius^2) )
      |
mask overlap (sum of contributions where mask is True)
      |
object contribution ratio ( overlap_sum / total_footprint_sum )
      |
threshold (e.g. >= 0.5)
      |
associated Gaussian
```

### Key Differences from Baseline

1. **Footprint vs Center**: A Gaussian whose center falls just outside the mask (rejected by Step 5) can now be associated if its footprint significantly overlaps the mask.
2. **Partial Overlap**: Provides a floating-point `association_score` [0, 1] instead of a strict binary in/out based solely on the center.

### Experimental Assumptions

- **Symmetric Footprint**: We approximate the footprint using a single scalar radius derived from the maximum physical scale axis. We ignore the full 3D covariance and quaternion rotation.
- **No Occlusion**: Transmittance and depth sorting are ignored. A Gaussian perfectly hidden behind a wall will still be associated if it projects into the mask area.
- **Tunable Threshold**: The `threshold` is exposed as an argument (default 0.5) and is an experimental parameter, not a finalized IITB value.
- **Performance**: We limit the footprint evaluation to a local bounding box of 3 std devs (`3 * radius`) to keep CPU processing feasible.

---

## Milestone 7 -- Depth-Aware Filtering [COMPLETE]

**Modules**: `src/association/depth_evidence.py`, `src/association/depth_filter.py`  
**Validate**: `python src/test_depth_filter.py`  
**IMPORTANT**: `DepthEvidence` is currently an interface for estimated depth information. This milestone uses synthetic depth maps for validation. A real depth-generation source (like SAM3 or a monocular depth model) must be connected later.

### How It Works

Two Gaussians can appear at the same image location but exist at different depths. Depth-aware filtering checks whether the Gaussian's 3D depth is consistent with the depth observed around the object mask.

Progression:
- **STEP 5**: Center projection inside mask
- **STEP 6**: Image-space footprint overlap + alpha contribution
- **STEP 7**: Image-space footprint overlap + alpha contribution + depth consistency

```text
Gaussian from Step 6
      |
sample depth_map at footprint pixels (where valid_mask == True)
      |
median observed depth
      |
relative error = abs(gaussian_Z - observed_depth) / observed_depth
      |
error <= max_relative_depth_error (e.g. 10%)
      |
accept/reject
```

### Depth Evidence Interface

`DepthEvidence` acts as a generic container for image-space depth data.
- **Convention**: Camera-space `Z` (depth), where `Z > 0` is in front of the camera.
- **Valid Mask**: Identifies pixels where the depth estimate is reliable.
- **Confidence Map**: Optional `[0, 1]` confidence scores for depth estimates.

This separates the association logic from the specific depth-generation model (SAM3, COLMAP dense stereo, etc.), keeping Layer 2 modular. COLMAP provides camera/scene geometry, while `DepthEvidence` provides image-space depth evidence.

### Experimental Assumptions

- **Median Sampling**: Uses the median depth within the footprint to be robust against outliers and edges.
- **Tunable Thresholds**: The 10% relative depth error limit (`max_relative_depth_error = 0.10`) is a tunable prototype value, not scientifically validated.
- **Insufficient Depth Policy**: If a Gaussian projects to an area with no valid depth evidence, the default policy is to `"reject"` it, but this is configurable to `"keep"`.
- **Synthetic Limitation**: We do NOT run real depth inference yet. Tests rely on deterministic, synthetic geometry.

---

## Milestone 8 -- Multi-View Evidence Fusion [COMPLETE]

**Module**: `src/association/multiview.py`  
**Validate**: `python src/test_multiview.py`  
**IMPORTANT**: This is an experimental evidence-fusion prototype. It is not a final validated Bayesian model.

### How It Works

A single drone frame can make an uncertain 2D->3D association. Multiple frames can observe the same Gaussian from different viewpoints. Multi-view fusion combines these observations into a more stable confidence score.

Progression:
- **STEP 5**: Center projection
- **STEP 6**: Footprint + alpha contribution
- **STEP 7**: Footprint + alpha + depth consistency
- **STEP 8**: Evidence from multiple views -> confidence fusion

```text
Per-View Observation (association_ratio, depth_error, frame_weight)
      |
depth_confidence = exp(-relative_depth_error / depth_scale)
      |
view_evidence = association_ratio * depth_confidence * frame_weight
      |
fused_score = sum(view_evidence) / sum(frame_weight)
      |
accepted = (fused_score >= fused_score_threshold)
```

### Key Concepts

- **ViewEvidence**: Container for per-frame evidence (`association_ratio`, `relative_depth_error`, `frame_weight`).
- **Weighted Aggregation**: Allows dynamic weighting of frames (e.g., higher weight for frames with stronger 2D detection confidence).
- **Missing-Depth Policy**: Governs how missing depth evidence is treated. Recommended and default is `"zero"` (missing depth provides no positive depth confidence).
- **Object Identity**: Aggregation strictly separates evidence by `(object_id, gaussian_id)`.

### Prototype & Tunable Parameters

- `depth_scale`: Controls the exponential decay of depth confidence. Default `0.10` corresponds roughly to a 10% relative error tolerance.
- `fused_score_threshold`: The final cutoff (default `0.5`) for accepting the Gaussian.
- `missing_depth_policy`: Default `"zero"`.
- `frame_weight`: Default `1.0`.

---

## Milestone 9 -- DBSCAN Spatial Cleanup [COMPLETE]

**Module**: `src/association/dbscan.py`  
**Validate**: `python src/test_dbscan_cleanup.py`  
**Dependency**: `scikit-learn`  
**IMPORTANT**: The DBSCAN parameters are prototype values and must be calibrated for the actual Layer-1 3DGS coordinate scale.

### How It Works

After multi-view fusion, some incorrect Gaussian IDs may remain isolated in 3D space. This can occur due to depth noise, alpha projection anomalies, or segmentation boundary bleeding. DBSCAN groups nearby candidate Gaussians and identifies isolated points as noise.

Progression:
- **STEP 5**: Center projection
- **STEP 6**: Footprint + alpha contribution
- **STEP 7**: Footprint + alpha + depth consistency
- **STEP 8**: Evidence from multiple views -> confidence fusion
- **STEP 9**: Spatial clustering cleanup -> cleaned 3D Gaussian IDs

```text
Fused Gaussians (object candidates)
      |
retrieve XYZ coordinates
      |
DBSCAN per-object (eps, min_samples)
      |
identify noise points (label -1)
      |
remove noise / preserve all non-noise clusters
      |
cleaned Gaussian IDs
```

### Key Concepts

- **Spatial Coherence Prior**: This is NOT semantic segmentation. It merely asserts that Gaussians belonging to the same object should generally occupy a spatially coherent region.
- **Per-Object Clustering**: Clustering runs independently for each 2D object ID. Nearby Gaussians belonging to different objects are never merged.
- **Default Policy**: We preserve *all* non-noise clusters by default. We do not arbitrarily cull down to the single largest cluster, as objects may consist of multiple disjoint parts.

### Prototype & Tunable Parameters

- `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default `0.05` is purely for synthetic validation and heavily dependent on the coordinate scale.
- `min_samples`: The number of samples in a neighborhood for a point to be considered a core point. Default `3`.

---

## Milestone 10 -- 3D Object Representation [COMPLETE]

**Module**: `src/objects/object_3d.py`  
**Validate**: `python src/test_object_3d.py`  

### How It Works

After cleaning the spatial point assignments in Step 9, this stage condenses the individual Gaussian IDs back into an explicit, object-level struct ready for topological mapping and serialization.

Progression:
- **STEP 8**: Multi-view fusion (confidence scoring)
- **STEP 9**: Spatial clustering cleanup (Gaussian ID filtering)
- **STEP 10**: Object3D Representation (AABB and Centroid derivation)

### Key Concepts

- **Coordinate System**: Bounding boxes and centroids directly inherit the Layer-1 3DGS world coordinate frame stored in `GaussianScene`. **No camera-space transformations** occur here.
- **Centroid**: The `center_3d` is evaluated as the arithmetic mean of the object's surviving Gaussian X, Y, and Z positions. **Note:** This is a *prototype object center estimate*, not necessarily the physically exact geometrical center.
- **Bounding Box**: Calculated as a strict, min-max Axis-Aligned Bounding Box (AABB) encapsulating all surviving Gaussian coordinates. 
- **Identity & Traceability**: The exact `gaussian_ids` surviving from Step 9 are preserved verbatim and mapped continuously, alongside an aggregated `mean_confidence` from Step 8's fused scores.

### Limitations

- Oriented Bounding Boxes (OBB) are not yet supported.
- Semantic relations (ON, UNDER, BESIDE) and Scene Graph extraction are **NOT** handled within this milestone.

---

## Milestone 11 -- 3D ON Relationship Extraction [COMPLETE]

**Module**: `src/relationships/on_relation.py`  
**Validate**: `python src/test_on_relation.py`  

### How It Works

This is a prototype geometric ON detector designed to deduce whether `Object A` (Subject) sits ON `Object B` (Support). It relies on the bounding geometries of `Object3D` models processed in M10.

Why is this necessary? Determining spatial relationships from 2D pixel overlaps alone (e.g., "Car" bounding box overlaps with "Road" pixels) is inherently inaccurate. A car mask overlapping a road mask could represent a car physically on the road, but it could equally represent a flying object passing between the drone and the road. We mandate genuine 3D geometric contact evaluation.

### Key Concepts & Calculations

- **Directionality**: Relationships possess explicit subjects and supports (`subject → ON → support`).
- **Vertical Gap Validation**: The geometric gap from the subject's base (lowest bounding coordinate) to the support's roof (highest bounding coordinate) must fall within a strict, configurable interval to confirm contact.
- **Footprint Overlap (`overlap_ratio`)**: Ensures the subject actually rests upon the horizontal real-estate of the support. This is computed as `Intersection Area / Subject Area` over the axes perpendicular to the configured up-axis.
- **Configurable `up_axis`**: Implemented with dynamic axis selection (`x`, `y`, or `z`) because the physical gravity orientation is unconfirmed until real Layer-1 integration occurs.
- **Prototype Score Formulation (`on_score`)**: Computes a linear decay of confidence inversely proportional to the absolute vertical gap, multiplied strictly by the footprint overlap ratio.

### Limitations

- **Prototype Assumptions**: The numerical thresholds and resulting `on_score` are prototype geometric heuristics. They are NOT calibrated probabilities.
- **Lack of True Surfaces**: Bounding Box (AABB) intersection is employed rather than dense 3D surface mesh normal alignment, which remains a limitation bounded by the precision of prototype point-cloud clustering.
