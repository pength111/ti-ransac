# TI-RANSAC: Topology-Aware Instance Segmentation for Semi-Automatic Rock Discontinuity Annotation from 3D Point Clouds

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

TI-RANSAC extends standard RANSAC with four collaborative mechanisms to resolve the "infinite plane" and "false plane" artifacts in automated rock discontinuity segmentation from 3D point clouds:

1. **Randomized Hough Transform (RHT)** normal estimation — voting-based normal estimation that preserves sharp edges
2. **Curvature-guided seed point selection** — statistical filtering to exclude non-planar features
3. **Octree-based spatial local sampling** — enforces spatial locality during random sampling
4. **Euclidean connectivity topology verification (DBSCAN)** — decouples coplanar but spatially disconnected joint surfaces

## Installation

```bash
git clone https://github.com/<your-username>/ti-ransac.git
cd ti-ransac
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- scikit-learn ≥ 1.0
- Plotly ≥ 5.0 (optional, for 3D visualization)

### Hardware Requirements

- Standard desktop computer with ≥ 16 GB RAM
- Multi-core CPU recommended (the algorithm uses parallel processing via ThreadPoolExecutor)

## Quick Start

### Synthetic Benchmark (Dodecahedron)

Reproduces the controlled experiment from the paper (Section 3.1):

```bash
python benchmark_dodecahedron.py
```

This generates a regular dodecahedron (12 faces, 1.2M points) with Gaussian noise (σ = 0.01 m), runs TI-RANSAC, and reports per-plane RMSE and orientation accuracy.

### Custom Point Cloud

```bash
python example_usage.py
```

Edit `example_usage.py` to set `input_file` to your `.txt`, `.csv`, `.xyz`, or `.npy` file (N×3 or N×D array).

## Usage

```python
import numpy as np
from ransac_discontinuity_detection import TIRANSAC
from utils import calculate_orientation

# Load point cloud (N × 3 array)
points = np.loadtxt("your_point_cloud.txt")

# Initialize TI-RANSAC
ransac = TIRANSAC(
    voxel_size=0.005,           # 5 mm voxel downsampling
    angle_threshold=25.0,       # Normal angle threshold (degrees)
    min_points=50,              # Minimum points per instance
    connectivity_factor=1.5,    # Spatial connectivity relaxation factor k₁
    k2=1.5,                     # DBSCAN connectivity relaxation factor k₂
    rht_neighbors=30,           # RHT neighborhood size
    rht_iterations=50,          # RHT iterations per point
    n_jobs=-1                   # Use all CPU cores
)

# Preprocessing
points_down = ransac.voxel_downsample(points)
normals = ransac.compute_normals(points_down)
curvature = ransac.compute_curvature(points_down)

# Extract discontinuity instances
planes, labels = ransac.extract_planes(points_down, normals, curvature)

# Output per-instance parameters
for plane in planes:
    dip_dir, dip_angle = calculate_orientation(plane['params'])
    print(f"Plane {plane['id']}: {plane['count']} points, "
          f"Orientation: {dip_dir:.1f}°∠{dip_angle:.1f}°, "
          f"RMSE: {plane['rmse']*1000:.2f} mm")
```

## Repository Structure

```
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── CITATION.cff                           # Citation metadata
├── ransac_discontinuity_detection.py      # TI-RANSAC core algorithm
├── utils.py                               # Utility functions (orientation, export, viz)
├── example_usage.py                       # End-to-end usage example on custom data
└── benchmark_dodecahedron.py              # Synthetic dodecahedron benchmark (Section 3.1)
```

## Input / Output

### Input

- 3D point cloud in `.txt`, `.csv`, `.xyz`, or `.npy` format (N×3 or N×D array, only first 3 columns used)

### Output

- `labeled_point_cloud.csv` — Point cloud with per-point instance labels
- `planes_summary.csv` — Per-instance parameters (ID, point count, plane equation, dip direction, dip angle, RMSE)
- `visualization.html` — Interactive 3D visualization (requires Plotly)

## Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `voxel_size` | L_voxel | 0.005 m | Voxel grid resolution for downsampling |
| `angle_threshold` | T_angle | 25° | Maximum normal angle deviation for inliers |
| `distance_threshold` | ε_d | adaptive | Maximum orthogonal distance to plane |
| `k2` | k₂ | 1.33–1.5 | DBSCAN connectivity relaxation factor |
| `min_points` | — | 50 | Minimum points per discontinuity instance |

## Reproducing Paper Results

### Synthetic Dodecahedron Benchmark (Section 3.1)

The script `ransac_discontinuity_detection.py` generates a regular dodecahedron with 1.2M points and Gaussian noise (σ = 0.01 m). Running the `__main__` block directly reproduces the geometric accuracy experiment from Table 2.

### Adit Case Study (Section 3.2)

The real-world adit point cloud data used in the case study cannot be shared publicly due to infrastructure security restrictions at the survey site. To reproduce the analysis on your own data, follow the Quick Start instructions above using identical parameter settings: `voxel_size=0.005`, `distance_threshold=0.015`, `angle_threshold=30°`, `k2=1.33`.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use TI-RANSAC in your research, please cite:

```bibtex
@article{ti-ransac,
  title     = {TI-RANSAC: Topology-Aware Instance Segmentation for Semi-Automatic Rock Discontinuity Annotation from 3D Point Clouds},
  journal   = {Computers & Geosciences},
  year      = {2026},
  author    = {Author 1 and Author 2 and Author 3}
}
```

See [CITATION.cff](CITATION.cff) for complete metadata.
