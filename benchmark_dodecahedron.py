"""
Synthetic dodecahedron benchmark for TI-RANSAC.

Reproduces the geometric accuracy experiment from Section 3.1 of the paper.
Generates a regular dodecahedron with 12 faces, adds Gaussian noise,
and evaluates segmentation quality against ground truth.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
from ransac_discontinuity_detection import TIRANSAC
from utils import calculate_orientation


def generate_dodecahedron_points(n_points_per_face=5000, noise_sigma=0.01):
    """Generate a regular dodecahedron point cloud for testing."""
    np.random.seed(42)

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # 20 vertices of a regular dodecahedron
    vertices = []
    for i in range(3):
        for j in range(2):
            for k in range(2):
                sign1 = 1 if j == 0 else -1
                sign2 = 1 if k == 0 else -1
                vert = [0, 0, 0]
                vert[i] = sign1
                vert[(i+1) % 3] = sign2 * phi
                vert[(i+2) % 3] = sign2 / phi
                vertices.append(vert)

    for i in range(3):
        for j in range(2):
            sign1 = 1 if j == 0 else -1
            vert = [0, 0, 0]
            vert[i] = sign1 * phi
            vert[(i+1) % 3] = sign2 * (1/phi)
            vert[(i+2) % 3] = 0
            vertices.append(vert)

    vertices = np.array(vertices)
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms

    # Compute 12 face vertex indices (dodecahedron dual to icosahedron)
    hull = ConvexHull(vertices)
    faces = hull.simplices  # triangular faces

    # Generate points for each face
    all_points = []
    for tri in faces:
        v1, v2, v3 = vertices[tri]
        # Uniform sampling within triangle
        r1 = np.sqrt(np.random.uniform(0, 1, n_points_per_face))
        r2 = np.random.uniform(0, 1, n_points_per_face)
        pts = (1 - r1[:, None]) * v1[None, :] + \
              r1[:, None] * (1 - r2[:, None]) * v2[None, :] + \
              r1[:, None] * r2[:, None] * v3[None, :]
        all_points.append(pts)

    points = np.vstack(all_points)
    points += np.random.normal(0, noise_sigma, points.shape)
    return points


if __name__ == "__main__":
    print("=" * 60)
    print("TI-RANSAC: Topology-Aware Instance Segmentation RANSAC")
    print("Synthetic Dodecahedron Benchmark")
    print("=" * 60)

    # Generate test data
    print("\nGenerating dodecahedron test data...")
    points = generate_dodecahedron_points(n_points_per_face=2000, noise_sigma=0.01)
    print(f"Total points: {len(points)}")

    # Create TI-RANSAC instance
    ransac = TIRANSAC(
        voxel_size=0.05,
        angle_threshold=25.0,
        min_points=100,
        pt=0.99,
        k=3,
        connectivity_factor=1.5,
        rht_neighbors=30,
        rht_iterations=50,
        n_jobs=-1
    )

    t0 = time.time()

    # Voxel downsampling
    points_down = ransac.voxel_downsample(points)

    # RHT normal computation
    normals = ransac.compute_normals(points_down)

    # Curvature computation
    curvature = ransac.compute_curvature(points_down)

    # TI-RANSAC discontinuity extraction
    planes, labels = ransac.extract_planes(points_down, normals, curvature,
                                           max_iterations=5000)

    t1 = time.time()

    print(f"\n{'='*60}")
    print(f"Total time: {t1 - t0:.2f}s")
    print(f"Extracted planes: {len(planes)}")
    print(f"{'='*60}")

    for plane in planes:
        dip_dir, dip_angle = calculate_orientation(plane['params'])
        print(f"  Plane {plane['id']}: {plane['count']} points, "
              f"Orientation: {dip_dir:.1f}°∠{dip_angle:.1f}°, "
              f"RMSE: {plane['rmse']*1000:.2f}mm")
