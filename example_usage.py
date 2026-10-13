"""
Example usage: load a point cloud from file and extract rock discontinuities
using TI-RANSAC. Supports txt, csv, xyz, and npy formats.
"""

import numpy as np
from ransac_discontinuity_detection import TIRANSAC
from utils import (calculate_orientation, save_labeled_point_cloud,
                   export_planes_summary, visualize_results)
import os


def load_point_cloud(file_path):
    """
    Load point cloud from file.

    Supported formats:
    - .txt, .xyz, .csv: plain text (space, comma, or semicolon delimited)
    - .npy: NumPy binary format

    Args:
        file_path: path to point cloud file

    Returns:
        points: Nx3 point cloud array
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.txt', '.xyz', '.csv']:
        try:
            points = np.loadtxt(file_path)
        except Exception:
            try:
                points = np.loadtxt(file_path, delimiter=',')
            except Exception:
                points = np.loadtxt(file_path, delimiter=';')

        if points.ndim == 1:
            points = points.reshape(-1, 3)
        if points.shape[1] > 3:
            points = points[:, :3]

    elif ext == '.npy':
        points = np.load(file_path)
        if points.shape[1] > 3:
            points = points[:, :3]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"Loaded point cloud: {file_path}")
    print(f"Point count: {len(points)}")
    print(f"Bounds:")
    print(f"  X: [{np.min(points[:, 0]):.3f}, {np.max(points[:, 0]):.3f}]")
    print(f"  Y: [{np.min(points[:, 1]):.3f}, {np.max(points[:, 1]):.3f}]")
    print(f"  Z: [{np.min(points[:, 2]):.3f}, {np.max(points[:, 2]):.3f}]")

    return points


def main():
    """Run TI-RANSAC on a point cloud file."""
    print("=" * 70)
    print("TI-RANSAC: Rock Discontinuity Instance Segmentation")
    print("=" * 70)

    # ---- Parameters ----
    # Point to your input file here. If the file does not exist, a synthetic
    # test point cloud is generated at this path instead.
    input_file = r"your_point_cloud.txt"

    params = {
        'voxel_size': 0.005,           # voxel size 5 mm
        'angle_threshold': 25.0,       # normal angle threshold (deg)
        'min_points': 50,              # minimum points per instance
        'pt': 0.99,                    # probability threshold
        'k': 3,                        # points to fit a plane
        'connectivity_factor': 1.5,    # spatial connectivity factor k1
        'rht_neighbors': 30,           # RHT neighborhood size
        'rht_iterations': 50,          # RHT iterations per point
        'curvature_k': 20,             # curvature neighborhood size
        'n_jobs': -1                   # use all CPU cores
    }

    max_iterations = 1000

    # ---- Load data ----
    print("\nStep 1: Load point cloud")
    print("-" * 70)

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Generating synthetic test data...")
        np.random.seed(42)

        n1, n2, n3 = 1000, 800, 600
        plane1 = np.column_stack([
            np.random.uniform(0, 10, n1),
            np.random.uniform(0, 10, n1),
            np.random.normal(5, 0.02, n1)])
        plane2 = np.column_stack([
            np.random.normal(5, 0.02, n2),
            np.random.uniform(0, 10, n2),
            np.random.uniform(0, 10, n2)])
        x3 = np.random.uniform(0, 10, n3)
        y3 = np.random.uniform(0, 10, n3)
        z3 = 0.5 * x3 + 0.3 * y3 + np.random.normal(0, 0.02, n3)
        plane3 = np.column_stack([x3, y3, z3])
        noise = np.random.uniform(0, 10, (200, 3))
        points = np.vstack([plane1, plane2, plane3, noise])
        np.savetxt(input_file, points, fmt='%.6f')
        print(f"Test data saved: {input_file}")
    else:
        points = load_point_cloud(input_file)

    # ---- Preprocessing ----
    print("\nStep 2: TI-RANSAC preprocessing")
    print("-" * 70)

    ransac = TIRANSAC(**params)
    points_down = ransac.voxel_downsample(points)

    print("\nStep 3: RHT normal estimation")
    print("-" * 70)
    normals = ransac.compute_normals(points_down)

    print("\nStep 4: Surface variation (curvature)")
    print("-" * 70)
    curvature = ransac.compute_curvature(points_down)

    # ---- Extract discontinuities ----
    print("\nStep 5: TI-RANSAC discontinuity extraction")
    print("-" * 70)
    planes, labels = ransac.extract_planes(
        points_down, normals, curvature, max_iterations
    )

    # ---- Save results ----
    print("\nStep 6: Save results")
    print("-" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_labeled = os.path.join(base_dir, "labeled_point_cloud.csv")
    output_summary = os.path.join(base_dir, "planes_summary.csv")
    output_viz = os.path.join(base_dir, "segmentation_result.ply")

    save_labeled_point_cloud(points_down, labels, output_labeled)
    export_planes_summary(planes, output_summary)

    try:
        visualize_results(points_down, labels, planes, output_viz)
    except Exception as e:
        print(f"Visualization failed: {e}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  Downsampled points: {len(points_down)}")
    print(f"  Extracted instances: {len(planes)}")
    classified = np.sum(labels >= 0)
    print(f"  Classified points: {classified} "
          f"({100*classified/len(points_down):.1f}%)")
    print(f"  Unclassified points: {np.sum(labels == -1)} "
          f"({100*np.sum(labels == -1)/len(points_down):.1f}%)")

    if planes:
        print(f"\nInstance details:")
        for plane in planes:
            dip_dir, dip_angle = calculate_orientation(plane['params'])
            rmse_mm = plane.get('rmse', 0) * 1000
            print(f"  Plane {plane['id']}: {plane['count']} points, "
                  f"DipDir {dip_dir:.1f}°, Dip {dip_angle:.1f}°, "
                  f"RMSE={rmse_mm:.2f}mm")

    print(f"\nOutput files:")
    print(f"  Labeled point cloud: {output_labeled}")
    print(f"  Plane summary: {output_summary}")
    if os.path.exists(output_viz):
        print(f"  Visualization: {output_viz}")

    print(f"\nParameters used:")
    print(f"  voxel_size={params['voxel_size']}mm, "
          f"angle_threshold={params['angle_threshold']}°, "
          f"connectivity_factor={params['connectivity_factor']}")
    print(f"  distance_threshold(Tgap)="
          f"{ransac.distance_threshold*1000:.1f}mm")


if __name__ == "__main__":
    main()
