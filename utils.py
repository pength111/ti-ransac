"""
Utility functions for TI-RANSAC: orientation computation, export, and visualization.
"""

import numpy as np
import math


def calculate_orientation(plane_params):
    """
    Compute dip direction and dip angle from plane parameters.

    Args:
        plane_params: (A, B, C, D) plane equation parameters

    Returns:
        dip_direction: dip direction in degrees
        dip_angle: dip angle in degrees
    """
    A, B, C, D = plane_params
    normal = np.array([A, B, C])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return 0.0, 0.0
    normal = normal / norm

    Nx, Ny, Nz = normal

    # Orient normal to upper hemisphere (consistent dip angle reference)
    if Nz > 0:
        Nx, Ny, Nz = -Nx, -Ny, -Nz

    # Dip Direction: clockwise angle relative to Y-axis (North)
    dip_direction = math.degrees(math.atan2(Nx, Ny))
    if dip_direction < 0:
        dip_direction += 360

    # Dip Angle: angle between normal and vertical axis
    nz_clipped = np.clip(abs(Nz), 0, 1)
    dip_angle = math.degrees(math.acos(nz_clipped))

    return dip_direction, dip_angle


def save_labeled_point_cloud(points, labels, output_file):
    """
    Save labeled point cloud to CSV.

    Args:
        points: Nx3 point cloud array
        labels: length-N label array (-1 = unassigned)
        output_file: output file path
    """
    with open(output_file, 'w') as f:
        f.write("X,Y,Z,Label\n")
        for point, label in zip(points, labels):
            f.write(f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f},{label}\n")
    print(f"Labeled point cloud saved: {output_file}")


def export_planes_summary(planes, output_file):
    """
    Export discontinuity parameter summary table (including RMSE).

    Args:
        planes: list of discontinuity instance dicts
        output_file: output CSV file path
    """
    with open(output_file, 'w') as f:
        f.write("PlaneID,PointCount,A,B,C,D,DipDirection,DipAngle,RMSE\n")
        for plane in planes:
            dip_dir, dip_angle = calculate_orientation(plane['params'])
            A, B, C, D = plane['params']
            rmse = plane.get('rmse', 0)
            f.write(f"{plane['id']},{plane['count']},"
                   f"{A:.6f},{B:.6f},{C:.6f},{D:.6f},"
                   f"{dip_dir:.2f},{dip_angle:.2f},{rmse:.6f}\n")
    print(f"Plane summary saved: {output_file}")


def visualize_results(points, labels, planes, output_path=None):
    """
    Visualize labeled point cloud using Open3D.

    Opens an interactive 3D window with per-instance coloring. If output_path
    is provided, also saves the colored point cloud as a PLY file.

    Args:
        points: Nx3 point cloud array
        labels: length-N label array (-1 = unassigned, shown in gray)
        planes: list of discontinuity instance dicts
        output_path: optional path to save colored PLY file
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Warning: open3d not installed, skipping visualization")
        return

    # Generate per-point colors (golden angle for distinct hues)
    colors = np.zeros((len(points), 3))
    golden_angle = 137.508
    for label in np.unique(labels):
        if label == -1:
            colors[labels == label] = [0.5, 0.5, 0.5]  # gray for unassigned
        else:
            import colorsys
            hue = (label * golden_angle) % 360 / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            colors[labels == label] = rgb

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if output_path:
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Colored point cloud saved: {output_path}")

    print(f"Visualizing {len(planes)} instances ({len(points)} points)...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"TI-RANSAC: {len(planes)} instances",
        width=1280, height=960,
        point_show_normal=False
    )
