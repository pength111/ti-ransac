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


def visualize_results(points, labels, planes, output_path="visualization.html"):
    """
    Create interactive 3D visualization using Plotly.

    Args:
        points: Nx3 point cloud array
        labels: length-N label array
        planes: list of discontinuity instance dicts
        output_path: output HTML file path
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed, skipping visualization")
        return

    colors = []
    for label in labels:
        if label == -1:
            colors.append('gray')
        else:
            hue = (label * 137.508) % 360
            colors.append(f'hsl({hue}, 70%, 50%)')

    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors, opacity=0.8),
            text=[f'Label: {label}' for label in labels],
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        title=f'TI-RANSAC Discontinuity Extraction ({len(planes)} instances)',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=False
    )

    fig.write_html(output_path)
    print(f"Visualization saved: {output_path}")
