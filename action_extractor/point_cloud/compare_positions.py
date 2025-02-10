#!/usr/bin/env python

import argparse
import numpy as np
import plotly.graph_objects as go

def load_and_plot_trajectories(file1, file2, out_html="compare_trajectories.html"):
    """
    Loads two .npy files (each Nx4x4 array of poses),
    extracts their 3D positions, and plots them in an interactive 3D figure.
    The figure is saved as an HTML file (no pop-up window needed).
    """

    # Load the Nx4x4 pose arrays
    poses1 = np.load(file1)  # shape (N, 4, 4)
    poses2 = np.load(file2)  # shape (M, 4, 4)

    # Extract the position portion from each 4x4 transform
    # (the translation vector is in the last column)
    pos1 = poses1[:, :3, 3]  # shape (N, 3)
    pos2 = poses2[:, :3, 3]  # shape (M, 3)

    # Create a Plotly figure
    fig = go.Figure()

    # Add the first trajectory in blue
    fig.add_trace(
        go.Scatter3d(
            x=pos1[:, 0],
            y=pos1[:, 1],
            z=pos1[:, 2],
            mode='markers+lines',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue'),
            name='Trajectory 1'
        )
    )

    # Add the second trajectory in red
    fig.add_trace(
        go.Scatter3d(
            x=pos2[:, 0],
            y=pos2[:, 1],
            z=pos2[:, 2],
            mode='markers+lines',
            marker=dict(size=4, color='red'),
            line=dict(color='red'),
            name='Trajectory 2'
        )
    )

    # Make the axes have equal aspect ratio
    fig.update_layout(scene=dict(aspectmode='data'))

    # Save to HTML
    fig.write_html(out_html)
    print(f"Saved interactive 3D plot to {out_html}.")
    print("You can open this HTML file in a web browser to rotate/zoom and compare the two trajectories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two 3D trajectories.")
    parser.add_argument("file1", help="Path to first .npy file of Nx4x4 poses.")
    parser.add_argument("file2", help="Path to second .npy file of Nx4x4 poses.")
    parser.add_argument("--out_html", default="compare_trajectories.html",
                        help="Output HTML file for the interactive 3D plot.")
    args = parser.parse_args()

    load_and_plot_trajectories(args.file1, args.file2, out_html=args.out_html)