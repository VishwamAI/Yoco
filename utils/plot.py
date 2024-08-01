# VishwamAI
# Copyright (C) 2024 Kasinadhsarma
#
# This file is part of the VishwamAI project.
#
# VishwamAI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# VishwamAI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with VishwamAI. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Kasinadhsarma

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns


def plot_2d_bounding_boxes(image, boxes, labels=None, colors=None):
    """Plot 2D bounding boxes on an image."""
    plt.imshow(image)
    ax = plt.gca()
    if colors is None:
        colors = plt.cm.hsv(np.linspace(0, 1, len(boxes)))
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colors[i], linewidth=2
        )
        ax.add_patch(rect)
        if labels:
            plt.text(x1, y1, labels[i], color=colors[i])
    plt.axis("off")
    plt.show()


def plot_segmentation_mask(image, mask, alpha=0.5):
    """Plot segmentation mask on an image."""
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha, cmap="jet")
    plt.axis("off")
    plt.show()


def plot_3d_bounding_boxes(points, boxes):
    """Plot 3D bounding boxes on a point cloud."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(pcd)
    for box in boxes:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(box)
        )
        bbox.color = (1, 0, 0)
        vis.add_geometry(bbox)
    vis.run()
    vis.destroy_window()


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def plot_precision_recall_curve(precision, recall):
    """Plot precision-recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, "b-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()


def plot_feature_map(feature_map):
    """Plot feature map or attention map."""
    plt.figure(figsize=(12, 8))
    num_features = feature_map.shape[0]
    rows = int(np.ceil(np.sqrt(num_features)))
    cols = int(np.ceil(num_features / rows))
    for i in range(num_features):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map[i], cmap="viridis")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Add more plotting functions as needed
