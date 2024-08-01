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

import json
import numpy as np
import cv2
from PIL import Image
import open3d as o3d


def load_image(file_path):
    """Load an image file."""
    return Image.open(file_path).convert("RGB")


def load_point_cloud(file_path):
    """Load a point cloud file."""
    return o3d.io.read_point_cloud(file_path)


def load_video(file_path):
    """Load a video file and return frames."""
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.BILINEAR)


def normalize_image(image):
    """Normalize an image."""
    return (image - image.mean()) / image.std()


def augment_2d(image):
    """Apply 2D data augmentation."""
    # Example: random horizontal flip
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def augment_3d(points):
    """Apply 3D data augmentation."""
    # Example: random rotation
    rotation = o3d.geometry.get_rotation_matrix_from_xyz(
        (np.random.rand(3) - 0.5) * 0.1
    )
    points.rotate(rotation)
    return points


def convert_2d_to_3d(points_2d, depth, camera_matrix):
    """Convert 2D points to 3D given depth and camera matrix."""
    points_2d_homogeneous = np.column_stack(
        (points_2d, np.ones(len(points_2d))))
    points_3d = np.dot(np.linalg.inv(camera_matrix), points_2d_homogeneous.T).T
    return points_3d * depth[:, np.newaxis]


def load_coco_annotations(file_path):
    """Load COCO format annotations."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_coco_annotations(annotations, file_path):
    """Save annotations in COCO format."""
    with open(file_path, "w") as f:
        json.dump(annotations, f)


# Add more utility functions as needed
