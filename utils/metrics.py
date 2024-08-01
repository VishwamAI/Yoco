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

import numpy as np


def precision_recall_f1(preds, targets, iou_threshold=0.5):
    """Calculate precision, recall, and F1 score."""
    tp, fp, fn = 0, 0, 0
    for pred, target in zip(preds, targets):
        iou = calculate_iou(pred, target)
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )
    return precision, recall, f1


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def mean_average_precision(preds, targets, iou_threshold=0.5):
    """Calculate mean Average Precision (mAP)."""
    aps = []
    for pred, target in zip(preds, targets):
        precision, recall, _ = precision_recall_f1(pred, target, iou_threshold)
        ap = np.trapz(recall, precision)
        aps.append(ap)
    return np.mean(aps)


def confusion_matrix(preds, targets, num_classes):
    """Calculate confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, target in zip(preds, targets):
        matrix[target, pred] += 1
    return matrix


def calculate_3d_iou(box1, box2):
    """Calculate 3D Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, z1, x2, y2, z2 = box1
    x1g, y1g, z1g, x2g, y2g, z2g = box2
    xi1, yi1, zi1 = max(x1, x1g), max(y1, y1g), max(z1, z1g)
    xi2, yi2, zi2 = min(x2, x2g), min(y2, y2g), min(z2, z2g)
    inter_volume = max(0, xi2 - xi1) * max(0, yi2 - yi1) * max(0, zi2 - zi1)
    box1_volume = (x2 - x1) * (y2 - y1) * (z2 - z1)
    box2_volume = (x2g - x1g) * (y2g - y1g) * (z2g - z1g)
    union_volume = box1_volume + box2_volume - inter_volume
    return inter_volume / union_volume if union_volume > 0 else 0


# Add more metrics as needed
