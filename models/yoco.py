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

import torch
import torch.nn as nn
import torchvision.models as models


class YOCO(nn.Module):
    def __init__(self, num_classes=80):
        super(YOCO, self).__init__()
        # Use a pre-trained MobileNetV2 as the backbone for feature extraction
        self.backbone = models.mobilenet_v2(pretrained=True).features

        # Additional convolutional layers for object detection and segmentation
        self.conv1 = nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        # Output layers for bounding box prediction and class probability prediction
        self.bbox_pred = nn.Conv2d(256, num_classes * 4, kernel_size=1)
        self.class_pred = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        bbox_pred = self.bbox_pred(x)
        class_pred = self.class_pred(x)

        return bbox_pred, class_pred
